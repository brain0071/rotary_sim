import casadi as ca
import numpy as np

# ==========================================
# 1. 物理参数配置 (来源于你的启动日志)
# ==========================================
# EKF 需要知道机器人的物理特性才能进行预测
# 这里填入日志中 "Loaded Parameters" 显示的数值
# ROBOT_PARAMS = {
#     'mass': np.array([4200.0]),
#     'inertia': np.array([3364.0, 5912.0, 5075.0]),
#     'add_mass': np.array([420.0, 420.0, 420.0, 336.0, 591.0, 507.0]),
#     'quadratic_damp': np.array([2327.0, 4042.0, 3135.0, 2829.0, 8535.0, 11000.0]),
#     'max_force_moment': np.array([7538.0, 7538.0, 9233.0, 8529.0, 8529.0, 11728.0])
# }

ROBOT_PARAMS = {
    'mass': np.array([25.00]),
    'inertia': np.array([0.64, 1.34, 1.38]),
    'add_mass': np.array([2.50, 27.90, 27.90, 0.00, 0.60, 0.60]),
    'quadratic_damp': np.array([27.36, 67.32, 67.32, 0.00, 0.28, 0.28]),
    'max_force_moment': np.array([158.33, 158.33, 158.33, 42.54, 55.98, 34.83])
}

class EKF_Estimator:
    def __init__(self, opt, dt):
        """
        初始化 EKF
        :param opt: mpc_node 传入的配置对象 (我们这里主要用它来兼容接口，参数主要从上方常量读取)
        :param dt: 采样时间 (0.05s / 20Hz)
        """
        self.dt = dt
        self.opt = opt # 保留引用以备后用
        self.params = ROBOT_PARAMS
        
        # --- 状态维度定义 ---
        # State: Pos(3) + Quat(4) + Vel(3) + Rate(3) + Dist_Lin(3) + Dist_Ang(3) = 19
        self.n_x = 19
        # Input: u [-1, 1] (6维)
        self.n_u = 6   
        # Measurement: Pos(3) + Quat(4) + Vel(3) + Rate(3) = 13
        self.n_z = 13  

        # --- 初始状态 ---
        self.x = np.zeros(self.n_x)
        self.x[3] = 1.0  # Quaternion [qw, qx, qy, qz]

        # --- 协方差矩阵初始化 ---
        self.P = np.eye(self.n_x) * 0.1

         # --- 噪声矩阵配置 ---
        # Process Noise (Q): 相信模型但在扰动项上给较大自由度
        # 对应: Pos(3), Quat(4), Vel(3), Rate(3), Dist_Lin(3), Dist_Ang(3)
        q_diag = np.concatenate([
            [0.001]*3,       # Pos
            [0.001]*4,       # Quat
            [0.1]*3,         # Vel
            [0.1]*3,         # Rate
            [1.0]*3,         # Dist_Lin (允许快速变化)
            [0.5]*3          # Dist_Ang
        ])
        self.Q = np.diag(q_diag) * self.dt

        # Measurement Noise (R): 传感器精度
        # 对应: Pos(3), Quat(4), Vel(3), Rate(3)
        r_diag = np.concatenate([
            [0.1]*3,         # GPS/Pos
            [0.01]*4,        # IMU/Quat
            [0.05]*3,        # DVL/Vel
            [0.02]*3         # Gyro/Rate
        ])
        self.R = np.diag(r_diag)

        # 构建 CasADi 模型
        self._build_model()

    def _build_model(self):
        """构建动力学方程 (匹配 Robot.py 的逻辑)"""
        # 1. 提取参数
        m = self.params['mass'][0]
        ma = ca.DM(self.params['add_mass'])
        I = ca.DM(self.params['inertia'])
        D_quad = ca.DM(self.params['quadratic_damp'])
        F_max = ca.DM(self.params['max_force_moment'])

        # 2. 符号变量
        x_sym = ca.SX.sym('x', self.n_x)
        u_sym = ca.SX.sym('u', self.n_u)

        # 解包状态
        pos = x_sym[0:3]
        quat = x_sym[3:7] # qw, qx, qy, qz
        vel = x_sym[7:10] # u, v, w (体坐标系)
        rate = x_sym[10:13]# p, q, r (体坐标系)
        dist_lin = x_sym[13:16] # 线性扰动估计
        dist_ang = x_sym[16:19] # 角扰动估计

        # 3. 运动学 (Kinematics)
        # 旋转矩阵 R (Body -> Earth)
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        R_mat = ca.vertcat(
            ca.horzcat(1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)),
            ca.horzcat(2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)),
            ca.horzcat(2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx**2+qy**2))
        )
        pos_dot = ca.mtimes(R_mat, vel)

        # 四元数导数
        Lambda = ca.vertcat(
            ca.horzcat(0, -rate[0], -rate[1], -rate[2]),
            ca.horzcat(rate[0], 0, rate[2], -rate[1]),
            ca.horzcat(rate[1], -rate[2], 0, rate[0]),
            ca.horzcat(rate[2], rate[1], -rate[0], 0)
        )
        quat_dot = 0.5 * ca.mtimes(Lambda, quat)

        # 4. 动力学 (Dynamics) - 严格匹配 Robot.py
        # Robot.py 逻辑: dot_u = (Force - Damp)/Mass (无科里奥利力耦合)
        
        # 线运动 (Surge, Sway, Heave)
        # Force = Input*Max - Damp*|v|*v + Disturbance
        forces_lin = u_sym[0:3] * F_max[0:3] - D_quad[0:3] * ca.fabs(vel) * vel + dist_lin
        acc_lin = forces_lin / (m + ma[0:3]) # 逐元素除以 (mass + added_mass)

        # 角运动 (Roll, Pitch, Yaw)
        forces_ang = u_sym[3:6] * F_max[3:6] - D_quad[3:6] * ca.fabs(rate) * rate + dist_ang
        acc_ang = forces_ang / (I + ma[3:6]) # 逐元素除

        # 5. 扰动模型 (Random Walk)
        dist_lin_dot = ca.SX.zeros(3)
        dist_ang_dot = ca.SX.zeros(3)

        # 整合
        x_dot = ca.vertcat(pos_dot, quat_dot, acc_lin, acc_ang, dist_lin_dot, dist_ang_dot)

        # --- RK4 积分 ---
        f = ca.Function('f', [x_sym, u_sym], [x_dot])
        k1 = f(x_sym, u_sym)
        k2 = f(x_sym + self.dt/2 * k1, u_sym)
        k3 = f(x_sym + self.dt/2 * k2, u_sym)
        k4 = f(x_sym + self.dt * k3, u_sym)
        x_next = x_sym + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 归一化四元数
        x_next[3:7] = x_next[3:7] / ca.norm_2(x_next[3:7])

        # 生成函数
        self.fn_predict = ca.Function('fn_predict', [x_sym, u_sym], [x_next])
        self.fn_jacobian = ca.Function('fn_jacobian', [x_sym, u_sym], [ca.jacobian(x_next, x_sym)])

    def predict(self, u_control):
        """
        预测步骤
        :param u_control: 当前控制输入 (归一化 [-1, 1])
        """
        # 确保输入维度正确
        if len(u_control) != self.n_u:
            # 如果输入有问题，做简单处理或报错
            u_control = np.zeros(self.n_u)

        x_pred_ca = self.fn_predict(self.x, u_control)
        self.x_pred = np.array(x_pred_ca).flatten()
        
        F_ca = self.fn_jacobian(self.x, u_control)
        F = np.array(F_ca)
        
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_meas):
        """
        更新步骤
        :param z_meas: 传感器测量值 [Pos(3), Quat(4), Vel(3), Rate(3)]
        """
        # 观测矩阵 H
        H = np.zeros((self.n_z, self.n_x))
        H[0:3, 0:3] = np.eye(3)
        H[3:7, 3:7] = np.eye(4)
        H[7:10, 7:10] = np.eye(3)
        H[10:13, 10:13] = np.eye(3)

        z_pred = H @ self.x_pred
        y = z_meas - z_pred
        
        # 简单的卡尔曼增益计算
        S = H @ self.P @ H.T + self.R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，跳过更新或使用伪逆
            return

        self.x = self.x_pred + K @ y
        self.P = (np.eye(self.n_x) - K @ H) @ self.P
        
        # 再次归一化
        self.x[3:7] /= np.linalg.norm(self.x[3:7])

    def step(self, measurement, control_input):
        self.predict(control_input)
        self.update(measurement)
        return self.x