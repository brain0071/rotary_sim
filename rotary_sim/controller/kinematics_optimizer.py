import os
import numpy as np
import sys

import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from rotary_sim.utils.utils import q_to_rot_mat, skew_symmetric
from copy import copy
import time



class MPC_Kinematics_Optimizer:

    def __init__(self, robot, kinematics_n_nodes, kinematics_q_cost, kinematics_r_cost, kinematics_dt):
        
        self.robot = robot
        self.max_kinematics_u = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        self.min_kinematics_u = np.array([-0.6, -0.6, -0.6, -0.6, -0.6, -0.6])

        self.robot_max_acceleration = robot.max_acceleration
        self.robot_max_angular_velocity = robot.max_angular_velocity

        self.N = kinematics_n_nodes
        self.T = self.N * kinematics_dt
        self.dt = kinematics_dt
        
        # dx dy dz x y z qw qx qy qz
        self.dp = cs.MX.sym("dp", 3)
        self.p = cs.MX.sym("p", 3)
        self.q = cs.MX.sym("q", 4)
        self.x = cs.vertcat(self.dp, self.p, self.q)

        du = cs.MX.sym("du")
        dv = cs.MX.sym("dv")
        dw = cs.MX.sym("dw")
        p = cs.MX.sym("p")
        q = cs.MX.sym("q")
        r = cs.MX.sym("r")
        self.u = cs.vertcat(du, dv, dw, p, q, r)

        self.acados_ocp_solver = {}
        self.acados_models_dir = ("/home/naodai/Workspace/rotary/ros2_ws/src/rotary_sim/acados_models")
    
        ocp = AcadosOcp()
        ocp.dims.N = self.N
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ocp.cost.W = np.diag(np.concatenate((kinematics_q_cost, kinematics_r_cost)))
        ocp.cost.W_e = np.diag(kinematics_q_cost)

        self.standard_kinematics = self.robot_kinematics()
        self.model_name = "kinematics"
        stand_acados_model = self.acados_setup_model(self.standard_kinematics(x=self.x, u=self.u)["x_next"], self.model_name)

        ocp.model = stand_acados_model
        nx = stand_acados_model.x.size()[0]
        nu = stand_acados_model.u.size()[0]

        ny = nx + nu
        x_ref = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        y_ref = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_ref_e = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        ocp.cost.yref = y_ref
        ocp.cost.yref_e = y_ref_e
        
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        
        ocp.constraints.x0 = x_ref
        ocp.constraints.lbu = np.array(self.min_kinematics_u)
        ocp.constraints.ubu = np.array(self.max_kinematics_u)

        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])
        ocp.solver_options.tf = self.T
        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.qp_solver_iter_max = 2000

        json_file = os.path.join(self.acados_models_dir, stand_acados_model.name + "_acados_ocp.json")
        self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file=json_file)

        self.max_opt_time = 0
        self.sum_opt_time = 0
        self.num_opt = 0

    def robot_kinematics(self):

        x_next = cs.vertcat(self.dp_kinematics(), self.p_kinematics(), self.q_kinematics())
        return cs.Function("x_next", [self.x, self.u], [x_next], ["x", "u"], ["x_next"])
    
    def dp_kinematics(self):
    
        dV = cs.mtimes(q_to_rot_mat(self.q), cs.vertcat(self.u[0] * self.robot_max_acceleration, self.u[1] * self.robot_max_acceleration, self.u[2] * self.robot_max_acceleration))
        self.dp[0] = self.dp[0] + self.dt * dV[0]
        self.dp[1] = self.dp[1] + self.dt * dV[1]
        self.dp[2] = self.dp[2] + self.dt * dV[2]
        return self.dp

    def p_kinematics(self):
        
        dV = cs.mtimes(q_to_rot_mat(self.q), cs.vertcat(self.u[0] * self.robot_max_acceleration, self.u[1] * self.robot_max_acceleration, self.u[2] * self.robot_max_acceleration))
        self.p[0] = self.p[0] + self.dp[0] + self.dt * dV[0]
        self.p[1] = self.p[1] + self.dp[1] + self.dt * dV[1]
        self.p[2] = self.p[2] + self.dp[2] + self.dt * dV[2]
        return self.p

    def q_kinematics(self):
        
        rate = self.u[3:] 
        self.q[0] = self.q[0] - 0.5 * (self.q[1] * (rate[0] * self.robot_max_angular_velocity) + self.q[2] * (rate[1] * self.robot_max_angular_velocity) + self.q[3] * (rate[2] * self.robot_max_angular_velocity)) * self.dt
        self.q[1] = self.q[1] + 0.5 * (self.q[0] * (rate[0] * self.robot_max_angular_velocity) + self.q[2] * (rate[2] * self.robot_max_angular_velocity) - self.q[3] * (rate[1] * self.robot_max_angular_velocity)) * self.dt
        self.q[2] = self.q[2] + 0.5 * (self.q[0] * (rate[1] * self.robot_max_angular_velocity) - self.q[1] * (rate[2] * self.robot_max_angular_velocity) + self.q[3] * (rate[0] * self.robot_max_angular_velocity)) * self.dt
        self.q[3] = self.q[3] + 0.5 * (self.q[0] * (rate[2] * self.robot_max_angular_velocity) + self.q[1] * (rate[1] * self.robot_max_angular_velocity) - self.q[2] * (rate[0] * self.robot_max_angular_velocity)) * self.dt
        return self.q
    
    def acados_setup_model(self, kinematics, model_name):

        def fill_in_acados_model(x, u, p, kinematics, name):
            model = AcadosModel()
            model.disc_dyn_expr = kinematics 
            model.x = x
            model.u = u
            model.p = p
            model.name = name
            return model

        acados_models = {}
        kinematics_ = kinematics
        acados_models = fill_in_acados_model(x=self.x, u=self.u, p=[], kinematics=kinematics_, name=model_name)
        return acados_models
        
    def set_reference_state(self, x_target):

        target = np.array((x_target))
        ref_u = np.array([0, 0, 0, 0, 0, 0])
        ref = np.concatenate((target, ref_u), axis=0)
        for j in range(self.N):
            # {x y z qw qx qy qz uu uv uw up uq ur}
            self.acados_ocp_solver.cost_set(j, "yref", ref)

        # {x y z qw qx qy qz}
        self.acados_ocp_solver.cost_set(self.N, "yref", target)

    def run_optimize(self, initial_state):

        x_init = initial_state
        x_init = np.stack(x_init)
        self.acados_ocp_solver.set(0, "lbx", x_init)
        self.acados_ocp_solver.set(0, "ubx", x_init)

        t = time.time()
        status = self.acados_ocp_solver.solve()
        opt_time = time.time() - t

        cost = self.acados_ocp_solver.get_cost()
        if opt_time > self.max_opt_time:
            self.max_opt_time = opt_time

        self.sum_opt_time += opt_time
        self.num_opt += 1
        u_opt_acados = np.ndarray((self.N, 6))

        for i in range(self.N):
            u_opt_acados[i, :] = self.acados_ocp_solver.get(i, "u")

        next_u = np.array(u_opt_acados[0, :])
        return next_u