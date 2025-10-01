import rclpy 
from rclpy.node import Node
from std_srvs.srv import SetBool
from mavros_msgs.msg import ActuatorControl
from mavros_msgs.msg import Altitude
from std_msgs.msg import Bool, Float32, Float32MultiArray
from geometry_msgs.msg import PoseStamped
import numpy as np
from rotary_mpc.controller.ros_mpc import ROS_MPC

class MPC_INDI_Wrapper(Node):
    
    def __init__(self):

        super().__init__('mpc_indi_node')
        self.controller_state = True
        # self.sunkezhen = 0
        self.load_params()
        
         # 4-DoF control {z, roll, pitch, yaw}
        # x: {delta_z, z, qw, qx, qy, qz, p, q, r}
        # u: {delta_vy, up, uq, ur}
        self.q_cost = np.array([0.01, 1, 1, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
        self.r_cost = np.array([0.1, 0.1, 0.1, 0.1])

        self.dt = 1 / self.control_freq
        self.timer = self.create_timer(self.dt, self.control_callback)

        self.gripper_value = 1100.0
        self.light_value = 1100.0
        
        self.ref_z = 0.0
        self.ref_att = np.array([1.0, 0.0, 0.0, 0.0])

        self.ros_mpc = ROS_MPC(self.mass, self.inertia, self.add_mass, self.quadratic_damp, self.max_force_moment, self.max_vel,
                 self.n_nodes, self.dt, self.q_cost, self.r_cost, self.exp_type)

        self.subscription = self.create_subscription(PoseStamped, '/target_pose', self.reference_callback, 10)
        self.control_pub = self.create_publisher(Float32MultiArray, 'mind_array_topic', 10)

        if self.exp_type == "real":
            self.create_subscription(Bool, '/light', self.light_callback, 10)
            self.create_subscription(Float32, '/gripper', self.gripper_callback, 10)
            self.motor_pub = self.create_publisher(ActuatorControl, "/mavros/actuator_control", 10)
            self.gripper_light_pub = self.create_publisher(Altitude, "/mavros/gripper_light_control", 10)
        else:
            self.pose_sim_pub = self.create_publisher(PoseStamped, 'sim_pose', 10)

        self.create_service(SetBool, "stop_signal", self.stop_callback)
  
    
    def load_params(self):
        
        self.declare_parameter('motor_max', [0.0])
        self.declare_parameter('mass', [0.0])
        self.declare_parameter('inertia', [0.0, 0.0, 0.0])
        self.declare_parameter('add_mass', [0.0]*6)
        self.declare_parameter('quadratic_damp', [0.0]*6)
        self.declare_parameter('max_force_moment', [0.0]*6)
        self.declare_parameter('max_vel', [0.0]*3)
   
        # Read parameters
        self.motor_max = self.get_parameter('motor_max').value
        self.mass = self.get_parameter('mass').value
        self.inertia = self.get_parameter('inertia').value
        self.add_mass = self.get_parameter('add_mass').value
        self.quadratic_damp = self.get_parameter('quadratic_damp').value
        self.max_force_moment = self.get_parameter('max_force_moment').value
        self.max_vel = self.get_parameter('max_vel').value

        self.declare_parameter('control_freq', 20)
        self.declare_parameter('t_horizon', 0.5)
        self.declare_parameter('n_nodes', 5)
        self.declare_parameter('exp_type', 'real')

        self.control_freq = self.get_parameter('control_freq').value
        self.t_horizon = self.get_parameter('t_horizon').value
        self.n_nodes = self.get_parameter('n_nodes').value
        self.exp_type = self.get_parameter('exp_type').value
        
        self.get_logger().info(f"Node name is: {self.get_name()}")
        
        self.get_logger().info(
            "\n===== Loaded Parameters =====\n"
            f"  motor_max        : {self.motor_max}\n"
            f"  mass             : {self.mass}\n"
            f"  inertia          : {self.inertia}\n"
            f"  add_mass         : {self.add_mass}\n"
            f"  quadratic_damp      : {self.quadratic_damp}\n"
            f"  max_force_moment : {self.max_force_moment}\n"
            f"  control_freq     : {self.control_freq}\n"
            f"  t_horizon        : {self.t_horizon}\n"
            f"  n_nodes          : {self.n_nodes}\n"
            f"  exp_type         : {self.exp_type}\n"
            "=============================="
            )
    
    def reference_callback(self, msg):

        self.ref_z_delta = [0,]
        self.ref_z = msg.pose.position.z
        self.ref_rate = [0, 0, 0]
        self.ref_att = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        self.ref = np.concatenate((self.ref_z_delta, np.concatenate((np.array([self.ref_z]), np.concatenate((self.ref_att, self.ref_rate))))))
        self.ros_mpc.set_reference(self.ref)
    
    def light_callback(self, msg):
        
        if msg.data == True:
            self.light_value = 1900.0
        else:
            self.light_value = 1100.0

    def gripper_callback(self, msg):

        self.gripper_value = msg.data



    def control_callback(self):
        
        if self.controller_state == True:
            print("Hello World.")
            # u = {delta_vz, up, uq, ur}
            # u = self.ros_mpc.optimize()
            # EKF
            
            # msg = Float32MultiArray()
            # msg.data = [u[0], u[1], u[2], u[3]]  
            # self.control_pub.publish(msg)
            
            if self.exp_type == "real":
                motor = ActuatorControl()
                motor.controls = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                self.motor_pub.publish(motor)
                gripper_light = Altitude()
                # gripper
                gripper_light.local = self.gripper_value
                # light
                gripper_light.relative = self.light_value            
                self.gripper_light_pub.publish(gripper_light)
            else:
                az = u[0] / self.dt * self.max_vel[2]
                self.ros_mpc.simulate(self.dt, az, u)

                sim_x = self.ros_mpc.get_current_sim_state()
                sim_pose = PoseStamped()
                sim_pose.pose.position.z = sim_x[1]
                sim_pose.pose.orientation.w, sim_pose.pose.orientation.x, sim_pose.pose.orientation.y, sim_pose.pose.orientation.z = sim_x[2:6]
                self.pose_sim_pub.publish(sim_pose)

        else:
            self.get_logger().info('The controller has been stopped.')

    def stop_callback(self, request, response):
        if request.data == True:
            self.controller_state = False
            response.success = True
            response.message = 'The controller has been successfully stopped.'
            return response
        else:
            self.controller_state = True
            response.success = False
            response.message = 'The controller is running.'

def main(args=None):
    rclpy.init(args=args)
    mpc_indi_wrapper = MPC_INDI_Wrapper()
    rclpy.spin(mpc_indi_wrapper)
    mpc_indi_wrapper.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()