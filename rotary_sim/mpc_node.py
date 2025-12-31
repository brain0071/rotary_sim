import rclpy 
from rclpy.node import Node
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseStamped
import numpy as np
from rotary_sim.controller.ros_mpc import ROS_MPC
from geometry_msgs.msg import Wrench, Vector3
from nav_msgs.msg import Odometry
import random

class Rotary_MPCWrapper(Node):
    
    def __init__(self):

        super().__init__('mpc_node')
        
        self.running = True
        self.load_params()
        
        self.q_cost = np.array([0.01, 0.01, 0.01, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
        self.r_cost = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        self.ref_pos = np.zeros((3,))
        self.ref_att = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.ref_a = np.zeros((3,))
        self.u_indi = np.zeros((3,))
        self.sim_a = np.zeros((3,))
        
        self.mpc_dt = 1 / self.mpc_freq
        self.timer_mpc = self.create_timer(self.mpc_dt, self.mpc_callback)
        
        self.indi_dt = 1 /self.indi_freq
        self.timer_indi = self.create_timer(self.indi_dt, self.indi_callback)
        
        self.ros_mpc = ROS_MPC(self.mass, self.inertia, self.add_mass, self.quadratic_damp, self.max_force_moment, self.max_vel,
                 self.n_nodes, self.mpc_dt, self.q_cost, self.r_cost)

        self.subscription = self.create_subscription(Odometry, '/reference', self.reference_callback, 10)
        self.pose_sim_pub = self.create_publisher(Odometry, 'sim_pose', 10)
        self.control_pub = self.create_publisher(Wrench, "/u", 10)
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

            self.declare_parameter('mpc_freq', 20)
            self.declare_parameter('indi_freq', 100)
            self.declare_parameter('t_horizon', 0.5)
            self.declare_parameter('n_nodes', 5)
            self.declare_parameter('exp_type', 'real')

            self.mpc_freq = self.get_parameter('mpc_freq').value
            self.indi_freq = self.get_parameter('indi_freq').value
            self.t_horizon = self.get_parameter('t_horizon').value
            self.n_nodes = self.get_parameter('n_nodes').value
        
        
            self.get_logger().info(f"Node name is: {self.get_name()}")
        
            self.get_logger().info(
                "\n===== Loaded Parameters =====\n"
                f"  motor_max        : {self.motor_max}\n"
                f"  mass             : {self.mass}\n"
                f"  inertia          : {self.inertia}\n"
                f"  add_mass         : {self.add_mass}\n"
                f"  quadratic_damp   : {self.quadratic_damp}\n"
                f"  max_force_moment : {self.max_force_moment}\n"
                f"  mpc_freq     : {self.mpc_freq}\n"
                f"  indi_freq        : {self.indi_freq}\n"
                f"  t_horizon        : {self.t_horizon}\n"
                f"  n_nodes          : {self.n_nodes}\n"
                "=============================="
                )
            
    def reference_callback(self, msg):
        
        self.ref_pos_delta = [0, 0, 0]
        self.ref_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,]
        self.ref_att = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,]
        self.ref_rate = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        self.ref = np.concatenate((self.ref_pos_delta, np.concatenate((self.ref_pos, np.concatenate((self.ref_att, self.ref_rate))))))
        self.ros_mpc.set_reference(self.ref)
    
    def indi_callback(self):
       
        force_u = (self.ref_a[0] - self.sim_a[0]) * self.mass[0] + self.uu * self.max_force_moment[0]
        force_v = (self.ref_a[1] - self.sim_a[1]) * self.mass[0] + self.uv * self.max_force_moment[1]
        force_w = (self.ref_a[2] - self.sim_a[2]) * self.mass[0] + self.uw * self.max_force_moment[2]

        self.uu = force_u / self.max_force_moment[0]
        self.uv = force_v / self.max_force_moment[1]
        self.uw = force_w / self.max_force_moment[2]
        
        if self.uu > 0.6:
            self.uu = 0.6

        if self.uu < -0.6:
            self.uu = -0.6
            
        if self.uv > 0.6:
            self.uv = 0.6

        if self.uv < -0.6:
            self.uv = -0.6
            
        if self.uw > 0.6:
            self.uw = 0.6

        if self.uw < -0.6:
            self.uw = -0.6
        
        self.u_indi = np.array([self.uu, self.uv, self.uw])
        self.sim_a = self.ros_mpc.simulate_indi(self.u_indi)            
            
            
    
    def control_callback(self):
        
        if self.running == True:
            u = self.ros_mpc.optimize()
            
            # disturbance = np.random.uniform(-0.3, 0.3, size=6)
            # u_d = u + disturbance
            ref_ax = u[0] / self.mpc_dt * self.max_vel[0]
            ref_ay = u[1] / self.mpc_dt * self.max_vel[1]
            ref_az = u[2] / self.mpc_dt * self.max_vel[2]
            self.ref_a = np.array([ref_ax, ref_ay, ref_az])
            self.ros_mpc.simulate(self.mpc_dt, self.sim_a, u)
            sim_x = self.ros_mpc.get_current_sim_state()
            sim_pose = PoseStamped()
            sim_pose.pose.position.x = sim_x[3]
            sim_pose.pose.position.y = sim_x[4]
            sim_pose.pose.position.z = sim_x[5]
            sim_pose.pose.orientation.w, sim_pose.pose.orientation.x, sim_pose.pose.orientation.y, sim_pose.pose.orientation.z = sim_x[6:10]
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
    mpc_wrapper = Rotary_MPCWrapper()
    rclpy.spin(mpc_wrapper)
    mpc_wrapper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()