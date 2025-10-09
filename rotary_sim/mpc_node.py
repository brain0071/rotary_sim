import rclpy 
from rclpy.node import Node
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseStamped
import numpy as np
from rotary_sim.controller.ros_mpc import ROS_MPC
from geometry_msgs.msg import Wrench, Vector3
from nav_msgs.msg import Odometry

class Rotary_MPCWrapper(Node):
    
    def __init__(self):

        super().__init__('mpc_node')
        
        self.running = True
        self.load_params()
        
        self.q_cost = np.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        self.r_cost = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        self.dt = 1 / self.control_freq
        self.timer = self.create_timer(self.dt, self.control_callback)

        self.ref_pos = np.zeros((3,))
        self.ref_att = np.array([1.0, 0.0, 0.0, 0.0])
        self.ref_vel = np.zeros((3,))
        self.ref_rate = np.zeros((3,))

        self.ros_mpc = ROS_MPC(self.mass, self.inertia, self.add_mass, self.quadratic_damp, self.max_force_moment, self.t_horizon,
                 self.n_nodes, self.q_cost, self.r_cost)

        self.subscription = self.create_subscription(Odometry, '/reference', self.reference_callback, 10)
        self.pose_sim_pub = self.create_publisher(Odometry, 'sim_pose', 10)
        self.control_pub = self.create_publisher(Wrench, "/u", 10)
        self.create_service(SetBool, "stop_signal", self.stop_callback)
    
    def load_params(self):
        
        self.declare_parameter('mass', [0.0])
        self.declare_parameter('inertia', [0.0, 0.0, 0.0])
        self.declare_parameter('add_mass', [0.0]*6)
        self.declare_parameter('quadratic_damp', [0.0]*6)
        self.declare_parameter('max_force_moment', [0.0]*6)
   
        # Read parameters
        self.mass = self.get_parameter('mass').value
        self.inertia = self.get_parameter('inertia').value
        self.add_mass = self.get_parameter('add_mass').value
        self.quadratic_damp = self.get_parameter('quadratic_damp').value
        self.max_force_moment = self.get_parameter('max_force_moment').value

        self.declare_parameter('control_freq', 20)
        self.declare_parameter('t_horizon', 0.5)
        self.declare_parameter('n_nodes', 5)

        self.control_freq = self.get_parameter('control_freq').value
        self.t_horizon = self.get_parameter('t_horizon').value
        self.n_nodes = self.get_parameter('n_nodes').value
        
        self.get_logger().info(f"Node name is: {self.get_name()}")
        
        self.get_logger().info(
            "\n===== Loaded Parameters =====\n"
            f"  mass             : {self.mass}\n"
            f"  inertia          : {self.inertia}\n"
            f"  add_mass         : {self.add_mass}\n"
            f"  quadratic_damp      : {self.quadratic_damp}\n"
            f"  max_force_moment : {self.max_force_moment}\n"
            f"  control_freq     : {self.control_freq}\n"
            f"  t_horizon        : {self.t_horizon}\n"
            f"  n_nodes          : {self.n_nodes}\n"
            "=============================="
            )
    
    def reference_callback(self, msg):

        self.ref_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,]
        self.ref_att = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,]
        self.ref_vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        self.ref_rate = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        self.ref = np.concatenate((self.ref_pos, np.concatenate((self.ref_att, np.concatenate((self.ref_vel, self.ref_rate))))))
        self.ros_mpc.set_reference(self.ref)
    


    def control_callback(self):
        
        if self.running == True:
            u = self.ros_mpc.optimize()
            self.ros_mpc.simulate(self.dt, u)
            sim_cur_state = self.ros_mpc.get_current_sim_state()

            sim_cur_p = Odometry()
            sim_cur_p.pose.pose.position.x, sim_cur_p.pose.pose.position.y, sim_cur_p.pose.pose.position.z = sim_cur_state[0: 3]
            sim_cur_p.pose.pose.orientation.w, sim_cur_p.pose.pose.orientation.x, sim_cur_p.pose.pose.orientation.y, sim_cur_p.pose.pose.orientation.z = sim_cur_state[3:7]
            sim_cur_p.twist.twist.linear.x, sim_cur_p.twist.twist.linear.y, sim_cur_p.twist.twist.linear.z = sim_cur_state[7:10]
            sim_cur_p.twist.twist.angular.x, sim_cur_p.twist.twist.angular.y, sim_cur_p.twist.twist.angular.z = sim_cur_state[10:13]

            self.pose_sim_pub.publish(sim_cur_p)

            control = Wrench()    
            control.force = Vector3(x=float(u[0]), y=float(u[1]), z=float(u[2]))
            control.torque = Vector3(x=float(u[3]), y=float(u[4]), z=float(u[5]))
            self.control_pub.publish(control)
            
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