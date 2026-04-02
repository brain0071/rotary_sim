import rclpy 
from rclpy.node import Node
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseStamped
import numpy as np
from rotary_sim.controller.ros_mpc import ROS_MPC
from geometry_msgs.msg import Wrench, Vector3, TwistStamped
from nav_msgs.msg import Odometry

class Rotary_Cascaded_MPCWrapper(Node):
    
    def __init__(self):

        super().__init__('mpc_node')
        
        self.running = True
        self.load_params()
        
        # {x, y, z, qw, qx, qy, qz}
        self.kinematics_q_cost = np.array([1, 1, 1, 1, 0.5, 0.5, 0.5])
        # {u, v, w, p, q ,r}
        self.kinematics_r_cost = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        # {u v w p q r}
        self.dynamics_q_cost = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # {uu uv uw up uq ur}
        self.dynamics_r_cost = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


        self.kinematics_dt = 1/ self.kinematics_control_freq
        self.kinematics_timer = self.create_timer(self.kinematics_dt, self.run_kinematics_MPC)

        self.dynamics_dt = 1/ self.dynamics_control_freq
        self.dynamics_timer = self.create_timer(self.dynamics_dt, self.run_dynamics_MPC)
        
        self.ref_pos = np.zeros((3,))
        self.ref_att = np.array([1.0, 0.0, 0.0, 0.0])
    
        self.ros_mpc = ROS_MPC(self.mass, self.inertia, self.add_mass, self.quadratic_damp, self.max_force_moment,   
                               self.max_velocity, self.max_angular_velocity, 
                               self.kinematics_n_nodes, self.kinematics_q_cost, self.kinematics_r_cost, self.kinematics_t_horizon, 
                               self.dynamics_n_nodes, self.dynamics_q_cost, self.dynamics_r_cost, self.dynamics_t_horizon)
        
        self.kinematics_u = np.zeros((6,))
        self.dynamics_u = np.zeros((6,))
        self.sim_v = np.zeros((6,))

        self.subscription = self.create_subscription(Odometry, '/reference', self.reference_callback, 10)
        self.pose_sim_pub = self.create_publisher(PoseStamped, 'sim_pose', 10)
        self.vel_sim_pub = self.create_publisher(TwistStamped, 'sim_velocity', 10)

        self.control_kinematics = self.create_publisher(Wrench, "/u_kinematics", 10)
        self.control_dynamics = self.create_publisher(Wrench, "/u_dynamics", 10)
        self.create_service(SetBool, "stop_signal", self.stop_callback)
    
    def load_params(self):
        
        self.declare_parameter('mass', [0.0])
        self.declare_parameter('inertia', [0.0, 0.0, 0.0])
        self.declare_parameter('add_mass', [0.0]*6)
        self.declare_parameter('quadratic_damp', [0.0]*6)
        self.declare_parameter('max_force_moment', [0.0]*6)
        self.declare_parameter('max_acceleration', 0.5)
        self.declare_parameter('max_angular_velocity', 0.5)     
   
        # Read parameters
        self.mass = self.get_parameter('mass').value
        self.inertia = self.get_parameter('inertia').value
        self.add_mass = self.get_parameter('add_mass').value
        self.quadratic_damp = self.get_parameter('quadratic_damp').value
        self.max_force_moment = self.get_parameter('max_force_moment').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
    
        
         # MPC
        self.declare_parameter('kinematics_control_freq', 20)
        self.declare_parameter('kinematics_n_nodes', 10)
        self.declare_parameter('kinematics_t_horizon', 0.1)
        self.declare_parameter('dynamics_control_freq', 100)
        self.declare_parameter('dynamics_n_nodes', 10)
        self.declare_parameter('dynamics_t_horizon', 0.1)
        
        self.kinematics_control_freq = self.get_parameter('kinematics_control_freq').value
        self.kinematics_n_nodes = self.get_parameter('kinematics_n_nodes').value
        self.kinematics_t_horizon = self.get_parameter('kinematics_t_horizon').value
        self.dynamics_control_freq = self.get_parameter('dynamics_control_freq').value
        self.dynamics_n_nodes = self.get_parameter('dynamics_n_nodes').value
        self.dynamics_t_horizon = self.get_parameter('dynamics_t_horizon').value
        
        self.get_logger().info(f"Node name is: {self.get_name()}")
        
        self.get_logger().info(
            "\n===== Loaded Parameters =====\n"
            f"  mass             : {self.mass}\n"
            f"  inertia          : {self.inertia}\n"
            f"  add_mass         : {self.add_mass}\n"
            f"  quadratic_damp      : {self.quadratic_damp}\n"
            f"  max_force_moment : {self.max_force_moment}\n"
            f"  max_velocity      : {self.max_velocity}\n"
            f"  max_angular_velocity : {self.max_angular_velocity}\n"
            f"  kinematics_control_freq     : {self.kinematics_control_freq}\n"
            f"  kinematics_n_nodes        : {self.kinematics_n_nodes}\n"
            f"  kinematics_t_horizon        : {self.kinematics_t_horizon}\n"
            f"  dynamics_control_freq          : {self.dynamics_control_freq}\n"
            f"  dynamics_n_nodes          : {self.dynamics_n_nodes}\n"
            f"  dynamics_t_horizon        : {self.dynamics_t_horizon}\n"
            "=============================="
            )
    
    def reference_callback(self, msg):

        self.ref_pos_delta = [0, 0, 0]
        self.ref_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,]
        self.ref_att = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,]
        self.ref = np.concatenate((self.ref_pos_delta, np.concatenate((self.ref_pos, self.ref_att))))
        self.ros_mpc.set_kinematics_reference(self.ref)
    
    def run_kinematics_MPC(self):
        
        if self.running:
            # (u, v, w, p, q, r)
            u = self.ros_mpc.kinematics_optimize()
            self.kinematics_u = u

            control = Wrench()
            control.force = Vector3(x=u[0], y=u[1], z=u[2])
            control.torque = Vector3(x=u[3], y=u[4], z=u[5])
            self.control_kinematics.publish(control)
        
            self.ros_mpc.set_dynamics_reference(self.kinematics_u)   
            self.ros_mpc.kinematics_simulate(self.kinematics_dt, self.sim_v)
            
            sim_cur_state = self.ros_mpc.get_kinematics_sim_state()
            sim_cur_p = PoseStamped()
            sim_cur_p.pose.position.x, sim_cur_p.pose.position.y, sim_cur_p.pose.position.z = sim_cur_state[0:3]
            sim_cur_p.pose.orientation.w, sim_cur_p.pose.orientation.x, sim_cur_p.pose.orientation.y, sim_cur_p.pose.orientation.z = sim_cur_state[3:6]
            self.pose_sim_pub.publish(sim_cur_p)


    # dynamics
    def run_dynamics_MPC(self):

        if self.running:
            # (uu, uv, uw, up, uq, ur)
            u = self.ros_mpc.dynamics_optimize()
            self.dynamics_u = u
            
            control = Wrench()
            control.force = Vector3(x=u[0], y=u[1], z=u[2])
            control.torque = Vector3(x=u[3], y=u[4], z=u[5])
            self.control_dynamics.publish(control)

            self.ros_mpc.dynamics_simulate(self.dynamics_dt, self.dynamics_u)
            sim_cur_state = self.ros_mpc.get_dynamics_sim_state()
        
            self.sim_v[:3] += sim_cur_state[0:3] 
            self.sim_v[3:6] = sim_cur_state[3:6]
            
            velocity_msg = TwistStamped()
            velocity_msg.header.stamp = self.get_clock().now().to_msg()  
            velocity_msg.twist.linear = Vector3(x=self.sim_v[0], y=self.sim_v[1], z=self.sim_v[2])
            velocity_msg.twist.angular = Vector3(x=self.sim_v[3], y=self.sim_v[4], z=self.sim_v[5])
            self.vel_sim_pub.publish(velocity_msg)

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
    mpc_wrapper = Rotary_Cascaded_MPCWrapper()
    rclpy.spin(mpc_wrapper)
    mpc_wrapper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()