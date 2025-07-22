import rclpy 
from rclpy.node import Node
from std_srvs.srv import SetBool
from mavros_msgs.msg import ActuatorControl
from mavros_msgs.msg import Altitude
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped
import numpy as np


class MPC_INDI_Wrapper(Node):
    
    def __init__(self):

        super().__init__('mpc_indi_node')
        self.controller_state = True
        self.load_params()
        
        self.dt = 1 / self.control_freq
        self.timer = self.create_timer(self.dt, self.control_callback)

        self.gripper_value = 1100.0
        self.light_value = 1100.0
        
        self.ref_z = 0.0
        self.ref_att = np.array([1.0, 0.0, 0.0, 0.0])
        self.ref_acc = np.zeros((2,))

        self.motor_pub = self.create_publisher(ActuatorControl, "/mavros/actuator_control", 10)
        self.gripper_light_pub = self.create_publisher(Altitude, "/mavros/gripper_light_control", 10)
        self.create_subscription(Bool, '/light', self.light_callback, 10)
        self.create_subscription(Float32, '/gripper', self.gripper_callback, 10)
        self.subscription = self.create_subscription(PoseStamped, '/target_pose', self.reference_callback, 10)
        self.create_service(SetBool, "stop_signal", self.stop_callback)

        
    
    def load_params(self):
        
        self.declare_parameter('motor_max', [0.0])
        self.declare_parameter('mass', [0.0])
        self.declare_parameter('inertia', [0.0, 0.0, 0.0])
        self.declare_parameter('add_mass', [0.0]*6)
        self.declare_parameter('linear_damp', [0.0]*6)
        self.declare_parameter('max_force_moment', [0.0]*6)
        self.declare_parameter('max_accel', [0.0]*3)
   
        # Read parameters
        self.motor_max = self.get_parameter('motor_max').value
        self.mass = self.get_parameter('mass').value
        self.inertia = self.get_parameter('inertia').value
        self.add_mass = self.get_parameter('add_mass').value
        self.linear_damp = self.get_parameter('linear_damp').value
        self.max_force_moment = self.get_parameter('max_force_moment').value
        self.max_accel = self.get_parameter('max_accel').value

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
            f"  linear_damp      : {self.linear_damp}\n"
            f"  max_force_moment : {self.max_force_moment}\n"
            f"  control_freq     : {self.control_freq}\n"
            f"  t_horizon        : {self.t_horizon}\n"
            f"  n_nodes          : {self.n_nodes}\n"
            f"  exp_type         : {self.exp_type}\n"
            "=============================="
            )
    
    def reference_callback(self, msg):
        
        self.ref_z = msg.pose.position.z
        self.ref_att = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        self.ref_acc = [msg.pose.position.x, msg.pose.position.y]
        


    
    def light_callback(self, msg):
        
        if msg.data == True:
            self.light_value = 1900.0
        else:
            self.light_value = 1100.0

    
    def gripper_callback(self, msg):
        
        self.gripper_value = msg.data



    def control_callback(self):
        
        if self.controller_state == True:
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
            pass

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