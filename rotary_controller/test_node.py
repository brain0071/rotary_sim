import rclpy 
from rclpy.node import Node
from std_srvs.srv import SetBool
from mavros_msgs.msg import ActuatorControl
# gripper and light
from mavros_msgs.msg import Altitude
from std_msgs.msg import Bool, Float32


class TEST_Wrapper(Node):
    
    def __init__(self):

        super().__init__('test_node')
        self.controller_state = True
        self.motor_pub = self.create_publisher(ActuatorControl, "/mavros/actuator_control", 10)
        self.gripper_light_pub = self.create_publisher(Altitude, "/mavros/gripper_light_control", 10)
        
        self.gripper_value = 1100.0
        self.light_value = 1100.0

        self.create_subscription(Bool, '/light', self.light_callback, 10)
        self.create_subscription(Float32, '/gripper', self.gripper_callback, 10)
        
        self.control_freq = 20
        self.dt = 1 / self.control_freq
        self.timer = self.create_timer(self.dt, self.control_callback)
        self.create_service(SetBool, "stop_signal", self.stop_callback)
        

    
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
    test_wrapper = TEST_Wrapper()
    rclpy.spin(test_wrapper)
    test_wrapper.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()