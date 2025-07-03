import rclpy 
from rclpy.node import Node
from std_srvs.srv import SetBool
from mavros_msgs.msg import ActuatorControl
# gripper
from mavros_msgs.msg import Thrust
# light
from mavros_msgs.msg import Param


class TEST_Wrapper(Node):
    
    def __init__(self):

        super().__init__('test_node')
        self.controller_state = True
        self.motor_pub = self.create_publisher(ActuatorControl, "/mavros/actuator_control", 10)
        self.control_freq = 20
        self.dt = 1 / self.control_freq
        self.timer = self.create_timer(self.dt, self.callback)
        self.create_service(SetBool, "stop_signal", self.stop_callback)
        
    def callback(self):
        
        if self.controller_state == True:
            motor = ActuatorControl()
            motor.controls = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.motor_pub.publish(motor)
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