from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rotary_controller',
            executable='test_node',
            name='test_node',
            output='screen',
        ),
    ])