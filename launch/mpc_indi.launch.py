from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    
    return LaunchDescription([
        Node(
            package='rotary_mpc',
            executable='mpc_indi_node',
            name='mpc_indi_node',
            output='screen',
            emulate_tty=True,
            parameters=[os.path.join(
                os.path.dirname(__file__), '../config/params.yaml')],
        ),
    
        Node(
            package='ocean_manual',
            executable='run_manual',
            name='manual_control_node',
            output='screen',
        )
    ])

