from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    
    return LaunchDescription([
        Node(
            package='rotary_sim',
            executable='mpc_node',
            name='mpc_node',
            output='screen',
            emulate_tty=True,
            parameters=[os.path.join(
                os.path.dirname(__file__), '../config/params.yaml')],
        ),
        
        Node(
            package='reference',
            executable='reference_node',
            name='reference_node',
            output='screen',
            emulate_tty=True,
        ),
        
        # Node(
        #     package='ocean_data',
        #     executable='log_data',  
        #     name='log_data',
        #     output='screen',
        #     emulate_tty=True,
        # ),
    ])

