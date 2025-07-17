from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE_NAME = 'base_detection'

def generate_launch_description():
    output_arg = DeclareLaunchArgument(
        'output', default_value='screen',
        description='Define onde o output dos nós será exibido (screen ou log)'
    )

    return LaunchDescription([
        output_arg,

        Node(
            package=PACKAGE_NAME,
            executable='base_detection',
            name='base_detection_node',
            output=LaunchConfiguration('output'),
            respawn=True, 
            respawn_delay=1.0
        ),
        Node(
            package=PACKAGE_NAME,
            executable='coordinate_processor',
            name='coordinate_processor_node',
            output=LaunchConfiguration('output'),
            respawn=True, 
            respawn_delay=1.0
        ),
        Node(
            package=PACKAGE_NAME,
            executable='coordinate_receiver',
            name='coordinate_receiver_node',
            output=LaunchConfiguration('output'),
            respawn=True, 
            respawn_delay=0.5
        ),
    ])
