from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='base_detection',
            executable='package_detector_node',
            name='package_detector',
            output='screen',
            emulate_tty=True,
        ),
    ])
