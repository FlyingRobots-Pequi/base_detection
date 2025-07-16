from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE_NAME = 'base_detection'
NODES_NAMES = ['base_detection', 'coordinate_receiver', 'coordinate_processor']

def generate_launch_description():
    output_arg = DeclareLaunchArgument(
        'output', default_value='screen',
        description='Define onde o output dos nós será exibido (screen ou log)'
    )

    node_args = dict(
        package=PACKAGE_NAME,
        output=LaunchConfiguration('output'),
        respawn=True,
        respawn_delay=1.0
    )

    nodes = [
        Node(
            **node_args,
            executable=node_name,
            name=node_name
        )
        for node_name in NODES_NAMES
    ]

    return LaunchDescription([
        output_arg,
        *nodes
    ])
