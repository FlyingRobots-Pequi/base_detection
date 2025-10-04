from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

NODES_NAMES = ['base_detection', 'coordinate_receiver', 'coordinate_processor']

def generate_launch_description():
    output_arg = DeclareLaunchArgument(
        'output', default_value='screen',
        description='Define onde o output dos nós será exibido (screen ou log)',
    )
    
    # Novo argumento para controlar o nível de log
    log_level_arg = DeclareLaunchArgument(
        'log_level', default_value='info',
        description='Nível de log dos nós (debug, info, warn, error)',
    )

    node_args = dict(
        package=NODES_NAMES[0],
        output=LaunchConfiguration('output'),
        respawn=True,
        respawn_delay=1.0,
        # Adiciona argumentos de log para cada nó
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
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
        log_level_arg,
        *nodes
    ])
