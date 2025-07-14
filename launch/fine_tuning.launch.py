from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Argumentos que podemos passar pela linha de comando
    collinearity_weight = LaunchConfiguration('collinearity_weight', default='15.0')
    line_tolerance = LaunchConfiguration('line_tolerance', default='0.3')
    min_line_points = LaunchConfiguration('min_line_points', default='3')
    output_dir = LaunchConfiguration('output_dir', default='/root/ros2_ws/tuning_results/default_run')
    rosbag_path = LaunchConfiguration('rosbag_path', default='/path/to/your/test.db3')

    # Nó do processador de coordenadas com parâmetros configuráveis
    coord_processor_node = Node(
        package='base_detection',
        executable='coordinate_processor',
        name='coordinate_processor',
        output='screen',
        parameters=[{
            'use_line_based_clustering': True,
            'collinearity_weight': collinearity_weight,
            'line_tolerance': line_tolerance,
            'min_line_points': min_line_points,
            'output_dir': output_dir,
        }]
    )

    # Comando para tocar o rosbag
    bag_play_process = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path, '--read-ahead-queue-size', '2000'],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('collinearity_weight', default_value='15.0', description='Weight for line identity in clustering.'),
        DeclareLaunchArgument('line_tolerance', default_value='0.3', description='RANSAC line tolerance.'),
        DeclareLaunchArgument('min_line_points', default_value='3', description='Minimum points to form a line.'),
        DeclareLaunchArgument('output_dir', default_value='/root/ros2_ws/tuning_results/default_run', description='Directory to save results.'),
        DeclareLaunchArgument('rosbag_path', default_value='/path/to/your/test.db3', description='Path to the rosbag for testing.'),

        coord_processor_node,
        bag_play_process,

        # Adiciona um manipulador de eventos que desliga tudo quando o rosbag termina.
        # Isso garante que o processo de lançamento termine e não cause timeout.
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=bag_play_process,
                on_exit=[Shutdown()]
            )
        ),
    ]) 