#!/usr/bin/env python3

import os
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseArray, Pose
import numpy as np
from sklearn.cluster import KMeans
from std_srvs.srv import Trigger  # Import for the service

# Define the relative path to the configuration file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'goto_setpoints.yaml')

class MyDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, indentless=False)

def represent_list(dumper, data):
    # If it's a list of numbers, use block style; otherwise, use flow style
    if all(isinstance(i, list) for i in data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=None)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

MyDumper.add_representer(list, represent_list)

def add_setpoint_to_file(file_path, new_setpoints):
    # Read the existing setpoints from the file
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Append the new setpoints to the list
    config['setpoints'].extend(new_setpoints)

    # Write the updated setpoints back to the file with the correct block style
    with open(file_path, 'w') as file:
        yaml.dump(config, file, Dumper=MyDumper, default_flow_style=False)


class CoordinateProcessor(Node):
    def __init__(self):
        super().__init__('coordinate_processor')

        # Subscription to delta_position topic
        self.subscription = self.create_subscription(
            Point,
            'delta_position',
            self.delta_position_callback,
            10)

        # Publisher for unique positions
        self.unique_positions_publisher = self.create_publisher(
            PoseArray,
            'unique_positions',
            10)

        # Service to trigger shutdown
        self.shutdown_service = self.create_service(
            Trigger,
            'coordinate_processor/shutdown',
            self.shutdown_service_callback)

        # Data structures
        self.positions_list = []  # List of received positions
        self.unique_positions = []  # List of unique positions

        # Parameters
        self.expected_bases = 3  # Number of bases in the arena

    def delta_position_callback(self, msg):
        # Receive Point message and append to positions_list
        position = [msg.x, msg.y]
        self.positions_list.append(position)
        self.get_logger().info(f"Received position: x={msg.x:.3f}, y={msg.y:.3f}")

    def process_positions(self):
        # Process positions_list to update unique_positions
        if len(self.positions_list) < self.expected_bases:
            self.get_logger().warn("Not enough positions collected to perform clustering.")
            return

        # Convert positions_list to numpy array
        positions_array = np.array(self.positions_list)

        # Perform KMeans clustering to group positions into expected number of bases
        try:
            kmeans = KMeans(n_clusters=self.expected_bases)
            kmeans.fit(positions_array)
            self.unique_positions = kmeans.cluster_centers_.tolist()

            # Prepare new setpoints to be added to the YAML file
            new_setpoints = []

            # Prepare PoseArray message for publishing
            pose_array = PoseArray()
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.header.frame_id = 'map'  # Replace with appropriate frame ID

            for pos in self.unique_positions:
                pose = Pose()
                pose.position.x = pos[0]
                pose.position.y = pos[1]
                pose.position.z = 0.0  # Assuming ground level
                # Orientation can be left at default (zero rotation)
                self.get_logger().info(f"Prepared coordinate ({pos[0]:.3f}, {pos[1]:.3f}, -1.5) to goto setpoints.")

                new_setpoint = [pos[0], pos[1], -1.5]
                new_setpoints.append(new_setpoint)

                pose_array.poses.append(pose)

            # Publish unique_positions
            self.unique_positions_publisher.publish(pose_array)  # Sends to drone_navigator
            self.get_logger().info(f"Published {len(self.unique_positions)} unique positions.")

            # Add all new setpoints to the YAML file at once
            add_setpoint_to_file(CONFIG_PATH, new_setpoints)
            self.get_logger().info(f"Saved {len(new_setpoints)} setpoints to {CONFIG_PATH}.")

        except Exception as e:
            self.get_logger().error(f"Error during clustering: {e}")
    
    def destroy_node(self):
        # Override destroy_node to process positions before shutting down
        self.get_logger().info("Node is shutting down. Processing positions before exit...")
        self.process_positions()
        super().destroy_node()

    def shutdown_service_callback(self, request, response):
        self.get_logger().info("Shutdown service called. Preparing to shut down...")
        # Call the destroy_node method to process positions and shut down
        self.destroy_node()
        response.success = True
        response.message = "CoordinateProcessor is shutting down."
        return response


def main(args=None):
    rclpy.init(args=args)
    coordinate_processor = CoordinateProcessor()

    try:
        rclpy.spin(coordinate_processor)
    except KeyboardInterrupt:
        pass
    finally:
        coordinate_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
