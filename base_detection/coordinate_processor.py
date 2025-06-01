#!/usr/bin/env python3

"""
Coordinate Processor Node for Base Detection System.

This module implements a ROS2 node that processes and clusters detected base coordinates,
saves them as setpoints, and provides a shutdown service. It uses K-means clustering
to identify unique base positions from multiple detections.

The node:
- Processes incoming delta positions
- Clusters positions to identify unique bases
- Saves setpoints to YAML configuration
- Publishes unique positions for navigation
- Provides graceful shutdown service

Dependencies:
    - ROS2
    - NumPy
    - scikit-learn
    - PyYAML
    - geometry_msgs
    - std_srvs
"""
import traceback
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseArray, Pose
import numpy as np
from sklearn.cluster import KMeans
from base_detection.variables import (
    DETECTED_COORDINATES_TOPIC,
    UNIQUE_POSITIONS_TOPIC
)

class CoordinateProcessor(Node):
    """
    A ROS2 node for processing and clustering base coordinates.

    This class processes incoming delta positions from base detections,
    clusters them to identify unique base locations, and saves them as
    setpoints for navigation.

    Attributes:
        positions_list (list): Raw detected positions
        unique_positions (list): Clustered unique base positions
        expected_bases (int): Expected number of bases in the arena
    """

    def __init__(self):
        """
        Initialize the CoordinateProcessor node.

        Sets up:
        - Subscription to delta positions
        - Publisher for unique positions
        - Shutdown service
        - Data structures for position processing
        """
        super().__init__('coordinate_processor')

        self.subscription = self.create_subscription(
            Point,
            DETECTED_COORDINATES_TOPIC,
            self.delta_position_callback,
            10)

        self.unique_positions_publisher = self.create_publisher(
            PoseArray,
            UNIQUE_POSITIONS_TOPIC,
            10)


        self.positions_list = []
        self.unique_positions = []
        self.expected_bases = 3

    def delta_position_callback(self, msg):
        """
        Process incoming delta position messages.

        Adds new position measurements to the positions list for
        later clustering and processing.

        Args:
            msg (geometry_msgs.msg.Point): Delta position message
        """
        try:
            position = [msg.x, msg.y]
            self.positions_list.append(position)
            self.get_logger().info(f"Received position: x={msg.x:.3f}, y={msg.y:.3f}")
        except Exception as e:
            self.get_logger().error(f"Error in delta_position_callback: {e}")
            traceback.print_exc()

    def process_positions(self):
        """
        Process collected positions to identify unique bases.

        Uses K-means clustering to identify unique base positions from
        collected measurements. Publishes results and saves setpoints
        to configuration file.

        Note:
            Requires at least expected_bases number of positions
            to perform clustering.
        """
        if len(self.positions_list) < self.expected_bases:
            self.get_logger().warn("Not enough positions collected to perform clustering.")
            return

        positions_array = np.array(self.positions_list)

        try:
            kmeans = KMeans(n_clusters=self.expected_bases)
            kmeans.fit(positions_array)
            self.unique_positions = kmeans.cluster_centers_.tolist()

            new_setpoints = []
            pose_array = PoseArray()
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.header.frame_id = 'map'

            for pos in self.unique_positions:
                pose = Pose()
                pose.position.x = pos[0]
                pose.position.y = pos[1]
                pose.position.z = 0.0
                self.get_logger().info(f"Prepared coordinate ({pos[0]:.3f}, {pos[1]:.3f}, -1.5) to goto setpoints.")

                new_setpoint = [pos[0], pos[1], -1.5]
                new_setpoints.append(new_setpoint)
                pose_array.poses.append(pose)

            self.unique_positions_publisher.publish(pose_array)
            self.get_logger().info(f"Published {len(self.unique_positions)} unique positions.")

        except Exception as e:
            self.get_logger().error(f"Error during clustering: {e}")
            traceback.print_exc()

def main(args=None):
    """
    Main entry point for the coordinate processor node.

    Args:
        args: Command line arguments (unused)
    """
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
