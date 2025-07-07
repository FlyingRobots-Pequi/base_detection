#!/usr/bin/env python3

"""
Coordinate Processor Node for Base Detection System.

This module implements a ROS2 node that processes and clusters detected base coordinates,
saves them as setpoints, and provides a shutdown service. It uses K-means clustering
to identify unique base positions from multiple absolute position detections.

The node:
- Processes incoming absolute positions
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
    ABSOLUTE_POINTS_TOPIC,
    UNIQUE_POSITIONS_TOPIC,
    INITIAL_BASE_EXCLUSION_RADIUS,
    INITIAL_BASE_X,
    INITIAL_BASE_Y
)


class CoordinateProcessor(Node):
    """
    A ROS2 node for processing and clustering base coordinates.

    This class processes incoming absolute positions from base detections,
    filters out positions too close to the initial base location to focus
    on remote bases, clusters them to identify unique base locations, and 
    saves them as setpoints for navigation.

    The node implements intelligent filtering to exclude detections near the
    origin (initial base position) with a configurable exclusion radius,
    ensuring only meaningful remote base detections are processed.

    Attributes:
        positions_list (list): Raw detected absolute positions (filtered)
        unique_positions (list): Clustered unique base positions
        expected_bases (int): Expected number of bases in the arena
    """

    def __init__(self):
        """
        Initialize the CoordinateProcessor node.

        Sets up:
        - Subscription to absolute positions
        - Publisher for unique positions
        - Shutdown service
        - Data structures for position processing
        """
        super().__init__("coordinate_processor")

        self.absolute_subscription = self.create_subscription(
            Point,
            ABSOLUTE_POINTS_TOPIC,
            self.absolute_position_callback,
            10)

        self.unique_positions_publisher = self.create_publisher(
            PoseArray, UNIQUE_POSITIONS_TOPIC, 10
        )

        self.positions_list = []
        self.unique_positions = []
        self.expected_bases = 5

    def is_near_initial_base(self, x, y):
        """
        Check if a position is too close to the initial base location.
        
        Args:
            x (float): X coordinate to check
            y (float): Y coordinate to check
            
        Returns:
            bool: True if position is within exclusion radius of initial base
        """
        distance = np.sqrt((x - INITIAL_BASE_X)**2 + (y - INITIAL_BASE_Y)**2)
        return distance <= INITIAL_BASE_EXCLUSION_RADIUS

    def absolute_position_callback(self, msg):
        """
        Process incoming absolute position messages.

        Filters out positions too close to the initial base and adds valid
        position measurements to the positions list for later clustering and processing.

        Args:
            msg (geometry_msgs.msg.Point): Absolute position message
        """
        try:
            # Filter out positions too close to initial base
            if self.is_near_initial_base(msg.x, msg.y):
                distance_to_origin = np.sqrt(msg.x**2 + msg.y**2)
                self.get_logger().info(f"Filtering out position near initial base: ({msg.x:.3f}, {msg.y:.3f}) - distance: {distance_to_origin:.3f}m < {INITIAL_BASE_EXCLUSION_RADIUS:.3f}m")
                return
                
            position = [msg.x, msg.y]
            self.positions_list.append(position)
            self.get_logger().info(f"Received and accepted remote base position: ({msg.x:.3f}, {msg.y:.3f})")
            
            if len(self.positions_list) >= self.expected_bases:
                self.process_positions()
                
        except Exception as e:
            self.get_logger().error(f"Error in absolute_position_callback: {e}")
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
            self.get_logger().warn(
                f"Not enough positions collected to perform clustering. Expected {self.expected_bases} positions, but got {len(self.positions_list)}."
            )
            return

        positions_array = np.array(self.positions_list)
        try:
            kmeans = KMeans(n_clusters=self.expected_bases)
            kmeans.fit(positions_array)
            self.unique_positions = kmeans.cluster_centers_.tolist()

        # Sort positions by distance from origin (0,0,0) - closest first
            self.unique_positions.sort(key=lambda pos: np.sqrt(pos[0]**2 + pos[1]**2))

            new_setpoints = []
            pose_array = PoseArray()
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.header.frame_id = 'map'

            for i, pos in enumerate(self.unique_positions):
                distance_from_origin = np.sqrt(pos[0]**2 + pos[1]**2)
                self.get_logger().info(f"Position {i+1}: ({pos[0]:.3f}, {pos[1]:.3f}) - distance: {distance_from_origin:.3f}m")
                
                pose = Pose()
                pose.position.x = pos[0]
                pose.position.y = pos[1]
                pose.position.z = 0.0

            new_setpoint = [pos[0], pos[1], -1.5]
            new_setpoints.append(new_setpoint)
            pose_array.poses.append(pose)

            self.unique_positions_publisher.publish(pose_array)
            self.get_logger().info(f"List of unique positions: {[f'({pos[0]:.3f}, {pos[1]:.3f})' for pos in self.unique_positions]}")
            self.get_logger().info(f"Published {len(self.unique_positions)} unique positions.")
        except Exception as e:
            self.get_logger().error(f"Error in process_positions: {e}")
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


if __name__ == "__main__":
    main()
