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
- Creates visual map for rqt_image_view

Dependencies:
    - ROS2
    - NumPy
    - scikit-learn
    - PyYAML
    - OpenCV
    - geometry_msgs
    - sensor_msgs
    - std_srvs
"""
import traceback
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseArray, Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from sklearn.cluster import KMeans, DBSCAN
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
        use_dbscan (bool): Whether to use DBSCAN instead of K-means
        outlier_threshold (float): IQR multiplier for outlier detection
        cv_bridge (CvBridge): Bridge for converting OpenCV images to ROS messages
    """

    def __init__(self):
        """
        Initialize the CoordinateProcessor node.

        Sets up:
        - Subscription to absolute positions
        - Publisher for unique positions
        - Publisher for visual map
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

        # Image publisher for visual map
        self.map_image_publisher = self.create_publisher(
            Image, "/base_detection/position_map", 10
        )

        self.positions_list = []
        self.unique_positions = []
        self.expected_bases = 5
        
        # Outlier detection parameters
        self.use_dbscan = False  # Use DBSCAN for better outlier handling
        self.outlier_threshold = 1.5  # IQR multiplier for outlier detection
        self.dbscan_eps = 0.9  # DBSCAN epsilon parameter (max distance between points in cluster)
        self.dbscan_min_samples = 3  # DBSCAN minimum samples per cluster
        
        # Image visualization parameters
        self.cv_bridge = CvBridge()
        self.map_width = 800
        self.map_height = 800
        self.map_scale = 10  # pixels per meter
        self.map_center_x = self.map_width // 2
        self.map_center_y = self.map_height // 2
        
        # Ground truth base positions
        self.ground_truth_bases = [
            (-0.24, -3.23),  # BASE_1
            (0.75, -5.05),   # BASE_2
            (5.16, -5.75),   # BASE_3
            (4.37, -2.30),   # BASE_4
            (5.69, -0.25),   # BASE_5
        ]

    def world_to_image(self, x, y):
        """
        Convert world coordinates to image pixel coordinates.
        
        Args:
            x (float): World X coordinate in meters
            y (float): World Y coordinate in meters
            
        Returns:
            tuple: (pixel_x, pixel_y) in image coordinates
        """
        pixel_x = int(self.map_center_x + x * self.map_scale)
        pixel_y = int(self.map_center_y - y * self.map_scale)  # Invert Y for image coordinates
        return pixel_x, pixel_y

    def create_position_map(self, all_positions, filtered_positions, cluster_centers, cluster_labels=None):
        """
        Create a visual map of all positions, filtered positions, and cluster centers.
        
        Args:
            all_positions (np.array): All detected positions
            filtered_positions (np.array): Positions after outlier removal
            cluster_centers (list): Final cluster center positions
            cluster_labels (np.array): Cluster labels for each filtered position (optional)
            
        Returns:
            np.array: RGB image array
        """
        # Create blank image
        img = np.ones((self.map_height, self.map_width, 3), dtype=np.uint8) * 255
        
        # Define colors
        outlier_color = (128, 128, 128)  # Gray for outliers
        raw_point_color = (255, 0, 0)   # Red for raw points
        filtered_point_color = (0, 255, 0)  # Green for filtered points
        cluster_colors = [
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 255, 0),    # Yellow
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 128, 255),    # Light blue
            (255, 0, 128),    # Pink
        ]
        cluster_center_color = (0, 0, 255)  # Blue for cluster centers
        initial_base_color = (0, 0, 0)      # Black for initial base
        ground_truth_color = (255, 165, 0)  # Orange for ground truth bases
        
        # Draw grid
        grid_spacing = int(1.0 * self.map_scale)  # 1 meter grid
        for i in range(0, self.map_width, grid_spacing):
            cv2.line(img, (i, 0), (i, self.map_height), (240, 240, 240), 1)
        for i in range(0, self.map_height, grid_spacing):
            cv2.line(img, (0, i), (self.map_width, i), (240, 240, 240), 1)
            
        # Draw axes
        cv2.line(img, (self.map_center_x, 0), (self.map_center_x, self.map_height), (200, 200, 200), 2)
        cv2.line(img, (0, self.map_center_y), (self.map_width, self.map_center_y), (200, 200, 200), 2)
        
        # Draw initial base position
        init_px, init_py = self.world_to_image(INITIAL_BASE_X, INITIAL_BASE_Y)
        if 0 <= init_px < self.map_width and 0 <= init_py < self.map_height:
            cv2.circle(img, (init_px, init_py), 15, initial_base_color, -1)
            cv2.circle(img, (init_px, init_py), int(INITIAL_BASE_EXCLUSION_RADIUS * self.map_scale), 
                      initial_base_color, 2)
        
        # Identify outliers (points in all_positions but not in filtered_positions)
        if len(all_positions) > 0 and len(filtered_positions) > 0:
            # Create a set of filtered positions for quick lookup
            filtered_set = set(map(tuple, filtered_positions))
            
            # Draw outliers
            for pos in all_positions:
                if tuple(pos) not in filtered_set:
                    px, py = self.world_to_image(pos[0], pos[1])
                    if 0 <= px < self.map_width and 0 <= py < self.map_height:
                        cv2.circle(img, (px, py), 4, outlier_color, -1)
        
        # Draw filtered positions with cluster colors if available
        if len(filtered_positions) > 0:
            for i, pos in enumerate(filtered_positions):
                px, py = self.world_to_image(pos[0], pos[1])
                if 0 <= px < self.map_width and 0 <= py < self.map_height:
                    if cluster_labels is not None and i < len(cluster_labels):
                        label = cluster_labels[i]
                        if label == -1:  # Noise point
                            color = outlier_color
                        else:
                            color = cluster_colors[label % len(cluster_colors)]
                    else:
                        color = filtered_point_color
                    cv2.circle(img, (px, py), 6, color, -1)
        
        # Draw cluster centers
        for i, center in enumerate(cluster_centers):
            px, py = self.world_to_image(center[0], center[1])
            if 0 <= px < self.map_width and 0 <= py < self.map_height:
                # Draw larger circle for cluster center
                cv2.circle(img, (px, py), 12, cluster_center_color, 3)
                # Add label
                cv2.putText(img, f"{i+1}", (px-8, py+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cluster_center_color, 2)
        
        # Draw ground truth bases
        for i, base_pos in enumerate(self.ground_truth_bases):
            px, py = self.world_to_image(base_pos[0], base_pos[1])
            if 0 <= px < self.map_width and 0 <= py < self.map_height:
                cv2.circle(img, (px, py), 10, ground_truth_color, -1)
                cv2.putText(img, f"GT_{i+1}", (px-15, py+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ground_truth_color, 1)
        
        # Add legend
        legend_y = 30
        cv2.putText(img, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        legend_items = [
            ("Initial Base", initial_base_color),
            ("Outliers", outlier_color),
            ("Filtered Points", filtered_point_color),
            ("Cluster Centers", cluster_center_color),
            ("Ground Truth Bases", ground_truth_color),
        ]
        
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + 25 * (i + 1)
            cv2.circle(img, (20, y_pos - 5), 5, color, -1)
            cv2.putText(img, label, (35, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add coordinate info
        info_text = f"Scale: {self.map_scale} px/m | Points: {len(all_positions)} | Filtered: {len(filtered_positions)} | Clusters: {len(cluster_centers)}"
        cv2.putText(img, info_text, (10, self.map_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img

    def publish_position_map(self, all_positions, filtered_positions, cluster_centers, cluster_labels=None):
        """
        Create and publish the position map image.
        
        Args:
            all_positions (np.array): All detected positions
            filtered_positions (np.array): Positions after outlier removal
            cluster_centers (list): Final cluster center positions
            cluster_labels (np.array): Cluster labels for each filtered position (optional)
        """
        try:
            # Create the map image
            map_img = self.create_position_map(all_positions, filtered_positions, cluster_centers, cluster_labels)
            
            # Convert to ROS Image message
            ros_image = self.cv_bridge.cv2_to_imgmsg(map_img, encoding="rgb8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "map"
            
            # Publish the image
            self.map_image_publisher.publish(ros_image)
            self.get_logger().info("Published position map image")
            
        except Exception as e:
            self.get_logger().error(f"Error creating/publishing position map: {e}")
            traceback.print_exc()

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

    def detect_outliers_iqr(self, positions):
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            positions (np.array): Array of [x, y] positions
            
        Returns:
            np.array: Boolean mask where True indicates non-outlier positions
        """
        if len(positions) < 4:  # Need at least 4 points for meaningful IQR
            return np.ones(len(positions), dtype=bool)
        
        # Calculate distances from origin for outlier detection
        distances = np.sqrt(np.sum(positions**2, axis=1))
        
        # Calculate IQR
        Q1 = np.percentile(distances, 25)
        Q3 = np.percentile(distances, 75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        # Create mask for non-outliers
        mask = (distances >= lower_bound) & (distances <= upper_bound)
        
        outliers_count = np.sum(~mask)
        if outliers_count > 0:
            self.get_logger().info(f"Detected and removing {outliers_count} outliers using IQR method")
            
        return mask

    def remove_outliers_2d(self, positions):
        """
        Remove outliers using 2D coordinate analysis with IQR method.
        
        Args:
            positions (np.array): Array of [x, y] positions
            
        Returns:
            np.array: Filtered positions without outliers
        """
        if len(positions) < 4:
            return positions
            
        # Apply IQR outlier detection for both X and Y coordinates
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # X coordinate outlier detection
        Q1_x = np.percentile(x_coords, 25)
        Q3_x = np.percentile(x_coords, 75)
        IQR_x = Q3_x - Q1_x
        x_mask = (x_coords >= Q1_x - self.outlier_threshold * IQR_x) & \
                 (x_coords <= Q3_x + self.outlier_threshold * IQR_x)
        
        # Y coordinate outlier detection  
        Q1_y = np.percentile(y_coords, 25)
        Q3_y = np.percentile(y_coords, 75)
        IQR_y = Q3_y - Q1_y
        y_mask = (y_coords >= Q1_y - self.outlier_threshold * IQR_y) & \
                 (y_coords <= Q3_y + self.outlier_threshold * IQR_y)
        
        # Combine masks (point must be non-outlier in both dimensions)
        combined_mask = x_mask & y_mask
        
        outliers_count = np.sum(~combined_mask)
        if outliers_count > 0:
            self.get_logger().info(f"Detected and removing {outliers_count} outliers using 2D IQR method")
            
        return positions[combined_mask]

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

        Uses K-means clustering or DBSCAN to identify unique base positions from
        collected measurements. Includes outlier detection and removal.
        Publishes results and saves setpoints to configuration file.

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
        
        # Remove outliers before clustering
        self.get_logger().info(f"Processing {len(positions_array)} positions with outlier detection")
        filtered_positions = self.remove_outliers_2d(positions_array)
        
        if len(filtered_positions) < 2:
            self.get_logger().warn("Not enough positions remaining after outlier removal for clustering")
            return
            
        self.get_logger().info(f"Proceeding with {len(filtered_positions)} positions after outlier removal")
        
        cluster_labels = None
        
        try:
            if self.use_dbscan:
                # Use DBSCAN clustering (more robust to outliers)
                dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
                cluster_labels = dbscan.fit_predict(filtered_positions)
                
                # Get unique cluster centers (excluding noise points labeled as -1)
                unique_labels = set(cluster_labels)
                if -1 in unique_labels:
                    unique_labels.remove(-1)  # Remove noise label
                    noise_count = np.sum(cluster_labels == -1)
                    self.get_logger().info(f"DBSCAN identified {noise_count} noise points")
                
                if len(unique_labels) == 0:
                    self.get_logger().warn("No valid clusters found by DBSCAN")
                    return
                    
                # Calculate cluster centers
                cluster_centers = []
                for label in unique_labels:
                    cluster_points = filtered_positions[cluster_labels == label]
                    center = np.mean(cluster_points, axis=0)
                    cluster_centers.append(center)
                    
                self.unique_positions = cluster_centers
                self.get_logger().info(f"DBSCAN found {len(unique_labels)} clusters")
                
            else:
                # Use K-means clustering
                n_clusters = min(self.expected_bases, len(filtered_positions))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(filtered_positions)
                self.unique_positions = kmeans.cluster_centers_.tolist()
                self.get_logger().info(f"K-means clustering with {n_clusters} clusters")

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
            
            # Publish visual map
            self.publish_position_map(positions_array, filtered_positions, self.unique_positions, cluster_labels)
            
            # Calculate and log detection accuracy
            self.calculate_detection_accuracy(self.unique_positions)
            
        except Exception as e:
            self.get_logger().error(f"Error in process_positions: {e}")
            traceback.print_exc()

    def calculate_detection_accuracy(self, cluster_centers):
        """
        Calculate accuracy metrics comparing detected cluster centers with ground truth.
        
        Args:
            cluster_centers (list): Detected cluster center positions
            
        Returns:
            dict: Accuracy metrics including distances and matches
        """
        if not cluster_centers or not self.ground_truth_bases:
            return {}
            
        # Convert to numpy arrays for easier calculation
        detected = np.array(cluster_centers)
        ground_truth = np.array(self.ground_truth_bases)
        
        # Calculate distance matrix between all detected and ground truth points
        distances = []
        matches = []
        
        for i, gt_pos in enumerate(ground_truth):
            min_distance = float('inf')
            closest_detected_idx = -1
            
            for j, det_pos in enumerate(detected):
                distance = np.sqrt((gt_pos[0] - det_pos[0])**2 + (gt_pos[1] - det_pos[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_detected_idx = j
                    
            distances.append(min_distance)
            matches.append((i, closest_detected_idx, min_distance))
            
        # Calculate metrics
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        min_distance = np.min(distances)
        
        # Count accurate detections (within 1 meter)
        accurate_detections = sum(1 for d in distances if d <= 1.0)
        detection_rate = accurate_detections / len(self.ground_truth_bases)
        
        # Log results
        self.get_logger().info(f"=== DETECTION ACCURACY ANALYSIS ===")
        self.get_logger().info(f"Ground Truth Bases: {len(self.ground_truth_bases)}")
        self.get_logger().info(f"Detected Clusters: {len(cluster_centers)}")
        self.get_logger().info(f"Average Distance Error: {avg_distance:.3f}m")
        self.get_logger().info(f"Min Distance Error: {min_distance:.3f}m")
        self.get_logger().info(f"Max Distance Error: {max_distance:.3f}m")
        self.get_logger().info(f"Accurate Detections (≤1m): {accurate_detections}/{len(self.ground_truth_bases)}")
        self.get_logger().info(f"Detection Rate: {detection_rate:.2%}")
        
        for i, (gt_idx, det_idx, distance) in enumerate(matches):
            gt_pos = self.ground_truth_bases[gt_idx]
            if det_idx >= 0:
                det_pos = cluster_centers[det_idx]
                self.get_logger().info(f"BASE_{gt_idx+1} ({gt_pos[0]:.2f}, {gt_pos[1]:.2f}) -> "
                                     f"Detected ({det_pos[0]:.2f}, {det_pos[1]:.2f}) | Error: {distance:.3f}m")
            else:
                self.get_logger().info(f"BASE_{gt_idx+1} ({gt_pos[0]:.2f}, {gt_pos[1]:.2f}) -> No detection found")
        
        return {
            'avg_distance': avg_distance,
            'max_distance': max_distance,
            'min_distance': min_distance,
            'accurate_detections': accurate_detections,
            'detection_rate': detection_rate,
            'matches': matches
        }


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
