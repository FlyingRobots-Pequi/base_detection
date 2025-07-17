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
- Creates visual markers for RViz

Dependencies:
    - ROS2
    - NumPy
    - scikit-learn
    - PyYAML
    - geometry_msgs
    - visualization_msgs
    - std_srvs
"""
import traceback
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from base_detection.variables import (
    ABSOLUTE_POINTS_TOPIC,
    UNIQUE_POSITIONS_TOPIC,
    INITIAL_BASE_EXCLUSION_RADIUS,
    INITIAL_BASE_X,
    INITIAL_BASE_Y,
    MIN_DISTANCE_BETWEEN_BASES,
    MAX_DETECTIONS_STORED,
    DETECTION_TIMEOUT
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
    """

    def __init__(self):
        """
        Initialize the CoordinateProcessor node.

        Sets up:
        - Subscription to absolute positions
        - Publisher for unique positions
        - Publisher for RViz markers
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

        # RViz markers publisher
        self.markers_publisher = self.create_publisher(
            MarkerArray, "/base_detection/visualization_markers", 10
        )

        self.positions_list = []  # List of [x, y, z] positions with altitude
        self.unique_positions = []  # List of unique base positions with altitudes
        self.expected_bases = 5  # Total de 5 bases na arena (corrigido de volta)
        
        # Frame-based detection grouping
        self.frame_detections = []  # Temporary storage for detections from current frame
        self.last_detection_time = 0.0
        self.frame_timeout = 0.1  # 100ms timeout to group detections from same frame
        
        # Advanced detection control
        self.first_detection_time = None  # Time of first detection
        self.last_clustering_time = 0.0   # Time of last clustering attempt
        self.clustering_min_interval = 10.0  # Minimum seconds between clustering attempts (aumentado para dar mais tempo de exploração)
        
        # Outlier detection parameters
        self.use_dbscan = True  # Use DBSCAN for better outlier handling (mudado para True)
        self.outlier_threshold = 1.5  # IQR multiplier for outlier detection
        self.dbscan_eps = 1.5  # DBSCAN epsilon parameter (aumentado para 1.5 para agrupar melhor as 5 bases)
        self.dbscan_min_samples = 3  # DBSCAN minimum samples per cluster (voltou para 3 para ser mais rigoroso)
        
        # Ground truth base positions (now including Z for better visualization)
        self.ground_truth_bases = [
            (-0.24, -3.23, 0.0),  # BASE_1
            (0.75, -5.05, 0.0),   # BASE_2
            (5.16, -5.75, 0.0),   # BASE_3
            (4.37, -2.30, 0.0),   # BASE_4
            (5.69, -0.25, 0.0),   # BASE_5
        ]
        
        # Marker ID counter
        self.marker_id_counter = 0

        # Setup visualization and processing
        self.get_logger().info("CoordinateProcessor initialized for 5-base detection with improved clustering")
        self.get_logger().info(f"Configuration: DBSCAN={self.use_dbscan}, eps={self.dbscan_eps}, min_samples={self.dbscan_min_samples}")
        self.get_logger().info(f"Filters: Initial base exclusion={INITIAL_BASE_EXCLUSION_RADIUS}m, Min base distance={MIN_DISTANCE_BETWEEN_BASES}m")
        self.get_logger().info(f"Limits: Max detections={MAX_DETECTIONS_STORED}, Detection timeout={DETECTION_TIMEOUT}s")
        self.get_logger().info(f"Expected bases: {self.expected_bases}, Clustering interval: {self.clustering_min_interval}s")
        
        # Create timer for periodic clustering check
        self.create_timer(10.0, self.periodic_clustering_check)  # Check every 10s (aumento do intervalo)

    def create_marker(self, marker_type, position, color, scale, text="", marker_id=None):
        """
        Create a RViz marker with specified properties.
        
        Args:
            marker_type: Marker type (Marker.SPHERE, Marker.CYLINDER, etc.)
            position: (x, y, z) position tuple
            color: (r, g, b, a) color tuple (0-1 range)
            scale: (x, y, z) scale tuple
            text: Text for TEXT_VIEW_FACING markers
            marker_id: Specific marker ID (auto-generated if None)
            
        Returns:
            Marker: Configured RViz marker
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        
        if marker_id is None:
            marker.id = self.marker_id_counter
            self.marker_id_counter += 1
        else:
            marker.id = marker_id
            
        marker.type = marker_type
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2]) if len(position) > 2 else 0.0
        marker.pose.orientation.w = 1.0
        
        # Scale
        marker.scale.x = float(scale[0])
        marker.scale.y = float(scale[1])
        marker.scale.z = float(scale[2])
        
        # Color
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        
        # Text
        if text:
            marker.text = text
            
        return marker

    def create_visualization_markers(self, all_positions, filtered_positions, cluster_centers, cluster_labels=None):
        """
        Create RViz markers for all detected data.
        
        Args:
            all_positions (np.array): All detected positions
            filtered_positions (np.array): Positions after outlier removal
            cluster_centers (list): Final cluster center positions
            cluster_labels (np.array): Cluster labels for each filtered position (optional)
            
        Returns:
            MarkerArray: Array of RViz markers
        """
        markers = MarkerArray()
        self.marker_id_counter = 0
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)
        
        # Ground truth bases
        for i, base_pos in enumerate(self.ground_truth_bases):
            # Base marker
            marker = self.create_marker(
                Marker.CYLINDER,
                (base_pos[0], base_pos[1], base_pos[2]),
                (1.0, 0.647, 0.0, 0.8),  # Orange
                (0.3, 0.3, 0.1)
            )
            markers.markers.append(marker)
            
            # Label
            label_marker = self.create_marker(
                Marker.TEXT_VIEW_FACING,
                (base_pos[0], base_pos[1], base_pos[2] + 0.3),
                (1.0, 0.647, 0.0, 1.0),  # Orange
                (0.3, 0.3, 0.3),
                f"GT_BASE_{i+1}"
            )
            markers.markers.append(label_marker)
        
        # Initial base position and exclusion radius
        if INITIAL_BASE_X != 0 or INITIAL_BASE_Y != 0:
            # Initial base marker
            initial_marker = self.create_marker(
                Marker.CYLINDER,
                (INITIAL_BASE_X, INITIAL_BASE_Y, 0.0),
                (0.0, 0.0, 0.0, 1.0),  # Black
                (0.4, 0.4, 0.1)
            )
            markers.markers.append(initial_marker)
            
            # Exclusion radius
            radius_marker = self.create_marker(
                Marker.CYLINDER,
                (INITIAL_BASE_X, INITIAL_BASE_Y, 0.0),
                (0.0, 0.0, 0.0, 0.2),  # Semi-transparent black
                (INITIAL_BASE_EXCLUSION_RADIUS * 2, INITIAL_BASE_EXCLUSION_RADIUS * 2, 0.02)
            )
            markers.markers.append(radius_marker)
            
            # Label
            initial_label = self.create_marker(
                Marker.TEXT_VIEW_FACING,
                (INITIAL_BASE_X, INITIAL_BASE_Y, 0.3),
                (0.0, 0.0, 0.0, 1.0),
                (0.3, 0.3, 0.3),
                "INITIAL_BASE"
            )
            markers.markers.append(initial_label)
        
        # Outliers (points in all_positions but not in filtered_positions)
        if len(all_positions) > 0 and len(filtered_positions) > 0:
            filtered_set = set(map(tuple, filtered_positions))
            
            for pos in all_positions:
                if tuple(pos) not in filtered_set:
                    outlier_marker = self.create_marker(
                        Marker.SPHERE,
                        (pos[0], pos[1], pos[2]),
                        (0.5, 0.5, 0.5, 0.6),  # Gray
                        (0.1, 0.1, 0.1)
                    )
                    markers.markers.append(outlier_marker)
        
        # Cluster colors for DBSCAN
        cluster_colors = [
            (1.0, 0.0, 1.0, 0.8),    # Magenta
            (0.0, 1.0, 1.0, 0.8),    # Cyan
            (1.0, 1.0, 0.0, 0.8),    # Yellow
            (1.0, 0.5, 0.0, 0.8),    # Orange
            (0.5, 0.0, 1.0, 0.8),    # Purple
            (0.0, 0.5, 1.0, 0.8),    # Light blue
            (1.0, 0.0, 0.5, 0.8),    # Pink
        ]
        
        # Filtered positions with cluster colors if available (DETECTED BASE POSITIONS)
        if len(filtered_positions) > 0:
            for i, pos in enumerate(filtered_positions):
                if cluster_labels is not None and i < len(cluster_labels):
                    label = cluster_labels[i]
                    if label == -1:  # Noise point
                        color = (0.5, 0.5, 0.5, 0.6)  # Gray
                    else:
                        color = cluster_colors[label % len(cluster_colors)]
                else:
                    color = (0.0, 1.0, 0.0, 0.8)  # Green
                
                # Marker for detected base position
                point_marker = self.create_marker(
                    Marker.SPHERE,
                    (pos[0], pos[1], pos[2]),
                    color,
                    (0.15, 0.15, 0.15)
                )
                markers.markers.append(point_marker)
        
        # Cluster centers (FINAL DETECTED BASES)
        for i, center in enumerate(cluster_centers):
            # Main center marker - larger and more prominent
            center_marker = self.create_marker(
                Marker.CYLINDER,
                (center[0], center[1], center[2]),
                (0.0, 0.0, 1.0, 1.0),  # Blue
                (0.4, 0.4, 0.2)  # Larger cylinder for prominence
            )
            markers.markers.append(center_marker)
            
            # Add detection confidence ring around the base
            confidence_ring = self.create_marker(
                Marker.CYLINDER,
                (center[0], center[1], center[2] + 0.1),
                (0.0, 0.0, 1.0, 0.3),  # Semi-transparent blue
                (0.8, 0.8, 0.05)  # Large flat ring
            )
            markers.markers.append(confidence_ring)
            
            # Label with "DETECTED BASE" prefix
            center_label = self.create_marker(
                Marker.TEXT_VIEW_FACING,
                (center[0], center[1], center[2] + 0.5),
                (0.0, 0.0, 1.0, 1.0),  # Blue
                (0.4, 0.4, 0.4),
                f"DETECTED_BASE_{i+1}\n({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})"
            )
            markers.markers.append(center_label)
        
        return markers

    def publish_visualization_markers(self, all_positions, filtered_positions, cluster_centers, cluster_labels=None):
        """
        Create and publish RViz markers for visualization.
        
        Args:
            all_positions (np.array): All detected positions
            filtered_positions (np.array): Positions after outlier removal
            cluster_centers (list): Final cluster center positions
            cluster_labels (np.array): Cluster labels for each filtered position (optional)
        """
        try:
            markers = self.create_visualization_markers(all_positions, filtered_positions, cluster_centers, cluster_labels)
            self.markers_publisher.publish(markers)
            self.get_logger().info(f"Published {len(markers.markers)} RViz markers")
            
        except Exception as e:
            self.get_logger().error(f"Error creating/publishing RViz markers: {e}")
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

    def is_too_close_to_existing(self, x, y):
        """
        Check if a new detection is too close to already stored detections.
        
        Args:
            x (float): X coordinate to check
            y (float): Y coordinate to check
            
        Returns:
            bool: True if position is too close to existing detections
        """
        for pos in self.positions_list:
            distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            if distance < MIN_DISTANCE_BETWEEN_BASES:
                return True
        return False

    def filter_duplicate_detections(self):
        """
        Remove duplicate detections that are too close to each other.
        Keeps the first detection in each cluster.
        """
        if len(self.positions_list) < 2:
            return
            
        filtered_positions = []
        
        for pos in self.positions_list:
            # Check if this position is too close to any already filtered position
            too_close = False
            for filtered_pos in filtered_positions:
                distance = np.sqrt((pos[0] - filtered_pos[0])**2 + (pos[1] - filtered_pos[1])**2)
                if distance < MIN_DISTANCE_BETWEEN_BASES:
                    too_close = True
                    break
            
            if not too_close:
                filtered_positions.append(pos)
        
        original_count = len(self.positions_list)
        self.positions_list = filtered_positions
        removed_count = original_count - len(filtered_positions)
        
        if removed_count > 0:
            self.get_logger().info(f"Filtered out {removed_count} duplicate detections (too close to existing ones)")
            self.get_logger().info(f"Remaining detections: {len(self.positions_list)}")

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
            positions (np.array): Array of [x, y, z] positions
            
        Returns:
            np.array: Filtered positions without outliers
        """
        if len(positions) < 4:
            return positions
            
        # Apply IQR outlier detection for X and Y coordinates (ignore Z for outlier detection)
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
        Process incoming absolute position messages with frame-based grouping.

        Groups detections that arrive within a short time window (same frame)
        and processes them together to eliminate temporal drift in clustering.

        Args:
            msg (geometry_msgs.msg.Point): Absolute position message
        """
        try:
            current_time = self.get_clock().now().nanoseconds / 1e9  # Convert to seconds
            
            # Filter out positions too close to initial base
            if self.is_near_initial_base(msg.x, msg.y):
                distance_to_origin = np.sqrt(msg.x**2 + msg.y**2)
                self.get_logger().info(f"❌ Filtering out position near initial base: ({msg.x:.3f}, {msg.y:.3f}) - distance: {distance_to_origin:.3f}m < {INITIAL_BASE_EXCLUSION_RADIUS:.3f}m")
                return
            
            # Check if this detection is part of the current frame or a new frame
            time_since_last = current_time - self.last_detection_time
            
            if time_since_last > self.frame_timeout and self.frame_detections:
                # Process previous frame's detections
                self.get_logger().info(f"📦 Processing frame with {len(self.frame_detections)} detections (time gap: {time_since_last:.3f}s)")
                self.process_frame_detections()
                self.frame_detections = []
            
            # Add current detection to frame
            position = [msg.x, msg.y, msg.z]
            self.frame_detections.append(position)
            self.last_detection_time = current_time
            
            self.get_logger().info(f"✅ Added detection to frame: ({msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}) - frame size: {len(self.frame_detections)}")
            
            # Create a timer to process the frame if no more detections arrive
            self.create_timer(self.frame_timeout, self.process_frame_timeout)
                
        except Exception as e:
            self.get_logger().error(f"Error in absolute_position_callback: {e}")
            traceback.print_exc()

    def process_frame_timeout(self):
        """
        Process accumulated frame detections when timeout is reached.
        """
        if self.frame_detections:
            self.get_logger().info(f"Frame timeout reached - processing {len(self.frame_detections)} detections")
            self.process_frame_detections()
            self.frame_detections = []

    def process_frame_detections(self):
        """
        Process detections from a single frame together with improved filtering.
        """
        if not self.frame_detections:
            return
            
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Set first detection time if not set
        if self.first_detection_time is None:
            self.first_detection_time = current_time
            
        # Add frame detections with duplicate filtering
        new_detections_added = 0
        for position in self.frame_detections:
            # Check if too close to existing detections
            if not self.is_too_close_to_existing(position[0], position[1]):
                self.positions_list.append(position)
                new_detections_added += 1
            else:
                self.get_logger().debug(f"Skipped duplicate detection: ({position[0]:.3f}, {position[1]:.3f})")
                
        self.get_logger().info(f"Added {new_detections_added}/{len(self.frame_detections)} new detections from frame. Total positions: {len(self.positions_list)}")
        
        # Trigger clustering based on multiple conditions
        should_cluster = False
        clustering_reason = ""
        
        # Condition 1: Reached expected number of bases
        if len(self.positions_list) >= self.expected_bases:
            should_cluster = True
            clustering_reason = f"reached expected bases ({self.expected_bases})"
            
        # Condition 2: Reached maximum stored detections
        elif len(self.positions_list) >= MAX_DETECTIONS_STORED:
            should_cluster = True
            clustering_reason = f"reached max detections ({MAX_DETECTIONS_STORED})"
            
        # Condition 3: Timeout since first detection
        elif (current_time - self.first_detection_time) >= DETECTION_TIMEOUT:
            should_cluster = True
            clustering_reason = f"timeout reached ({DETECTION_TIMEOUT:.1f}s since first detection)"
            
        # Respect minimum interval between clustering attempts
        time_since_last_clustering = current_time - self.last_clustering_time
        
        if should_cluster and time_since_last_clustering >= self.clustering_min_interval:
            self.get_logger().info(f"Triggering clustering: {clustering_reason}")
            self.last_clustering_time = current_time
            self.filter_duplicate_detections()  # Final cleanup before clustering
            self.process_positions()
        elif should_cluster:
            self.get_logger().info(f"Clustering delayed: {clustering_reason}, but waiting for min interval ({time_since_last_clustering:.1f}s < {self.clustering_min_interval:.1f}s)")

    def process_positions(self):
        """
        Process collected positions to identify unique bases using improved clustering.

        Uses DBSCAN clustering for better outlier handling and flexible number of clusters.
        Includes outlier detection and duplicate removal.
        """
        
        if len(self.positions_list) < 2:
            self.get_logger().warn(
                f"Not enough positions to perform clustering. Need at least 2 positions, but got {len(self.positions_list)}."
            )
            return

        positions_array = np.array(self.positions_list)  # Contains [x, y, z] data
        
        # Remove outliers before clustering
        self.get_logger().info(f"Processing {len(positions_array)} positions with outlier detection")
        filtered_positions = self.remove_outliers_2d(positions_array)
        
        if len(filtered_positions) < 2:
            self.get_logger().warn("Not enough positions remaining after outlier removal for clustering")
            return
            
        self.get_logger().info(f"Proceeding with {len(filtered_positions)} positions after outlier removal")
        
        cluster_labels = None
        
        try:
            # Extract 2D coordinates for clustering (x, y only)
            positions_2d = filtered_positions[:, :2]
            
            if self.use_dbscan:
                # Use DBSCAN clustering (more robust to outliers and finds optimal number of clusters)
                dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
                cluster_labels = dbscan.fit_predict(positions_2d)
                
                # Get unique cluster centers (excluding noise points labeled as -1)
                unique_labels = set(cluster_labels)
                noise_count = 0
                if -1 in unique_labels:
                    unique_labels.remove(-1)  # Remove noise label
                    noise_count = np.sum(cluster_labels == -1)
                    self.get_logger().info(f"DBSCAN identified {noise_count} noise points")
                
                if len(unique_labels) == 0:
                    self.get_logger().warn("No valid clusters found by DBSCAN - all points classified as noise")
                    return
                    
                # Calculate cluster centers with average Z values
                cluster_centers = []
                for label in unique_labels:
                    cluster_points = filtered_positions[cluster_labels == label]
                    center_2d = np.mean(cluster_points[:, :2], axis=0)  # Average x, y
                    center_z = np.mean(cluster_points[:, 2], axis=0)    # Average z
                    center = [center_2d[0], center_2d[1], center_z]
                    cluster_centers.append(center)
                    
                self.unique_positions = cluster_centers
                self.get_logger().info(f"DBSCAN found {len(unique_labels)} unique base clusters (noise points: {noise_count})")
                
            else:
                # Fallback to K-means clustering
                n_clusters = min(self.expected_bases, len(filtered_positions))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(positions_2d)
                
                # Calculate average Z values for each cluster
                cluster_centers = []
                cluster_labels = kmeans.predict(positions_2d)
                for i in range(n_clusters):
                    cluster_points = filtered_positions[cluster_labels == i]
                    center_2d = kmeans.cluster_centers_[i]
                    center_z = np.mean(cluster_points[:, 2], axis=0) if len(cluster_points) > 0 else 0.0
                    center = [center_2d[0], center_2d[1], center_z]
                    cluster_centers.append(center)
                
                self.unique_positions = cluster_centers
                self.get_logger().info(f"K-means clustering with {n_clusters} clusters")

            # Sort positions by distance from origin (0,0,0) - closest first
            self.unique_positions.sort(key=lambda pos: np.sqrt(pos[0]**2 + pos[1]**2))

            # Create and publish pose array
            pose_array = PoseArray()
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.header.frame_id = 'map'

            self.get_logger().info("=== DETECTED UNIQUE BASE POSITIONS ===")
            for i, pos in enumerate(self.unique_positions):
                distance_from_origin = np.sqrt(pos[0]**2 + pos[1]**2)
                self.get_logger().info(f"Base {i+1}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) - distance: {distance_from_origin:.3f}m")
                
                pose = Pose()
                pose.position.x = pos[0]
                pose.position.y = pos[1]
                pose.position.z = pos[2]
                pose_array.poses.append(pose)

            self.unique_positions_publisher.publish(pose_array)
            self.get_logger().info(f"Published {len(self.unique_positions)} unique base positions.")
            
            # Publish visual markers
            self.publish_visualization_markers(positions_array, filtered_positions, self.unique_positions, cluster_labels)
            
            # Calculate and log detection accuracy
            self.calculate_detection_accuracy(self.unique_positions)
            
            # Clear processed positions to allow for new detections
            self.positions_list.clear()
            self.first_detection_time = None
            self.get_logger().info("Cleared processed detections - ready for new exploration data")
            
        except Exception as e:
            self.get_logger().error(f"Error in process_positions: {e}")
            traceback.print_exc()

    def calculate_detection_accuracy(self, cluster_centers):
        """
        Calculate accuracy metrics comparing detected cluster centers with ground truth.
        
        Args:
            cluster_centers (list): Detected cluster center positions [x, y, z]
            
        Returns:
            dict: Accuracy metrics including distances and matches
        """
        if not cluster_centers or not self.ground_truth_bases:
            return {}
            
        # Convert to numpy arrays for easier calculation (use only x,y for distance)
        detected = np.array([[pos[0], pos[1]] for pos in cluster_centers])
        ground_truth = np.array([[pos[0], pos[1]] for pos in self.ground_truth_bases])
        
        # Calculate distance matrix between all detected and ground truth points
        distances = []
        matches = []
        
        for i, gt_pos in enumerate(self.ground_truth_bases):
            min_distance = float('inf')
            closest_detected_idx = -1
            
            for j, cluster_pos in enumerate(cluster_centers):
                # Calculate 2D distance (x, y only)
                distance = np.sqrt((gt_pos[0] - cluster_pos[0])**2 + (gt_pos[1] - cluster_pos[1])**2)
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
                self.get_logger().info(f"BASE_{gt_idx+1} ({gt_pos[0]:.2f}, {gt_pos[1]:.2f}, {gt_pos[2]:.2f}) -> "
                                     f"Detected ({det_pos[0]:.2f}, {det_pos[1]:.2f}, {det_pos[2]:.2f}) | Error: {distance:.3f}m")
            else:
                self.get_logger().info(f"BASE_{gt_idx+1} ({gt_pos[0]:.2f}, {gt_pos[1]:.2f}, {gt_pos[2]:.2f}) -> No detection found")
        
        return {
            'avg_distance': avg_distance,
            'max_distance': max_distance,
            'min_distance': min_distance,
            'accurate_detections': accurate_detections,
            'detection_rate': detection_rate,
            'matches': matches
        }

    def periodic_clustering_check(self):
        """
        Periodically check if clustering should be triggered due to timeout.
        This is a fallback mechanism to ensure clustering happens even if
        no new detections arrive for a long time.
        """
        if self.first_detection_time is None:
            return  # No detections yet
            
        current_time = self.get_clock().now().nanoseconds / 1e9
        time_since_first_detection = current_time - self.first_detection_time
        time_since_last_clustering = current_time - self.last_clustering_time

        if (time_since_first_detection >= DETECTION_TIMEOUT and 
            time_since_last_clustering >= self.clustering_min_interval and
            len(self.positions_list) >= 2):
            
            self.get_logger().info(f"Periodic check: Timeout reached ({time_since_first_detection:.1f}s since first detection). Triggering clustering.")
            self.last_clustering_time = current_time
            self.filter_duplicate_detections()
            self.process_positions()
        else:
            self.get_logger().debug(f"Periodic check: No timeout, first detection time: {self.first_detection_time:.1f}s, current time: {current_time:.1f}s")


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
