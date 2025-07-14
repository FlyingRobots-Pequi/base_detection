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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Point, PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from px4_msgs.msg import VehicleLocalPosition
from base_detection.variables import (
    ABSOLUTE_POINTS_TOPIC,
    UNIQUE_POSITIONS_TOPIC,
    VEHICLE_LOCAL_POSITION_TOPIC,
    INITIAL_BASE_EXCLUSION_RADIUS,
    INITIAL_BASE_X,
    INITIAL_BASE_Y
)
import os
from datetime import datetime
import csv
import json


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
        - Line-based clustering parameters
        """
        super().__init__("coordinate_processor")
        
        sensor_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )


        # Assinatura para posições absolutas detectadas
        self.absolute_subscription = self.create_subscription(
            Point,
            ABSOLUTE_POINTS_TOPIC,
            self.absolute_position_callback,
            10)

        # Assinatura para a posição local do veículo para estimar a direção do movimento
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            VEHICLE_LOCAL_POSITION_TOPIC,
            self.vehicle_local_position_callback,
            sensor_qos_profile)

        self.unique_positions_publisher = self.create_publisher(
            PoseArray, UNIQUE_POSITIONS_TOPIC, 10
        )

        # RViz markers publisher
        self.markers_publisher = self.create_publisher(
            MarkerArray, "/base_detection/visualization_markers", 10
        )

        self.positions_list = []
        self.unique_positions = []
        self.expected_bases = 5
        
        # Outlier detection parameters
        self.use_dbscan = False  # Use DBSCAN for better outlier handling
        self.outlier_threshold = 1.5  # IQR multiplier for outlier detection
        self.dbscan_eps = 0.9  # DBSCAN epsilon parameter (max distance between points in cluster)
        self.dbscan_min_samples = 3  # DBSCAN minimum samples per cluster
        
        # Parâmetros de clustering baseado em linhas (configuráveis via ROS)
        self.declare_parameter('use_line_based_clustering', True)
        self.declare_parameter('line_tolerance', 0.3)
        self.declare_parameter('min_line_points', 3)
        self.declare_parameter('line_detection_ransac_iterations', 100)
        self.declare_parameter('collinearity_weight', 15.0) # Novo! Peso para a identidade da linha.
        self.declare_parameter('output_dir', '/root/ros2_ws/tuning_results/default_run')

        self.use_line_based_clustering = self.get_parameter('use_line_based_clustering').value
        self.line_tolerance = self.get_parameter('line_tolerance').value
        self.min_line_points = self.get_parameter('min_line_points').value
        self.line_detection_ransac_iterations = self.get_parameter('line_detection_ransac_iterations').value
        self.collinearity_weight = self.get_parameter('collinearity_weight').value # Obter o valor
        
        self.drone_position_history = []
        self.max_position_history = 50 # Aumentar histórico para melhor estimativa
        
        # Adiciona a flag de estado para o "aquecimento" da clusterização inicial
        self.initial_clustering_done = False
        
        # Ground truth base positions
        self.ground_truth_bases = [
            (-0.24, -3.23),  # BASE_1
            (0.75, -5.05),   # BASE_2
            (5.16, -5.75),   # BASE_3
            (4.37, -2.30),   # BASE_4
            (5.69, -0.25),   # BASE_5
        ]
        
        # Marker ID counter
        self.marker_id_counter = 0
        
        # Get the ROS2 workspace base directory
        self.ros2_ws_base = os.path.expanduser("/root/ros2_ws")

        # Log configuration
        clustering_mode = "hybrid line-KMeans" if self.use_line_based_clustering else "traditional"
        self.get_logger().info(f"CoordinateProcessor initialized with {clustering_mode} clustering")
        self.get_logger().info(f"Line tolerance: {self.line_tolerance}m, Min line points: {self.min_line_points}")
        self.get_logger().info(f"Collinearity Weight: {self.collinearity_weight}")
        self.get_logger().info("🚁 Subscribed to vehicle position for accurate movement direction.")
        self.get_logger().info("Ready to wait for sufficient points for initial line-based clustering.")

    def on_shutdown(self):
        """
        Executado quando o nó está sendo desligado.
        Garante que a análise de acurácia final seja calculada e salva.
        """
        self.get_logger().info("Node is shutting down. Performing final accuracy analysis...")
        if self.unique_positions:
             # O diretório de resultados será parametrizado
            output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
            self.calculate_detection_accuracy(self.unique_positions, save_results=True, output_dir=output_dir)
        else:
            self.get_logger().warn("No unique positions were detected, skipping final analysis.")

    def absolute_position_callback(self, msg):
        """
        Callback for handling incoming absolute position messages.

        Filters out positions near the initial base and adds valid
        positions to the list for clustering. Triggers clustering
        process.

        Args:
            msg (Point): The incoming message with absolute coordinates.
        """
        x, y, z = msg.x, msg.y, msg.z

        # Filter out detections near the initial base
        if not self.is_near_initial_base(x, y):
            self.positions_list.append([x, y, z])
            self.get_logger().info(f"Received new detection at ({x:.2f}, {y:.2f}). Total points: {len(self.positions_list)}")
            
            # A lógica de processamento verificará se há pontos suficientes para clusterizar.
            self.process_positions()
        else:
            # Log para sabermos que um ponto foi ignorado
            self.get_logger().debug(f"Filtered out a detection at ({x:.2f}, {y:.2f}) near the initial base.")

    def vehicle_local_position_callback(self, msg):
        """
        Callback for vehicle local position updates.

        Maintains a history of the drone's position to estimate its
        movement direction.

        Args:
            msg (VehicleLocalPosition): The incoming local position message.
        """
        # We only need X and Y for movement direction estimation
        current_position = [msg.x, msg.y]
        
        # Add to history if moved a bit
        if not self.drone_position_history or np.linalg.norm(np.array(current_position) - np.array(self.drone_position_history[-1])) > 0.01:
            self.drone_position_history.append(current_position)
        
        # Keep the history size limited
        if len(self.drone_position_history) > self.max_position_history:
            self.drone_position_history.pop(0)

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
                (base_pos[0], base_pos[1], 0.0),
                (1.0, 0.647, 0.0, 0.8),  # Orange
                (0.3, 0.3, 0.1)
            )
            markers.markers.append(marker)
            
            # Label
            label_marker = self.create_marker(
                Marker.TEXT_VIEW_FACING,
                (base_pos[0], base_pos[1], 0.3),
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
                        (pos[0], pos[1], 0.0),
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
        
        # Filtered positions with cluster colors if available
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
                
                point_marker = self.create_marker(
                    Marker.SPHERE,
                    (pos[0], pos[1], 0.0),
                    color,
                    (0.15, 0.15, 0.15)
                )
                markers.markers.append(point_marker)
        
        # Cluster centers
        for i, center in enumerate(cluster_centers):
            # Center marker
            center_marker = self.create_marker(
                Marker.SPHERE,
                (center[0], center[1], 0.0),
                (0.0, 0.0, 1.0, 1.0),  # Blue
                (0.25, 0.25, 0.25)
            )
            markers.markers.append(center_marker)
            
            # Label
            center_label = self.create_marker(
                Marker.TEXT_VIEW_FACING,
                (center[0], center[1], 0.3),
                (0.0, 0.0, 1.0, 1.0),  # Blue
                (0.3, 0.3, 0.3),
                f"CLUSTER_{i+1}"
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

    def estimate_drone_movement_direction(self):
        """
        Estima a direção primária do movimento do drone usando o histórico de posições do veículo.
        """
        if len(self.drone_position_history) < 2:
            self.get_logger().warn("Not enough vehicle position history to estimate movement direction. Defaulting to X-axis.")
            return np.array([1.0, 0.0])
        
        positions = np.array(self.drone_position_history)
        
        # Calculate movement vectors between consecutive positions
        movements = []
        for i in range(1, len(positions)):
            movement = positions[i] - positions[i-1]
            if np.linalg.norm(movement) > 0.1:  # Only consider significant movements
                movements.append(movement)
        
        if not movements:
            return np.array([1.0, 0.0])  # Default direction
        
        movements = np.array(movements)
        
        # Calculate average movement direction
        avg_movement = np.mean(movements, axis=0)
        
        # Normalize to get unit vector
        direction_magnitude = np.linalg.norm(avg_movement)
        if direction_magnitude > 0:
            primary_direction = avg_movement / direction_magnitude
        else:
            primary_direction = np.array([1.0, 0.0])
        
        self.get_logger().info(f"🧭 Estimated drone movement direction: ({primary_direction[0]:.3f}, {primary_direction[1]:.3f})")
        return primary_direction

    def detect_lines_ransac(self, positions, direction_hint=None):
        """
        Detect lines in position data using RANSAC algorithm.
        
        Args:
            positions (np.array): Array of positions [x, y]
            direction_hint (np.array): Hint for expected line direction
            
        Returns:
            list: List of dictionaries with 'line_params', 'inliers', 'points'
        """
        if len(positions) < self.min_line_points:
            return []
        
        lines = []
        remaining_points = list(range(len(positions)))
        
        while len(remaining_points) >= self.min_line_points:
            best_line = None
            best_inliers = []
            best_score = 0
            
            # RANSAC iterations
            for _ in range(self.line_detection_ransac_iterations):
                # Randomly sample 2 points
                if len(remaining_points) < 2:
                    break
                    
                sample_indices = np.random.choice(remaining_points, 2, replace=False)
                p1, p2 = positions[sample_indices]
                
                # Calculate line parameters (ax + by + c = 0)
                if np.allclose(p1, p2):
                    continue
                    
                direction = p2 - p1
                direction_norm = np.linalg.norm(direction)
                if direction_norm == 0:
                    continue
                    
                direction = direction / direction_norm
                
                # Line equation: (y - y1) = m(x - x1) -> ax + by + c = 0
                # Where a = -m, b = 1, c = m*x1 - y1
                if abs(direction[0]) > abs(direction[1]):  # More horizontal
                    # Use form y = mx + c
                    m = direction[1] / direction[0] if direction[0] != 0 else 0
                    c = p1[1] - m * p1[0]
                    line_params = [-m, 1, -c]  # ax + by + c = 0 form
                else:  # More vertical
                    # Use form x = my + c
                    m = direction[0] / direction[1] if direction[1] != 0 else 0
                    c = p1[0] - m * p1[1]
                    line_params = [1, -m, -c]  # ax + by + c = 0 form
                
                # Find inliers
                inliers = []
                for i in remaining_points:
                    point = positions[i]
                    # Distance from point to line: |ax + by + c| / sqrt(a² + b²)
                    distance = abs(line_params[0] * point[0] + line_params[1] * point[1] + line_params[2])
                    distance /= np.sqrt(line_params[0]**2 + line_params[1]**2)
                    
                    if distance <= self.line_tolerance:
                        inliers.append(i)
                
                # Score this line
                score = len(inliers)
                
                # Bonus for alignment with movement direction hint
                if direction_hint is not None:
                    line_direction = np.array([-line_params[1], line_params[0]])
                    line_direction /= np.linalg.norm(line_direction)
                    alignment = abs(np.dot(direction, direction_hint))
                    score += alignment * 2  # Bonus for alignment
                
                if score > best_score and len(inliers) >= self.min_line_points:
                    best_line = line_params
                    best_inliers = inliers
                    best_score = score
            
            # Add the best line found
            if best_line is not None and len(best_inliers) >= self.min_line_points:
                line_points = positions[best_inliers]
                lines.append({
                    'line_params': best_line,
                    'inliers': best_inliers,
                    'points': line_points,
                    'score': best_score
                })
                
                # Remove inliers from remaining points
                remaining_points = [i for i in remaining_points if i not in best_inliers]
                
                self.get_logger().info(f"📏 Detected line with {len(best_inliers)} points, score: {best_score:.2f}")
            else:
                break  # No more good lines found
        
        return lines

    def run_hybrid_line_kmeans(self, positions):
        """
        Executa um clustering híbrido que usa a detecção de linhas para guiar o K-Means.

        Args:
            positions (np.array): Pontos 3D [x, y, z] a serem clusterizados.

        Returns:
            list: Uma lista de centros de cluster (posições 3D).
            np.array: Os rótulos de cluster para cada ponto de entrada.
        """
        self.get_logger().info(f"🚁 Running Hybrid Line-KMeans clustering with {len(positions)} points.")
        positions_2d = positions[:, :2]

        # Passo 1: Detectar linhas e atribuir IDs
        movement_direction = self.estimate_drone_movement_direction()
        detected_lines = self.detect_lines_ransac(positions_2d, movement_direction)

        line_ids = np.full(len(positions), -1, dtype=int)
        if not detected_lines:
            self.get_logger().warning("⚠️ No lines detected. Proceeding with standard K-Means.")
            # Se nenhuma linha for encontrada, todos os line_ids permanecem -1 (outliers).
            # O K-Means irá agrupar apenas com base na posição.
        else:
            for i, line in enumerate(detected_lines):
                line_ids[line['inliers']] = i
        
        # Passo 2: Criar características aprimoradas
        # Usamos apenas X e Y para a parte espacial do clustering, mas o Z será mantido para o cálculo do centro final.
        line_feature = self.collinearity_weight * line_ids.reshape(-1, 1)
        enhanced_features = np.hstack([positions_2d, line_feature])
        
        # Passo 3: Normalizar características
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(enhanced_features)
        
        # Passo 4: Executar K-Means nos dados aprimorados e normalizados
        n_clusters = min(self.expected_bases, len(positions))
        if n_clusters <= 0:
             self.get_logger().warn("Not enough data to form any clusters.")
             return [], np.array([])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        labels = kmeans.labels_
        
        # Passo 5: "Desnormalizar" - Calcular os centros no espaço de coordenadas REAL
        cluster_centers = []
        for i in range(n_clusters):
            # Pega os pontos ORIGINAIS (3D) que pertencem a este cluster
            points_in_cluster = positions[labels == i]
            if len(points_in_cluster) > 0:
                # Calcula a média dos pontos originais para encontrar o centro real
                center = np.mean(points_in_cluster, axis=0)
                cluster_centers.append(center)
        
        self.get_logger().info(f"✅ Hybrid clustering found {len(cluster_centers)} clusters.")
        return cluster_centers, labels

    def process_positions(self):
        """
        Process collected positions to identify unique bases.

        Uses line-based clustering to identify unique base positions from
        collected measurements, prioritizing points that form lines in the
        direction of drone movement. Falls back to K-means or DBSCAN if 
        line detection fails. Includes outlier detection and removal.
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

        positions_array = np.array(self.positions_list)  # Now contains [x, y, z] data
        
        # Simple CSV logging for pre-clustering data
        try:
            csv_dir = "/root/ros2_ws/clustering_logs"
            os.makedirs(csv_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_file = os.path.join(csv_dir, f"positions_before_clustering_{timestamp_str}.csv")
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x_m', 'y_m', 'z_m', 'distance_from_origin_m', 'timestamp'])
                
                for pos in positions_array:
                    distance = np.sqrt(pos[0]**2 + pos[1]**2)
                    writer.writerow([pos[0], pos[1], pos[2], distance, datetime.now().isoformat()])
            
            self.get_logger().info(f"📊 Saved {len(positions_array)} positions before clustering to: {csv_file}")
        except Exception as e:
            self.get_logger().warning(f"CSV logging failed: {e}")
        
        # Remove outliers before clustering
        self.get_logger().info(f"Processing {len(positions_array)} positions with outlier detection")
        filtered_positions = self.remove_outliers_2d(positions_array)
        
        if len(filtered_positions) < 2:
            self.get_logger().warn("Not enough positions remaining after outlier removal for clustering")
            return
            
        self.get_logger().info(f"Proceeding with {len(filtered_positions)} positions after outlier removal")
        
        # Lógica de "espera"
        if not self.initial_clustering_done and len(filtered_positions) < self.min_line_points:
            self.get_logger().info(
                f"Postponing initial clustering. Waiting for at least {self.min_line_points} points "
                f"to ensure good line detection. Current points: {len(filtered_positions)}."
            )
            return

        try:
            if self.use_line_based_clustering:
                # Caminho primário: Clustering Híbrido
                self.unique_positions, cluster_labels = self.run_hybrid_line_kmeans(filtered_positions)
                clustering_method = "hybrid-line-kmeans"
            else:
                # Fallback para o K-Means tradicional se o método de linhas for desativado
                self.get_logger().info("📊 Using fallback K-means clustering as configured.")
                n_clusters = min(self.expected_bases, len(filtered_positions))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(filtered_positions)
                self.unique_positions = kmeans.cluster_centers_.tolist()
                cluster_labels = kmeans.labels_
                clustering_method = "K-means"

            # Marca que a clusterização inicial foi bem-sucedida
            if not self.initial_clustering_done and len(self.unique_positions) > 0:
                self.get_logger().info("✅ Initial clustering successful. System is now fully operational.")
                self.initial_clustering_done = True
            
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
            self.get_logger().info(f"Published {len(self.unique_positions)} unique positions using {clustering_method} clustering.")
            
            # CSV logging for post-clustering results
            try:
                csv_file_clusters = os.path.join(csv_dir, f"cluster_centers_{clustering_method}_{timestamp_str}.csv")
                with open(csv_file_clusters, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['cluster_id', 'x_m', 'y_m', 'z_m', 'distance_from_origin_m', 'clustering_method', 'timestamp'])
                    
                    for i, pos in enumerate(self.unique_positions):
                        distance = np.sqrt(pos[0]**2 + pos[1]**2)
                        writer.writerow([i, pos[0], pos[1], pos[2], distance, clustering_method, datetime.now().isoformat()])
                
                self.get_logger().info(f"📊 Saved {len(self.unique_positions)} cluster centers ({clustering_method}) to: {csv_file_clusters}")
            except Exception as e:
                self.get_logger().warning(f"Cluster CSV logging failed: {e}")
            
            # Publish visual markers
            self.publish_visualization_markers(positions_array, filtered_positions, self.unique_positions, cluster_labels)
            
            # Calculate and log detection accuracy
            self.calculate_detection_accuracy(self.unique_positions)
            
        except Exception as e:
            self.get_logger().error(f"Error in process_positions: {e}")
            traceback.print_exc()

    def calculate_detection_accuracy(self, cluster_centers, save_results=False, output_dir="."):
        """
        Calculate accuracy metrics comparing detected cluster centers with ground truth.
        
        Args:
            cluster_centers (list): Detected cluster center positions
            save_results (bool): If True, saves metrics to a JSON file.
            output_dir (str): Directory to save the results file.
            
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
        
        metrics = {
            'avg_distance_error': avg_distance,
            'max_distance_error': max_distance,
            'min_distance_error': min_distance,
            'accurate_detections': accurate_detections,
            'detection_rate': detection_rate,
            'detected_clusters': len(cluster_centers),
            'matches': matches
        }

        if save_results:
            try:
                os.makedirs(output_dir, exist_ok=True)
                # Usar um timestamp para evitar sobreescrever arquivos
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = os.path.join(output_dir, f'accuracy_results_{timestamp}.json')
                
                # Salva os parâmetros usados nesta execução junto com os resultados
                params_used = {
                    'collinearity_weight': self.collinearity_weight,
                    'line_tolerance': self.line_tolerance,
                    'min_line_points': self.min_line_points
                }
                
                full_results = {
                    'parameters': params_used,
                    'metrics': metrics
                }
                
                with open(filepath, 'w') as f:
                    json.dump(full_results, f, indent=4)
                self.get_logger().info(f"💾 Accuracy results saved to {filepath}")
                
            except Exception as e:
                self.get_logger().error(f"Failed to save accuracy results: {e}")
        
        return metrics


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
        # Chama o hook de desligamento antes de destruir o nó
        coordinate_processor.on_shutdown()
        coordinate_processor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
