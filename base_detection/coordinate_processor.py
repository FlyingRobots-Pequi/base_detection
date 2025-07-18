#!/usr/bin/env python3

"""
ROS2 node to process and cluster detected base coordinates.

Clusters absolute positions to find unique bases, saves them as setpoints,
publishes them for navigation, and creates RViz markers.
"""
import traceback
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from geometry_msgs.msg import Point, PoseArray, Pose
from px4_msgs.msg import VehicleLocalPosition, VehicleOdometry
import numpy as np
import os
from datetime import datetime
import csv
import json

from .variables import (
    ABSOLUTE_POINTS_TOPIC,
    UNIQUE_POSITIONS_TOPIC,
    VEHICLE_LOCAL_POSITION_TOPIC,
    HIGH_ACCURACY_POINT_TOPIC,
    CONFIRMED_BASES_TOPIC,
    VEHICLE_ODOMETRY_TOPIC,
)
from .parameters import get_coordinate_processor_params
from .clustering import run_hybrid_line_kmeans
from .visualization import MarkerManager
from .utils import remove_outliers_2d
from sklearn.cluster import KMeans


class CoordinateProcessor(Node):
    """
    Processes and clusters absolute base coordinates to identify unique base locations.

    This node collects potential base detections, periodically clusters them to find
    stable base positions, and publishes the results for navigation and visualization.
    """

    def __init__(self):
        """Initializes the node, subscriptions, publishers, and a processing timer."""
        super().__init__("coordinate_processor")

        self.params = get_coordinate_processor_params(self)
        self.marker_manager = MarkerManager(self)

        self.positions_list = []
        self.unique_positions = []
        self.confirmed_bases = []
        self.drone_position_history = []
        self.max_position_history = 50
        self.initial_clustering_done = False
        self.has_odometry = False
        
        sensor_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.absolute_subscription = self.create_subscription(
            Point, ABSOLUTE_POINTS_TOPIC, self.absolute_position_callback, 10
        )
        self.high_accuracy_subscription = self.create_subscription(
            Point,
            HIGH_ACCURACY_POINT_TOPIC,
            self.high_accuracy_callback,
            10,
        )
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            VEHICLE_LOCAL_POSITION_TOPIC,
            self.vehicle_local_position_callback,
            sensor_qos_profile,
        )
        self.odometry_sub = self.create_subscription(
            VehicleOdometry,
            VEHICLE_ODOMETRY_TOPIC,
            self.vehicle_odometry_callback,
            sensor_qos_profile,
        )
        self.unique_positions_publisher = self.create_publisher(
            PoseArray, UNIQUE_POSITIONS_TOPIC, 10
        )
        self.confirmed_bases_publisher = self.create_publisher(
            PoseArray, CONFIRMED_BASES_TOPIC, 10
        )

        # Timer for periodic processing
        self.processing_timer = self.create_timer(2.0, self.process_positions)

        self.get_logger().info(
            "CoordinateProcessor initialized and ready to process points."
        )

    def on_shutdown(self):
        """Performs final accuracy analysis on shutdown."""
        self.get_logger().info("Node shutting down. Performing final analysis...")
        if self.unique_positions:
            self.calculate_detection_accuracy(
                self.unique_positions,
                save_results=True,
                output_dir=self.params.output_dir,
            )
        else:
            self.get_logger().warn(
                "No unique positions detected, skipping final analysis."
            )

    def absolute_position_callback(self, msg: Point):
        """Callback to collect incoming absolute position detections."""
        if not self.is_near_initial_base(msg.x, msg.y):
            self.positions_list.append([msg.x, msg.y, msg.z])
            self.get_logger().debug(
                f"Collected new detection. Total points: {len(self.positions_list)}"
            )
        else:
            self.get_logger().debug(
                f"Filtered out a detection at ({msg.x:.2f}, {msg.y:.2f}) near the initial base."
            )

    def high_accuracy_callback(self, msg: Point):
        """Callback for high-accuracy detections, which are treated as confirmed bases."""
        new_base = [msg.x, msg.y]

        # Ignore bases detected near the initial home position (0,0)
        if self.is_near_initial_base(new_base[0], new_base[1]):
            self.get_logger().debug(
                f"Ignoring high-accuracy point {new_base} near initial base."
            )
            return

        # Check if this base is too close to an already confirmed base
        is_duplicate = False
        for confirmed_base in self.confirmed_bases:
            distance = np.linalg.norm(np.array(new_base) - np.array(confirmed_base))
            if distance < self.params.clustering.min_distance_between_bases:
                is_duplicate = True
                break
        
        if not is_duplicate:
            self.get_logger().info(f"New confirmed base added at: {new_base}")
            self.confirmed_bases.append(new_base)
            self._publish_confirmed_bases() # Publish the updated list

            # Filter out points in positions_list that are close to the new confirmed base
            radius = self.params.confirmed_base_filter_radius
            points_to_keep = []
            for point in self.positions_list:
                if np.linalg.norm(np.array(point[:2]) - np.array(new_base)) > radius:
                    points_to_keep.append(point)
            
            removed_count = len(self.positions_list) - len(points_to_keep)
            if removed_count > 0:
                self.get_logger().info(f"Removed {removed_count} nearby points from processing list.")
            self.positions_list = points_to_keep

    def vehicle_odometry_callback(self, msg: VehicleOdometry):
        """Stores vehicle position history from odometry data."""
        self.has_odometry = True
        current_position = [msg.position[0], msg.position[1]]
        self._update_drone_position_history(current_position)

    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        """Stores vehicle position history (fallback if no odometry)."""
        if self.has_odometry:
            return  # Odometry has priority

        current_position = [msg.x, msg.y]
        self._update_drone_position_history(current_position)

    def _update_drone_position_history(self, current_position: list):
        """Helper to append a new position to the history list if it's moved enough."""
        if (
            not self.drone_position_history
            or np.linalg.norm(
                np.array(current_position) - np.array(self.drone_position_history[-1])
            )
            > 0.01
        ):
            self.drone_position_history.append(current_position)
        if len(self.drone_position_history) > self.max_position_history:
            self.drone_position_history.pop(0)

    def _run_clustering(self, filtered_positions: np.ndarray, n_clusters: int, sample_weights: np.ndarray):
        """Runs the appropriate clustering algorithm based on configuration."""
        if self.params.clustering.use_line_based:
            # Note: The custom line-based clustering does not support weights yet.
            # We will ignore weights for this method for now.
            movement_direction = self.estimate_drone_movement_direction()
            unique_positions, labels = run_hybrid_line_kmeans(
                filtered_positions,
                self.params.clustering,
                n_clusters,
                movement_direction,
                self.get_logger(),
            )
            method = "hybrid-line-kmeans"
        else:
            # Fallback to traditional K-Means
            n_clusters = min(n_clusters, len(filtered_positions))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(filtered_positions, sample_weight=sample_weights)
            unique_positions = kmeans.cluster_centers_.tolist()
            labels = kmeans.labels_
            method = "K-means"

        if not self.initial_clustering_done and unique_positions:
            self.get_logger().info(
                "✅ Initial clustering successful. System is now fully operational."
            )
            self.initial_clustering_done = True

        return unique_positions, labels, method

    def process_positions(self):
        """
        Periodically processes collected positions to identify unique bases.
        This method is called by a timer.
        """
        # Always publish markers to keep RViz updated, even if there's nothing new
        # This will also handle clearing markers if everything is empty
        defer_visualization = False

        if len(self.positions_list) < self.params.expected_bases:
            self.get_logger().info(
                f"Waiting for more points... ({len(self.positions_list)}/{self.params.expected_bases})"
            )
            defer_visualization = True

        # If all bases are confirmed, no need to cluster
        num_bases_to_find = self.params.expected_bases - len(self.confirmed_bases)
        if num_bases_to_find <= 0:
            self.get_logger().info("All expected bases have been confirmed. Skipping clustering.")
            self.unique_positions = self.confirmed_bases
            self._publish_unique_positions()
            # Defer visualization to the final block to keep logic clean
            defer_visualization = True

        if defer_visualization:
            self.marker_manager.publish_markers(
                all_positions=np.array(self.positions_list),
                filtered_positions=np.array(self.unique_positions),
                cluster_centers=self.unique_positions,
                ground_truth_bases=self.params.ground_truth_bases,
                initial_base_params=self.params.initial_base,
                cluster_labels=None,
            )
            return

        positions_array = np.array(self.positions_list)
        self._log_pre_clustering_data(positions_array)

        # Step 1: Filter outliers
        try:
            filtered_positions_with_weights = remove_outliers_2d(
                positions_array, self.params.outlier_threshold, self.get_logger()
            )

            if len(filtered_positions_with_weights) < self.params.clustering.min_line_points:
                self.get_logger().info(
                    f"Waiting for more points after filtering ({len(filtered_positions_with_weights)}/{self.params.clustering.min_line_points})"
                )
                return
        except Exception as e:
            self.get_logger().error(f"Error during outlier removal: {e}")
            traceback.print_exc()
            return

        # Step 2: Run clustering
        try:
            # Separate positions and weights for clustering
            filtered_positions = filtered_positions_with_weights[:, :2]
            sample_weights = filtered_positions_with_weights[:, 2]

            unique_positions, cluster_labels, method = self._run_clustering(
                filtered_positions, num_bases_to_find, sample_weights
            )
            # Combine confirmed bases with newly found clusters
            self.unique_positions = self.confirmed_bases + unique_positions
        except Exception as e:
            self.get_logger().error(f"Clustering failed: {e}", exc_info=True)
            # Visualize points even if clustering fails to aid debugging
            self.marker_manager.publish_markers(
                all_positions=positions_array,
                filtered_positions=filtered_positions,
                cluster_centers=[],
                ground_truth_bases=self.params.ground_truth_bases,
                initial_base_params=self.params.initial_base,
                cluster_labels=None,
            )
            return

        # Step 3: Post-processing and visualization
        try:
            self.unique_positions.sort(key=lambda pos: np.linalg.norm(pos))
            self._publish_unique_positions()
            self._log_post_clustering_data(method)
            # Combine all points for visualization
            self.marker_manager.publish_markers(
                all_positions=positions_array,
                filtered_positions=filtered_positions,
                cluster_centers=self.unique_positions,
                ground_truth_bases=self.params.ground_truth_bases,
                initial_base_params=self.params.initial_base,
                cluster_labels=cluster_labels,
            )
            self.calculate_detection_accuracy(self.unique_positions)
        except Exception as e:
            self.get_logger().error(
                f"Post-clustering processing or visualization failed: {e}",
                exc_info=True,
            )

    def _publish_unique_positions(self):
        """Publishes the unique base positions as a PoseArray."""
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"
        for pos in self.unique_positions:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = pos[0], pos[1], 0.0
            pose.orientation.w = 1.0  # Set default orientation
            pose_array.poses.append(pose)
        self.unique_positions_publisher.publish(pose_array)
        self.get_logger().info(
            f"Published {len(self.unique_positions)} unique positions."
        )

    def _publish_confirmed_bases(self):
        """Publishes the list of confirmed bases as a PoseArray."""
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"
        for pos in self.confirmed_bases:
            pose = Pose()
            # Assuming pos is [x, y], z is 0
            pose.position.x, pose.position.y, pose.position.z = pos[0], pos[1], 0.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
        self.confirmed_bases_publisher.publish(pose_array)
        self.get_logger().info(f"Published {len(self.confirmed_bases)} confirmed bases.")

    def estimate_drone_movement_direction(self) -> np.ndarray:
        """Estimates the primary drone movement direction from position history."""
        if len(self.drone_position_history) < 2:
            return np.array([1.0, 0.0])
        
        positions = np.array(self.drone_position_history)
        movements = np.diff(positions, axis=0)
        significant_movements = movements[np.linalg.norm(movements, axis=1) > 0.1]

        if len(significant_movements) == 0:
            return np.array([1.0, 0.0])

        avg_movement = np.mean(significant_movements, axis=0)
        norm = np.linalg.norm(avg_movement)
        return avg_movement / norm if norm > 0 else np.array([1.0, 0.0])

    def is_near_initial_base(self, x: float, y: float) -> bool:
        """Checks if a position is within the exclusion radius of the initial base."""
        params = self.params.initial_base
        return (
            np.sqrt((x - params.x) ** 2 + (y - params.y) ** 2)
            <= params.exclusion_radius
        )

    def _log_pre_clustering_data(self, positions_array: np.ndarray):
        """Logs the collected raw data points to a CSV file before processing."""
        try:
            log_dir = self.params.clustering_logs_dir
            os.makedirs(log_dir, exist_ok=True)
            filepath = os.path.join(
                log_dir,
                f"positions_before_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["x_m", "y_m", "z_m", "distance_from_origin_m", "timestamp"]
                )
                for pos in positions_array:
                    writer.writerow(
                        [
                            pos[0],
                            pos[1],
                            pos[2],
                            np.linalg.norm(pos[:2]),
                            datetime.now().isoformat(),
                        ]
                    )
            self.get_logger().debug(
                f"Saved {len(positions_array)} raw points to {filepath}"
            )
        except Exception as e:
            self.get_logger().warning(f"Pre-clustering CSV logging failed: {e}")

    def _log_post_clustering_data(self, clustering_method: str):
        """Logs the final cluster centers to a CSV file."""
        try:
            log_dir = self.params.clustering_logs_dir
            os.makedirs(log_dir, exist_ok=True)
            filepath = os.path.join(
                log_dir,
                f"cluster_centers_{clustering_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )
            with open(filepath, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "cluster_id",
                            "x_m",
                            "y_m",
                            "z_m",
                            "distance_from_origin_m",
                            "clustering_method",
                            "timestamp",
                        ]
                    )
                    for i, pos in enumerate(self.unique_positions):
                        writer.writerow(
                            [
                                i,
                                pos[0],
                                pos[1],
                                pos[2] if len(pos) > 2 else 0.0, # Handle 2D and 3D points
                                np.linalg.norm(pos[:2]),
                                clustering_method,
                                datetime.now().isoformat(),
                            ]
                        )
            self.get_logger().debug(
                f"Saved {len(self.unique_positions)} cluster centers to {filepath}"
            )
        except Exception as e:
            self.get_logger().warning(f"Post-clustering CSV logging failed: {e}")

    def calculate_detection_accuracy(
        self, cluster_centers, save_results=False, output_dir="."
    ):
        """Calculates accuracy by comparing detected cluster centers with ground truth."""
        if not cluster_centers or not self.params.ground_truth_bases:
            return {}
            
        detected = np.array(cluster_centers)
        ground_truth = np.array(self.params.ground_truth_bases)
        
        matches = []
        unmatched_gt = list(range(len(ground_truth)))

        for i, det_pos in enumerate(detected):
            distances = np.linalg.norm(ground_truth[unmatched_gt] - det_pos[:2], axis=1)
            if distances.size > 0:
                min_dist_idx = np.argmin(distances)
                gt_idx = unmatched_gt.pop(min_dist_idx)
                matches.append(
                    {
                        "detected_idx": i,
                        "gt_idx": gt_idx,
                        "error": distances[min_dist_idx],
                    }
                )

        if not matches:
            return {}

        errors = [m["error"] for m in matches]
        avg_error = np.mean(errors)
        detection_rate = len(matches) / len(ground_truth)

        self.get_logger().info(
            f"Accuracy: Avg Error={avg_error:.3f}m, Detection Rate={detection_rate:.2%}"
        )

        if save_results:
            self._save_accuracy_results(
                errors, len(matches), detection_rate, output_dir
            )

    def _save_accuracy_results(
        self, errors, detected_count, detection_rate, output_dir
    ):
        """Saves accuracy metrics to a JSON file."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(
            output_dir,
            f'accuracy_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        )

            metrics = {
                "avg_distance_error": np.mean(errors) if errors else 0,
                "max_distance_error": np.max(errors) if errors else 0,
                "min_distance_error": np.min(errors) if errors else 0,
                "detected_clusters": detected_count,
                "detection_rate": detection_rate,
                }
            
            full_results = {
            "parameters": {
                "collinearity_weight": self.params.clustering.collinearity_weight,
                "line_tolerance": self.params.clustering.line_tolerance,
                "min_line_points": self.params.clustering.min_line_points,
            },
            "metrics": metrics,
            }

            with open(filepath, "w") as f:
                json.dump(full_results, f, indent=4)
            self.get_logger().info(f"💾 Accuracy results saved to {filepath}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to save accuracy results: {e}")


def main(args=None):
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
