from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from typing import List
import numpy as np

from .parameters import InitialBaseParams


class MarkerManager:
    """Manages the creation and publication of RViz visualization markers."""

    def __init__(self, node: Node):
        """
        Initializes the MarkerManager.

        Args:
            node: The ROS 2 node to use for publishing and logging.
        """
        self._node = node
        self._logger = node.get_logger()
        self._publisher = node.create_publisher(
            MarkerArray, "/base_detection/visualization_markers", 10
        )
        self._marker_id_counter = 0

    def publish_markers(
        self,
        all_positions: np.ndarray,
        filtered_positions: np.ndarray,
        cluster_centers: List[np.ndarray],
        ground_truth_bases: List[List[float]],
        initial_base_params: InitialBaseParams,
        cluster_labels: np.ndarray = None,
    ):
        """
        Creates and publishes all visualization markers.
        """
        try:
            # First, publish a marker array to clear all previous markers in our namespace
            clear_markers = MarkerArray()
            clear_marker = Marker()
            clear_marker.ns = "base_detection"
            clear_marker.action = Marker.DELETEALL
            clear_markers.markers.append(clear_marker)
            self._publisher.publish(clear_markers)

            # Then, publish the new markers
            markers = self._create_visualization_markers(
                all_positions,
                filtered_positions,
                cluster_centers,
                ground_truth_bases,
                initial_base_params,
                cluster_labels,
            )
            self._publisher.publish(markers)
            self._logger.info(f"Published {len(markers.markers)} RViz markers.")
        except Exception as e:
            self._logger.error(f"Error creating/publishing RViz markers: {e}")

    def _create_marker(
        self, marker_type, position, color, scale, text="", marker_id=None
    ):
        """Helper function to create a single RViz marker."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self._node.get_clock().now().to_msg()
        marker.ns = "base_detection"

        if marker_id is None:
            marker.id = self._marker_id_counter
            self._marker_id_counter += 1
        else:
            marker.id = marker_id

        marker.type = marker_type
        marker.action = Marker.ADD

        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2]) if len(position) > 2 else 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x, marker.scale.y, marker.scale.z = (
            float(scale[0]),
            float(scale[1]),
            float(scale[2]),
        )
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
            float(color[0]),
            float(color[1]),
            float(color[2]),
            float(color[3]),
        )

        if text:
            marker.text = text

        return marker

    def _add_ground_truth_markers(
        self, markers: MarkerArray, ground_truth_bases: List[List[float]]
    ):
        """Adds markers for ground truth base positions."""
        for i, base_pos in enumerate(ground_truth_bases):
            markers.markers.append(
                self._create_marker(
                    Marker.CYLINDER,
                    (base_pos[0], base_pos[1], 0.0),
                    (1.0, 0.647, 0.0, 0.8),
                    (0.3, 0.3, 0.1),
                )
            )
            markers.markers.append(
                self._create_marker(
                    Marker.TEXT_VIEW_FACING,
                    (base_pos[0], base_pos[1], 0.3),
                    (1.0, 0.647, 0.0, 1.0),
                    (0.3, 0.3, 0.3),
                    f"GT_BASE_{i+1}",
                )
            )

    def _add_initial_base_markers(
        self, markers: MarkerArray, params: InitialBaseParams
    ):
        """Adds markers for the initial base position and exclusion radius."""
        if params.x != 0 or params.y != 0:
            markers.markers.append(
                self._create_marker(
                    Marker.CYLINDER,
                    (params.x, params.y, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                    (0.4, 0.4, 0.1),
                )
            )
            markers.markers.append(
                self._create_marker(
                    Marker.CYLINDER,
                    (params.x, params.y, 0.0),
                    (0.0, 0.0, 0.0, 0.2),
                    (params.exclusion_radius * 2, params.exclusion_radius * 2, 0.02),
                )
            )
            markers.markers.append(
                self._create_marker(
                    Marker.TEXT_VIEW_FACING,
                    (params.x, params.y, 0.3),
                    (0.0, 0.0, 0.0, 1.0),
                    (0.3, 0.3, 0.3),
                    "INITIAL_BASE",
                )
            )

    def _add_outlier_markers(
        self, markers: MarkerArray, all_positions, filtered_positions
    ):
        """Adds markers for outlier points."""
        if len(all_positions) > 0 and len(filtered_positions) > 0:
            filtered_set = set(map(tuple, filtered_positions))
            for pos in all_positions:
                if tuple(pos) not in filtered_set:
                    markers.markers.append(
                        self._create_marker(
                            Marker.SPHERE,
                            (pos[0], pos[1], 0.0),
                            (0.5, 0.5, 0.5, 0.6),
                            (0.1, 0.1, 0.1),
                        )
                    )

    def _add_clustered_point_markers(
        self, markers: MarkerArray, filtered_positions, cluster_labels
    ):
        """Adds markers for the filtered points, colored by cluster."""
        cluster_colors = [
            (1.0, 0.0, 1.0, 0.8),
            (0.0, 1.0, 1.0, 0.8),
            (1.0, 1.0, 0.0, 0.8),
            (1.0, 0.5, 0.0, 0.8),
            (0.5, 0.0, 1.0, 0.8),
            (0.0, 0.5, 1.0, 0.8),
            (1.0, 0.0, 0.5, 0.8),
        ]

        if len(filtered_positions) > 0:
            for i, pos in enumerate(filtered_positions):
                color = (0.0, 1.0, 0.0, 0.8)
                if cluster_labels is not None and i < len(cluster_labels):
                    label = cluster_labels[i]
                    if label == -1:
                        color = (0.5, 0.5, 0.5, 0.6)
                    else:
                        color = cluster_colors[label % len(cluster_colors)]

                markers.markers.append(
                    self._create_marker(
                        Marker.SPHERE, (pos[0], pos[1], 0.0), color, (0.15, 0.15, 0.15)
                    )
                )

    def _add_cluster_center_markers(self, markers: MarkerArray, cluster_centers):
        """Adds markers for the calculated cluster centers."""
        for i, center in enumerate(cluster_centers):
            markers.markers.append(
                self._create_marker(
                    Marker.SPHERE,
                    (center[0], center[1], 0.0),
                    (0.0, 0.0, 1.0, 1.0),
                    (0.25, 0.25, 0.25),
                )
            )
            markers.markers.append(
                self._create_marker(
                    Marker.TEXT_VIEW_FACING,
                    (center[0], center[1], 0.3),
                    (0.0, 0.0, 1.0, 1.0),
                    (0.3, 0.3, 0.3),
                    f"CLUSTER_{i+1}",
                )
            )

    def _create_visualization_markers(
        self,
        all_positions,
        filtered_positions,
        cluster_centers,
        ground_truth_bases,
        initial_base_params,
        cluster_labels=None,
    ):
        """Creates RViz markers for all detected data points and clusters."""
        markers = MarkerArray()
        self._marker_id_counter = 0

        self._add_ground_truth_markers(markers, ground_truth_bases)
        self._add_initial_base_markers(markers, initial_base_params)
        self._add_outlier_markers(markers, all_positions, filtered_positions)
        self._add_clustered_point_markers(markers, filtered_positions, cluster_labels)
        self._add_cluster_center_markers(markers, cluster_centers)

        return markers
