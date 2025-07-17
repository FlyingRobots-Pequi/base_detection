from dataclasses import dataclass
from typing import List
from rclpy.node import Node


@dataclass
class HSVFilterParams:
    """Parameters for HSV color filtering."""

    lower: List[int]
    upper: List[int]


@dataclass
class ImageInferencerParams:
    """Parameters for the ImageInferencer node."""

    model_path: str
    detection_threshold: float
    hsv_filter: HSVFilterParams


@dataclass
class CameraParams:
    """Camera intrinsic and extrinsic parameters."""

    name: str
    bias_x: float
    bias_y: float
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float
    rgb_width: int
    rgb_height: int
    depth_width: int
    depth_height: int


@dataclass
class MotionParams:
    """Parameters for motion compensation and stability assessment."""

    compensation_enabled: bool
    adaptive_timeout_enabled: bool
    stability_threshold: float
    processing_delay_estimate: float
    base_frame_timeout: float
    min_frame_timeout: float
    max_frame_timeout: float
    velocity_scale_factor: float
    confidence_factor: float
    velocity_outlier_threshold: float
    outlier_velocity_threshold: float
    outlier_distance_threshold: float


@dataclass
class CoordinateReceiverParams:
    """Parameters for the CoordinateReceiver node."""

    csv_logs_dir: str
    camera: "CameraParams"
    motion: "MotionParams"
    high_accuracy_pixel_threshold: float
    confirmed_base_filter_radius: float


@dataclass
class InitialBaseParams:
    """Parameters for the initial base exclusion zone."""

    exclusion_radius: float
    x: float
    y: float


@dataclass
class ClusteringParams:
    """Parameters for the clustering algorithm."""

    use_line_based: bool
    line_tolerance: float
    min_line_points: int
    ransac_iterations: int
    collinearity_weight: float
    min_distance_between_bases: float


class CoordinateProcessorParams:
    """Parameters for the CoordinateProcessor node."""
    def __init__(self, node: Node):
        self.expected_bases = node.declare_parameter(
            "expected_bases", 5
        ).get_parameter_value().integer_value
        self.outlier_threshold = node.declare_parameter(
            "outlier_threshold", 2.0
        ).get_parameter_value().double_value
        self.output_dir = node.declare_parameter(
            "output_dir", "/root/ros2_ws/tuning_results/default_run"
        ).get_parameter_value().string_value
        self.clustering_logs_dir = node.declare_parameter(
            "clustering_logs_dir", "/root/ros2_ws/clustering_logs"
        ).get_parameter_value().string_value
        self.initial_base = get_initial_base_params(node)

        # Declare the flattened list and its dimension
        node.declare_parameter(
            "ground_truth_bases",
            [-0.24, -3.23, 0.75, -5.05, 5.16, -5.75, 4.37, -2.30, 5.69, -0.25],
        )
        node.declare_parameter("ground_truth_bases_dim", 2)
        
        # Get the flattened list and reshape it
        flat_bases = node.get_parameter("ground_truth_bases").value
        dim = node.get_parameter("ground_truth_bases_dim").value
        self.ground_truth_bases = [
            flat_bases[i : i + dim] for i in range(0, len(flat_bases), dim)
        ]

        self.clustering = get_clustering_params(node)
        self.confirmed_base_filter_radius = node.declare_parameter(
            "confirmed_base_filter_radius", 0.1
        ).get_parameter_value().double_value


def get_image_inferencer_params(node: Node) -> ImageInferencerParams:
    """Declare and get parameters for the ImageInferencer node."""
    node.declare_parameter(
        "model_path", "/root/ros2_ws/src/base_detection/base_detection/best.pt"
    )
    node.declare_parameter("detection_threshold", 0.9)
    node.declare_parameter("hsv_filter.lower", [42, 30, 120])
    node.declare_parameter("hsv_filter.upper", [135, 190, 220])

    hsv_filter = HSVFilterParams(
        lower=node.get_parameter("hsv_filter.lower").value,
        upper=node.get_parameter("hsv_filter.upper").value,
    )

    return ImageInferencerParams(
        model_path=node.get_parameter("model_path").value,
        detection_threshold=node.get_parameter("detection_threshold").value,
        hsv_filter=hsv_filter,
    )


def get_coordinate_receiver_params(node: Node) -> CoordinateReceiverParams:
    """Declare and get parameters for the CoordinateReceiver node."""
    node.declare_parameter("csv_logs_dir", "/root/ros2_ws/detection_logs")
    node.declare_parameter("motion_compensation_enabled", True)
    node.declare_parameter("adaptive_timeout_enabled", True)
    node.declare_parameter("stability_threshold", 0.1)
    node.declare_parameter("processing_delay_estimate", 0.1)
    node.declare_parameter("base_frame_timeout", 1.0)
    node.declare_parameter("min_frame_timeout", 0.5)
    node.declare_parameter("max_frame_timeout", 2.0)
    node.declare_parameter("velocity_scale_factor", 1.0)
    node.declare_parameter("confidence_factor", 0.9)
    node.declare_parameter("velocity_outlier_threshold", 2.0)
    node.declare_parameter("outlier_velocity_threshold", 2.0)
    node.declare_parameter("outlier_distance_threshold", 5.0)
    node.declare_parameter("high_accuracy_pixel_threshold", 15.0)
    node.declare_parameter("confirmed_base_filter_radius", 0.3)

    node.declare_parameter("camera.name", "D455")
    node.declare_parameter("camera.bias_x", 0.02447)
    node.declare_parameter("camera.bias_y", 0.0)
    node.declare_parameter("camera.fx", 610.0)
    node.declare_parameter("camera.fy", 610.0)
    node.declare_parameter("camera.cx", 320.0)
    node.declare_parameter("camera.cy", 240.0)
    node.declare_parameter("camera.baseline", 0.075)
    node.declare_parameter("camera.rgb_width", 640)
    node.declare_parameter("camera.rgb_height", 480)
    node.declare_parameter("camera.depth_width", 640)
    node.declare_parameter("camera.depth_height", 480)

    camera_params = CameraParams(
        name=node.get_parameter("camera.name").value,
        bias_x=node.get_parameter("camera.bias_x").value,
        bias_y=node.get_parameter("camera.bias_y").value,
        fx=node.get_parameter("camera.fx").value,
        fy=node.get_parameter("camera.fy").value,
        cx=node.get_parameter("camera.cx").value,
        cy=node.get_parameter("camera.cy").value,
        baseline=node.get_parameter("camera.baseline").value,
        rgb_width=node.get_parameter("camera.rgb_width").value,
        rgb_height=node.get_parameter("camera.rgb_height").value,
        depth_width=node.get_parameter("camera.depth_width").value,
        depth_height=node.get_parameter("camera.depth_height").value,
    )

    motion_params = MotionParams(
        compensation_enabled=node.get_parameter("motion_compensation_enabled").value,
        adaptive_timeout_enabled=node.get_parameter("adaptive_timeout_enabled").value,
        stability_threshold=node.get_parameter("stability_threshold").value,
        processing_delay_estimate=node.get_parameter("processing_delay_estimate").value,
        base_frame_timeout=node.get_parameter("base_frame_timeout").value,
        min_frame_timeout=node.get_parameter("min_frame_timeout").value,
        max_frame_timeout=node.get_parameter("max_frame_timeout").value,
        velocity_scale_factor=node.get_parameter("velocity_scale_factor").value,
        confidence_factor=node.get_parameter("confidence_factor").value,
        velocity_outlier_threshold=node.get_parameter("velocity_outlier_threshold").value,
        outlier_velocity_threshold=node.get_parameter("outlier_velocity_threshold").value,
        outlier_distance_threshold=node.get_parameter("outlier_distance_threshold").value,
    )

    return CoordinateReceiverParams(
        csv_logs_dir=node.get_parameter("csv_logs_dir").value,
        camera=camera_params,
        motion=motion_params,
        high_accuracy_pixel_threshold=node.get_parameter("high_accuracy_pixel_threshold").value,
        confirmed_base_filter_radius=node.get_parameter(
            "confirmed_base_filter_radius"
        ).value,
    )


def get_coordinate_processor_params(node: Node) -> CoordinateProcessorParams:
    """Declare and get parameters for the CoordinateProcessor node."""
    return CoordinateProcessorParams(node)


def get_initial_base_params(node: Node) -> InitialBaseParams:
    """Declare and get parameters for the initial base."""
    node.declare_parameter("initial_base.exclusion_radius", 0.7)
    node.declare_parameter("initial_base.x", 0.0)
    node.declare_parameter("initial_base.y", 0.0)
    
    return InitialBaseParams(
        exclusion_radius=node.get_parameter("initial_base.exclusion_radius").value,
        x=node.get_parameter("initial_base.x").value,
        y=node.get_parameter("initial_base.y").value,
    )

def get_clustering_params(node: Node) -> ClusteringParams:
    """Declare and get parameters for clustering."""
    node.declare_parameter("use_line_based_clustering", True)
    node.declare_parameter("min_distance_between_bases", 1.0)
    node.declare_parameter("line_tolerance", 0.3)
    node.declare_parameter("min_line_points", 3)
    node.declare_parameter("line_detection_ransac_iterations", 100)
    node.declare_parameter("collinearity_weight", 15.0)

    return ClusteringParams(
        use_line_based=node.get_parameter("use_line_based_clustering").value,
        line_tolerance=node.get_parameter("line_tolerance").value,
        min_line_points=node.get_parameter("min_line_points").value,
        ransac_iterations=node.get_parameter("line_detection_ransac_iterations").value,
        collinearity_weight=node.get_parameter("collinearity_weight").value,
        min_distance_between_bases=node.get_parameter(
            "min_distance_between_bases"
        ).value,
    )
