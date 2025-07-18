"""
ROS2 node to process base detection coordinates with motion compensation.

Receives detections, depth images, and vehicle telemetry to calculate
real-world 3D positions of bases.
"""

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point, PoseArray
from cv_bridge import CvBridge
import numpy as np
from px4_msgs.msg import VehicleLocalPosition
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from base_detection.variables import (
    DETECTED_COORDINATES_TOPIC,
    DELTA_POINTS_TOPIC,
    ABSOLUTE_POINTS_TOPIC,
    HIGH_ACCURACY_POINT_TOPIC,
    CONFIRMED_BASES_TOPIC,
    DEPTH_IMAGE_TOPIC,
    VEHICLE_LOCAL_POSITION_TOPIC,
)
from base_detection.parameters import get_coordinate_receiver_params
from base_detection.utils import calculate_motion_compensation


class CoordinateReceiver(Node):
    """
    Processes detected coordinates and depth data to calculate real-world positions.
    Handles resolution differences and uses vehicle telemetry for motion compensation.
    """

    def __init__(self):
        """Initializes the node, subscriptions, publishers, and parameters."""
        super().__init__("coordinate_receiver")
        self.params = get_coordinate_receiver_params(self)
        self.bridge = CvBridge()

        self.latest_depth = None
        self.confirmed_bases = []
        self.vehicle_state = {
            "x": 0.0,
            "y": 0.0,
            "alt": 0.0,
            "yaw": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "az": 0.0,
            "timestamp": 0.0,
        }

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            Float32MultiArray, DETECTED_COORDINATES_TOPIC, self.listener_callback, 10
        )
        self.depth_subscription = self.create_subscription(
            Image, DEPTH_IMAGE_TOPIC, self.depth_callback, 10
        )
        self.confirmed_bases_subscription = self.create_subscription(
            PoseArray, CONFIRMED_BASES_TOPIC, self.confirmed_bases_callback, 10
        )
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition, 
            VEHICLE_LOCAL_POSITION_TOPIC, 
            self.vehicle_local_position_callback, 
            qos_profile,
        )
        self.delta_publisher = self.create_publisher(Point, DELTA_POINTS_TOPIC, 10)
        self.absolute_publisher = self.create_publisher(
            Point, ABSOLUTE_POINTS_TOPIC, 10
        )
        self.high_accuracy_publisher = self.create_publisher(
            Point, HIGH_ACCURACY_POINT_TOPIC, 10
        )

        self.get_logger().info("CoordinateReceiver initialized.")

    def confirmed_bases_callback(self, msg: PoseArray):
        """Callback to update the list of confirmed bases."""
        self.confirmed_bases = [
            (pose.position.x, pose.position.y) for pose in msg.poses
        ]
        self.get_logger().debug(f"Updated confirmed bases: {len(self.confirmed_bases)} bases.")

    def depth_callback(self, msg: Image):
        """Callback to process and store incoming depth images."""
        self.latest_depth = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        ).astype(np.float32)

    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        """Callback to update vehicle position, orientation, and motion data."""
        self.vehicle_state.update(
            {
                "x": msg.x,
                "y": msg.y,
                "alt": msg.z,
                "yaw": msg.heading,
                "vx": msg.vx,
                "vy": msg.vy,
                "vz": msg.vz,
                "ax": msg.ax,
                "ay": msg.ay,
                "az": msg.az,
                "timestamp": time.time(),
            }
        )

    def listener_callback(self, msg: Float32MultiArray):
        """Processes batched base detections with motion compensation."""
        if self.latest_depth is None:
            self.get_logger().warning("No depth data available, skipping processing.")
            return
            
        detections = [msg.data[i : i + 5] for i in range(0, len(msg.data), 5)]
        if not detections:
            return
            
        self.get_logger().debug(f"Processing {len(detections)} detections.")

        compensated_pose = self._get_compensated_pose()
        correction_vector = np.array([0.0, 0.0])

        # First Pass: Find correction vector using a confirmed base
        if self.confirmed_bases:
            for x1, y1, x2, y2, score in detections:
                result = self._process_one_detection(x1, y1, x2, y2, compensated_pose)
                if result:
                    absolute_point, _, _, _ = result
                    current_pos = np.array([absolute_point.x, absolute_point.y])
                    for confirmed_pos in self.confirmed_bases:
                        dist = np.linalg.norm(current_pos - np.array(confirmed_pos))
                        # Use a threshold to match a detection to a confirmed base
                        if dist < self.params.confirmed_base_filter_radius: 
                            correction_vector = current_pos - np.array(confirmed_pos)
                            self.get_logger().info(f"Correction vector calculated: {correction_vector}")
                            break  # Found a match, use this correction
                if not np.all(correction_vector == 0.0):
                    break # Exit outer loop once correction is found

        # Second Pass: Process all detections and apply correction
        for x1, y1, x2, y2, score in detections:
            result = self._process_one_detection(x1, y1, x2, y2, compensated_pose)
            if result:
                absolute_point, delta_point, is_high_accuracy = result
                
                # Apply correction
                corrected_x = absolute_point.x - correction_vector[0]
                corrected_y = absolute_point.y - correction_vector[1]
                corrected_absolute_point = Point(x=corrected_x, y=corrected_y, z=absolute_point.z)

                self.delta_publisher.publish(delta_point)

                if is_high_accuracy:
                    self.get_logger().info(
                        "High accuracy point detected! Publishing to dedicated topic."
                    )
                    self.high_accuracy_publisher.publish(corrected_absolute_point)
                else:
                    self.absolute_publisher.publish(corrected_absolute_point)

    def _get_compensated_pose(self):
        """Gets the vehicle pose, compensated for processing delay."""
        return calculate_motion_compensation(
            detection_timestamp=time.time(),
            vehicle_timestamp=self.vehicle_state["timestamp"],
            current_pose=(
                self.vehicle_state["x"],
                self.vehicle_state["y"],
                self.vehicle_state["alt"],
                self.vehicle_state["yaw"],
            ),
            current_velocity=(
                self.vehicle_state["vx"],
                self.vehicle_state["vy"],
                self.vehicle_state["vz"],
            ),
            current_acceleration=(
                self.vehicle_state["ax"],
                self.vehicle_state["ay"],
                self.vehicle_state["az"],
            ),
            motion_params=self.params.motion,
            logger=self.get_logger(),
        )

    def _process_one_detection(self, x1, y1, x2, y2, compensated_pose):
        """Transforms a single detection into 3D coordinates."""
        cam = self.params.camera
        mid_x_rgb = (x1 + x2) / 2
        mid_y_rgb = (y1 + y2) / 2

        # High accuracy check in pixel space
        image_center_x = cam.rgb_width / 2
        image_center_y = cam.rgb_height / 2
        pixel_dist = np.sqrt(
            (mid_x_rgb - image_center_x) ** 2 + (mid_y_rgb - image_center_y) ** 2
        )
        is_high_accuracy = pixel_dist < self.params.high_accuracy_pixel_threshold

        # Calculate a normalized weight (0 to 1). 1 is best (center of image).
        max_dist = np.sqrt(image_center_x**2 + image_center_y**2)
        weight = max(0.0, 1.0 - (pixel_dist / max_dist))

        depth_h, depth_w = self.latest_depth.shape
        mid_x_depth = int(mid_x_rgb * (depth_w / cam.rgb_width))
        mid_y_depth = int(mid_y_rgb * (depth_h / cam.rgb_height))

        if not (0 <= mid_y_depth < depth_h and 0 <= mid_x_depth < depth_w):
            return None

        # --- Depth Sampling with Median Filter ---
        patch_size = 5
        half_patch = patch_size // 2
        
        y_start = max(0, mid_y_depth - half_patch)
        y_end = min(depth_h, mid_y_depth + half_patch + 1)
        x_start = max(0, mid_x_depth - half_patch)
        x_end = min(depth_w, mid_x_depth + half_patch + 1)

        depth_patch = self.latest_depth[y_start:y_end, x_start:x_end]
        valid_depths = depth_patch[np.isfinite(depth_patch) & (depth_patch > 0)]

        if valid_depths.size == 0:
            return None # No valid depth points in the patch

        depth_value = np.median(valid_depths)
        # --- End of Depth Sampling ---

        if np.isnan(depth_value) or depth_value <= 0:
            return None

        z = depth_value / 1000.0
        comp_x, comp_y, comp_z, comp_yaw = compensated_pose

        delta_x = (mid_x_depth - cam.cx) * z / cam.fx + cam.baseline - cam.bias_x
        delta_y = (mid_y_depth - cam.cy) * z / cam.fy - cam.bias_y

        cos_yaw, sin_yaw = np.cos(comp_yaw), np.sin(comp_yaw)
        global_x = comp_x + (delta_x * cos_yaw - delta_y * sin_yaw)
        global_y = comp_y + (delta_x * sin_yaw + delta_y * cos_yaw)

        # Use the 'z' field to pass the weight of the detection
        absolute_point = Point(x=global_x, y=global_y, z=weight)
        delta_point = Point(x=delta_x, y=delta_y, z=comp_z - z)

        return absolute_point, delta_point, is_high_accuracy


def main(args=None):
    """Initializes and spins the CoordinateReceiver node."""
    rclpy.init(args=args)
    coordinate_receiver = CoordinateReceiver()
    rclpy.spin(coordinate_receiver)
    coordinate_receiver.destroy_node()


if __name__ == "__main__":
    main()
