#!/usr/bin/env python3
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
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import numpy as np
from px4_msgs.msg import VehicleLocalPosition
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from .variables import (
    DETECTED_COORDINATES_TOPIC,
    DELTA_POINTS_TOPIC,
    ABSOLUTE_POINTS_TOPIC,
    DEPTH_IMAGE_TOPIC,
    VEHICLE_LOCAL_POSITION_TOPIC,
)
from .parameters import get_coordinate_receiver_params
from .utils import assess_motion_stability, calculate_motion_compensation


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

        self.get_logger().info("CoordinateReceiver initialized.")

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

        for x1, y1, x2, y2, score in detections:
            result = self._process_one_detection(x1, y1, x2, y2, compensated_pose)
            if result:
                absolute_point, delta_point, is_remote = result
                self.delta_publisher.publish(delta_point)
                if is_remote:
                    self.absolute_publisher.publish(absolute_point)

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

        depth_h, depth_w = self.latest_depth.shape
        mid_x_depth = int(mid_x_rgb * (depth_w / cam.rgb_width))
        mid_y_depth = int(mid_y_rgb * (depth_h / cam.rgb_height))

        if not (0 <= mid_y_depth < depth_h and 0 <= mid_x_depth < depth_w):
            return None

        depth_value = self.latest_depth[mid_y_depth, mid_x_depth]
        if np.isnan(depth_value) or depth_value <= 0:
            return None

        z = depth_value / 1000.0
        comp_x, comp_y, comp_z, comp_yaw = compensated_pose

        delta_x = (mid_x_depth - cam.cx) * z / cam.fx + cam.baseline - cam.bias_x
        delta_y = (mid_y_depth - cam.cy) * z / cam.fy - cam.bias_y

        cos_yaw, sin_yaw = np.cos(comp_yaw), np.sin(comp_yaw)
        global_x = comp_x + (delta_x * cos_yaw - delta_y * sin_yaw)
        global_y = comp_y + (delta_x * sin_yaw + delta_y * cos_yaw)

        absolute_point = Point(x=global_x, y=global_y, z=comp_z - z)
        delta_point = Point(x=delta_x, y=delta_y, z=comp_z - z)

        # A simplified check. For full decoupling, this parameter would
        # be part of a shared config or this node's own parameters.
        is_remote = np.sqrt(global_x**2 + global_y**2) > 0.7

        return absolute_point, delta_point, is_remote


def main(args=None):
    rclpy.init(args=args)
    node = CoordinateReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
