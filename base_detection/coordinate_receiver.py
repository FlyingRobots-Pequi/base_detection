#!/usr/bin/env python3
"""
Coordinate Receiver Node for Base Detection System.

This module implements a ROS2 node that receives and processes base detection coordinates
along with depth information from a D435i camera. It calculates real-world 3D positions
of detected bases and publishes delta positions for navigation.

The node handles:
- Processing of detected base coordinates
- Depth image processing
- Vehicle position tracking
- 3D position calculation with camera parameters
- Safety checks and failsafe triggers

Dependencies:
    - ROS2
    - OpenCV
    - NumPy
    - cv_bridge
    - PX4 Messages
    - Sensor Messages
    - Geometry Messages
"""

import traceback
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
from px4_msgs.msg import VehicleLocalPosition
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from base_detection.variables import (
    DEPTH_IMAGE_TOPIC,
    VEHICLE_LOCAL_POSITION_TOPIC,
    DETECTED_COORDINATES_TOPIC,
    DELTA_POSITION_TOPIC,
    D455_BASELINE as BASELINE,
    D455_BIAS_X as BIAS_X,
    D455_BIAS_Y as BIAS_Y,
    D455_FX_DEPTH as FX_DEPTH,
    D455_FY_DEPTH as FY_DEPTH,
    D455_CX_DEPTH as CX_DEPTH,
    D455_CY_DEPTH as CY_DEPTH
)


class CoordinateReceiver(Node):
    """
    A ROS2 node for processing and transforming detected base coordinates into real-world positions.

    This class subscribes to detected coordinates, depth images, and vehicle position data,
    processes this information to calculate real-world 3D positions of detected bases, and
    publishes the results for navigation purposes.

    Attributes:
        bias_x (float): X-axis bias correction value
        bias_y (float): Y-axis bias correction value
        fx_depth (float): Focal length in pixels (X-axis)
        fy_depth (float): Focal length in pixels (Y-axis)
        cx_depth (float): Principal point X coordinate
        cy_depth (float): Principal point Y coordinate
        baseline (float): Baseline between RGB and depth cameras
        latest_depth (numpy.ndarray): Latest processed depth image
        current_altitude (float): Current vehicle altitude
        current_x (float): Current vehicle X position
        current_y (float): Current vehicle Y position
        current_yaw (float): Current vehicle yaw angle
        failsafe_triggered (bool): Flag indicating if failsafe has been triggered
    """

    def __init__(self):
        """
        Initialize the CoordinateReceiver node.

        Args:
            bias_x (float): X-axis bias correction value. Defaults to BIAS_X.
            bias_y (float): Y-axis bias correction value. Defaults to BIAS_Y.
            fx_depth (float): Focal length in pixels (X-axis). Defaults to FX_DEPTH.
            fy_depth (float): Focal length in pixels (Y-axis). Defaults to FY_DEPTH.
            cx_depth (float): Principal point X coordinate. Defaults to 639.5.
            cy_depth (float): Principal point Y coordinate. Defaults to 359.5.
            baseline (float): Baseline between RGB and depth cameras. Defaults to 0.025.
        """
        super().__init__('coordinate_receiver')
        
        # QoS Profile Definition
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.subscription = self.create_subscription(
            Float32MultiArray, 
            DETECTED_COORDINATES_TOPIC, 
            self.listener_callback, 
            10)
        
        self.depth_subscription = self.create_subscription(
            Image, 
            DEPTH_IMAGE_TOPIC, 
            self.depth_callback, 
            10)
        
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition, 
            VEHICLE_LOCAL_POSITION_TOPIC, 
            self.vehicle_local_position_callback, 
            qos_profile)
        
        self.delta_publisher = self.create_publisher(
            Point, 
            DELTA_POSITION_TOPIC, 
            10)
        

        self.bridge = CvBridge()

        self.latest_depth = None
        self.failsafe_triggered = False

    def depth_callback(self, msg):
        """
        Process incoming depth image messages.

        Converts ROS Image messages to OpenCV format and stores them for later use
        in coordinate processing.

        Args:
            msg (sensor_msgs.msg.Image): The incoming depth image message

        Note:
            Depth values are stored in millimeters and converted to meters during processing
        """
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth = depth_image.astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Error in depth_callback: {e}")
            traceback.print_exc()
    
    def vehicle_local_position_callback(self, msg):
        """
        Process vehicle position updates and check safety conditions.

        Updates the stored vehicle position and orientation, and checks if the altitude
        exceeds safety thresholds.

        Args:
            msg (px4_msgs.msg.VehicleLocalPosition): Vehicle position message

        Note:
            Triggers failsafe if altitude exceeds -0.13m (NED frame)
        """
        try:
            self.current_altitude = msg.z
            self.current_x = msg.x
            self.current_y = msg.y
            self.current_yaw = msg.heading

            if self.current_altitude > -0.13:
                self.get_logger().error("Altitude exceeds safe threshold! Engaging failsafe.")
                self.failsafe_triggered = True
        except Exception as e:
            self.get_logger().error(f"Error in vehicle_local_position_callback: {e}")
            traceback.print_exc()

    def listener_callback(self, msg):
        """
        Process detected base coordinates and calculate real-world positions.

        This method:
        1. Validates the bounding box dimensions
        2. Calculates the midpoint in image coordinates
        3. Retrieves depth information
        4. Transforms image coordinates to real-world coordinates
        5. Applies necessary corrections and transformations
        6. Publishes the resulting delta position

        Args:
            msg (std_msgs.msg.Float32MultiArray): Array containing bounding box coordinates
                [x1, y1, x2, y2] where (x1,y1) is the top-left corner and (x2,y2) is the
                bottom-right corner of the detected base

        Note:
            Implements various safety checks including:
            - Bounding box aspect ratio validation
            - Depth value validation
            - Coordinate transformation verification
        """
        try:
            # Unpack coordinates
            x1, y1, x2, y2 = msg.data
            self.get_logger().info(f"Received coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Calculate bounding box dimensions
            bbox_width = abs(x2 - x1)
            bbox_height = abs(y2 - y1)

            # Validate bounding box shape
            aspect_ratio = bbox_width / bbox_height if bbox_height != 0 else float('inf')
            tolerance = 0.1
            if abs(aspect_ratio - 1) <= tolerance:

                mid_x_rgb = int((x1 + x2) / 2)
                mid_y_rgb = int((y1 + y2) / 2)
                self.get_logger().info(f"Midpoint of the bounding box (RGB): ({mid_x_rgb}, {mid_y_rgb})")

                if self.latest_depth is not None:
                    mid_x_depth = mid_x_rgb
                    mid_y_depth = mid_y_rgb

                    # Get and validate depth value
                    depth_value = self.latest_depth[mid_y_depth, mid_x_depth]
                    if np.isnan(depth_value) or depth_value <= 0:
                        self.get_logger().warning("Invalid depth value encountered.")
                        return

                    # Convert depth to meters
                    z = depth_value / 1000.0

                    delta_x = (mid_x_depth - CX_DEPTH) * z / FX_DEPTH if FX_DEPTH != 0 else 0
                    delta_y = (mid_y_depth - CY_DEPTH) * z / FY_DEPTH if FY_DEPTH != 0 else 0

                    # Apply corrections
                    delta_x += BASELINE
                    delta_x -= BIAS_X 
                    delta_y -= BIAS_Y 

                    self.get_logger().info(f"Delta real x: {delta_x:.3f} meters, Delta real y: {delta_y:.3f} meters")
                    
                    # Publish results
                    delta_point = Point()
                    delta_point.x = delta_x
                    delta_point.y = delta_y
                    delta_point.z = 0.0
                    self.delta_publisher.publish(delta_point)
                    
                else:
                    self.get_logger().warning("No depth data available.")
            else:
                self.get_logger().info("Bounding box is not approximately square. Ignoring this detection.")
        except ValueError as ve:
            self.get_logger().error(f"Value error in listener_callback: {ve}")
            traceback.print_exc()
        except Exception as e:
            self.get_logger().error(f"Unexpected error in listener_callback: {e}")
            traceback.print_exc()


def main(args=None):
    """
    Main entry point for the coordinate receiver node.

    Args:
        args: Command line arguments (unused)
    """
    rclpy.init(args=args)
    coordinate_receiver = CoordinateReceiver()

    try:
        rclpy.spin(coordinate_receiver)
    except KeyboardInterrupt:
        pass
    finally:
        coordinate_receiver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
