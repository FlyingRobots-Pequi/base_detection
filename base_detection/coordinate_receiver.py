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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from base_detection.variables import (
    DEPTH_IMAGE_TOPIC,
    VEHICLE_LOCAL_POSITION_TOPIC,
    DETECTED_COORDINATES_TOPIC,
    DELTA_POINTS_TOPIC,
    ABSOLUTE_POINTS_TOPIC,
    D455_BASELINE as BASELINE,
    D455_BIAS_X as BIAS_X,
    D455_BIAS_Y as BIAS_Y,
    D455_FX_DEPTH as FX_DEPTH,
    D455_FY_DEPTH as FY_DEPTH,
    D455_CX_DEPTH as CX_DEPTH,
    D455_CY_DEPTH as CY_DEPTH,
    D455_RGB_WIDTH,
    D455_RGB_HEIGHT,
    D455_DEPTH_WIDTH,
    D455_DEPTH_HEIGHT,
    INITIAL_BASE_EXCLUSION_RADIUS,
    INITIAL_BASE_X,
    INITIAL_BASE_Y
)


class CoordinateReceiver(Node):
    """
    A ROS2 node for processing and transforming detected base coordinates into real-world positions.

    This class subscribes to detected coordinates, depth images, and vehicle position data,
    processes this information to calculate real-world 3D positions of detected bases, and
    publishes both relative (delta) and absolute coordinates for navigation purposes.

    The node implements robust coordinate transformation that automatically handles different
    resolutions between RGB and depth cameras by calculating precise scaling factors,
    eliminating the need for coordinate clamping and improving position accuracy.

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
        rgb_width (int): RGB image width for coordinate transformation
        rgb_height (int): RGB image height for coordinate transformation
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
        super().__init__("coordinate_receiver")

        # QoS Profile Definition
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
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
            qos_profile)
        
        self.delta_publisher = self.create_publisher(
            Point, 
            DELTA_POINTS_TOPIC, 
            10)
        
        self.absolute_publisher = self.create_publisher(
            Point, 
            ABSOLUTE_POINTS_TOPIC, 
            10)
        

        self.bridge = CvBridge()

        self.latest_depth = None
        self.failsafe_triggered = False
        
        # Initialize vehicle position variables
        self.current_altitude = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # Initialize RGB image dimensions (will be detected automatically)
        self.rgb_width = None
        self.rgb_height = None

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
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.latest_depth = depth_image.astype(np.float32)

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
        except Exception as e:
            self.get_logger().error(f"Error in vehicle_local_position_callback: {e}")
            traceback.print_exc()

    def listener_callback(self, msg):
        """
        Process detected base coordinates and calculate real-world positions.

        This method now handles batched detections from a single frame:
        1. Parses batched detection data (groups of 5: x1,y1,x2,y2,score)
        2. Processes all detections with the same timestamp
        3. Calculates real-world positions for each detection
        4. Publishes all positions from the frame

        Args:
            msg (std_msgs.msg.Float32MultiArray): Array containing batched bounding box coordinates
                Format: [x1,y1,x2,y2,score, x1,y1,x2,y2,score, ...] for multiple detections

        Note:
            Batched processing ensures all detections from the same frame are processed
            simultaneously, eliminating temporal drift in clustering.
        """
        try:
            # Parse batched detections (groups of 5: x1,y1,x2,y2,score)
            if len(msg.data) % 5 != 0:
                self.get_logger().warning(f"Invalid detection data length: {len(msg.data)} (should be multiple of 5)")
                return
                
            num_detections = len(msg.data) // 5
            self.get_logger().info(f"Processing {num_detections} detections from single frame")
            
            # Store all positions from this frame with same timestamp
            frame_positions = []
            current_time = self.get_clock().now()
            
            for i in range(num_detections):
                idx = i * 5
                x1, y1, x2, y2, score = msg.data[idx:idx+5]
                
                self.get_logger().info(f"Detection {i+1}: p1=({x1:.3f}, {y1:.3f}), p2=({x2:.3f}, {y2:.3f}), score={score:.3f}")

                # Calculate bounding box dimensions
                bbox_width = abs(x2 - x1)
                bbox_height = abs(y2 - y1)

                # Validate bounding box shape
                aspect_ratio = (
                    bbox_width / bbox_height if bbox_height != 0 else float("inf")
                )
                tolerance = 0.15
                if abs(aspect_ratio - 1) > tolerance:
                    self.get_logger().info(f"Detection {i+1}: Bounding box is not approximately square. Ignoring this detection.")
                    continue

                # cálculo do centro da bounding‐box
                mid_x_rgb = int((x1 + x2) / 2)
                mid_y_rgb = int((y1 + y2) / 2)
                self.get_logger().info(f"Detection {i+1}: Midpoint (RGB): ({mid_x_rgb:.3f}, {mid_y_rgb:.3f})")

                # busca depth_value, usando valor padrão se latest_depth for None ou inválido
                if self.latest_depth is not None:
                    # Get depth image dimensions
                    depth_height, depth_width = self.latest_depth.shape
                    
                    # Auto-detect RGB image dimensions from coordinates if not set
                    if self.rgb_width is None or self.rgb_height is None:
                        # Estimate RGB dimensions based on coordinate ranges and common camera resolutions
                        max_coord_x = max(x1, x2)
                        max_coord_y = max(y1, y2)
                        
                        # Use standard D455 RGB resolution as default
                        self.rgb_width = D455_RGB_WIDTH
                        self.rgb_height = D455_RGB_HEIGHT
                        
                        # Override if coordinates suggest different resolution
                        if max_coord_x > 1920:
                            self.get_logger().warning(f"Coordinates exceed standard RGB width. Using coordinate-based estimation.")
                            self.rgb_width = max(int(max_coord_x * 1.1), D455_RGB_WIDTH)
                        if max_coord_y > 1080:
                            self.get_logger().warning(f"Coordinates exceed standard RGB height. Using coordinate-based estimation.")
                            self.rgb_height = max(int(max_coord_y * 1.1), D455_RGB_HEIGHT)
                    
                    # Calculate robust scaling factors for coordinate transformation
                    scale_x = depth_width / self.rgb_width
                    scale_y = depth_height / self.rgb_height
                    
                    # Transform RGB coordinates to depth coordinate space
                    mid_x_depth = int(mid_x_rgb * scale_x)
                    mid_y_depth = int(mid_y_rgb * scale_y)
                    
                    self.get_logger().info(f"Detection {i+1}: Transformed coordinates: RGB({mid_x_rgb}, {mid_y_rgb}) -> Depth({mid_x_depth}, {mid_y_depth})")
                    
                    # Validate transformed coordinates are within depth image bounds
                    if (mid_x_depth < 0 or mid_x_depth >= depth_width or 
                        mid_y_depth < 0 or mid_y_depth >= depth_height):
                        self.get_logger().error(f"Detection {i+1}: Transformed coordinates ({mid_x_depth}, {mid_y_depth}) are outside depth image bounds ({depth_width}x{depth_height}).")
                        continue

                    # Get and validate depth value
                    depth_value = self.latest_depth[mid_y_depth, mid_x_depth]
                    if np.isnan(depth_value) or depth_value <= 0:
                        self.get_logger().warning(f"Detection {i+1}: Invalid depth value at ({mid_x_depth:.3f}, {mid_y_depth:.3f}): {depth_value:.3f}")
                        continue

                    # Convert depth to meters
                    z = depth_value / 1000.0
                    self.get_logger().info(f"Detection {i+1}: Depth at center: {z:.3f} meters")

                    # Calculate the real altitude of the base relative to the ground
                    base_altitude_relative_to_drone = -z  # Convert camera distance to altitude offset
                    base_ground_altitude = self.current_altitude + base_altitude_relative_to_drone
                    
                    self.get_logger().info(f"Detection {i+1}: Base altitude calculation: drone_alt={self.current_altitude:.3f}m, depth={z:.3f}m, base_ground_alt={base_ground_altitude:.3f}m")

                    delta_x = (mid_x_depth - CX_DEPTH) * z / FX_DEPTH if FX_DEPTH != 0 else 0
                    delta_y = (mid_y_depth - CY_DEPTH) * z / FY_DEPTH if FY_DEPTH != 0 else 0

                    # Apply corrections
                    delta_x += BASELINE
                    delta_x -= BIAS_X 
                    delta_y -= BIAS_Y 

                    self.get_logger().info(f"Detection {i+1}: Delta real (x,y): ({delta_x:.3f}, {delta_y:.3f}) meters")
                    
                    # Calculate absolute coordinates
                    cos_yaw = np.cos(self.current_yaw)
                    sin_yaw = np.sin(self.current_yaw)
                    
                    # Transform relative coordinates to global frame
                    global_x = self.current_x + (delta_x * cos_yaw - delta_y * sin_yaw)
                    global_y = self.current_y + (delta_x * sin_yaw + delta_y * cos_yaw)
                    
                    self.get_logger().info(f"Detection {i+1}: Absolute position (x,y): ({global_x:.3f}, {global_y:.3f}) meters")
                    
                    # Store this detection's position data
                    frame_positions.append({
                        'delta': (delta_x, delta_y, base_ground_altitude),
                        'absolute': (global_x, global_y, base_ground_altitude),
                        'score': score,
                        'near_initial': self.is_near_initial_base(global_x, global_y)
                    })
                    
                else:
                    self.get_logger().warning(f"Detection {i+1}: No depth data available. Using default for simulation.")
                    depth_value = np.float32(-0.4)
                    z = depth_value / 1000.0
                    
                    # Calculate the real altitude of the base (fallback case)
                    base_altitude_relative_to_drone = -z
                    base_ground_altitude = self.current_altitude + base_altitude_relative_to_drone

                    # cálculo de delta_x e delta_y
                    delta_x = ((mid_x_rgb - CX_DEPTH) * z / FX_DEPTH) if FX_DEPTH != 0 else 0
                    delta_y = ((mid_y_rgb - CY_DEPTH) * z / FY_DEPTH) if FY_DEPTH != 0 else 0

                    # aplica correções de baseline e bias
                    delta_x += BASELINE - BIAS_X
                    delta_y -= BIAS_Y

                    self.get_logger().info(f"Detection {i+1}: Delta real x,y (fallback): ({delta_x:.3f}, {delta_y:.3f}) meters")
                    
                    # Calculate absolute coordinates (fallback)
                    cos_yaw = np.cos(self.current_yaw)
                    sin_yaw = np.sin(self.current_yaw)
                    global_x = self.current_x + (delta_x * cos_yaw - delta_y * sin_yaw)
                    global_y = self.current_y + (delta_x * sin_yaw + delta_y * cos_yaw)
                    
                    # Store this detection's position data (fallback)
                    frame_positions.append({
                        'delta': (delta_x, delta_y, base_ground_altitude),
                        'absolute': (global_x, global_y, base_ground_altitude),
                        'score': score,
                        'near_initial': self.is_near_initial_base(global_x, global_y)
                    })

            # Publish all positions from this frame
            remote_bases_count = 0
            for pos_data in frame_positions:
                delta_x, delta_y, base_ground_altitude = pos_data['delta']
                global_x, global_y, _ = pos_data['absolute']
                
                # Always publish relative coordinates for debugging
                delta_point = Point()
                delta_point.x = delta_x
                delta_point.y = delta_y
                delta_point.z = base_ground_altitude
                self.delta_publisher.publish(delta_point)
                
                # Only publish absolute coordinates for remote bases
                if not pos_data['near_initial']:
                    absolute_point = Point()
                    absolute_point.x = global_x
                    absolute_point.y = global_y
                    absolute_point.z = base_ground_altitude
                    self.absolute_publisher.publish(absolute_point)
                    remote_bases_count += 1
                else:
                    distance_to_origin = np.sqrt(global_x**2 + global_y**2)
                    self.get_logger().info(f"Base detected near initial position (distance: {distance_to_origin:.3f}m). Filtering out from absolute positions.")
            
            self.get_logger().info(f"Published {len(frame_positions)} delta positions and {remote_bases_count} remote absolute positions from frame")
                    
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


if __name__ == "__main__":
    main()
