#!/usr/bin/env python3
"""
Coordinate Receiver Node for Base Detection System with Motion Compensation.

This module implements a ROS2 node that receives and processes base detection coordinates
along with depth information from a D435i camera. It calculates real-world 3D positions
of detected bases and publishes delta positions for navigation.

The node handles:
- Processing of detected base coordinates
- Depth image processing
- Vehicle position tracking with motion compensation
- 3D position calculation with camera parameters
- Safety checks and failsafe triggers
- Motion compensation for improved precision during vehicle movement

Motion Compensation Features:
- Compensates for vehicle movement during processing delays
- Adaptive frame timeout based on vehicle velocity
- Motion-aware outlier detection
- Velocity-based detection confidence adjustment

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
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from px4_msgs.msg import VehicleLocalPosition
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import csv
import os
from datetime import datetime
import time
from base_detection.variables import (
    DETECTED_COORDINATES_TOPIC,
    DELTA_POINTS_TOPIC,
    ABSOLUTE_POINTS_TOPIC,
    DEPTH_IMAGE_TOPIC,
    VEHICLE_LOCAL_POSITION_TOPIC,
    D455_BIAS_X,
    D455_BIAS_Y,
    D455_FX_DEPTH,
    D455_FY_DEPTH,
    D455_CX_DEPTH,
    D455_CY_DEPTH,
    D455_BASELINE,
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
    A ROS2 node for processing and transforming detected base coordinates into real-world positions
    with enhanced motion compensation capabilities.

    This class subscribes to detected coordinates, depth images, and vehicle position data,
    processes this information to calculate real-world 3D positions of detected bases, and
    publishes both relative (delta) and absolute coordinates for navigation purposes.

    The node implements robust coordinate transformation that automatically handles different
    resolutions between RGB and depth cameras by calculating precise scaling factors,
    eliminating the need for coordinate clamping and improving position accuracy.

    Motion Compensation Features:
    - Compensates for vehicle movement during processing delays using kinematic equations
    - Implements adaptive frame timeout based on vehicle velocity magnitude  
    - Provides motion-aware outlier detection considering vehicle trajectory
    - Adjusts detection confidence based on motion stability factors
    - Uses predictive position estimation with velocity/acceleration data

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
        current_vx (float): Current vehicle X velocity
        current_vy (float): Current vehicle Y velocity
        current_vz (float): Current vehicle Z velocity
        current_ax (float): Current vehicle X acceleration
        current_ay (float): Current vehicle Y acceleration
        current_az (float): Current vehicle Z acceleration
        vehicle_timestamp (float): Timestamp of last vehicle position update
        failsafe_triggered (bool): Flag indicating if failsafe has been triggered
        rgb_width (int): RGB image width for coordinate transformation
        rgb_height (int): RGB image height for coordinate transformation
        motion_compensation_enabled (bool): Enable/disable motion compensation
        adaptive_timeout_enabled (bool): Enable/disable adaptive frame timeout
        motion_stability_threshold (float): Velocity threshold for motion stability assessment
        processing_delay_estimate (float): Estimated processing delay for motion compensation
    """

    def __init__(self):
        """
        Initialize the CoordinateReceiver node with motion compensation capabilities.

        Sets up subscriptions, publishers, and motion compensation parameters for enhanced
        precision during vehicle movement.
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
        
        # Motion compensation: Add velocity and acceleration tracking
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vz = 0.0
        self.current_ax = 0.0
        self.current_ay = 0.0
        self.current_az = 0.0
        self.vehicle_timestamp = 0.0
        
        # Initialize RGB image dimensions (will be detected automatically)
        self.rgb_width = None
        self.rgb_height = None
        
        # Motion compensation configuration
        self.motion_compensation_enabled = True
        self.adaptive_timeout_enabled = True
        self.motion_stability_threshold = 0.5  # m/s - threshold for considering motion "stable"
        self.processing_delay_estimate = 0.1   # seconds - estimated processing delay
        
        # Motion compensation parameters for adaptive behavior
        self.base_frame_timeout = 0.2          # Base timeout for hovering (200ms)
        self.min_frame_timeout = 0.05          # Minimum timeout for high-speed (50ms)
        self.max_frame_timeout = 0.3           # Maximum timeout for very stable hovering (300ms)
        self.velocity_scale_factor = 0.1       # How much velocity affects timeout
        
        # Motion-aware detection parameters
        self.motion_confidence_factor = 1.0    # Confidence adjustment based on motion
        self.velocity_outlier_threshold = 2.0  # m/s - velocity above which to apply stricter outlier detection
        
        # Motion-aware outlier detection parameters
        self.outlier_velocity_threshold = 2.0  # m/s - above this velocity, use relaxed outlier detection
        self.outlier_distance_threshold = 5.0  # meters - max reasonable detection distance from drone
        
        # CSV Logging System for detection points
        self.setup_csv_logging()
        
        self.get_logger().info("CoordinateReceiver node initialized with enhanced motion compensation and CSV logging")

    def setup_csv_logging(self):
        """
        Setup CSV logging system for saving detection points at different processing stages.
        
        Creates CSV files with timestamps and proper headers for tracking:
        - Raw camera coordinates (pixel coordinates)
        - Delta coordinates (relative to camera in meters)
        - Motion-compensated coordinates (with vehicle motion)
        - Absolute global coordinates (world frame)
        """
        try:
            # Create logs directory if it doesn't exist
            self.csv_logs_dir = "/root/ros2_ws/detection_logs"
            os.makedirs(self.csv_logs_dir, exist_ok=True)
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Define CSV file paths
            self.csv_raw_file = os.path.join(self.csv_logs_dir, f"raw_detections_{timestamp}.csv")
            self.csv_processed_file = os.path.join(self.csv_logs_dir, f"processed_detections_{timestamp}.csv")
            
            # CSV Headers
            raw_headers = [
                'timestamp', 'detection_id', 'frame_id',
                'camera_x_px', 'camera_y_px', 'depth_value_mm',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence_score',
                'reference_type'  # 'RAW_CAMERA_DETECTION'
            ]
            
            processed_headers = [
                'timestamp', 'detection_id', 'frame_id',
                'delta_x_m', 'delta_y_m', 'delta_z_m',
                'vehicle_x_m', 'vehicle_y_m', 'vehicle_z_m', 'vehicle_yaw_rad',
                'vehicle_vx_ms', 'vehicle_vy_ms', 'vehicle_vz_ms',
                'compensated_x_m', 'compensated_y_m', 'compensated_z_m', 'compensated_yaw_rad',
                'global_x_m', 'global_y_m', 'global_z_m',
                'motion_compensated', 'near_initial_base', 'is_outlier',
                'reference_type'  # 'DELTA_COORDS', 'MOTION_COMPENSATED', 'ABSOLUTE_GLOBAL'
            ]
            
            # Create CSV files with headers
            with open(self.csv_raw_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(raw_headers)
                
            with open(self.csv_processed_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(processed_headers)
                
            self.get_logger().info(f"CSV logging initialized:")
            self.get_logger().info(f"  Raw detections: {self.csv_raw_file}")
            self.get_logger().info(f"  Processed coords: {self.csv_processed_file}")
            
            # Initialize CSV counters
            self.csv_detection_counter = 0
            self.csv_frame_counter = 0
            
        except Exception as e:
            self.get_logger().error(f"Failed to setup CSV logging: {e}")
            self.csv_raw_file = None
            self.csv_processed_file = None

    def log_raw_detection_to_csv(self, camera_x, camera_y, depth_mm, bbox, confidence, reference_type="RAW_CAMERA_DETECTION"):
        """
        Log raw camera detection data to CSV.
        
        Args:
            camera_x, camera_y: Pixel coordinates in camera frame
            depth_mm: Depth value in millimeters
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence score
            reference_type: Type of coordinate reference
        """
        if self.csv_raw_file is None:
            return
            
        try:
            self.csv_detection_counter += 1
            timestamp = datetime.now().isoformat()
            
            row = [
                timestamp, self.csv_detection_counter, self.csv_frame_counter,
                camera_x, camera_y, depth_mm,
                bbox[0], bbox[1], bbox[2], bbox[3], confidence,
                reference_type
            ]
            
            with open(self.csv_raw_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
        except Exception as e:
            self.get_logger().error(f"Failed to log raw detection to CSV: {e}")

    def log_processed_coordinates_to_csv(self, delta_coords, vehicle_state, compensated_state, 
                                       global_coords, flags, reference_type):
        """
        Log processed coordinate data to CSV.
        
        Args:
            delta_coords: (delta_x, delta_y, delta_z) in meters
            vehicle_state: (x, y, z, yaw, vx, vy, vz) vehicle state
            compensated_state: (x, y, z, yaw) motion-compensated state
            global_coords: (global_x, global_y, global_z) in world frame
            flags: Dict with 'motion_compensated', 'near_initial_base', 'is_outlier'
            reference_type: Type of coordinate reference
        """
        if self.csv_processed_file is None:
            return
            
        try:
            timestamp = datetime.now().isoformat()
            
            row = [
                timestamp, self.csv_detection_counter, self.csv_frame_counter,
                delta_coords[0], delta_coords[1], delta_coords[2],
                vehicle_state[0], vehicle_state[1], vehicle_state[2], vehicle_state[3],
                vehicle_state[4], vehicle_state[5], vehicle_state[6],
                compensated_state[0], compensated_state[1], compensated_state[2], compensated_state[3],
                global_coords[0], global_coords[1], global_coords[2],
                flags.get('motion_compensated', False),
                flags.get('near_initial_base', False),
                flags.get('is_outlier', False),
                reference_type
            ]
            
            with open(self.csv_processed_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
        except Exception as e:
            self.get_logger().error(f"Failed to log processed coordinates to CSV: {e}")

    def calculate_motion_compensation(self, detection_timestamp):
        """
        Calculate motion compensation for vehicle movement during processing delay.
        
        Uses kinematic equations to predict where the vehicle was at detection time
        based on current position, velocity, and acceleration.
        
        Args:
            detection_timestamp (float): Timestamp when detection occurred
            
        Returns:
            tuple: (compensated_x, compensated_y, compensated_z, compensated_yaw)
        """
        if not self.motion_compensation_enabled:
            return self.current_x, self.current_y, self.current_altitude, self.current_yaw
            
        # Calculate time delta (processing delay)
        time_delta = detection_timestamp - self.vehicle_timestamp
        
        # If time delta is too large or negative, don't apply compensation
        if abs(time_delta) > 1.0 or time_delta < 0:
            self.get_logger().debug(f"⚠️ Large time delta ({time_delta:.3f}s), skipping motion compensation")
            return self.current_x, self.current_y, self.current_altitude, self.current_yaw
        
        # Apply kinematic compensation: x = x0 + v*t + 0.5*a*t²
        compensated_x = self.current_x - (self.current_vx * time_delta + 0.5 * self.current_ax * time_delta**2)
        compensated_y = self.current_y - (self.current_vy * time_delta + 0.5 * self.current_ay * time_delta**2)
        compensated_z = self.current_altitude - (self.current_vz * time_delta + 0.5 * self.current_az * time_delta**2)
        
        # For yaw, we don't have angular velocity/acceleration, so use current value
        compensated_yaw = self.current_yaw
        
        # Calculate motion magnitude for logging
        velocity_magnitude = np.sqrt(self.current_vx**2 + self.current_vy**2)
        acceleration_magnitude = np.sqrt(self.current_ax**2 + self.current_ay**2)
        position_compensation = np.sqrt((self.current_x - compensated_x)**2 + (self.current_y - compensated_y)**2)
        
        if position_compensation > 0.01:  # Only log significant compensations
            self.get_logger().info(f"🔧 Motion Compensation Applied:")
            self.get_logger().info(f"  ⏱️ Time delta: {time_delta:.3f}s")
            self.get_logger().info(f"  🏃 Velocity: {velocity_magnitude:.3f}m/s, Accel: {acceleration_magnitude:.3f}m/s²")
            self.get_logger().info(f"  📍 Position compensation: {position_compensation:.3f}m")
        
        return compensated_x, compensated_y, compensated_z, compensated_yaw

    def calculate_adaptive_frame_timeout(self):
        """
        Calculate adaptive frame timeout based on vehicle velocity.
        
        Higher velocity = shorter timeout for more responsive detection.
        Lower velocity = longer timeout for better clustering of stable detections.
        
        Returns:
            float: Adaptive timeout value in seconds
        """
        if not self.adaptive_timeout_enabled:
            return self.base_frame_timeout
            
        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(self.current_vx**2 + self.current_vy**2 + self.current_vz**2)
        
        # Calculate adaptive timeout: faster = shorter timeout
        # timeout = base_timeout * (1 - velocity_scale_factor * velocity_magnitude)
        adaptive_timeout = self.base_frame_timeout * (1.0 - self.velocity_scale_factor * velocity_magnitude)
        
        # Clamp to min/max bounds
        adaptive_timeout = max(self.min_frame_timeout, min(self.max_frame_timeout, adaptive_timeout))
        
        self.get_logger().debug(f"🕒 Adaptive timeout: {adaptive_timeout:.3f}s (velocity: {velocity_magnitude:.3f}m/s)")
        
        return adaptive_timeout

    def assess_motion_stability(self):
        """
        Assess vehicle motion stability for detection confidence adjustment.
        
        Returns:
            tuple: (stability_factor, is_stable)
                - stability_factor: 0.0-1.0, where 1.0 is perfectly stable
                - is_stable: bool indicating if motion is below stability threshold
        """
        velocity_magnitude = np.sqrt(self.current_vx**2 + self.current_vy**2)
        acceleration_magnitude = np.sqrt(self.current_ax**2 + self.current_ay**2)
        
        # Calculate stability based on velocity and acceleration
        # Lower velocity and acceleration = higher stability
        velocity_stability = max(0.0, 1.0 - velocity_magnitude / (self.motion_stability_threshold * 2))
        acceleration_stability = max(0.0, 1.0 - acceleration_magnitude / 1.0)  # 1.0 m/s² as reference
        
        # Combined stability factor (weighted average)
        stability_factor = 0.7 * velocity_stability + 0.3 * acceleration_stability
        is_stable = velocity_magnitude < self.motion_stability_threshold
        
        return stability_factor, is_stable

    def calculate_motion_aware_confidence(self, base_score):
        """
        Adjust detection confidence based on motion stability.
        
        Args:
            base_score (float): Original detection confidence score
            
        Returns:
            float: Motion-adjusted confidence score
        """
        stability_factor, is_stable = self.assess_motion_stability()
        
        # Adjust confidence: stable motion = higher confidence, unstable = lower
        adjusted_score = base_score * (0.8 + 0.2 * stability_factor)
        
        velocity_magnitude = np.sqrt(self.current_vx**2 + self.current_vy**2)
        
        if not is_stable:
            self.get_logger().debug(f"⚡ Motion instability detected: vel={velocity_magnitude:.3f}m/s, "
                                   f"confidence: {base_score:.3f} → {adjusted_score:.3f}")
        
        return adjusted_score

    def is_motion_outlier(self, position, expected_trajectory=None):
        """
        Determine if a detected position is a motion-based outlier.
        
        Args:
            position (tuple): Detected position (x, y)
            expected_trajectory (tuple, optional): Expected position based on vehicle motion
            
        Returns:
            bool: True if position is likely a motion outlier
        """
        if expected_trajectory is None:
            return False
            
        # Calculate distance from expected trajectory
        distance_from_expected = np.sqrt((position[0] - expected_trajectory[0])**2 + 
                                        (position[1] - expected_trajectory[1])**2)
        
        # Dynamic threshold based on velocity
        velocity_magnitude = np.sqrt(self.current_vx**2 + self.current_vy**2)
        
        if velocity_magnitude > self.velocity_outlier_threshold:
            # Higher velocity = more strict outlier detection
            outlier_threshold = 0.5  # 50cm threshold during high-speed movement
        else:
            # Lower velocity = more lenient outlier detection
            outlier_threshold = 1.0  # 1m threshold during stable movement
            
        is_outlier = distance_from_expected > outlier_threshold
        
        if is_outlier:
            self.get_logger().warning(f"🚫 Motion outlier detected: distance from expected = {distance_from_expected:.3f}m "
                                     f"(threshold: {outlier_threshold:.3f}m, velocity: {velocity_magnitude:.3f}m/s)")
        
        return is_outlier

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
        Process vehicle position updates with enhanced motion data tracking.

        Updates the stored vehicle position, orientation, velocity, and acceleration data
        for motion compensation calculations.

        Args:
            msg (px4_msgs.msg.VehicleLocalPosition): Vehicle position message with motion data

        Note:
            Triggers failsafe if altitude exceeds -0.13m (NED frame)
            Now also tracks velocity and acceleration for motion compensation
        """
        try:
            # Position and orientation
            self.current_altitude = msg.z
            self.current_x = msg.x
            self.current_y = msg.y
            self.current_yaw = msg.heading
            
            # Motion compensation: Track velocity and acceleration
            self.current_vx = msg.vx
            self.current_vy = msg.vy
            self.current_vz = msg.vz
            self.current_ax = msg.ax
            self.current_ay = msg.ay
            self.current_az = msg.az
            
            # Update timestamp for motion compensation calculations
            self.vehicle_timestamp = time.time()
            
            # Log motion data periodically for debugging
            velocity_magnitude = np.sqrt(self.current_vx**2 + self.current_vy**2 + self.current_vz**2)
            if velocity_magnitude > 0.1:  # Only log when moving
                self.get_logger().debug(f"🚁 Vehicle motion: vel={velocity_magnitude:.3f}m/s, "
                                       f"pos=({self.current_x:.2f}, {self.current_y:.2f}, {self.current_altitude:.2f})")
            
        except Exception as e:
            self.get_logger().error(f"Error in vehicle_local_position_callback: {e}")
            traceback.print_exc()

    def listener_callback(self, msg):
        """
        Process detected base coordinates with enhanced motion compensation.

        This method now handles batched detections from a single frame with motion compensation:
        1. Parses batched detection data (groups of 5: x1,y1,x2,y2,score)
        2. Applies motion compensation for vehicle movement during processing
        3. Calculates real-world positions for each detection with motion-aware confidence
        4. Publishes all positions from the frame with enhanced precision

        Args:
            msg (std_msgs.msg.Float32MultiArray): Array containing batched bounding box coordinates
                Format: [x1,y1,x2,y2,score, x1,y1,x2,y2,score, ...] for multiple detections

        Note:
            Enhanced with motion compensation to improve precision during vehicle movement.
            Includes adaptive frame timeout and motion-aware outlier detection.
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
            detection_timestamp = time.time()
            
            # Calculate motion compensation for this detection frame
            compensated_x, compensated_y, compensated_z, compensated_yaw = self.calculate_motion_compensation(detection_timestamp)
            
            # Assess motion stability for confidence adjustment
            stability_factor, is_stable = self.assess_motion_stability()
            motion_status = "🟢 Stable" if is_stable else "🟡 Dynamic"
            self.get_logger().info(f"🚁 Motion status: {motion_status} (stability: {stability_factor:.2f})")
            
            # Process frame detections and save them as a batch
            frame_positions = []
            
            # Increment frame counter for CSV logging
            self.csv_frame_counter += 1
            
            # Parse detections from message data
            detections = []
            for i in range(num_detections):
                idx = i * 5
                x1, y1, x2, y2, score = msg.data[idx:idx+5]
                detections.append([x1, y1, x2, y2, score])
            
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, score = detection

                # Calculate motion-aware confidence adjustment
                adjusted_score = self.calculate_motion_aware_confidence(score)

                # Calculate center coordinates
                mid_x_rgb = (x1 + x2) / 2
                mid_y_rgb = (y1 + y2) / 2

                # Log raw detection to CSV
                bbox = [x1, y1, x2, y2]
                self.log_raw_detection_to_csv(mid_x_rgb, mid_y_rgb, 0, bbox, score, "RAW_CAMERA_DETECTION")

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

                    # Update raw detection log with actual depth value
                    self.log_raw_detection_to_csv(mid_x_depth, mid_y_depth, depth_value, bbox, score, "RAW_CAMERA_DETECTION_WITH_DEPTH")

                    # Convert depth to meters
                    z = depth_value / 1000.0
                    self.get_logger().info(f"Detection {i+1}: Depth at center: {z:.3f} meters")

                    # Calculate the real altitude of the base relative to the ground
                    # Use motion-compensated altitude for more accurate calculation
                    base_altitude_relative_to_drone = -z  # Convert camera distance to altitude offset
                    base_ground_altitude = compensated_z + base_altitude_relative_to_drone
                    
                    self.get_logger().info(f"Detection {i+1}: Base altitude calculation: "
                                          f"compensated_alt={compensated_z:.3f}m, depth={z:.3f}m, "
                                          f"base_ground_alt={base_ground_altitude:.3f}m")

                    delta_x = (mid_x_depth - D455_CX_DEPTH) * z / D455_FX_DEPTH if D455_FX_DEPTH != 0 else 0
                    delta_y = (mid_y_depth - D455_CY_DEPTH) * z / D455_FY_DEPTH if D455_FY_DEPTH != 0 else 0

                    # Apply corrections
                    delta_x += D455_BASELINE
                    delta_x -= D455_BIAS_X 
                    delta_y -= D455_BIAS_Y 

                    self.get_logger().info(f"Detection {i+1}: Delta real (x,y): ({delta_x:.3f}, {delta_y:.3f}) meters")
                    
                    # Log delta coordinates to CSV
                    delta_coords = (delta_x, delta_y, base_ground_altitude)
                    vehicle_state = (self.current_x, self.current_y, self.current_altitude, self.current_yaw,
                                   self.current_vx, self.current_vy, self.current_vz)
                    compensated_state = (compensated_x, compensated_y, compensated_z, compensated_yaw)
                    temp_global_coords = (0, 0, base_ground_altitude)  # Will be updated below
                    flags = {'motion_compensated': True, 'near_initial_base': False, 'is_outlier': False}
                    self.log_processed_coordinates_to_csv(delta_coords, vehicle_state, compensated_state, 
                                                        temp_global_coords, flags, "DELTA_COORDS")

                    # Calculate absolute coordinates using motion-compensated position
                    cos_yaw = np.cos(compensated_yaw)
                    sin_yaw = np.sin(compensated_yaw)
                    
                    # CORRECTED: Transform relative coordinates to global frame with motion compensation
                    # delta_x and delta_y represent the offset from camera center to base in camera frame
                    # We need to transform this to world frame and add to drone position to get base position
                    base_offset_world_x = (delta_x * cos_yaw - delta_y * sin_yaw)
                    base_offset_world_y = (delta_x * sin_yaw + delta_y * cos_yaw)
                    
                    # Calculate absolute base position in world frame
                    global_x = compensated_x + base_offset_world_x
                    global_y = compensated_y + base_offset_world_y
                    
                    # Motion-aware outlier detection (optional)
                    expected_global_pos = (compensated_x, compensated_y)
                    is_outlier = self.is_motion_outlier((global_x, global_y), expected_global_pos)
                    if is_outlier:
                        self.get_logger().warning(f"Detection {i+1}: Flagged as motion outlier, but processing anyway")
                    
                    self.get_logger().info(f"Detection {i+1}: Motion-compensated absolute position: "
                                          f"({global_x:.3f}, {global_y:.3f}) meters")
                    
                    # Check if near initial base
                    near_initial = self.is_near_initial_base(global_x, global_y)
                    
                    # Log motion-compensated absolute coordinates to CSV
                    global_coords = (global_x, global_y, base_ground_altitude)
                    flags = {
                        'motion_compensated': True, 
                        'near_initial_base': near_initial, 
                        'is_outlier': is_outlier
                    }
                    self.log_processed_coordinates_to_csv(delta_coords, vehicle_state, compensated_state, 
                                                        global_coords, flags, "ABSOLUTE_GLOBAL")
                    
                    # ENHANCED DEBUGGING: Comprehensive coordinate transformation analysis
                    self.get_logger().info(f"Detection {i+1}: === COORDINATE TRANSFORMATION DEBUGGING ===")
                    self.get_logger().info(f"  Camera coordinates (pixels): ({mid_x_depth}, {mid_y_depth})")
                    self.get_logger().info(f"  Camera intrinsics: fx={D455_FX_DEPTH:.1f}, fy={D455_FY_DEPTH:.1f}, cx={D455_CX_DEPTH:.1f}, cy={D455_CY_DEPTH:.1f}")
                    self.get_logger().info(f"  Depth value: {z:.3f} meters")
                    self.get_logger().info(f"  Raw delta (camera frame): ({(mid_x_depth - D455_CX_DEPTH) * z / D455_FX_DEPTH:.3f}, {(mid_y_depth - D455_CY_DEPTH) * z / D455_FY_DEPTH:.3f})")
                    self.get_logger().info(f"  Corrected delta (with bias/baseline): ({delta_x:.3f}, {delta_y:.3f})")
                    self.get_logger().info(f"  Drone position (compensated): ({compensated_x:.3f}, {compensated_y:.3f}) yaw={compensated_yaw:.3f}rad={np.degrees(compensated_yaw):.1f}°")
                    
                    # TEST: Calculate multiple coordinate interpretations
                    
                    # Standard interpretation (current)
                    standard_x = compensated_x + (delta_x * cos_yaw - delta_y * sin_yaw)
                    standard_y = compensated_y + (delta_x * sin_yaw + delta_y * cos_yaw)
                    
                    # Alternative 1: Inverted X direction
                    alt1_x = compensated_x + (-delta_x * cos_yaw - delta_y * sin_yaw)
                    alt1_y = compensated_y + (-delta_x * sin_yaw + delta_y * cos_yaw)
                    
                    # Alternative 2: Inverted Y direction
                    alt2_x = compensated_x + (delta_x * cos_yaw + delta_y * sin_yaw)
                    alt2_y = compensated_y + (delta_x * sin_yaw - delta_y * cos_yaw)
                    
                    # Alternative 3: Both X and Y inverted
                    alt3_x = compensated_x + (-delta_x * cos_yaw + delta_y * sin_yaw)
                    alt3_y = compensated_y + (-delta_x * sin_yaw - delta_y * cos_yaw)
                    
                    # Alternative 4: No rotation (sanity check)
                    alt4_x = compensated_x + delta_x
                    alt4_y = compensated_y + delta_y
                    
                    self.get_logger().info(f"  === COORDINATE ALTERNATIVES ===")
                    self.get_logger().info(f"  Standard (current):  ({standard_x:.3f}, {standard_y:.3f})")
                    self.get_logger().info(f"  Alt1 (-X):          ({alt1_x:.3f}, {alt1_y:.3f})")
                    self.get_logger().info(f"  Alt2 (-Y):          ({alt2_x:.3f}, {alt2_y:.3f})")
                    self.get_logger().info(f"  Alt3 (-X,-Y):       ({alt3_x:.3f}, {alt3_y:.3f})")
                    self.get_logger().info(f"  Alt4 (no rotation): ({alt4_x:.3f}, {alt4_y:.3f})")
                    
                    # Distance from drone to calculated base position
                    dist_to_base = np.sqrt(delta_x**2 + delta_y**2)
                    self.get_logger().info(f"  Distance drone->base: {dist_to_base:.3f}m")
                    
                    # Use standard interpretation for now
                    global_x = standard_x
                    global_y = standard_y
                    
                    # GROUND TRUTH COMPARISON for debugging
                    ground_truth_bases = [
                        (-0.24, -3.23, 0.0),  # BASE_1
                        (0.75, -5.05, 0.0),   # BASE_2
                        (5.16, -5.75, 0.0),   # BASE_3
                        (4.37, -2.30, 0.0),   # BASE_4
                        (5.69, -0.25, 0.0),   # BASE_5
                    ]
                    
                    # Find closest ground truth base for each interpretation
                    interpretations = [
                        ("Standard", standard_x, standard_y),
                        ("Alt1(-X)", alt1_x, alt1_y),
                        ("Alt2(-Y)", alt2_x, alt2_y),
                        ("Alt3(-X,-Y)", alt3_x, alt3_y),
                        ("Alt4(NoRot)", alt4_x, alt4_y)
                    ]
                    
                    self.get_logger().info(f"  === GROUND TRUTH DISTANCE ANALYSIS ===")
                    for name, x, y in interpretations:
                        min_dist = float('inf')
                        closest_base = -1
                        for i, (gt_x, gt_y, gt_z) in enumerate(ground_truth_bases):
                            dist = np.sqrt((x - gt_x)**2 + (y - gt_y)**2)
                            if dist < min_dist:
                                min_dist = dist
                                closest_base = i + 1
                        self.get_logger().info(f"  {name:12}: closest to BASE_{closest_base}, distance = {min_dist:.3f}m")
                    
                    # Store this detection's position data with motion compensation
                    frame_positions.append({
                        'delta': (delta_x, delta_y, base_ground_altitude),
                        'absolute': (global_x, global_y, base_ground_altitude),
                        'score': adjusted_score,  # Use motion-adjusted score
                        'near_initial': self.is_near_initial_base(global_x, global_y),
                        'motion_compensated': True
                    })
                    
                else:
                    self.get_logger().warning(f"Detection {i+1}: No depth data available. Using default for simulation.")
                    depth_value = np.float32(-0.4)
                    z = depth_value / 1000.0
                    
                    # Calculate the real altitude of the base (fallback case with motion compensation)
                    base_altitude_relative_to_drone = -z
                    base_ground_altitude = compensated_z + base_altitude_relative_to_drone

                # cálculo de delta_x e delta_y
                    delta_x = ((mid_x_rgb - D455_CX_DEPTH) * z / D455_FX_DEPTH) if D455_FX_DEPTH != 0 else 0
                    delta_y = ((mid_y_rgb - D455_CY_DEPTH) * z / D455_FY_DEPTH) if D455_FY_DEPTH != 0 else 0

                # aplica correções de baseline e bias
                    delta_x += D455_BASELINE - D455_BIAS_X
                    delta_y -= D455_BIAS_Y

                    self.get_logger().info(f"Detection {i+1}: Delta real x,y (fallback): ({delta_x:.3f}, {delta_y:.3f}) meters")
                    
                    # Calculate absolute coordinates (fallback with motion compensation)
                    cos_yaw = np.cos(compensated_yaw)
                    sin_yaw = np.sin(compensated_yaw)
                    global_x = compensated_x + (delta_x * cos_yaw - delta_y * sin_yaw)
                    global_y = compensated_y + (delta_x * sin_yaw + delta_y * cos_yaw)
                    
                    # Log fallback coordinates to CSV
                    delta_coords = (delta_x, delta_y, base_ground_altitude)
                    vehicle_state = (self.current_x, self.current_y, self.current_altitude, self.current_yaw,
                            self.current_vx, self.current_vy, self.current_vz)
                    compensated_state = (compensated_x, compensated_y, compensated_z, compensated_yaw)
                    global_coords = (global_x, global_y, base_ground_altitude)
                    flags = {
                        'motion_compensated': True, 
                        'near_initial_base': self.is_near_initial_base(global_x, global_y), 
                        'is_outlier': False
                    }
                    self.log_processed_coordinates_to_csv(delta_coords, vehicle_state, compensated_state, 
                                                        global_coords, flags, "FALLBACK_NO_DEPTH")
                    
                    # Store this detection's position data (fallback with motion compensation)
                    frame_positions.append({
                        'delta': (delta_x, delta_y, base_ground_altitude),
                        'absolute': (global_x, global_y, base_ground_altitude),
                        'score': adjusted_score,  # Use motion-adjusted score
                        'near_initial': self.is_near_initial_base(global_x, global_y),
                        'motion_compensated': True
                    })

            # Publish all positions from this frame
            remote_bases_count = 0
            motion_compensated_count = 0
            
            for pos_data in frame_positions:
                delta_x, delta_y, base_ground_altitude = pos_data['delta']
                global_x, global_y, _ = pos_data['absolute']
                
                if pos_data.get('motion_compensated', False):
                    motion_compensated_count += 1
                
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
            
            # Enhanced logging with motion compensation info
            compensation_status = f"🔧 {motion_compensated_count}/{len(frame_positions)} motion-compensated" if motion_compensated_count > 0 else ""
            
            self.get_logger().info(f"✅ Published {len(frame_positions)} delta positions and "
                                  f"{remote_bases_count} remote absolute positions from frame {compensation_status}")
                    
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
