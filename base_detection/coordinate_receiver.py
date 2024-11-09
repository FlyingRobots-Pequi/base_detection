#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
from px4_msgs.msg import VehicleLocalPosition
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy


class CoordinateReceiver(Node):
    def __init__(self, bias_x: float = 0.1, bias_y: float = 0.1, fx_depth:float = 925.1, fy_depth:float = 925.1, cx_depth:float = 639.5, cy_depth:float = 359.5, baseline:float = 0.025):
        super().__init__('coordinate_receiver')
        # QoS Profile Definition
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )
        
        
        self.subscription = self.create_subscription(
            Float32MultiArray, 'detected_coordinates', self.listener_callback, 10)
        
        # Subscription to the depth image
        self.depth_subscription = self.create_subscription(
            Image, '/hermit/camera/d435i/depth/image_rect_raw', self.depth_callback, 10)
        
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        
        self.delta_publisher = self.create_publisher(Point, 'delta_position', 10)


        # Initialize CvBridge
        self.bridge = CvBridge()

        # Variable to store the latest depth image
        self.latest_depth = None

        self.bias_x = bias_x  # Adjust for any known biases
        self.bias_y = bias_y

        # Camera intrinsic parameters for the depth camera
        self.fx_depth = fx_depth  # Focal length in pixels (depth camera)
        self.fy_depth = fy_depth
        self.cx_depth = cx_depth  # Principal point (depth camera)
        self.cy_depth = cy_depth

        # Baseline between the RGB and depth cameras
        self.baseline = baseline # 2.5 cm

    def depth_callback(self, msg):
        try:
            # Convert the depth image to a NumPy array
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth = depth_image.astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Error in depth_callback: {e}")
    
    def vehicle_local_position_callback(self, msg):
        # Get current altitude and yaw angle from the message
        self.current_altitude = msg.z  # Altitude in NED frame (z is negative upwards)
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_yaw = msg.heading

        # Check for failsafe condition
        if self.current_altitude > -0.13:
            self.get_logger().error("Altitude exceeds safe threshold! Engaging failsafe.")
            self.failsafe_triggered = True  # Set the failsafe flag to true

    def listener_callback(self, msg):
        try:
            # Unpack the coordinates
            x1, y1, x2, y2 = msg.data
            self.get_logger().info(f"Received coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Calculate the width and height of the bounding box in pixels
            bbox_width = abs(x2 - x1)
            bbox_height = abs(y2 - y1)
            self.get_logger().info(f"Bounding box width: {bbox_width}, height: {bbox_height}")

            # Check if the bounding box is approximately square (within 10% tolerance)
            aspect_ratio = bbox_width / bbox_height if bbox_height != 0 else float('inf')
            tolerance = 0.1  # 10% tolerance
            if abs(aspect_ratio - 1) <= tolerance:
                self.get_logger().info("Bounding box is approximately square.")

                # Calculate the midpoint of the bounding box
                mid_x_rgb = int((x1 + x2) / 2)
                mid_y_rgb = int((y1 + y2) / 2)
                self.get_logger().info(f"Midpoint of the bounding box (RGB): ({mid_x_rgb}, {mid_y_rgb})")

                # Adjust for the offset between RGB and depth cameras
                # Since the depth and RGB images might have different resolutions and FOVs, consider transforming coordinates
                # For simplicity, we'll assume aligned images. In practice, you might need to use depth-to-color alignment functions

                if self.latest_depth is not None:
                    # Map RGB midpoint to depth image coordinates if necessary
                    # For this example, let's assume images are aligned and have the same resolution
                    mid_x_depth = mid_x_rgb
                    mid_y_depth = mid_y_rgb

                    # Get the depth value at the corresponding point
                    depth_value = self.latest_depth[mid_y_depth, mid_x_depth]
                    if np.isnan(depth_value) or depth_value <= 0:
                        self.get_logger().warning("Invalid depth value encountered.")
                        return

                    z = depth_value / 1000.0  # Convert from millimeters to meters
                    self.get_logger().info(f"Depth at midpoint: {z:.3f} meters")

                    # Camera intrinsic parameters for the depth camera
                    fx = self.fx_depth
                    fy = self.fy_depth
                    cx = self.cx_depth
                    cy = self.cy_depth

                    # Calculate real-world coordinates (X, Y, Z) in the camera coordinate system
                    # X = (u - cx) * Z / fx
                    # Y = (v - cy) * Z / fy
                    delta_x = (mid_x_depth - cx) * z / fx
                    delta_y = (mid_y_depth - cy) * z / fy

                    # Adjust for the baseline (offset) between the RGB and depth cameras
                    delta_x += self.baseline  # Assuming baseline along the X-axis

                    # Adjust for biasesq
                    delta_x -= self.bias_x 
                    delta_y -= self.bias_y 
                    
                    # delta_x += self.current_x
                    # delta_y += self.current_y # Tirar essse comentario quanto for fazer o voo envolta da arena.

                    self.get_logger().info(f"Delta real x: {delta_x:.3f} meters, Delta real y: {delta_y:.3f} meters")
                    
                    # Create and publish the Point message
                    delta_point = Point()
                    delta_point.x = delta_x
                    delta_point.y = delta_y
                    delta_point.z = 0.0  # Set to zero or include depth if applicable
                    self.delta_publisher.publish(delta_point)
                    
                else:
                    self.get_logger().warning("No depth data available.")
            else:
                self.get_logger().info("Bounding box is not approximately square. Ignoring this detection.")
        except ValueError as ve:
            self.get_logger().error(f"Value error in listener_callback: {ve}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in listener_callback: {e}")


def main(args=None):
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
