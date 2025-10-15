"""Realiza a detecçao e a extracao do ponto 3D referente as bases em tempo real"""
# rclpy
import rclpy
from rclpy.node import Node
import message_filters 

# ros messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from px4_msgs.msg import VehicleLocalPosition

# model dependencies
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Qos configuration
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

class ImageInferencer(Node):
    """
    ROS2 node for real-time base detection.

    Processes RGB images with HSV filtering and a YOLO model. Publishes detected
    coordinates and a visualization image.
    """

    def _init_(self):
        """Initializes the node, publishers, subscription, and YOLO model."""
        super()._init_("Drone_Base_Detection")
        self.get_logger().info("Base Detection Node Initialized")

        # set model parameters
        self.model_path = "/root/ros2_ws/src/base_detection/base_detection/best.pt"
        self.detection_threshold = 0.9
        self.hsv_filter_lower = np.array([42, 30, 120])
        self.hsv_filter_upper = np.array([135, 190, 220])

        
        #create d455 config subscriber
        self.k_sub = self.create_subscription(CameraInfo, "/camera/color/camera_info", self.intrinsics_callback, 10)
        self.camera_intrinsics = None

        # create d455 depth and image sub 
        self.depth_sub = message_filters.Subscriber(self, Image, "/camera/depth/depth_image")
        self.image_sub = message_filters.Subscriber(self, Image, "/camera/color/image_raw")

        #time sincronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1
            )
        
        self.ts.registerCallback(self._inferenzzia)
        
        # Publisher for drone position and trajectory markers
        self.drone_position_publisher = self.create_publisher(
            MarkerArray, "/base_detection/markers", 10
        )
        
        # Subscription for vehicle local position
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition, 
            "/fmu/out/vehicle_local_position", 
            self.vehicle_local_position_callback, 
            qos_profile,
        )
        
        # Drone trajectory storage (keeps all history)
        self.drone_trajectory = []
        self.actual_position = VehicleLocalPosition()
        
        # Heartbeat timer (log every 2 second)
        self.frame_count = 0
        self.detection_count = 0
        self.create_timer(2.0, self.heartbeat_callback)
        
        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {device}")

        self.model.to(device)

    def heartbeat_callback(self):
        """Log heartbeat message every second with processing statistics."""
        self.get_logger().info(
            f"Base Detection Active - Processed {self.frame_count} frames, "
            f"Detected bases in {self.detection_count} frames"
        )
        self.frame_count = 0
        self.detection_count = 0

    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        """Callback to update vehicle position and publish drone position marker with trajectory."""
        # Add current position to trajectory (keeps all history)
        self.drone_trajectory.append([msg.x, msg.y, -msg.z])  # Use actual drone altitude

        # save actual position
        self.actual_position = msg

        # Publish drone position and trajectory markers
        self.publish_drone_position_marker(msg.x, msg.y)

    def publish_drone_position_marker(self, x: float, y: float):
        """Publishes drone position and trajectory as markers in RViz."""
        marker_array = MarkerArray()
        
        # 1. Current drone position marker
        drone_marker = Marker()
        drone_marker.header.frame_id = "map"
        drone_marker.header.stamp = self.get_clock().now().to_msg()
        drone_marker.ns = "drone_position"
        drone_marker.id = 0
        drone_marker.type = Marker.SPHERE
        drone_marker.action = Marker.ADD
        
        drone_marker.pose.position.x = x
        drone_marker.pose.position.y = y
        drone_marker.pose.position.z = self.drone_trajectory[-1][2] if self.drone_trajectory else 0.0
        drone_marker.pose.orientation.w = 1.0
        
        drone_marker.scale.x = 0.3
        drone_marker.scale.y = 0.3
        drone_marker.scale.z = 0.3
        
        drone_marker.color.r = 0.0
        drone_marker.color.g = 1.0
        drone_marker.color.b = 0.0
        drone_marker.color.a = 0.8
        
        marker_array.markers.append(drone_marker)
        
        # 2. Trajectory line marker (if we have enough points)
        if len(self.drone_trajectory) > 1:
            trajectory_marker = Marker()
            trajectory_marker.header.frame_id = "map"
            trajectory_marker.header.stamp = self.get_clock().now().to_msg()
            trajectory_marker.ns = "drone_trajectory"
            trajectory_marker.id = 1
            trajectory_marker.type = Marker.LINE_STRIP
            trajectory_marker.action = Marker.ADD
            
            # Add all trajectory points
            for point in self.drone_trajectory:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                trajectory_marker.points.append(p)
            
            trajectory_marker.scale.x = 0.05  # Line width
            
            trajectory_marker.color.r = 0.0
            trajectory_marker.color.g = 0.5
            trajectory_marker.color.b = 1.0
            trajectory_marker.color.a = 0.6
            
            marker_array.markers.append(trajectory_marker)
        
        self.drone_position_publisher.publish(marker_array)

    def _inferenzzia(self, color_data, depth_data):
        """Callback to process an image, run inference, and publish results."""
        img = self.bridge.imgmsg_to_cv2(color_data, desired_encoding="bgr8")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_filter_lower, self.hsv_filter_upper)

        result = np.zeros_like(img)
        result[mask > 0] = [255, 255, 255]

        # Run inference with verbose=False to suppress YOLO logs
        results_fly = self.model(result, verbose=False)[0]
        
        # Update frame counter
        self.frame_count += 1

        frame_detections = []
        for result_fly in results_fly.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result_fly
            if score > self.detection_threshold:

                # --- Centroid Refinement ---
                # Fallback to bbox center if centroid fails
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                try:
                    # Crop the HSV mask to the bounding box
                    roi = mask[int(y1) : int(y2), int(x1) : int(x2)]
                    
                    # Calculate moments for the cropped mask
                    moments = cv2.moments(roi)
                    if moments["m00"] > 0:
                        # Calculate centroid and convert to global coordinates
                        c_x = int(moments["m10"] / moments["m00"]) + x1
                        c_y = int(moments["m01"] / moments["m00"]) + y1
                        center_x, center_y = float(c_x), float(c_y)
                except Exception as e:
                    # Remove or reduce logging here - only log at debug level
                    self.get_logger().debug(f"Centroid calculation failed: {e}. Falling back to bbox center.")
                
                # Use a tiny bounding box around the centroid for publishing
                # This ensures the receiver calculates the exact centroid without changing message format.
                frame_detections.append([center_x -1, center_y -1, center_x + 1, center_y + 1, score])

                cv2.rectangle(
                    result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4
                )
                cv2.circle(result, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

                cv2.putText(
                    result,
                    f"{score:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # get 3d points coordinates

        arr = np.zeros((30, 3))
        valid_pos = 0
        if frame_detections:
            depth_img = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding='32FC1')
            depth_img_resized = cv2.resize(
                depth_img, 
                (1280, 720),         # Dimensões alvo (largura, altura)
                interpolation=cv2.INTER_NEAREST # Método de interpolação correto para profundidade
            )

            for detection in frame_detections:
                u = int((detection[0] + detection[2]) / 2)
                v = int((detection[1] + detection[3]) / 2)
                depth = depth_img_resized[v, u]

                # Convert to 3D poin
                if valid_pos == 0:
                    camera_frame_3d = self.get_points_to_3d(u, v, depth)
                    x_b = -camera_frame_3d[1] - 0.13
                    y_b = -camera_frame_3d[0] - 0.05
                    z_b = -camera_frame_3d[2]

                    x_world = self.actual_position.x + (x_b * np.cos(self.actual_position.yaw) - y_b * np.sin(self.actual_position.yaw))
                    y_world = self.actual_position.y + (x_b * np.sin(self.actual_position.yaw) + y_b * np.cos(self.actual_position.yaw))
                    z_world = self.actual_position.z - z_b
                    point_3d = (x_world, y_world, z_world)
                    self.get_logger().info(f"Base Position in World Frame: X: {x_world:.3f}m, Y: {y_world:.3f}m, Z: {z_world:.3f}m")

                    arr[valid_pos] = point_3d
                    valid_pos += 1
                else:
                    camera_frame_3d = self.get_points_to_3d(u, v, depth)
                    x_b = -camera_frame_3d[1] - 0.13
                    y_b = -camera_frame_3d[0] - 0.05
                    z_b = -camera_frame_3d[2]

                    x_world = self.actual_position.x + (x_b * np.cos(self.actual_position.yaw) - y_b * np.sin(self.actual_position.yaw))
                    y_world = self.actual_position.y + (x_b * np.sin(self.actual_position.yaw) + y_b * np.cos(self.actual_position.yaw))
                    z_world = self.actual_position.z + z_b
                    point_3d = (x_world, y_world, z_world)

                    aux =  arr[0:valid_pos] - np.array(point_3d)
                    aux = aux**2
                    mask = aux < self.cluster_thresholds
                    if mask:
                        arr[mask] = (arr[mask] + aux[mask])/2
                    else:
                        arr[valid_pos] = point_3d
                        valid_pos += 1

                    
                self.get_logger().info(f"Detected 3D Point: {point_3d}")
            
        # inferred_image_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")

    def get_points_to_3d(self, x, y, depth):
        if depth <= 0.0 or np.isnan(depth) or np.isinf(depth):
            self.get_logger().error(f"invalid depth ==> value: {depth}")
            return
        
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        Z = float(depth)
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        
        self.get_logger().info(
            f"Ponto 3D para o pixel ({x}, {y}) | Profundidade: {Z:.3f}m -> "
            f"[X: {X:.3f}m, Y: {Y:.3f}m, Z: {Z:.3f}m]"
        )

        return (X, Y, Z)
    
    def intrinsics_callback(self, msg):
        self.camera_intrinsics = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5]
        }
        self.get_logger().info(f"Camera intrinsics received: {self.camera_intrinsics}")
        # Unsubscribe after receiving the intrinsics once
        self.k_sub  # keep a reference to avoid garbage collection
        self.destroy_subscription(self.k_sub)
        self.get_logger().info("Unsubscribed from camera intrinsics topic.")

def main(args=None):
    """Initializes and runs the ROS2 node."""
    rclpy.init(args=args)
    image_inferencer = ImageInferencer()

    try:
        rclpy.spin(image_inferencer)
    except KeyboardInterrupt:
        pass
    finally:
        image_inferencer.destroy_node()
        rclpy.shutdown()


if _name_ == "_main_":
    main()