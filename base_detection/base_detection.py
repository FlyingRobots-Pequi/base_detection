"""
ROS2 node for real-time base detection.
Processes RGB images, applies HSV color filtering, and uses a YOLO model for detection.
Publishes detected base coordinates and a visualization image.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Bool, Int32
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from visualization_msgs.msg import Marker, MarkerArray
from px4_msgs.msg import VehicleLocalPosition
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from base_detection.variables import (
    COLOR_IMAGE_TOPIC,
    INFERRED_IMAGE_TOPIC,
    DETECTED_COORDINATES_TOPIC,
    NUMBER_OF_BASES_TOPIC,
    COLOR_IMAGE_TOPIC2,
    BOOL_DETECTOR_TOPIC,
    VEHICLE_LOCAL_POSITION_TOPIC,
)
from base_detection.parameters import get_image_inferencer_params


class ImageInferencer(Node):
    """
    ROS2 node for real-time base detection.

    Processes RGB images with HSV filtering and a YOLO model. Publishes detected
    coordinates and a visualization image.
    """

    def __init__(self):
        """Initializes the node, publishers, subscription, and YOLO model."""
        super().__init__("Drone_Base_Detection")
        self.get_logger().info("Base Detection Node Initialized")

        self.params = get_image_inferencer_params(self)

        self.publisher_ = self.create_publisher(Image, INFERRED_IMAGE_TOPIC, 10)

        self.coord_publisher = self.create_publisher(
            Float32MultiArray, DETECTED_COORDINATES_TOPIC, 10
        )

        self.bool_detector = self.create_publisher(Bool, BOOL_DETECTOR_TOPIC, 10)

        self.number_of_bases_publisher = self.create_publisher(
            Int32, NUMBER_OF_BASES_TOPIC, 10
        )

        self.subscription = self.create_subscription(
            Image, COLOR_IMAGE_TOPIC2, self._inferenzzia, 10
        )
        
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
            VEHICLE_LOCAL_POSITION_TOPIC, 
            self.vehicle_local_position_callback, 
            qos_profile,
        )
        
        # Drone trajectory storage (keeps all history)
        self.drone_trajectory = []
        
        self.bridge = CvBridge()
        self.model = YOLO(self.params.model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {device}")

        self.model.to(device)

    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        """Callback to update vehicle position and publish drone position marker with trajectory."""
        # Add current position to trajectory (keeps all history)
        self.drone_trajectory.append([msg.x, msg.y, -msg.z])  # Use actual drone altitude
        
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
                from geometry_msgs.msg import Point
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

    def _inferenzzia(self, data):
        """Callback to process an image, run inference, and publish results."""
        img = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_bound = np.array(self.params.hsv_filter.lower)
        upper_bound = np.array(self.params.hsv_filter.upper)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        result = np.zeros_like(img)
        result[mask > 0] = [255, 255, 255]

        results_fly = self.model(result)[0]

        frame_detections = []
        for result_fly in results_fly.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result_fly
            if score > self.params.detection_threshold:

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
                    self.get_logger().warn(f"Centroid calculation failed: {e}. Falling back to bbox center.")
                
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

        # --- Improved Publishing Logic ---

        # Publish whether any base was detected in this frame
        has_base_msg = Bool()
        has_base_msg.data = bool(frame_detections)
        self.bool_detector.publish(has_base_msg)

        # Publish the number of detected bases in this frame
        num_bases_msg = Int32()
        num_bases_msg.data = len(frame_detections)
        self.number_of_bases_publisher.publish(num_bases_msg)

        # Publish coordinates only if detections were made
        if frame_detections:
            coord_msg = Float32MultiArray()
            flat_detections = [
                item for detection in frame_detections for item in detection
            ]
            coord_msg.data = flat_detections
            self.coord_publisher.publish(coord_msg)
            self.get_logger().debug(
                f"Published {len(frame_detections)} detections in batch"
            )

        inferred_image_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
        self.publisher_.publish(inferred_image_msg)


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


if __name__ == "__main__":
    main()
