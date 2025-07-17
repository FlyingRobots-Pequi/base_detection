"""
Base Detection Node for Robotics Competition.

This module implements a ROS2 node that performs real-time base detection using
computer vision and deep learning. It processes RGB images from a D435i camera,
applies color-based filtering, and uses a YOLO model for detection.

The node:
- Subscribes to RGB camera feed
- Processes images using HSV color filtering
- Performs inference using a YOLO model
- Publishes detected base coordinates and visualization

Dependencies:
    - ROS2
    - OpenCV
    - PyTorch
    - Ultralytics YOLO
    - cv_bridge
    - NumPy
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from base_detection.variables import (
    COLOR_IMAGE_TOPIC,
    INFERRED_IMAGE_TOPIC,
    DETECTED_COORDINATES_TOPIC,
    COLOR_IMAGE_TOPIC2,
)


class ImageInferencer(Node):
    """
    A ROS2 node for real-time base detection using computer vision and deep learning.

    This class processes RGB images from a camera feed, applies HSV color filtering
    to isolate potential base regions, and uses a pre-trained YOLO model for
    accurate base detection. It publishes both the detected coordinates and
    a visualization of the detection results.

    Attributes:
        threshold_helmet (float): Confidence threshold for detection (default: 0.9)
        x1 (float): Left coordinate of detected bounding box
        y1 (float): Top coordinate of detected bounding box
        x2 (float): Right coordinate of detected bounding box
        y2 (float): Bottom coordinate of detected bounding box
        model (YOLO): Pre-trained YOLO model for base detection
    """

    def __init__(self):
        """
        Initialize the ImageInferencer node.

        Sets up:
        - Publishers for inference results and coordinates
        - Subscription to RGB camera feed
        - YOLO model loading and configuration
        - CUDA device selection if available
        """
        super().__init__("brota_na_base")
        self.get_logger().info("Base Detection Node Initialized")

        self.publisher_ = self.create_publisher(Image, INFERRED_IMAGE_TOPIC, 10)

        self.coord_publisher = self.create_publisher(
            Float32MultiArray, DETECTED_COORDINATES_TOPIC, 10
        )

        self.subscription = self.create_subscription(
            Image, COLOR_IMAGE_TOPIC2, self._inferenzzia, 10
        )
        self.bridge = CvBridge()
        self.model = YOLO("/root/ros2_ws/src/base_detection/base_detection/best.pt")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {device}")
        
        self.model.to(device)

        # Initialize bounding box coordinates
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None

        self.threshold_helmet = 0.9

    def _inferenzzia(self, data):
        """
        Process incoming RGB images and perform base detection.

        This method:
        1. Converts ROS image message to OpenCV format
        2. Applies HSV color filtering
        3. Runs YOLO inference on filtered image
        4. Collects all valid detections from the frame
        5. Publishes all detections as a single batch message

        Args:
            data (sensor_msgs.msg.Image): Input RGB image from camera

        Note:
            HSV filter values are tuned for the specific base color:
            - Hue: [42, 135]
            - Saturation: [30, 190]
            - Value: [120, 220]
        """
        img = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

        # Apply HSV color filtering
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([42, 30, 120]), np.array([135, 190, 220]))

        # Create binary mask result
        result = np.zeros_like(img)
        result[mask > 0] = [255, 255, 255]

        # Run YOLO inference
        results_fly = self.model(result)[0]

        # Collect all valid detections from this frame
        frame_detections = []
        
        for result_fly in results_fly.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result_fly
            if score > self.threshold_helmet:
                # Store detection data
                frame_detections.append([x1, y1, x2, y2, score])
                
                # Draw detection visualization
                cv2.rectangle(result, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 4)
                
                # Calculate center for visualization
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)
                
                cv2.putText(result, 
                        f"{score:.2f}", 
                        (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2, 
                        cv2.LINE_AA)

        # Publish all detections from this frame as a single batched message
        if frame_detections:
            # Flatten the list: [x1,y1,x2,y2,score, x1,y1,x2,y2,score, ...]
            coord_msg = Float32MultiArray()
            coord_msg.data = [item for detection in frame_detections for item in detection]
            self.coord_publisher.publish(coord_msg)
            
            self.get_logger().info(f"Published {len(frame_detections)} detections in batch")
        
        # Publish visualization image
        inferred_image_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
        self.publisher_.publish(inferred_image_msg)


def main(args=None):
    """
    Main entry point for the base detection node.

    Args:
        args: Command line arguments (unused)
    """
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
