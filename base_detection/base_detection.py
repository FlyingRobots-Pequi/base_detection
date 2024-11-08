import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import concurrent.futures
import cv2
import numpy as np
import torch

class ImageInferencer(Node):
    def __init__(self):
        super().__init__('brota_na_base')
        self.publisher_ = self.create_publisher(Image, 'inferred_image_capiche', 10)
        self.subscription = self.create_subscription(Image, '/hermit/camera/d435i/color/image_raw', self._inferenzzia, 10)
        self.bridge = CvBridge()
        self.model = YOLO('/ros2_ws/src/base_detection/base_detection/best.pt')
        self.threshold_helmet = 0.9
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        
        self.coord_publisher = self.create_publisher(
            Float32MultiArray, 'detected_coordinates', 10)

    def _inferenzzia(self, data):
        img = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([42, 30, 120]), np.array([135, 190, 220]))

        result = np.zeros_like(img)
        result[mask > 0] = [255, 255, 255]

        results_fly = self.model(result)[0]

        for result_fly in results_fly.boxes.data.tolist():
            self.x1, self.y1, self.x2, self.y2, score, class_id = result_fly

            if score > self.threshold_helmet:
                cv2.rectangle(result, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), (0, 255, 0), 4)
                cv2.putText(result, str(score), (int(self.x1), int(self.y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Publish the coordinates
                coord_msg = Float32MultiArray()
                coord_msg.data = [self.x1, self.y1, self.x2, self.y2]
                self.coord_publisher.publish(coord_msg)

        inferred_image_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
        self.publisher_.publish(inferred_image_msg)

def main(args=None):
    rclpy.init(args=args)
    image_inferencer = ImageInferencer()

    try:
        rclpy.spin(image_inferencer)
    except KeyboardInterrupt:
        pass
    finally:
        image_inferencer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()