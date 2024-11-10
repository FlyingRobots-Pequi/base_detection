import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
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
        
        blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
        
        lab_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_img)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        preprocessed_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        hsv = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2HSV)
        
        # NEW: Mascara azul
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # NEW: Mascara amarela
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # NEW: Combinacao das duas mascaras e passa para o modelo
        # TODO: Verificar se o modelo atual vai conseguir identificar as bases, pois a matriz de 0 e 1 para ser reconhecida mudou de padrao.
        combined_mask = cv2.bitwise_or(mask_blue, mask_yellow)
        
        result = np.zeros_like(preprocessed_img)
        result[combined_mask > 0] = [255, 255, 255]
        
        results_fly = self.model(result)[0]
        
        for result_fly in results_fly.boxes.data.tolist():
            self.x1, self.y1, self.x2, self.y2, score, class_id = result_fly

            if score > self.threshold_helmet:
                cv2.rectangle(preprocessed_img, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), (0, 255, 0), 4)
                cv2.putText(preprocessed_img, f"{score:.2f}", (int(self.x1), int(self.y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                
                coord_msg = Float32MultiArray()
                coord_msg.data = [self.x1, self.y1, self.x2, self.y2]
                self.coord_publisher.publish(coord_msg)

        inferred_image_msg = self.bridge.cv2_to_imgmsg(preprocessed_img, encoding="bgr8")
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
