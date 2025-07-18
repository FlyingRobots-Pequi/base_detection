import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from .variables import (
    DETECTED_COORDINATES_TOPIC,
    ALIGNMENT_CORRECTION_TOPIC,
    COLOR_IMAGE_TOPIC2
)

# NÂO ESTA TESTADO, VAI DAR ERRO N SEI AONDE

def is_rectangular(contour, width, height):
    """
    Verifica se um contorno é minimamente retangular.
    """
    # Evita divisão por zero se a forma for inválida
    if min(width, height) <= 0:
        return False
        
    # Calcula a área do contorno
    contour_area = cv2.contourArea(contour)
    
    # Calcula a área do bounding box de área mínima
    bbox_area = width * height
    
    # Se a área do contorno for muito menor que a do bbox, não é retangular
    if contour_area < 0.3 * bbox_area:
        return False
    
    # Calcula a razão entre largura e altura
    aspect_ratio = max(width, height) / min(width, height)
    
    # Se for muito alongado (como uma linha), não é um pacote
    if aspect_ratio > 5.0:
        return False
    
    # Se for muito pequeno em uma dimensão, pode ser ruído
    if min(width, height) < 10:
        return False
    
    return True

class PackageDetector(Node):
    def __init__(self):
        super().__init__('package_detector')
        self.bridge = CvBridge()
        self.image_subscriber = self.create_subscription(
            Image,
            COLOR_IMAGE_TOPIC2,
            self.image_callback,
            10)
        self.bbox_subscriber = self.create_subscription(
            Float32MultiArray,
            DETECTED_COORDINATES_TOPIC,
            self.bbox_callback,
            10)
        self.alignment_publisher = self.create_publisher(
            Float32MultiArray, 
            ALIGNMENT_CORRECTION_TOPIC,
            10
        )
        
        self.last_bbox = None
        self.get_logger().info('Package Detector node has been started with advanced detection logic.')

    def image_callback(self, msg):
        if self.last_bbox is None:
            return

        detections = [self.last_bbox[i:i+5] for i in range(0, len(self.last_bbox), 5)]
        x_min, y_min, x_max, y_max, _ = detections[0]

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        y_min, y_max = int(max(0, y_min)), int(min(cv_image.shape[0], y_max))
        x_min, x_max = int(max(0, x_min)), int(min(cv_image.shape[1], x_max))
        
        sub_frame = cv_image[y_min:y_max, x_min:x_max]

        if sub_frame.size == 0:
            self.get_logger().warn("Bounding box resulted in an empty sub-frame. Skipping.")
            return

        # --- Etapa 1: Filtragem Multi-Cinza Avançada ---
        hsv = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2HSV)
        
        gray_masks = [
            cv2.inRange(hsv, np.array([0, 0, 20]), np.array([180, 60, 120])), # Escuro
            cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 50, 160])), # Médio
            cv2.inRange(hsv, np.array([0, 0, 60]), np.array([180, 40, 200])), # Claro
            cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 30, 240])) # Muito claro/reflexivo
        ]
        
        mask = gray_masks[0]
        for m in gray_masks[1:]:
            mask = cv2.bitwise_or(mask, m)

        # --- Etapa 2: Operações Morfológicas ---
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # --- Etapa 3: Validação do Contorno ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 100:
            return

        rect = cv2.minAreaRect(largest_contour)
        (center_x_local, center_y_local), (width, height), angle = rect

        # --- Etapa 4: Validação da Forma ---
        if not is_rectangular(largest_contour, width, height):
            self.get_logger().debug("Detected object is not rectangular enough to be a package.")
            return

        # --- Lógica de Correção de Orientação e Posição (mantida) ---
        if width > height:
            width, height = height, width
            angle += 90

        orientation_error = angle
        if orientation_error > 45:
            orientation_error -= 90
        
        center_x_global = center_x_local + x_min
        center_y_global = center_y_local + y_min
        
        img_height, img_width, _ = cv_image.shape
        target_x = img_width / 2
        target_y = img_height / 2
        
        position_error_x = center_x_global - target_x
        position_error_y = center_y_global - target_y

        alignment_msg = Float32MultiArray()
        alignment_msg.data = [orientation_error, position_error_x, position_error_y]
        self.alignment_publisher.publish(alignment_msg)
        self.get_logger().info(
            f"Publishing Validated Alignment: "
            f"Rot(deg): {orientation_error:.2f}, "
            f"Pos(px): ({position_error_x:.2f}, {position_error_y:.2f})"
        )

    def bbox_callback(self, msg):
        if msg.data:
            self.last_bbox = msg.data
        else:
            self.last_bbox = None

def main(args=None):
    rclpy.init(args=args)
    package_detector_node = PackageDetector()
    rclpy.spin(package_detector_node)
    package_detector_node.destroy_node()

if __name__ == '__main__':
    main() 