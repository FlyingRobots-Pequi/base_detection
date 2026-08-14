"""Realiza a detecçao e a extracao do ponto 3D referente as bases em tempo real"""
# rclpy
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros
from tf2_geometry_msgs import do_transform_point

# ros messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray
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

    def __init__(self):
        """Initializes the node, publishers, subscription, and YOLO model."""
        super().__init__("Drone_Base_Detection")
        self.get_logger().info("Base Detection Node Initialized")

        # set model parameters
        self.model_path = "/root/ros2_ws/src/base_detection/base_detection/best.pt"
        self.detection_threshold = 0.9
        self.hsv_filter_lower = np.array([42, 30, 120])
        self.hsv_filter_upper = np.array([135, 190, 220])

        self.arr = np.zeros((30, 3))
        self.valid_pos = 0

        # downward_rgb_camera (hermit/model.sdf): mounted 0.24m below
        # base_link, pitched 90 deg to look straight down. No depth needed:
        # each pixel's ray is cast in camera_down_optical_frame and
        # intersected with the ground plane (Z=0 in "map") using the real
        # TF2 chain map->base_link->camera_down_optical_frame -- that chain
        # already carries the drone's full attitude/position, so this is
        # robust without any hand-rolled heading trig. Assumes the target is
        # on the ground; fine for landing (only ground-level bases are
        # landable), not for the elevated bases.
        self.camera_intrinsics = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.k_sub = self.create_subscription(CameraInfo, "/camera/down/camera_info", self.intrinsics_callback, 10)
        self.image_sub = self.create_subscription(Image, "/camera/down/image", self._inferenzzia, 10)

        # Publisher for drone position and trajectory markers
        self.drone_position_publisher = self.create_publisher(
            MarkerArray, "/base_detection/markers", 10
        )
        
        self.base_publisher = self.create_publisher(PoseArray, "/base_detection/bases", 10)
        self.debug_image_publisher = self.create_publisher(Image, "/base_detection/debug_image", 10)

        # Erro de pixel NORMALIZADO ((u-cx)/fx, (v-cy)/fy) da deteccao mais
        # proxima do centro da imagem, no frame camera_down_optical_frame.
        # E o sinal de realimentacao do servo visual da missao (fase ALIGN):
        # levar esse erro a zero poe o drone no nadir da base, o que tambem
        # zera o vies de paralaxe das bases elevadas.
        self.pixel_error_publisher = self.create_publisher(
            PointStamped, "/base_detection/target_pixel_error", 10
        )

        # Portao de nadir, em METROS de afastamento no chao (nao em angulo):
        # so entra no mapa deteccao cujo ponto projetado esteja a menos disto
        # do ponto sob o drone. Motivo do portao: base ELEVADA vista de
        # esguelha entra deslocada ~h*tan(theta) e a media exponencial
        # mistura/duplica bases vizinhas.
        # Ja foi um limiar ANGULAR fixo, que na pratica muda de tamanho com a
        # altura: varrendo a 1.5 m ele so aceitava base quase exatamente sob o
        # drone e a varredura inteira mapeou ZERO bases. Em metros o criterio
        # independe da altitude de voo.
        self.declare_parameter("nadir_gate_m", 1.0)
        self.nadir_gate_m = self.get_parameter("nadir_gate_m").value

        # A missao publica aqui a altura REAL da base apos pousar nela
        # (x,y = qual base; z = altura medida no toque, VehicleLocalPosition).
        # Corrige o Z=0 assumido pela projecao no plano do chao.
        self.height_update_sub = self.create_subscription(
            PointStamped, "/base_detection/base_height_update",
            self.base_height_update_callback, 10,
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

        self.cluster_thresholds = 0.45


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

        # 3. Detected base markers
        for i in range(self.valid_pos):
            base_marker = Marker()
            base_marker.header.frame_id = "map"
            base_marker.header.stamp = self.get_clock().now().to_msg()
            base_marker.ns = "detected_bases"
            base_marker.id = i
            base_marker.type = Marker.CYLINDER
            base_marker.action = Marker.ADD

            base_marker.pose.position.x = float(self.arr[i][0])
            base_marker.pose.position.y = float(self.arr[i][1])
            base_marker.pose.position.z = float(self.arr[i][2])
            base_marker.pose.orientation.w = 1.0

            base_marker.scale.x = 0.3
            base_marker.scale.y = 0.3
            base_marker.scale.z = 0.05

            base_marker.color.r = 1.0
            base_marker.color.g = 0.6
            base_marker.color.b = 0.0
            base_marker.color.a = 0.9

            marker_array.markers.append(base_marker)

        self.drone_position_publisher.publish(marker_array)

    def base_height_update_callback(self, msg: PointStamped):
        """Sobrescreve o Z da base mais proxima de (x,y) com a altura medida.

        Publicado pela missao apos confirmar o pouso: no toque, a altitude do
        drone (VehicleLocalPosition) E a altura da base -- dado que a projecao
        por plano Z=0 nao tem como estimar para bases elevadas.
        """
        if self.valid_pos == 0:
            return
        xy = np.array([msg.point.x, msg.point.y])
        distances = np.linalg.norm(self.arr[:self.valid_pos, :2] - xy, axis=1)
        idx = int(np.argmin(distances))
        if distances[idx] < self.cluster_thresholds:
            self.arr[idx][2] = msg.point.z
            self.get_logger().info(
                f"Base {idx}: altura corrigida para {msg.point.z:.2f} m (pouso)"
            )

    def _inferenzzia(self, color_data):
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

        if frame_detections:
            self.detection_count += 1

        # get 3d points coordinates via ground-plane intersection (no depth sensor)
        if frame_detections and self.camera_intrinsics is not None:
            fx = self.camera_intrinsics['fx']
            fy = self.camera_intrinsics['fy']
            cx = self.camera_intrinsics['cx']
            cy = self.camera_intrinsics['cy']

            # Servo visual: publica o erro normalizado da deteccao mais
            # central. Norma pequena == drone praticamente no nadir da base.
            best_error = None
            for detection in frame_detections:
                u = (detection[0] + detection[2]) / 2
                v = (detection[1] + detection[3]) / 2
                err = ((u - cx) / fx, (v - cy) / fy)
                norm = float(np.hypot(err[0], err[1]))
                if best_error is None or norm < best_error[2]:
                    best_error = (err[0], err[1], norm)

            error_msg = PointStamped()
            error_msg.header.stamp = color_data.header.stamp
            error_msg.header.frame_id = "camera_down_optical_frame"
            error_msg.point.x = best_error[0]
            error_msg.point.y = best_error[1]
            error_msg.point.z = best_error[2]  # norma, p/ threshold no consumidor
            self.pixel_error_publisher.publish(error_msg)

            for detection in frame_detections:
                u = int((detection[0] + detection[2]) / 2)
                v = int((detection[1] + detection[3]) / 2)

                point_3d = self.get_points_to_3d(u, v)
                if point_3d is None:
                    continue

                # Portao de nadir em metros: descarta do MAPA a deteccao cujo
                # ponto no chao esteja longe da vertical do drone (o servo
                # visual segue usando todas, pelo topico de erro de pixel).
                nadir = self.nadir_point()
                if nadir is not None:
                    offset = float(np.hypot(point_3d[0] - nadir[0],
                                            point_3d[1] - nadir[1]))
                    if offset > self.nadir_gate_m:
                        continue

                if self.valid_pos == 0:
                    self.get_logger().info(f"Base Position in World Frame: X: {point_3d[0]:.3f}m, Y: {point_3d[1]:.3f}m, Z: {point_3d[2]:.3f}m")
                    self.arr[self.valid_pos] = point_3d
                    self.valid_pos += 1
                else:
                    # Define limite de 1 metro (para XY apenas)
                    distance_threshold_xy = 1.0  

                    # Calcula distância XY (ignorando Z)
                    distances_xy = np.linalg.norm(self.arr[:self.valid_pos, :2] - np.array(point_3d)[:2], axis=1)
                    min_dist_xy = np.min(distances_xy)
                    idx = np.argmin(distances_xy)

                    if min_dist_xy < distance_threshold_xy:
                        # Mesma base → média exponencial
                        self.arr[idx] = 0.8 * self.arr[idx] + 0.2 * np.array(point_3d)
                    else:
                        # Nova base
                        self.arr[self.valid_pos] = point_3d
                        self.valid_pos += 1

                bases = PoseArray()
                bases.header.stamp = self.get_clock().now().to_msg()
                bases.header.frame_id = "map"
                for i in range(self.valid_pos):
                    pose = Pose()
                    pose.position.x = float(self.arr[i][0])
                    pose.position.y = float(self.arr[i][1])
                    pose.position.z = float(self.arr[i][2])
                    pose.orientation.w = 1.0  # orientação neutra
                    bases.poses.append(pose)

                self.base_publisher.publish(bases)

                self.get_logger().debug(f"Detected 3D Point: {point_3d}")

        debug_image_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
        debug_image_msg.header = color_data.header
        self.debug_image_publisher.publish(debug_image_msg)

    def nadir_point(self):
        """(x, y) do ponto sob o drone no frame "map", ou None sem TF."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "map", "camera_down_optical_frame", Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        return (tf.transform.translation.x, tf.transform.translation.y)

    def get_points_to_3d(self, u, v):
        """Back-projects a down-camera pixel onto the ground plane (Z=0 in "map"), no depth sensor.

        Casts the pixel's ray in camera_down_optical_frame and transforms
        both the camera origin and a point along the ray into "map" via TF2
        (map->base_link->camera_down_optical_frame), then intersects that
        ray with Z=0. TF already carries the drone's real attitude/position,
        so this is robust without hand-rolled heading trig. Assumes the
        target sits on the ground -- correct for landing on ground-level
        bases, not for the elevated ones.
        """
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        x_opt = (u - cx) / fx
        y_opt = (v - cy) / fy

        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "camera_down_optical_frame", Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF map<-camera_down_optical_frame indisponível: {e}", throttle_duration_sec=2.0)
            return None

        origin_camera = PointStamped()
        origin_camera.header.frame_id = "camera_down_optical_frame"
        origin_camera.point.x = 0.0
        origin_camera.point.y = 0.0
        origin_camera.point.z = 0.0

        ray_camera = PointStamped()
        ray_camera.header.frame_id = "camera_down_optical_frame"
        ray_camera.point.x = x_opt
        ray_camera.point.y = y_opt
        ray_camera.point.z = 1.0

        origin_map = do_transform_point(origin_camera, transform).point
        ray_map = do_transform_point(ray_camera, transform).point

        direction = np.array([
            ray_map.x - origin_map.x,
            ray_map.y - origin_map.y,
            ray_map.z - origin_map.z,
        ])

        # Ray has to actually point toward the ground (Z decreasing) to hit Z=0.
        if direction[2] >= 0.0:
            self.get_logger().warn(
                "Raio da câmera não aponta pro chão (attitude estranha?)",
                throttle_duration_sec=2.0,
            )
            return None

        t = -origin_map.z / direction[2]
        if t <= 0.0:
            return None

        x_world = origin_map.x + t * direction[0]
        y_world = origin_map.y + t * direction[1]
        z_world = 0.0

        self.get_logger().debug(
            f"Ponto no chão para o pixel ({u}, {v}) -> "
            f"[X: {x_world:.3f}m, Y: {y_world:.3f}m, Z: {z_world:.3f}m]"
        )

        return (x_world, y_world, z_world)

    def intrinsics_callback(self, msg):
        self.camera_intrinsics = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5],
        }
        self.get_logger().info(f"Camera intrinsics received: {self.camera_intrinsics}")
        self.destroy_subscription(self.k_sub)

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