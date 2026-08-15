"""Realiza a detecçao e a extracao do ponto 3D referente as bases em tempo real"""
# rclpy
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import message_filters
import tf2_ros
from tf2_geometry_msgs import do_transform_point

# ros messages
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from px4_msgs.msg import VehicleLocalPosition

# model dependencies
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import time
import math

# Qos configuration
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
    qos_profile_sensor_data,
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

        # downward_rgb_camera + downward_depth_camera (hermit/model.sdf):
        # mounted 0.24m below base_link, pitched 90 deg to look straight down.
        # Real depth gives the true distance to whatever the pixel hits, so
        # bases at any height are localized correctly (a flat ground-plane
        # assumption would put the elevated bases on the floor below them).
        # The point is then transformed into "map" via the TF2 chain
        # map->base_link->camera_down_optical_frame, which already carries the
        # drone's full attitude/position -- no hand-rolled heading trig.
        self.camera_intrinsics = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.image_sub = message_filters.Subscriber(
            self, Image, "/camera/down/image",
            qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(
            self, Image, "/camera/down/depth",
            qos_profile=qos_profile_sensor_data)
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
        
        self.base_publisher = self.create_publisher(PoseArray, "/base_detection/bases", 10)
        # Somente as bases observadas no quadro atual. O mapa acima e
        # acumulado e serve para visualizacao, mas nao deve disparar uma
        # aproximacao autonoma usando uma observacao antiga.
        self.current_base_publisher = self.create_publisher(
            PoseArray, "/base_detection/current_bases", 10)
        self.current_body_base_publisher = self.create_publisher(
            PoseArray, "/base_detection/current_bases_body", 10)
        self.debug_image_publisher = self.create_publisher(Image, "/base_detection/debug_image", 10)

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
        
        # Heartbeat timer (log every 2 second)
        self.frame_count = 0
        self.detection_count = 0
        self.certified_frame_count = 0
        self.body_frame_count = 0
        self.create_timer(2.0, self.heartbeat_callback)
        
        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {device}")

        self.model.to(device)

        # YOLO em todos os ~30 quadros/s consumia um nucleo inteiro e reduzia
        # o fator de tempo do Gazebo para ~0.1. O controle precisa de deteccao
        # frequente, mas 8 Hz ainda fornece as cinco confirmacoes em <1 s.
        self.max_inference_hz = 8.0
        self.last_inference_wall_time = 0.0
        # O circulo roxo e o alvo fixo no centro optico, nao uma nova
        # deteccao. A caixa verde + centro vermelho da cruz sao movidos ate ele.


    def heartbeat_callback(self):
        """Log heartbeat message every second with processing statistics."""
        self.get_logger().info(
            f"Base Detection Active - Processed {self.frame_count} frames, "
            f"Detected={self.detection_count}, certified={self.certified_frame_count}, "
            f"body={self.body_frame_count} frames"
        )
        self.frame_count = 0
        self.detection_count = 0
        self.certified_frame_count = 0
        self.body_frame_count = 0

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

    def _inferenzzia(self, color_data, depth_data):
        """Callback to process an image, run inference, and publish results."""
        now_wall = time.monotonic()
        if now_wall - self.last_inference_wall_time < 1.0 / self.max_inference_hz:
            return
        self.last_inference_wall_time = now_wall
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

                # --- Centroid Refinement + shape validation ---
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                cross_found = False

                try:
                    # Crop the HSV mask to the bounding box
                    roi = mask[int(y1) : int(y2), int(x1) : int(x2)]

                    # The base plate is a rotated square with a circular
                    # marker (white ring, black cross) inside it. The plate
                    # is the same filtered color as the ring's interior, so a
                    # plain centroid of all white pixels in the (axis-aligned)
                    # box also averages in whatever extra plate area falls in
                    # the box corners -- an amount that varies with the
                    # plate's rotation, and it visibly pulled the centroid off
                    # the true center (confirmed against ground truth: a bias
                    # that didn't track localization drift, i.e. not noise).
                    # A per-white-component approach doesn't work either: the
                    # cross's arms reach the ring, so the ring's white
                    # interior is itself split into 4 disconnected sectors by
                    # the cross -- there's no single white blob that
                    # represents "the target".
                    #
                    # The cross itself is the reliable landmark: it's a
                    # symmetric shape whose pixel centroid is its true
                    # intersection point, and unlike the ring/plate/corner
                    # triangles, it never touches the crop's own boundary
                    # (it's fully inside the ring). So: find contours in the
                    # *unfiltered* (black) pixels, keep the ones that don't
                    # touch the crop edges, and take the largest -- that's the
                    # cross, and its moments centroid is the target center.
                    # Verified against ground truth: this lands right on the
                    # cross intersection where the old whole-ROI moments
                    # landed visibly off it.
                    inverted = cv2.bitwise_not(roi)
                    contours, _ = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    roi_h, roi_w = roi.shape
                    min_area = 0.005 * roi_h * roi_w
                    cross_contour, cross_area = None, 0.0
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area < min_area:
                            continue
                        bx, by, bw, bh = cv2.boundingRect(contour)
                        touches_edge = bx <= 1 or by <= 1 or bx + bw >= roi_w - 1 or by + bh >= roi_h - 1
                        if touches_edge:
                            continue
                        if area > cross_area:
                            cross_contour, cross_area = contour, area

                    if cross_contour is not None:
                        moments = cv2.moments(cross_contour)
                        if moments["m00"] > 0:
                            c_x = moments["m10"] / moments["m00"] + x1
                            c_y = moments["m01"] / moments["m00"] + y1
                            # A cruz valida o pouso somente quando sua
                            # intersecao esta na regiao central da caixa YOLO.
                            # Isso rejeita letras/bordas pretas laterais que
                            # podem formar um contorno interno semelhante.
                            box_w = max(1.0, x2 - x1)
                            box_h = max(1.0, y2 - y1)
                            offset_x = abs(c_x - (x1 + x2) / 2.0) / box_w
                            offset_y = abs(c_y - (y1 + y2) / 2.0) / box_h
                            if offset_x <= 0.22 and offset_y <= 0.22:
                                center_x, center_y = float(c_x), float(c_y)
                                cross_found = True
                except Exception as e:
                    # Remove or reduce logging here - only log at debug level
                    self.get_logger().debug(f"Centroid calculation failed: {e}. Falling back to bbox center.")

                # Reject detections with no cross inside. The arena has large
                # light-colored structures (the cluttered_environment near the
                # spawn: floor, roof, walls, window frames) that pass the HSV
                # filter and score high on the YOLO model, producing confident
                # false positives on plain surfaces. Verified in flight: with
                # the camera over bare floor near the origin the node kept
                # reporting a base ~0.5 m below the drone -- the projection was
                # right, there just was no base there. A real landing pad
                # always shows the black cross fully enclosed by the ring, so
                # requiring it separates real pads from blank surfaces without
                # touching the model or the HSV thresholds. Bases accumulate
                # forever in self.arr, so one false positive poisons the map
                # permanently -- worth being strict here.
                if not cross_found:
                    self.get_logger().debug(
                        f"Deteccao descartada (score {score:.2f}): sem cruz -- provavel falso positivo")
                    cv2.rectangle(
                        result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
                    )
                    continue

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
            self.certified_frame_count += 1

        if frame_detections:
            self.detection_count += 1

        # A camera inferior do modelo tem HFOV conhecido de 1.518 rad. O
        # CameraInfo sincronizado pode chegar primeiro de outro publisher com
        # centro/resolucao incompatíveis; isso colocou o alvo roxo à esquerda
        # embora o drone já estivesse fisicamente sobre a base. Deriva os
        # intrínsecos do quadro que estamos realmente processando: centro
        # geométrico exato e pixels quadrados, conforme o SDF da câmera.
        img_h, img_w = img.shape[:2]
        focal = img_w / (2.0 * math.tan(1.518 / 2.0))
        self.camera_intrinsics = {
            'fx': focal,
            'fy': focal,
            'cx': img_w / 2.0,
            'cy': img_h / 2.0,
        }

        current_points = []
        current_body_points = []

        # get 3d points coordinates via real depth from the down camera
        if frame_detections:
            depth_img = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding='32FC1')

        if frame_detections:
            for detection in frame_detections:
                u = int((detection[0] + detection[2]) / 2)
                v = int((detection[1] + detection[3]) / 2)
                depth = self._median_depth(depth_img, u, v)

                point_body = self.get_point_in_base(u, v, depth)
                if point_body is None:
                    continue
                current_body_points.append(point_body)

                # O servo visual nao depende do mapa. Antes, get_points_to_3d
                # vinha primeiro e uma falha momentanea da TF do RTAB-Map
                # eliminava tambem o ponto corporal, embora pixel e depth
                # continuassem validos. Isso fazia todas as descidas abortarem
                # perto de 0.85 m com o detector ainda positivo.
                point_3d = self.get_points_to_3d(u, v, depth)
                if point_3d is None:
                    continue

                current_points.append(point_3d)

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
                    elif self.valid_pos < self.arr.shape[0]:
                        # Nova base
                        self.arr[self.valid_pos] = point_3d
                        self.valid_pos += 1
                    else:
                        # self.arr tem tamanho fixo e nao havia checagem de limite:
                        # ao encher, o proximo indice estourava com IndexError e
                        # derrubava o no inteiro em pleno voo (visto em teste real,
                        # "index 30 is out of bounds"). A arena tem 6 bases, entao
                        # encher os 30 slots significa que algo esta gerando bases
                        # espurias -- tipicamente deriva da localizacao fazendo a
                        # mesma base reaparecer deslocada. Derrubar a deteccao e a
                        # pior resposta possivel: perde-se tudo, inclusive as bases
                        # boas ja mapeadas.
                        self.get_logger().warn(
                            f'Limite de {self.arr.shape[0]} bases atingido -- ignorando '
                            f'deteccao nova. Provavel deriva da localizacao gerando '
                            f'duplicatas.', throttle_duration_sec=5.0)

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


        # Publica inclusive a lista vazia para o consumidor saber que a base
        # saiu do campo de visao, em vez de reutilizar silenciosamente o ultimo
        # quadro positivo.
        current = PoseArray()
        current.header.stamp = self.get_clock().now().to_msg()
        current.header.frame_id = "map"
        for point in current_points:
            pose = Pose()
            pose.position.x = float(point[0])
            pose.position.y = float(point[1])
            pose.position.z = float(point[2])
            pose.orientation.w = 1.0
            current.poses.append(pose)
        self.current_base_publisher.publish(current)

        body_current = PoseArray()
        body_current.header.stamp = self.get_clock().now().to_msg()
        body_current.header.frame_id = "base_link"
        for point in current_body_points:
            pose = Pose()
            pose.position.x = float(point[0])
            pose.position.y = float(point[1])
            pose.position.z = float(point[2])
            pose.orientation.w = 1.0
            body_current.poses.append(pose)
        if current_body_points:
            self.body_frame_count += 1
        self.current_body_base_publisher.publish(body_current)


        # Referencia visual do servo: o pouso so e liberado depois que o ponto
        # vermelho (centro da cruz) permanece dentro deste circulo roxo.
        # Para a mira visual, use explicitamente o centro do próprio quadro.
        # Assim ela nunca depende de CameraInfo ou de ordem de publishers.
        target_u = result.shape[1] // 2
        target_v = result.shape[0] // 2
        cv2.circle(result, (target_u, target_v), 14, (255, 0, 255), 3)
        cv2.circle(result, (target_u, target_v), 2, (255, 0, 255), -1)

        cv2.line(result, (target_u - 24, target_v),
                 (target_u - 16, target_v), (255, 0, 255), 2)
        cv2.line(result, (target_u + 16, target_v),
                 (target_u + 24, target_v), (255, 0, 255), 2)
        cv2.line(result, (target_u, target_v - 24),
                 (target_u, target_v - 16), (255, 0, 255), 2)
        cv2.line(result, (target_u, target_v + 16),
                 (target_u, target_v + 24), (255, 0, 255), 2)

        # O painel Image do RViz e menor que 640x480 e estava exibindo somente
        # um recorte em zoom ~100%, fazendo o centro parecer deslocado. Publica
        # um preview completo 320x240; isto nao altera pixels/intrinsecos usados
        # no controle, apenas a visualizacao.
        debug_preview = cv2.resize(
            result, (320, 240), interpolation=cv2.INTER_AREA)
        debug_image_msg = self.bridge.cv2_to_imgmsg(
            debug_preview, encoding="bgr8")
        debug_image_msg.header = color_data.header
        self.debug_image_publisher.publish(debug_image_msg)

    def _median_depth(self, depth_img, u, v, window=5):
        """Median depth over a small window around (u, v).

        A single-pixel depth read is noisy, especially at oblique angles
        (far/edge-of-frame detections), where each pixel covers more physical
        area. The median over a neighborhood rejects outliers (flying pixels,
        sensor noise) far better than one sample.
        """
        half = window // 2
        height, width = depth_img.shape
        u0, u1 = max(0, u - half), min(width, u + half + 1)
        v0, v1 = max(0, v - half), min(height, v + half + 1)
        patch = depth_img[v0:v1, u0:u1]

        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return float("nan")

        return float(np.median(valid))

    def get_points_to_3d(self, u, v, depth):
        """Back-projects a down-camera pixel to a "map"-frame point using real depth.

        Standard pinhole projection into camera_down_optical_frame, then TF2
        transforms that point through map->base_link->camera_down_optical_frame
        (the static camera TF from ros2_bridge.launch.py, plus map->base_link
        from RTAB-Map/localization). Real depth means this works for both
        ground-level and elevated bases.
        """
        if depth <= 0.0 or np.isnan(depth) or np.isinf(depth):
            self.get_logger().error(f"invalid depth ==> value: {depth}", throttle_duration_sec=2.0)
            return None

        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        Z = float(depth)
        point_camera = PointStamped()
        point_camera.header.frame_id = "camera_down_optical_frame"
        point_camera.point.x = (u - cx) * Z / fx
        point_camera.point.y = (v - cy) * Z / fy
        point_camera.point.z = Z

        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "camera_down_optical_frame", Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF map<-camera_down_optical_frame indisponível: {e}", throttle_duration_sec=2.0)
            return None

        point_map = do_transform_point(point_camera, transform).point

        self.get_logger().debug(
            f"Ponto para o pixel ({u}, {v}) | profundidade: {Z:.3f}m -> "
            f"[X: {point_map.x:.3f}m, Y: {point_map.y:.3f}m, Z: {point_map.z:.3f}m]"
        )

        return (point_map.x, point_map.y, point_map.z)

    def get_point_in_base(self, u, v, depth):
        """Ponto da base no corpo calculado diretamente pelos pixels.

        A montagem conhecida da camera inferior mapeia: cima da imagem para
        frente do drone e esquerda da imagem para esquerda do drone. Fazer
        lookup TF aqui chegou a publicar erro quase zero com a base visivelmente
        no canto do quadro; esta formula evita qualquer TF dinamica/duplicada.
        """
        if depth <= 0.0 or np.isnan(depth) or np.isinf(depth):
            return None
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        body_x = -(v - cy) * depth / fy
        body_y = -(u - cx) * depth / fx
        body_z = -(float(depth) + 0.24)
        return (body_x, body_y, body_z)

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
