"""Ground truth do Gazebo como fonte de localizacao (so simulacao).

Existe porque a odometria visual do RTAB-Map nao sobrevive ao movimento da
varredura nesta arena: perde tracking, reseta pra origem, e cada reset entra no
EKF da PX4 como salto de posicao (medido: 22 resets de pos_ne, estimativa
divergindo pra 10 m fora da arena com o drone parado no ar). Com localizacao
perfeita a camada de missao pode ser validada de ponta a ponta; a odometria
visual vira um problema separado e a missao volta pra ela sem mudar uma linha
-- basta trocar a origem do mesmo topico de odometria.

Le a odometria real publicada pelo gz (bridgeada em /odom_gt), renomeia os
frames pro padrao do repo (odom -> base_link) e:
  * republica em /odom_gt_fixed (consumido pelo ros_odometry_to_vehicle_odometry,
    que converte pra vehicle_visual_odometry da PX4);
  * transmite o TF odom->base_link, que no modo VSLAM seria do rgbd_odometry.

O map->odom (identidade) vem de um static_transform_publisher no launch: com
ground truth a odometria ja e global, entao nao ha correcao de SLAM a aplicar.
"""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from tf2_ros import TransformBroadcaster


class GroundTruthOdometry(Node):
    def __init__(self):
        super().__init__("gt_odometry")

        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", True)

        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.publish_tf = self.get_parameter("publish_tf").value

        # O bridge do gz entrega BEST_EFFORT; casar a QoS ou nada chega.
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(Odometry, "/odom_gt", self._on_odom, qos)
        self.pub = self.create_publisher(Odometry, "/odom_gt_fixed", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.count = 0
        self.create_timer(5.0, self._heartbeat)
        self.get_logger().info(
            "Ground truth odometry ativa: /odom_gt -> /odom_gt_fixed + TF "
            f"{self.odom_frame}->{self.base_frame}")

    def _heartbeat(self):
        if self.count == 0:
            self.get_logger().warn(
                "Nenhuma amostra em /odom_gt -- o bridge do gz esta no ar?")
        self.count = 0

    def _on_odom(self, msg: Odometry):
        self.count += 1

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame
        out.pose = msg.pose
        out.twist = msg.twist
        # Ground truth nao tem incerteza, mas o EKF2 rejeita variancia zero:
        # entrega um piso pequeno e coerente entre posicao e orientacao.
        cov = list(out.pose.covariance)
        for i in (0, 7, 14, 21, 28, 35):
            if cov[i] <= 0.0:
                cov[i] = 1e-4
        out.pose.covariance = cov
        self.pub.publish(out)

        if self.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = msg.header.stamp
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = msg.pose.pose.position.x
            tf.transform.translation.y = msg.pose.pose.position.y
            tf.transform.translation.z = msg.pose.pose.position.z
            tf.transform.rotation = msg.pose.pose.orientation
            self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
