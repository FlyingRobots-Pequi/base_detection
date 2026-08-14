"""Primitivas de voo por cima do offboard_control (rtabmap_drone_example), sem tocar nele.

O offboard_control.cpp ja expoe tudo que a missao precisa via /cmd_vel + /joy:

- ARMAR:    Twist com linear.z < -0.4 e angular.z < -0.4 sustentado ~1s
            (update_state, offboard_control.cpp:165-169). O arm() decola sozinho
            para 2 m acima da posicao atual.
- DESARMAR: Twist com linear.z < -0.4 e angular.z > +0.4 (linhas 170-175).
- VELOCIDADE XY/YAW: Twist continuo a >=10 Hz (o node troca para velocity control
            5 s depois de armar; parar de publicar por >0.1 s faz ele LATCHAR a
            pose atual como goal de posicao -- e o nosso "hold").
- VELOCIDADE Z: so vale com /joy buttons[10] == 1 (velocity2d_, linhas 302-311).
            Convencao ENU do Twist: linear.z > 0 sobe (o node converte para NED).

Cuidado com o teleop_twist_joy: ele tambem assina /joy, mas so publica /cmd_vel
com o enable_button 9 pressionado (joy_config.yaml). Nunca setar buttons[9] aqui.

Uso tick-based: o chamador (FSM da missao) chama um publish_* por tick a >=10 Hz.
Nao publicar nada = "hold" (latch de posicao do offboard_control).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from px4_msgs.msg import VehicleLandDetected, VehicleLocalPosition, VehicleStatus

# Indices no /joy (ver offboard_control.cpp e joy_config.yaml)
UNLOCK_VZ_BUTTON = 10   # buttons[10]=1 destrava velocidade Z no offboard_control
TELEOP_ENABLE_BUTTON = 9  # NUNCA setar: e o enable do teleop_twist_joy
JOY_NUM_BUTTONS = 12
JOY_NUM_AXES = 8

# Magnitude dos gestos: precisa passar do limiar 0.4 do offboard_control
GESTURE_MAG = 0.8


class PX4Gestures:
    """Publica /cmd_vel + /joy e observa o estado da PX4 (fmu/out/*).

    Anexa publishers/subscriptions ao node dado. Os topicos fmu/* sao RELATIVOS,
    entao o node deve rodar sob o namespace da PX4 (ex.: -r __ns:=/pequi/hermit),
    igual ao offboard_control. /cmd_vel e /joy sao absolutos (globais).
    """

    def __init__(self, node: Node):
        self._node = node

        self.cmd_vel_pub = node.create_publisher(Twist, "/cmd_vel", 10)
        self.joy_pub = node.create_publisher(Joy, "/joy", 10)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.status = VehicleStatus()
        self.land_detected = VehicleLandDetected()
        self.local_position = VehicleLocalPosition()
        self._have_status = False
        self._have_local_position = False

        node.create_subscription(
            VehicleStatus, "fmu/out/vehicle_status", self._on_status, qos)
        node.create_subscription(
            VehicleLandDetected, "fmu/out/vehicle_land_detected", self._on_land, qos)
        node.create_subscription(
            VehicleLocalPosition, "fmu/out/vehicle_local_position", self._on_lpos, qos)

    # ---- estado observado -------------------------------------------------

    def _on_status(self, msg):
        self.status = msg
        self._have_status = True

    def _on_land(self, msg):
        self.land_detected = msg

    def _on_lpos(self, msg):
        self.local_position = msg
        self._have_local_position = True

    @property
    def ready(self) -> bool:
        return self._have_status and self._have_local_position

    @property
    def armed(self) -> bool:
        return self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED

    @property
    def offboard(self) -> bool:
        return self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    @property
    def landed(self) -> bool:
        return bool(self.land_detected.landed)

    @property
    def altitude(self) -> float:
        """Altitude acima da origem local (m, positivo para cima). NED: -z."""
        return -self.local_position.z

    # ---- primitivas (1 chamada = 1 tick; chamar a >=10 Hz) ----------------

    def publish_arm_gesture(self):
        """Gesto de armar (stick esquerdo baixo-direita). Repetir por >1 s."""
        self._publish_twist(lz=-GESTURE_MAG, az=-GESTURE_MAG)

    def publish_disarm_gesture(self):
        """Gesto de desarmar (stick esquerdo baixo-esquerda). Repetir por >1 s."""
        self._publish_twist(lz=-GESTURE_MAG, az=+GESTURE_MAG)

    def publish_velocity(self, vx=0.0, vy=0.0, vz=None, yaw_rate=0.0):
        """Comanda velocidade no frame do corpo (ENU do Twist).

        vx frente, vy esquerda, vz para cima (None = manter altitude travada),
        yaw_rate anti-horario. Com vz != None, publica junto o /joy com o botao
        de unlock para o offboard_control aceitar o eixo Z.
        Gestos exigem |linear.z| > 0.4 E |angular.z| > 0.4 simultaneos; para nao
        disparar por acidente, o par (vz, yaw_rate) e saturado fora dessa zona.
        """
        lz = 0.0 if vz is None else vz
        az = yaw_rate
        if abs(lz) > 0.4 and abs(az) > 0.4:
            az = 0.39 if az > 0 else -0.39  # nunca casar um gesto sem querer
        # O unlock e PEGAJOSO no offboard_control (velocity2d_ so muda quando
        # chega um /joy). Sem republicar com unlock=False ao voltar pro modo 2D,
        # o eixo Z ficaria solto em vz=0, sem o latch de posicao nem o P de
        # altitude -- e o drone afundaria devagar durante a navegacao.
        self._publish_joy(unlock_vz=vz is not None)
        self._publish_twist(lx=vx, ly=vy, lz=lz, az=az)

    def hold(self):
        """Nao publica nada: apos 0.1 s o offboard_control latcha a pose atual."""
        pass

    def lock_vz(self):
        """Devolve o eixo Z ao controle de POSICAO do offboard_control.

        Obrigatorio antes de entregar o /cmd_vel ao Nav2: o unlock e pegajoso
        (o offboard_control so reavalia quando chega um /joy). Vindo de uma fase
        vertical, o Z continuaria em controle de VELOCIDADE com vz=0 -- sem
        latch de altitude, o drone afunda ate o chao enquanto o Nav2 navega.
        """
        self._publish_joy(unlock_vz=False)

    # ---- helpers ----------------------------------------------------------

    def _publish_twist(self, lx=0.0, ly=0.0, lz=0.0, az=0.0):
        msg = Twist()
        msg.linear.x = float(lx)
        msg.linear.y = float(ly)
        msg.linear.z = float(lz)
        msg.angular.z = float(az)
        self.cmd_vel_pub.publish(msg)

    def _publish_joy(self, unlock_vz=False):
        msg = Joy()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.axes = [0.0] * JOY_NUM_AXES
        msg.buttons = [0] * JOY_NUM_BUTTONS
        if unlock_vz:
            msg.buttons[UNLOCK_VZ_BUTTON] = 1
        self.joy_pub.publish(msg)
