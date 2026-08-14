"""Missao autonoma: sobrevoar a arena, achar as bases e pousar em todas (so RGB).

Substitui o clique de "2D Goal Pose" do RViz por uma FSM que orquestra o que ja
existe -- Nav2 (XY), offboard_control (PX4) e base_detection (percepcao):

  BOOT -> SEARCH -> [por base] GOTO -> ALIGN -> DESCEND -> CONFIRM
                         ^                                    |
                         +------- TAKEOFF <---- DISARM <------+
                                     |
                             (todas visitadas) -> RETURN (pousa na origem) -> END

Regra de ouro: /cmd_vel tem UM dono por vez. Na SEARCH/GOTO o dono e o Nav2
(via velocity_smoother); nas fases verticais (ALIGN/DESCEND) esta FSM cancela o
goal do Nav2 e assume com as primitivas de px4_gestures. Nunca os dois juntos.

Altitude: o Nav2 e 2D; o Z fica travado pelo latch de posicao do
offboard_control e so muda nas fases verticais (unlock via /joy, ver
px4_gestures.py). DESCEND publica vz<0 continuamente; pausar a descida e
publicar vz=0.0 (NAO None: sem o unlock o offboard_control voltaria a puxar o
drone para o current_goal_.z antigo).

Rodar sob o namespace da PX4 (fmu/* relativos), acoes do Nav2 sao absolutas:
  ros2 run base_detection mission_control --ros-args -r __ns:=/pequi/hermit \
      -p use_sim_time:=true
"""

import math

import numpy as np
import rclpy
import tf2_ros
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped
from nav2_msgs.action import FollowWaypoints, NavigateToPose

from base_detection.px4_gestures import PX4Gestures

TICK_HZ = 20.0


class MissionControl(Node):
    # Fases do retorno e do pouso final: imunes ao prazo global de missao (que
    # manda voltar para casa), senao o proprio retorno seria abortado no meio.
    RETURN_STATES = ("RETURN", "RETURN_GOTO", "ALIGN", "DESCEND", "CONFIRM",
                     "DISARM", "PAUSE", "LATCH_ALT", "RECOVER_ALT", "RECENTER",
                     "END")

    def __init__(self):
        super().__init__("mission_control")

        # -- parametros ------------------------------------------------------
        # Area de varredura (boustrophedon). Cobre o retangulo das bases da
        # arena default (x 0.6..5.1, y 2.0..5.4) com folga MINIMA e SEM
        # encostar na borda sul: o primeiro row em y=0 levava o drone por cima
        # do cluttered_environment (1.07,-0.69) -- features rasas correndo
        # rapido na imagem, odometria visual morreu e o drone caiu (validado
        # na pratica). A camera frontal ve a base de y>=1.5 de qualquer jeito
        # (FOV 1.518 a ~2.8 m de altura).
        # Area dimensionada para conter as SEIS bases da arena (medidas no gz):
        #   ground_land_base_1 (4.29, 2.00)   ground_land_base_2 (0.66, 4.66)
        #   ground_land_base_3 (5.13, 5.41)   ground_land_base_4 (2.77, 3.18)
        #   high_base_1      (-0.23, 2.98)    high_base_2      (3.17, 4.46)
        # (as elevadas sao links do arena_spawn, que fica em (2.571, 3.980)
        # girado -90 deg -- world = origem + (y_link, -x_link))
        # Uma versao anterior ia so de x 0.5..4.6 e deixava base_3 e high_base_1
        # FORA da varredura: a missao achava 3 bases + o pad de decolagem.
        # As colunas caem em x = -0.3, 1.5, 3.3, 5.1 e cada uma mapeia +-1 m
        # (nadir_gate_m), entao a cobertura e continua e toca as seis.
        self.declare_parameter("search_x_min", -0.3)
        self.declare_parameter("search_x_max", 5.3)
        self.declare_parameter("search_y_min", 1.7)
        self.declare_parameter("search_y_max", 5.7)
        self.declare_parameter("search_row_step", 1.8)

        # Geofence = paredes da arena com folga. A arena (arena_spawn em
        # (2.5707, 3.9802) girada -90 deg) ocupa X [-1.13, 6.63], Y [-0.91,
        # 6.86] no mundo. A folga cobre o descasamento entre posicao ESTIMADA e
        # REAL: com os resets de pos_ne do EKF o drone chegou a (4.49, 6.53) --
        # 33 cm da parede norte -- achando estar dentro da rota. Ao furar a
        # cerca a missao aborta o Nav2 e recua para o centro.
        # Folga de 0.5 m das paredes -- e a POSICAO REAL que importa aqui.
        # (Ja foi fence_y_min=-0.2 com o pad de decolagem em y=0: a cerca
        # disparava com o drone parado em casa, so pela deriva normal.)
        self.declare_parameter("fence_x_min", -0.6)
        self.declare_parameter("fence_x_max", 6.1)
        self.declare_parameter("fence_y_min", -0.4)
        self.declare_parameter("fence_y_max", 6.3)
        self.declare_parameter("fence_center", [2.5, 3.0])
        self.declare_parameter("skip_search", False)   # ir direto as bases ja mapeadas
        # Altitude de varredura. A decolagem do offboard_control sobe 2 m, o que
        # deixava a busca a ~1.5-2 m: campo de visao da camera de baixo estreito
        # (~2.8 m a 1.5 m de altura) para linhas separadas por row_step, e a
        # varredura inteira mapeou ZERO bases. A 3.2 m a faixa vista passa de
        # 6 m e cobre a folga entre as colunas.
        # Altitudes medidas pelo TF do mapa (z de base_link em "map"), nao pelo
        # EKF: a origem do EKF e redefinida a cada rearme. Referencia: o drone
        # pousado no chao marca ~0.34 m (offset do corpo no modelo).
        self.declare_parameter("search_altitude", 3.2)
        self.declare_parameter("min_altitude", 1.5)
        # True: pousa assim que achar base nova e RETOMA a varredura de onde
        # parou (o que a equipe pediu). False: varre tudo primeiro e so depois
        # visita as bases mapeadas.
        self.declare_parameter("land_during_search", True)
        self.declare_parameter("align_gain", 2.0)      # P do servo visual (por erro normalizado)
        self.declare_parameter("align_max_vel", 0.35)  # m/s teto do servo
        self.declare_parameter("align_tol", 0.06)      # norma p/ considerar alinhado
        self.declare_parameter("descend_speed", 0.3)   # m/s (< 0.4: nunca vira gesto)
        self.declare_parameter("descend_abort_norm", 0.35)  # desalinhou demais: pausa vz
        self.declare_parameter("visited_radius", 0.8)  # base a menos disto de uma visitada = mesma
        self.declare_parameter("home_xy", [0.0, 0.0])
        # Voltar para a base inicial e OBRIGATORIO, entao ha dois prazos:
        #  - tf_wait_timeout: PICK_BASE sem TF por tanto tempo -> volta em vez
        #    de ficar plantado no ar esperando (ja aconteceu).
        #  - mission_timeout: teto da missao inteira. Estourou, para de caçar
        #    base e vai para casa com bateria/tempo de sobra.
        self.declare_parameter("tf_wait_timeout", 25.0)
        self.declare_parameter("mission_timeout", 900.0)

        p = self.get_parameter
        self.align_gain = p("align_gain").value
        self.align_max_vel = p("align_max_vel").value
        self.align_tol = p("align_tol").value
        self.descend_speed = p("descend_speed").value
        self.descend_abort_norm = p("descend_abort_norm").value
        self.visited_radius = p("visited_radius").value
        self.home_xy = list(p("home_xy").value)

        # -- estado ----------------------------------------------------------
        self.gestures = PX4Gestures(self)
        self.state = "BOOT"
        self.state_start = None
        self.bases = []           # [(x, y, z)] vindos de /base_detection/bases
        self.visited = []         # [(x, y)] ja pousadas (inclui descartes)
        self.target = None        # (x, y) da base corrente
        self.pixel_error = None   # (ex, ey, norma) mais recente
        self.pixel_error_stamp = None
        self.align_ok_ticks = 0
        self.touch_altitude = None
        self.goal_handle = None   # handle da acao Nav2 corrente
        self.goal_done = False
        self.goal_result_ok = False
        self._pending_goal_future = None
        self._goal_retry = None       # (client, goal, feedback) p/ reenviar apos rejeicao
        self._goal_retry_at = None
        self._search_waypoints = []   # rota completa da varredura
        self._search_from = 0         # indice do 1o waypoint do goal corrente
        self._search_index = 0        # progresso DENTRO do goal corrente
        self._resume_search_at = None # onde retomar apos um pouso incremental
        self._mission_start = None    # marcado quando o relogio de sim comeca

        # -- IO ---------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(PoseArray, "/base_detection/bases",
                                 self._on_bases, 10)
        self.create_subscription(PointStamped, "/base_detection/target_pixel_error",
                                 self._on_pixel_error, 10)
        self.height_update_pub = self.create_publisher(
            PointStamped, "/base_detection/base_height_update", 10)

        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.waypoints_client = ActionClient(self, FollowWaypoints, "/follow_waypoints")

        self.timer = self.create_timer(1.0 / TICK_HZ, self.tick)
        self.get_logger().info("Mission control iniciado")

    # ---- subscriptions -----------------------------------------------------

    def _on_bases(self, msg: PoseArray):
        self.bases = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]

    def _on_pixel_error(self, msg: PointStamped):
        self.pixel_error = (msg.point.x, msg.point.y, msg.point.z)
        self.pixel_error_stamp = Time.from_msg(msg.header.stamp)

    def map_height(self):
        """Altura do drone no frame "map" (metros), via TF, ou None.

        Preferir SEMPRE isto a `gestures.altitude` para limiares de voo: a
        altitude do EKF e relativa a uma origem que a PX4 REDEFINE a cada
        rearme -- depois do primeiro pouso ela vinha negativa (-1.28 m com o
        drone no ar) e as guardas disparavam sem sentido. O frame "map" e o
        mesmo do inicio ao fim da missao.
        """
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_link", Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        return tf.transform.translation.z

    def map_pose(self):
        """Pose do drone no frame "map" (ENU), via TF: (x, y, yaw) ou None.

        NAO usar local_position da PX4 para isso: ela e NED (x=Norte, y=Leste)
        enquanto bases, waypoints e cerca vivem no ENU do mapa (x=Leste,
        y=Norte). Misturar os dois trocava os eixos -- a cerca acusava violacao
        com o drone em posicao legal e o recuo "ao centro" saia na diagonal
        errada, ate pousar em cima do cenario.
        """
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_link", Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.x, t.y, yaw)

    def pixel_error_fresh(self) -> bool:
        if self.pixel_error is None or self.pixel_error_stamp is None:
            return False
        age = (self.get_clock().now() - self.pixel_error_stamp).nanoseconds / 1e9
        return age < 1.0

    # ---- Nav2 helpers ------------------------------------------------------

    def _on_waypoint_feedback(self, feedback):
        """Guarda em que waypoint a varredura esta, para poder retomar dali."""
        self._search_index = feedback.feedback.current_waypoint

    def _send_goal(self, client, goal_msg, feedback_cb=None):
        """Dispara uma acao Nav2; goal_done/goal_result_ok sao setados no fim.

        Rejeicao NAO marca goal_done: guarda o par (client, goal) e o tick
        reenvia com backoff (o bt_navigator rejeita enquanto o lifecycle ainda
        esta ativando -- visto na pratica logo apos subir a stack)."""
        self.goal_handle = None
        self.goal_done = False
        self.goal_result_ok = False
        self._goal_retry = (client, goal_msg, feedback_cb)
        self._goal_retry_at = None
        future = client.send_goal_async(goal_msg, feedback_callback=feedback_cb)
        future.add_done_callback(self._on_goal_response)
        self._pending_goal_future = future

    def _on_goal_response(self, future):
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().warn("Goal Nav2 rejeitado; reenviando em 2 s")
            self._goal_retry_at = self.get_clock().now() + Duration(seconds=2.0)
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_goal_result)

    def _maybe_retry_goal(self):
        if (self._goal_retry_at is not None and self._goal_retry is not None
                and self.get_clock().now() >= self._goal_retry_at):
            client, goal_msg, feedback_cb = self._goal_retry
            self._goal_retry_at = None
            future = client.send_goal_async(goal_msg, feedback_callback=feedback_cb)
            future.add_done_callback(self._on_goal_response)

    def _on_goal_result(self, future):
        result = future.result()
        # status 4 = SUCCEEDED (action_msgs/GoalStatus)
        self.goal_result_ok = (result is not None and result.status == 4)
        self.goal_done = True

    def cancel_nav_goal(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None

    def make_pose(self, x, y, yaw=math.pi / 2):
        # yaw default = pi/2 (Norte, o heading de spawn): a odometria visual
        # depende da camera frontal ver o conteudo texturizado ao Norte. O DWB
        # esta com rotacao congelada (nav2_params), entao isto e so o goal
        # nominal -- mas mantem os checkers e o RViz coerentes.
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    def search_waypoints(self):
        """Boustrophedon em COLUNAS (pernas longas ao longo de Y, Norte-Sul).

        Com o yaw congelado pro Norte, perna longa em Y = movimento
        frontal/traseiro -- o caso amigavel pra odometria visual F2M (features
        aproximando/afastando com alta sobreposicao). Pernas longas em X eram
        strafe lateral sustentado e derrubavam o tracking (medido: registro
        falhando ate a EKF resetar 22x e o drone cair). O passo lateral fica
        restrito ao hop curto de row_step entre colunas."""
        p = self.get_parameter
        x_min, x_max = p("search_x_min").value, p("search_x_max").value
        y_min, y_max = p("search_y_min").value, p("search_y_max").value
        step = p("search_row_step").value
        poses = []
        x = x_min
        south_to_north = True
        while x <= x_max + 1e-6:
            ys = (y_min, y_max) if south_to_north else (y_max, y_min)
            poses.append(self.make_pose(x, ys[0]))
            poses.append(self.make_pose(x, ys[1]))
            south_to_north = not south_to_north
            x += step
        return poses

    # ---- selecao de bases --------------------------------------------------

    def next_base(self):
        """Base mapeada mais proxima ainda nao visitada (ou None).

        Distancias no frame do MAPA: as bases vem de /base_detection/bases em
        "map"; usar a local_position (NED) da PX4 aqui compararia eixos
        trocados e elegeria a base errada."""
        pose = self.map_pose()
        if pose is None:
            return None
        px, py, _ = pose
        best = None
        for (bx, by, _bz) in self.bases:
            # o pad de decolagem aparece no mapa como base em ~home; quem pousa
            # nele e o RETURN no fim, nao o laco de bases
            if np.hypot(bx - self.home_xy[0], by - self.home_xy[1]) < self.visited_radius:
                continue
            if any(np.hypot(bx - vx, by - vy) < self.visited_radius
                   for (vx, vy) in self.visited):
                continue
            d = np.hypot(bx - px, by - py)
            if best is None or d < best[0]:
                best = (d, bx, by)
        return None if best is None else (best[1], best[2])

    # ---- servo visual ------------------------------------------------------

    def servo_velocity(self):
        """Erro de pixel (frame optico) -> velocidade XY no frame do corpo.

        Rotaciona o vetor de erro para base_link via TF (so rotacao), evitando
        hardcode de convencao optica. O erro aponta DO eixo otico PARA a base,
        entao a velocidade e +ganho*erro (ir na direcao da base).
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", "camera_down_optical_frame", Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        q = tf.transform.rotation
        # matriz de rotacao a partir do quaternion (so os dois primeiros eixos)
        x, y, z, w = q.x, q.y, q.z, q.w
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - z * w)
        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x * x + z * z)
        # (a linha r2x seria o componente vertical; Z e controlado a parte)
        ex, ey, _ = self.pixel_error
        vx = r00 * ex + r01 * ey
        vy = r10 * ex + r11 * ey
        v = np.array([vx, vy]) * self.align_gain
        n = np.linalg.norm(v)
        if n > self.align_max_vel:
            v *= self.align_max_vel / n
        return float(v[0]), float(v[1])

    # ---- FSM ---------------------------------------------------------------

    def elapsed(self):
        return (self.get_clock().now() - self.state_start).nanoseconds / 1e9

    def goto_state(self, state):
        g = self.gestures
        self.get_logger().info(
            f"[{self.state} -> {state}] alt={g.altitude:.2f} armed={g.armed} "
            f"bases={len(self.bases)} visitadas={len(self.visited)}")
        self.state = state
        self.state_start = self.get_clock().now()
        self.align_ok_ticks = 0

    def tick(self):
        g = self.gestures

        if self.state_start is None:
            if self.get_clock().now().nanoseconds == 0:
                return  # use_sim_time: esperar o /clock
            self.state_start = self.get_clock().now()
            self._mission_start = self.state_start

        self._maybe_retry_goal()

        # Prazo global: passou do teto e ainda esta caçando base -> vai para
        # casa. So as fases do retorno e do pouso final escapam, senao o drone
        # abortaria a propria aterrissagem.
        if (self._mission_start is not None
                and self.state not in self.RETURN_STATES
                and (self.get_clock().now() - self._mission_start).nanoseconds / 1e9
                    > self.get_parameter("mission_timeout").value):
            self.get_logger().error(
                "Tempo de missao esgotado: voltando para a base inicial")
            self.cancel_nav_goal()
            self.goto_state("RETURN")
            return

        # Nas fases dirigidas pelo Nav2 o eixo Z tem que estar no controle de
        # POSICAO do offboard_control. O unlock e pegajoso: sem reafirmar o
        # lock aqui, o Z ficava em velocidade com vz=0 depois de uma fase
        # vertical e o drone afundava ate a guarda disparar -- em loop.
        if self.state in ("SEARCH", "GOTO", "RETURN_GOTO", "PICK_BASE"):
            g.lock_vz()

        handler = getattr(self, f"state_{self.state.lower()}", None)
        if handler is None:
            self.get_logger().error(f"Estado desconhecido: {self.state}")
            self.timer.cancel()
            return
        handler(g)

    def state_boot(self, g):
        # offboard_control arma e decola sozinho; esperar o hover estabilizar
        # E os action servers do Nav2 ativarem (lifecycle demora; goal mandado
        # antes disso e rejeitado na cara).
        if not (self.nav_client.server_is_ready()
                and self.waypoints_client.server_is_ready()):
            if self.elapsed() > 20.0:
                self.get_logger().warn("Aguardando action servers do Nav2...",
                                       throttle_duration_sec=10.0)
            return
        if g.ready and g.armed and g.offboard and g.altitude > 1.5:
            if self.elapsed() > 4.0:
                if self.get_parameter("skip_search").value:
                    self.goto_state("PICK_BASE")
                else:
                    self.goto_state("CLIMB")

    def state_climb(self, g):
        """Sobe ate a altitude de varredura antes de entregar o Nav2."""
        target = self.get_parameter("search_altitude").value
        h = self.map_height()
        if h is not None and h < target - 0.2 and self.elapsed() < 40.0:
            g.publish_velocity(0.0, 0.0, vz=+0.5)
            return
        self._resume_state = "SEARCH"
        self._search_index = 0
        self._search_from = 0
        self.goto_state("LATCH_ALT")

    def _altitude_guard(self, g, resume_state):
        """Guardas de voo durante navegacao: altitude minima e geofence.

        Retorna True se assumiu o controle (o estado chamador deve retornar)."""
        h = self.map_height()
        if h is not None and h < self.get_parameter("min_altitude").value:
            self.get_logger().warn(
                f"Altura {h:.2f} m (TF map) durante {self.state}: recuperando")
            self.cancel_nav_goal()
            self._resume_state = resume_state
            self.goto_state("RECOVER_ALT")
            return True

        pose = self.map_pose()
        if pose is None:
            return False  # sem TF nao da para julgar; a guarda de altitude ja cobre
        p = self.get_parameter
        x, y, _ = pose
        if not (p("fence_x_min").value <= x <= p("fence_x_max").value
                and p("fence_y_min").value <= y <= p("fence_y_max").value):
            self.get_logger().error(
                f"GEOFENCE furada em ({x:.2f}, {y:.2f}) durante {self.state}: "
                "abortando e recuando ao centro")
            self.cancel_nav_goal()
            self._resume_state = resume_state
            self.goto_state("RECENTER")
            return True
        return False

    def state_recenter(self, g):
        """Recuo cego para o centro da arena, sem Nav2 (que pode estar com a
        pose errada). Velocidade direta no frame do mapa, ate voltar pra dentro
        da cerca com folga."""
        pose = self.map_pose()
        if pose is None:
            g.publish_velocity(0.0, 0.0)
            return
        p = self.get_parameter
        cx, cy = p("fence_center").value
        x, y, yaw = pose
        dx = cx - x
        dy = cy - y
        dist = math.hypot(dx, dy)
        if dist < 0.8 or self.elapsed() > 40.0:
            resume = getattr(self, "_resume_state", "PICK_BASE")
            if resume == "SEARCH":
                # retoma de onde parou, nao do inicio
                self.start_search(self._search_from + self._search_index)
            elif resume in ("GOTO", "RETURN_GOTO") and self.target is not None:
                self._send_goal(self.nav_client,
                                NavigateToPose.Goal(pose=self.make_pose(*self.target)))
            self.goto_state(resume)
            return
        # /cmd_vel e no frame do CORPO (FLU): rotaciona o erro do mapa pelo yaw
        # REAL do drone, em vez de assumir que ele continua apontado ao Norte.
        speed = 0.3
        ux, uy = dx / dist, dy / dist
        vx_body = (ux * math.cos(yaw) + uy * math.sin(yaw)) * speed
        vy_body = (-ux * math.sin(yaw) + uy * math.cos(yaw)) * speed
        g.publish_velocity(vx_body, vy_body)

    def state_recover_alt(self, g):
        h = self.map_height()
        if h is not None and h < self.get_parameter("search_altitude").value - 0.3:
            # vz do publish_velocity e ENU do Twist: POSITIVO sobe. (Ja foi -0.3
            # aqui por engano: a guarda de altitude empurrava o drone contra o
            # chao -- altitude baixa disparava a guarda, que descia mais.)
            g.publish_velocity(0.0, 0.0, vz=+0.3)
            return
        self.goto_state("LATCH_ALT")

    def state_latch_alt(self, g):
        """Silencio deliberado para o offboard_control gravar a nova altitude.

        Ele so atualiza o alvo de posicao quando o /cmd_vel fica velho (>0.1 s)
        E o eixo Z ainda esta destravado. Sem esta pausa, a subida do CLIMB/
        RECOVER_ALT era descartada: ao devolver o controle ao Nav2 o drone
        voltava ao alvo de z anterior, a guarda disparava de novo e virava um
        sobe-desce infinito.
        """
        if self.elapsed() < 0.8:
            return  # nao publicar nada: e a pausa que dispara o latch
        resume = getattr(self, "_resume_state", "PICK_BASE")
        if resume == "SEARCH":
            self.start_search(self._search_from + self._search_index)
        elif resume in ("GOTO", "RETURN_GOTO") and self.target is not None:
            self._send_goal(self.nav_client,
                            NavigateToPose.Goal(pose=self.make_pose(*self.target)))
        self.goto_state(resume)

    def start_search(self, from_index=0):
        """Dispara a varredura a partir de um waypoint (0 = do inicio)."""
        if not self._search_waypoints:
            self._search_waypoints = self.search_waypoints()
        self._search_from = from_index
        self._search_index = 0
        remaining = self._search_waypoints[from_index:]
        if not remaining:
            return False
        self._send_goal(self.waypoints_client,
                        FollowWaypoints.Goal(poses=remaining),
                        feedback_cb=self._on_waypoint_feedback)
        return True

    def state_search(self, g):
        # Nav2 e o dono do /cmd_vel; base_detection acumula em background
        if self._altitude_guard(g, "SEARCH"):
            return

        # Pouso INCREMENTAL: achou base nova -> interrompe a varredura, pousa
        # nela e retoma dali. O indice do waypoint corrente vem do feedback do
        # FollowWaypoints, entao a retomada nao refaz o trecho ja varrido.
        if self.get_parameter("land_during_search").value:
            target = self.next_base()
            if target is not None:
                self._resume_search_at = self._search_from + self._search_index
                self.get_logger().info(
                    f"Base nova em ({target[0]:.2f}, {target[1]:.2f}) durante a "
                    f"varredura: pousando antes de retomar (waypoint "
                    f"{self._resume_search_at})")
                self.cancel_nav_goal()
                self.target = target
                self._send_goal(self.nav_client,
                                NavigateToPose.Goal(pose=self.make_pose(*target)))
                self.goto_state("GOTO")
                return

        if self.goal_done:
            self._resume_search_at = None  # varredura terminou de fato
            self.goto_state("PICK_BASE")

    def state_pick_base(self, g):
        if self.map_pose() is None:
            # Sem TF nao da para medir distancia ate as bases. Esperar, NAO
            # concluir: tratar isso como "acabaram as bases" mandava a missao
            # direto pro RETURN com o mapa cheio (visto na pratica: 4 bases
            # mapeadas, nenhuma visitada). Mas esperar PARA SEMPRE deixava o
            # drone plantado no ar quando o TF nao voltava -- entao, passado o
            # limite, volta para casa em vez de ficar preso.
            if self.elapsed() > self.get_parameter("tf_wait_timeout").value:
                self.get_logger().error(
                    "PICK_BASE sem TF por tempo demais: voltando para a base inicial")
                self.goto_state("RETURN")
                return
            self.get_logger().warn("PICK_BASE aguardando TF map->base_link...",
                                   throttle_duration_sec=5.0)
            return
        target = self.next_base()
        if target is None:
            self.goto_state("RETURN")
            return
        self.target = target
        self._send_goal(self.nav_client,
                        NavigateToPose.Goal(pose=self.make_pose(*target)))
        self.goto_state("GOTO")

    def state_goto(self, g):
        if self._altitude_guard(g, "GOTO"):
            return
        if self.goal_done:
            # sucesso ou nao, se ha deteccao fresca da para alinhar; senao pula
            if self.pixel_error_fresh() or self.goal_result_ok:
                self.goto_state("ALIGN")
            else:
                self.get_logger().warn(
                    f"GOTO falhou sem deteccao em {self.target}; descartando base")
                self.visited.append(self.target)
                self.goto_state("PICK_BASE")

    def state_align(self, g):
        if self.elapsed() > 45.0:
            if self.target == tuple(self.home_xy):
                # em casa nao ha o que descartar: o Nav2 ja nos deixou em cima
                # da origem; desce as cegas (pouso final, precisao secundaria)
                self.get_logger().warn("ALIGN timeout em casa; descendo as cegas")
                self.goto_state("DESCEND")
            else:
                self.get_logger().warn("ALIGN timeout; descartando base")
                self.visited.append(self.target)
                self.goto_state("PICK_BASE")
            return
        if not self.pixel_error_fresh():
            g.publish_velocity(0.0, 0.0)  # hover ativo enquanto espera deteccao
            return
        v = self.servo_velocity()
        if v is None:
            g.publish_velocity(0.0, 0.0)
            return
        g.publish_velocity(v[0], v[1])
        if self.pixel_error[2] < self.align_tol:
            self.align_ok_ticks += 1
            if self.align_ok_ticks > TICK_HZ:  # 1 s estavel no nadir
                self.goto_state("DESCEND")
        else:
            self.align_ok_ticks = 0

    def state_descend(self, g):
        if g.landed:
            # altura do TOQUE medida no frame do mapa: e ela que vira a
            # altura real da base publicada para a percepcao, e a
            # referencia da decolagem seguinte.
            self.touch_altitude = self.map_height()
            if self.touch_altitude is None:
                self.touch_altitude = 0.34  # chao, se o TF falhar no instante
            self.goto_state("CONFIRM")
            return
        v = self.servo_velocity() if self.pixel_error_fresh() else None
        if v is None:
            # perdeu a base de vista (FOV encolhe perto do chao): desce reto.
            # Ja estavamos alinhados no nadir; a inercia lateral e ~zero.
            g.publish_velocity(0.0, 0.0, vz=-self.descend_speed)
        elif self.pixel_error[2] > self.descend_abort_norm:
            # desalinhou demais: pausa a descida (vz=0.0 mantem o unlock!)
            g.publish_velocity(v[0], v[1], vz=0.0)
        else:
            g.publish_velocity(v[0], v[1], vz=-self.descend_speed)

    def state_confirm(self, g):
        # anuncia a altura real da base para corrigir o mapa (Z=0 assumido)
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.point.x = float(self.target[0])
        msg.point.y = float(self.target[1])
        msg.point.z = float(self.touch_altitude)
        self.height_update_pub.publish(msg)
        self.visited.append(self.target)
        self.get_logger().info(
            f"POUSO CONFIRMADO em {self.target} (altura {self.touch_altitude:.2f} m)")
        self.goto_state("DISARM")

    def state_disarm(self, g):
        g.publish_disarm_gesture()
        if not g.armed:
            self.goto_state("PAUSE")

    def state_pause(self, g):
        if self.elapsed() > 3.0:
            if self.state_after_pause() == "END":
                self.goto_state("END")
            else:
                self.goto_state("REARM")

    def state_after_pause(self):
        """END se acabou de pousar em casa (RETURN concluido); senao rearma."""
        if self.target == tuple(self.home_xy) and len(self.visited) > 0:
            return "END"
        return "REARM"

    def state_rearm(self, g):
        g.publish_arm_gesture()
        if g.armed:
            self.goto_state("WAIT_TAKEOFF")

    def state_wait_takeoff(self, g):
        # Retoma a varredura interrompida por um pouso incremental, do waypoint
        # onde ela parou; so quando nao ha mais varredura pendente e que o laco
        # cai no PICK_BASE (bases mapeadas mas nao visitadas) e depois RETURN.
        h = self.map_height()
        if (self._resume_search_at is not None and self.touch_altitude is not None
                and h is not None and h > self.touch_altitude + 1.8):
            resume_at = self._resume_search_at
            self._resume_search_at = None
            if self.start_search(resume_at):
                self.goto_state("SEARCH")
                return
            self.goto_state("PICK_BASE")
            return
        return self._state_wait_takeoff_plain(g)

    def _state_wait_takeoff_plain(self, g):
        # arm() do offboard_control decola 2 m acima da posicao atual; esperar
        # QUASE o topo (+1.8): sair cedo (+1.0) deixava o ALIGN seguinte a
        # ~1.8 m do chao, perto demais -- a base ocupa a imagem inteira e o
        # YOLO para de detectar (visto na pratica: ALIGN timeout apos pouso).
        h = self.map_height()
        if (self.touch_altitude is not None and h is not None
                and h > self.touch_altitude + 1.8):
            self.goto_state("PICK_BASE")

    def state_return(self, g):
        # pousar de volta na origem: mesma sequencia, alvo = home
        self.target = tuple(self.home_xy)
        self._send_goal(self.nav_client,
                        NavigateToPose.Goal(pose=self.make_pose(*self.home_xy)))
        self.goto_state("RETURN_GOTO")

    def state_return_goto(self, g):
        if self._altitude_guard(g, "RETURN_GOTO"):
            return
        # Sem prazo aqui, um goal que nunca conclui deixaria o drone parado no
        # ar de bateria a zero. Passado o limite, segue para o pouso de onde
        # estiver: descer perto de casa e melhor que nao descer.
        if self.goal_done or self.elapsed() > 120.0:
            if not self.goal_done:
                self.get_logger().warn(
                    "Retorno nao concluiu no prazo: pousando na posicao atual")
                self.cancel_nav_goal()
            self.goto_state("ALIGN")  # pad de decolagem tambem e detectavel

    def state_end(self, g):
        self.get_logger().info(
            f"==== MISSAO CONCLUIDA: {len(self.visited)} pousos ====")
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = MissionControl()
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
