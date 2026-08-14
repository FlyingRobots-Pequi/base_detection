"""Teste de bancada das primitivas de voo (px4_gestures) contra o SITL.

Valida a hipotese central da missao SEM percepcao: da para pousar, desarmar e
rearmar so com os gestos de /cmd_vel + /joy, com o offboard_control sobrevivendo
ao ciclo (risco: update_state chama rclcpp::shutdown() se o nav_state sair de
OFFBOARD, offboard_control.cpp:158-162).

Sequencia (FSM a 20 Hz):
  WAIT_READY -> WAIT_AIRBORNE (offboard_control arma e decola sozinho ao subir)
  -> DESCEND (vz < 0 via /joy unlock) -> WAIT_LAND (vehicle_land_detected)
  -> DISARM (gesto) -> PAUSE -> ARM (gesto) -> WAIT_TAKEOFF -> PASS

Rodar sob o namespace da PX4, com o stack (simulation.sh + bridge + offboard) up:
  ros2 run base_detection px4_gesture_test --ros-args -r __ns:=/pequi/hermit \
      -p use_sim_time:=true
"""

import rclpy
from rclpy.node import Node

from base_detection.px4_gestures import PX4Gestures

DESCEND_SPEED = 0.3     # m/s; < 0.4 para nunca casar um gesto por acidente
TICK_HZ = 20.0
STATE_TIMEOUT_S = 90.0  # failsafe por estado


class GestureTest(Node):
    def __init__(self):
        super().__init__("px4_gesture_test")
        self.gestures = PX4Gestures(self)
        self.state = "WAIT_READY"
        # Com use_sim_time, now() vale 0 ate o primeiro /clock chegar; capturar
        # aqui estouraria o timeout no primeiro tick com clock valido. O tick()
        # inicializa quando o clock ja esta correndo.
        self.state_start = None
        self.takeoff_alt = None  # altitude no momento do rearm, para medir subida
        self.timer = self.create_timer(1.0 / TICK_HZ, self.tick)
        self.get_logger().info("Gesture test iniciado")

    def elapsed(self):
        return (self.get_clock().now() - self.state_start).nanoseconds / 1e9

    def goto(self, state):
        self.get_logger().info(
            f"[{self.state} -> {state}] alt={self.gestures.altitude:.2f} "
            f"armed={self.gestures.armed} offboard={self.gestures.offboard} "
            f"landed={self.gestures.landed}")
        self.state = state
        self.state_start = self.get_clock().now()

    def finish(self, verdict):
        self.get_logger().info(f"==== RESULTADO: {verdict} ====")
        self.timer.cancel()
        raise SystemExit(0 if verdict == "PASS" else 1)

    def tick(self):
        g = self.gestures

        if self.state_start is None:
            if self.get_clock().now().nanoseconds == 0:
                return  # sim time ainda nao chegou
            self.state_start = self.get_clock().now()

        if self.elapsed() > STATE_TIMEOUT_S:
            self.finish(f"FAIL (timeout em {self.state})")

        if self.state == "WAIT_READY":
            if g.ready:
                self.goto("WAIT_AIRBORNE")

        elif self.state == "WAIT_AIRBORNE":
            # offboard_control arma e sobe 2 m sozinho; esperar ele chegar la
            if g.armed and g.offboard and g.altitude > 1.5:
                self.goto("STABILIZE")

        elif self.state == "STABILIZE":
            # deixar o setpoint de takeoff assentar antes de comandar velocidade
            if self.elapsed() > 4.0:
                self.goto("DESCEND")

        elif self.state == "DESCEND":
            g.publish_velocity(vz=-DESCEND_SPEED)
            if g.landed:
                self.goto("DISARM")

        elif self.state == "DISARM":
            g.publish_disarm_gesture()
            if not g.armed:
                self.goto("PAUSE")

        elif self.state == "PAUSE":
            # 3 s parado no chao; NAO publicar nada (hold)
            if not g.offboard:
                # o ponto critico do teste: a PX4 saiu de offboard ao desarmar?
                self.finish("FAIL (PX4 saiu de OFFBOARD apos desarmar -- "
                            "offboard_control deve ter morrido)")
            if self.elapsed() > 3.0:
                self.goto("ARM")

        elif self.state == "ARM":
            g.publish_arm_gesture()
            if g.armed:
                self.takeoff_alt = g.altitude
                self.goto("WAIT_TAKEOFF")

        elif self.state == "WAIT_TAKEOFF":
            # arm() do offboard_control mira 2 m acima; aceitar >1 m de subida
            if g.altitude > self.takeoff_alt + 1.0:
                self.finish("PASS")


def main(args=None):
    rclpy.init(args=args)
    node = GestureTest()
    try:
        rclpy.spin(node)
    except SystemExit as e:
        raise e
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
