# Missão RGB-only: mapear e pousar em todas as bases

Branch `mission/rgb-only` — resolve a task usando **só a câmera RGB de baixo**
(sem depth): varre a arena, mapeia as bases por interseção raio↔chão e pousa em
todas, uma a uma, voltando à origem no fim.

## Nós e responsabilidades

| Nó | Arquivo | Papel |
|---|---|---|
| `base_detection` | `base_detection/base_detection.py` | YOLO+HSV na câmera de baixo; projeta pixel→mundo via TF (plano Z=0); clusteriza e publica `/base_detection/bases`; publica erro de pixel p/ servo visual |
| `mission_control` | `base_detection/mission_control.py` | FSM da missão: subida, varredura (Nav2 FollowWaypoints), ida à base (NavigateToPose), alinhamento (servo visual), descida, pouso, desarme/rearme, retorno |
| `px4_gestures` | `base_detection/px4_gestures.py` | Primitivas por cima do `offboard_control`: arm/disarm pelos gestos de joystick, vz pelo destrave do `/joy` (buttons[10]) e `lock_vz()` para devolver o Z ao controle de posição |
| `gt_odometry` | `base_detection/gt_odometry.py` | **Só simulação**: pose real do gz como fonte de localização (`localization_source:=gt`), no lugar da odometria visual |

## A FSM

```
BOOT → CLIMB → SEARCH → PICK_BASE → GOTO → ALIGN → DESCEND → CONFIRM → DISARM → PAUSE → REARM → WAIT_TAKEOFF
                  ↑___________________________________________________________________________________|
                (sem base nova) → RETURN → RETURN_GOTO → ALIGN → ... → PAUSE → END

guardas (em qualquer fase de navegação): altitude < 1.3 m → RECOVER_ALT
                                          fora da cerca   → RECENTER
```

**Regra de ouro**: `/cmd_vel` tem um dono por vez — Nav2 nas fases horizontais,
`px4_gestures` nas verticais. E o eixo Z tem dois regimes: nas fases do Nav2 a
missão reafirma `lock_vz()` a cada ciclo (Z em controle de posição); nas fases
verticais o `/joy` destrava o Z e a missão manda `vz`.

## Convenções que causaram bugs reais — leia antes de mexer

1. **NED × ENU.** A PX4 fala NED (x=Norte, y=Leste); bases, waypoints e cerca
   vivem no ENU do mapa (x=Leste, y=Norte). Use `map_pose()` (TF `map→base_link`),
   nunca `local_position` da PX4, para qualquer conta no frame do mapa. Misturar
   os dois fazia a cerca acusar violação com o drone em posição legal e o recuo
   sair na diagonal errada, até pousar em cima do cenário.
2. **`vz` é ENU do Twist: positivo SOBE.** Já esteve invertido no `RECOVER_ALT`
   — a guarda de altitude empurrava o drone contra o chão, em espiral.
3. **O unlock do Z é pegajoso.** O `offboard_control` só reavalia quando chega
   um `/joy`. Voltando de uma fase vertical sem `lock_vz()`, o Z fica em
   velocidade com vz=0, sem latch, e o drone afunda durante a navegação.
4. **Nada de gesto por acidente.** Toda velocidade da missão fica < 0.4 m/s: os
   gestos de arm/disarm disparam com |linear.z| e |angular.z| simultâneos > 0.4.

## Configuração do Nav2 que este drone exige

- **Yaw congelado** (`max_vel_theta: 0.0`, sem críticos de heading, goals com
  yaw=π/2): girar tira a câmera do conteúdo texturizado e mata a odometria
  visual. O drone navega de strafe (`min_vel_x/y` negativos).
- **`velocity_smoother` holonômico** (`max_velocity: [0.35, 0.35, 0.0]`): o
  default do Nav2 é diferencial e o `0.0` do meio **zera a velocidade lateral**
  — com o yaw congelado, isso trava a varredura inteira ("Failed to make
  progress" em loop).
- **`spin` tem que continuar em `behavior_plugins`**: o BT default referencia a
  ação no load; sem o server, o lifecycle aborta o bringup inteiro.
- **Costmap global rolling, sem `static_layer`**: não depende do `/map` do
  RTAB-Map, então funciona igual nos dois modos de localização.

## Percepção: o que o portão de nadir faz

A projeção assume base no plano Z=0. Base **elevada** vista de esguelha entra no
mapa deslocada ~h·tan(θ), e a média exponencial mistura bases vizinhas. Por isso
só entram no mapa detecções cujo ponto projetado esteja a menos de
`nadir_gate_m` (1 m) da vertical do drone — critério em **metros**, não em
ângulo: um limiar angular fixo cobre uma área que encolhe com a altura, e
varrendo baixo ele rejeitava tudo (varredura inteira com zero bases mapeadas).

O servo visual (fase ALIGN) usa **todas** as detecções, pelo tópico de erro de
pixel: levar o erro a zero põe o drone no nadir da base, o que também zera o
viés de paralaxe. E a altura real da base é medida **no pouso** (a missão
publica em `/base_detection/base_height_update`), única forma de conhecê-la sem
sensor de profundidade.

## Localização: dois modos

`localization_source:=vslam` (default, fiel ao drone real) usa a odometria
visual do RTAB-Map. **Ela não sustenta o voo da varredura nesta arena**: perde
tracking, reseta pra origem, e cada reset entra no EKF como salto de posição
(medido: 22 resets de `pos_ne`, estimativa fugindo para 70 m com o drone parado
no ar; o drone gira e arranca atrás de uma pose fantasma). Item aberto.

`localization_source:=gt` usa a pose real do gz. Troca **apenas** a origem do TF
`odom→base_link` e da odometria mandada ao EKF; percepção, Nav2 e missão ficam
idênticos. Detalhe crítico: `normalize_to_first_sample` tem que ficar
**desligada** nesse modo — ela torna a pose relativa ao instante de partida
(paliativo para a visão) e corrompe um dado que já é absoluto.

## Ajustes de PX4 (airframe 22000_gz_hermit)

- `EKF2_EV_CTRL 9` + `EKF2_HGT_REF 0`: altura pelo **barômetro**. A visão quase
  não observa movimento vertical e derivava ~2 m numa varredura de 9 min; com a
  PX4 segurando o z estimado, o drone afundava fisicamente até o chão.
- `MPC_TILTMAX_AIR 20`, `MPC_XY_VEL_MAX 1.0`: bounda a excursão quando um salto
  de EV vira correção agressiva de atitude.

E no `offboard_control.cpp`: o relatch de posição **preserva o z** em navegação
2D (senão cada pausa do Nav2 rebaixa a altitude de voo: 2.8 → 2.4 → 1.7 m), e o
nó **não morre mais** ao sair de OFFBOARD (antes: ninguém publicava setpoints →
failsafe → queda).

## Rodar

Container (`IMAGE_NAME=uav-px4-simulator-local:base_detection ./run.sh`) e, na
primeira vez, `cd /root/ros2_ws && colcon build --symlink-install && source
install/setup.bash`. Depois, um comando por terminal — todos via
`docker exec -it px4_container bash`:

```bash
# 1 — simulação
simulation.sh

# 2 — navegação (espere "Managed nodes are active")
ros2 launch rtabmap_drone_example ros2_bridge.launch.py localization_source:=gt

# 3 — offboard (arma e decola sozinho)
ros2 run rtabmap_drone_example offboard_control --ros-args -r __ns:=/pequi/hermit -p use_sim_time:=true

# 4 — percepção
ros2 run base_detection base_detection --ros-args -p use_sim_time:=true

# 5 — missão
ros2 run base_detection mission_control --ros-args -r __ns:=/pequi/hermit -p use_sim_time:=true
```

**Entre o 2 e o 3, confira a árvore de TF** — quando ela racha, a projeção das
bases falha e a missão trava esperando:

```bash
ros2 run tf2_ros tf2_echo map camera_down_optical_frame   # tem que dar Translation
```

Parâmetros úteis da missão: `land_during_search:=false` (varre tudo antes de
pousar, em vez do pouso incremental), `skip_search:=true` (vai direto às bases
já mapeadas), `search_altitude`, `search_x_min/max`, `search_y_min/max`.
`px4_gesture_test` valida as primitivas de voo isoladamente.

No RViz, em modo `gt` os displays de SLAM (Map, MapCloud, MapGraph, Rtabmap
plan) ficam vazios de propósito — o RTAB-Map não roda. Olhe Local costmap, TF e
o Image com `/base_detection/debug_image`.

**Antes de acusar a missão de qualquer coisa, valide a localização**: compare a
estimativa do EKF com a pose física do gz (`gz model -m hermit_0 -p` contra
`px4-listener vehicle_local_position`, lembrando ENU↔NED). Boa parte das falhas
desta task foi missão obedecendo fielmente uma pose errada.

## Limites conhecidos / próximos passos

- Odometria visual (acima) — o item aberto de maior peso.
- Bases elevadas: XY correto exige nadir; o Z real só é conhecido após o 1º
  pouso nelas.
- `FollowWaypoints` não replaneja a ordem; waypoints fixos por parâmetros
  `search_*`.
- Falta rodar o teste de variações de cenário com `set_land_bases.py`.
