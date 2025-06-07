import functools
import traceback


def log_exception(func):
    """
    Decorator que captura exceções, formata o traceback e retorna
    um dict com detalhes do erro e dos parâmetros de entrada/saída.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as exc:
            tb_str = traceback.format_exc()
            # Log exception usando ROS2 logger se self for um Node
            if args and hasattr(args[0], "get_logger"):
                logger = args[0].get_logger()
                logger.error(f"Exception in {func.__name__}: {exc}")
                logger.error(tb_str)
            else:
                # fallback genérico
                print(f"Exception in {func.__name__}: {exc}")
                print(tb_str)
            return {
                "traceback": tb_str,
                "function_name": func.__name__,
                "error": exc,
                "inputs": {"args": args, "kwargs": kwargs},
                "output": None,
            }

    return wrapper


# Common
DETECTED_COORDINATES_TOPIC = "detected_coordinates"

# Base_detection
COLOR_IMAGE_TOPIC = f"/hermit/camera/d435i/color/image_raw"
COLOR_IMAGE_TOPIC2 = f"/hermit/d455/color/image_raw"
INFERRED_IMAGE_TOPIC = "inferred_image_capiche"

# Coordinate Receiver
DELTA_POSITION_TOPIC = "delta_position"
DEPTH_IMAGE_TOPIC = f"/hermit/camera/d435i/depth/image_rect_raw"
VEHICLE_LOCAL_POSITION_TOPIC = "/fmu/out/vehicle_local_position"

# Coordinate Processor
UNIQUE_POSITIONS_TOPIC = "unique_positions"
NUM_BASES = 3

# Variables para a camera D435i
D435I_BIAS_X = 0.1
D435I_BIAS_Y = 0.1
D435I_FX_DEPTH = 925.1
D435I_FY_DEPTH = 925.1
D435I_CX_DEPTH = 639.5
D435I_CY_DEPTH = 359.5
D435I_BASELINE = 0.025

# Variables para a camera D455
D455_BIAS_X = 0.02447  # m, translation X from Infrared2 to IMU (extrinsic) [1]
D455_BIAS_Y = -0.00130  # m, translation Y from Infrared2 to IMU (extrinsic) [1]
D455_FX_DEPTH = 917.06  # px, focal length X for 1280×720 depth stream [2]
D455_FY_DEPTH = 917.06  # px, focal length Y for 1280×720 depth stream [2]
D455_CX_DEPTH = 639.5  # px, principal point X [2]
D455_CY_DEPTH = 359.5  # px, principal point Y [2]
D455_BASELINE = 0.052  # m, stereo baseline between IR cameras [2]
