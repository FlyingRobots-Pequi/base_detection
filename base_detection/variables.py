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
D455_FX_DEPTH = 917.06 # px, focal length X for 1280×720 depth stream [2]
D455_FY_DEPTH = 917.06 # px, focal length Y for 1280×720 depth stream [2]
D455_CX_DEPTH = 639.5  # px, principal point X [2]
D455_CY_DEPTH = 359.5  # px, principal point Y [2]
D455_BASELINE = 0.052  # m, stereo baseline between IR cameras [2]