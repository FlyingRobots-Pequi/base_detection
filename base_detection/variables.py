# Common
DETECTED_COORDINATES_TOPIC = "base_detection/detected_coordinates"

# Base_detection
COLOR_IMAGE_TOPIC = f"/hermit/camera/d435i/color/image_raw"
COLOR_IMAGE_TOPIC2 = f"camera/color/image_raw"
#COLOR_IMAGE_TOPIC2 = f"/hermit/d455/color/image_raw"
INFERRED_IMAGE_TOPIC = "base_detection/inferred_image_capiche"

# Coordinate Receiver
DELTA_POINTS_TOPIC = "base_detection/delta_points"
ABSOLUTE_POINTS_TOPIC = "base_detection/absolute_points"
#DEPTH_IMAGE_TOPIC = f"/hermit/camera/d435i/depth/image_rect_raw"
DEPTH_IMAGE_TOPIC = f"camera/depth/depth_image"
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

# Camera resolution constants for coordinate transformation
D455_RGB_WIDTH = 1920     # Standard RGB resolution width
D455_RGB_HEIGHT = 1080    # Standard RGB resolution height
D455_DEPTH_WIDTH = 1280   # Depth stream resolution width
D455_DEPTH_HEIGHT = 720   # Depth stream resolution height

# Base filtering constants
INITIAL_BASE_EXCLUSION_RADIUS = 1.5  # meters - radius around (0,0) to exclude detections (aumentado de 0.7)
INITIAL_BASE_X = 0.0                 # X coordinate of initial base position
INITIAL_BASE_Y = 0.0                 # Y coordinate of initial base position

# Advanced filtering constants
MIN_DISTANCE_BETWEEN_BASES = 1.0     # meters - minimum distance between detected bases
MAX_DETECTIONS_STORED = 75           # maximum number of detections to store before clustering (aumentado para permitir mais detecções)
DETECTION_TIMEOUT = 45.0             # seconds - timeout to trigger clustering (aumentado para dar mais tempo para encontrar as 5 bases)