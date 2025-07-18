# Common
DETECTED_COORDINATES_TOPIC = "base_detection/detected_coordinates"

# Base_detection
COLOR_IMAGE_TOPIC = f"/hermit/camera/d435i/color/image_raw"
COLOR_IMAGE_TOPIC2 = f"camera/color/image_raw"
INFERRED_IMAGE_TOPIC = "base_detection/inferred_image_capiche"
BOOL_DETECTOR_TOPIC = "base_detection/has_base"
NUMBER_OF_BASES_TOPIC = "base_detection/number_of_bases"
# Coordinate Receiver
DELTA_POINTS_TOPIC = "base_detection/delta_points"
ABSOLUTE_POINTS_TOPIC = "base_detection/absolute_points"
HIGH_ACCURACY_POINT_TOPIC = "base_detection/high_accuracy_point"
DEPTH_IMAGE_TOPIC = f"camera/depth/depth_image"
VEHICLE_LOCAL_POSITION_TOPIC = "/fmu/out/vehicle_local_position"
VEHICLE_ODOMETRY_TOPIC = "/fmu/out/vehicle_odometry"

# Coordinate Processor
UNIQUE_POSITIONS_TOPIC = "base_detection/unique_positions"
CONFIRMED_BASES_TOPIC = "base_detection/confirmed_bases"
