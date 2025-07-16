import numpy as np
from logging import Logger
from typing import Tuple, List
from .parameters import ClusteringParams, InitialBaseParams, MotionParams
from scipy import stats
from scipy.spatial.transform import Rotation


def detect_outliers_iqr(
    positions: np.ndarray, outlier_threshold: float, logger: Logger
) -> np.ndarray:
    """Detects outliers using the Interquartile Range (IQR) method."""
    if len(positions) < 4:
        return np.ones(len(positions), dtype=bool)

    distances = np.sqrt(np.sum(positions**2, axis=1))

    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1

    lower_bound = q1 - outlier_threshold * iqr
    upper_bound = q3 + outlier_threshold * iqr

    mask = (distances >= lower_bound) & (distances <= upper_bound)

    outliers_count = np.sum(~mask)
    if outliers_count > 0:
        logger.info(f"Detected and removing {outliers_count} outliers using IQR method")

    return mask


def remove_outliers_2d(points_3d, threshold, logger):
    """
    Removes outliers from a 3D point cloud (x, y, weight) based on the Z-score of the x and y coordinates.
    Keeps the z-coordinate (weight) associated with the inliers.
    """
    if len(points_3d) == 0:
        return np.array([])
        
    points_2d = points_3d[:, :2]
    mean = np.mean(points_2d, axis=0)
    std = np.std(points_2d, axis=0)
    
    # Avoid division by zero if all points are the same
    std[std == 0] = 1 
    
    z_scores = np.abs((points_2d - mean) / std)
    
    # Check Z-scores for both x and y dimensions
    inliers_mask = np.all(z_scores < threshold, axis=1)
    
    num_outliers = len(points_3d) - np.sum(inliers_mask)
    if num_outliers > 0:
        logger.info(f"Filtered out {num_outliers} outliers based on Z-score.")
        
    return points_3d[inliers_mask]


def detect_lines_ransac(
    positions: np.ndarray,
    clustering_params: ClusteringParams,
    logger: Logger,
    direction_hint: np.ndarray = None,
) -> List[dict]:
    """Detects lines in position data using the RANSAC algorithm."""
    if len(positions) < clustering_params.min_line_points:
        return []

    lines = []
    remaining_points = list(range(len(positions)))

    while len(remaining_points) >= clustering_params.min_line_points:
        best_line = None
        best_inliers = []
        best_score = 0

        for _ in range(clustering_params.ransac_iterations):
            if len(remaining_points) < 2:
                break

            sample_indices = np.random.choice(remaining_points, 2, replace=False)
            p1, p2 = positions[sample_indices]

            if np.allclose(p1, p2):
                continue

            direction = p2 - p1
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0:
                continue

            direction /= direction_norm

            if abs(direction[0]) > abs(direction[1]):
                m = direction[1] / direction[0] if direction[0] != 0 else 0
                c = p1[1] - m * p1[0]
                line_params = [-m, 1, -c]
            else:
                m = direction[0] / direction[1] if direction[1] != 0 else 0
                c = p1[0] - m * p1[1]
                line_params = [1, -m, -c]

            inliers = []
            for i in remaining_points:
                point = positions[i]
                dist = abs(
                    line_params[0] * point[0]
                    + line_params[1] * point[1]
                    + line_params[2]
                )
                dist /= np.sqrt(line_params[0] ** 2 + line_params[1] ** 2)

                if dist <= clustering_params.line_tolerance:
                    inliers.append(i)

            score = len(inliers)
            if direction_hint is not None:
                line_direction = np.array([-line_params[1], line_params[0]])
                line_direction /= np.linalg.norm(line_direction)
                alignment = abs(np.dot(direction, direction_hint))
                score += alignment * 2

            if score > best_score and len(inliers) >= clustering_params.min_line_points:
                best_line = line_params
                best_inliers = inliers
                best_score = score

        if (
            best_line is not None
            and len(best_inliers) >= clustering_params.min_line_points
        ):
            lines.append(
                {
                    "line_params": best_line,
                    "inliers": best_inliers,
                    "points": positions[best_inliers],
                    "score": best_score,
                }
            )
            remaining_points = [i for i in remaining_points if i not in best_inliers]
            logger.debug(
                f"📏 Detected line with {len(best_inliers)} points, score: {best_score:.2f}"
            )
        else:
            break
    return lines


def is_near_initial_base(x: float, y: float, params: InitialBaseParams) -> bool:
    """Checks if a position is within the exclusion radius of the initial base."""
    distance = np.sqrt((x - params.x) ** 2 + (y - params.y) ** 2)
    return distance <= params.exclusion_radius


def assess_motion_stability(
    vx: float, vy: float, ax: float, ay: float, stability_threshold: float
) -> Tuple[float, bool]:
    """Assesses vehicle motion stability based on velocity and acceleration."""
    velocity_magnitude = np.sqrt(vx**2 + vy**2)
    acceleration_magnitude = np.sqrt(ax**2 + ay**2)

    velocity_stability = max(0.0, 1.0 - velocity_magnitude / (stability_threshold * 2))
    acceleration_stability = max(0.0, 1.0 - acceleration_magnitude / 1.0)

    stability_factor = 0.7 * velocity_stability + 0.3 * acceleration_stability
    is_stable = velocity_magnitude < stability_threshold
    return stability_factor, is_stable


def calculate_motion_compensation(
    detection_timestamp: float,
    vehicle_timestamp: float,
    current_pose: Tuple[float, float, float, float],
    current_velocity: Tuple[float, float, float],
    current_acceleration: Tuple[float, float, float],
    motion_params: MotionParams,
    logger: Logger,
) -> Tuple[float, float, float, float]:
    """Calculates vehicle position at detection time to compensate for processing delay."""
    current_x, current_y, current_altitude, current_yaw = current_pose
    vx, vy, vz = current_velocity
    ax, ay, az = current_acceleration

    if not motion_params.compensation_enabled:
        return current_pose

    time_delta = detection_timestamp - vehicle_timestamp

    if abs(time_delta) > 1.0 or time_delta < 0:
        logger.debug(
            f"⚠️ Large time delta ({time_delta:.3f}s), skipping motion compensation"
        )
        return current_pose

    compensated_x = current_x - (vx * time_delta + 0.5 * ax * time_delta**2)
    compensated_y = current_y - (vy * time_delta + 0.5 * ay * time_delta**2)
    compensated_z = current_altitude - (vz * time_delta + 0.5 * az * time_delta**2)
    compensated_yaw = current_yaw

    velocity_magnitude = np.sqrt(vx**2 + vy**2)
    position_compensation = np.sqrt(
        (current_x - compensated_x) ** 2 + (current_y - compensated_y) ** 2
    )

    if position_compensation > 0.01:
        accel_magnitude = np.sqrt(ax**2 + ay**2)
        logger.debug("🔧 Motion Compensation Applied:")
        logger.debug(f"  ⏱️ Time delta: {time_delta:.3f}s")
        logger.debug(
            f"  🏃 Velocity: {velocity_magnitude:.3f}m/s, Accel: {accel_magnitude:.3f}m/s²"
        )
        logger.debug(f"  📍 Position compensation: {position_compensation:.3f}m")

    return compensated_x, compensated_y, compensated_z, compensated_yaw
