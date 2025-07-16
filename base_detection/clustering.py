import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from logging import Logger

from .parameters import ClusteringParams
from .utils import detect_lines_ransac


def run_hybrid_line_kmeans(
    positions: np.ndarray,
    clustering_params: ClusteringParams,
    expected_bases: int,
    movement_direction: np.ndarray,
    logger: Logger,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Runs a hybrid clustering algorithm using line detection to guide K-Means.

    Args:
        positions: 3D points [x, y, z] to be clustered.
        clustering_params: Dataclass containing parameters for clustering.
        expected_bases: The expected number of clusters (bases).
        movement_direction: The estimated primary direction of drone movement.
        logger: The ROS 2 node logger.

    Returns:
        A tuple containing a list of cluster centers (3D positions) and cluster labels.
    """
    logger.debug(
        f"🚁 Running Hybrid Line-KMeans clustering with {len(positions)} points."
    )
    positions_2d = positions[:, :2]

    # Step 1: Detect lines and assign IDs
    detected_lines = detect_lines_ransac(
        positions_2d, clustering_params, logger, movement_direction
    )

    line_ids = np.full(len(positions), -1, dtype=int)
    if not detected_lines:
        logger.warning("⚠️ No lines detected. Proceeding with standard K-Means.")
    else:
        for i, line in enumerate(detected_lines):
            line_ids[line["inliers"]] = i

    # Step 2: Create enhanced features for clustering
    line_feature = clustering_params.collinearity_weight * line_ids.reshape(-1, 1)
    enhanced_features = np.hstack([positions_2d, line_feature])

    # Step 3: Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(enhanced_features)

    # Step 4: Run K-Means on the enhanced and normalized data
    n_clusters = min(expected_bases, len(positions))
    if n_clusters <= 0:
        logger.warn("Not enough data to form any clusters.")
        return [], np.array([])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    labels = kmeans.labels_

    # Step 5: Calculate cluster centers in the original coordinate space
    cluster_centers = []
    for i in range(n_clusters):
        points_in_cluster = positions[labels == i]
        if len(points_in_cluster) > 0:
            center = np.mean(points_in_cluster, axis=0)
            cluster_centers.append(center)

    logger.info(f"✅ Hybrid clustering found {len(cluster_centers)} clusters.")
    return cluster_centers, labels
