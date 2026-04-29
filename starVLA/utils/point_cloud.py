"""Point cloud post-processing utilities.

Statistical Outlier Removal (SOR) with fixed-size resampling, applied at
load time by both the training dataset and the visualizer so the on-disk
point clouds stay unchanged.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial import cKDTree


def clean_point_cloud(
    pts: np.ndarray,
    k: int = 16,
    std_ratio: float = 2.0,
    num_points: int = 1024,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Remove statistical outliers and resample to a fixed point count.

    Applies SOR using k-NN mean distances, then resamples the kept points
    — with replacement if necessary — to exactly ``num_points``.

    Args:
        pts: (N, 3) point cloud.
        k: Number of nearest neighbors used to compute the per-point mean
            distance.
        std_ratio: Points whose mean distance exceeds
            ``global_mean + std_ratio * global_std`` are removed.
        num_points: Fixed output point count.
        rng: Optional numpy Generator for deterministic resampling. If
            omitted, ``np.random.default_rng()`` is used.

    Returns:
        (num_points, 3) float32 array.

    Raises:
        ValueError: If ``pts`` is empty.
    """
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts must have shape (N, 3), got {pts.shape}")
    n = pts.shape[0]
    if n == 0:
        raise ValueError("clean_point_cloud received an empty point cloud")

    pts = pts.astype(np.float32, copy=False)
    rng = rng or np.random.default_rng()

    if n < k + 1:
        warnings.warn(
            f"clean_point_cloud: N={n} < k+1={k + 1}, skipping SOR and resampling only.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _resample(pts, num_points, rng)

    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std
    keep_mask = mean_dists <= threshold
    kept = pts[keep_mask]

    if kept.shape[0] == 0:
        warnings.warn(
            "clean_point_cloud: SOR removed all points; falling back to original cloud.",
            RuntimeWarning,
            stacklevel=2,
        )
        kept = pts

    return _resample(kept, num_points, rng)


def _resample(
    pts: np.ndarray, num_points: int, rng: np.random.Generator
) -> np.ndarray:
    n = pts.shape[0]
    replace = n < num_points
    idx = rng.choice(n, size=num_points, replace=replace)
    return pts[idx].astype(np.float32, copy=False)
