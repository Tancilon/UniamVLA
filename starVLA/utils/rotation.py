"""6D rotation representation utilities (Zhou et al. CVPR 2019).

For state/action encoding in VLA models, 6D rotation is preferred over
quaternion or Euler angles because it provides a continuous, unambiguous
mapping from SO(3) to R^6 — critical for stable neural network regression.

References:
    Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
    CVPR 2019, https://arxiv.org/abs/1812.07035
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _R


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> rotation matrix (3, 3).

    Input does not need to be unit-length; scipy normalises it internally.
    A zero-norm quaternion raises ValueError.
    """
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape != (4,):
        raise ValueError(f"quat must have shape (4,), got {quat.shape}")
    return _R.from_quat(quat).as_matrix().astype(np.float32)


def euler_to_matrix(euler: np.ndarray, convention: str = "XYZ") -> np.ndarray:
    """Euler angles -> rotation matrix (3, 3). Default intrinsic XYZ."""
    euler = np.asarray(euler, dtype=np.float32)
    if euler.shape != (3,):
        raise ValueError(f"euler must have shape (3,), got {euler.shape}")
    return _R.from_euler(convention, euler).as_matrix().astype(np.float32)


def matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) -> 6D representation (first two columns flattened)."""
    R = np.asarray(R, dtype=np.float32)
    if R.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3), got {R.shape}")
    # Take first two columns, flatten column-major: [R[:,0], R[:,1]]
    return R[:, :2].flatten(order="F").astype(np.float32)


def quat_to_6d(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 6D rotation."""
    return matrix_to_6d(quat_to_matrix(quat))


def euler_to_6d(euler: np.ndarray, convention: str = "XYZ") -> np.ndarray:
    """Euler angles -> 6D rotation."""
    return matrix_to_6d(euler_to_matrix(euler, convention))


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle rotation vector (rotvec, magnitude=angle) → rotation matrix (3, 3).

    The 3-vector encodes the rotation as: axis = vec / ||vec||, angle = ||vec||.
    Used by robosuite/LIBERO for `ee_ori` field. Compatible with
    scipy.spatial.transform.Rotation.from_rotvec.
    """
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    if axis_angle.shape != (3,):
        raise ValueError(f"axis_angle must have shape (3,), got {axis_angle.shape}")
    return _R.from_rotvec(axis_angle).as_matrix().astype(np.float32)


def axis_angle_to_6d(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle (rotvec) → 6D rotation."""
    return matrix_to_6d(axis_angle_to_matrix(axis_angle))


def matrix_from_6d(x6d: np.ndarray) -> np.ndarray:
    """6D representation -> rotation matrix via Gram-Schmidt orthogonalization.

    Used primarily for round-trip testing and inverse transformations.
    Raises ValueError on degenerate input (zero a1 or a2 parallel to a1).
    """
    x6d = np.asarray(x6d, dtype=np.float32)
    if x6d.shape != (6,):
        raise ValueError(f"x6d must have shape (6,), got {x6d.shape}")

    # x6d is column-major flattened: first 3 = column 0, last 3 = column 1
    a1 = x6d[:3]
    a2 = x6d[3:6]

    norm_a1 = np.linalg.norm(a1)
    if norm_a1 < 1e-8:
        raise ValueError(
            f"matrix_from_6d: a1 is near-zero (norm={norm_a1:.2e}). "
            f"Input 6D may be corrupted."
        )
    b1 = a1 / norm_a1

    b2 = a2 - np.dot(b1, a2) * b1
    norm_b2 = np.linalg.norm(b2)
    if norm_b2 < 1e-8:
        raise ValueError(
            f"matrix_from_6d: a2 is parallel to a1 (projected norm={norm_b2:.2e}). "
            f"Cannot complete Gram-Schmidt."
        )
    b2 = b2 / norm_b2

    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32)
