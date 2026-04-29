"""Pure geometry utility functions for dataset preprocessing.

No simulator dependency — all functions take numpy arrays as input.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image


def mat_to_6d(R: np.ndarray) -> list[float]:
    """Convert 3x3 rotation matrix to 6D representation (first two columns).

    Returns [col0[0], col0[1], col0[2], col1[0], col1[1], col1[2]].
    """
    return R[:, :2].T.flatten().tolist()


def rotation_6d_to_matrix_np(d6) -> np.ndarray:
    """Convert 6D rotation representation to 3x3 matrix via Gram-Schmidt.

    Numpy equivalent of pose_utils.rotation_6d_to_matrix (PyTorch version).

    Args:
        d6: Sequence of 6 floats [a1x, a1y, a1z, a2x, a2y, a2z].

    Returns:
        (3, 3) rotation matrix.
    """
    d6 = np.asarray(d6, dtype=np.float64)
    a1 = d6[:3]
    a2 = d6[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def get_camera_intrinsic_from_fovy(
    fovy: float, width: int = 256, height: int = 256
) -> dict:
    """Compute camera intrinsic parameters from MuJoCo's vertical FOV.

    Args:
        fovy: Vertical field of view in degrees.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Dict with keys: fx, fy, cx, cy, width, height.
    """
    f = 0.5 * height / np.tan(np.radians(fovy / 2.0))
    return {
        "fx": float(f),
        "fy": float(f),
        "cx": width / 2.0,
        "cy": height / 2.0,
        "width": width,
        "height": height,
    }


def linearize_depth(
    depth_buf: np.ndarray, znear: float, zfar: float
) -> np.ndarray:
    """Convert raw OpenGL depth buffer to metric depth in meters.

    MuJoCo's sim.render(depth=True) returns the nonlinear GL depth
    buffer in [0, 1].  The linearization formula is:
        depth_metric = znear * zfar / (zfar - buf * (zfar - znear))
    Pixels with buf == 0 are treated as invalid (no geometry) and
    kept at 0.
    """
    out = np.zeros_like(depth_buf, dtype=np.float32)
    valid = depth_buf > 0
    out[valid] = znear * zfar / (zfar - depth_buf[valid] * (zfar - znear))
    return out


def depth_to_world_points(
    depth: np.ndarray,
    seg_mask: np.ndarray,
    intrinsic: dict,
    cam_R: np.ndarray,
    cam_t: np.ndarray,
    target_id: int,
    num_points: int = 1024,
) -> np.ndarray | None:
    """Convert depth + segmentation to world-frame point cloud for target object.

    Args:
        depth: (H, W) float32 depth image.
        seg_mask: (H, W) int segmentation mask with instance IDs.
        intrinsic: Dict with fx, fy, cx, cy.
        cam_R: (3, 3) camera-to-world rotation matrix (OpenGL/MuJoCo convention).
        cam_t: (3,) camera-to-world translation vector (camera position in world).
        target_id: Instance ID of the target object in seg_mask.
        num_points: Fixed output point count.

    Returns:
        (num_points, 3) float32 array in world frame, or None if target not visible.
    """
    H, W = depth.shape
    mask = (seg_mask == target_id) & (depth > 0)

    if not np.any(mask):
        return None

    v_idx, u_idx = np.where(mask)
    d = depth[v_idx, u_idx]

    fx, fy = intrinsic["fx"], intrinsic["fy"]
    cx, cy = intrinsic["cx"], intrinsic["cy"]

    # Back-project to OpenCV camera frame (x-right, y-down, z-forward)
    x_cv = (u_idx - cx) * d / fx
    y_cv = (v_idx - cy) * d / fy
    z_cv = d
    # Convert to OpenGL camera frame (x-right, y-up, z-backward)
    # required because MuJoCo cam_xmat uses OpenGL convention
    pts_cam = np.stack([x_cv, -y_cv, -z_cv], axis=-1)

    pts_world = (cam_R @ pts_cam.T).T + cam_t

    n = len(pts_world)
    if n > num_points:
        indices = np.random.choice(n, num_points, replace=False)
    elif n < num_points:
        indices = np.random.choice(n, num_points, replace=True)
    else:
        indices = np.arange(n)

    return pts_world[indices].astype(np.float32)


def crop_target_from_seg(
    rgb: np.ndarray,
    seg_mask: np.ndarray,
    target_id: int,
    output_size: int = 224,
    padding: int = 5,
) -> Image.Image | None:
    """Crop the bounding box of the target object from RGB using segmentation mask.

    Args:
        rgb: (H, W, 3) uint8 RGB image.
        seg_mask: (H, W) int segmentation mask.
        target_id: Instance ID of target object.
        output_size: Resize crop to this square size.
        padding: Pixels to pad around bounding box.

    Returns:
        PIL Image of size (output_size, output_size), or None if target not visible.
    """
    mask = seg_mask == target_id
    if not np.any(mask):
        return None

    rows, cols = np.where(mask)
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()

    H, W = rgb.shape[:2]
    r_min = max(0, r_min - padding)
    r_max = min(H - 1, r_max + padding)
    c_min = max(0, c_min - padding)
    c_max = min(W - 1, c_max + padding)

    crop = rgb[r_min : r_max + 1, c_min : c_max + 1]
    img = Image.fromarray(crop)
    return img.resize((output_size, output_size), Image.BILINEAR)


def extract_instruction_from_filename(filename: str) -> str:
    """Extract natural language instruction from LIBERO task filename.

    Pattern: {SCENE_PREFIX}_{instruction}_demo.hdf5
    Example: LIVING_ROOM_SCENE2_pick_up_the_black_bowl_..._demo.hdf5
             -> "pick up the black bowl ..."
    """
    stem = Path(filename).stem  # strip .hdf5
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]

    # Try to strip scene prefix (e.g., LIVING_ROOM_SCENE2_)
    match = re.match(r"^.*?SCENE\d+_", stem)
    if match:
        stem = stem[match.end() :]

    return stem.replace("_", " ")
