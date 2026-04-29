"""Smoke test: ported utility modules import and expose expected callables."""
import numpy as np


def test_geometry_imports():
    from starVLA.utils.geometry import (
        crop_target_from_seg,
        depth_to_world_points,
        extract_instruction_from_filename,
        get_camera_intrinsic_from_fovy,
        linearize_depth,
        mat_to_6d,
    )
    R = np.eye(3, dtype=np.float32)
    sixd = mat_to_6d(R)
    assert len(sixd) == 6


def test_rotation_imports():
    from starVLA.utils.rotation import quat_to_6d, euler_to_6d, axis_angle_to_6d
    aa = np.zeros(3, dtype=np.float32)
    sixd = axis_angle_to_6d(aa)
    assert sixd.shape == (6,)


def test_point_cloud_imports():
    from starVLA.utils.point_cloud import clean_point_cloud
    pts = np.random.randn(1024, 3).astype(np.float32)
    cleaned = clean_point_cloud(pts)
    assert cleaned.shape[1] == 3
