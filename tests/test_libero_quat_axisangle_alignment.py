"""P2 probe (spec §8.1): verify _quat2axisangle(robot0_eef_quat) matches HDF5 obs/ee_ori.

Skipped automatically if libero_env deps are missing. Run inside libero_env when
testing the design end-to-end:

    pytest tests/test_libero_quat_axisangle_alignment.py -v
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

libero = pytest.importorskip("libero.libero")
h5py = pytest.importorskip("h5py")


# Canonical robosuite quat→axisangle (mirrors examples/LIBERO/eval_files/eval_libero.py:_quat2axisangle).
def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


HDF5_GLOB_HINT = "dataset/libero2uam/raw/libero_spatial/**/*.hdf5"


@pytest.mark.libero_env
def test_eval_quat2axisangle_matches_hdf5_ee_ori(tmp_path):
    candidates = list(Path(".").glob(HDF5_GLOB_HINT))
    if not candidates:
        pytest.skip(f"No LIBERO HDF5 files matching {HDF5_GLOB_HINT}")
    h5_path = candidates[0]

    with h5py.File(h5_path, "r") as f:
        demo = next(iter(f["data"].values()))
        ee_ori = demo["obs/ee_ori"][:]   # (T, 3) axis-angle from HDF5
        ee_quat = demo["obs/ee_states"][:, 3:7] if "ee_states" in demo["obs"] else demo["obs/ee_quat"][:]

    n = min(10, ee_ori.shape[0])
    diffs = []
    for t in range(n):
        recovered = _quat2axisangle(ee_quat[t])
        diffs.append(np.abs(recovered - ee_ori[t]).max())

    max_diff = max(diffs)
    assert max_diff < 1e-5, (
        f"_quat2axisangle output diverges from HDF5 obs/ee_ori "
        f"(max abs diff over {n} timesteps = {max_diff:.3e}; tol 1e-5)"
    )
