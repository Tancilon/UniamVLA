"""Local unit tests for the UamVLA CALVIN preprocessing + training pipeline.

These tests run on any machine without requiring calvin_env / PyBullet.
Coverage:
  - statistics.yaml schema migration (Task 1)
  - mixture registry entry (Task 2)
  - synthetic dataset → UamVLADataset chain (Task 3)
  - CLI arg parsing (Task 4)
  - training yaml loadable (Task 5)
"""
from __future__ import annotations

import json         # noqa: F401 — used by Task 3 fixture builder
import subprocess   # noqa: F401 — used by Task 4 CLI tests
import sys          # noqa: F401 — used by Task 4 CLI tests
from pathlib import Path

import numpy as np
import pytest      # noqa: F401 — used by Task 3 (importorskip) / Task 5 (importorskip)
import yaml

ROOT = Path(__file__).resolve().parent.parent  # noqa: F841 — used by Task 4 CLI_PATH


# ============================================================================
# Task 1: _write_statistics schema migration
# ============================================================================

def _make_synthetic_calvin_samples(n: int = 6) -> list[dict]:
    """Build n CALVIN-shape sample dicts. Values are deterministic but varied
    across samples so q01 != q99 (otherwise q99 normalization would degenerate
    to identity and downstream tests can't tell stats were applied).
    """
    rng = np.random.default_rng(seed=42)
    samples = []
    for i in range(n):
        # 15-dim robot_obs: tcp(3) + euler(3) + gripper_width(1) + joints(7) + gripper_cmd(1)
        # Ranges chosen to match the values we probed from a real CALVIN frame:
        #   tcp ≈ [0.10, -0.09, 0.57], euler ≈ [3.10, 0.02, 1.50], gripper_width ≈ 0.08,
        #   joints ≈ 7-tuple within [-π, π], gripper_cmd ∈ {-1, 1}
        tcp = np.array([0.10, -0.09, 0.57], dtype=np.float64) + 0.05 * rng.standard_normal(3)
        euler = np.array([3.10, 0.02, 1.50], dtype=np.float64) + 0.05 * rng.standard_normal(3)
        gripper_width = 0.08 + 0.005 * rng.standard_normal()
        joints = np.array([-0.59, 0.90, 1.74, -1.84, -0.92, 1.60, 0.59], dtype=np.float64) \
                 + 0.1 * rng.standard_normal(7)
        gripper_cmd = 1.0
        robot_obs_15 = np.concatenate([
            tcp, euler, [gripper_width], joints, [gripper_cmd],
        ]).tolist()

        rel_actions_7 = (0.01 * rng.standard_normal(7)).tolist()
        action_24 = rel_actions_7 + [0.0] * 17

        samples.append({
            "id": f"calvin_debug_ep00000_step{i:04d}",
            "episode_id": "calvin_debug_ep00000",
            "step_idx": i,
            "total_steps": n,
            "image": [
                f"images/obs/static/calvin_debug_ep00000_step{i:04d}.jpg",
                f"images/obs/wrist/calvin_debug_ep00000_step{i:04d}.jpg",
            ],
            "instruction": "test instruction",
            "embodiment": "franka_calvin",
            "action_dim": 7,
            "action": action_24,
            "action_mask": [1] * 7 + [0] * 17,
            "robot_obs": robot_obs_15,
            "dataset_source": "calvin_debug",
        })
    return samples


def test_write_statistics_schema(tmp_path):
    """`_write_statistics` emits the new state_stats[franka_calvin] block."""
    from tools.preprocess.calvin_preprocessor import _write_statistics

    samples = _make_synthetic_calvin_samples(n=6)
    intrinsics = {
        "static": {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
        "wrist":  {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
    }

    _write_statistics(samples, tmp_path, intrinsics)

    with open(tmp_path / "statistics.yaml") as f:
        stats = yaml.safe_load(f)

    # Top-level shape
    assert set(stats.keys()) == {
        "view_names", "max_action_dim", "embodiment_stats",
        "state_stats", "cameras", "point_cloud",
    }, f"unexpected top-level keys: {sorted(stats.keys())}"

    # state_stats[franka_calvin] presence + shape
    assert "franka_calvin" in stats["state_stats"]
    fs = stats["state_stats"]["franka_calvin"]
    assert set(fs.keys()) == {"arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"}, \
        f"unexpected state field keys: {sorted(fs.keys())}"

    expected_lengths = {"arm_0.ee_pose": 9, "arm_0.joint_pos": 7, "gripper_0": 1}
    expected_substats = {"q01", "q99", "min", "max", "mean", "std"}
    for path, length in expected_lengths.items():
        assert set(fs[path].keys()) == expected_substats, (
            f"{path}: substats {sorted(fs[path].keys())} != {sorted(expected_substats)}"
        )
        for k in expected_substats:
            assert len(fs[path][k]) == length, (
                f"{path}.{k}: len={len(fs[path][k])}, expected {length}"
            )


def test_write_statistics_no_legacy_keys(tmp_path):
    """`robot_obs_mean` / `robot_obs_std` are removed from the new schema."""
    from tools.preprocess.calvin_preprocessor import _write_statistics

    samples = _make_synthetic_calvin_samples(n=4)
    intrinsics = {
        "static": {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
        "wrist":  {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
    }

    _write_statistics(samples, tmp_path, intrinsics)

    with open(tmp_path / "statistics.yaml") as f:
        stats = yaml.safe_load(f)

    assert "robot_obs_mean" not in stats, \
        "Legacy key 'robot_obs_mean' must be removed in the new schema."
    assert "robot_obs_std" not in stats, \
        "Legacy key 'robot_obs_std' must be removed in the new schema."


# ============================================================================
# Task 2: mixture registry entry
# ============================================================================

def test_calvin_uamvla_mixture_registered():
    from starVLA.dataloader.uamvla_dataset import DATASET_NAMED_MIXTURES

    assert "calvin_uamvla" in DATASET_NAMED_MIXTURES, (
        f"Missing 'calvin_uamvla' mixture; available: {list(DATASET_NAMED_MIXTURES)}"
    )
    assert DATASET_NAMED_MIXTURES["calvin_uamvla"] == [
        ("task_D_D", 1.0, "franka_calvin"),
    ]
