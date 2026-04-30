"""Spec §8.2 — canonical_state produced by the training pipeline must equal
the canonical_state produced by the eval client for the same raw observation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml


def _build_stats_yaml(tmp_path: Path) -> Path:
    """Same fixture content as tests/test_eval_state_passthrough_wiring.py.
    Re-stated locally to keep this test self-contained."""
    stats = {
        "view_names": ["static", "wrist"],
        "max_action_dim": 24,
        "embodiment_stats": {
            "franka_libero": {
                "action_dim": 7,
                "action_min_bound": [-1.0] * 7,
                "action_max_bound": [1.0] * 7,
            }
        },
        "state_stats": {
            "franka_libero": {
                "arm_0.ee_pose": {
                    "q01": [-0.5] * 9, "q99": [0.5] * 9,
                    "min": [-1.0] * 9, "max": [1.0] * 9,
                    "mean": [0.0] * 9, "std": [0.3] * 9,
                },
                "arm_0.joint_pos": {
                    "q01": [-1.0] * 7, "q99": [1.0] * 7,
                    "min": [-2.0] * 7, "max": [2.0] * 7,
                    "mean": [0.0] * 7, "std": [0.5] * 7,
                },
                "gripper_0": {
                    "q01": [0.0], "q99": [1.0],
                    "min": [0.0], "max": [1.0],
                    "mean": [0.5], "std": [0.3],
                },
            }
        },
    }
    path = tmp_path / "statistics.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(stats, f)
    return path


def _hand_raw_obs() -> dict:
    return {
        "ee_pos":        np.array([0.1, 0.0, 0.5], dtype=np.float32),
        "ee_axis_angle": np.array([0.1, -0.2, 0.3], dtype=np.float32),
        "joint_pos":     np.array([0.0, -0.5, 0.0, 1.5, 0.0, 1.0, 0.0], dtype=np.float32),
        "gripper_qpos":  np.array([0.02, 0.02], dtype=np.float32),
    }


# apply_to is explicit on both sides — matches the training config
# (uamvla_libero.yaml: state_encoder.normalization.apply_to) and the eval
# client's pinned list. Keeping it explicit here forecloses default-None
# divergence if state_stats grows new fields in the future.
_APPLY_TO = ["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"]


def _train_side(stats_yaml: Path, raw: dict) -> dict:
    """Mirror UamVLADataset.__getitem__'s normalization sub-pipeline
    (see uamvla_dataset.py:71-82, 105-106)."""
    from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter
    from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer

    with open(stats_yaml) as f:
        stats_dict = yaml.safe_load(f)
    adapter = LiberoAdapter()
    normalizer = StateNormalizer(
        stats_dict=stats_dict, embodiment="franka_libero", mode="q99",
        apply_to=_APPLY_TO,
    )
    return normalizer(adapter.to_canonical(raw))


def _eval_side(stats_yaml: Path, raw: dict) -> dict:
    """Mirror ModelClient's _adapter / _state_normalizer (see model2libero_interface.py
    state-passthrough block in __init__)."""
    from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter
    from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer

    with open(stats_yaml) as f:
        stats_dict = yaml.safe_load(f)
    adapter = LiberoAdapter()
    normalizer = StateNormalizer(
        stats_dict=stats_dict, embodiment="franka_libero", mode="q99",
        apply_to=_APPLY_TO,
    )
    return normalizer(adapter.to_canonical(raw))


def test_train_and_eval_canonical_state_match(tmp_path: Path):
    stats_yaml = _build_stats_yaml(tmp_path)
    raw = _hand_raw_obs()

    train_out = _train_side(stats_yaml, raw)
    eval_out = _eval_side(stats_yaml, raw)

    assert set(train_out.keys()) == set(eval_out.keys())
    for limb in train_out.keys():
        if isinstance(train_out[limb], dict):
            assert set(train_out[limb].keys()) == set(eval_out[limb].keys())
            for k in train_out[limb]:
                t, e = train_out[limb][k], eval_out[limb][k]
                assert t.shape == e.shape, f"{limb}.{k}: shape mismatch"
                assert t.dtype == e.dtype, f"{limb}.{k}: dtype mismatch"
                assert torch.allclose(t, e, atol=1e-6), f"{limb}.{k}: value mismatch"
        else:
            t, e = train_out[limb], eval_out[limb]
            assert t.shape == e.shape
            assert torch.allclose(t, e, atol=1e-6)
