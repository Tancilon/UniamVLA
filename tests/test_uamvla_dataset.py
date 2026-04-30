# tests/test_uamvla_dataset.py
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

# Real test dataset (has state_stats block needed for normalization tests)
_REAL_DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "uamvla_test" / "libero_spatial"


@pytest.fixture
def sample_dataset_dir():
    """Return path to the real UamVLA test dataset (libero_spatial)."""
    if not _REAL_DATASET_DIR.exists():
        pytest.skip(f"Real test dataset not found at {_REAL_DATASET_DIR}")
    return _REAL_DATASET_DIR


@pytest.fixture
def mock_jsonl_dir(tmp_path):
    """Create a minimal preprocessor-output-shaped directory."""
    sample = {
        "id": "test_00_ep0000_step0000",
        "episode_id": "test_00_ep0000",
        "step_idx": 0,
        "total_steps": 5,
        "image": ["images/obs/static/test_00_ep0000_step0000.jpg",
                  "images/obs/wrist/test_00_ep0000_step0000.jpg"],
        "instruction": "test task",
        "embodiment": "franka_libero",
        "action_dim": 7,
        "action": [0.1] * 7 + [0.0] * 17,
        "action_mask": [1] * 7 + [0] * 17,
        "ee_pos": [0.0, 0.0, 0.0],
        "ee_axis_angle": [0.0, 0.0, 0.0],
        "joint_pos": [0.0] * 7,
        "gripper_qpos": [0.04, 0.04],
        "dataset_source": "test",
    }
    (tmp_path / "data.jsonl").write_text(json.dumps(sample) + "\n")
    stats = {
        "view_names": ["static", "wrist"],
        "max_action_dim": 24,
        "embodiment_stats": {
            "franka_libero": {
                "action_dim": 7,
                "action_min_bound": [-1.0] * 7,
                "action_max_bound": [1.0] * 7,
            },
        },
        # Minimal state_stats required by StateNormalizer (default mode=q99)
        "state_stats": {
            "franka_libero": {
                "arm_0.ee_pose": {
                    "q01": [-1.0] * 9,
                    "q99": [1.0] * 9,
                    "min": [-1.0] * 9,
                    "max": [1.0] * 9,
                    "mean": [0.0] * 9,
                    "std": [1.0] * 9,
                },
                "arm_0.joint_pos": {
                    "q01": [-1.0] * 7,
                    "q99": [1.0] * 7,
                    "min": [-1.0] * 7,
                    "max": [1.0] * 7,
                    "mean": [0.0] * 7,
                    "std": [1.0] * 7,
                },
                "gripper_0": {
                    "q01": [0.0],
                    "q99": [1.0],
                    "min": [0.0],
                    "max": [1.0],
                    "mean": [0.5],
                    "std": [0.5],
                },
            },
        },
    }
    import yaml
    (tmp_path / "statistics.yaml").write_text(yaml.dump(stats))
    return tmp_path


def test_dataset_getitem_yields_expected_fields(mock_jsonl_dir, monkeypatch):
    from starVLA.dataloader.uamvla_dataset import UamVLADataset

    # Mock image loading (real .jpg fixtures not present)
    from PIL import Image
    monkeypatch.setattr(Image, "open", lambda p: Image.new("RGB", (640, 640)))

    ds = UamVLADataset(data_root=mock_jsonl_dir, embodiment="franka_libero", action_horizon=8)
    sample = ds[0]
    # Required fields per spec §4.6
    assert "image" in sample and len(sample["image"]) == 2
    assert "lang" in sample
    assert "action" in sample and sample["action"].shape == (8, 7)
    assert "canonical_state" in sample
    assert "arm_0" in sample["canonical_state"]
    assert "ee_pose" in sample["canonical_state"]["arm_0"]
    # Action normalization: with min=-1, max=1, normalized = action (no change in [-1,1] target range)
    np.testing.assert_allclose(sample["action"][0], [0.1] * 7, atol=1e-5)


def test_uamvla_collate_fn_stacks_correctly(mock_jsonl_dir, monkeypatch):
    from starVLA.dataloader.uamvla_dataset import UamVLADataset, collate_fn
    from PIL import Image
    monkeypatch.setattr(Image, "open", lambda p: Image.new("RGB", (640, 640)))

    ds = UamVLADataset(mock_jsonl_dir, embodiment="franka_libero", action_horizon=8)
    samples = [ds[0]]  # batch of 1
    batch = collate_fn(samples)
    # Phase 1: collate returns the list (framework handles stacking) — verify it's a list
    assert isinstance(batch, list)
    assert len(batch) == 1


def test_canonical_state_normalized_to_unit_range(sample_dataset_dir):
    """After normalization, every field in canonical_state should be in [-1, 1]."""
    import torch
    from starVLA.dataloader.uamvla_dataset import UamVLADataset

    ds = UamVLADataset(
        data_root=sample_dataset_dir,
        embodiment="franka_libero",
        action_horizon=8,
        normalization={
            "mode": "q99",
            "apply_to": ["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"],
        },
    )
    sample = ds[0]
    cs = sample["canonical_state"]
    for field in (cs["arm_0"]["ee_pose"], cs["arm_0"]["joint_pos"], cs["gripper_0"]):
        assert torch.all(field >= -1.0 - 1e-6), f"value < -1: {field}"
        assert torch.all(field <= 1.0 + 1e-6), f"value > 1: {field}"


def test_canonical_state_mode_none_passes_raw_values(sample_dataset_dir):
    """mode=none must NOT modify canonical_state values."""
    import torch
    from starVLA.dataloader.uamvla_dataset import UamVLADataset
    from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter

    ds = UamVLADataset(
        data_root=sample_dataset_dir,
        embodiment="franka_libero",
        action_horizon=8,
        normalization={"mode": "none", "apply_to": []},
    )
    sample = ds[0]
    raw = ds.samples[0]
    expected = LiberoAdapter().to_canonical(raw)
    assert torch.allclose(sample["canonical_state"]["arm_0"]["ee_pose"],
                          expected["arm_0"]["ee_pose"])
