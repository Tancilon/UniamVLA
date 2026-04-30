# tests/test_uamvla_dataset.py
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest


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
