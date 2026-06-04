from __future__ import annotations

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.transform.base import PreserveRawModalityTransform


def test_preserve_raw_modality_transform_copies_numpy_and_tensor_values():
    transform = PreserveRawModalityTransform(
        apply_to=["state.robot_obs", "action.x"],
        output_key="uamvla_raw_reconvla",
    )
    robot_obs = np.arange(15, dtype=np.float32).reshape(1, 15)
    action_x = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    data = {
        "state.robot_obs": robot_obs,
        "action.x": action_x,
    }

    out = transform(data)

    assert "uamvla_raw_reconvla" in out
    assert np.allclose(out["uamvla_raw_reconvla"]["state.robot_obs"], robot_obs)
    assert torch.allclose(out["uamvla_raw_reconvla"]["action.x"], action_x)

    robot_obs[...] = -1
    action_x.fill_(-2)
    assert not np.allclose(out["uamvla_raw_reconvla"]["state.robot_obs"], robot_obs)
    assert not torch.allclose(out["uamvla_raw_reconvla"]["action.x"], action_x)


def test_pack_sample_preserves_reconvla_raw_robot_obs_and_action(monkeypatch):
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

    dataset = object.__new__(LeRobotSingleDataset)
    dataset.data_cfg = {"image_resize": 8, "include_state": True}
    dataset._modality_keys = {
        "video": ["video.primary_image", "video.wrist_image"],
        "language": ["annotation.human.action.task_description"],
        "action": ["action.x", "action.y"],
        "state": ["state.robot_obs"],
    }

    image = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    data = {
        "video.primary_image": image,
        "video.wrist_image": image,
        "annotation.human.action.task_description": ["move"],
        "action.x": np.ones((5, 1), dtype=np.float32),
        "action.y": np.ones((5, 1), dtype=np.float32) * 2,
        "state.robot_obs": np.ones((1, 15), dtype=np.float32) * 3,
        "uamvla_raw_reconvla": {
            "action.x": np.ones((5, 1), dtype=np.float32) * 4,
            "action.y": np.ones((5, 1), dtype=np.float32) * 5,
            "state.robot_obs": np.arange(15, dtype=np.float32).reshape(1, 15),
        },
    }

    sample = LeRobotSingleDataset._pack_sample(dataset, data)

    assert sample["uamvla_raw_action"].shape == (5, 2)
    assert np.allclose(sample["uamvla_raw_action"][0], [4.0, 5.0])
    assert sample["uamvla_raw_state"]["robot_obs"].shape == (1, 15)
