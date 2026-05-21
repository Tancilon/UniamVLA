import numpy as np
import torch

from starVLA.model.framework.VLM4A.UamVLAOFT import UamVLAOFT


def test_aux_sidecar_paths_and_collate_masks(tmp_path):
    model = object.__new__(UamVLAOFT)
    model.sidecar_root = tmp_path
    model.aux_state_slice = {
        "target_pose_rot6d": [15, 21],
        "target_pose_trans": [21, 24],
        "static_cam_rot6d": [24, 30],
        "static_cam_trans": [30, 33],
    }
    model._image_target_cache = {}
    model._load_image_future = lambda traj, base: None
    model._load_image_action_future = lambda traj, base: torch.ones(3, 8, 8)

    for subdir, value in [
        ("depths/static/3", np.ones((4, 4), dtype=np.float32)),
        ("grounding_masks/static/3", np.ones((1, 20, 20), dtype=np.float32)),
        ("affordance_heatmaps/static/3", np.ones((1, 20, 20), dtype=np.float32)),
    ]:
        path = tmp_path / subdir
        path.mkdir(parents=True)
        np.save(path / "7.npy", value)

    sample = {
        "__trajectory_id": 3,
        "__base_index": 7,
        "image": [],
        "lang": "open drawer",
        "action": np.zeros((8, 7), dtype=np.float32),
        "state": np.zeros((1, 33), dtype=np.float32),
    }
    out = UamVLAOFT._unpack_lerobot_sample(model, sample)
    assert out["depth_target"].shape == (1, 4, 4)
    assert out["grounding_mask"].shape == (1, 20, 20)
    assert out["grounding_level"] == "object"
    assert out["affordance_heatmap"].shape == (1, 20, 20)
    assert out["image_action_future"].shape == (3, 8, 8)

    batch = UamVLAOFT._collate_aux(
        model,
        [out, {"action": torch.zeros(8, 7)}],
        {"input_ids": torch.zeros(2, 4, dtype=torch.long)},
    )
    assert torch.equal(batch["depth_mask"], torch.tensor([True, False]))
    assert torch.equal(batch["grounding_mask_mask"], torch.tensor([True, False]))
    assert torch.equal(batch["affordance_mask"], torch.tensor([True, False]))
    assert torch.equal(batch["action_conditioned_future_mask"], torch.tensor([True, False]))
    assert batch["grounding_level"] == ["object", ""]
