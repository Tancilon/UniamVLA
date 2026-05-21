import sys
import types

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.UamVLAOFT import UamVLAOFT


class _AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class _Tokenizer:
    @staticmethod
    def convert_tokens_to_ids(_token):
        return 99


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


def test_grounding_head_key_matches_grounding_mask_presence_mask(monkeypatch):
    class _FakePoseHead(nn.Module):
        pass

    class _FakeGroundingHead(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.mask_key = "grounding_mask_mask"

    aux_heads_pkg = types.ModuleType("starVLA.model.modules.uamvla.aux_heads")
    aux_heads_pkg.__path__ = []
    pose_module = types.ModuleType("starVLA.model.modules.uamvla.aux_heads.pose_head")
    pose_module.PoseHead = _FakePoseHead
    grounding_module = types.ModuleType(
        "starVLA.model.modules.uamvla.aux_heads.grounding_head"
    )
    grounding_module.GroundingMaskDenoisingHead = _FakeGroundingHead
    monkeypatch.setitem(sys.modules, aux_heads_pkg.__name__, aux_heads_pkg)
    monkeypatch.setitem(sys.modules, pose_module.__name__, pose_module)
    monkeypatch.setitem(
        sys.modules,
        grounding_module.__name__,
        grounding_module,
    )

    model = object.__new__(UamVLAOFT)
    nn.Module.__init__(model)
    model.aux_heads = nn.ModuleDict()
    model.qwen_vl_interface = _AttrDict(
        image_token_id=99,
        processor=_AttrDict(tokenizer=_Tokenizer()),
        model=_AttrDict(config=_AttrDict(hidden_size=4)),
    )
    model.config = _AttrDict(
        framework=_AttrDict(
            aux_heads=_AttrDict(
                grounding=_AttrDict(enabled=True),
            ),
        ),
    )

    UamVLAOFT._maybe_build_aux_heads(model)

    assert list(model.aux_heads.keys()) == ["grounding_mask"]
    batch = {
        "grounding_mask": torch.ones(2, 1, 20, 20),
        "grounding_mask_mask": torch.tensor([True, False]),
    }
    mask = UamVLAOFT._resolve_head_mask(
        model,
        "grounding_mask",
        batch,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert torch.equal(mask, torch.tensor([True, False]))
