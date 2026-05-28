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
    model._load_image_future = lambda traj, base, sidecar_root=None: None
    model._load_image_action_future = (
        lambda traj, base, sidecar_root=None: torch.ones(3, 8, 8)
    )

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
    assert torch.equal(batch["pose_mask"], torch.tensor([False, False]))
    assert torch.equal(batch["depth_mask"], torch.tensor([True, False]))
    assert torch.equal(batch["grounding_mask_mask"], torch.tensor([True, False]))
    assert torch.equal(batch["affordance_mask"], torch.tensor([True, False]))
    assert torch.equal(batch["action_conditioned_future_mask"], torch.tensor([True, False]))
    assert batch["grounding_level"] == ["object", ""]


def test_aux_sidecar_lookup_uses_dataset_name_root(tmp_path):
    model = object.__new__(UamVLAOFT)
    wrong_root = tmp_path / "lerobot_libero_spatial"
    right_root = tmp_path / "lerobot_libero_goal"
    model.sidecar_root = wrong_root
    model.sidecar_roots = {
        "lerobot_libero_spatial": wrong_root,
        "lerobot_libero_goal": right_root,
    }
    model.aux_state_slice = {
        "target_pose_rot6d": [15, 21],
        "target_pose_trans": [21, 24],
        "static_cam_rot6d": [24, 30],
        "static_cam_trans": [30, 33],
    }
    model._image_target_cache = {}
    model._image_target_cache_maxsize = 16
    model._load_image_future = lambda traj, base, sidecar_root=None: None
    model._load_image_action_future = lambda traj, base, sidecar_root=None: None

    pc_dir = right_root / "point_clouds" / "3"
    pc_dir.mkdir(parents=True)
    expected = np.full((1024, 3), 7.0, dtype=np.float32)
    np.save(pc_dir / "5.npy", expected)

    sample = {
        "__dataset_name": "lerobot_libero_goal",
        "__trajectory_id": 3,
        "__base_index": 5,
        "image": [],
        "lang": "open drawer",
        "action": np.zeros((8, 7), dtype=np.float32),
        "state": np.zeros((1, 33), dtype=np.float32),
    }

    out = UamVLAOFT._unpack_lerobot_sample(model, sample)

    assert "point_cloud" in out
    assert torch.equal(out["point_cloud"], torch.as_tensor(expected))


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


def test_aux_heads_use_configured_320_vision_layout(monkeypatch):
    class _FakePoseHead(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

    class _FakeHead(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

    class _FakeVAE(nn.Module):
        latent_channels = 4

        def __init__(self, _path):
            super().__init__()

    def _install_module(name, attr_name, cls):
        module = types.ModuleType(name)
        setattr(module, attr_name, cls)
        monkeypatch.setitem(sys.modules, name, module)

    aux_heads_pkg = types.ModuleType("starVLA.model.modules.uamvla.aux_heads")
    aux_heads_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, aux_heads_pkg.__name__, aux_heads_pkg)
    _install_module("starVLA.model.modules.uamvla.aux_heads.pose_head", "PoseHead", _FakePoseHead)
    _install_module("starVLA.model.modules.uamvla.aux_heads.future_head", "FutureHead", _FakeHead)
    _install_module("starVLA.model.modules.uamvla.aux_heads.recon_head", "ReconHead", _FakeHead)
    _install_module(
        "starVLA.model.modules.uamvla.aux_heads.depth_head",
        "DepthDenoisingHead",
        _FakeHead,
    )
    _install_module(
        "starVLA.model.modules.uamvla.aux_heads.grounding_head",
        "GroundingMaskDenoisingHead",
        _FakeHead,
    )
    _install_module(
        "starVLA.model.modules.uamvla.aux_heads.affordance_head",
        "AffordanceHeatmapDenoisingHead",
        _FakeHead,
    )
    _install_module(
        "starVLA.model.modules.uamvla.aux_heads.action_conditioned_future_head",
        "ActionConditionedFutureHead",
        _FakeHead,
    )
    _install_module(
        "starVLA.model.modules.uamvla.components.pixel_decoder.vae",
        "VAEPixelDecoder",
        _FakeVAE,
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
            obs_image_size=[320, 320],
            vae=_AttrDict(path="fake-vae"),
            action_model=_AttrDict(action_dim=7, action_horizon=8),
            aux_heads=_AttrDict(
                future=_AttrDict(enabled=True, target_resize=320, n_patches=400),
                recon=_AttrDict(enabled=True, target_resize=320, n_patches=400),
                depth=_AttrDict(enabled=True, target_size=20),
                grounding=_AttrDict(enabled=True, target_size=20),
                affordance=_AttrDict(enabled=True, target_size=20),
                action_conditioned_future=_AttrDict(
                    enabled=True,
                    target_resize=320,
                    n_patches=400,
                ),
            ),
        ),
    )

    UamVLAOFT._maybe_build_aux_heads(model)

    assert model.aux_heads["future"].kwargs["patches_per_view"] == 100
    assert model.aux_heads["future"].kwargs["n_patches"] == 100
    assert model.aux_heads["future"].kwargs["target_resize"] == 160
    assert model.aux_heads["recon"].kwargs["patches_per_view"] == 100
    assert model.aux_heads["recon"].kwargs["n_patches"] == 100
    assert model.aux_heads["recon"].kwargs["target_resize"] == 160
    assert model.aux_heads["depth"].kwargs["patches_per_view"] == 100
    assert model.aux_heads["depth"].kwargs["target_size"] == 10
    assert model.aux_heads["grounding_mask"].kwargs["patches_per_view"] == 100
    assert model.aux_heads["grounding_mask"].kwargs["target_size"] == 10
    assert model.aux_heads["affordance"].kwargs["patches_per_view"] == 100
    assert model.aux_heads["affordance"].kwargs["target_size"] == 10
    assert model.aux_heads["action_conditioned_future"].kwargs["patches_per_view"] == 100
    assert model.aux_heads["action_conditioned_future"].kwargs["n_patches"] == 100
    assert model.aux_heads["action_conditioned_future"].kwargs["target_resize"] == 160
