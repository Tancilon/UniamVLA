from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np


def _load_uamvla_oft_module(monkeypatch):
    class _Registry:
        def register(self, _name):
            return lambda cls: cls

    qwen = types.ModuleType("starVLA.model.framework.VLM4A.QwenOFT")
    qwen.Qwenvl_OFT = type("Qwenvl_OFT", (), {})
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.VLM4A.QwenOFT", qwen)

    helpers = types.ModuleType("starVLA.model.modules.uamvla.collator_helpers")
    helpers.stack_optional_tensor_fields = lambda *args, **kwargs: None
    helpers.stack_optional_string_fields = lambda *args, **kwargs: {}
    helpers.stack_pose_gt = lambda *args, **kwargs: None
    helpers.stack_static_cam_extrinsic = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "starVLA.model.modules.uamvla.collator_helpers",
        helpers,
    )

    tools = types.ModuleType("starVLA.model.tools")
    tools.FRAMEWORK_REGISTRY = _Registry()
    monkeypatch.setitem(sys.modules, "starVLA.model.tools", tools)

    pose_utils = types.ModuleType(
        "starVLA.model.modules.uamvla.components.pose.pose_utils",
    )
    pose_utils.rotation_6d_to_matrix = lambda value: value
    monkeypatch.setitem(
        sys.modules,
        "starVLA.model.modules.uamvla.components.pose.pose_utils",
        pose_utils,
    )

    class _FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        @property
        def ndim(self):
            return self.array.ndim

        @property
        def shape(self):
            return self.array.shape

        def squeeze(self, axis=None):
            return _FakeTensor(np.squeeze(self.array, axis=axis))

        def to(self, dtype=None):
            if dtype is None:
                return self
            return _FakeTensor(self.array.astype(dtype))

        def __getitem__(self, item):
            return _FakeTensor(self.array[item])

        def permute(self, *dims):
            return _FakeTensor(np.transpose(self.array, axes=dims))

        def float(self):
            return _FakeTensor(self.array.astype(np.float32))

        def __truediv__(self, value):
            return _FakeTensor(self.array / value)

    torch = types.ModuleType("torch")
    torch.inference_mode = lambda: (lambda fn: fn)
    torch.float32 = np.float32
    torch.is_tensor = lambda value: isinstance(value, _FakeTensor)
    torch.as_tensor = lambda value, dtype=None: _FakeTensor(
        np.asarray(value, dtype=dtype),
    )
    torch.from_numpy = lambda value: _FakeTensor(np.asarray(value))
    nn = types.ModuleType("torch.nn")
    torch.nn = nn
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)

    path = Path("starVLA/model/framework/VLM4A/UamVLAOFT.py")
    spec = importlib.util.spec_from_file_location("uamvla_oft_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unpack_lerobot_sample_loads_image_target_by_trajectory_and_base(
    tmp_path,
    monkeypatch,
):
    module = _load_uamvla_oft_module(monkeypatch)
    target_dir = tmp_path / "image_targets" / "3"
    target_dir.mkdir(parents=True)
    module.Image.new("RGB", (2, 2), color=(10, 20, 30)).save(
        target_dir / "0.png",
    )
    module.Image.new("RGB", (2, 2), color=(40, 50, 60)).save(
        target_dir / "1.png",
    )

    model = object.__new__(module.UamVLAOFT)
    model.sidecar_root = tmp_path
    model._image_target_cache = OrderedDict()
    model._image_target_cache_maxsize = 8
    model._load_image_future = lambda traj, base: None
    model._load_image_action_future = lambda traj, base: None
    model.aux_state_slice = {
        "target_pose_rot6d": (15, 21),
        "target_pose_trans": (21, 24),
        "static_cam_rot6d": (24, 30),
        "static_cam_trans": (30, 33),
    }
    sample = {
        "image": [module.Image.new("RGB", (4, 4))],
        "lang": "open drawer",
        "action": np.zeros((1, 7), dtype=np.float32),
        "state": np.zeros((1, 33), dtype=np.float32),
        "__trajectory_id": 3,
        "__base_index": 1,
    }

    out = module.UamVLAOFT._unpack_lerobot_sample(model, sample)

    assert "image_target" in out
    assert out["image_target"].array.shape == (3, 2, 2)
    np.testing.assert_allclose(
        out["image_target"].array[:, 0, 0],
        np.array([40, 50, 60], dtype=np.float32) / 255.0,
    )
    assert list(model._image_target_cache.keys()) == [(3, 1)]


def test_image_future_video_path_uses_lerobot_chunk_metadata(
    tmp_path,
    monkeypatch,
):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps({
            "chunks_size": 1000,
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
        })
    )

    module = _load_uamvla_oft_module(monkeypatch)
    model = object.__new__(module.UamVLAOFT)
    model.sidecar_root = tmp_path

    model._init_lerobot_video_path_config()

    assert model._image_future_video_path(999) == (
        tmp_path / "videos" / "chunk-000" / "video.primary_image" /
        "episode_000999.mp4"
    )
    assert model._image_future_video_path(1000) == (
        tmp_path / "videos" / "chunk-001" / "video.primary_image" /
        "episode_001000.mp4"
    )


def test_image_future_frame_index_is_episode_terminal_frame(monkeypatch):
    module = _load_uamvla_oft_module(monkeypatch)

    assert module.UamVLAOFT._image_future_frame_index(video_length=65) == 64
    assert module.UamVLAOFT._image_future_frame_index(video_length=1) == 0
    assert module.UamVLAOFT._image_future_frame_index(video_length=0) is None


def test_aux_metric_log_key_drops_duplicate_head_prefix(monkeypatch):
    module = _load_uamvla_oft_module(monkeypatch)

    assert module.UamVLAOFT._aux_metric_log_key("recon", "recon_loss") == (
        "recon_loss_raw"
    )
    assert module.UamVLAOFT._aux_metric_log_key("future", "future_loss") == (
        "future_loss_raw"
    )
    assert module.UamVLAOFT._aux_metric_log_key("pose", "pose_loss") == (
        "pose_loss_raw"
    )
    assert module.UamVLAOFT._aux_metric_log_key(
        "pose", "pose_translation_residual_mean"
    ) == "pose_translation_residual_mean_raw"


def test_visualize_batch_builds_wandb_images_without_heavy_model(monkeypatch):
    module = _load_uamvla_oft_module(monkeypatch)

    class _WandbImage:
        def __init__(self, image, caption=None):
            self.image = image
            self.caption = caption

    wandb = types.ModuleType("wandb")
    wandb.Image = _WandbImage
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    class _FakeModel(module.UamVLAOFT):
        action_horizon = 2

        def __init__(self):
            self.training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def _visualization_forward_context(self, selected):
            return (
                selected,
                module.np.ones(
                    (len(selected), self.action_horizon, 7), dtype=module.np.float32,
                ),
                None,
                {},
            )

    sample = {
        "image": [module.Image.new("RGB", (24, 16), color=(255, 0, 0))],
        "lang": "open the drawer",
        "action": module.np.zeros((2, 7), dtype=module.np.float32),
    }

    model = _FakeModel()
    out = module.UamVLAOFT.visualize_batch(model, [sample], n_samples=1)

    assert list(out) == ["viz/uamvla_oft/action_0"]
    logged = out["viz/uamvla_oft/action_0"]
    assert logged.caption == "open the drawer"
    assert logged.image.mode == "RGB"
    assert logged.image.width > 24
    assert logged.image.height > 16
    assert model.training is True


def test_collect_aux_head_visualizations_dispatches_all_enabled_heads(monkeypatch):
    module = _load_uamvla_oft_module(monkeypatch)

    class _Hidden:
        shape = (2, 4, 8)
        device = "cpu"

    class _Head:
        def __init__(self, name):
            self.name = name
            self.camera_params = {"intrinsic": {"fx": 1.0}} if name == "pose" else None
            self.calls = []

        def visualize(self, hidden_states, batch, mask, num_samples, **kwargs):
            self.calls.append((hidden_states, batch, mask, num_samples, kwargs))
            return [f"{self.name}-image"]

    model = object.__new__(module.UamVLAOFT)
    model.aux_heads = {
        "recon": _Head("recon"),
        "future": _Head("future"),
        "pose": _Head("pose"),
    }
    model._resolve_head_mask = (
        lambda head_name, batch, batch_size, device: batch[f"{head_name}_mask"]
    )
    batch = {
        "recon_mask": "recon-mask",
        "future_mask": "future-mask",
        "pose_mask": "pose-mask",
    }

    out = {"viz/uamvla_oft/action_0": "action-image"}
    module.UamVLAOFT._collect_aux_head_visualizations(
        model, _Hidden(), batch, num_samples=1, outputs=out,
    )

    assert out == {
        "viz/uamvla_oft/action_0": "action-image",
        "viz/uamvla_oft/recon_0": "recon-image",
        "viz/uamvla_oft/future_0": "future-image",
        "viz/uamvla_oft/pose_0": "pose-image",
    }
    assert model.aux_heads["recon"].calls[0][2] == "recon-mask"
    assert model.aux_heads["future"].calls[0][2] == "future-mask"
    assert model.aux_heads["pose"].calls[0][2] == "pose-mask"
    assert model.aux_heads["pose"].calls[0][4]["camera_params"] == (
        model.aux_heads["pose"].camera_params
    )
