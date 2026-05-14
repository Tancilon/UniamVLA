from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_uamvla_oft_module(monkeypatch):
    class _Registry:
        def register(self, _name):
            return lambda cls: cls

    qwen = types.ModuleType("starVLA.model.framework.VLM4A.QwenOFT")
    qwen.Qwenvl_OFT = type("Qwenvl_OFT", (), {})
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.VLM4A.QwenOFT", qwen)

    helpers = types.ModuleType("starVLA.model.modules.uamvla.collator_helpers")
    helpers.stack_optional_tensor_fields = lambda *args, **kwargs: None
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

    torch = types.ModuleType("torch")
    torch.inference_mode = lambda: (lambda fn: fn)
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
