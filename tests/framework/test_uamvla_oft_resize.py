from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image


class _AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _load_uamvla_oft_module(monkeypatch):
    class _Registry:
        def register(self, _name):
            return lambda cls: cls

    qwen_oft = types.ModuleType("starVLA.model.framework.VLM4A.QwenOFT")
    qwen_oft.Qwenvl_OFT = object
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.VLM4A.QwenOFT", qwen_oft)

    collator_helpers = types.ModuleType("starVLA.model.modules.uamvla.collator_helpers")
    collator_helpers.stack_optional_string_fields = lambda *a, **kw: None
    collator_helpers.stack_optional_tensor_fields = lambda *a, **kw: None
    collator_helpers.stack_pose_gt = lambda *a, **kw: None
    collator_helpers.stack_static_cam_extrinsic = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "starVLA.model.modules.uamvla.collator_helpers", collator_helpers)

    tools = types.ModuleType("starVLA.model.tools")
    tools.FRAMEWORK_REGISTRY = _Registry()
    monkeypatch.setitem(sys.modules, "starVLA.model.tools", tools)

    path = Path("starVLA/model/framework/VLM4A/UamVLAOFT.py")
    spec = importlib.util.spec_from_file_location("uamvla_oft_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _model_with_image_size(module, image_size):
    model = object.__new__(module.UamVLAOFT)
    model.config = _AttrDict(
        framework=_AttrDict(
            obs_image_size=[image_size, image_size],
        ),
    )
    return model


def test_force_resize_640_converts_numpy_arrays_to_640_pil(monkeypatch):
    module = _load_uamvla_oft_module(monkeypatch)
    model = object.__new__(module.UamVLAOFT)

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    out = module.UamVLAOFT._force_resize_640(model, [image])

    assert isinstance(out[0], Image.Image)
    assert out[0].size == (640, 640)


def test_force_resize_640_keeps_existing_640_pil(monkeypatch):
    module = _load_uamvla_oft_module(monkeypatch)
    model = object.__new__(module.UamVLAOFT)

    image = Image.new("RGB", (640, 640))
    out = module.UamVLAOFT._force_resize_640(model, [image])

    assert out[0] is image


def test_force_resize_uses_configured_320_layout(monkeypatch):
    module = _load_uamvla_oft_module(monkeypatch)
    model = _model_with_image_size(module, 320)

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    out = module.UamVLAOFT._force_resize_640(model, [image])

    assert isinstance(out[0], Image.Image)
    assert out[0].size == (320, 320)


def test_qwen_vision_layout_derives_320_aux_grid(monkeypatch):
    module = _load_uamvla_oft_module(monkeypatch)
    model = _model_with_image_size(module, 320)

    layout = module.UamVLAOFT._qwen_vision_layout(model)

    assert layout == {
        "image_size": 320,
        "grid_size": 10,
        "patches_per_view": 100,
        "target_size": 10,
        "target_resize": 160,
    }
