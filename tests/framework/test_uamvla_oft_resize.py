from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image


def _load_uamvla_oft_module(monkeypatch):
    class _Registry:
        def register(self, _name):
            return lambda cls: cls

    qwen_oft = types.ModuleType("starVLA.model.framework.VLM4A.QwenOFT")
    qwen_oft.Qwenvl_OFT = object
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.VLM4A.QwenOFT", qwen_oft)

    collator_helpers = types.ModuleType("starVLA.model.modules.uamvla.collator_helpers")
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
