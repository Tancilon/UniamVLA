from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


class _FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def shape(self):
        return self._array.shape

    def to(self, dtype=None):
        return _FakeTensor(self._array.astype(np.float32))

    def squeeze(self, axis=None):
        return _FakeTensor(np.squeeze(self._array, axis=axis))

    def reshape(self, *shape):
        return _FakeTensor(self._array.reshape(*shape))

    def numpy(self):
        return self._array

    def __getitem__(self, item):
        return _FakeTensor(self._array[item])


def _load_gr00t_with_torch_stub(monkeypatch):
    class _QwenGR00T:
        pass

    class _UamVLAOFT:
        pass

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = _FakeTensor
    torch_mod.float32 = np.float32
    torch_mod.is_tensor = lambda value: isinstance(value, _FakeTensor)
    torch_mod.as_tensor = lambda value, dtype=None: _FakeTensor(value)
    torch_mod.inference_mode = lambda: (lambda fn: fn)
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.Module = object
    torch_mod.nn = torch_nn
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn)

    qwen_gr00t = types.ModuleType("starVLA.model.framework.VLM4A.QwenGR00T")
    qwen_gr00t.Qwen_GR00T = _QwenGR00T
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.VLM4A.QwenGR00T", qwen_gr00t)

    uamvla_oft = types.ModuleType("starVLA.model.framework.VLM4A.UamVLAOFT")
    uamvla_oft.UamVLAOFT = _UamVLAOFT
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.VLM4A.UamVLAOFT", uamvla_oft)

    class _Registry:
        def register(self, name):
            return lambda cls: cls

    tools = types.ModuleType("starVLA.model.tools")
    tools.FRAMEWORK_REGISTRY = _Registry()
    monkeypatch.setitem(sys.modules, "starVLA.model.tools", tools)

    path = Path("starVLA/model/framework/VLM4A/UamVLAGR00T.py")
    spec = importlib.util.spec_from_file_location("uamvla_gr00t_state_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gr00t_extracts_first_7_dims_from_libero_33d_state(monkeypatch):
    module = _load_gr00t_with_torch_stub(monkeypatch)
    packed = np.arange(33, dtype=np.float32).reshape(1, 33)
    state = module.UamVLAGR00T._extract_gr00t_state_from_packed_calvin_state(packed)
    assert tuple(state.shape) == (1, 7)
    assert np.allclose(state.numpy(), np.arange(7, dtype=np.float32).reshape(1, 7))
