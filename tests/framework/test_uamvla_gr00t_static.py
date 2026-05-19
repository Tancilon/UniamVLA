from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml


class _AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _load_uamvla_gr00t_module(monkeypatch):
    registered = []

    class _Registry:
        def register(self, name):
            registered.append(name)
            return lambda cls: cls

    class _QwenGR00T:
        def __init__(self, config):
            super().__init__()
            self.config = config

    class _UamVLAOFT:
        def __init__(self, config):
            raise AssertionError("UamVLAGR00T must not call UamVLAOFT.__init__")

        _unpack_lerobot_sample = lambda self, sample: sample
        _force_resize_640 = lambda self, image_list: image_list
        _collate_aux = lambda self, examples, qwen_inputs: {"input_ids": qwen_inputs["input_ids"]}

    qwen_gr00t = types.ModuleType("starVLA.model.framework.VLM4A.QwenGR00T")
    qwen_gr00t.Qwen_GR00T = _QwenGR00T
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.VLM4A.QwenGR00T", qwen_gr00t)

    uamvla_oft = types.ModuleType("starVLA.model.framework.VLM4A.UamVLAOFT")
    uamvla_oft.UamVLAOFT = _UamVLAOFT
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.VLM4A.UamVLAOFT", uamvla_oft)

    tools = types.ModuleType("starVLA.model.tools")
    tools.FRAMEWORK_REGISTRY = _Registry()
    monkeypatch.setitem(sys.modules, "starVLA.model.tools", tools)

    path = Path("starVLA/model/framework/VLM4A/UamVLAGR00T.py")
    spec = importlib.util.spec_from_file_location("uamvla_gr00t_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._registered_names = registered
    return module


def test_uamvla_gr00t_calvin_d_config_uses_horizon_8():
    cfg = yaml.safe_load(Path("starVLA/config/training/uamvla_gr00t_calvin_d.yaml").read_text())

    assert cfg["framework"]["name"] == "UamVLAGR00T"
    assert cfg["framework"]["action_model"]["action_model_type"] == "DiT-B"
    assert cfg["framework"]["action_model"]["action_horizon"] == 8
    assert cfg["framework"]["action_model"]["state_dim"] == 7
    assert cfg["framework"]["action_model"]["repeated_diffusion_steps"] == 8
    assert "Action" not in cfg["framework"]["qwenvl"]["base_vlm"]
    assert cfg["datasets"]["vla_data"]["data_mix"] == "uamvla_calvin_d_h8"


def test_uamvla_gr00t_registers_framework_and_uses_flow_matching_forward(monkeypatch):
    module = _load_uamvla_gr00t_module(monkeypatch)
    monkeypatch.setattr(module.torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())

    assert "UamVLAGR00T" in module._registered_names

    class _FakeQwenInterface:
        image_token_id = 99

        def __init__(self):
            self.instructions = None

        def build_qwenvl_inputs(self, images, instructions):
            self.instructions = instructions
            return {"input_ids": torch.full((len(images), 400), 99, dtype=torch.long)}

        def __call__(self, **kwargs):
            hidden = torch.ones((kwargs["input_ids"].shape[0], 400, 16), dtype=torch.float32)
            return types.SimpleNamespace(hidden_states=[hidden])

    class _FakeActionModel:
        def __init__(self):
            self.calls = []

        def __call__(self, vl_embs, actions, state):
            self.calls.append((vl_embs, actions, state))
            return torch.tensor(2.5)

    model = object.__new__(module.UamVLAGR00T)
    model.qwen_vl_interface = _FakeQwenInterface()
    model.action_model = _FakeActionModel()
    model.action_horizon = 8
    model.aux_heads = {}
    model.config = _AttrDict(
        framework=_AttrDict(
            action_model=_AttrDict(repeated_diffusion_steps=2),
        ),
    )

    sample = {
        "image": [object()],
        "lang": "open the drawer",
        "action": np.arange(10 * 7, dtype=np.float32).reshape(10, 7),
    }

    out = module.UamVLAGR00T.forward(model, [sample])

    assert out["action_loss"].item() == 2.5
    assert model.qwen_vl_interface.instructions == ["open the drawer"]
    vl_embs, actions, state = model.action_model.calls[0]
    assert vl_embs.shape == (2, 400, 16)
    assert actions.shape == (2, 8, 7)
    assert state is None
    np.testing.assert_allclose(actions[0].numpy(), sample["action"][-8:])


def test_uamvla_gr00t_forward_repeats_extracted_calvin_robot_obs_state(monkeypatch):
    module = _load_uamvla_gr00t_module(monkeypatch)
    monkeypatch.setattr(module.torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())

    class _FakeQwenInterface:
        image_token_id = 99

        def build_qwenvl_inputs(self, images, instructions):
            return {"input_ids": torch.full((len(images), 400), 99, dtype=torch.long)}

        def __call__(self, **kwargs):
            hidden = torch.ones((kwargs["input_ids"].shape[0], 400, 16), dtype=torch.float32)
            return types.SimpleNamespace(hidden_states=[hidden])

    class _FakeActionModel:
        def __init__(self):
            self.calls = []

        def __call__(self, vl_embs, actions, state):
            self.calls.append((vl_embs, actions, state))
            return torch.tensor(1.0)

    model = object.__new__(module.UamVLAGR00T)
    model.qwen_vl_interface = _FakeQwenInterface()
    model.action_model = _FakeActionModel()
    model.action_horizon = 8
    model.aux_heads = {}
    model.config = _AttrDict(
        framework=_AttrDict(
            action_model=_AttrDict(repeated_diffusion_steps=2, state_dim=7),
        ),
    )

    packed_state = np.arange(33, dtype=np.float32).reshape(1, 33)
    sample = {
        "__trajectory_id": 1,
        "image": [object()],
        "lang": "open the drawer",
        "action": np.arange(10 * 7, dtype=np.float32).reshape(10, 7),
        "state": packed_state,
    }

    module.UamVLAGR00T.forward(model, [sample])

    _vl_embs, _actions, state = model.action_model.calls[0]
    assert state.shape == (2, 1, 7)
    np.testing.assert_allclose(state[0].numpy(), packed_state[:, :7])
    np.testing.assert_allclose(state[1].numpy(), packed_state[:, :7])


def test_uamvla_gr00t_init_does_not_walk_into_uamvla_oft_init(monkeypatch):
    module = _load_uamvla_gr00t_module(monkeypatch)
    init_calls = []

    def _fake_init_gr00t_components(self, config):
        init_calls.append(config)
        self.config = config
        self.action_horizon = 8

    module.UamVLAGR00T._init_gr00t_components = _fake_init_gr00t_components
    module.UamVLAGR00T._init_uamvla_sidecars = lambda self: None
    module.UamVLAGR00T._maybe_build_aux_heads = lambda self: None

    model = module.UamVLAGR00T("cfg")

    assert model.config == "cfg"
    assert model.action_horizon == 8
    assert init_calls == ["cfg"]


def test_uamvla_gr00t_unpacks_calvin_robot_obs_as_action_aligned_state(monkeypatch):
    module = _load_uamvla_gr00t_module(monkeypatch)
    model = object.__new__(module.UamVLAGR00T)

    packed_state = np.arange(33, dtype=np.float32).reshape(1, 33)
    sample = {
        "__trajectory_id": 1,
        "state": packed_state,
    }

    out = module.UamVLAGR00T._unpack_lerobot_sample(model, sample)

    assert out["state"].shape == (1, 7)
    np.testing.assert_allclose(out["state"].numpy(), packed_state[:, :7])
