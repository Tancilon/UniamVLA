from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
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


def test_uamvla_gr00t_calvin_config_uses_horizon_8():
    cfg = yaml.safe_load(Path("starVLA/config/training/uamvla_gr00t_calvin.yaml").read_text())

    assert cfg["framework"]["name"] == "UamVLAGR00T"
    assert cfg["framework"]["action_model"]["action_model_type"] == "DiT-B"
    assert cfg["framework"]["action_model"]["action_horizon"] == 8
    assert cfg["framework"]["action_model"]["state_dim"] == 7
    assert cfg["framework"]["action_model"]["repeated_diffusion_steps"] == 8
    assert "Action" not in cfg["framework"]["qwenvl"]["base_vlm"]
    assert cfg["datasets"]["vla_data"]["data_mix"] == "uamvla_calvin_d_h8"


def test_calvin_registry_contains_abc_h8_mix(monkeypatch):
    datasets = types.ModuleType("starVLA.dataloader.gr00t_lerobot.datasets")

    class _ModalityConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    datasets.ModalityConfig = _ModalityConfig
    monkeypatch.setitem(sys.modules, "starVLA.dataloader.gr00t_lerobot.datasets", datasets)

    base_transform = types.ModuleType("starVLA.dataloader.gr00t_lerobot.transform.base")
    base_transform.ComposedModalityTransform = lambda transforms: transforms
    monkeypatch.setitem(sys.modules, "starVLA.dataloader.gr00t_lerobot.transform.base", base_transform)

    state_action = types.ModuleType("starVLA.dataloader.gr00t_lerobot.transform.state_action")
    state_action.StateActionToTensor = lambda **kwargs: ("to_tensor", kwargs)
    state_action.StateActionTransform = lambda **kwargs: ("transform", kwargs)
    monkeypatch.setitem(sys.modules, "starVLA.dataloader.gr00t_lerobot.transform.state_action", state_action)

    embodiment_tags = types.ModuleType("starVLA.dataloader.gr00t_lerobot.embodiment_tags")
    embodiment_tags.EmbodimentTag = types.SimpleNamespace(FRANKA="franka")
    monkeypatch.setitem(sys.modules, "starVLA.dataloader.gr00t_lerobot.embodiment_tags", embodiment_tags)

    path = Path("examples/calvin/train_files/data_registry/data_config.py")
    spec = importlib.util.spec_from_file_location("calvin_data_config_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.DATASET_NAMED_MIXTURES["uamvla_calvin_abc_h8"] == [
        ("lerobot_calvin_abc", 1.0, "uamvla_calvin_franka_h8"),
    ]


def test_uamvla_gr00t_config_defines_aux_loss_control_and_new_heads():
    cfg = yaml.safe_load(Path("starVLA/config/training/uamvla_gr00t_calvin.yaml").read_text())
    assert cfg["framework"]["aux_loss_control"]["enabled"] is True
    assert cfg["framework"]["aux_loss_control"]["aux_budget"] == 1.0
    assert cfg["framework"]["obs_image_size"] == [320, 320]
    assert cfg["datasets"]["vla_data"]["image_resize"] == 320
    assert "warmup_steps" not in cfg["framework"]["aux_loss_control"]
    assert "aux_ratio_cap" not in cfg["framework"]["aux_loss_control"]
    assert "action_loss_ema_beta" not in cfg["framework"]["aux_loss_control"]
    heads = cfg["framework"]["aux_heads"]
    for name in ["depth", "action_conditioned_future", "grounding", "affordance"]:
        assert name in heads
        assert heads[name]["enabled"] is False
    assert heads["future"]["target_resize"] == 160
    assert heads["recon"]["target_resize"] == 160
    assert heads["action_conditioned_future"]["target_resize"] == 160
    assert heads["depth"]["target_size"] == 10
    assert heads["grounding"]["target_size"] == 10
    assert heads["affordance"]["target_size"] == 10


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


def test_uamvla_gr00t_image_token_assert_uses_configured_layout(monkeypatch):
    module = _load_uamvla_gr00t_module(monkeypatch)
    model = object.__new__(module.UamVLAGR00T)
    model.qwen_vl_interface = types.SimpleNamespace(image_token_id=99)
    model._qwen_patches_per_view = lambda: 100
    examples = [{"image": [object(), object()]}]

    module.UamVLAGR00T._assert_image_token_count(
        model,
        torch.full((1, 200), 99, dtype=torch.long),
        examples,
    )

    with pytest.raises(RuntimeError, match="expected 200"):
        module.UamVLAGR00T._assert_image_token_count(
            model,
            torch.full((1, 800), 99, dtype=torch.long),
            examples,
        )


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


def test_uamvla_gr00t_forward_uses_aux_suite_when_present(monkeypatch):
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
        def __call__(self, vl_embs, actions, state):
            return torch.tensor(2.0)

    class _FakeSuite:
        def __init__(self):
            self.calls = []

        def __call__(self, action_loss, hidden_states, batch, masks, global_step=0):
            self.calls.append((action_loss, hidden_states, batch, masks, global_step))
            return torch.tensor(0.25), {"aux_total_post_budget": torch.tensor(0.25)}

    model = object.__new__(module.UamVLAGR00T)
    model.qwen_vl_interface = _FakeQwenInterface()
    model.action_model = _FakeActionModel()
    model.action_horizon = 8
    model.aux_heads = {}
    model.aux_suite = _FakeSuite()
    model.config = _AttrDict(framework=_AttrDict(action_model=_AttrDict(repeated_diffusion_steps=1)))

    sample = {
        "image": [object()],
        "lang": "open",
        "action": np.zeros((8, 7), dtype=np.float32),
    }
    out = module.UamVLAGR00T.forward(model, [sample], global_step=3)
    assert out["action_loss"].item() == 2.25
    assert out["aux_total_post_budget"].item() == 0.25
    assert model.aux_suite.calls[0][4] == 3


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


def test_uamvla_gr00t_predict_action_ignores_aux_sidecars(monkeypatch):
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
        def predict_action(self, hidden, state):
            assert hidden.shape == (1, 400, 16)
            assert state is None
            return torch.zeros(1, 8, 7)

    def _forbid_aux_unpack(self, sample):
        forbidden = {
            "depth_target",
            "grounding_mask",
            "affordance_heatmap",
            "image_action_future",
        }
        assert not forbidden.intersection(sample)
        return sample

    model = object.__new__(module.UamVLAGR00T)
    model.qwen_vl_interface = _FakeQwenInterface()
    model.action_model = _FakeActionModel()
    model.action_horizon = 8
    model.config = _AttrDict(framework=_AttrDict(action_model=_AttrDict(state_dim=0)))
    model._prepare_examples = types.MethodType(lambda self, examples: examples, model)
    model._unpack_lerobot_sample = types.MethodType(_forbid_aux_unpack, model)

    out = module.UamVLAGR00T.predict_action(
        model,
        {"image": [object()], "lang": "open drawer"},
    )

    assert out["normalized_actions"].shape == (1, 8, 7)


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
