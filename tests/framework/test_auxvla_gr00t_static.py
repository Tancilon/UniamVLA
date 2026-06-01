from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


class _AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _load_auxvla_module(monkeypatch):
    registered = []

    class _Registry:
        def register(self, name):
            registered.append(name)
            return lambda cls: cls

    class _BaseFramework(torch.nn.Module):
        def __init__(self):
            super().__init__()

    tools = types.ModuleType("starVLA.model.tools")
    tools.FRAMEWORK_REGISTRY = _Registry()
    monkeypatch.setitem(sys.modules, "starVLA.model.tools", tools)

    base_framework = types.ModuleType("starVLA.model.framework.base_framework")
    base_framework.baseframework = _BaseFramework
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.base_framework", base_framework)

    share_tools = types.ModuleType("starVLA.model.framework.share_tools")
    share_tools.merge_framework_config = lambda _default, cfg: cfg
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.share_tools", share_tools)

    action_header = types.ModuleType("starVLA.model.modules.action_model.GR00T_ActionHeader")
    action_header.get_action_model = lambda config: _FakeActionModel()
    monkeypatch.setitem(
        sys.modules,
        "starVLA.model.modules.action_model.GR00T_ActionHeader",
        action_header,
    )

    uamvla_oft = types.ModuleType("starVLA.model.framework.VLM4A.UamVLAOFT")
    uamvla_oft.UamVLAOFT = _FakeUamVLAOFT
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.VLM4A.UamVLAOFT", uamvla_oft)

    path = Path("starVLA/model/framework/VLM4A/AuxVLAGR00T.py")
    spec = importlib.util.spec_from_file_location("auxvla_gr00t_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module._registered_names = registered
    return module


class _FakeUamVLAOFT:
    def __init__(self, config):
        raise AssertionError("AuxVLAGR00T must not call UamVLAOFT.__init__")

    _unpack_lerobot_sample = lambda self, sample: sample
    _init_uamvla_sidecar_roots = lambda self: None
    _init_lerobot_video_path_config = lambda self: None
    _sidecar_root_for_sample = lambda self, sample: None
    _image_future_video_path = lambda self, trajectory_id, sidecar_root=None: Path("missing.mp4")
    _image_future_frame_index = staticmethod(
        lambda video_length: video_length - 1 if video_length > 0 else None
    )
    _load_npy_sidecar = staticmethod(lambda path: None)
    _load_depth_target = lambda self, trajectory_id, base_index, sidecar_root=None: None
    _load_grounding_mask = lambda self, trajectory_id, base_index, sidecar_root=None: None
    _load_affordance_heatmap = lambda self, trajectory_id, base_index, sidecar_root=None: None
    _image_action_future_frame_index = (
        lambda self, base_index, video_length: base_index if base_index < video_length else None
    )
    _load_image_future = lambda self, trajectory_id, base_index, sidecar_root=None: None
    _load_image_action_future = lambda self, trajectory_id, base_index, sidecar_root=None: None
    _maybe_build_aux_heads = lambda self: None
    _maybe_build_aux_loss_control = lambda self: None
    _aux_head_cfg_without_layout_keys = staticmethod(lambda cfg: dict(cfg))
    _force_resize_640 = lambda self, image_list: image_list
    _resolve_head_mask = lambda self, name, batch, batch_size, device: batch[f"{name}_mask"]
    _global_step_from_kwargs = staticmethod(lambda kwargs: int(kwargs.get("global_step", 0) or 0))

    def _collate_aux(self, examples, qwen_inputs):
        return {
            "input_ids": qwen_inputs["input_ids"],
            "recon_mask": torch.tensor([True] * len(examples)),
        }


class _FakeActionModel:
    def __init__(self):
        self.calls = []

    def __call__(self, vl_embs, actions, state):
        self.calls.append((vl_embs, actions, state))
        return torch.tensor(2.0)

    def predict_action(self, hidden, state):
        self.calls.append((hidden, None, state))
        return torch.zeros(hidden.shape[0], 8, 7)


def test_auxvla_gr00t_registers_framework(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    assert "AuxVLAGR00T" in module._registered_names


def test_auxvla_delegates_uamvla_sidecar_loaders(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    expected_methods = [
        "_image_future_video_path",
        "_image_future_frame_index",
        "_load_npy_sidecar",
        "_load_depth_target",
        "_load_grounding_mask",
        "_load_affordance_heatmap",
        "_image_action_future_frame_index",
        "_load_image_future",
        "_load_image_action_future",
    ]
    for name in expected_methods:
        assert hasattr(module.AuxVLAGR00T, name), name

    model = object.__new__(module.AuxVLAGR00T)
    assert model._image_future_frame_index(3) == 2
    assert model._load_npy_sidecar(Path("missing.npy")) is None
    assert model._aux_head_cfg_without_layout_keys({"enabled": True}) == {"enabled": True}


def test_make_aux_input_ids_marks_boi_eoi_span(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    out = module._make_aux_input_ids(
        batch_size=2,
        seq_len=8,
        boi_ids=[1, 3],
        eoi_ids=[3, 6],
        image_token_id=-200,
        device=torch.device("cpu"),
    )
    assert torch.equal(out[0], torch.tensor([0, -200, -200, -200, 0, 0, 0, 0]))
    assert torch.equal(out[1], torch.tensor([0, 0, 0, -200, -200, -200, -200, 0]))


def test_make_aux_input_ids_rejects_missing_span(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    with pytest.raises(RuntimeError, match="missing visual span"):
        module._make_aux_input_ids(
            batch_size=1,
            seq_len=4,
            boi_ids=[None],
            eoi_ids=[None],
            image_token_id=-200,
            device=torch.device("cpu"),
        )


def test_select_single_view_primary_and_wrist(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    first, second = object(), object()
    assert module.AuxVLAGR00T._select_single_view(model, [first, second], "primary") is first
    assert module.AuxVLAGR00T._select_single_view(model, [first, second], "wrist") is second


def test_select_single_view_rejects_missing_wrist(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    with pytest.raises(RuntimeError, match="wrist"):
        module.AuxVLAGR00T._select_single_view(model, [object()], "wrist")


def test_forward_routes_reconvla_hidden_to_gr00t_action_and_aux_suite(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    monkeypatch.setattr(module.torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())

    class _FakeReconInterface:
        image_token_id = -200
        image_embed_len = 4

        def build_qwenvl_inputs(self, images, instructions):
            assert len(images) == 1
            assert len(images[0]) == 1
            assert instructions == ["open drawer"]
            return {"input_ids": torch.zeros(1, 3, dtype=torch.long)}

        def __call__(self, **kwargs):
            hidden = torch.ones(1, 6, 16)
            return types.SimpleNamespace(hidden_states=[hidden], boi_ids=[1], eoi_ids=[4])

    class _FakeSuite:
        def __init__(self):
            self.calls = []

        def __call__(self, action_loss, hidden_states, batch, masks, global_step=0):
            self.calls.append((action_loss, hidden_states, batch, masks, global_step))
            assert torch.equal(batch["input_ids"][0], torch.tensor([0, -200, -200, -200, -200, 0]))
            return torch.tensor(0.5), {"aux_total_post_budget": torch.tensor(0.5)}

    model = object.__new__(module.AuxVLAGR00T)
    model.qwen_vl_interface = _FakeReconInterface()
    model.action_model = _FakeActionModel()
    model.action_horizon = 8
    model.aux_heads = {"recon": object()}
    model.aux_suite = _FakeSuite()
    model.config = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(single_view_mode="primary", synthetic_image_token_id=-200),
            action_model=_AttrDict(repeated_diffusion_steps=2, state_dim=7),
        ),
    )
    model._prepare_examples = types.MethodType(lambda self, examples: examples, model)

    sample = {
        "image": [object(), object()],
        "lang": "open drawer",
        "action": np.arange(10 * 7, dtype=np.float32).reshape(10, 7),
        "state": np.arange(7, dtype=np.float32).reshape(1, 7),
    }

    out = module.AuxVLAGR00T.forward(model, [sample], global_step=9)

    assert out["action_loss"].item() == 2.5
    vl_embs, actions, state = model.action_model.calls[0]
    assert vl_embs.shape == (2, 6, 16)
    assert actions.shape == (2, 8, 7)
    assert state.shape == (2, 1, 7)
    assert model.aux_suite.calls[0][4] == 9


def test_predict_action_uses_single_view_without_aux(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    monkeypatch.setattr(module.torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())

    class _FakeReconInterface:
        image_token_id = -200
        image_embed_len = 4

        def build_qwenvl_inputs(self, images, instructions):
            assert len(images[0]) == 1
            return {"input_ids": torch.zeros(1, 3, dtype=torch.long)}

        def __call__(self, **kwargs):
            hidden = torch.ones(1, 6, 16)
            return types.SimpleNamespace(hidden_states=[hidden], boi_ids=[1], eoi_ids=[4])

    model = object.__new__(module.AuxVLAGR00T)
    model.qwen_vl_interface = _FakeReconInterface()
    model.action_model = _FakeActionModel()
    model.action_horizon = 8
    model.config = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(single_view_mode="primary", synthetic_image_token_id=-200),
            action_model=_AttrDict(state_dim=0),
        ),
    )
    model._prepare_examples = types.MethodType(lambda self, examples: examples, model)

    out = module.AuxVLAGR00T.predict_action(
        model,
        {"image": [object(), object()], "lang": "open drawer"},
    )
    assert out["normalized_actions"].shape == (1, 8, 7)
