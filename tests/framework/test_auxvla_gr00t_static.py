from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image


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
    _to_action_numpy = staticmethod(lambda action: np.asarray(action, dtype=np.float32))
    _to_rgb_pil = staticmethod(
        lambda image: image.convert("RGB")
        if isinstance(image, Image.Image)
        else Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    )
    _fit_for_viz = staticmethod(lambda image, height=160: image)
    _draw_action_comparison = staticmethod(
        lambda pred_action, gt_action=None: Image.new("RGB", (16, 16), "white")
    )

    @classmethod
    def _make_visualization_canvas(cls, images, pred_action, gt_action=None):
        return Image.new("RGB", (16, 16), "white")

    @classmethod
    def _pil_to_normalized_chw(cls, image, device=None):
        tensor = torch.zeros(3, 4, 4)
        return tensor.to(device) if device is not None else tensor

    @classmethod
    def _make_visualization_image_batch(cls, examples, device=None):
        return torch.zeros(len(examples), 1, 3, 4, 4, device=device)

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


class _FakeReconInterfaceForInit:
    instances = []

    def __init__(self, config):
        self.config = config
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=3584))
        self.applied_lora = None
        self.__class__.instances.append(self)

    def apply_language_lora(self, lora_cfg):
        self.applied_lora = lora_cfg


class _NamedParamModule(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.register_parameter(name, torch.nn.Parameter(torch.ones(())))


class _FakeLoRAInterface(torch.nn.Module):
    def __init__(self, lora_enabled=True):
        super().__init__()
        self.lora_enabled = lora_enabled
        self.base_weight = torch.nn.Parameter(torch.ones(()))
        self.lora_A = torch.nn.Parameter(torch.ones(()))


class _FakeARReconInterface(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generate_normalized_actions(
        self,
        images,
        instructions,
        robot_obs,
        input_mode,
        action_horizon,
        action_dim,
    ):
        self.calls.append(
            {
                "images": images,
                "instructions": instructions,
                "robot_obs": robot_obs,
                "input_mode": input_mode,
                "action_horizon": action_horizon,
                "action_dim": action_dim,
            }
        )
        batch_size = len(instructions)
        return np.full(
            (batch_size, int(action_horizon), int(action_dim)),
            0.25,
            dtype=np.float32,
        )


def _make_predict_model(module, inference_mode="gr00t", ar_input_mode="official_compose"):
    model = object.__new__(module.AuxVLAGR00T)
    torch.nn.Module.__init__(model)
    model.config = _AttrDict(
        framework=_AttrDict(
            name="AuxVLAGR00T",
            reconvla=_AttrDict(
                inference_mode=inference_mode,
                ar_input_mode=ar_input_mode,
                single_view_mode="concat_vertical",
            ),
            action_model=_AttrDict(
                action_horizon=5,
                action_dim=7,
                state_dim=7,
            ),
        )
    )
    model.action_horizon = 5
    model.action_model = _FakeActionModel()
    model.qwen_vl_interface = _FakeARReconInterface()
    model._prepare_examples = lambda examples, require_reconvla_target=False: examples
    model._encode_reconvla_hidden = lambda examples: (
        {"input_ids": torch.ones(len(examples), 4, dtype=torch.long)},
        torch.ones(len(examples), 4, 3584, dtype=torch.bfloat16),
    )
    model._action_model_compute_dtype = lambda hidden_dtype: torch.float32
    model._state_batch_or_none = (
        lambda examples, device, dtype: torch.zeros(len(examples), 1, 7, device=device, dtype=dtype)
    )
    return model


def test_auxvla_gr00t_registers_framework(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    assert "AuxVLAGR00T" in module._registered_names


def test_auxvla_lora_defaults_are_disabled(monkeypatch):
    module = _load_auxvla_module(monkeypatch)

    default_cfg = module.AuxVLAGR00TDefaultConfig()
    lora_cfg = default_cfg.reconvla["lora"]

    assert lora_cfg["enabled"] is False
    assert lora_cfg["r"] == 16
    assert lora_cfg["lora_alpha"] == 32
    assert lora_cfg["lora_dropout"] == 0.05
    assert lora_cfg["init_lora_weights"] is True
    assert lora_cfg["bias"] == "none"
    assert lora_cfg["task_type"] == "CAUSAL_LM"
    assert lora_cfg["train_mm_projector"] is True
    assert lora_cfg["train_mm_inv_projector"] is False
    assert lora_cfg["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_cfg_to_plain_dict_handles_wrapped_lora_target_modules(monkeypatch):
    from starVLA.training.trainer_utils.config_tracker import wrap_config

    module = _load_auxvla_module(monkeypatch)
    cfg = wrap_config(
        OmegaConf.create(
            {
                "enabled": True,
                "r": 16,
                "target_modules": ["q_proj", "v_proj"],
            }
        )
    )

    plain = module._cfg_to_plain_dict(cfg)

    assert plain == {
        "enabled": True,
        "r": 16,
        "target_modules": ["q_proj", "v_proj"],
    }


def test_auxvla_init_applies_reconvla_lora_when_enabled(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    _FakeReconInterfaceForInit.instances = []
    monkeypatch.setattr(module, "ReconVLAInterface", _FakeReconInterfaceForInit)

    cfg = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(
                lora=_AttrDict(
                    enabled=True,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj"],
                )
            ),
            action_model=_AttrDict(
                diffusion_model_cfg=_AttrDict(cross_attention_dim=0),
                action_horizon=8,
            ),
        ),
        datasets=_AttrDict(vla_data=_AttrDict()),
    )

    model = module.AuxVLAGR00T(cfg)

    assert model.qwen_vl_interface is _FakeReconInterfaceForInit.instances[0]
    assert model.qwen_vl_interface.applied_lora is cfg.framework.reconvla.lora
    assert cfg.framework.action_model.diffusion_model_cfg.cross_attention_dim == 3584


def test_reconvla_lora_init_lora_weights_is_forwarded_to_peft(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    captured = {}

    class _FakeTaskType:
        CAUSAL_LM = "CAUSAL_LM"

    class _FakeLoraConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def _fake_get_peft_model(model, _peft_cfg):
        model.lora_A = torch.nn.Parameter(torch.ones(()))
        return model

    fake_peft = types.ModuleType("peft")
    fake_peft.TaskType = _FakeTaskType
    fake_peft.LoraConfig = _FakeLoraConfig
    fake_peft.get_peft_model = _fake_get_peft_model
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    class _FakeReconModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(2, 2)

    interface = object.__new__(module.ReconVLAInterface)
    torch.nn.Module.__init__(interface)
    interface.model = _FakeReconModel()
    interface.lora_enabled = False

    module.ReconVLAInterface.apply_language_lora(
        interface,
        {
            "enabled": True,
            "r": 32,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "init_lora_weights": "gaussian",
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "train_mm_projector": False,
            "train_mm_inv_projector": False,
            "target_modules": ["q_proj"],
        },
    )

    assert captured["init_lora_weights"] == "gaussian"


def test_auxvla_get_lr_groups_routes_lora_and_action_params(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    torch.nn.Module.__init__(model)
    model.qwen_vl_interface = _FakeLoRAInterface(lora_enabled=True)
    model.action_model = _NamedParamModule("action_weight")
    model.aux_heads = torch.nn.ModuleDict({"recon": _NamedParamModule("aux_weight")})
    model.extra_weight = torch.nn.Parameter(torch.ones(()))
    model.config = _AttrDict(
        framework=_AttrDict(reconvla=_AttrDict(lora=_AttrDict(enabled=True))),
        trainer=_AttrDict(freeze_modules=None),
    )

    groups = module.AuxVLAGR00T.get_lr_groups(
        model,
        _AttrDict(
            base=1.0e-4,
            qwen_vl_interface=1.0e-5,
            action_model=2.0e-4,
            aux_heads=3.0e-4,
        ),
    )

    by_name = {group["name"]: group for group in groups}
    assert by_name["qwen_vl_interface"]["lr"] == 1.0e-5
    assert by_name["action_model"]["lr"] == 2.0e-4
    assert by_name["aux_heads"]["lr"] == 3.0e-4
    assert by_name["base"]["lr"] == 1.0e-4

    qwen_param_ids = {id(param) for param in by_name["qwen_vl_interface"]["params"]}
    all_group_param_ids = {id(param) for group in groups for param in group["params"]}
    assert qwen_param_ids == {id(model.qwen_vl_interface.lora_A)}
    assert id(model.qwen_vl_interface.base_weight) not in all_group_param_ids
    assert id(model.action_model.action_weight) in {
        id(param) for param in by_name["action_model"]["params"]
    }
    assert id(model.aux_heads["recon"].aux_weight) in {
        id(param) for param in by_name["aux_heads"]["params"]
    }
    assert id(model.extra_weight) in {id(param) for param in by_name["base"]["params"]}


def test_auxvla_get_lr_groups_rejects_freezing_qwen_interface_with_lora(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    torch.nn.Module.__init__(model)
    model.qwen_vl_interface = _FakeLoRAInterface(lora_enabled=True)
    model.action_model = _NamedParamModule("action_weight")
    model.config = _AttrDict(
        framework=_AttrDict(reconvla=_AttrDict(lora=_AttrDict(enabled=True))),
        trainer=_AttrDict(freeze_modules="qwen_vl_interface"),
    )

    with pytest.raises(RuntimeError, match="freeze_modules"):
        module.AuxVLAGR00T.get_lr_groups(
            model,
            _AttrDict(base=1.0e-4, qwen_vl_interface=1.0e-5, action_model=2.0e-4),
        )


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


def test_select_single_view_concat_vertical(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    model._qwen_image_size = lambda: 4
    primary = Image.new("RGB", (2, 2), (255, 0, 0))
    wrist = Image.new("RGB", (2, 2), (0, 0, 255))

    out = module.AuxVLAGR00T._select_single_view(
        model,
        [primary, wrist],
        "concat_vertical",
    )

    assert out.size == (4, 4)
    assert out.getpixel((1, 0)) == (255, 0, 0)
    assert out.getpixel((1, 1)) == (255, 0, 0)
    assert out.getpixel((1, 2)) == (0, 0, 255)
    assert out.getpixel((1, 3)) == (0, 0, 255)


def test_select_single_view_rejects_missing_wrist(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    with pytest.raises(RuntimeError, match="wrist"):
        module.AuxVLAGR00T._select_single_view(model, [object()], "wrist")


def test_reconvla_style_image_target_composes_crop_and_wrist(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)

    crop = torch.zeros(3, 2, 2)
    crop[0].fill_(1.0)
    primary = Image.new("RGB", (2, 2), (0, 255, 0))
    wrist = Image.new("RGB", (2, 2), (0, 0, 255))
    example = {
        "image": [primary, wrist],
        "image_target": crop,
    }

    out = module.AuxVLAGR00T._compose_reconvla_style_image_target(model, example)

    assert out is not example
    assert out["image_target"].shape == (3, 384, 384)
    assert out["image_target"].dtype == torch.float32
    assert out["image_target"].min().item() >= 0.0
    assert out["image_target"].max().item() <= 1.0

    crop_height = 384 * 14 // 27
    top = out["image_target"][:, :crop_height]
    bottom = out["image_target"][:, crop_height:]
    assert top[0].mean().item() > 0.99
    assert top[1].mean().item() < 0.01
    assert top[2].mean().item() < 0.01
    assert bottom[0].mean().item() < 0.01
    assert bottom[1].mean().item() < 0.01
    assert bottom[2].mean().item() > 0.99
    assert torch.equal(example["image_target"], crop)


def test_reconvla_style_image_target_requires_crop(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    example = {
        "image": [
            Image.new("RGB", (2, 2), "red"),
            Image.new("RGB", (2, 2), "blue"),
        ],
    }

    with pytest.raises(RuntimeError, match="image_target"):
        module.AuxVLAGR00T._compose_reconvla_style_image_target(model, example)


def test_reconvla_style_image_target_requires_wrist(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    example = {
        "image": [Image.new("RGB", (2, 2), "red")],
        "image_target": torch.zeros(3, 2, 2),
    }

    with pytest.raises(RuntimeError, match="wrist"):
        module.AuxVLAGR00T._compose_reconvla_style_image_target(model, example)


def test_prepare_examples_can_require_reconvla_target(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    crop = torch.ones(3, 2, 2)
    sample = {
        "image": [
            Image.new("RGB", (2, 2), "black"),
            Image.new("RGB", (2, 2), "blue"),
        ],
        "image_target": crop,
        "lang": "open drawer",
        "action": torch.zeros(8, 7),
    }

    out = module.AuxVLAGR00T._prepare_examples(
        model,
        [sample],
        require_reconvla_target=True,
    )

    assert out[0]["image_target"].shape == (3, 384, 384)
    assert sample["image_target"].shape == (3, 2, 2)


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
    model._prepare_examples = types.MethodType(
        lambda self, examples, require_reconvla_target=False: examples,
        model,
    )

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


def test_forward_casts_action_inputs_to_action_model_dtype(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    monkeypatch.setattr(module.torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())

    class _FakeReconInterface:
        image_token_id = -200
        image_embed_len = 4

        def build_qwenvl_inputs(self, images, instructions):
            return {"input_ids": torch.zeros(1, 3, dtype=torch.long)}

        def __call__(self, **kwargs):
            hidden = torch.ones(1, 6, 16, dtype=torch.bfloat16)
            return types.SimpleNamespace(hidden_states=[hidden], boi_ids=[1], eoi_ids=[4])

    class _FloatActionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.ones(()))
            self.calls = []

        def forward(self, vl_embs, actions, state):
            self.calls.append((vl_embs, actions, state))
            assert vl_embs.dtype == self.marker.dtype
            assert actions.dtype == self.marker.dtype
            assert state.dtype == self.marker.dtype
            return self.marker * 0 + torch.tensor(1.0)

    model = object.__new__(module.AuxVLAGR00T)
    torch.nn.Module.__init__(model)
    model.qwen_vl_interface = _FakeReconInterface()
    model.action_model = _FloatActionModel()
    model.action_horizon = 8
    model.aux_heads = {}
    model.config = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(single_view_mode="primary", synthetic_image_token_id=-200),
            action_model=_AttrDict(repeated_diffusion_steps=2, state_dim=7),
        ),
    )
    model._prepare_examples = types.MethodType(
        lambda self, examples, require_reconvla_target=False: examples,
        model,
    )

    sample = {
        "image": [object()],
        "lang": "open drawer",
        "action": np.arange(10 * 7, dtype=np.float32).reshape(10, 7),
        "state": np.arange(7, dtype=np.float32).reshape(1, 7),
    }

    out = module.AuxVLAGR00T.forward(model, [sample])

    assert out["action_loss"].item() == 1.0


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
    model._prepare_examples = types.MethodType(
        lambda self, examples, require_reconvla_target=False: examples,
        model,
    )

    out = module.AuxVLAGR00T.predict_action(
        model,
        {"image": [object(), object()], "lang": "open drawer"},
    )
    assert out["normalized_actions"].shape == (1, 8, 7)


def test_predict_action_casts_hidden_to_action_model_dtype(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    monkeypatch.setattr(module.torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())

    class _FakeReconInterface:
        image_token_id = -200
        image_embed_len = 4

        def build_qwenvl_inputs(self, images, instructions):
            return {"input_ids": torch.zeros(1, 3, dtype=torch.long)}

        def __call__(self, **kwargs):
            hidden = torch.ones(1, 6, 16, dtype=torch.bfloat16)
            return types.SimpleNamespace(hidden_states=[hidden], boi_ids=[1], eoi_ids=[4])

    class _FloatPredictActionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.ones(()))
            self.calls = []

        def predict_action(self, hidden, state):
            self.calls.append((hidden, state))
            assert hidden.dtype == self.marker.dtype
            assert state.dtype == self.marker.dtype
            return torch.zeros(hidden.shape[0], 8, 7, dtype=self.marker.dtype)

    model = object.__new__(module.AuxVLAGR00T)
    torch.nn.Module.__init__(model)
    model.qwen_vl_interface = _FakeReconInterface()
    model.action_model = _FloatPredictActionModel()
    model.config = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(single_view_mode="primary", synthetic_image_token_id=-200),
            action_model=_AttrDict(state_dim=7),
        ),
    )
    model._prepare_examples = types.MethodType(
        lambda self, examples, require_reconvla_target=False: examples,
        model,
    )

    out = module.AuxVLAGR00T.predict_action(
        model,
        {
            "image": [object()],
            "lang": "open drawer",
            "state": np.arange(7, dtype=np.float32).reshape(1, 7),
        },
    )

    assert out["normalized_actions"].shape == (1, 8, 7)


def test_predict_action_defaults_to_gr00t_path(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    monkeypatch.setattr(module.torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())
    model = _make_predict_model(module, inference_mode="gr00t")
    example = {
        "image": [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)],
        "lang": "open the drawer",
        "state": np.zeros((1, 7), dtype=np.float32),
        "robot_obs": np.zeros(15, dtype=np.float32),
    }

    out = module.AuxVLAGR00T.predict_action(model, [example])

    assert out["normalized_actions"].shape == (1, 8, 7)
    assert model.qwen_vl_interface.calls == []
    assert len(model.action_model.calls) == 1


def test_predict_action_reconvla_ar_returns_normalized_chunk(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = _make_predict_model(
        module,
        inference_mode="reconvla_ar_normalized",
        ar_input_mode="official_compose",
    )
    example = {
        "image": [np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)],
        "lang": "move the slider left",
        "robot_obs": np.arange(15, dtype=np.float32),
    }

    out = module.AuxVLAGR00T.predict_action(model, [example])

    assert out["normalized_actions"].shape == (1, 5, 7)
    assert out["normalized_actions"].dtype == np.float32
    assert np.allclose(out["normalized_actions"], 0.25)
    assert model.qwen_vl_interface.calls[0]["input_mode"] == "official_compose"
    assert model.qwen_vl_interface.calls[0]["action_horizon"] == 5
    assert model.qwen_vl_interface.calls[0]["action_dim"] == 7
    assert model.qwen_vl_interface.calls[0]["robot_obs"].shape == (1, 15)
    assert len(model.action_model.calls) == 0


def test_predict_action_reconvla_ar_requires_raw_robot_obs(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = _make_predict_model(module, inference_mode="reconvla_ar_normalized")
    example = {
        "image": [np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)],
        "lang": "turn on the lightbulb",
    }

    with pytest.raises(RuntimeError, match="15-D robot_obs"):
        module.AuxVLAGR00T.predict_action(model, [example])


def test_predict_action_reconvla_ar_accepts_uamvla_raw_state(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = _make_predict_model(module, inference_mode="reconvla_ar_normalized")
    example = {
        "image": [np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)],
        "lang": "push the block right",
        "uamvla_raw_state": {"robot_obs": np.arange(15, dtype=np.float32)},
    }

    out = module.AuxVLAGR00T.predict_action(model, [example])

    assert out["normalized_actions"].shape == (1, 5, 7)
    assert np.allclose(model.qwen_vl_interface.calls[0]["robot_obs"][0], np.arange(15, dtype=np.float32))


def test_visualize_batch_accepts_distributed_flag_and_uses_reconvla_context(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    monkeypatch.setattr(module.torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())

    class _FloatPredictActionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.ones(()))
            self.calls = []

        def predict_action(self, hidden, state):
            self.calls.append((hidden, state))
            assert hidden.dtype == self.marker.dtype
            return torch.zeros(hidden.shape[0], 8, 7, dtype=self.marker.dtype)

    class _FakeVizHead:
        def __init__(self):
            self.calls = []

        def visualize(self, hidden_states, batch_dict, mask, num_samples, **kwargs):
            self.calls.append((hidden_states, batch_dict, mask, num_samples, kwargs))
            assert torch.equal(mask, torch.tensor([True]))
            return [Image.new("RGB", (8, 8), "blue")]

    model = object.__new__(module.AuxVLAGR00T)
    torch.nn.Module.__init__(model)
    model.action_model = _FloatPredictActionModel()
    model.action_horizon = 8
    recon_head = _FakeVizHead()
    model.aux_heads = {"recon": recon_head}
    model.config = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(single_view_mode="primary", synthetic_image_token_id=-200),
            action_model=_AttrDict(state_dim=0),
        ),
    )
    hidden = torch.ones(1, 6, 16, dtype=torch.bfloat16)
    recon_inputs = {"input_ids": torch.zeros(1, 6, dtype=torch.long)}
    model._prepare_examples = types.MethodType(
        lambda self, examples, require_reconvla_target=False: examples,
        model,
    )
    model._encode_reconvla_hidden = types.MethodType(
        lambda self, examples: (recon_inputs, hidden),
        model,
    )

    sample = {
        "image": [Image.new("RGB", (4, 4), "red")],
        "lang": "open drawer",
        "action": np.zeros((8, 7), dtype=np.float32),
    }

    out = module.AuxVLAGR00T.visualize_batch(
        model,
        [sample],
        n_samples=1,
        distributed_all_ranks=True,
    )

    assert "viz/auxvla_gr00t/action_0" in out
    assert "viz/auxvla_gr00t/recon_0" in out
    assert model.action_model.calls
    assert recon_head.calls
