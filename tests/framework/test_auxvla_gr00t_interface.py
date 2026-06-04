from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


class _AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _load_module(monkeypatch):
    class _Registry:
        def register(self, _name):
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

    class _UamVLAOFT:
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
        _unpack_lerobot_sample = lambda self, sample: sample
        _collate_aux = lambda self, examples, qwen_inputs: {}
        _resolve_head_mask = lambda self, name, batch, batch_size, device: None
        _force_resize_640 = lambda self, image_list: image_list
        _global_step_from_kwargs = staticmethod(lambda kwargs: 0)
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

    uamvla_oft = types.ModuleType("starVLA.model.framework.VLM4A.UamVLAOFT")
    uamvla_oft.UamVLAOFT = _UamVLAOFT
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.VLM4A.UamVLAOFT", uamvla_oft)

    path = Path("starVLA/model/framework/VLM4A/AuxVLAGR00T.py")
    spec = importlib.util.spec_from_file_location("auxvla_gr00t_interface_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    vocab_size = 1000
    model_max_length = 2048

    def __init__(self):
        self.calls = []

    @classmethod
    def from_pretrained(cls, path, use_fast=False):
        tokenizer = cls()
        tokenizer.path = path
        tokenizer.use_fast = use_fast
        return tokenizer

    def decode(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


class _FakeAutoConfig:
    @classmethod
    def from_pretrained(cls, path):
        return types.SimpleNamespace(
            hidden_size=3584,
            image_embed_len=729,
            recon_enable=True,
            reconstruct_image=True,
            mm_vision_tower="./siglip-so400m-patch14-384",
        )


class _FakeImageProcessor:
    image_mean = [0.5, 0.5, 0.5]

    def preprocess(self, image, return_tensors="pt"):
        assert return_tensors == "pt"
        return {"pixel_values": torch.full((1, 3, 384, 384), float(image))}


class _FakeVisionTower(torch.nn.Module):
    is_loaded = True
    image_processor = _FakeImageProcessor()

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)

    def load_model(self, device_map=None):
        self.loaded_device_map = device_map


class _FakeInnerReconModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_projector = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.GELU(),
            torch.nn.Linear(2, 2),
        )
        self.mm_inv_projector = torch.nn.Linear(2, 2)
        self.q_proj = torch.nn.Linear(2, 2)
        self.k_proj = torch.nn.Linear(2, 2)
        self.v_proj = torch.nn.Linear(2, 2)
        self.o_proj = torch.nn.Linear(2, 2)
        self.gate_proj = torch.nn.Linear(2, 2)
        self.up_proj = torch.nn.Linear(2, 2)
        self.down_proj = torch.nn.Linear(2, 2)
        self.initialize_calls = []

    def initialize_vision_modules(self, model_args, fsdp=None):
        self.initialize_calls.append((model_args, fsdp))
        self.pixel_decoder = torch.nn.Linear(2, 2)
        self.mm_inv_projector = torch.nn.Linear(2, 2)


class _FakeReconModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(
            hidden_size=3584,
            image_embed_len=729,
            recon_enable=True,
            reconstruct_image=True,
        )
        self.vision_tower = _FakeVisionTower()
        self.model = _FakeInnerReconModel()
        self.forward_kwargs = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        model = cls()
        model.path = path
        model.from_pretrained_kwargs = kwargs
        if "config" in kwargs:
            model.config = kwargs["config"]
        return model

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.vision_tower

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        hidden = torch.ones(1, 5, 3584)
        return types.SimpleNamespace(hidden_states=[hidden], boi_ids=[1], eoi_ids=[3])


def _install_reconvla_fakes(monkeypatch):
    for package_name in ["recon", "recon.model", "recon.model.language_model"]:
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    fake_lm = types.ModuleType("recon.model.language_model.recon_qwen")
    fake_lm.ReconQwen2ForCausalLM = _FakeReconModel
    monkeypatch.setitem(sys.modules, fake_lm.__name__, fake_lm)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = _FakeAutoConfig
    fake_transformers.AutoTokenizer = _FakeTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_mm_utils = types.ModuleType("recon.mm_utils")
    fake_mm_utils.tokenizer_image_token = (
        lambda prompt, tokenizer, image_token_index, return_tensors=None: torch.tensor(
            [11, image_token_index, 12]
        )
    )
    monkeypatch.setitem(sys.modules, fake_mm_utils.__name__, fake_mm_utils)

    fake_action_tokenizer = types.ModuleType("recon.action_tokenizer")

    class _FakeActionTokenizer:
        action_token_begin_idx = 744

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, action):
            action = np.asarray(action).reshape(-1)
            token_ids = [900 + idx for idx in range(action.shape[0])]
            return token_ids, " ".join(str(token_id) for token_id in token_ids)

    fake_action_tokenizer.ActionTokenizer = _FakeActionTokenizer
    fake_action_tokenizer.encode_actions = (
        lambda sentence, action_tokenizer, statistics=None: action_tokenizer(
            np.asarray([float(value) for value in sentence.split(" ")], dtype=np.float32)
        )
    )
    fake_action_tokenizer.encode_robot_obs = (
        lambda sentence, action_tokenizer, statistics=None: (
            list(range(200, 215)),
            sentence,
        )
    )
    monkeypatch.setitem(sys.modules, fake_action_tokenizer.__name__, fake_action_tokenizer)

    fake_constants = types.ModuleType("recon.constants")
    fake_constants.DEFAULT_IMAGE_TOKEN = "<image>"
    monkeypatch.setitem(sys.modules, fake_constants.__name__, fake_constants)

    fake_conversation = types.ModuleType("recon.conversation")

    class _FakeConversation:
        roles = ("USER", "ASSISTANT")

        def __init__(self):
            self.system = ""
            self.messages = []

        def copy(self):
            return _FakeConversation()

        def append_message(self, role, message):
            self.messages.append((role, message))

        def get_prompt(self):
            return "\n".join(
                [self.system]
                + [
                    f"{role}: {'' if message is None else message}"
                    for role, message in self.messages
                ]
            )

    fake_conversation.default_conversation = _FakeConversation()
    monkeypatch.setitem(sys.modules, fake_conversation.__name__, fake_conversation)
    return fake_mm_utils


def _install_fake_peft(monkeypatch, add_lora_param=True):
    fake_peft = types.ModuleType("peft")

    class _FakeTaskType:
        CAUSAL_LM = "CAUSAL_LM"

    class _FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.target_modules = kwargs["target_modules"]

    def _fake_get_peft_model(model, lora_config):
        for param in model.parameters():
            param.requires_grad_(False)
        model.peft_config_seen = lora_config
        targets = set(lora_config.target_modules)
        if add_lora_param:
            for name, module in model.named_modules():
                leaf_name = name.rsplit(".", 1)[-1]
                if name in targets or leaf_name in targets:
                    module.register_parameter("lora_A", torch.nn.Parameter(torch.ones(1)))
        return model

    fake_peft.TaskType = _FakeTaskType
    fake_peft.LoraConfig = _FakeLoraConfig
    fake_peft.get_peft_model = _fake_get_peft_model
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    return fake_peft


def test_reconvla_interface_loads_and_disables_internal_recon(monkeypatch):
    module = _load_module(monkeypatch)
    _install_reconvla_fakes(monkeypatch)

    cfg = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(
                model_path="ckpt/pretrain-checkpoint-10388",
                attn_implementation="sdpa",
                synthetic_image_token_id=-200,
                disable_internal_recon_loss=True,
            )
        )
    )

    interface = module.ReconVLAInterface(cfg)

    assert interface.model.config.recon_enable is False
    assert interface.model.config.reconstruct_image is False
    assert interface.image_token_id == -200
    assert interface.image_embed_len == 729
    assert interface.model.from_pretrained_kwargs["attn_implementation"] == "sdpa"
    assert interface.model.from_pretrained_kwargs["ignore_mismatched_sizes"] is False


def test_reconvla_interface_initializes_internal_recon_modules(monkeypatch):
    module = _load_module(monkeypatch)
    _install_reconvla_fakes(monkeypatch)
    loaded_prefixes = []
    monkeypatch.setattr(
        module.ReconVLAInterface,
        "_load_reconvla_checkpoint_prefixes",
        lambda self, prefixes: loaded_prefixes.append(prefixes),
    )
    cfg = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(
                model_path="ckpt/pretrain-checkpoint-10388",
                mm_pixel_decoder="ckpt/pretrained_vae",
                disable_internal_recon_loss=False,
            )
        )
    )

    interface = module.ReconVLAInterface(cfg)

    inner_model = interface.model.get_model()
    assert interface.model.config.recon_enable is True
    assert interface.model.config.reconstruct_image is False
    assert inner_model.initialize_calls
    assert inner_model.initialize_calls[0][0].mm_pixel_decoder == "ckpt/pretrained_vae"
    assert loaded_prefixes == [("model.mm_inv_projector.", "model.pixel_decoder.")]


def test_reconvla_interface_overrides_local_vision_tower_path(monkeypatch):
    module = _load_module(monkeypatch)
    _install_reconvla_fakes(monkeypatch)

    cfg = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(
                model_path="ckpt/pretrain-checkpoint-10388",
                vision_tower_path="ckpt/siglip-so400m-patch14-384",
            )
        )
    )

    interface = module.ReconVLAInterface(cfg)

    loaded_config = interface.model.from_pretrained_kwargs["config"]
    assert loaded_config.mm_vision_tower == "ckpt/siglip-so400m-patch14-384"
    assert interface.model.config.mm_vision_tower == "ckpt/siglip-so400m-patch14-384"


def test_reconvla_interface_applies_lora_and_freezes_multimodal(monkeypatch):
    module = _load_module(monkeypatch)
    _install_reconvla_fakes(monkeypatch)
    _install_fake_peft(monkeypatch)

    cfg = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(
                model_path="ckpt/pretrain-checkpoint-10388",
                vision_tower_path="ckpt/siglip-so400m-patch14-384",
                lora=_AttrDict(
                    enabled=True,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    train_mm_projector=False,
                    train_mm_inv_projector=False,
                ),
            )
        )
    )

    interface = module.ReconVLAInterface(cfg)
    interface.apply_language_lora(cfg.framework.reconvla.lora)

    assert interface.lora_enabled is True
    assert interface.model.peft_config_seen.kwargs["r"] == 16
    assert interface.model.peft_config_seen.kwargs["lora_alpha"] == 32
    assert interface.model.peft_config_seen.kwargs["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    trainable = [name for name, param in interface.model.named_parameters() if param.requires_grad]
    assert trainable == [
        "model.q_proj.lora_A",
        "model.k_proj.lora_A",
        "model.v_proj.lora_A",
        "model.o_proj.lora_A",
    ]
    assert all(not p.requires_grad for p in interface.model.get_vision_tower().parameters())
    assert all(not p.requires_grad for p in interface.model.get_model().mm_projector.parameters())
    assert all(not p.requires_grad for p in interface.model.get_model().mm_inv_projector.parameters())


def test_reconvla_interface_keeps_mm_projector_lora_trainable(monkeypatch):
    module = _load_module(monkeypatch)
    _install_reconvla_fakes(monkeypatch)
    _install_fake_peft(monkeypatch)

    cfg = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(
                model_path="ckpt/pretrain-checkpoint-10388",
                lora=_AttrDict(
                    enabled=True,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj"],
                    train_mm_projector=True,
                    train_mm_inv_projector=False,
                ),
            )
        )
    )

    interface = module.ReconVLAInterface(cfg)
    interface.apply_language_lora(cfg.framework.reconvla.lora)

    assert "model.mm_projector.0" in interface.model.peft_config_seen.kwargs["target_modules"]
    assert "model.mm_projector.2" in interface.model.peft_config_seen.kwargs["target_modules"]
    trainable = [name for name, param in interface.model.named_parameters() if param.requires_grad]
    assert "model.q_proj.lora_A" in trainable
    assert "model.mm_projector.0.lora_A" in trainable
    assert "model.mm_projector.2.lora_A" in trainable
    assert all(
        not param.requires_grad
        for name, param in interface.model.get_model().mm_projector.named_parameters()
        if "lora_" not in name
    )
    assert all(not p.requires_grad for p in interface.model.get_model().mm_inv_projector.parameters())
    assert all(not p.requires_grad for p in interface.model.get_vision_tower().parameters())


def test_reconvla_interface_rejects_lora_without_trainable_adapters(monkeypatch):
    module = _load_module(monkeypatch)
    _install_reconvla_fakes(monkeypatch)
    _install_fake_peft(monkeypatch, add_lora_param=False)

    cfg = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(
                model_path="ckpt/pretrain-checkpoint-10388",
                lora=_AttrDict(
                    enabled=True,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj"],
                ),
            )
        )
    )

    interface = module.ReconVLAInterface(cfg)
    with pytest.raises(RuntimeError, match="No trainable LoRA parameters"):
        interface.apply_language_lora(cfg.framework.reconvla.lora)


def test_reconvla_interface_builds_single_image_batch(monkeypatch):
    module = _load_module(monkeypatch)
    fake_mm_utils = _install_reconvla_fakes(monkeypatch)

    prompts = []

    def _tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors=None):
        prompts.append((prompt, image_token_index, return_tensors))
        return torch.tensor([101, image_token_index, 102], dtype=torch.long)

    fake_mm_utils.tokenizer_image_token = _tokenizer_image_token

    cfg = _AttrDict(framework=_AttrDict(reconvla=_AttrDict(model_path="ckpt/pretrain-checkpoint-10388")))
    interface = module.ReconVLAInterface(cfg)
    batch = interface.build_qwenvl_inputs(images=[[2]], instructions=["open drawer"])

    assert prompts == [("<image>\nopen drawer", -200, None)]
    assert batch["input_ids"].shape == (1, 3)
    assert torch.equal(batch["attention_mask"], torch.ones(1, 3, dtype=torch.bool))
    assert batch["images"].shape == (1, 3, 384, 384)
    assert batch["image_sizes"] == [(384, 384)]
    assert batch["boi_ids"] == [1]
    assert batch["eoi_ids"] == [729]


def test_reconvla_interface_forward_forwards_kwargs(monkeypatch):
    module = _load_module(monkeypatch)
    _install_reconvla_fakes(monkeypatch)

    cfg = _AttrDict(framework=_AttrDict(reconvla=_AttrDict(model_path="ckpt/pretrain-checkpoint-10388")))
    interface = module.ReconVLAInterface(cfg)
    out = interface(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        images=torch.zeros(1, 3, 384, 384),
        image_sizes=[(384, 384)],
        boi_ids=[1],
        eoi_ids=[3],
        output_hidden_states=True,
        return_dict=True,
    )

    assert out.hidden_states[-1].shape == (1, 5, 3584)
    assert out.boi_ids == [1]
    assert out.eoi_ids == [3]
    assert interface.model.forward_kwargs["output_hidden_states"] is True
    assert interface.model.forward_kwargs["return_dict"] is True
    assert "boi_ids" not in interface.model.forward_kwargs
    assert "eoi_ids" not in interface.model.forward_kwargs
