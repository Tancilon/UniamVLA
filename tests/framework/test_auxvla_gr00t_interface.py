from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


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
        _maybe_build_aux_heads = lambda self: None
        _maybe_build_aux_loss_control = lambda self: None
        _aux_head_cfg_without_layout_keys = staticmethod(lambda cfg: dict(cfg))
        _unpack_lerobot_sample = lambda self, sample: sample
        _collate_aux = lambda self, examples, qwen_inputs: {}
        _resolve_head_mask = lambda self, name, batch, batch_size, device: None
        _force_resize_640 = lambda self, image_list: image_list
        _global_step_from_kwargs = staticmethod(lambda kwargs: 0)

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

    def __init__(self):
        self.calls = []

    @classmethod
    def from_pretrained(cls, path, use_fast=False):
        tokenizer = cls()
        tokenizer.path = path
        tokenizer.use_fast = use_fast
        return tokenizer


class _FakeImageProcessor:
    image_mean = [0.5, 0.5, 0.5]

    def preprocess(self, image, return_tensors="pt"):
        assert return_tensors == "pt"
        return {"pixel_values": torch.full((1, 3, 384, 384), float(image))}


class _FakeVisionTower:
    is_loaded = True
    image_processor = _FakeImageProcessor()

    def load_model(self, device_map=None):
        self.loaded_device_map = device_map


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
        self.forward_kwargs = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        model = cls()
        model.path = path
        model.from_pretrained_kwargs = kwargs
        return model

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
    fake_transformers.AutoTokenizer = _FakeTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_mm_utils = types.ModuleType("recon.mm_utils")
    fake_mm_utils.tokenizer_image_token = (
        lambda prompt, tokenizer, image_token_index, return_tensors=None: torch.tensor(
            [11, image_token_index, 12]
        )
    )
    monkeypatch.setitem(sys.modules, fake_mm_utils.__name__, fake_mm_utils)
    return fake_mm_utils


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
