"""AuxVLAGR00T framework: ReconVLA/Ross-Qwen2 backbone + GR00T action head.

This framework is intentionally isolated from UamVLAGR00T. It uses the
ReconVLA checkpoint as the VLM backbone, then feeds its hidden states into the
existing GR00T action head and optional UamVLA external aux heads.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image

from starVLA.model.framework.VLM4A.UamVLAOFT import UamVLAOFT
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)

_RECONVLA_CALVIN_TARGET_SIZE = 384
_RECONVLA_CALVIN_CROP_NUMERATOR = 14
_RECONVLA_CALVIN_CROP_DENOMINATOR = 27


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_reconvla_pythonpath() -> None:
    reconvla_root = _repo_root() / "third_party" / "ReconVLA" / "reconvla"
    root_str = str(reconvla_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_to_plain_dict(cfg) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, (list, tuple)):
        return [_cfg_to_plain_dict(value) for value in cfg]
    if hasattr(cfg, "_is_list_config") and cfg._is_list_config():
        return [_cfg_to_plain_dict(value) for value in cfg.values()]
    if isinstance(cfg, dict):
        return {key: _cfg_to_plain_dict(value) for key, value in cfg.items()}
    if hasattr(cfg, "items"):
        return {key: _cfg_to_plain_dict(value) for key, value in cfg.items()}
    return cfg


def _freeze_module(module) -> None:
    if module is None or not hasattr(module, "parameters"):
        return
    for param in module.parameters():
        param.requires_grad_(False)


def _to_rgb_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _make_aux_input_ids(
    batch_size: int,
    seq_len: int,
    boi_ids,
    eoi_ids,
    image_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    aux_input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    if len(boi_ids) != batch_size or len(eoi_ids) != batch_size:
        raise RuntimeError(
            f"visual span batch mismatch: batch_size={batch_size}, "
            f"boi={len(boi_ids)}, eoi={len(eoi_ids)}"
        )

    for batch_idx, (boi, eoi) in enumerate(zip(boi_ids, eoi_ids)):
        if boi is None or eoi is None:
            raise RuntimeError(f"sample {batch_idx}: missing visual span from ReconVLA output")
        boi_i = int(boi.item() if torch.is_tensor(boi) else boi)
        eoi_i = int(eoi.item() if torch.is_tensor(eoi) else eoi)
        if boi_i < 0 or eoi_i < boi_i or eoi_i >= seq_len:
            raise RuntimeError(
                f"sample {batch_idx}: invalid visual span boi={boi_i}, eoi={eoi_i}, seq_len={seq_len}"
            )
        aux_input_ids[batch_idx, boi_i : eoi_i + 1] = int(image_token_id)
    return aux_input_ids


@dataclass
class AuxVLAGR00TDefaultConfig:
    name: str = "AuxVLAGR00T"
    reconvla: dict = field(
        default_factory=lambda: {
            "model_path": "ckpt/pretrain-checkpoint-10388",
            "vision_tower_path": "ckpt/siglip-so400m-patch14-384",
            "attn_implementation": "flash_attention_2",
            "single_view_mode": "primary",
            "inference_mode": "gr00t",
            "ar_input_mode": "official_compose",
            "action_stat_path": "third_party/ReconVLA/reconvla/statistics.yaml",
            "double_instruction": True,
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
            "synthetic_image_token_id": -200,
            "disable_internal_recon_loss": True,
            "lora": {
                "enabled": False,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "init_lora_weights": True,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "train_mm_projector": True,
                "train_mm_inv_projector": False,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 8,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
                "cross_attention_dim": 3584,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )
    obs_image_size: Optional[list] = None
    vae: dict = field(default_factory=lambda: {"path": "ckpt/pretrained_vae"})
    aux_loss_control: dict = field(default_factory=dict)
    aux_heads: dict = field(default_factory=dict)


class ReconVLAInterface(nn.Module):
    def __init__(self, config):
        super().__init__()
        _ensure_reconvla_pythonpath()
        from recon.model.language_model.recon_qwen import ReconQwen2ForCausalLM
        import recon.mm_utils as recon_mm_utils
        from transformers import AutoConfig, AutoTokenizer

        self.config = config
        recon_cfg = config.framework.get("reconvla", {})
        self.model_path = str(recon_cfg.get("model_path", "ckpt/pretrain-checkpoint-10388"))
        self.image_token_id = int(recon_cfg.get("synthetic_image_token_id", -200))
        self._tokenizer_image_token = recon_mm_utils.tokenizer_image_token
        self._process_images = getattr(recon_mm_utils, "process_images", None)

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "ignore_mismatched_sizes": False,
        }
        attn_implementation = recon_cfg.get("attn_implementation", None)
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.processor = SimpleNamespace(tokenizer=self.tokenizer)
        model_config = AutoConfig.from_pretrained(self.model_path)
        vision_tower_path = self._resolve_vision_tower_path(recon_cfg, model_config)
        if vision_tower_path is not None:
            model_config.mm_vision_tower = vision_tower_path
        load_kwargs["config"] = model_config
        self.model = ReconQwen2ForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        self.model.config.use_cache = False

        if bool(recon_cfg.get("disable_internal_recon_loss", True)):
            self.model.config.recon_enable = False
            self.model.config.reconstruct_image = False

        vision_tower = self.model.get_vision_tower()
        if not getattr(vision_tower, "is_loaded", True):
            vision_tower.load_model(device_map=None)
        self.image_processor = vision_tower.image_processor
        self.image_embed_len = int(getattr(self.model.config, "image_embed_len", 729))
        self.lora_enabled = False

    def _resolve_vision_tower_path(self, recon_cfg, model_config):
        configured = recon_cfg.get("vision_tower_path", None) or recon_cfg.get("mm_vision_tower", None)
        if configured:
            return str(configured)

        original = getattr(model_config, "mm_vision_tower", None)
        if not original:
            return None
        original_str = str(original)
        original_path = Path(original_str)
        if original_path.is_absolute() or original_path.exists():
            return original_str if original_path.exists() else None

        relative = original_str[2:] if original_str.startswith("./") else original_str
        candidates = [
            Path(self.model_path).parent / relative,
            _repo_root() / relative,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _base_reconvla_model(self):
        if hasattr(self.model, "get_base_model"):
            return self.model.get_base_model()
        base_model = getattr(self.model, "base_model", None)
        if base_model is not None:
            wrapped_model = getattr(base_model, "model", None)
            if wrapped_model is not None:
                return wrapped_model
        return self.model

    def _inner_reconvla_model(self):
        base_model = self._base_reconvla_model()
        if hasattr(base_model, "get_model"):
            return base_model.get_model()
        wrapped_model = getattr(base_model, "model", None)
        if wrapped_model is not None:
            return wrapped_model
        return base_model

    def _matched_lora_targets(self, target_modules: list[str]) -> set[str]:
        targets = {str(target) for target in target_modules}
        matched: set[str] = set()
        for name, _module in self.model.named_modules():
            leaf_name = name.rsplit(".", 1)[-1]
            if name in targets:
                matched.add(name)
            elif leaf_name in targets:
                matched.add(leaf_name)
        return matched

    def _module_path(self, target_module) -> str | None:
        for name, module in self.model.named_modules():
            if module is target_module:
                return name
        return None

    def _linear_lora_targets_under(self, module_attr: str) -> list[str]:
        inner_model = self._inner_reconvla_model()
        module = getattr(inner_model, module_attr, None)
        if module is None:
            return []
        module_path = self._module_path(module)
        if module_path is None:
            return []

        targets: list[str] = []
        for child_name, child_module in module.named_modules():
            if isinstance(child_module, nn.Linear):
                targets.append(
                    module_path
                    if child_name == ""
                    else f"{module_path}.{child_name}"
                )
        return targets

    def _expand_lora_targets(self, lora_cfg: dict) -> list[str]:
        target_modules = [str(target) for target in (lora_cfg.get("target_modules") or [])]
        if bool(lora_cfg.get("train_mm_projector", False)):
            target_modules.extend(self._linear_lora_targets_under("mm_projector"))
        if bool(lora_cfg.get("train_mm_inv_projector", False)):
            target_modules.extend(self._linear_lora_targets_under("mm_inv_projector"))
        return list(dict.fromkeys(target_modules))

    def _freeze_module_except_lora(self, module) -> None:
        if module is None or not hasattr(module, "named_parameters"):
            _freeze_module(module)
            return
        for name, param in module.named_parameters():
            param.requires_grad_("lora_" in name.lower())

    def _freeze_non_language_multimodal_modules(self, lora_cfg: dict) -> None:
        base_model = self._base_reconvla_model()
        inner_model = self._inner_reconvla_model()
        vision_tower = None
        if hasattr(base_model, "get_vision_tower"):
            vision_tower = base_model.get_vision_tower()
        elif hasattr(inner_model, "get_vision_tower"):
            vision_tower = inner_model.get_vision_tower()
        _freeze_module(vision_tower)
        if bool(lora_cfg.get("train_mm_projector", False)):
            self._freeze_module_except_lora(getattr(inner_model, "mm_projector", None))
        else:
            _freeze_module(getattr(inner_model, "mm_projector", None))
        if bool(lora_cfg.get("train_mm_inv_projector", False)):
            self._freeze_module_except_lora(getattr(inner_model, "mm_inv_projector", None))
        else:
            _freeze_module(getattr(inner_model, "mm_inv_projector", None))

    def lora_trainable_parameter_names(self) -> list[str]:
        return [
            name
            for name, param in self.model.named_parameters()
            if param.requires_grad and "lora_" in name.lower()
        ]

    def apply_language_lora(self, lora_cfg) -> None:
        lora_cfg = _cfg_to_plain_dict(lora_cfg)
        if not bool(lora_cfg.get("enabled", False)):
            return
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "AuxVLAGR00T LoRA requires peft. Install peft or set "
                "framework.reconvla.lora.enabled=false."
            ) from exc

        target_modules = self._expand_lora_targets(lora_cfg)
        if not target_modules:
            raise ValueError("framework.reconvla.lora.target_modules must be non-empty")
        matched_targets = self._matched_lora_targets(target_modules)
        if not matched_targets:
            raise ValueError(
                "No ReconVLA modules matched LoRA target_modules="
                f"{target_modules}. Expected Qwen2/Ross names like q_proj, k_proj, "
                "v_proj, o_proj, gate_proj, up_proj, down_proj."
            )

        task_type_name = str(lora_cfg.get("task_type", "CAUSAL_LM"))
        task_type = getattr(TaskType, task_type_name, task_type_name)
        peft_cfg = LoraConfig(
            task_type=task_type,
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            init_lora_weights=lora_cfg.get("init_lora_weights", True),
            target_modules=target_modules,
            bias=str(lora_cfg.get("bias", "none")),
        )
        self.model = get_peft_model(self.model, peft_cfg)
        self._freeze_non_language_multimodal_modules(lora_cfg)
        trainable_lora = self.lora_trainable_parameter_names()
        if not trainable_lora:
            raise RuntimeError(
                "No trainable LoRA parameters were created for AuxVLAGR00T. "
                "Check framework.reconvla.lora.target_modules."
            )
        self.lora_enabled = True
        logger.info(
            "AuxVLAGR00T LoRA enabled with %d trainable adapter tensors",
            len(trainable_lora),
        )

    @property
    def device(self):
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            return torch.device("cpu")
        return first_param.device

    def _process_image(self, image):
        image_size = getattr(image, "size", None)
        if self._process_images is not None:
            pixel_values = self._process_images([image], self.image_processor, self.model.config)
            if pixel_values.ndim == 4:
                pixel_values = pixel_values[0]
        else:
            pixel_values = self.image_processor.preprocess(
                image,
                return_tensors="pt",
            )["pixel_values"][0]
        if image_size is None:
            image_size = (pixel_values.shape[-1], pixel_values.shape[-2])
        return pixel_values, tuple(image_size)

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        if solutions is not None:
            raise ValueError("AuxVLAGR00T does not support language/action-token solutions")
        if len(images) != len(instructions):
            raise AssertionError("Images and instructions must have the same length")

        input_ids = []
        image_tensors = []
        image_sizes = []
        boi_ids = []
        eoi_ids = []
        for sample_images, instruction in zip(images, instructions):
            if len(sample_images) != 1:
                raise RuntimeError(
                    f"ReconVLA single-view path expects exactly one image, got {len(sample_images)}"
                )
            prompt = f"<image>\n{instruction}"
            ids = self._tokenizer_image_token(
                prompt,
                self.tokenizer,
                self.image_token_id,
                return_tensors=None,
            )
            if not torch.is_tensor(ids):
                ids = torch.as_tensor(ids, dtype=torch.long)
            image_positions = torch.where(ids == self.image_token_id)[0]
            if image_positions.numel() != 1:
                raise RuntimeError(
                    f"Expected exactly one image token in ReconVLA prompt, got {image_positions.numel()}"
                )
            boi = int(image_positions[0].item())
            boi_ids.append(boi)
            eoi_ids.append(boi + self.image_embed_len - 1)
            input_ids.append(ids)
            pixel_values, image_size = self._process_image(sample_images[0])
            image_tensors.append(pixel_values)
            image_sizes.append(image_size)

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)
        images_tensor = torch.stack(image_tensors, dim=0)
        device = self.device
        return {
            "input_ids": padded_input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "images": images_tensor.to(device),
            "image_sizes": image_sizes,
            "boi_ids": boi_ids,
            "eoi_ids": eoi_ids,
        }

    def forward(self, **kwargs):
        boi_ids = kwargs.pop("boi_ids", None)
        eoi_ids = kwargs.pop("eoi_ids", None)
        kwargs.setdefault("use_cache", False)
        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("return_dict", True)
        outputs = self.model(**kwargs)
        if boi_ids is not None:
            setattr(outputs, "boi_ids", boi_ids)
        if eoi_ids is not None:
            setattr(outputs, "eoi_ids", eoi_ids)
        return outputs


@FRAMEWORK_REGISTRY.register("AuxVLAGR00T")
class AuxVLAGR00T(baseframework):
    """ReconVLA-backed GR00T framework with optional UamVLA aux heads."""

    _init_uamvla_sidecar_roots = UamVLAOFT._init_uamvla_sidecar_roots
    _init_lerobot_video_path_config = UamVLAOFT._init_lerobot_video_path_config
    _sidecar_root_for_sample = UamVLAOFT._sidecar_root_for_sample
    _image_future_video_path = UamVLAOFT._image_future_video_path
    _image_future_frame_index = staticmethod(UamVLAOFT._image_future_frame_index)
    _load_npy_sidecar = staticmethod(UamVLAOFT._load_npy_sidecar)
    _load_depth_target = UamVLAOFT._load_depth_target
    _load_grounding_mask = UamVLAOFT._load_grounding_mask
    _load_affordance_heatmap = UamVLAOFT._load_affordance_heatmap
    _image_action_future_frame_index = UamVLAOFT._image_action_future_frame_index
    _load_image_future = UamVLAOFT._load_image_future
    _load_image_action_future = UamVLAOFT._load_image_action_future
    _maybe_build_aux_heads = UamVLAOFT._maybe_build_aux_heads
    _maybe_build_aux_loss_control = UamVLAOFT._maybe_build_aux_loss_control
    _aux_head_cfg_without_layout_keys = staticmethod(UamVLAOFT._aux_head_cfg_without_layout_keys)
    _collate_aux = UamVLAOFT._collate_aux
    _resolve_head_mask = UamVLAOFT._resolve_head_mask
    _force_resize_640 = UamVLAOFT._force_resize_640
    _to_action_numpy = staticmethod(UamVLAOFT._to_action_numpy)
    _to_rgb_pil = staticmethod(UamVLAOFT._to_rgb_pil)
    _fit_for_viz = staticmethod(UamVLAOFT._fit_for_viz)
    _draw_action_comparison = staticmethod(UamVLAOFT._draw_action_comparison)
    _make_visualization_canvas = UamVLAOFT.__dict__["_make_visualization_canvas"]
    _pil_to_normalized_chw = UamVLAOFT.__dict__["_pil_to_normalized_chw"]
    _make_visualization_image_batch = UamVLAOFT.__dict__["_make_visualization_image_batch"]

    def __init__(self, config) -> None:
        baseframework.__init__(self)
        self.config = merge_framework_config(AuxVLAGR00TDefaultConfig, config)
        self._init_reconvla_components()
        self._init_uamvla_sidecars()
        self.aux_heads = nn.ModuleDict()
        self._maybe_build_aux_heads()
        if hasattr(self, "_maybe_build_aux_loss_control"):
            self._maybe_build_aux_loss_control()

    def _init_reconvla_components(self) -> None:
        from starVLA.model.modules.action_model.GR00T_ActionHeader import (
            get_action_model as get_gr00t_action_model,
        )

        self.qwen_vl_interface = ReconVLAInterface(self.config)
        recon_cfg = _cfg_get(self.config.framework, "reconvla", {})
        lora_cfg = _cfg_get(recon_cfg, "lora", {})
        if bool(_cfg_get(lora_cfg, "enabled", False)):
            self.qwen_vl_interface.apply_language_lora(lora_cfg)
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = hidden_size
        self.action_model = get_gr00t_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _reconvla_lora_enabled(self) -> bool:
        qwen_interface = getattr(self, "qwen_vl_interface", None)
        if bool(getattr(qwen_interface, "lora_enabled", False)):
            return True
        framework_cfg = _cfg_get(getattr(self, "config", None), "framework", {})
        recon_cfg = _cfg_get(framework_cfg, "reconvla", {})
        lora_cfg = _cfg_get(recon_cfg, "lora", {})
        return bool(_cfg_get(lora_cfg, "enabled", False))

    def _trainer_freeze_patterns(self) -> list[str]:
        trainer_cfg = _cfg_get(getattr(self, "config", None), "trainer", {})
        freeze_modules = _cfg_get(trainer_cfg, "freeze_modules", "")
        if freeze_modules is None:
            return []
        if isinstance(freeze_modules, str):
            return [pattern.strip() for pattern in freeze_modules.split(",") if pattern.strip()]
        if isinstance(freeze_modules, (list, tuple)):
            return [str(pattern).strip() for pattern in freeze_modules if str(pattern).strip()]
        return [str(freeze_modules).strip()] if str(freeze_modules).strip() else []

    def _resolve_module_path(self, module_path: str):
        module = self
        for attr in module_path.split("."):
            module = getattr(module, attr)
        return module

    def _frozen_param_ids_from_config(self) -> set[int]:
        frozen_param_ids: set[int] = set()
        for freeze_path in self._trainer_freeze_patterns():
            try:
                module = self._resolve_module_path(freeze_path)
            except AttributeError:
                logger.warning("freeze module path does not exist: %s", freeze_path)
                continue
            if hasattr(module, "parameters"):
                frozen_param_ids.update(id(param) for param in module.parameters())
        return frozen_param_ids

    def get_lr_groups(self, lr_cfg):
        lr_cfg = _cfg_to_plain_dict(lr_cfg)
        base_lr = float(lr_cfg.get("base", 1.0e-4))
        lora_enabled = self._reconvla_lora_enabled()
        freeze_patterns = self._trainer_freeze_patterns()
        if lora_enabled and any(
            pattern == "qwen_vl_interface" or pattern.startswith("qwen_vl_interface.")
            for pattern in freeze_patterns
        ):
            raise RuntimeError(
                "AuxVLAGR00T LoRA is enabled, but trainer.freeze_modules includes "
                "`qwen_vl_interface`. This would freeze LoRA adapters; set "
                "trainer.freeze_modules=null and rely on the framework LR groups."
            )

        excluded_param_ids = self._frozen_param_ids_from_config()
        if lora_enabled and hasattr(getattr(self, "qwen_vl_interface", None), "named_parameters"):
            for name, param in self.qwen_vl_interface.named_parameters():
                if "lora_" not in name.lower():
                    excluded_param_ids.add(id(param))

        used_param_ids: set[int] = set()
        param_groups: list[dict] = []

        def _module_params_for_group(module_name: str, module) -> list[torch.nn.Parameter]:
            if (
                lora_enabled
                and (
                    module_name == "qwen_vl_interface"
                    or module_name.startswith("qwen_vl_interface.")
                )
                and hasattr(module, "named_parameters")
            ):
                return [
                    param
                    for name, param in module.named_parameters()
                    if "lora_" in name.lower()
                ]
            if not hasattr(module, "parameters"):
                return []
            return list(module.parameters())

        def _append_group(name: str, params, lr: float) -> None:
            group_params = [
                param
                for param in params
                if param.requires_grad
                and id(param) not in excluded_param_ids
                and id(param) not in used_param_ids
            ]
            if not group_params:
                return
            param_groups.append({"params": group_params, "lr": float(lr), "name": name})
            used_param_ids.update(id(param) for param in group_params)

        for module_name, lr in lr_cfg.items():
            if module_name == "base":
                continue
            try:
                module = self._resolve_module_path(module_name)
            except AttributeError:
                logger.warning("LR module path does not exist: %s", module_name)
                continue
            _append_group(
                module_name,
                _module_params_for_group(str(module_name), module),
                float(lr),
            )

        remaining_params = [
            param
            for param in self.parameters()
            if param.requires_grad
            and id(param) not in excluded_param_ids
            and id(param) not in used_param_ids
        ]
        _append_group("base", remaining_params, base_lr)
        return param_groups

    def _init_uamvla_sidecars(self) -> None:
        self._init_uamvla_sidecar_roots()
        self._init_lerobot_video_path_config()
        self._image_target_cache: "OrderedDict[tuple[str, int, int], torch.Tensor]" = OrderedDict()
        self._image_target_cache_maxsize = int(
            self.config.datasets.vla_data.get("image_target_cache_maxsize", 256)
        )
        self.aux_state_slice = self.config.datasets.vla_data.get(
            "aux_state_slice",
            {
                "target_pose_rot6d": [15, 21],
                "target_pose_trans": [21, 24],
                "static_cam_rot6d": [24, 30],
                "static_cam_trans": [30, 33],
            },
        )

    def _qwen_image_size(self) -> int:
        vision_tower = self.qwen_vl_interface.model.get_vision_tower()
        vision_cfg = getattr(vision_tower, "config", None)
        return int(getattr(vision_cfg, "image_size", 384))

    def _qwen_vision_layout(self) -> dict[str, int]:
        patches = int(getattr(self.qwen_vl_interface, "image_embed_len", 729))
        grid_size = int(round(patches**0.5))
        if grid_size * grid_size != patches:
            raise RuntimeError(
                f"ReconVLA image_embed_len must be square for aux heads, got {patches}"
            )
        decode_size = int(
            getattr(self.qwen_vl_interface.model.config, "decode_image_size", grid_size * 16)
        )
        return {
            "image_size": self._qwen_image_size(),
            "grid_size": grid_size,
            "patches_per_view": patches,
            "target_size": grid_size,
            "target_resize": decode_size,
        }

    def _qwen_patches_per_view(self) -> int:
        return self._qwen_vision_layout()["patches_per_view"]

    def _compose_reconvla_style_image_target(self, example: dict) -> dict:
        """Build ReconVLA CALVIN target_image: top crop + bottom wrist image."""
        if "image_target" not in example:
            raise RuntimeError(
                "AuxVLAGR00T recon training requires `image_target` sidecar crop "
                "to compose ReconVLA-style crop-plus-wrist target."
            )
        image_list = example.get("image")
        if not isinstance(image_list, (list, tuple)) or len(image_list) < 2:
            raise RuntimeError(
                "AuxVLAGR00T recon training requires a wrist image at "
                "`example['image'][1]` to compose ReconVLA-style target."
            )

        crop = example["image_target"]
        if not torch.is_tensor(crop):
            crop = torch.as_tensor(np.asarray(crop), dtype=torch.float32)
        if crop.ndim != 3 or crop.shape[0] not in {1, 3, 4}:
            raise RuntimeError(
                "AuxVLAGR00T recon target crop must be CHW with 1, 3, or 4 "
                f"channels, got shape {tuple(crop.shape)}."
            )

        target_size = _RECONVLA_CALVIN_TARGET_SIZE
        crop_height = (
            target_size
            * _RECONVLA_CALVIN_CROP_NUMERATOR
            // _RECONVLA_CALVIN_CROP_DENOMINATOR
        )
        wrist_height = target_size - crop_height
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

        crop_img = _to_rgb_pil(crop).resize((target_size, crop_height), resample)
        wrist_img = _to_rgb_pil(image_list[1]).resize((target_size, wrist_height), resample)
        combined = Image.new("RGB", (target_size, target_size))
        combined.paste(crop_img, (0, 0))
        combined.paste(wrist_img, (0, crop_height))

        arr = np.array(combined, dtype=np.uint8)
        out = dict(example)
        out["image_target"] = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return out

    def _prepare_examples(
        self,
        examples: List[dict],
        require_reconvla_target: bool = False,
    ) -> List[dict]:
        prepared = [
            self._unpack_lerobot_sample(example)
            if "__trajectory_id" in example
            else example
            for example in examples
        ]
        if require_reconvla_target:
            prepared = [
                self._compose_reconvla_style_image_target(example)
                for example in prepared
            ]
        return prepared

    def _unpack_lerobot_sample(self, sample: dict) -> dict:
        out = UamVLAOFT._unpack_lerobot_sample(self, sample)
        if "state" in sample:
            out["state"] = self._extract_gr00t_state_from_packed_calvin_state(sample["state"])
        return out

    @staticmethod
    def _extract_gr00t_state_from_packed_calvin_state(state) -> torch.Tensor:
        if not torch.is_tensor(state):
            state = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        else:
            state = state.to(dtype=torch.float32)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)
        elif state.ndim > 1:
            state = state.reshape(-1, state.shape[-1])[0]
        if state.shape[-1] < 7:
            raise RuntimeError(
                f"Expected packed state with at least 7 dims, got shape {tuple(state.shape)}."
            )
        return state[..., :7].reshape(1, 7)

    def _state_batch_or_none(self, examples: List[dict], device, dtype):
        state_dim = int(self.config.framework.action_model.get("state_dim", 0) or 0)
        if state_dim <= 0 or "state" not in examples[0]:
            return None
        states = [
            self._extract_gr00t_state_from_packed_calvin_state(example["state"])
            for example in examples
        ]
        return torch.as_tensor(np.asarray(states), device=device, dtype=dtype)

    def _action_model_compute_dtype(self, fallback_dtype: torch.dtype) -> torch.dtype:
        parameters = getattr(self.action_model, "parameters", None)
        if parameters is None:
            return fallback_dtype
        for param in parameters():
            if torch.is_floating_point(param):
                return param.dtype
        return fallback_dtype

    def _prepare_gr00t_action_inputs(
        self,
        examples: List[dict],
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        action_dtype = self._action_model_compute_dtype(hidden.dtype)
        gt_actions = [example["action"] for example in examples]
        actions = torch.as_tensor(
            np.asarray([
                action.detach().float().cpu().numpy()
                if torch.is_tensor(action)
                else np.asarray(action)
                for action in gt_actions
            ]),
            device=hidden.device,
            dtype=action_dtype,
        )
        actions_target = actions[:, -self.action_horizon :, :]
        if actions_target.shape[1] != self.action_horizon:
            raise RuntimeError(
                f"Expected at least {self.action_horizon} action steps, got {actions.shape[1]}."
            )
        state = self._state_batch_or_none(examples, hidden.device, action_dtype)
        return hidden.to(dtype=action_dtype), actions_target, state

    def _select_single_view(self, image_list, mode: str):
        if mode == "concat_vertical":
            if not isinstance(image_list, (list, tuple)) or len(image_list) < 2:
                raise RuntimeError("single_view_mode=concat_vertical requires primary and wrist images")
            image_size = self._qwen_image_size()
            top_height = image_size // 2
            bottom_height = image_size - top_height
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            primary = _to_rgb_pil(image_list[0]).resize((image_size, top_height), resample)
            wrist = _to_rgb_pil(image_list[1]).resize((image_size, bottom_height), resample)
            combined = Image.new("RGB", (image_size, image_size))
            combined.paste(primary, (0, 0))
            combined.paste(wrist, (0, top_height))
            return combined
        if not isinstance(image_list, (list, tuple)):
            return image_list
        if mode == "primary":
            if len(image_list) < 1:
                raise RuntimeError("single_view_mode=primary requires at least one image")
            return image_list[0]
        if mode == "wrist":
            if len(image_list) < 2:
                raise RuntimeError("single_view_mode=wrist requires a wrist image at index 1")
            return image_list[1]
        raise ValueError(
            f"Unsupported single_view_mode `{mode}`. "
            "Expected `primary`, `wrist`, or `concat_vertical`."
        )

    def _single_view_images(self, examples: List[dict]) -> list[list]:
        recon_cfg = self.config.framework.get("reconvla", {})
        mode = recon_cfg.get("single_view_mode", "primary")
        return [[self._select_single_view(example["image"], mode)] for example in examples]

    def _encode_reconvla_hidden(self, examples: List[dict]):
        batch_images = self._single_view_images(examples)
        instructions = [example["lang"] for example in examples]
        recon_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **recon_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.hidden_states[-1]
        image_token_id = int(
            self.config.framework.get("reconvla", {}).get("synthetic_image_token_id", -200)
        )
        aux_input_ids = _make_aux_input_ids(
            batch_size=hidden.shape[0],
            seq_len=hidden.shape[1],
            boi_ids=outputs.boi_ids,
            eoi_ids=outputs.eoi_ids,
            image_token_id=image_token_id,
            device=hidden.device,
        )
        recon_inputs["aux_input_ids"] = aux_input_ids
        recon_inputs["model_input_ids"] = recon_inputs["input_ids"]
        recon_inputs["input_ids"] = aux_input_ids
        return recon_inputs, hidden

    @staticmethod
    def _aux_metric_log_key(head_name: str, metric_name: str) -> str:
        prefix = f"{head_name}_"
        metric_core = (
            metric_name[len(prefix) :]
            if metric_name.startswith(prefix)
            else metric_name
        )
        return f"{head_name}_{metric_core}_raw"

    def _compute_aux_training_losses(
        self,
        total: torch.Tensor,
        hidden: torch.Tensor,
        batch_dict: dict,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        aux_heads = getattr(self, "aux_heads", {})
        if hasattr(self, "aux_suite"):
            masks = {
                name: self._resolve_head_mask(name, batch_dict, hidden.shape[0], hidden.device)
                for name in aux_heads
            }
            aux_loss, aux_metrics = self.aux_suite(
                action_loss=total,
                hidden_states=hidden,
                batch=batch_dict,
                masks=masks,
                global_step=global_step,
            )
            return total + aux_loss, aux_metrics

        log_metrics: dict = {}
        for name, head in aux_heads.items():
            mask = self._resolve_head_mask(name, batch_dict, hidden.shape[0], hidden.device)
            out = head.compute_loss(hidden, batch_dict, mask=mask)
            if out.loss is not None:
                total = total + out.loss
                log_metrics[f"{name}_loss_weighted"] = out.loss.detach()
            for metric_name, metric_value in out.metrics.items():
                log_metrics[self._aux_metric_log_key(name, metric_name)] = metric_value
        return total, log_metrics

    def _reconvla_inference_mode(self) -> str:
        recon_cfg = self.config.framework.get("reconvla", {})
        return str(recon_cfg.get("inference_mode", "gr00t"))

    def _reconvla_ar_input_mode(self) -> str:
        recon_cfg = self.config.framework.get("reconvla", {})
        return str(recon_cfg.get("ar_input_mode", "official_compose"))

    @staticmethod
    def _extract_reconvla_ar_robot_obs(example: dict) -> np.ndarray:
        if "robot_obs" in example:
            robot_obs = example["robot_obs"]
        elif (
            isinstance(example.get("uamvla_raw_state"), dict)
            and "robot_obs" in example["uamvla_raw_state"]
        ):
            robot_obs = example["uamvla_raw_state"]["robot_obs"]
        else:
            raise RuntimeError(
                "AuxVLAGR00T reconvla_ar_normalized inference requires raw 15-D robot_obs "
                "at `example['robot_obs']` or `example['uamvla_raw_state']['robot_obs']`."
            )

        robot_obs = np.asarray(robot_obs, dtype=np.float32).copy().reshape(-1)
        if robot_obs.shape[0] != 15:
            raise RuntimeError(
                "AuxVLAGR00T reconvla_ar_normalized inference requires raw 15-D robot_obs, "
                f"got shape {robot_obs.shape}."
            )
        return robot_obs

    def _reconvla_ar_robot_obs_batch(self, examples: List[dict]) -> np.ndarray:
        return np.stack(
            [self._extract_reconvla_ar_robot_obs(example) for example in examples],
            axis=0,
        ).astype(np.float32)

    def _reconvla_ar_images(self, examples: List[dict], input_mode: str) -> list:
        if input_mode == "official_compose":
            return [example["image"] for example in examples]
        if input_mode == "auxvla_compose":
            return self._single_view_images(examples)
        raise ValueError(
            f"Unsupported framework.reconvla.ar_input_mode={input_mode!r}. "
            "Expected `official_compose` or `auxvla_compose`."
        )

    def forward(self, examples: List[dict], **kwargs) -> dict:
        examples = self._prepare_examples(
            examples,
            require_reconvla_target="recon" in getattr(self, "aux_heads", {}),
        )
        recon_inputs, hidden = self._encode_reconvla_hidden(examples)

        with torch.autocast("cuda", dtype=torch.float32):
            hidden_for_action, actions_target, state = self._prepare_gr00t_action_inputs(
                examples,
                hidden,
            )
            repeat = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            actions_repeated = actions_target.repeat(repeat, 1, 1)
            hidden_repeated = hidden_for_action.repeat(repeat, 1, 1)
            state_repeated = state.repeat(repeat, 1, 1) if state is not None else None
            total = self.action_model(hidden_repeated, actions_repeated, state_repeated)

        log_metrics = {"action_loss_fm": total.detach()}
        batch_dict = self._collate_aux(examples, recon_inputs)
        assert "input_ids" in batch_dict, "future/recon heads require input_ids"
        total, aux_metrics = self._compute_aux_training_losses(
            total,
            hidden,
            batch_dict,
            global_step=UamVLAOFT._global_step_from_kwargs(kwargs),
        )
        log_metrics.update(aux_metrics)
        return {"action_loss": total, **log_metrics}

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        examples = self._prepare_examples(examples)

        inference_mode = self._reconvla_inference_mode()
        if inference_mode == "reconvla_ar_normalized":
            input_mode = self._reconvla_ar_input_mode()
            robot_obs = self._reconvla_ar_robot_obs_batch(examples)
            pred_actions = self.qwen_vl_interface.generate_normalized_actions(
                images=self._reconvla_ar_images(examples, input_mode),
                instructions=[example["lang"] for example in examples],
                robot_obs=robot_obs,
                input_mode=input_mode,
                action_horizon=int(
                    self.config.framework.action_model.get(
                        "action_horizon",
                        self.action_horizon,
                    )
                ),
                action_dim=int(self.config.framework.action_model.get("action_dim", 7)),
            )
            return {"normalized_actions": np.asarray(pred_actions, dtype=np.float32)}

        if inference_mode != "gr00t":
            raise ValueError(
                f"Unsupported framework.reconvla.inference_mode={inference_mode!r}. "
                "Expected `gr00t` or `reconvla_ar_normalized`."
            )

        _recon_inputs, hidden = self._encode_reconvla_hidden(examples)
        action_dtype = self._action_model_compute_dtype(hidden.dtype)
        hidden_for_action = hidden.to(dtype=action_dtype)
        state = self._state_batch_or_none(examples, hidden.device, action_dtype)
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(hidden_for_action, state)
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def _visualization_forward_context(
        self,
        selected: list[dict],
    ) -> tuple[list[dict], np.ndarray, torch.Tensor, dict]:
        """Run one ReconVLA + GR00T inference pass for action and aux visualizers."""
        examples = self._prepare_examples(
            selected,
            require_reconvla_target="recon" in getattr(self, "aux_heads", {}),
        )
        recon_inputs, hidden = self._encode_reconvla_hidden(examples)
        action_dtype = self._action_model_compute_dtype(hidden.dtype)
        hidden_for_action = hidden.to(dtype=action_dtype)
        state = self._state_batch_or_none(examples, hidden.device, action_dtype)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(hidden_for_action, state)

        batch_dict = self._collate_aux(examples, recon_inputs)
        batch_dict["image"] = self._make_visualization_image_batch(
            examples,
            device=hidden.device,
        )
        batch_dict["instruction"] = [example["lang"] for example in examples]

        return examples, pred_actions.detach().cpu().numpy(), hidden, batch_dict

    @torch.inference_mode()
    def visualize_batch(
        self,
        batch: List[dict],
        n_samples: int = 1,
        distributed_all_ranks: bool = False,
    ) -> dict:
        """Visualize AuxVLAGR00T action predictions plus enabled aux heads."""
        if not isinstance(batch, list):
            batch = [batch]
        limit = min(max(int(n_samples), 0), len(batch))
        if limit == 0:
            return {}

        selected = batch[:limit]
        was_training = bool(getattr(self, "training", False))
        self.eval()
        try:
            examples, pred_actions, hidden_states, batch_dict = self._visualization_forward_context(selected)
            if pred_actions.ndim == 2:
                pred_actions = pred_actions[None, ...]

            try:
                import wandb
            except ImportError:
                wandb = None

            outputs = {}
            for idx, sample in enumerate(examples):
                pred = pred_actions[idx]
                horizon = int(getattr(self, "action_horizon", pred.shape[0]))

                gt_action = None
                if "action" in sample:
                    gt_action = self._to_action_numpy(sample["action"])
                    if gt_action.ndim >= 2:
                        gt_action = gt_action[-horizon:, : pred.shape[-1]]

                images = sample.get("image", [])
                if isinstance(images, Image.Image) or not isinstance(images, (list, tuple)):
                    images = [images]

                canvas = self._make_visualization_canvas(
                    images=list(images),
                    pred_action=pred,
                    gt_action=gt_action,
                )
                caption = str(sample.get("lang", ""))
                outputs[f"viz/auxvla_gr00t/action_{idx}"] = (
                    wandb.Image(canvas, caption=caption) if wandb is not None else canvas
                )

            return self._collect_aux_head_visualizations_auxvla_gr00t(
                hidden_states,
                batch_dict,
                num_samples=limit,
                outputs=outputs,
                distributed_all_ranks=distributed_all_ranks,
            )
        finally:
            if was_training:
                self.train()

    def _collect_aux_head_visualizations_auxvla_gr00t(
        self,
        hidden_states: torch.Tensor,
        batch_dict: dict,
        num_samples: int,
        outputs: dict,
        distributed_all_ranks: bool = False,
    ) -> dict:
        """Append enabled aux-head visualizations under ``viz/auxvla_gr00t``."""
        aux_heads = getattr(self, "aux_heads", {})
        if not aux_heads:
            return outputs

        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        profile = os.environ.get("UAMVLA_AUX_PROFILE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0") or 0)
        for name, head in aux_heads.items():
            if not hasattr(head, "visualize"):
                continue
            try:
                mask = self._resolve_head_mask(name, batch_dict, batch_size, device)
                valid_count = int(mask.to(dtype=torch.bool).sum().item())
                if distributed_all_ranks and dist.is_initialized():
                    valid_any = torch.tensor([int(valid_count > 0)], device=device, dtype=torch.int32)
                    dist.all_reduce(valid_any, op=dist.ReduceOp.MIN)
                    if int(valid_any.item()) == 0:
                        if profile:
                            print(
                                f"[uamvla-aux-profile][rank{rank}] "
                                f"step=viz head={name} skip valid={valid_count}/{batch_size}",
                                flush=True,
                            )
                        continue
                if profile:
                    print(
                        f"[uamvla-aux-profile][rank{rank}] "
                        f"step=viz head={name} start valid={valid_count}/{batch_size}",
                        flush=True,
                    )
                    t_start = time.perf_counter()
                else:
                    t_start = None
                kwargs = {}
                if name == "pose":
                    kwargs["camera_params"] = getattr(head, "camera_params", None)
                images = head.visualize(
                    hidden_states,
                    batch_dict,
                    mask=mask,
                    num_samples=num_samples,
                    **kwargs,
                )
                for idx, image in enumerate(images or []):
                    outputs[f"viz/auxvla_gr00t/{name}_{idx}"] = image
                if profile:
                    print(
                        f"[uamvla-aux-profile][rank{rank}] "
                        f"step=viz head={name} done elapsed={time.perf_counter() - t_start:.3f}s",
                        flush=True,
                    )
            except Exception as exc:
                logger.warning("AuxVLAGR00T %s visualization failed: %s", name, exc)
        return outputs
