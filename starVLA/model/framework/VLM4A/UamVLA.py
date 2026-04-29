"""UamVLA framework: Qwen3-VL-8B + ModularStateEncoder + 4 aux heads.

@FRAMEWORK_REGISTRY.register("UamVLA")

See: docs/superpowers/specs/2026-04-29-starvla-migration-design.md
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.uamvla.backbone_wrapper import build_uamvla_backbone
from starVLA.model.modules.uamvla.aux_heads.action_head import ActionHead
from starVLA.model.modules.uamvla.aux_heads.pose_head import PoseHead
from starVLA.model.modules.uamvla.aux_heads.future_head import FutureHead
from starVLA.model.modules.uamvla.aux_heads.recon_head import ReconHead


@dataclass
class UamVLADefaultConfig:
    name: str = "UamVLA"
    embodiment: dict = field(default_factory=lambda: {"name": "franka_libero", "action_dim": 7})
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "ckpt/Qwen3-VL-8B-Instruct",
        "attn_implementation": "flash_attention_2",
    })
    action_model: dict = field(default_factory=lambda: {
        "future_action_window_size": 7,
        "num_bins": 256,
        "action_dim": 7,
    })
    state_encoder: dict = field(default_factory=lambda: {"type": "modular", "register_special_tokens": True})
    vae: dict = field(default_factory=lambda: {"path": "ckpt/pretrained_vae"})
    aux_heads: dict = field(default_factory=lambda: {
        "action": {"enabled": True, "loss_weight": 1.0, "lr": 1.0e-4},
        "pose":   {"enabled": True, "loss_weight": 0.5, "lr": 1.0e-4},
        "future": {"enabled": True, "loss_weight": 0.1, "lr": 1.0e-4},
        "recon":  {"enabled": True, "loss_weight": 0.1, "lr": 1.0e-4},
    })


def _build_aux_heads(framework_cfg, hidden_size, vae=None, vision_extra=None, lm_head=None, action_token_begin_id=0):
    """Construct enabled aux heads. Mirrors UamVLA models/uamvla_model.py:197-218 logic."""
    heads = {}
    cfg_heads = framework_cfg.aux_heads
    if cfg_heads.action.get("enabled", True):
        heads["action"] = ActionHead(
            hidden_size=hidden_size, lm_head=lm_head,
            action_token_begin_id=action_token_begin_id,
            loss_weight=float(cfg_heads.action.get("loss_weight", 1.0)),
        )
    if cfg_heads.pose.get("enabled", True):
        heads["pose"] = PoseHead(
            hidden_size=hidden_size,
            **{k: v for k, v in cfg_heads.pose.items() if k not in ("enabled", "lr")},
        )
    if cfg_heads.future.get("enabled", True) and vae is not None:
        heads["future"] = FutureHead(
            hidden_size=hidden_size, vae=vae, **(vision_extra or {}),
            **{k: v for k, v in cfg_heads.future.items() if k not in ("enabled", "lr")},
        )
    if cfg_heads.recon.get("enabled", True) and vae is not None:
        heads["recon"] = ReconHead(
            hidden_size=hidden_size, vae=vae, **(vision_extra or {}),
            **{k: v for k, v in cfg_heads.recon.items() if k not in ("enabled", "lr")},
        )
    return heads


@FRAMEWORK_REGISTRY.register("UamVLA")
class UamVLA(baseframework):
    """UamVLA: Qwen3-VL backbone + state encoder + multi-aux-head VLA model.

    Single-loss contract with starVLA trainer: forward returns
    {"action_loss": <sum of head losses>, ...per-head metrics}.
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(UamVLADefaultConfig, config)

        self.qwen_vl_interface = build_uamvla_backbone(self.config)
        hidden_size = self.qwen_vl_interface.config.hidden_size

        # Aux heads
        from starVLA.model.modules.uamvla.components.pixel_decoder.vae import VAEPixelDecoder
        fut_on = self.config.framework.aux_heads.get("future", {}).get("enabled", True)
        rec_on = self.config.framework.aux_heads.get("recon", {}).get("enabled", True)
        needs_vae = fut_on or rec_on
        self.vae = VAEPixelDecoder(self.config.framework.vae.path) if needs_vae else None

        # vision_extra: inferred from backbone config (Qwen3-VL ppv=400 for 640px)
        vision_extra = self._derive_vision_extra(hidden_size, needs_vae)

        self.aux_heads = nn.ModuleDict(_build_aux_heads(
            self.config.framework,
            hidden_size=hidden_size,
            vae=self.vae,
            vision_extra=vision_extra,
            lm_head=self.qwen_vl_interface.get_lm_head(),
            action_token_begin_id=self.config.framework.get("action_token_begin_id", 0),
        ))

        self.action_horizon = int(self.config.framework.action_model.future_action_window_size) + 1

    def _derive_vision_extra(self, hidden_size, needs_vae):
        if not needs_vae:
            return {}
        # TODO(Phase 2): derive from cfg if multiple image sizes are supported.
        ppv = 400  # Qwen3-VL @ 640px / patch16 / merge2 → 20x20 grid
        side = int(ppv ** 0.5)
        assert side * side == ppv, "ppv must be a perfect square (DiT requires a square spatial grid)"
        return {
            "image_mean": [0.5, 0.5, 0.5],
            "image_std":  [0.5, 0.5, 0.5],
            "image_token_id": self.qwen_vl_interface.image_token_id,  # backbone wrapper exposes this
            "patches_per_view": ppv,
            "n_patches": ppv,
            "target_resize": side * 16,  # 320 for ppv=400
        }

    # TODO(Task 18): implement forward(examples) → {"action_loss": ..., ...per-head metrics}
    # TODO(Task 19): implement predict_action(examples) → {"normalized_actions": ...}
    # TODO(Task 20): implement visualize_batch(examples, outputs)
    # TODO(Task 21): implement get_lr_groups() and supports_training_tag(tag)
