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
from starVLA.model.modules.uamvla.collator_helpers import (
    stack_canonical,
    stack_optional_tensor_fields,
    stack_pose_gt,
    stack_static_cam_extrinsic,
)


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

    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Training forward.

        Aux head losses are summed into a single ``"action_loss"`` entry so
        the starVLA trainer (which only knows about ``action_loss``) can
        backprop through the multi-head model. Per-head losses + metrics
        are also returned (detached) for logging.

        Args:
            examples: per-sample list of dicts. Each example carries the
                inputs the backbone + heads need (image, lang,
                canonical_state, plus optional fields like image_target,
                image_future, point_cloud, pose_gt, static_cam_extrinsic).

        Returns:
            ``{"action_loss": Tensor, "<head>_loss": Tensor, "<head>_<metric>": ...}``
        """
        qwen_inputs = self.qwen_vl_interface.build_inputs(
            images=[e["image"] for e in examples],
            instructions=[e["lang"] for e in examples],
            canonical_state=stack_canonical([e["canonical_state"] for e in examples]),
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = self.qwen_vl_interface(
                **qwen_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = backbone_out.hidden_states[-1]  # (B, L, H)

        batch_dict = self._collate_for_heads(examples, qwen_inputs)

        total = torch.tensor(0.0, device=hidden.device, requires_grad=True)
        log_metrics: dict = {}
        for name, head in self.aux_heads.items():
            mask = batch_dict.get(f"{name}_mask")
            if mask is None:
                mask = torch.ones(
                    hidden.shape[0], dtype=torch.bool, device=hidden.device,
                )
            out = head.compute_loss(hidden, batch_dict, mask=mask)
            if out.loss is not None:
                total = total + out.loss
                log_metrics[f"{name}_loss"] = out.loss.detach()
            for mk, mv in out.metrics.items():
                log_metrics[f"{name}_{mk}"] = mv

        return {**log_metrics, "action_loss": total}

    def _collate_for_heads(self, examples: List[dict], qwen_inputs: dict) -> dict:
        """Stack per-sample optional fields into batch tensors with masks.

        Combines:
            * the qwen-side input ids / labels / pixel_values that
              ``build_inputs`` already produced,
            * the per-sample optional tensor fields
              (``image_target``, ``image_future``, ``point_cloud``)
              with batch-zero-padding + masks,
            * pose targets (nested dict),
            * static cam extrinsic (nested dict),
            * head-mask aliases (``recon_mask``, ``future_mask``).

        ``labels`` is read via ``qwen_inputs.get("labels")``: the current
        Qwen3-VL ``build_inputs`` does NOT emit labels (action-token labels
        are produced by the dataloader's collate path or the ActionHead's
        own labelization step — Task 24 territory). ``None`` flows through
        the batch dict and ActionHead.compute_loss will fail if it tries to
        read ``batch["labels"]`` on real training data; that gap is resolved
        in Task 24 / Task 31, not here.
        """
        batch_dict: dict = {
            "input_ids":   qwen_inputs["input_ids"],
            "labels":      qwen_inputs.get("labels"),
            "image":       qwen_inputs.get("pixel_values"),
            "instruction": [e["lang"] for e in examples],
        }

        # Optional tensor fields with batch-zero-padding + masks.
        batch_dict.update(stack_optional_tensor_fields(
            examples, ["image_target", "image_future", "point_cloud"],
        ))

        # Aux head mask aliases (heads read f"{head_name}_mask").
        if "image_target_mask" in batch_dict:
            batch_dict["recon_mask"] = batch_dict["image_target_mask"]
        if "image_future_mask" in batch_dict:
            batch_dict["future_mask"] = batch_dict["image_future_mask"]

        # Pose targets (nested dict) + pose_mask.
        pose_out = stack_pose_gt(examples)
        if pose_out is not None:
            batch_dict["pose_gt"] = pose_out["pose_gt"]
            batch_dict["pose_mask"] = pose_out["pose_mask"]

        # Static cam extrinsic (per-sample, optional).
        cam_out = stack_static_cam_extrinsic(examples)
        if cam_out is not None:
            batch_dict["static_cam_extrinsic"] = cam_out["static_cam_extrinsic"]
            batch_dict["static_cam_extrinsic_mask"] = cam_out["static_cam_extrinsic_mask"]

        return batch_dict

    # TODO(Task 19): implement predict_action(examples) → {"normalized_actions": ...}
    # TODO(Task 20): implement visualize_batch(examples, outputs)
    # TODO(Task 21): implement get_lr_groups() and supports_training_tag(tag)
