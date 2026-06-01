"""AuxVLAGR00T framework: ReconVLA/Ross-Qwen2 backbone + GR00T action head.

This framework is intentionally isolated from UamVLAGR00T. It uses the
ReconVLA checkpoint as the VLM backbone, then feeds its hidden states into the
existing GR00T action head and optional UamVLA external aux heads.
"""
from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.UamVLAOFT import UamVLAOFT
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_reconvla_pythonpath() -> None:
    reconvla_root = _repo_root() / "third_party" / "ReconVLA" / "reconvla"
    root_str = str(reconvla_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


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
            "attn_implementation": "flash_attention_2",
            "single_view_mode": "primary",
            "synthetic_image_token_id": -200,
            "disable_internal_recon_loss": True,
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
        raise NotImplementedError("ReconVLAInterface is implemented in Task 4")


@FRAMEWORK_REGISTRY.register("AuxVLAGR00T")
class AuxVLAGR00T(baseframework):
    """ReconVLA-backed GR00T framework with optional UamVLA aux heads."""

    _init_uamvla_sidecar_roots = UamVLAOFT._init_uamvla_sidecar_roots
    _init_lerobot_video_path_config = UamVLAOFT._init_lerobot_video_path_config
    _sidecar_root_for_sample = UamVLAOFT._sidecar_root_for_sample
    _maybe_build_aux_heads = UamVLAOFT._maybe_build_aux_heads
    _maybe_build_aux_loss_control = UamVLAOFT._maybe_build_aux_loss_control
    _aux_head_cfg_without_layout_keys = UamVLAOFT._aux_head_cfg_without_layout_keys
    _collate_aux = UamVLAOFT._collate_aux
    _resolve_head_mask = UamVLAOFT._resolve_head_mask
    _force_resize_640 = UamVLAOFT._force_resize_640

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
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = hidden_size
        self.action_model = get_gr00t_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

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

    def _prepare_examples(self, examples: List[dict]) -> List[dict]:
        return [
            self._unpack_lerobot_sample(example)
            if "__trajectory_id" in example
            else example
            for example in examples
        ]

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

    def _select_single_view(self, image_list, mode: str):
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
        raise ValueError(f"Unsupported single_view_mode `{mode}`. Expected `primary` or `wrist`.")

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

    def forward(self, examples: List[dict], **kwargs) -> dict:
        examples = self._prepare_examples(examples)
        recon_inputs, hidden = self._encode_reconvla_hidden(examples)
        gt_actions = [example["action"] for example in examples]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.as_tensor(
                np.asarray([
                    action.detach().float().cpu().numpy()
                    if torch.is_tensor(action)
                    else np.asarray(action)
                    for action in gt_actions
                ]),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            actions_target = actions[:, -self.action_horizon :, :]
            if actions_target.shape[1] != self.action_horizon:
                raise RuntimeError(
                    f"Expected at least {self.action_horizon} action steps, got {actions.shape[1]}."
                )
            repeat = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            actions_repeated = actions_target.repeat(repeat, 1, 1)
            hidden_repeated = hidden.repeat(repeat, 1, 1)
            state = self._state_batch_or_none(examples, hidden.device, hidden.dtype)
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
        _recon_inputs, hidden = self._encode_reconvla_hidden(examples)
        state = self._state_batch_or_none(examples, hidden.device, hidden.dtype)
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(hidden, state)
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
