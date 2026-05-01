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
from starVLA.model.modules.uamvla.backbone_wrapper import (
    _replace_state_tokens,
    build_uamvla_backbone,
)
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
from starVLA.model.modules.uamvla.state_encoder.special_tokens import ACTION_START_TOKEN


def _move_tensors_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(v, device) for v in value)
    return value


@dataclass
class UamVLADefaultConfig:
    name: str = "UamVLA"
    embodiment: dict = field(default_factory=lambda: {"name": "franka_libero", "action_dim": 7})
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "ckpt/Qwen3-VL-8B-Instruct",
        "attn_implementation": "flash_attention_2",
    })
    action_model: dict = field(default_factory=lambda: {
        "future_action_window_size": 7, # action_horizon = 7 + 1 = 8
        "num_bins": 256,
        "action_dim": 7,
    })
    state_encoder: dict = field(default_factory=lambda: {"type": "modular", "register_special_tokens": True})
    vae: dict = field(default_factory=lambda: {"path": "ckpt/pretrained_vae"})
    aux_heads: dict = field(default_factory=lambda: {
        "action": {"enabled": True, "loss_weight": 1.0, "lr": 1.0e-4},
        "pose":   {"enabled": True, "loss_weight": 0.5, "lr": 1.0e-4},
        "future": {"enabled": True, "loss_weight": 1.0, "lr": 1.0e-4},
        "recon":  {"enabled": True, "loss_weight": 1.0, "lr": 1.0e-4},
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
    # Merge vision_extra (derived defaults) with cfg-side kwargs. yaml takes
    # precedence on conflict (e.g., both currently provide `target_resize`):
    # vision_extra fills in derived ppv-based defaults the user usually doesn't
    # touch; cfg lets the user override any of them from the training yaml.
    if cfg_heads.future.get("enabled", True) and vae is not None:
        _future_cfg = {k: v for k, v in cfg_heads.future.items() if k not in ("enabled", "lr")}
        heads["future"] = FutureHead(
            hidden_size=hidden_size, vae=vae,
            **{**(vision_extra or {}), **_future_cfg},
        )
    if cfg_heads.recon.get("enabled", True) and vae is not None:
        _recon_cfg = {k: v for k, v in cfg_heads.recon.items() if k not in ("enabled", "lr")}
        heads["recon"] = ReconHead(
            hidden_size=hidden_size, vae=vae,
            **{**(vision_extra or {}), **_recon_cfg},
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
        # Use the wrapper's hidden_size property which handles both transformers 4.x
        # (top-level config.hidden_size) and 5.x (text_config.hidden_size) layouts.
        hidden_size = self.qwen_vl_interface.hidden_size

        # Build ActionTokenizer and resize embedding table to include <ACT_i> tokens.
        from starVLA.model.modules.uamvla.data.action_tokenizer import ActionTokenizer
        _tokenizer = self.qwen_vl_interface.get_tokenizer()
        self._tokenizer = _tokenizer
        _n_bins = int(self.config.framework.action_model.get("num_bins", 256))
        self.action_tokenizer = ActionTokenizer(_tokenizer, n_bins=_n_bins)
        try:
            self.qwen_vl_interface.resize_token_embeddings(len(_tokenizer))
        except TypeError:
            # _tokenizer is a mock or does not support len(); skip resize.
            pass

        # Resolve action_token_begin_id from the newly added <ACT_0> token.
        try:
            _act0_id = int(_tokenizer.convert_tokens_to_ids("<ACT_0>"))
        except (TypeError, ValueError):
            _act0_id = 0
        self._act0_id = _act0_id

        # Verify <ACT_i> token IDs are contiguous (required for ActionLogitsProcessor
        # and _decode_generated_actions). Skip on mock tokenizers that return the
        # same id for every <ACT_*> query.
        try:
            _act_last_id = int(_tokenizer.convert_tokens_to_ids(f"<ACT_{_n_bins - 1}>"))
            if _act_last_id != _act0_id and _act_last_id != _act0_id + _n_bins - 1:
                raise RuntimeError(
                    f"Action token IDs are not contiguous: <ACT_0>={_act0_id}, "
                    f"<ACT_{_n_bins - 1}>={_act_last_id}, "
                    f"expected {_act0_id + _n_bins - 1}. "
                    "ActionLogitsProcessor and _decode_generated_actions require "
                    "contiguous IDs."
                )
        except (TypeError, ValueError):
            # Mock tokenizer in unit tests; cannot verify.
            pass

        # Resolve action_start_id from the structural <|action_start|> token registered
        # at backbone-build time. Hard-error on silent unk_token_id collisions so that
        # a missing register_structural_tokens() call surfaces here, not at training time.
        try:
            _action_start_id = int(_tokenizer.convert_tokens_to_ids(ACTION_START_TOKEN))
            _unk_id = getattr(_tokenizer, "unk_token_id", None)
            if _action_start_id is None or (
                _unk_id is not None and _action_start_id == _unk_id
            ):
                raise RuntimeError(
                    f"{ACTION_START_TOKEN} was not registered as a special token. "
                    f"Did you call register_structural_tokens() in build_uamvla_backbone?"
                )
        except (TypeError, ValueError):
            # Mock tokenizer in unit tests — convert_tokens_to_ids returned a non-int.
            # NOTE: Do NOT widen this to `except Exception`; doing so would swallow
            # the RuntimeError above and silently mis-tokenize at training time.
            _action_start_id = -1
        self.action_start_id = _action_start_id

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
            action_token_begin_id=_act0_id,
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

    def _build_labels_and_extend(
        self, examples: list, qwen_inputs: dict
    ) -> tuple[dict, torch.Tensor]:
        """Append action tokens to prompt inputs and build labels tensor.

        Returns:
            extended_qwen_inputs: copy of qwen_inputs with input_ids and
                attention_mask extended by H*7 positions per sample.
            labels: (B, L_total) — -100 at prompt+padding positions,
                action token IDs at action positions (or -100 for padded timesteps).
        """
        B = len(examples)
        H = self.action_horizon          # total action steps
        _act_dim = int(self.config.framework.embodiment.get("action_dim", 7))
        n_act = H * _act_dim             # tokens per sample
        device = qwen_inputs["input_ids"].device
        pad_id = int(
            self._tokenizer.pad_token_id
            if self._tokenizer is not None
            and self._tokenizer.pad_token_id is not None
            else 0
        )

        prompt_ids = qwen_inputs["input_ids"]       # (B, L_p)
        prompt_mask = qwen_inputs["attention_mask"] # (B, L_p)
        L_p = prompt_ids.shape[1]
        L_total = L_p + n_act

        new_ids = torch.full((B, L_total), pad_id, dtype=torch.long, device=device)
        new_mask = torch.zeros(B, L_total, dtype=torch.long, device=device)
        labels = torch.full((B, L_total), -100, dtype=torch.long, device=device)

        for i, ex in enumerate(examples):
            real_len = int(prompt_mask[i].sum().item())
            # Copy real prompt tokens (right-padded layout from build_inputs)
            new_ids[i, :real_len] = prompt_ids[i, :real_len]
            new_mask[i, :real_len] = 1

            action = ex["action"]         # (H, 7) float32 tensor
            action_msk = ex["action_mask"]  # (H, 7) long tensor

            tok_ids: list[int] = []
            lbl_ids: list[int] = []
            for h in range(H):
                step_ids = self.action_tokenizer.encode(action[h].detach().cpu().numpy())  # list[int] len 7
                tok_ids.extend(step_ids)
                # Padded timestep (all-zero mask row) → -100 in labels
                step_valid = bool(action_msk[h].any().item())
                lbl_ids.extend(step_ids if step_valid else [-100] * len(step_ids))

            act_tensor = torch.tensor(tok_ids, dtype=torch.long, device=device)
            lbl_tensor = torch.tensor(lbl_ids, dtype=torch.long, device=device)

            new_ids[i, real_len:real_len + n_act] = act_tensor
            new_mask[i, real_len:real_len + n_act] = 1
            labels[i, real_len:real_len + n_act] = lbl_tensor

        # Extend mm_token_type_ids if present (action positions get type 0 = text)
        extended = {**qwen_inputs, "input_ids": new_ids, "attention_mask": new_mask}
        if "mm_token_type_ids" in qwen_inputs and qwen_inputs["mm_token_type_ids"] is not None:
            ext_mm = torch.zeros(B, L_total, dtype=qwen_inputs["mm_token_type_ids"].dtype, device=device)
            ext_mm[:, :L_p] = qwen_inputs["mm_token_type_ids"]
            extended["mm_token_type_ids"] = ext_mm

        return extended, labels

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

        qwen_inputs, labels = self._build_labels_and_extend(examples, qwen_inputs)
        qwen_inputs = _move_tensors_to_device(qwen_inputs, self._backbone_input_device())
        labels = labels.to(qwen_inputs["input_ids"].device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = self.qwen_vl_interface(
                **qwen_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = backbone_out.hidden_states[-1]  # (B, L, H)

        batch_dict = self._collate_for_heads(examples, qwen_inputs, labels=labels)

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

    def _backbone_input_device(self) -> torch.device:
        try:
            return self.qwen_vl_interface.get_embed_tokens().weight.device
        except AttributeError:
            return next(self.qwen_vl_interface.parameters()).device

    def _collate_for_heads(self, examples: List[dict], qwen_inputs: dict, labels=None) -> dict:
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

        ``labels`` is the (B, L_total) tensor produced by
        ``_build_labels_and_extend``: -100 at prompt+padding positions and
        action token IDs at action positions. It is passed in directly rather
        than read from ``qwen_inputs`` (which never carries labels).
        """
        # NOTE: do not put qwen_inputs["pixel_values"] under "image" here. Qwen3-VL's
        # pixel_values is a flattened patch tensor (N_patches, dim) that cannot be
        # sliced as (B, V, C, H, W). The "image" key is populated only in the
        # visualize path (see visualize_batch) where aux heads expect raw views.
        batch_dict: dict = {
            "input_ids":   qwen_inputs["input_ids"],
            "labels":      labels,
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

        return _move_tensors_to_device(batch_dict, qwen_inputs["input_ids"].device)

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        """Eval-time inference: autoregressive action chunk generation.

        Builds prompt + state-spliced inputs_embeds, calls model.generate with
        ActionLogitsProcessor active by default, and decodes the generated
        action tokens via ID-range scan.

        Args:
            examples: a single example dict or list of example dicts. Each must have:
              - "image": List[PIL.Image] or np.ndarray (multi-view per spec §4.6)
              - "lang": str instruction
              - "canonical_state": canonical state tensor / dict (or omit for all examples
                if state-less inference is desired; mixing within a batch raises ValueError)
            **kwargs:
              - constrain_action_logits (bool, default True): if True, apply
                ActionLogitsProcessor to constrain post-<|action_start|> logits to
                action-bin tokens. Set False for ablations / debugging.
              - other kwargs: ignored (compat with diffusion-style frameworks).

        Returns:
            {"normalized_actions": np.ndarray of shape (B, H, action_dim)}.
            Values are in the same normalized action space as the dataset's
            ``example["action"]`` (e.g. BOUNDS_Q99 for LIBERO via the dataset's
            upstream normalizer). Callers comparing against ground-truth actions
            must already have matching normalization applied.
        """
        if not isinstance(examples, list):
            examples = [examples]

        # Validate per-example "image" is a list/tuple of views, NOT a single PIL.
        # Without this guard, `for img in e["image"]` would iterate over PIL pixels
        # and produce a confusing crash deep inside to_pil_preserve.
        for i, e in enumerate(examples):
            if not isinstance(e["image"], (list, tuple)):
                raise TypeError(
                    f"examples[{i}]['image'] must be a list/tuple of views; "
                    f"got {type(e['image']).__name__}. Wrap a single view as [img]."
                )

        from deployment.model_server.tools.image_tools import to_pil_preserve
        from transformers import LogitsProcessorList
        from starVLA.model.modules.uamvla.inference import ActionLogitsProcessor

        # Reject mixed canonical_state presence (stack_canonical can't handle partial dicts).
        states = [e.get("canonical_state") for e in examples]
        has_state = [s is not None for s in states]
        if any(has_state) and not all(has_state):
            raise ValueError(
                "predict_action requires canonical_state for all examples or none; "
                "mixed presence is not supported."
            )

        # Inference uses left padding so each row's last non-pad prompt token
        # aligns to the same column (required for batched generate to step
        # uniformly). NOTE: this mutates a process-global tokenizer attribute;
        # do not call predict_action from multiple threads in the same process.
        tokenizer = self.qwen_vl_interface.tokenizer
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            qwen_inputs = self.qwen_vl_interface.build_inputs(
                images=[[to_pil_preserve(img) for img in e["image"]] for e in examples],
                instructions=[e["lang"] for e in examples],
                canonical_state=stack_canonical(states) if all(has_state) else None,
            )
        finally:
            tokenizer.padding_side = old_padding_side

        H = self.action_horizon
        action_dim = int(self.config.framework.embodiment.get("action_dim", 7))
        chunk_len = H * action_dim
        n_bins = int(self.config.framework.action_model.get("num_bins", 256))

        constrain_action_logits = kwargs.get("constrain_action_logits", True)
        logits_processor = None
        if constrain_action_logits:
            logits_processor = LogitsProcessorList([
                ActionLogitsProcessor(
                    action_start_id=self.action_start_id,
                    action_begin_id=self._act0_id,
                    n_bins=n_bins,
                    action_chunk_len=chunk_len,
                )
            ])

        # autocast must cover both the prefill (state_encoder Linear layers run on
        # fp32 canonical_state vs. bf16 weights without it) and generate().
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen_kwargs = self._build_prefill_generate_kwargs(qwen_inputs)
            generated_ids = self.qwen_vl_interface.model.generate(
                **gen_kwargs,
                max_new_tokens=chunk_len,
                min_new_tokens=chunk_len,
                logits_processor=logits_processor,
                do_sample=False,
                use_cache=True,
            )

        normalized_actions = self._decode_generated_actions(
            generated_ids,
            prompt_len=gen_kwargs["inputs_embeds"].shape[1],
            H=H,
            action_dim=action_dim,
        )
        return {"normalized_actions": normalized_actions}

    def _build_prefill_generate_kwargs(self, qwen_inputs: dict) -> dict:
        """Prefill: embed input_ids, splice in state-token embeddings, and return
        kwargs suitable for model.generate().

        The state splice is identical to the training forward path. We keep the
        original input_ids in the generation kwargs so generation-time helpers
        and logits processors can see the textual prompt.

        This method does NOT run any transformer layers — only embedding lookup
        and state splicing. The transformer runs inside model.generate().
        """
        iface = self.qwen_vl_interface
        embed_tokens = iface.get_embed_tokens()
        qwen_inputs = _move_tensors_to_device(qwen_inputs, embed_tokens.weight.device)
        input_ids = qwen_inputs["input_ids"]
        canonical_state = qwen_inputs.get("canonical_state")
        base_embeds = embed_tokens(input_ids)

        if canonical_state is not None:
            state_embeds = iface.state_encoder(canonical_state)
            inputs_embeds = _replace_state_tokens(
                inputs_embeds=base_embeds,
                input_ids=input_ids,
                state_embeds=state_embeds,
                embodiment=iface.embodiment,
                tokenizer=iface.tokenizer,
            )
        else:
            inputs_embeds = base_embeds

        # NOTE: do NOT pass mm_token_type_ids — Qwen3VLForConditionalGeneration's
        # _validate_model_kwargs rejects unknown kwargs, and the LM-head class's
        # forward signature does not declare this field. The model derives image
        # positions internally from <|image_pad|> tokens in input_ids during
        # prefill (we keep input_ids in gen_kwargs for exactly this reason). The
        # training path needs to pass mm_token_type_ids only because the wrapper
        # nulls input_ids before calling the inner Qwen3VLModel.
        gen_kwargs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": qwen_inputs.get("attention_mask"),
            "pixel_values": qwen_inputs.get("pixel_values"),
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
        }
        return {k: v for k, v in gen_kwargs.items() if v is not None}

    def _extract_generated_tail(
        self,
        generated_ids: torch.Tensor,
        prompt_len: int,
        chunk_len: int,
    ) -> torch.Tensor:
        """Return generated new-token IDs from either prompt+new or new-only sequences.

        HuggingFace `generate(inputs_embeds=...)` may return (B, S_prompt + N_new)
        or (B, N_new) depending on model and version. Normalize to "new tokens only"
        before downstream decoding.
        """
        seq_len = int(generated_ids.shape[1])

        # New-only return: usually exactly chunk_len, or shorter if generation stopped early.
        if seq_len <= chunk_len:
            return generated_ids

        # Prompt+new return: slice after the prompt. This is the expected path when
        # input_ids are supplied to generate().
        if seq_len >= prompt_len:
            return generated_ids[:, prompt_len:]

        # Defensive fallback for unusual wrappers: keep the final action window.
        return generated_ids[:, -chunk_len:]

    def _decode_generated_actions(
        self,
        generated_ids: torch.Tensor,
        prompt_len: int,
        H: int,
        action_dim: int,
    ) -> np.ndarray:
        """Extract action tokens from the generated tail and decode to normalized actions.

        Strategy: normalize generate() return shape via _extract_generated_tail, then
        scan for action token ids in [_act0_id, _act0_id + n_bins). Tokens outside
        the range are skipped defensively. Length is normalized to H * action_dim
        via mid_bin padding (action ≈ 0) on shortfall, or truncation on overflow.

        Args:
            generated_ids: (B, S_prompt + N_new) or (B, N_new) long tensor from model.generate
            prompt_len: number of prompt tokens (= inputs_embeds.shape[1])
            H: action horizon
            action_dim: action dimensionality (typically 7 for Franka)

        Returns:
            np.ndarray of shape (B, H, action_dim), float32, in normalized action space
        """
        B = generated_ids.shape[0]
        chunk_len = H * action_dim
        n_bins = int(self.config.framework.action_model.get("num_bins", 256))
        act_min = self._act0_id
        act_max = self._act0_id + n_bins  # exclusive
        mid_id = self._act0_id + n_bins // 2

        new_ids = self._extract_generated_tail(generated_ids, prompt_len, chunk_len)

        decoded = np.zeros((B, H, action_dim), dtype=np.float32)
        for b in range(B):
            row = new_ids[b]
            mask = (row >= act_min) & (row < act_max)
            action_ids = row[mask].tolist()
            if len(action_ids) < chunk_len:
                action_ids += [mid_id] * (chunk_len - len(action_ids))
            elif len(action_ids) > chunk_len:
                action_ids = action_ids[:chunk_len]
            chunk = self.action_tokenizer.decode(action_ids).reshape(H, action_dim)
            decoded[b] = chunk
        return decoded

    def visualize_batch(self, batch, n_samples: int = 1) -> dict:
        """Iterate aux heads and call .visualize() on each, gathering wandb.Image entries.

        The trainer hands us the same ``List[dict]`` it passes to ``forward()``;
        we mirror the forward preprocessing path (build_inputs → label extend →
        backbone forward → _collate_for_heads) so per-head ``visualize`` sees a
        dict with the same keys it sees in ``compute_loss``.
        """
        if isinstance(batch, list):
            examples = batch
        elif isinstance(batch, dict) and "examples" in batch:
            examples = batch["examples"]
        else:
            raise TypeError(
                f"visualize_batch expects List[dict] (per-sample examples) or "
                f"{{'examples': List[dict]}}, got {type(batch).__name__}."
            )

        qwen_inputs = self.qwen_vl_interface.build_inputs(
            images=[e["image"] for e in examples],
            instructions=[e["lang"] for e in examples],
            canonical_state=stack_canonical([e["canonical_state"] for e in examples]),
        )
        qwen_inputs, labels = self._build_labels_and_extend(examples, qwen_inputs)
        qwen_inputs = _move_tensors_to_device(qwen_inputs, self._backbone_input_device())
        labels = labels.to(qwen_inputs["input_ids"].device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = self.qwen_vl_interface(
                **qwen_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = backbone_out.hidden_states[-1]

        batch_dict = self._collate_for_heads(examples, qwen_inputs, labels=labels)

        # Aux heads' visualize methods read batch["image"][i, view_idx] expecting
        # a (B, V, C, H, W) tensor normalized with mean=0.5/std=0.5 (so the
        # default vis_draw.tensor_to_pil round-trip recovers the original RGB).
        # The dataset emits e["image"] as List[PIL.Image] (transforms=None), so
        # build that tensor here, only when actually visualizing.
        def _pil_view_to_normed(view) -> torch.Tensor:
            if isinstance(view, torch.Tensor):
                return view  # assume caller already handled normalization
            # np.array (not np.asarray) — PIL's buffer is read-only, and
            # torch.from_numpy on a non-writable view emits a defensive
            # UserWarning. Copying once per view is cheap during viz.
            arr = np.array(view, dtype=np.uint8)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # (C,H,W) [0,1]
            return (t - 0.5) / 0.5  # (C,H,W) in [-1, 1]

        batch_dict["image"] = torch.stack([
            torch.stack([_pil_view_to_normed(v) for v in e["image"]])
            for e in examples
        ]).to(hidden.device)

        out = {}
        for name, head in self.aux_heads.items():
            if not hasattr(head, "visualize"):
                continue
            mask = batch_dict.get(f"{name}_mask")
            if mask is None:
                mask = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
            if not mask.any():
                continue
            # Forward optional head-owned viz state (e.g. PoseHead.camera_params,
            # loaded from stats_path at construction). Heads that don't carry
            # such state simply ignore the kwarg via their **kwargs.
            extra_kwargs = {}
            cam_params = getattr(head, "camera_params", None)
            if cam_params is not None:
                extra_kwargs["camera_params"] = cam_params
            imgs = head.visualize(hidden, batch_dict, mask, num_samples=n_samples, **extra_kwargs)
            for i, img in enumerate(imgs):
                out[f"viz/{name}/{i}"] = img
        return out

    def get_lr_groups(self, lr_cfg) -> list:
        groups = []
        used_params = set()

        def add_group(name: str, params, lr: float) -> None:
            unique_params = []
            for param in params:
                if not param.requires_grad:
                    continue
                param_id = id(param)
                if param_id in used_params:
                    continue
                used_params.add(param_id)
                unique_params.append(param)
            if unique_params:
                groups.append({"name": name, "params": unique_params, "lr": float(lr)})

        # Order matters for shared modules: state_encoder is nested under the
        # backbone wrapper, and ActionHead borrows the backbone lm_head.
        add_group(
            "state_encoder",
            self.qwen_vl_interface.state_encoder.parameters(),
            lr_cfg.state_encoder,
        )
        for name, head in self.aux_heads.items():
            head_cfg = self.config.framework.aux_heads[name]
            add_group(
                f"aux_head_{name}",
                head.parameters(),
                head_cfg.get("lr", lr_cfg.base),
            )
        add_group(
            "qwen_vl_interface",
            self.qwen_vl_interface.parameters(),
            lr_cfg.qwen_vl_interface,
        )
        return groups

    def supports_training_tag(self, tag: str) -> bool:
        # Phase 1: VLA only, no VLM co-training
        return tag == "vla"
