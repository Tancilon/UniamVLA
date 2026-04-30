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
from starVLA.model.modules.uamvla.state_encoder.special_tokens import ACTION_START_TOKEN


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
        batch_dict: dict = {
            "input_ids":   qwen_inputs["input_ids"],
            "labels":      labels,
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

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        """Eval-time inference: run backbone + action head only; returns normalized actions.

        Only the action head runs at inference time — pose, future, and recon
        heads are skipped.

        Args:
            examples: a single example dict or a list of example dicts.  Each
                dict must have:
                  - "image"  : List[PIL.Image] or np.ndarray (multi-view per §4.6)
                  - "lang"   : str instruction
                  - "canonical_state": canonical state tensor / array

        Returns:
            {"normalized_actions": np.ndarray of shape (B, T, 7)}
        """
        if not isinstance(examples, list):
            examples = [examples]

        from deployment.model_server.tools.image_tools import to_pil_preserve

        # e["image"] is List[PIL.Image] per spec §4.6 (multi-view).
        # to_pil_preserve handles nested lists natively (recurses into list).
        qwen_inputs = self.qwen_vl_interface.build_inputs(
            images=[[to_pil_preserve(img) for img in e["image"]] for e in examples],
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

        pred = self.aux_heads["action"].predict(hidden, batch=qwen_inputs)
        normalized_actions = self._decode_action_tokens(pred, batch_size=len(examples))

        return {"normalized_actions": normalized_actions}

    def _decode_action_tokens(self, head_output, batch_size: int):
        """Decode (B, L) argmax token ids → (B, T, 7) normalized actions as np.ndarray.

        head_output is HeadOutput(predictions={"token_ids": Tensor(B, L)}) from
        ActionHead.predict.

        Phase 1 note: actual decoding is gated on Task 24 (dataloader labels
        wiring) and Task 33 (L5 eval dry-run).  _extract_action_tokens_for_sample
        raises NotImplementedError until that infrastructure exists.
        """
        import numpy as np

        action_tokenizer = self.action_tokenizer

        pred_ids = head_output.predictions["token_ids"]  # (B, L)
        H = self.action_horizon  # T

        decoded = np.zeros((batch_size, H, 7), dtype=np.float32)
        for i in range(batch_size):
            token_ids_i = self._extract_action_tokens_for_sample(pred_ids[i], H)  # length H*7 list
            chunk = action_tokenizer.decode(token_ids_i).reshape(H, 7)
            decoded[i] = chunk
        return decoded

    def _extract_action_tokens_for_sample(self, pred_ids_row, action_horizon: int):
        """Find the action_horizon*7 action token positions in pred_ids_row ((L,) tensor).

        Phase 1 strategy (in priority order):
          1. Use labels mask from upstream batch (labels != -100 marks action positions).
             Requires Task 24 (dataloader plugin) to wire up labels at eval time.
          2. Scan input_ids for action_token_begin_id sentinel, then take the
             next action_horizon*7 positions.
             Requires action_token_begin_id to be correctly set in config.

        Neither is wired up yet.  This method raises NotImplementedError until
        Task 24 + Task 33 provide the supporting infrastructure.  Do not attempt
        a heuristic implementation here — see the migration design doc §4.3.
        """
        raise NotImplementedError(
            "predict_action decoding requires labels mask or action_token_begin_id "
            "from the dataloader. Wire up via Task 24 (dataloader plugin) and Task 33 (L5 eval). "
            "See docs/superpowers/specs/2026-04-29-starvla-migration-design.md §4.3."
        )

    def visualize_batch(self, batch: dict, n_samples: int = 1) -> dict:
        """Iterate aux heads and call .visualize() on each, gathering wandb.Image entries."""
        out = {}
        # Recompute hidden states (the trainer doesn't always persist them)
        hidden = self._backbone_forward_for_viz(batch)
        for name, head in self.aux_heads.items():
            if not hasattr(head, "visualize"):
                continue
            mask = batch.get(f"{name}_mask")
            if mask is None or mask.any():
                imgs = head.visualize(hidden, batch, mask, num_samples=n_samples)
                for i, img in enumerate(imgs):
                    out[f"viz/{name}/{i}"] = img
        return out

    def _backbone_forward_for_viz(self, batch):
        """Run backbone forward on a viz batch — returns hidden states only."""
        if "examples" in batch:
            from starVLA.model.modules.uamvla.collator_helpers import stack_canonical
            examples = batch["examples"]
            qwen_inputs = self.qwen_vl_interface.build_inputs(
                images=[e["image"] for e in examples],
                instructions=[e["lang"] for e in examples],
                canonical_state=stack_canonical([e["canonical_state"] for e in examples]),
            )
        else:
            # Assume batch is a pre-stacked dict already compatible with backbone forward
            qwen_inputs = batch
        with torch.no_grad():
            backbone_out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
        return backbone_out.hidden_states[-1]

    def get_lr_groups(self, lr_cfg) -> list:
        # Deduplicate: exclude state_encoder params from the qwen_vl_interface group
        # to avoid double-counting (state_encoder is a submodule of qwen_vl_interface).
        state_enc_params = set(id(p) for p in self.qwen_vl_interface.state_encoder.parameters())
        qwen_params = [p for p in self.qwen_vl_interface.parameters() if id(p) not in state_enc_params]
        groups = [
            {"name": "qwen_vl_interface",
             "params": qwen_params,
             "lr": float(lr_cfg.qwen_vl_interface)},
            {"name": "state_encoder",
             "params": list(self.qwen_vl_interface.state_encoder.parameters()),
             "lr": float(lr_cfg.state_encoder)},
        ]
        for name, head in self.aux_heads.items():
            head_cfg = self.config.framework.aux_heads[name]
            groups.append({
                "name": f"aux_head_{name}",
                "params": list(head.parameters()),
                "lr": float(head_cfg.get("lr", lr_cfg.base)),
            })
        return groups

    def supports_training_tag(self, tag: str) -> bool:
        # Phase 1: VLA only, no VLM co-training
        return tag == "vla"
