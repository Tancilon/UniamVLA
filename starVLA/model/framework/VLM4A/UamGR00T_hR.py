"""UamGR00T_hR: UamVLAGR00T with dedicated h_R reconstructive tokens.

Sequence layout:
    [instruction] → [image×N_views] → [recon×patches_per_view]

Key design choices:
  - Recon tokens (🎯) attend to full instruction + image context (ReconVLA §3.2).
  - Action model sees only [instruction + image] hidden states — recon tokens
    are stripped before passing to GR00T DiT to avoid contamination.
  - Aux heads (ReconHead, DepthHead, …) receive the full hidden sequence.
"""
from __future__ import annotations

import numpy as np
from typing import List

import torch

from starVLA.model.framework.VLM4A.UamGR00T import UamVLAGR00T
from starVLA.model.framework.VLM4A.UamVLAOFT import UamVLAOFT
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("UamGR00T_hR")
class UamVLAGR00T_hR(UamVLAGR00T):
    """UamVLAGR00T variant with dedicated h_R tokens for ReconHead."""

    _DEFAULT_RECON_TOKEN = "🎯"

    def __init__(self, config) -> None:
        super().__init__(config)
        self._register_recon_token()

    # ──────────────────────────────────────────────────────────────────
    #  h_R token registration
    # ──────────────────────────────────────────────────────────────────
    def _register_recon_token(self) -> None:
        """Register recon token and patch ReconHead + VLM interface."""
        self.recon_token: str | None = None
        self.recon_token_id: int | None = None

        if "recon" not in self.aux_heads:
            return

        cfg_recon = self.config.framework.aux_heads.get("recon", {})
        token_str = cfg_recon.get("recon_token", self._DEFAULT_RECON_TOKEN)
        ids = self.qwen_vl_interface.processor.tokenizer(
            token_str, add_special_tokens=False,
        )["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(
                f"recon_token={token_str!r} must tokenize to exactly 1 token, "
                f"got {ids}. Set framework.aux_heads.recon.recon_token."
            )
        self.recon_token = token_str
        self.recon_token_id = ids[0]

        # Switch ReconHead to use dedicated recon token positions.
        self.aux_heads["recon"]._condition_token_id = self.recon_token_id

        # Upgrade VLM interface to accept text_suffix.
        from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
        from starVLA.model.modules.vlm.QWen3_hR import _QWen3_VL_hR_Interface
        if (isinstance(self.qwen_vl_interface, _QWen3_VL_Interface)
                and not isinstance(self.qwen_vl_interface, _QWen3_VL_hR_Interface)):
            self.qwen_vl_interface.__class__ = _QWen3_VL_hR_Interface

    # ──────────────────────────────────────────────────────────────────
    #  Encode: inject recon tokens after images
    # ──────────────────────────────────────────────────────────────────
    def _encode_qwen_hidden(self, examples: List[dict]):
        """Like parent but injects recon tokens AFTER images via text_suffix."""
        batch_images = [self._force_resize_640(example["image"]) for example in examples]
        examples = [
            {**example, "image": images}
            for example, images in zip(examples, batch_images)
        ]
        instructions = [example["lang"] for example in examples]

        text_suffix = (
            self.recon_token * self._qwen_patches_per_view()
            if self.recon_token else None
        )
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            text_suffix=text_suffix,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = qwenvl_outputs.hidden_states[-1]
        self._assert_image_token_count(qwen_inputs["input_ids"], examples)
        return examples, qwen_inputs, hidden

    # ──────────────────────────────────────────────────────────────────
    #  Helper: strip recon token positions for action model
    # ──────────────────────────────────────────────────────────────────
    def _hidden_for_action(
        self, hidden: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return hidden states with recon token positions removed.

        Action model should not attend to h_R tokens — those are purely
        for visual reconstruction supervision.
        """
        if not self.recon_token:
            return hidden
        keep = input_ids != self.recon_token_id  # (B, L) bool
        B, _, H = hidden.shape
        L_new = int(keep[0].sum().item())
        out = hidden.new_empty(B, L_new, H)
        for b in range(B):
            out[b] = hidden[b, keep[b]]
        return out

    # ──────────────────────────────────────────────────────────────────
    #  Forward: action model uses stripped hidden; aux heads use full
    # ──────────────────────────────────────────────────────────────────
    def forward(self, examples: List[dict], **kwargs) -> dict:
        examples = self._prepare_examples(examples)
        examples, qwen_inputs, hidden = self._encode_qwen_hidden(examples)
        gt_actions = [example["action"] for example in examples]

        # Strip recon tokens before action model
        hidden_action = self._hidden_for_action(hidden, qwen_inputs["input_ids"])

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.as_tensor(
                np.asarray([
                    action.detach().float().cpu().numpy() if torch.is_tensor(action)
                    else np.asarray(action)
                    for action in gt_actions
                ]),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            actions_target = actions[:, -self.action_horizon:, :]
            if actions_target.shape[1] != self.action_horizon:
                raise RuntimeError(
                    f"Expected at least {self.action_horizon} action steps, "
                    f"got {actions.shape[1]}."
                )

            repeated_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            actions_repeated = actions_target.repeat(repeated_steps, 1, 1)
            hidden_action_repeated = hidden_action.repeat(repeated_steps, 1, 1)

            state = self._state_batch_or_none(examples, hidden.device, hidden.dtype)
            state_repeated = state.repeat(repeated_steps, 1, 1) if state is not None else None

            total = self.action_model(hidden_action_repeated, actions_repeated, state_repeated)

        log_metrics = {"action_loss_fm": total.detach()}

        # Aux heads use full hidden (including h_R positions)
        batch_dict = self._collate_aux(examples, qwen_inputs)
        assert "input_ids" in batch_dict
        if hasattr(self, "_compute_aux_training_losses"):
            total, aux_metrics = self._compute_aux_training_losses(
                total, hidden, batch_dict,
                global_step=UamVLAOFT._global_step_from_kwargs(kwargs),
            )
            log_metrics.update(aux_metrics)
        elif hasattr(self, "aux_suite"):
            masks = {
                name: self._resolve_head_mask(name, batch_dict, hidden.shape[0], hidden.device)
                for name in self.aux_heads
            }
            aux_loss, aux_metrics = self.aux_suite(
                action_loss=total, hidden_states=hidden, batch=batch_dict,
                masks=masks, global_step=int(kwargs.get("global_step", 0) or 0),
            )
            total = total + aux_loss
            log_metrics.update(aux_metrics)
        else:
            for name, head in self.aux_heads.items():
                mask = self._resolve_head_mask(name, batch_dict, hidden.shape[0], hidden.device)
                out = head.compute_loss(hidden, batch_dict, mask=mask)
                if out.loss is not None:
                    total = total + out.loss
                    log_metrics[f"{name}_loss_weighted"] = out.loss.detach()
                for metric_name, metric_value in out.metrics.items():
                    log_metrics[self._aux_metric_log_key(name, metric_name)] = metric_value

        return {"action_loss": total, **log_metrics}

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        examples = self._prepare_examples(examples)
        examples, qwen_inputs, hidden = self._encode_qwen_hidden(examples)
        # Strip recon tokens — action model should not see h_R positions
        hidden_action = self._hidden_for_action(hidden, qwen_inputs["input_ids"])
        state = self._state_batch_or_none(examples, hidden_action.device, hidden_action.dtype)
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(hidden_action, state)
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
