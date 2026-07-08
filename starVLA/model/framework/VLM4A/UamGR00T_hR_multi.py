"""Multi-head dedicated token frameworks for UamVLAGR00T.

Both frameworks assign each enabled aux head a dedicated single-token
placeholder that is injected AFTER all image tokens in the LLM sequence.
This gives every head's tokens causal attention to the full instruction +
image context (ReconVLA §3.2).

The GR00T action model always receives hidden states with ALL aux head
tokens stripped, so action prediction is unaffected by the extra tokens.

Registered framework names:
  UamGR00T_hR_A  — Strategy A: all head tokens concatenated in a fixed
                   arbitrary order (no deliberate cross-head dependencies).
  UamGR00T_hR_B  — Strategy B: semantic hierarchy order
                   grounding → affordance → recon → depth → future,
                   where later heads can attend to earlier heads.

Sequence layouts:
  Strategy A:
    [instruction] → [images×views] →
    [recon×N][depth×N][grounding×N][affordance×N][future×N][acf×N]

  Strategy B:
    [instruction] → [images×views] →
    [grounding×N] → [affordance×N] → [recon×N] → [depth×N] →
    [future×N] → [acf×N]

YAML configuration (add under framework.aux_heads.<name>):
    head_token: "<single-token-string>"   # default shown in _DEFAULT_HEAD_TOKENS
"""
from __future__ import annotations

import numpy as np
from typing import List

import torch

from starVLA.model.framework.VLM4A.UamGR00T import UamVLAGR00T
from starVLA.model.framework.VLM4A.UamVLAOFT import UamVLAOFT
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.uamvla.aux_heads.dynamic_suite import DynamicAuxSuite


# Config key → aux_heads registration key
_CFG_TO_HEAD_KEY: dict[str, str] = {
    "grounding": "grounding_mask",
    "affordance": "affordance",
    "recon": "recon",
    "depth": "depth",
    "future": "future",
    "action_conditioned_future": "action_conditioned_future",
}

# Default placeholder token per head (all single-token emojis; verified at runtime)
_DEFAULT_HEAD_TOKENS: dict[str, str] = {
    "grounding": "🟨",
    "affordance": "🟩",
    "recon": "🎯",
    "depth": "✨",
    "future": "🟧",
    "action_conditioned_future": "🔶",
}


class _MultiHeadBase(UamVLAGR00T):
    """Shared base for Strategy-A and Strategy-B multi-head frameworks.

    Subclasses must implement ``_head_order()`` to specify token ordering.
    """

    def __init__(self, config) -> None:
        super().__init__(config)          # UamVLAGR00T.__init__
        self._register_all_head_tokens()
        self._maybe_build_dynamic_aux_suite()

    # ──────────────────────────────────────────────────────────────────
    #  Token registration
    # ──────────────────────────────────────────────────────────────────
    def _register_all_head_tokens(self) -> None:
        """Register a dedicated special token for every enabled aux head."""
        self._head_token_str: dict[str, str] = {}    # cfg_key → token str
        self._head_token_id: dict[str, int] = {}     # cfg_key → token id

        tok = self.qwen_vl_interface.processor.tokenizer
        seen_ids: dict[int, str] = {}  # id → cfg_key, for collision check

        for cfg_key, head_key in _CFG_TO_HEAD_KEY.items():
            if head_key not in self.aux_heads:
                continue

            cfg_head = self.config.framework.aux_heads.get(cfg_key, {})
            token_str = cfg_head.get("head_token", _DEFAULT_HEAD_TOKENS[cfg_key])
            ids = tok(token_str, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise RuntimeError(
                    f"head_token for '{cfg_key}'={token_str!r} must tokenize to "
                    f"exactly 1 token, got {ids}. "
                    "Set framework.aux_heads.<head>.head_token."
                )
            tid = ids[0]
            if tid in seen_ids:
                raise RuntimeError(
                    f"Token collision: '{cfg_key}' and '{seen_ids[tid]}' both "
                    f"map to token id {tid} ({token_str!r}). "
                    "Use distinct head_token values per head."
                )
            seen_ids[tid] = cfg_key
            self._head_token_str[cfg_key] = token_str
            self._head_token_id[cfg_key] = tid

            # Patch the head to use its dedicated token for hidden-state slicing.
            head = self.aux_heads[head_key]
            # ReconHead uses _condition_token_id; all others use image_token_id.
            if hasattr(head, "_condition_token_id"):
                head._condition_token_id = tid
            if hasattr(head, "image_token_id"):
                head.image_token_id = tid

        # Upgrade VLM interface to support text_suffix parameter.
        from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface  # noqa: keep near usage
        from starVLA.model.modules.vlm.QWen3_hR import _QWen3_VL_hR_Interface
        if (isinstance(self.qwen_vl_interface, _QWen3_VL_Interface)
                and not isinstance(self.qwen_vl_interface, _QWen3_VL_hR_Interface)):
            self.qwen_vl_interface.__class__ = _QWen3_VL_hR_Interface

    # ──────────────────────────────────────────────────────────────────
    #  Dynamic aux suite (optional)
    # ──────────────────────────────────────────────────────────────────
    def _maybe_build_dynamic_aux_suite(self) -> None:
        """Build DynamicAuxSuite when framework.dynamic_aux is configured.

        The suite is stored as self.aux_suite so that forward() can detect
        it and delegate to _compute_aux_training_losses.
        """
        dyn_cfg = getattr(getattr(self.config, "framework", None), "dynamic_aux", None)
        if dyn_cfg is None:
            self.aux_suite = None
            return

        self.aux_suite = DynamicAuxSuite.from_config(
            aux_heads=self.aux_heads,
            config=self.config,
        )

    def _compute_aux_training_losses(
        self,
        action_loss: "torch.Tensor",
        hidden: "torch.Tensor",
        batch_dict: dict,
        global_step: int = 0,
    ):
        """Delegate to DynamicAuxSuite when configured; else static weights.

        Returns (total_loss, log_metrics_dict).
        """
        if self.aux_suite is not None:
            masks = {
                n: self._resolve_head_mask(n, batch_dict, hidden.shape[0], hidden.device)
                for n in self.aux_heads
            }
            return self.aux_suite(
                action_loss=action_loss,
                hidden_states=hidden,
                batch=batch_dict,
                masks=masks,
                global_step=global_step,
            )

        # ── Fallback: static per-head weights (original behaviour) ────
        log_metrics = {}
        total = action_loss
        for name, head in self.aux_heads.items():
            mask = self._resolve_head_mask(
                name, batch_dict, hidden.shape[0], hidden.device
            )
            out = head.compute_loss(hidden, batch_dict, mask=mask)
            if out.loss is not None:
                total = total + out.loss
                log_metrics[f"loss/{name}_weighted"] = out.loss.detach()
            for mn, mv in out.metrics.items():
                log_metrics[self._aux_metric_log_key(name, mn)] = mv
        return total, log_metrics

    def _head_order(self) -> list[str]:
        """Return cfg_key ordering for the text_suffix. Override in subclasses."""
        raise NotImplementedError

    # ──────────────────────────────────────────────────────────────────
    #  Suffix builder
    # ──────────────────────────────────────────────────────────────────
    def _build_all_head_suffix(self) -> str | None:
        """Concatenate head tokens in _head_order() for text_suffix."""
        ppv = self._qwen_patches_per_view()
        parts = [
            self._head_token_str[k] * ppv
            for k in self._head_order()
            if k in self._head_token_str
        ]
        return "".join(parts) if parts else None

    # ──────────────────────────────────────────────────────────────────
    #  Encode: inject all head tokens after images
    # ──────────────────────────────────────────────────────────────────
    def _encode_qwen_hidden(self, examples: List[dict]):
        batch_images = [self._force_resize_640(e["image"]) for e in examples]
        examples = [
            {**e, "image": imgs}
            for e, imgs in zip(examples, batch_images)
        ]
        instructions = [e["lang"] for e in examples]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            text_suffix=self._build_all_head_suffix(),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = out.hidden_states[-1]
        self._assert_image_token_count(qwen_inputs["input_ids"], examples)
        return examples, qwen_inputs, hidden

    # ──────────────────────────────────────────────────────────────────
    #  Strip ALL head tokens before action model
    # ──────────────────────────────────────────────────────────────────
    def _hidden_for_action(
        self, hidden: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Remove every registered head token position from hidden states."""
        if not self._head_token_id:
            return hidden
        all_ids = set(self._head_token_id.values())
        keep = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        for tid in all_ids:
            keep &= (input_ids != tid)
        B, _, H = hidden.shape
        L_new = int(keep[0].sum().item())
        out = hidden.new_empty(B, L_new, H)
        for b in range(B):
            out[b] = hidden[b, keep[b]]
        return out

    # ──────────────────────────────────────────────────────────────────
    #  Forward: action uses stripped hidden, aux heads use full hidden
    # ──────────────────────────────────────────────────────────────────
    def forward(self, examples: List[dict], **kwargs) -> dict:
        examples = self._prepare_examples(examples)
        examples, qwen_inputs, hidden = self._encode_qwen_hidden(examples)
        gt_actions = [e["action"] for e in examples]

        hidden_action = self._hidden_for_action(hidden, qwen_inputs["input_ids"])

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.as_tensor(
                np.asarray([
                    a.detach().float().cpu().numpy() if torch.is_tensor(a)
                    else np.asarray(a)
                    for a in gt_actions
                ]),
                device=hidden.device, dtype=hidden.dtype,
            )
            actions_target = actions[:, -self.action_horizon:, :]
            if actions_target.shape[1] != self.action_horizon:
                raise RuntimeError(
                    f"Expected {self.action_horizon} action steps, "
                    f"got {actions.shape[1]}."
                )
            r = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            state = self._state_batch_or_none(examples, hidden.device, hidden.dtype)
            total = self.action_model(
                hidden_action.repeat(r, 1, 1),
                actions_target.repeat(r, 1, 1),
                state.repeat(r, 1, 1) if state is not None else None,
            )

        log_metrics = {"action_loss_fm": total.detach()}

        batch_dict = self._collate_aux(examples, qwen_inputs)
        assert "input_ids" in batch_dict
        total, aux_m = self._compute_aux_training_losses(
            total, hidden, batch_dict,
            global_step=UamVLAOFT._global_step_from_kwargs(kwargs),
        )
        log_metrics.update(aux_m)

        return {"action_loss": total, **log_metrics}

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        examples = self._prepare_examples(examples)
        examples, qwen_inputs, hidden = self._encode_qwen_hidden(examples)
        hidden_action = self._hidden_for_action(hidden, qwen_inputs["input_ids"])
        state = self._state_batch_or_none(examples, hidden_action.device, hidden_action.dtype)
        with torch.autocast("cuda", dtype=torch.float32):
            pred = self.action_model.predict_action(hidden_action, state)
        return {"normalized_actions": pred.detach().cpu().numpy()}


# ══════════════════════════════════════════════════════════════════════
#  Strategy A — Parallel (true engineering parallel via block mask)
# ══════════════════════════════════════════════════════════════════════
@FRAMEWORK_REGISTRY.register("UamGR00T_hR_A")
class UamVLAGR00T_hR_A(_MultiHeadBase):
    """All head tokens concatenated after images; cross-head attention blocked.

    Each aux-head token block can attend to [instruction + image] tokens and
    its own block, but is masked from attending to every other head's tokens.
    This gives true parallel conditioning: every head sees only the shared
    visual context, with no information leak from sibling heads.

    Requires attn_implementation=sdpa (flash_attention_2 does not support
    arbitrary 4-D attention masks).

    Sequence: [images] → [recon×N][depth×N][grounding×N][affordance×N]
                         [future×N][action_conditioned_future×N]
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        # Verify the attention backend supports 4-D additive masks.
        attn_impl = getattr(
            getattr(self.qwen_vl_interface, "model", None),
            "config", None,
        )
        attn_impl = getattr(attn_impl, "attn_implementation", None)
        if attn_impl == "flash_attention_2":
            raise RuntimeError(
                "UamGR00T_hR_A uses a 4-D block-parallel attention mask to "
                "prevent cross-head token attention, which is not supported by "
                "flash_attention_2. Set `framework.qwenvl.attn_implementation: "
                "sdpa` in your YAML config and re-launch."
            )

    def _head_order(self) -> list[str]:
        return [
            "recon",
            "depth",
            "grounding",
            "affordance",
            "future",
            "action_conditioned_future",
        ]

    # ──────────────────────────────────────────────────────────────────
    #  4-D block-parallel attention mask
    # ──────────────────────────────────────────────────────────────────
    def _build_parallel_aux_mask(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Build a (B, 1, L, L) additive float mask where aux-head token
        blocks are mutually invisible: head_i tokens cannot attend to
        head_j tokens (i ≠ j), but each block still attends to all
        instruction + image tokens via normal causal attention.

        Non-head positions follow the standard lower-triangular causal rule.
        Padding positions (key side) are preserved as -inf from the original
        2-D padding mask baked into the causal mask.
        """
        B, L = input_ids.shape
        device = input_ids.device

        # --- causal lower-triangular base (True = can attend) ----------
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))

        # --- assign group index per position ----------------------------
        # head_group[b, i] = h (0-based) if position i is head-h's token
        #                   = -1 otherwise (instruction / image / padding)
        head_group = torch.full((B, L), -1, dtype=torch.long, device=device)
        for h_idx, tid in enumerate(self._head_token_id.values()):
            head_group[input_ids == tid] = h_idx

        # --- vectorised cross-block mask --------------------------------
        # cross_block[b, q, k] = True iff q and k belong to *different* heads
        q_group = head_group.unsqueeze(2)   # (B, L, 1)
        k_group = head_group.unsqueeze(1)   # (B, 1, L)
        cross_block = (q_group >= 0) & (k_group >= 0) & (q_group != k_group)

        # parallel_mask: causal AND NOT cross-block  →  (B, L, L) bool
        parallel_mask = causal.unsqueeze(0) & ~cross_block

        # --- convert to additive float mask (B, 1, L, L) ----------------
        # 0.0 = allowed, -inf = blocked  (HuggingFace sdpa convention)
        float_mask = torch.zeros(B, 1, L, L, dtype=torch.bfloat16, device=device)
        float_mask.masked_fill_(~parallel_mask.unsqueeze(1), float("-inf"))
        return float_mask

    # ──────────────────────────────────────────────────────────────────
    #  Override encode: inject 4-D mask before LLM forward
    # ──────────────────────────────────────────────────────────────────
    def _encode_qwen_hidden(self, examples: List[dict]):
        batch_images = [self._force_resize_640(e["image"]) for e in examples]
        examples = [
            {**e, "image": imgs}
            for e, imgs in zip(examples, batch_images)
        ]
        instructions = [e["lang"] for e in examples]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            text_suffix=self._build_all_head_suffix(),
        )

        # Replace the 2-D padding mask with our 4-D block-parallel mask.
        # The causal lower-triangular base already handles padding because
        # left-padded positions appear before all valid tokens and are
        # never attended to by the causal rule.
        parallel_mask = self._build_parallel_aux_mask(qwen_inputs["input_ids"])
        qwen_inputs = {**qwen_inputs, "attention_mask": parallel_mask}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = out.hidden_states[-1]
        self._assert_image_token_count(qwen_inputs["input_ids"], examples)
        return examples, qwen_inputs, hidden


# ══════════════════════════════════════════════════════════════════════
#  Strategy B — Hierarchy (semantic dependency order)
# ══════════════════════════════════════════════════════════════════════
@FRAMEWORK_REGISTRY.register("UamGR00T_hR_B")
class UamVLAGR00T_hR_B(_MultiHeadBase):
    """Head tokens in semantic hierarchy order.

    Sequence: [images] → [grounding×N] → [affordance×N] → [recon×N]
                         → [depth×N] → [future×N]
                         → [action_conditioned_future×N]

    Later heads can attend to earlier heads:
      - affordance sees grounding  (knows which object to interact with)
      - recon sees grounding+affordance  (knows target and interaction point)
      - depth sees semantic context (focuses on target area)
      - future sees full scene understanding
    """

    def _head_order(self) -> list[str]:
        return [
            "grounding",
            "affordance",
            "recon",
            "depth",
            "future",
            "action_conditioned_future",
        ]
