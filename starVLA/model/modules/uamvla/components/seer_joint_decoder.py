"""SeerJointDecoder: unified self-attention + Seer block mask.

Replaces the two independent cross-attn modules (FutureCrossAttnBranch and
SeerMLPActionHead) with a single joint self-attention decoder that processes
future and action tokens in the same sequence as Qwen hidden:

    [qwen_hidden (L+K) | future_query_tokens (N_f) | action_pred_tokens (T_act)]
              ↓  masked self-attention (Seer block mask)
    [qwen_out          | future_tokens (N_f)        | action_feats (T_act)     ]

Seer block mask (Seer seer_model.py §3):
  qwen(L+K)    → sees: qwen only   (no future, no action)
  future(N_f)  → sees: qwen+future (not action)
  action(T_act)→ sees: everything

This exactly mirrors Seer's causal obs/action token visibility rules while
keeping Qwen3-VL as the backbone (replacing GPT2).

Architecture:
  1. Concatenate [qwen_hidden | future_query | action_query]
  2. Seer block mask prevents qwen from seeing future/action (gradient isolation)
  3. L joint self-attn blocks
  4. Slice future_tokens → SeerViTDecoder per view (MAE reconstruction loss)
  5. Slice action_feats  → MLP → arm (Tanh) + gripper (±1)
"""
from __future__ import annotations

import logging
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.uamvla.components.future_cross_attn import SeerViTDecoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Joint self-attention block (bidirectional, accepts additive attn_mask)
# ---------------------------------------------------------------------------

class _JointSelfAttnBlock(nn.Module):
    """Standard pre-norm transformer block with full self-attention.

    Accepts an additive ``attn_mask`` (0=allowed, -inf=blocked) and an
    optional ``key_padding_mask`` (True=ignore) — same interface as
    ``nn.MultiheadAttention``.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# SeerJointDecoder
# ---------------------------------------------------------------------------

class SeerJointDecoder(nn.Module):
    """Seer-aligned joint decoder for UamGR00T_DT.

    Args:
        d_model:            Qwen LLM hidden size.
        num_views:          Camera views (e.g. 2 for primary+wrist).
        num_obs_tokens:     Future query tokens per view (Seer base = 9).
        action_dim:         Total action dimensionality (arm + gripper).
        action_pred_steps:  Action chunk length (T_act).
        num_joint_layers:   Depth of the joint self-attn stack.
        num_heads:          Joint self-attn heads.
        state_dim:          Proprioception dim; 0 = no state conditioning.
        hidden_dim:         Action MLP intermediate width.
        gripper_loss_ratio: Weight for BCE gripper loss (Seer default 0.01).
        patch_size:         Pixel patch side for SeerViTDecoder (default 16).
        image_size:         Input image side (default 224).
        decoder_dim:        ViT decoder hidden size (default = d_model).
        num_decoder_heads:  ViT decoder attn heads (default 16).
        num_decoder_blocks: ViT decoder depth (Seer uses 2).
        future_weight:      Reconstruction loss weight.
    """
    def __init__(
        self,
        d_model: int,
        num_views: int = 2,
        num_obs_tokens: int = 9,
        action_dim: int = 7,
        action_pred_steps: int = 3,
        num_joint_layers: int = 2,
        num_heads: int = 16,
        state_dim: int = 0,
        hidden_dim: int | None = None,
        gripper_loss_ratio: float = 0.01,
        patch_size: int = 16,
        image_size: int = 224,
        decoder_dim: int | None = None,
        num_decoder_heads: int = 16,
        num_decoder_blocks: int = 2,
        future_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = d_model // 2

        self.N_f   = num_views * num_obs_tokens   # total future tokens
        self.N_act = action_pred_steps
        self.action_dim = action_dim
        self._gripper_loss_ratio = gripper_loss_ratio

        # Learnable query tokens (moved from FutureCrossAttnBranch and SeerMLPActionHead)
        self.future_query_tokens = nn.Parameter(torch.zeros(1, self.N_f, d_model))
        self.action_pred_tokens  = nn.Parameter(torch.zeros(1, action_pred_steps, d_model))
        nn.init.xavier_uniform_(self.future_query_tokens.view(1, -1, d_model))
        nn.init.normal_(self.action_pred_tokens, std=0.02)

        # Optional state encoder: biases action_pred_tokens with current-frame state.
        self.state_encoder = (
            nn.Linear(state_dim, d_model) if state_dim > 0 else None
        )

        # Joint self-attention layers with Seer block mask.
        self.joint_layers = nn.ModuleList([
            _JointSelfAttnBlock(d_model, num_heads)
            for _ in range(num_joint_layers)
        ])

        # Per-view Seer MAE decoders (unchanged from FutureCrossAttnBranch).
        self.vit_decoders = nn.ModuleList([
            SeerViTDecoder(
                d_model=d_model,
                num_obs_tokens=num_obs_tokens,
                patch_size=patch_size,
                image_size=image_size,
                decoder_dim=decoder_dim,
                num_heads=num_decoder_heads,
                num_blocks=num_decoder_blocks,
                future_weight=future_weight,
            )
            for _ in range(num_views)
        ])

        # Action MLP decoder (unchanged from SeerMLPActionHead).
        mlp_out = hidden_dim // 2
        self.action_decoder  = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, mlp_out), nn.ReLU(),
        )
        self.arm_decoder     = nn.Sequential(nn.Linear(mlp_out, action_dim - 1), nn.Tanh())
        self.gripper_decoder = nn.Linear(mlp_out, 1)   # raw logit for BCEWithLogits

        logger.info(
            "[SeerJointDecoder] N_f=%d  N_act=%d  num_joint_layers=%d  "
            "num_obs_tokens=%d  patch_size=%d  state_encoder=%s",
            self.N_f, action_pred_steps, num_joint_layers,
            num_obs_tokens, patch_size,
            "enabled" if self.state_encoder is not None else "DISABLED",
        )
    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _build_block_mask(self, L_q: int, device: torch.device) -> torch.Tensor:
        """Build Seer-style additive block mask (0=allowed, -inf=blocked).

        Visibility rules:
          qwen(L_q)    → qwen only   (VLM representation shielded from prediction tokens)
          future(N_f)  → qwen+future (obs tokens see context but not action)
          action(T_act)→ all         (action tokens see full context incl. future)
        """
        N = L_q + self.N_f + self.N_act
        mask = torch.full((N, N), float("-inf"), device=device)
        # qwen → qwen
        mask[:L_q, :L_q] = 0.0
        # future → qwen + future
        mask[L_q : L_q + self.N_f, : L_q + self.N_f] = 0.0
        # action → all
        mask[L_q + self.N_f :, :] = 0.0
        return mask    # (N, N)

    def _build_key_padding_mask(
        self,
        L_q: int,
        B: int,
        qwen_pad_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Extend Qwen padding mask to cover the full joint sequence.

        ``qwen_pad_mask`` is (B, L_q) with 1=valid, 0=pad (from qwen attention_mask).
        nn.MultiheadAttention's ``key_padding_mask`` convention: True=IGNORE.
        """
        if qwen_pad_mask is None:
            return None
        qwen_kpm = ~qwen_pad_mask.bool()                        # True=pad → ignore
        extra    = torch.zeros(B, self.N_f + self.N_act, dtype=torch.bool, device=device)
        return torch.cat([qwen_kpm, extra], dim=1)              # (B, L_q+N_f+N_act)

    def _run_joint(
        self,
        qwen_hidden: torch.Tensor,
        state: torch.Tensor | None,
        qwen_pad_mask: torch.Tensor | None,
    ):
        """Run the joint self-attention and return (future_tokens, action_feats)."""
        B, L_q, D = qwen_hidden.shape
        device, dtype = qwen_hidden.device, qwen_hidden.dtype

        # Expand learnable query tokens to batch size
        fq = self.future_query_tokens.expand(B, -1, -1).to(dtype)  # (B, N_f, D)
        aq = self.action_pred_tokens.expand(B, -1, -1).to(dtype)   # (B, N_act, D)

        # Optional: bias action tokens with current-frame state
        if state is not None and self.state_encoder is not None:
            s = state.float()
            if s.ndim == 3:
                s = s[:, -1, :]           # current frame (B, state_dim)
            elif s.ndim == 1:
                s = s.unsqueeze(0)
            aq = aq + self.state_encoder(s).to(dtype).unsqueeze(1)

        # Build joint sequence: [qwen | future | action]
        seq = torch.cat([qwen_hidden, fq, aq], dim=1)  # (B, L_q+N_f+N_act, D)

        # Build Seer block mask and optional padding mask
        block_mask = self._build_block_mask(L_q, device)
        kpm        = self._build_key_padding_mask(L_q, B, qwen_pad_mask, device)

        # Run joint self-attention layers
        for layer in self.joint_layers:
            seq = layer(seq, attn_mask=block_mask, key_padding_mask=kpm)

        future_tokens = seq[:, L_q : L_q + self.N_f]      # (B, N_f, D)
        action_feats  = seq[:, L_q + self.N_f :]           # (B, N_act, D)
        return future_tokens, action_feats
    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        qwen_hidden: torch.Tensor,
        qwen_pad_mask: torch.Tensor | None,
        state: torch.Tensor | None,
        gt_actions: torch.Tensor,
        future_rgb_list: List[torch.Tensor] | None,
    ):
        """Training forward: Seer MAE reconstruction loss + SmoothL1+BCE action loss.

        Args:
            qwen_hidden:     (B, L, D) — Qwen last hidden states (+ state tokens appended).
            qwen_pad_mask:   (B, L) 1=valid 0=pad from qwen attention_mask; or None.
            state:           (B, K, state_dim) or None — for action state conditioning.
            gt_actions:      (B, T_act, action_dim) — ground-truth actions.
            future_rgb_list: list[Tensor(B,3,H,W)] per view, or None (pretrain w/o action).

        Returns:
            future_loss, action_loss  — scalar losses.
        """
        future_tokens, action_feats = self._run_joint(qwen_hidden, state, qwen_pad_mask)

        # ── Future reconstruction loss (Seer MAE per view) ─────────────────────
        future_loss = qwen_hidden.new_zeros(())
        if future_rgb_list is not None and len(future_rgb_list) > 0:
            num_views = len(self.vit_decoders)
            obs_per_view = self.N_f // num_views
            for v, dec in enumerate(self.vit_decoders):
                obs_v = future_tokens[:, v * obs_per_view : (v + 1) * obs_per_view]
                future_loss = future_loss + dec.compute_loss(obs_v, future_rgb_list[v])
            future_loss = future_loss / num_views
        else:
            # ZeRO-3 dummy: touch decoder params
            for dec in self.vit_decoders:
                future_loss = future_loss + (dec.decoder_pred.weight * 0.0).sum()

        # ── Action loss (SmoothL1 arm + 0.01×BCE gripper) ──────────────────────
        feat          = self.action_decoder(action_feats)     # (B, T_act, mlp_out)
        arm_pred      = self.arm_decoder(feat)                # (B, T_act, action_dim-1) Tanh
        gripper_logit = self.gripper_decoder(feat)            # (B, T_act, 1)

        gt = gt_actions.to(arm_pred.dtype)
        gt_arm     = gt[..., :-1]
        gt_gripper = (gt[..., -1:] > 0).float()

        arm_loss     = F.smooth_l1_loss(arm_pred, gt_arm)
        gripper_loss = F.binary_cross_entropy_with_logits(gripper_logit, gt_gripper)
        action_loss  = arm_loss + self._gripper_loss_ratio * gripper_loss

        return future_loss, action_loss

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self,
        qwen_hidden: torch.Tensor,
        qwen_pad_mask: torch.Tensor | None,
        state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Inference: return predicted actions (B, T_act, action_dim).

        Gripper output is binarised to {-1, +1} from logit sign to match
        eval_calvin.py:700 threshold-at-0 protocol.
        """
        _, action_feats = self._run_joint(qwen_hidden, state, qwen_pad_mask)
        feat          = self.action_decoder(action_feats)
        arm           = self.arm_decoder(feat)                        # Tanh [-1,1]
        gripper_logit = self.gripper_decoder(feat)
        gripper       = torch.where(
            gripper_logit > 0,
            torch.ones_like(gripper_logit),    # open  → +1
            -torch.ones_like(gripper_logit),   # close → -1
        )
        return torch.cat([arm, gripper], dim=-1)   # (B, T_act, action_dim)

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype




