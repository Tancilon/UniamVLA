"""SeerJointDecoder: unified self-attention + Seer block mask.

Replaces the two independent cross-attn modules (FutureCrossAttnBranch and
SeerMLPActionHead) with a single joint self-attention decoder that processes
future and action tokens in the same sequence as Qwen hidden:

    [qwen_hidden (L+K) | future_query_tokens (N_f) | action_pred_tokens (T_act)]
              ↓  masked self-attention (Seer block mask)
    [qwen_out          | future_tokens (N_f)        | action_feats (T_act)     ]

Seer block mask — aligned with seer_model.py generate_attention_mask:
  qwen(L+K)    → qwen only          (B tokens never keys for A tokens)
  future_t     → qwen only          (per-timestep independent; no cross-B key access)
  action_t     → qwen + future_t    (same-timestep obs_tokens only; Seer exception rule)

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
        history_frames: int = 1,    # K total query groups (Seer: sequence_length)
        atten_goal: int = 0,        # last atten_goal groups not supervised (Seer default 3)
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = d_model // 2

        self.N_f    = num_views * num_obs_tokens   # tokens per query group (future)
        self.N_act  = action_pred_steps             # tokens per query group (action)
        self.K      = history_frames                # total query groups
        self.K_sup  = max(1, history_frames - atten_goal)  # supervised groups
        self.action_dim = action_dim
        self._gripper_loss_ratio = gripper_loss_ratio

        # Query tokens: shared base + per-timestep positional embedding.
        # Aligns with Seer (seer_model.py:190): obs_tokens (1,1,N,D) shared across all
        # S timesteps; position_embedding (1,S,1,D) added to distinguish timesteps.
        # Parameters:
        #   future_query_base  (1, N_f,   D) — shared semantic base, xavier init
        #   action_pred_base   (1, N_act, D) — shared semantic base, normal init
        #   temporal_pos_emb   (K, 1,     D) — one vector per timestep, broadcast to
        #                                       both future and action tokens (Seer style)
        self.future_query_base  = nn.Parameter(torch.zeros(1, self.N_f,   d_model))
        self.action_pred_base   = nn.Parameter(torch.zeros(1, self.N_act, d_model))
        self.temporal_pos_emb   = nn.Parameter(torch.zeros(self.K, 1,     d_model))
        nn.init.xavier_uniform_(self.future_query_base.squeeze(0))
        nn.init.normal_(self.action_pred_base, std=0.02)
        # temporal_pos_emb: zeros init (Seer initialises position embeddings at zero)

        self.state_encoder = None  # removed: state is now injected into Qwen input_embeds
        # (state_dim kept in signature for YAML backward compat; value is intentionally unused)

        self._num_heads = num_heads  # stored for (B*H, N, N) mask expansion in _run_joint

        self.joint_layers = nn.ModuleList([
            _JointSelfAttnBlock(d_model, num_heads)
            for _ in range(num_joint_layers)
        ])

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

        mlp_out = hidden_dim // 2
        self.action_decoder  = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, mlp_out), nn.ReLU(),
        )
        self.arm_decoder     = nn.Sequential(nn.Linear(mlp_out, action_dim - 1), nn.Tanh())
        self.gripper_decoder = nn.Linear(mlp_out, 1)

        logger.info(
            "[SeerJointDecoder] K=%d  K_sup=%d  N_f=%d  N_act=%d  "
            "num_joint_layers=%d  num_obs_tokens=%d  patch_size=%d  state_encoder=%s",
            self.K, self.K_sup, self.N_f, self.N_act,
            num_joint_layers, num_obs_tokens, patch_size,
            "enabled" if self.state_encoder is not None else "DISABLED",
        )
    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _build_block_mask(
        self,
        L_q: int,
        device: torch.device,
        B: int = 1,
        qwen_frame_ends: "list[list[int]] | None" = None,
        state_token_positions: "list[list[int]] | None" = None,
    ) -> torch.Tensor:
        """Build per-sample Seer temporal causal block mask.

        Sequence layout:
          [qwen(L_q) | f_0(N_f)|a_0(N_act) | f_1|a_1 | ... | f_{K-1}|a_{K-1}]

        Visibility rules:
          qwen      -> qwen only
          future_t  -> qwen[0..end_t[b])  (per-sample causal boundary)
                    -> ALSO: state token of frame t+atten_goal (atten_goal_state)
          action_t  -> qwen[0..end_t[b]) + future_t

        qwen_frame_ends: list[B][K] — per-sample frame end positions.
        state_token_positions: list[B][K] — position of each injected state token.
          When provided, future_t may additionally attend the state token of the
          goal timestep t+atten_goal (Seer pretrain --atten_goal_state flag).
          atten_goal is derived from self.K - self.K_sup.

        Returns (B, N, N) — caller expands to (B*num_heads, N, N) for MHA.
        """
        K   = self.K
        N_f = self.N_f
        N_a = self.N_act
        N   = L_q + K * (N_f + N_a)
        _atten_goal = K - self.K_sup  # = atten_goal stored implicitly via K_sup

        masks = []
        for b in range(B):
            mask = torch.full((N, N), float("-inf"), device=device)
            mask[:L_q, :L_q] = 0.0

            for t in range(K):
                f_s = L_q + t * (N_f + N_a)
                f_e = f_s + N_f
                a_s = f_e
                a_e = a_s + N_a

                end_t = (qwen_frame_ends[b][t] + 1) if qwen_frame_ends is not None else L_q

                mask[f_s:f_e, :end_t] = 0.0          # future_t -> qwen[0..end_t)
                mask[a_s:a_e, :end_t] = 0.0          # action_t -> qwen[0..end_t)
                mask[a_s:a_e, f_s:f_e] = 0.0         # action_t -> future_t

                # atten_goal_state (Seer pretrain --atten_goal_state):
                # future_t may additionally attend the state token of the goal frame
                # t + atten_goal.  This gives future prediction a goal-conditioning
                # signal: "predict the image given you know where the robot will be
                # atten_goal steps later."  Only valid when state tokens are injected.
                if state_token_positions is not None and _atten_goal > 0:
                    goal_t = t + _atten_goal
                    if goal_t < K:
                        goal_state_pos = state_token_positions[b][goal_t]
                        mask[f_s:f_e, goal_state_pos] = 0.0

            masks.append(mask)

        return torch.stack(masks)  # (B, N, N)

    def _build_key_padding_mask(
        self,
        L_q: int,
        B: int,
        qwen_pad_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if qwen_pad_mask is None:
            return None
        qwen_kpm = ~qwen_pad_mask.bool()
        extra    = torch.zeros(B, self.K * (self.N_f + self.N_act), dtype=torch.bool, device=device)
        return torch.cat([qwen_kpm, extra], dim=1)

    def _run_joint(
        self,
        qwen_hidden: torch.Tensor,
        qwen_pad_mask: torch.Tensor | None,
        qwen_frame_ends: "list[list[int]] | None" = None,
        state_token_positions: "list[list[int]] | None" = None,
    ):
        """Run joint self-attn → return (future_all, action_all) each (B, K*N, D)."""
        B, L_q, D = qwen_hidden.shape
        device, dtype = qwen_hidden.device, qwen_hidden.dtype

        # Expand K query groups: shared_base + temporal_pos_emb → (K, N, D) → (B, K*N, D)
        # Mirrors Seer: token_t = obs_tokens_base + position_embedding[t]
        fq_knd = (self.future_query_base + self.temporal_pos_emb).to(dtype)  # (K, N_f, D)
        aq_knd = (self.action_pred_base  + self.temporal_pos_emb).to(dtype)  # (K, N_act, D)
        fq = fq_knd.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, self.K * self.N_f,   D)
        aq = aq_knd.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, self.K * self.N_act, D)

        # Interleave into [qwen | f_0|a_0 | f_1|a_1 | ...]
        parts = [qwen_hidden]
        for t in range(self.K):
            parts.append(fq[:, t * self.N_f : (t + 1) * self.N_f])
            parts.append(aq[:, t * self.N_act : (t + 1) * self.N_act])
        seq = torch.cat(parts, dim=1)  # (B, L_q + K*(N_f+N_act), D)

        block_mask = self._build_block_mask(L_q, device, B, qwen_frame_ends,
                                             state_token_positions)
        # Expand (B, N, N) → (B*num_heads, N, N) as required by nn.MultiheadAttention.
        H = self._num_heads
        N = block_mask.shape[-1]
        block_mask = block_mask.unsqueeze(1).expand(-1, H, -1, -1).reshape(B * H, N, N)
        kpm        = self._build_key_padding_mask(L_q, B, qwen_pad_mask, device)

        for layer in self.joint_layers:
            seq = layer(seq, attn_mask=block_mask, key_padding_mask=kpm)

        # Slice out future and action tokens for all K groups
        future_out = torch.zeros(B, self.K, self.N_f, D, device=device, dtype=dtype)
        action_out = torch.zeros(B, self.K, self.N_act, D, device=device, dtype=dtype)
        for t in range(self.K):
            f_s = L_q + t * (self.N_f + self.N_act)
            future_out[:, t] = seq[:, f_s : f_s + self.N_f]
            action_out[:, t] = seq[:, f_s + self.N_f : f_s + self.N_f + self.N_act]

        return future_out, action_out   # (B, K, N_f, D), (B, K, N_act, D)
    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        qwen_hidden: torch.Tensor,
        qwen_pad_mask: torch.Tensor | None,
        gt_actions: torch.Tensor,
        future_rgb_list: List[torch.Tensor] | None,
        action_history: "np.ndarray | None" = None,
        future_rgb_history: "List | None" = None,
        qwen_frame_ends: "list[list[int]] | None" = None,
        state_token_positions: "list[list[int]] | None" = None,
    ):
        """Dense per-timestep training (Seer-aligned).

        Dense mode (action_history / future_rgb_history provided):
          - Computes loss for K_sup supervised timesteps (t=0..K_sup-1).
          - action_history: ndarray (B, K, T_act, action_dim)
          - future_rgb_history: list[K] of list[num_views](B,3,H,W) tensors

        Single-step mode (backward compat, only gt_actions / future_rgb_list):
          - Uses query group t=K-1 (last = current timestep).

        qwen_frame_ends: passed to _build_block_mask to prevent future-frame leakage.

        Returns:
            future_loss, action_loss — scalar losses.
        """
        future_out, action_out = self._run_joint(qwen_hidden, qwen_pad_mask,
                                                  qwen_frame_ends, state_token_positions)
        # future_out: (B, K, N_f, D),  action_out: (B, K, N_act, D)

        num_views    = len(self.vit_decoders)
        obs_per_view = self.N_f // num_views

        dense = (action_history is not None) and (future_rgb_history is not None)

        # ── Future reconstruction loss ──────────────────────────────────────────
        future_loss = qwen_hidden.new_zeros(())
        if dense:
            for t in range(self.K_sup):
                for v, dec in enumerate(self.vit_decoders):
                    obs_v = future_out[:, t, v * obs_per_view : (v + 1) * obs_per_view]
                    rgb_v = future_rgb_history[t][v]
                    future_loss = future_loss + dec.compute_loss(obs_v, rgb_v)
            future_loss = future_loss / (self.K_sup * num_views)
        elif future_rgb_list is not None and len(future_rgb_list) > 0:
            t = self.K - 1   # current timestep group
            for v, dec in enumerate(self.vit_decoders):
                obs_v = future_out[:, t, v * obs_per_view : (v + 1) * obs_per_view]
                future_loss = future_loss + dec.compute_loss(obs_v, future_rgb_list[v])
            future_loss = future_loss / num_views
        else:
            for dec in self.vit_decoders:
                future_loss = future_loss + (dec.decoder_pred.weight * 0.0).sum()

        # ── Action loss (SmoothL1 arm + 0.01×BCE gripper) ──────────────────────
        action_loss = qwen_hidden.new_zeros(())
        if dense:
            import numpy as _np
            act_hist = torch.as_tensor(
                _np.asarray(action_history), device=qwen_hidden.device, dtype=qwen_hidden.dtype
            )  # (B, K, T_act, action_dim)
            for t in range(self.K_sup):
                feats         = self.action_decoder(action_out[:, t])
                arm_pred      = self.arm_decoder(feats)
                gripper_logit = self.gripper_decoder(feats)
                gt            = act_hist[:, t].to(arm_pred.dtype)
                action_loss   = action_loss + (
                    F.smooth_l1_loss(arm_pred, gt[..., :-1])
                    + self._gripper_loss_ratio * F.binary_cross_entropy_with_logits(
                        gripper_logit, (gt[..., -1:] > 0).float()
                    )
                )
            action_loss = action_loss / self.K_sup
        else:
            t             = self.K - 1
            feats         = self.action_decoder(action_out[:, t])
            arm_pred      = self.arm_decoder(feats)
            gripper_logit = self.gripper_decoder(feats)
            gt            = gt_actions.to(arm_pred.dtype)
            action_loss   = (
                F.smooth_l1_loss(arm_pred, gt[..., :-1])
                + self._gripper_loss_ratio * F.binary_cross_entropy_with_logits(
                    gripper_logit, (gt[..., -1:] > 0).float()
                )
            )

        return future_loss, action_loss

    @torch.no_grad()
    def predict_action(
        self,
        qwen_hidden: torch.Tensor,
        qwen_pad_mask: torch.Tensor | None,
        qwen_frame_ends: "list[list[int]] | None" = None,
        state_token_positions: "list[list[int]] | None" = None,
    ) -> torch.Tensor:
        """Inference: use LAST query group (t=K-1 = current timestep)."""
        _, action_out = self._run_joint(qwen_hidden, qwen_pad_mask,
                                        qwen_frame_ends, state_token_positions)
        # Use last group (current timestep)
        feats         = self.action_decoder(action_out[:, self.K - 1])
        arm           = self.arm_decoder(feats)
        gripper_logit = self.gripper_decoder(feats)
        gripper       = torch.where(
            gripper_logit > 0,
            torch.ones_like(gripper_logit),
            -torch.ones_like(gripper_logit),
        )
        return torch.cat([arm, gripper], dim=-1)

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype




