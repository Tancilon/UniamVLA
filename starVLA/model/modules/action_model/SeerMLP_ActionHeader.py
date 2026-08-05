"""SeerMLPActionHead: Seer-style lightweight MLP action decoder.

Seer-faithful action prediction for the UamGR00T_DT framework.
Action prediction tokens cross-attend to [future_tokens | qwen_hidden],
then an MLP decodes to arm + gripper actions (direct MSE regression).

This is the Qwen-framework analog of Seer's design where obs_tokens and
action_pred_token are peers in the same transformer sequence:
  - Seer:   [obs_tokens | action_pred_token] in GPT2 self-attention
  - Here:   action_pred_tokens → cross-attn(KV=[future_tokens | qwen_hidden]) → MLP

Architecture:
  action_pred_tokens (1, T_act, D)  ← learnable, analog of Seer's action_pred_token
      + optional state_token (1, 1, D)
        ↓  cross-attention
  KV = [future_tokens (B, N_f, D) | qwen_hidden (B, S, D)]
        ↓
  LayerNorm + FF residual
        ↓
  MLP: Linear(D, H) → ReLU → Linear(H, H//2) → ReLU
        ↓
  arm_decoder:     Linear(H//2, action_dim-1) + Tanh      [6-DoF arm]
  gripper_decoder: Linear(H//2, 1) + Sigmoid               [1-D gripper]
        ↓ concat
  pred_actions (B, T_act, action_dim)

Loss: MSE(pred_actions, gt_actions)  — direct regression, no diffusion steps.

Config keys (under framework.action_model):
    action_dim         int  — total action dimensionality (arm + gripper = 7)
    action_horizon     int  — T_act, number of steps in the predicted chunk
    hidden_dim         int  — MLP intermediate width (default d_model // 2)
    state_dim          int  — proprioception dim; 0 = disable state injection
    num_heads          int  — cross-attention heads (default 8)
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SeerMLPActionHead(nn.Module):
    """Seer-aligned lightweight action head for UamGR00T_DT.

    Args:
        d_model:          LLM hidden dimension (D).
        action_dim:       Total action dimensionality (arm + gripper).
        action_pred_steps: Number of action steps to predict (chunk size, T_act).
        hidden_dim:       MLP hidden width.  Default: d_model // 2.
        state_dim:        Proprioception dim.  0 → no state injection.
        num_heads:        Cross-attention heads (must divide d_model).
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        action_pred_steps: int,
        hidden_dim: int | None = None,
        state_dim: int = 0,
        num_heads: int = 8,
        gripper_loss_ratio: float = 0.01,   # Seer train_utils.py:165 loss_gripper_action_ratio
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = d_model // 2
        self.action_dim = action_dim
        self.action_pred_steps = action_pred_steps
        self._gripper_loss_ratio = gripper_loss_ratio

        # Learnable action prediction tokens — analog of Seer's action_pred_token.
        self.action_pred_tokens = nn.Parameter(
            torch.zeros(1, action_pred_steps, d_model)
        )
        nn.init.normal_(self.action_pred_tokens, std=0.02)

        # Optional robot-state encoder → biases action_pred_tokens.
        self.state_encoder = (
            nn.Linear(state_dim, d_model) if state_dim > 0 else None
        )

        # Cross-attention: action_pred_tokens attend to [future_tokens | qwen_hidden].
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )

        # Feed-forward residual after cross-attention.
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Seer-style MLP action decoder.
        mlp_out = hidden_dim // 2
        self.action_decoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mlp_out),
            nn.ReLU(),
        )
        # Split head: arm (action_dim-1 DoF, Tanh) + gripper logit (1D, no Sigmoid).
        # Sigmoid is applied at inference; training uses BCEWithLogitsLoss.
        self.arm_decoder     = nn.Sequential(nn.Linear(mlp_out, action_dim - 1), nn.Tanh())
        self.gripper_decoder = nn.Linear(mlp_out, 1)   # raw logit; Sigmoid in predict_action

        logger.info(
            "[SeerMLPActionHead] action_dim=%d  T_act=%d  hidden_dim=%d  "
            "state_dim=%d  num_heads=%d",
            action_dim, action_pred_steps, hidden_dim, state_dim, num_heads,
        )

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _get_features(
        self,
        qwen_hidden: torch.Tensor,
        future_tokens: torch.Tensor,
        state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Cross-attend then MLP → (B, T_act, mlp_out).  Shared by train and inference."""
        B = qwen_hidden.shape[0]

        # Q: action_pred_tokens, optionally shifted by current state.
        q = self.action_pred_tokens.expand(B, -1, -1)       # (B, T_act, D)
        if state is not None and self.state_encoder is not None:
            s = state.float()
            if s.ndim == 3:
                s = s[:, -1, :]                             # (B, state_dim) — current frame
            elif s.ndim == 1:
                s = s.unsqueeze(0)
            s_tok = self.state_encoder(s).unsqueeze(1)      # (B, 1, D)
            q = q + s_tok.to(q.dtype)

        # KV: future_tokens concatenated with Qwen hidden states.
        kv = torch.cat([future_tokens, qwen_hidden], dim=1) # (B, N_f+S, D)

        # Cross-attention with pre-norm.
        q_norm  = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))                    # (B, T_act, D)
        return self.action_decoder(q)                       # (B, T_act, mlp_out)

    def _decode(
        self,
        qwen_hidden: torch.Tensor,
        future_tokens: torch.Tensor,
        state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Inference decode → (B, T_act, action_dim).

        Gripper output is binarised to {-1, +1} by logit sign, matching
        eval_calvin.py:700 which thresholds at 0 before env.step().
        Returning sigmoid [0,1] would always be > 0, making gripper always open.
        """
        feat         = self._get_features(qwen_hidden, future_tokens, state)
        arm          = self.arm_decoder(feat)              # (B, T_act, action_dim-1) Tanh [-1,1]
        gripper_logit= self.gripper_decoder(feat)          # (B, T_act, 1) raw logit
        gripper      = torch.where(
            gripper_logit > 0,
            torch.ones_like(gripper_logit),                # open  → +1
            -torch.ones_like(gripper_logit),               # close → -1
        )
        return torch.cat([arm, gripper], dim=-1)           # (B, T_act, action_dim)

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def forward(
        self,
        qwen_hidden: torch.Tensor,
        future_tokens: torch.Tensor,
        gt_actions: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training forward: SmoothL1(arm) + BCEWithLogits(gripper).

        Seer loss alignment (train_utils.py:135):
          arm:     SmoothL1Loss  — robust to outliers
          gripper: BCEWithLogitsLoss on binary 0/1 label
                   GT gripper is thresholded at 0 (>0 → open=1, ≤0 → close=0).

        Args:
            qwen_hidden:   (B, S, D)
            future_tokens: (B, N_f, D)
            gt_actions:    (B, T_act, action_dim)  last dim = [...arm..., gripper]
            state:         (B, state_dim) | (B, 1, state_dim) | (B, K, state_dim) | None
        Returns:
            Scalar loss.
        """
        feat         = self._get_features(qwen_hidden, future_tokens, state)
        arm_pred     = self.arm_decoder(feat)               # (B, T_act, action_dim-1)  Tanh
        gripper_logit= self.gripper_decoder(feat)           # (B, T_act, 1)  raw logit

        gt = gt_actions.to(arm_pred.dtype)
        gt_arm     = gt[..., :-1]                           # (B, T_act, action_dim-1)
        gt_gripper = (gt[..., -1:] > 0).float()            # 0/1 binary label

        arm_loss     = F.smooth_l1_loss(arm_pred, gt_arm)
        gripper_loss = F.binary_cross_entropy_with_logits(gripper_logit, gt_gripper)
        # Seer train_utils.py:165: loss_gripper_action_ratio=0.01 keeps gripper from
        # dominating arm gradients (BCE and SmoothL1 have very different scales).
        return arm_loss + self._gripper_loss_ratio * gripper_loss

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self,
        qwen_hidden: torch.Tensor,
        future_tokens: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference: return predicted actions (B, T_act, action_dim)."""
        return self._decode(qwen_hidden, future_tokens, state)

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


def get_action_model(config=None) -> SeerMLPActionHead:
    """Factory: build SeerMLPActionHead from global framework config."""
    cfg = config.framework.action_model
    d_llm = config.framework.qwenvl.get("vl_hidden_dim", 2048)
    return SeerMLPActionHead(
        d_model=d_llm,
        action_dim=int(cfg.action_dim),
        action_pred_steps=int(cfg.action_horizon),
        hidden_dim=int(cfg.get("hidden_dim", d_llm // 2)),
        state_dim=int(cfg.get("state_dim", 0) or 0),
        num_heads=int(cfg.get("num_heads", 8)),
    )
