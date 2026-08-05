"""FutureCrossAttnBranch: future prediction via cross-attention + DiT diffusion decoder
in projected VLM feature space (x_channel = z_channel, not pixel space).

Architecture:

  learnable future_query_tokens  (1, N_f, D)   [N_f = num_views × num_obs_tokens]
              │
  L-layer CrossAttention          Q = future_query_tokens
                                  KV = cat([qwen_hidden, hist_tokens, state_token])
              │
  future_tokens  (B, N_f, D)
      │
      ├── → appended to vl_embs for GR00T cross-attention   [action conditioning]
      │
      └── FutureDiffusionDecoder (per view, VLM feature-space DiT)  [future supervision]
            ln_pre: LayerNorm(D)
            condition_proj: Linear(D, z_channel)       [SHARED for condition AND target]
            rearrange → (B, z_channel, grid_h, grid_w) [2D spatial feature map]
            ReconDenoiser (DiT DDPM, cosine schedule)
              denoises (B, z_channel, grid_h, grid_h) in projected VLM feature space
            DDPM loss vs GT future_vlm_feat projected through same condition_proj

Condition and target share condition_proj → DiT operates in one unified semantic space.
At convergence future_tokens ≈ vl_feat, so condition ≈ target (ideal for diffusion).

Constraint: num_obs_tokens must be a perfect square (e.g. 49 = 7×7).
"""
from __future__ import annotations

import math
import logging
from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from starVLA.model.modules.uamvla.components.denoiser.scheduler import ReconDenoiser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-view diffusion decoder (ReconVLA DiT, pixel space)
# ---------------------------------------------------------------------------

class FutureDiffusionDecoder(nn.Module):
    """Decode per-view future query tokens into projected VLM features via DiT DDPM.

    Condition and target share condition_proj so the DiT operates in one unified
    z_channel-dimensional projected semantic feature space (not pixel space).

    Pipeline:
      obs_tokens (B, N, D)           [predicted future tokens from cross-attn]
        → ln_pre: LayerNorm(D, no learnable params)
        → condition_proj: Linear(D, z_channel)   ← SHARED with target projection
        → rearrange to (B, z_channel, grid_h, grid_w)   [2D spatial cond map]
        → ReconDenoiser: DiT denoises (B, z_channel, grid_h, grid_h)

      GT target: Qwen ViT tokens (B, N, D)
        → same ln_pre + condition_proj
        → rearrange to (B, z_channel, grid_h, grid_h)   [2D spatial target map]

    Args:
        d_model:        LLM hidden dimension.
        num_obs_tokens: Tokens for this view (must be a perfect square).
        z_channel:      Projection dim shared by condition and target (= DiT x_channel).
        dit_embed_dim:  DiT transformer hidden size.
        dit_depth:      Number of DiT blocks.
        future_weight:  Loss weight applied to the DDPM loss.
        dit_timesteps:  Inference timestep respacing (e.g. ``"100"``).
    """

    def __init__(
        self,
        d_model: int,
        num_obs_tokens: int,
        z_channel: int,
        dit_embed_dim: int,
        dit_depth: int,
        future_weight: float = 0.5,
        dit_timesteps: str = "1000",
    ) -> None:
        super().__init__()

        grid_h = int(math.sqrt(num_obs_tokens))
        if grid_h * grid_h != num_obs_tokens:
            raise ValueError(
                f"FutureDiffusionDecoder requires num_obs_tokens to be a perfect square "
                f"(needed for 2D spatial condition map), got {num_obs_tokens}."
            )
        self.grid_h = grid_h
        # target_res = grid_h: each pixel of the 7×7 target is conditioned by one obs token.
        # (Identical spatial resolution → no interpolation, clean per-token alignment.)
        self.target_res = grid_h
        self.future_weight = future_weight

        # Conditioning: LN (no learnable params, same as ReconHead) → linear project.
        self.ln_pre = nn.LayerNorm(d_model, elementwise_affine=False)
        self.condition_proj = nn.Linear(d_model, z_channel)

        # DiT DDPM denoiser — projected VLM feature space, x_channel=z_channel.
        # Condition and target both live in the same z_channel-dim space (shared proj),
        # so x_channel == z_channel by design.
        self.denoiser = ReconDenoiser(
            x_channel=z_channel,        # feature space, not RGB (was 3)
            z_channel=z_channel,
            embed_dim=dit_embed_dim,
            depth=dit_depth,
            n_patches=num_obs_tokens,   # DiT spatial grid = sqrt(num_obs_tokens)
            timesteps=dit_timesteps,
        )

    # ------------------------------------------------------------------

    def _prepare_condition(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, D) → LayerNorm → Linear → (B, z_channel, grid_h, grid_h)."""
        z = self.condition_proj(self.ln_pre(obs_tokens))
        return rearrange(z, "b (h w) c -> b c h w", h=self.grid_h)

    def compute_loss(
        self, obs_tokens: torch.Tensor, future_vlm_feat: torch.Tensor
    ) -> torch.Tensor:
        """VLM feature-space DDPM training loss for one camera view.

        Condition (from predicted future tokens) and target (from Qwen ViT GT) both
        pass through the shared condition_proj → same z_channel space → no mismatch.

        Args:
            obs_tokens:      (B, num_obs_tokens, D) — predicted future tokens from cross-attn.
            future_vlm_feat: (B, num_obs_tokens, D) — GT VLM tokens from frozen Qwen ViT.
        Returns:
            Scalar loss (future_weight × mean DDPM loss).
        """
        z = self._prepare_condition(obs_tokens)                             # (B, z_channel, 7, 7)
        # Project GT VLM features through the same condition_proj (shared weights).
        target = self.condition_proj(self.ln_pre(future_vlm_feat.to(z.dtype)))  # (B, N, z_channel)
        target = rearrange(target, "b (h w) c -> b c h w", h=self.grid_h)  # (B, z_channel, 7, 7)
        target = target.detach()
        loss = self.denoiser(z=z, target=target).mean()
        return self.future_weight * loss

    @torch.no_grad()
    def sample(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        """Sample a future projected VLM feature map via DDPM reverse diffusion.

        Returns:
            (B, z_channel, grid_h, grid_h) — denoised projected VLM feature map.
        """
        z = self._prepare_condition(obs_tokens)
        return self.denoiser.sample(z)   # (B, z_channel, grid_h, grid_h)


# ---------------------------------------------------------------------------
# Single cross-attention layer (query attends to KV)
# ---------------------------------------------------------------------------

class _CrossAttnLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return q


# ---------------------------------------------------------------------------
# FutureCrossAttnBranch
# ---------------------------------------------------------------------------

class FutureCrossAttnBranch(nn.Module):
    """Seer cross-attention + ReconVLA DiT diffusion decoder for future prediction.

    Args:
        d_model:               LLM hidden dimension (D).
        num_views:             Number of camera views (e.g. 2 for primary+wrist).
        num_obs_tokens:        Future query tokens per view.  Must be a perfect
                               square (e.g. 49 = 7×7 to align with Qwen3-VL 224px).
        num_cross_attn_layers: Depth of the cross-attention stack.
        num_heads:             Attention heads for cross-attention layers.
        state_dim:             Robot state dimension for state-token injection (0 = off).
        future_weight:         Loss weight applied to per-view DDPM losses.
        z_channel:             Projection dimension for DiT condition.
        dit_embed_dim:         DiT transformer hidden size.
        dit_depth:             Number of DiT blocks.
        dit_timesteps:         Inference respacing string (e.g. ``"100"``).
    """

    def __init__(
        self,
        d_model: int,
        num_views: int = 2,
        num_obs_tokens: int = 49,
        num_cross_attn_layers: int = 2,
        num_heads: int = 8,
        state_dim: int = 0,
        future_weight: float = 0.5,
        z_channel: int = 512,
        dit_embed_dim: int = 512,
        dit_depth: int = 3,
        dit_timesteps: str = "1000",
    ) -> None:
        super().__init__()

        self.num_views = num_views
        self.num_obs_tokens = num_obs_tokens    # per view
        self.n_f = num_views * num_obs_tokens   # total future tokens N_f

        # Learnable future query tokens — split evenly across views.
        self.future_query_tokens = nn.Parameter(torch.zeros(1, self.n_f, d_model))
        nn.init.xavier_uniform_(self.future_query_tokens.view(1, -1, d_model))

        # L-layer cross-attention stack.
        self.cross_attn_layers = nn.ModuleList([
            _CrossAttnLayer(d_model, num_heads)
            for _ in range(num_cross_attn_layers)
        ])

        # Optional robot-state token.
        self.state_proj = nn.Linear(state_dim, d_model) if state_dim > 0 else None

        # Per-view DiT diffusion decoders.
        self.diff_decoders = nn.ModuleList([
            FutureDiffusionDecoder(
                d_model=d_model,
                num_obs_tokens=num_obs_tokens,
                z_channel=z_channel,
                dit_embed_dim=dit_embed_dim,
                dit_depth=dit_depth,
                future_weight=future_weight,
                dit_timesteps=dit_timesteps,
            )
            for _ in range(num_views)
        ])

        logger.info(
            "[FutureCrossAttnBranch] N_f=%d  num_views=%d  num_obs_tokens=%d  "
            "feat_grid=%dx%d  z_channel=%d  dit_embed_dim=%d  dit_depth=%d  "
            "(VLM feature-space DDPM, x_channel=z_channel=%d)  "
            "state_proj=%s (state_dim=%d)",
            self.n_f, num_views, num_obs_tokens,
            self.diff_decoders[0].grid_h, self.diff_decoders[0].grid_h,
            z_channel, dit_embed_dim, dit_depth, z_channel,
            "enabled" if self.state_proj is not None else "DISABLED",
            state_dim,
        )

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _build_kv(
        self,
        hidden: torch.Tensor,
        hist_tokens: torch.Tensor | None,
        state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Concatenate Qwen hidden states, optional history tokens, optional state token."""
        parts = [hidden]
        if hist_tokens is not None:
            parts.append(hist_tokens)
        if state is not None and self.state_proj is not None:
            s = state.float()
            if s.ndim == 3:
                s = s.squeeze(1)
            elif s.ndim == 1:
                s = s.unsqueeze(0)
            st = self.state_proj(s).unsqueeze(1)    # (B, 1, D)
            parts.append(st.to(hidden.dtype))
        return torch.cat(parts, dim=1)

    def _cross_attend(self, kv: torch.Tensor) -> torch.Tensor:
        """Run future_query_tokens through all cross-attention layers."""
        B = kv.shape[0]
        q = self.future_query_tokens.expand(B, -1, -1).to(kv.dtype)
        for layer in self.cross_attn_layers:
            q = layer(q, kv)
        return q                                    # (B, N_f, D)

    def _dummy_loss(
        self, future_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Zero-weight dummy loss to keep ZeRO-3 parameter collectives aligned.

        Touches condition_proj, ln_pre, AND denoiser parameters so all ranks
        participate in allgather even when there are no VLM feature supervision targets.
        """
        dummy = future_tokens.new_zeros(())
        for v, dec in enumerate(self.diff_decoders):
            obs_v = future_tokens[:1, v * self.num_obs_tokens:(v + 1) * self.num_obs_tokens]
            z = dec._prepare_condition(obs_v)            # touches condition_proj + ln_pre
            dummy_target = z.detach().clone()
            denoiser_out = dec.denoiser(z=z, target=dummy_target)  # touches all DiT params
            dummy = dummy + denoiser_out.mean() * 0.0
        return dummy

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
        hist_tokens: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference path: return future_tokens for action conditioning only."""
        kv = self._build_kv(hidden, hist_tokens, state)
        return self._cross_attend(kv)

    def compute_loss(
        self,
        hidden: torch.Tensor,
        batch: dict,
        hist_tokens: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        future_vlm_features: List[torch.Tensor] | None = None,
    ):
        """Training path: compute future_tokens and VLM feature-space DiT DDPM loss.

        Args:
            hidden:               Qwen last hidden states (B, seq_len, D).
            batch:                Collated batch dict (used for dummy-loss fallback).
            hist_tokens:          Optional (B, T_hist, D) from HistoryVisionEncoder.
            state:                Optional robot state (B, state_dim) or (B, 1, state_dim).
            future_vlm_features:  list[Tensor(B, num_obs_tokens, D)] — GT VLM tokens from
                                  frozen Qwen ViT, one entry per view.  If None or empty,
                                  returns a zero-weight dummy loss.

        Returns:
            future_tokens: (B, N_f, D) — appended to vl_embs for GR00T.
            future_loss:   Scalar DDPM loss (0.0 if no VLM feature targets present).
        """
        kv = self._build_kv(hidden, hist_tokens, state)
        future_tokens = self._cross_attend(kv)          # (B, N_f, D)

        if future_vlm_features is None or len(future_vlm_features) == 0:
            return future_tokens, self._dummy_loss(future_tokens)

        total_loss = future_tokens.new_zeros(())
        for v, dec in enumerate(self.diff_decoders):
            obs_v = future_tokens[:, v * self.num_obs_tokens:(v + 1) * self.num_obs_tokens]
            feat_v = future_vlm_features[v].to(hidden.device, hidden.dtype)
            total_loss = total_loss + dec.compute_loss(obs_v, feat_v)

        return future_tokens, total_loss / self.num_views

    @torch.no_grad()
    def sample_future(
        self,
        hidden: torch.Tensor,
        hist_tokens: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> List[torch.Tensor]:
        """Sample future RGB frames (one per view) via DDPM reverse diffusion.

        Returns:
            List of (B, 3, target_res, target_res) tensors in [0, 1], one per view.
        """
        future_tokens = self.forward(hidden, hist_tokens, state)
        results = []
        for v, dec in enumerate(self.diff_decoders):
            obs_v = future_tokens[:, v * self.num_obs_tokens:(v + 1) * self.num_obs_tokens]
            results.append(dec.sample(obs_v))
        return results
