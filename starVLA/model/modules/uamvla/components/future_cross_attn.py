"""FutureCrossAttnBranch: Seer-style future prediction via cross-attention + DiT diffusion decoder.

Architecture (aligned with ReconVLA + Seer):

  learnable future_query_tokens  (1, N_f, D)   [N_f = num_views × num_obs_tokens]
              │
  L-layer CrossAttention          Q = future_query_tokens
                                  KV = cat([qwen_hidden, hist_tokens, state_token])
              │
  future_tokens  (B, N_f, D)
      │
      ├── → appended to vl_embs for GR00T cross-attention   [action conditioning]
      │
      └── FutureDiffusionDecoder (per view, ReconVLA-style DiT)  [future supervision]
            ln_pre: LayerNorm(D)
            condition_proj: Linear(D, z_channel)
            rearrange → (B, z_channel, grid_h, grid_w)       [2D spatial cond map]
            ReconDenoiser (DiT DDPM, cosine schedule)
              denoises (B, 3, target_res, target_res) in pixel space
            MSE-like DDPM loss vs future_rgb resized to target_res

Constraint: num_obs_tokens must be a perfect square (e.g. 49 = 7×7).
target_res = sqrt(num_obs_tokens) = 7 (coarse future state representation).
"""
from __future__ import annotations

import math
import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from starVLA.model.modules.uamvla.components.denoiser.scheduler import ReconDenoiser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-view diffusion decoder (ReconVLA DiT, pixel space)
# ---------------------------------------------------------------------------

class FutureDiffusionDecoder(nn.Module):
    """Decode per-view future query tokens into a future RGB frame via DiT DDPM.

    Conditioning pipeline (mirrors ReconHead._spatial_condition):
      obs_tokens (B, N, D)
        → ln_pre: LayerNorm(D, no learnable params)
        → condition_proj: Linear(D, z_channel)
        → rearrange to (B, z_channel, grid_h, grid_w)   [2D spatial cond]
        → ReconDenoiser: DiT denoises (B, 3, target_res, target_res)

    Args:
        d_model:        LLM hidden dimension.
        num_obs_tokens: Tokens for this view (must be a perfect square).
        z_channel:      Projection dimension for the DiT condition.
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

        # DiT DDPM denoiser — pixel space, x_channel=3 (RGB).
        self.denoiser = ReconDenoiser(
            x_channel=3,
            z_channel=z_channel,
            embed_dim=dit_embed_dim,
            depth=dit_depth,
            n_patches=num_obs_tokens,   # DiT input_size = sqrt(49) = 7
            timesteps=dit_timesteps,
        )

    # ------------------------------------------------------------------

    def _prepare_condition(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, D) → LayerNorm → Linear → (B, z_channel, grid_h, grid_h)."""
        z = self.condition_proj(self.ln_pre(obs_tokens))
        return rearrange(z, "b (h w) c -> b c h w", h=self.grid_h)

    def compute_loss(
        self, obs_tokens: torch.Tensor, future_rgb: torch.Tensor
    ) -> torch.Tensor:
        """DDPM training loss for one camera view.

        Args:
            obs_tokens: (B, num_obs_tokens, D) — future query tokens for this view.
            future_rgb: (B, 3, H, W) float in [0, 1] — ground-truth future frame.
        Returns:
            Scalar loss (future_weight × mean DDPM loss).
        """
        z = self._prepare_condition(obs_tokens)          # (B, z_channel, 7, 7)
        # Resize to target_res and rescale [0,1] → [-1,1] for DDPM.
        target = F.interpolate(
            future_rgb.float(), (self.target_res, self.target_res), mode="bilinear", align_corners=False
        )
        target = target * 2.0 - 1.0                     # → [-1, 1]
        target = target.to(z.dtype)
        loss = self.denoiser(z=z, target=target).mean()  # ReconDenoiser returns (B,) losses
        return self.future_weight * loss

    @torch.no_grad()
    def sample(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        """Sample a future RGB frame via DDPM reverse diffusion.

        Returns:
            (B, 3, target_res, target_res) float in [0, 1].
        """
        z = self._prepare_condition(obs_tokens)
        sampled = self.denoiser.sample(z)                # (B, 3, 7, 7) in [-1, 1]
        return (sampled / 2.0 + 0.5).clamp(0.0, 1.0)


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
            "target_res=%dx%d  z_channel=%d  dit_embed_dim=%d  dit_depth=%d",
            self.n_f, num_views, num_obs_tokens,
            self.diff_decoders[0].target_res, self.diff_decoders[0].target_res,
            z_channel, dit_embed_dim, dit_depth,
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

        Touches every diff_decoder parameter so all ranks participate in the
        allgather even when there are no future-rgb supervision targets.
        """
        dummy = future_tokens.new_zeros(())
        for v, dec in enumerate(self.diff_decoders):
            obs_v = future_tokens[:1, v * self.num_obs_tokens:(v + 1) * self.num_obs_tokens]
            z = dec._prepare_condition(obs_v)
            dummy = dummy + z.sum() * 0.0
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
    ):
        """Training path: compute future_tokens and DiT DDPM future prediction loss.

        Args:
            hidden:      Qwen last hidden states (B, seq_len, D).
            batch:       Must contain ``"future_rgb"`` — list[Tensor(B,3,H,W)] in [0,1],
                         one entry per view, in the same order as the camera views.
                         If absent, returns a zero-weight dummy loss.
            hist_tokens: Optional (B, T_hist, D) from HistoryVisionEncoder.
            state:       Optional robot state (B, state_dim) or (B, 1, state_dim).

        Returns:
            future_tokens: (B, N_f, D) — appended to vl_embs for GR00T.
            future_loss:   Scalar DDPM loss (0.0 if no targets present).
        """
        kv = self._build_kv(hidden, hist_tokens, state)
        future_tokens = self._cross_attend(kv)          # (B, N_f, D)

        future_rgb_list: List[torch.Tensor] | None = batch.get("future_rgb", None)
        if future_rgb_list is None or len(future_rgb_list) == 0:
            return future_tokens, self._dummy_loss(future_tokens)

        total_loss = future_tokens.new_zeros(())
        for v, dec in enumerate(self.diff_decoders):
            obs_v = future_tokens[:, v * self.num_obs_tokens:(v + 1) * self.num_obs_tokens]
            target_v = future_rgb_list[v].to(hidden.device, hidden.dtype)
            total_loss = total_loss + dec.compute_loss(obs_v, target_v)

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
