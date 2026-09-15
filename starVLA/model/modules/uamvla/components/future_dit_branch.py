"""FutureDiTBranch: Flow Matching DiT for future image token prediction.

Architecture:
  future_query_tokens (B, N_obs, D)   ← condition from _run_joint
       ↓  mean-pool + AdaLN
  FutureDiT(noisy_gt_tokens, timestep)
       ↓  velocity prediction
  pred_x0 = one-step estimate of clean tokens
       ↓
  FuturePixelDecoder                  → (B, 3, H, W)
       ↓
  pixel_loss + token_loss

Training uses Flow Matching (same schedule as GR00T_ActionHeader):
  t ~ Beta(alpha, beta), clamped to [0, noise_s]
  x_t = (1-t)*noise + t*x0_gt
  velocity_gt = x0_gt - noise
  token_loss = MSE(v_pred, velocity_gt)
  pixel_loss = MSE(PixelDecoder(pred_x0), future_rgb)

Inference: 4-step Euler integration from pure noise.
"""
from __future__ import annotations

import math
import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1D sincos positional embedding helper
# ---------------------------------------------------------------------------

def _get_1d_sincos_pos_embed(embed_dim: int, n: int) -> np.ndarray:
    """1D sincos position embedding, shape (n, embed_dim)."""
    assert embed_dim % 2 == 0
    position = np.arange(n)[:, np.newaxis]          # (n, 1)
    div = np.exp(np.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
    pe = np.zeros((n, embed_dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div)
    pe[:, 1::2] = np.cos(position * div)
    return pe


# ---------------------------------------------------------------------------
# FutureDiT — lightweight token-sequence DiT
# ---------------------------------------------------------------------------

class FutureDiT(nn.Module):
    """Token-sequence DiT adapted from denoiser/dit.py.

    Differences from the spatial DiT:
    - x_embedder: Linear (token sequence, not PatchEmbed)
    - pos_embed:  1D sincos (num_img_tokens positions)
    - z_embedder: mean-pools future_query_tokens, then Linear → AdaLN condition
    - final_layer output: d_model (same as input, for velocity prediction)
    """

    def __init__(
        self,
        d_model: int,
        hidden_size: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        num_img_tokens: int = 49,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        from starVLA.model.modules.uamvla.components.denoiser.common import (
            TimestepEmbedder, FinalLayer,
        )
        from starVLA.model.modules.uamvla.components.denoiser.dit import DiTBlock

        self.num_img_tokens = num_img_tokens
        self.d_model = d_model

        # Token embedders
        self.x_embedder = nn.Linear(d_model, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.z_embedder = nn.Linear(d_model, hidden_size)   # applied after mean-pool

        # 1D frozen sincos position embedding
        pos = _get_1d_sincos_pos_embed(hidden_size, num_img_tokens)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos).float().unsqueeze(0),     # (1, N, H)
            requires_grad=False,
        )

        # DiT blocks (reuse from denoiser/dit.py)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Output projection back to d_model (velocity in token space)
        # FinalLayer expects (hidden_size, patch_size=1, out_channels=d_model)
        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)
        nn.init.xavier_uniform_(self.z_embedder.weight)
        nn.init.zeros_(self.z_embedder.bias)

    def forward(
        self,
        x: torch.Tensor,               # (B, N_img, d_model)  noisy tokens
        future_query: torch.Tensor,     # (B, N_obs, d_model)  condition
        timestep: torch.Tensor,         # (B,)  long, discretized
    ) -> torch.Tensor:                  # (B, N_img, d_model)  predicted velocity
        dtype = self.x_embedder.weight.dtype
        x = x.to(dtype)
        future_query = future_query.to(dtype)

        # Embed noisy tokens + positional encoding
        x = self.x_embedder(x) + self.pos_embed.to(dtype)   # (B, N_img, H)

        # Timestep + condition embeddings
        t_emb = self.t_embedder(timestep.to(x.device))        # (B, H)
        z_emb = self.z_embedder(future_query.mean(dim=1))      # (B, H) — mean-pool query
        c = (t_emb + z_emb).unsqueeze(1).expand(-1, self.num_img_tokens, -1)  # (B, N, H)

        # DiT blocks
        for block in self.blocks:
            x = block(x, c)

        # Project back to d_model (velocity prediction)
        x = self.final_layer(x, c)      # (B, N_img, d_model)
        return x


# ---------------------------------------------------------------------------
# FuturePixelDecoder — 49 tokens → 224×224 pixels via inverse patchify
# ---------------------------------------------------------------------------

class FuturePixelDecoder(nn.Module):
    """Inverse patchify: N_img tokens (B, N, D) → (B, 3, H, W).

    Each of the N=7×7=49 tokens covers a (H/7)×(W/7) = 32×32 pixel patch.
    A single Linear projects each token to its patch pixels.
    """

    def __init__(
        self,
        d_model: int,
        num_img_tokens: int = 49,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        grid = int(round(num_img_tokens ** 0.5))
        assert grid * grid == num_img_tokens, \
            f"num_img_tokens={num_img_tokens} must be a perfect square"
        self.patch_px = image_size // grid      # pixel size per patch (32)
        self.grid = grid
        self.proj = nn.Linear(d_model, self.patch_px ** 2 * 3)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, D) → (B, 3, H, W) float."""
        B = tokens.shape[0]
        patches = self.proj(tokens)             # (B, N, patch_px²×3)
        g, p = self.grid, self.patch_px
        x = patches.reshape(B, g, g, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, g * p, g * p)
        return x


# ---------------------------------------------------------------------------
# FutureDiTBranch — full flow-matching wrapper (drop-in for SeerViTDecoder)
# ---------------------------------------------------------------------------

class FutureDiTBranch(nn.Module):
    """Flow Matching future prediction branch.

    Replaces SeerViTDecoder as a drop-in, with the same compute_loss interface:
      compute_loss(obs_tokens, future_rgb, gt_img_tokens) → scalar loss

    Training computes two losses:
      token_loss  = MSE(v_pred, velocity_gt)          in token space
      pixel_loss  = MSE(PixelDecoder(pred_x0), rgb)   in pixel space
      total       = token_loss + pixel_loss_weight * pixel_loss

    Args:
        d_model:           Qwen LLM hidden size (= GT token dim).
        num_img_tokens:    Qwen image tokens per view (49 for 224px / patch16 / merge2).
        image_size:        Image side in pixels (224).
        dit_hidden:        DiT internal width (lightweight, default 256).
        dit_depth:         DiT blocks (default 2).
        dit_num_heads:     DiT attention heads (default 8).
        num_inference_steps: Euler steps at inference (default 4).
        token_loss_weight: Weight for velocity prediction loss (default 1.0).
        pixel_loss_weight: Weight for pixel reconstruction loss (default 0.1).
        noise_beta_alpha / _beta / _s:  Flow matching noise schedule (GR00T defaults).
        num_timestep_buckets: Discretization for TimestepEmbedder (default 1000).
    """

    def __init__(
        self,
        d_model: int,
        num_img_tokens: int = 49,
        image_size: int = 224,
        dit_hidden: int = 256,
        dit_depth: int = 2,
        dit_num_heads: int = 8,
        num_inference_steps: int = 4,
        token_loss_weight: float = 1.0,
        pixel_loss_weight: float = 0.1,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        num_timestep_buckets: int = 1000,
    ) -> None:
        super().__init__()

        self.dit = FutureDiT(
            d_model=d_model,
            hidden_size=dit_hidden,
            depth=dit_depth,
            num_heads=dit_num_heads,
            num_img_tokens=num_img_tokens,
        )
        self.pixel_decoder = FuturePixelDecoder(
            d_model=d_model,
            num_img_tokens=num_img_tokens,
            image_size=image_size,
        )

        self.beta_dist = Beta(
            torch.tensor(noise_beta_alpha, dtype=torch.float32),
            torch.tensor(noise_beta_beta, dtype=torch.float32),
        )
        self.noise_s = noise_s
        self.num_timestep_buckets = num_timestep_buckets
        self.num_inference_steps = num_inference_steps
        self.token_loss_weight = token_loss_weight
        self.pixel_loss_weight = pixel_loss_weight
        self.num_img_tokens = num_img_tokens
        self.d_model = d_model
        self.image_size = image_size

        logger.info(
            "[FutureDiTBranch] d_model=%d  num_img_tokens=%d  "
            "dit_hidden=%d  dit_depth=%d  pixel_loss_weight=%.3f",
            d_model, num_img_tokens, dit_hidden, dit_depth, pixel_loss_weight,
        )

    # ------------------------------------------------------------------
    # Noise schedule helpers
    # ------------------------------------------------------------------

    def _sample_time(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample t from Beta distribution, bias toward t=1 (clean signal)."""
        t = self.beta_dist.sample([B]).to(device=device, dtype=dtype)
        return self.noise_s * (1.0 - t)          # (B,), values in (0, noise_s]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        obs_tokens: torch.Tensor,       # (B, N_obs, D)  future query condition
        future_rgb: torch.Tensor,       # (B, 3, H, W)   ground-truth future frame
        gt_img_tokens: torch.Tensor,    # (B, N_img, D)  GT tokens from frozen Qwen ViT
    ) -> torch.Tensor:
        """Flow matching training loss (token + pixel).

        Args:
            obs_tokens:    Future query output from _run_joint (condition).
            future_rgb:    Ground-truth future frame in [0, 1].
            gt_img_tokens: GT image tokens from frozen Qwen visual encoder.
        Returns:
            Scalar loss = token_loss_weight * token_loss
                        + pixel_loss_weight * pixel_loss
        """
        B, device, dtype = obs_tokens.shape[0], obs_tokens.device, obs_tokens.dtype

        gt = gt_img_tokens.to(device=device, dtype=dtype)   # (B, N_img, D)

        # ── Flow matching noise schedule ──────────────────────────────────
        t = self._sample_time(B, device, dtype)             # (B,)
        noise = torch.randn_like(gt)
        x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * gt
        velocity_gt = gt - noise                            # flow matching target

        t_disc = (t * self.num_timestep_buckets).long()     # discretize for embedder

        # ── DiT velocity prediction ───────────────────────────────────────
        v_pred = self.dit(x_t, obs_tokens, t_disc)          # (B, N_img, D)
        token_loss = F.mse_loss(v_pred, velocity_gt)

        # ── One-step x0 estimate → pixel decoder ─────────────────────────
        pred_x0 = x_t + (1.0 - t[:, None, None]) * v_pred  # (B, N_img, D)
        pred_px = self.pixel_decoder(pred_x0)               # (B, 3, H', W')

        # Resize GT rgb to decoder output size if necessary
        target = future_rgb.to(device=device, dtype=torch.float32)
        if target.shape[-2:] != pred_px.shape[-2:]:
            target = F.interpolate(
                target, size=pred_px.shape[-2:], mode="bilinear", align_corners=False
            )
        pixel_loss = F.mse_loss(pred_px.float(), target)

        return (self.token_loss_weight * token_loss
                + self.pixel_loss_weight * pixel_loss)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        obs_tokens: torch.Tensor,   # (B, N_obs, D)
    ) -> torch.Tensor:              # (B, 3, H, W) predicted future pixels
        """4-step Euler integration from pure noise → pixel image."""
        B, device, dtype = obs_tokens.shape[0], obs_tokens.device, obs_tokens.dtype
        x = torch.randn(B, self.num_img_tokens, self.d_model, device=device, dtype=dtype)
        dt = 1.0 / self.num_inference_steps

        for step in range(self.num_inference_steps):
            t_cont = step * dt
            t_disc = int(t_cont * self.num_timestep_buckets)
            t_tensor = torch.full((B,), t_disc, dtype=torch.long, device=device)
            v = self.dit(x, obs_tokens, t_tensor)
            x = x + dt * v                               # Euler step

        return self.pixel_decoder(x)                     # (B, 3, H, W)
