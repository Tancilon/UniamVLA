"""FutureCrossAttnBranch: future prediction via cross-attention + Seer-style MAE decoder.

Fully Seer-aligned decoder (seer_model.py:407):

  learnable future_query_tokens  (1, N_f, D)   [N_f = num_views × num_obs_tokens]
              │
  L-layer CrossAttention          Q = future_query_tokens
                                  KV = cat([qwen_hidden, state_tokens])
              │
  future_tokens  (B, N_f, D)
      │
      ├── → appended to vl_embs for action conditioning
      │
      └── SeerViTDecoder (per view, MAE-style ViT decoder)  [future supervision]
            9 obs_tokens (B, 9, D) → obs_projector
            196 mask_tokens (learnable) + 2D sincos pos_embed
            → 2 × ViT Block (full self-attention)
            → decoder_pred: Linear(decoder_dim, patch_size²×3)
            → normalized patch MSE (Seer §3.3)

Seer alignment:
  - num_obs_tokens = 9 per view (semantic compression, not spatial 1:1 grid)
  - patch_size = 16 → 196 full-resolution patches (14×14)
  - Decoder: obs_tokens + mask_tokens → ViT blocks (Seer's image_decoder)
  - Loss: MSE on mean/var normalized patch targets (Seer's train_utils.py:150)
"""
from __future__ import annotations

import logging
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sincos position embedding helpers (adapted from Seer / MAE)
# ---------------------------------------------------------------------------

def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """2D sincos position embedding for a grid_size × grid_size uniform grid.

    Returns:
        (grid_size², embed_dim) float32 array.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos"
    half = embed_dim // 4
    omega = np.arange(half, dtype=np.float32) / half
    omega = 1.0 / (10000 ** omega)

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    gh, gw = np.meshgrid(grid_h, grid_w, indexing="ij")   # (H, W) each
    gh = gh.reshape(-1)   # (H*W,)
    gw = gw.reshape(-1)

    out_h = np.outer(gh, omega)   # (H*W, half)
    out_w = np.outer(gw, omega)
    emb = np.concatenate([np.sin(out_h), np.cos(out_h), np.sin(out_w), np.cos(out_w)], axis=1)
    return emb.astype(np.float32)   # (H*W, embed_dim)


# ---------------------------------------------------------------------------
# ViT decoder block (full bidirectional self-attention, no causal mask)
# ---------------------------------------------------------------------------

class _ViTBlock(nn.Module):
    """One transformer block with full (non-causal) self-attention + FFN."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full self-attention (no mask → obs and mask tokens attend to each other)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Per-view Seer-style MAE decoder
# ---------------------------------------------------------------------------

class SeerViTDecoder(nn.Module):
    """Seer-aligned MAE-style image reconstruction decoder.

    Seer seer_model.py:260-270 analog (translated to our parameterisation):
      obs_tokens (B, N_obs, D)  — semantic future token from cross-attention
        → obs_projector: Linear(D, decoder_dim)
      mask_tokens (B, N_mask, decoder_dim)  — learnable, one per target patch
      cat([obs_embed, mask_tokens]) + pos_embed (2D sincos, frozen)
        → 2 × ViT Block (bidirectional self-attention)
        → decoder_norm + decoder_pred: Linear(decoder_dim, patch_size²×3)
      Loss: MSE on mean/variance-normalized patches  (Seer train_utils.py:150)

    Args:
        d_model:         LLM hidden dimension.
        num_obs_tokens:  Obs tokens per view (Seer default = 9).
        patch_size:      Pixel patch side (Seer default = 16 → 196 patches for 224px).
        image_size:      Input image side (default 224).
        decoder_dim:     ViT decoder hidden size (default = d_model).
        num_heads:       ViT decoder attention heads (default 16).
        num_blocks:      ViT decoder depth (Seer uses 2).
        future_weight:   Scalar multiplier for the reconstruction loss.
    """

    def __init__(
        self,
        d_model: int,
        num_obs_tokens: int = 9,
        patch_size: int = 16,
        image_size: int = 224,
        decoder_dim: int | None = None,
        num_heads: int = 16,
        num_blocks: int = 2,
        future_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if decoder_dim is None:
            decoder_dim = d_model

        num_patches = (image_size // patch_size) ** 2   # e.g. 196 for 224/16
        obs_grid    = int(round(num_obs_tokens ** 0.5))
        if obs_grid * obs_grid != num_obs_tokens:
            raise ValueError(
                f"SeerViTDecoder: num_obs_tokens={num_obs_tokens} must be a perfect square "
                f"(needed for 2D sincos pos-embed of the obs-token grid). "
                f"Seer default is 9 (3×3)."
            )

        self.num_obs_tokens = num_obs_tokens
        self.num_patches    = num_patches
        self.patch_size     = patch_size
        self.future_weight  = future_weight
        self.decoder_dim    = decoder_dim

        # Project obs tokens from LLM dim to decoder dim.
        self.obs_projector = nn.Linear(d_model, decoder_dim)

        # Learnable mask tokens (one per target patch).
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Frozen 2D sincos position embedding (obs grid + mask grid concatenated).
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_obs_tokens + num_patches, decoder_dim),
            requires_grad=False,
        )
        self._init_pos_embed(obs_grid, image_size // patch_size, decoder_dim)

        # ViT decoder blocks (full bidirectional self-attention).
        self.decoder_blocks = nn.ModuleList([
            _ViTBlock(decoder_dim, num_heads) for _ in range(num_blocks)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size * 3)

        logger.info(
            "[SeerViTDecoder] num_obs=%d  num_patches=%d  patch_size=%d  "
            "decoder_dim=%d  num_heads=%d  num_blocks=%d",
            num_obs_tokens, num_patches, patch_size, decoder_dim, num_heads, num_blocks,
        )

    def _init_pos_embed(self, obs_grid: int, mask_grid: int, embed_dim: int) -> None:
        """Initialize frozen 2D sincos position embeddings."""
        obs_pos  = _get_2d_sincos_pos_embed(embed_dim, obs_grid)   # (N_obs, D)
        mask_pos = _get_2d_sincos_pos_embed(embed_dim, mask_grid)  # (N_mask, D)
        pos = np.concatenate([obs_pos, mask_pos], axis=0)          # (N_obs+N_mask, D)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) float [0,1] → (B, N_patches, patch_size²×3) raster order."""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, h * w, p * p * C)

    def compute_loss(
        self, obs_tokens: torch.Tensor, future_rgb: torch.Tensor
    ) -> torch.Tensor:
        """Seer-aligned MAE reconstruction loss for one camera view.

        Seer train_utils.py:150: normalizes target patches by their mean/var
        before computing MSE — equivalent to MAE's normalized-pixel MSE.

        Args:
            obs_tokens:  (B, num_obs_tokens, D) — semantic future tokens.
            future_rgb:  (B, 3, H, W) float [0,1] — ground-truth future frame.
        Returns:
            Scalar loss (future_weight × mean normalized-patch MSE).
        """
        B = obs_tokens.shape[0]
        dtype = obs_tokens.dtype

        # Project obs tokens to decoder dim.
        obs_embed = self.obs_projector(obs_tokens)              # (B, N_obs, decoder_dim)

        # Expand learnable mask tokens.
        mask_tokens = self.mask_token.expand(B, self.num_patches, -1)  # (B, N_mask, D_dec)

        # Concatenate and add position embedding.
        x = torch.cat([obs_embed, mask_tokens], dim=1)          # (B, N_obs+N_mask, D_dec)
        x = x + self.pos_embed.to(dtype=dtype)

        # ViT decoder forward (full bidirectional self-attention).
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # Predict target patches from mask-token positions only.
        pred = self.decoder_pred(x[:, self.num_obs_tokens:])    # (B, N_mask, patch_dim)

        # Patchify ground-truth, then normalize (Seer train_utils.py:150).
        # Auto-resize to expected image_size if dataset frames differ
        # (e.g. raw Calvin frames are 200×200 but decoder expects 224×224).
        expected_px = int(self.num_patches ** 0.5) * self.patch_size
        rgb = future_rgb.to(device=obs_tokens.device, dtype=torch.float32)
        if rgb.shape[-2] != expected_px or rgb.shape[-1] != expected_px:
            rgb = torch.nn.functional.interpolate(
                rgb, size=(expected_px, expected_px),
                mode="bilinear", align_corners=False,
            )
        target = self.patchify(rgb.to(dtype=dtype))
        mean = target.mean(dim=-1, keepdim=True)
        var  = target.var(dim=-1, keepdim=True)
        target = ((target - mean) / (var + 1e-6) ** 0.5).detach()

        return self.future_weight * F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Single cross-attention layer (query attends to KV)
# ---------------------------------------------------------------------------

class _CrossAttnLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_norm  = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return q


# ---------------------------------------------------------------------------
# FutureCrossAttnBranch
# ---------------------------------------------------------------------------

class FutureCrossAttnBranch(nn.Module):
    """Seer cross-attention + MAE ViT decoder for future prediction.

    Args:
        d_model:               LLM hidden dimension (D).
        num_views:             Number of camera views.
        num_obs_tokens:        Future query tokens per view (Seer default = 9).
                               Must be a perfect square (for 2D sincos pos-embed).
        num_cross_attn_layers: Cross-attention stack depth.
        num_heads:             Cross-attention heads.
        state_dim:             Robot state dimension (0 = disabled).
        future_weight:         Reconstruction loss weight.
        patch_size:            Pixel patch side for reconstruction (Seer = 16).
        image_size:            Input image side (default 224).
        decoder_dim:           ViT decoder hidden size (default = d_model).
        num_decoder_heads:     ViT decoder attention heads (default 16).
        num_decoder_blocks:    ViT decoder depth (Seer default 2).
    """

    def __init__(
        self,
        d_model: int,
        num_views: int = 2,
        num_obs_tokens: int = 9,
        num_cross_attn_layers: int = 2,
        num_heads: int = 8,
        state_dim: int = 0,
        future_weight: float = 1.0,
        patch_size: int = 16,
        image_size: int = 224,
        decoder_dim: int | None = None,
        num_decoder_heads: int = 16,
        num_decoder_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.num_views      = num_views
        self.num_obs_tokens = num_obs_tokens    # per view
        self.n_f            = num_views * num_obs_tokens   # total future tokens N_f

        # Learnable future query tokens.
        self.future_query_tokens = nn.Parameter(torch.zeros(1, self.n_f, d_model))
        nn.init.xavier_uniform_(self.future_query_tokens.view(1, -1, d_model))

        # Cross-attention stack.
        self.cross_attn_layers = nn.ModuleList([
            _CrossAttnLayer(d_model, num_heads)
            for _ in range(num_cross_attn_layers)
        ])

        # Optional state projection (supports K-frame state: (B, K, state_dim)).
        self.state_proj = nn.Linear(state_dim, d_model) if state_dim > 0 else None

        # Per-view Seer-style MAE decoders.
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

        logger.info(
            "[FutureCrossAttnBranch] N_f=%d  num_views=%d  num_obs=%d  "
            "patch_size=%d  num_patches=%d  decoder_dim=%s  "
            "state_proj=%s (state_dim=%d)  [Seer MAE decoder]",
            self.n_f, num_views, num_obs_tokens,
            patch_size, self.vit_decoders[0].num_patches,
            decoder_dim or d_model,
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
        """Concatenate Qwen hidden states, optional history tokens, state tokens.

        Supports K-frame state: state can be (B, K, state_dim) or (B, state_dim).
        All K state frames are projected and concatenated as KV context.
        """
        parts = [hidden]
        if hist_tokens is not None:
            parts.append(hist_tokens)
        if state is not None and self.state_proj is not None:
            s = state.float()
            if s.ndim == 2:
                s = s.unsqueeze(1)          # (B, 1, state_dim) — single frame
            # s: (B, K, state_dim)  — supports K-frame state
            st = self.state_proj(s)         # (B, K, D)
            parts.append(st.to(hidden.dtype))
        return torch.cat(parts, dim=1)

    def _cross_attend(self, kv: torch.Tensor) -> torch.Tensor:
        B = kv.shape[0]
        q = self.future_query_tokens.expand(B, -1, -1).to(kv.dtype)
        for layer in self.cross_attn_layers:
            q = layer(q, kv)
        return q    # (B, N_f, D)

    def _dummy_loss(self, future_tokens: torch.Tensor) -> torch.Tensor:
        """Zero-weight dummy loss for ZeRO-3 parameter collective alignment."""
        dummy = future_tokens.new_zeros(())
        for dec in self.vit_decoders:
            dummy = dummy + (dec.decoder_pred.weight * 0.0).sum()
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
        """Inference: return future_tokens for action conditioning."""
        kv = self._build_kv(hidden, hist_tokens, state)
        return self._cross_attend(kv)

    def compute_loss(
        self,
        hidden: torch.Tensor,
        batch: dict,
        hist_tokens: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        future_rgb_list: List[torch.Tensor] | None = None,
    ):
        """Training: compute future_tokens + Seer MAE reconstruction loss.

        Args:
            hidden:          Qwen last hidden states (B, seq_len, D).
            batch:           Collated batch dict (unused, kept for API compat).
            hist_tokens:     Optional (B, T_hist, D) — usually None when K frames
                             are already embedded in Qwen hidden.
            state:           Optional (B, state_dim) | (B, 1, state_dim) |
                             (B, K, state_dim) — K-frame state supported.
            future_rgb_list: list[Tensor(B, 3, H, W)] float [0,1], one per view.
                             None → zero-weight dummy loss.
        Returns:
            future_tokens: (B, N_f, D)
            future_loss:   Scalar MAE reconstruction loss.
        """
        kv           = self._build_kv(hidden, hist_tokens, state)
        future_tokens = self._cross_attend(kv)

        if future_rgb_list is None or len(future_rgb_list) == 0:
            return future_tokens, self._dummy_loss(future_tokens)

        total_loss = future_tokens.new_zeros(())
        for v, dec in enumerate(self.vit_decoders):
            obs_v = future_tokens[:, v * self.num_obs_tokens:(v + 1) * self.num_obs_tokens]
            total_loss = total_loss + dec.compute_loss(obs_v, future_rgb_list[v])

        return future_tokens, total_loss / self.num_views
