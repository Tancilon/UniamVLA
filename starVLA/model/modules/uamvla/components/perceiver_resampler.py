"""PerceiverResampler: compress a variable-length visual token sequence to a fixed-size latent.

Ported from third_party/Seer/models/perceiver_resampler.py.
Changes vs. original:
  - Removed einops_exts dependency; rearrange_many replaced with plain einops.rearrange.
  - Simplified forward signature: accepts (b, T, n_tokens, D) instead of (b, T, F, v, D)
    so callers don't need to distinguish frames (F) from views (v).
  - Retained optional frame / media-time positional embeddings for multi-timestep use.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import einsum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exists(val):
    return val is not None


def _FeedForward(dim: int, mult: int = 4) -> nn.Sequential:
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


# ---------------------------------------------------------------------------
# Perceiver cross-attention block
# ---------------------------------------------------------------------------

class PerceiverAttention(nn.Module):
    """Cross-attention: latents (queries) attend to media tokens (keys/values).

    Args:
        dim: Feature dimension.
        dim_head: Dimension per attention head.
        heads: Number of attention heads.
    """

    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       Media (image) features  — (b, T, n_media, D)
            latents: Learnable latent queries — (b, T, n_latents, D)
        Returns:
            Updated latents — (b, T, n_latents, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        h = self.heads

        q = self.to_q(latents)                          # (b, T, n_latents, inner)
        kv_input = torch.cat((x, latents), dim=-2)      # attend to media + latents
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # split heads
        q = rearrange(q, "b t n (h d) -> b h t n d", h=h)
        k = rearrange(k, "b t n (h d) -> b h t n d", h=h)
        v = rearrange(v, "b t n (h d) -> b h t n d", h=h)

        q = q * self.scale
        sim = einsum("b h t i d, b h t j d -> b h t i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h t i j, b h t j d -> b h t i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)")
        return self.to_out(out)


# ---------------------------------------------------------------------------
# PerceiverResampler
# ---------------------------------------------------------------------------

class PerceiverResampler(nn.Module):
    """Compress a variable-length token sequence to ``num_latents`` tokens.

    Input shape expected by ``forward``: **(b, T, n_tokens, D)**

    where
      - b = batch size
      - T = number of timesteps / media clips (e.g. K-1 historical frames)
      - n_tokens = tokens per timestep (e.g. 49 for a 224-px Qwen view)
      - D = feature dimension

    Args:
        dim: Input (and output) feature dimension.
        depth: Number of (PerceiverAttention + FFN) blocks.
        dim_head: Head dimension for attention.
        heads: Number of attention heads.
        num_latents: Number of compressed output tokens per timestep.
        max_num_media: If set, adds learnable media-time positional embeddings.
        ff_mult: Feed-forward hidden-dimension multiplier.
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int = 3,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 10,
        max_num_media: int | None = None,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.num_latents = num_latents

        # Learnable compressed queries, shared across batch and time.
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # Optional temporal positional embedding (one vector per media clip).
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if _exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                _FeedForward(dim=dim, mult=ff_mult),
            ])
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, T, n_tokens, D) — media features for T timesteps.
               In multi-frame, multi-view scenarios, T = num_frames × num_views
               (frames and views are pre-merged by the caller, frame-major order:
               [f0v0, f0v1, f1v0, f1v1, ...]). Positional information is encoded
               per T slot via media_time_embs, so no structural information is lost.

        Returns:
            (b, T, num_latents, D) — compressed latent tokens.
        """
        b, T, n, d = x.shape

        if _exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]          # broadcast over n dimension

        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)                      # (b, T, num_latents, D)
