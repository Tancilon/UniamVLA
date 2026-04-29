"""Task query reader modules for extracting and spatializing backbone latents.

Ported from old-version/uamvla/model/query_reader.py with import path updates.
"""

import math
from typing import Callable, cast

import torch
import torch.nn as nn
import einops
from typing_extensions import override

__all__ = ["QueryReaderLayer", "TaskQueryReader", "QueryToSpatial"]

rearrange_fn: Callable[..., torch.Tensor] = cast(
    Callable[..., torch.Tensor], einops.rearrange
)


class QueryReaderLayer(nn.Module):
    """Single pre-norm reader layer with cross-attention, self-attention, and FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        nn.Module.__init__(self)

        self.cross_attn_ln_q = nn.LayerNorm(hidden_dim)
        self.cross_attn_ln_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )

        self.self_attn_ln = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )

        self.ffn_ln = nn.LayerNorm(hidden_dim)
        ffn_hidden = int(hidden_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    @override
    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_norm = self.cross_attn_ln_q(queries)
        kv_norm = self.cross_attn_ln_kv(memory)
        cross_out = self.cross_attn(
            q_norm, kv_norm, kv_norm,
            key_padding_mask=memory_key_padding_mask, need_weights=False,
        )[0]
        queries = queries + cross_out

        q_norm = self.self_attn_ln(queries)
        self_out = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)[0]
        queries = queries + self_out

        queries = queries + self.ffn(self.ffn_ln(queries))
        return queries


class TaskQueryReader(nn.Module):
    """Task-specific readout transformer for compact latent extraction."""

    def __init__(
        self,
        num_queries: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        nn.Module.__init__(self)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        self.queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([
            QueryReaderLayer(
                hidden_dim=hidden_dim, num_heads=num_heads,
                ffn_ratio=ffn_ratio, dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_dim)

    @override
    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = memory.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.layers:
            queries = layer(
                queries=queries, memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        return self.final_ln(queries)


class QueryToSpatial(nn.Module):
    """Project query latents to spatial conditioning for the DiT denoiser."""

    def __init__(self, num_queries: int, hidden_dim: int, n_patches: int) -> None:
        nn.Module.__init__(self)
        self.n_patches = n_patches
        self.h = int(math.sqrt(n_patches))
        self.w = self.h
        assert self.h * self.w == n_patches

        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.spatial_proj = nn.Linear(num_queries, n_patches)

    @override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.ln(z)
        z = z.transpose(1, 2)
        z = self.spatial_proj(z)
        z = z.transpose(1, 2)
        z = rearrange_fn(z, "b (h w) c -> b c h w", h=self.h, w=self.w)
        return z
