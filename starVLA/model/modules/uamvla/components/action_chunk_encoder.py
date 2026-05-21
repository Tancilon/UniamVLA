from __future__ import annotations

import math

import torch
import torch.nn as nn


def _build_sincos_pos_embed(max_horizon: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(max_horizon, dim)
    position = torch.arange(max_horizon, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class ActionChunkEncoder(nn.Module):
    """Encode normalized action chunks into one hidden conditioning vector."""

    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        action_embed_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        max_horizon: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(action_dim, action_embed_dim)
        self.pos_embed = nn.Parameter(
            _build_sincos_pos_embed(max_horizon, action_embed_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=action_embed_dim,
            nhead=num_heads,
            dim_feedforward=action_embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(action_embed_dim, hidden_size)

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        if action_chunk.ndim != 3:
            raise ValueError(
                f"action_chunk must have shape [B, T, A], got {tuple(action_chunk.shape)}"
            )
        horizon = action_chunk.shape[1]
        if horizon > self.pos_embed.shape[1]:
            raise ValueError(
                f"action horizon {horizon} exceeds max_horizon {self.pos_embed.shape[1]}"
            )
        x = self.input_proj(action_chunk.float())
        x = x + self.pos_embed[:, :horizon, :].to(dtype=x.dtype, device=x.device)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.output_proj(pooled)
