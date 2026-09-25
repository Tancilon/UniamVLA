"""Per-level, per-view history aggregation for Qwen3-VL DeepStack."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class HistoryAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        if dim % num_heads:
            raise ValueError("History feature width must be divisible by num_heads")
        self.num_heads = num_heads
        self.norm_q = nn.RMSNorm(dim)
        self.norm_kv = nn.RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, query, memory):
        # Fold views into the batch: attention never crosses camera boundaries.
        b, v, n, d = query.shape
        q = self.norm_q(query)
        kv = self.norm_kv(memory)

        def split(x):
            return x.reshape(b * v, -1, self.num_heads, d // self.num_heads).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            split(self.q_proj(q)), split(self.k_proj(kv)), split(self.v_proj(kv)),
            dropout_p=0.0,
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, v, n, d))


class DeepStackHistoryFusion(nn.Module):
    """H oldest-to-newest history frames -> three current-sized feature maps.

    Each level has independent parameters. Cameras share those parameters but
    attend only within their own history. No state persists between calls.
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        if dim % 2:
            raise ValueError("Sinusoidal time encoding requires an even feature width")
        self.attentions = nn.ModuleList([HistoryAttention(dim, num_heads) for _ in range(3)])
        self.gates = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(3)])
        self.norms = nn.ModuleList([nn.RMSNorm(dim) for _ in range(3)])

    def forward(self, history, current, steps):
        if history.ndim != 6 or current.ndim != 5:
            raise ValueError("Expected history [B,3,H,2,N,D], current [B,3,2,N,D]")
        b, levels, t, views, n, d = history.shape
        if (levels, views) != (3, 2) or t < 1 or current.shape != (b, levels, views, n, d):
            raise ValueError(f"Invalid history/current shapes: {history.shape}, {current.shape}")
        steps = torch.as_tensor(steps, device=current.device, dtype=torch.float32)
        if steps.shape != (b,) or (steps < 0).any():
            raise ValueError("steps must contain one nonnegative episode index per sample")
        # Recompute in float32 so model.to(bfloat16) at serving time does not
        # quantize the frequencies differently from training.
        frequencies = torch.exp(
            torch.arange(0, d, 2, device=current.device, dtype=torch.float32) * (-math.log(10000.0) / d)
        )
        angles = steps[:, None] * frequencies[None]
        time = torch.cat([angles.sin(), angles.cos()], dim=-1).to(current.dtype)
        outputs = []
        for level in range(3):
            mem = history[:, level]
            context = self.attentions[level](
                mem[:, -1], mem.permute(0, 2, 1, 3, 4).reshape(b, views, t * n, d),
            )
            visual = current[:, level] + time[:, None, None]
            gate = self.gates[level](torch.cat([context, visual], dim=-1)).sigmoid()
            outputs.append(self.norms[level](gate * visual + (1 - gate) * context))
        return torch.stack(outputs, dim=1)
