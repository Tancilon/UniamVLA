import torch
import torch.nn as nn


class TaskAdapter(nn.Module):
    """Bottleneck MLP adapter with residual connection for task-specific specialization."""

    def __init__(self, dim: int, bottleneck_dim: int = 512) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.ln(x))
