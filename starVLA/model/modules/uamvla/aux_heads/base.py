from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class HeadOutput:
    loss: torch.Tensor | None
    metrics: dict[str, float]
    predictions: dict[str, Any] | None


class AuxHead(ABC, nn.Module):
    """Abstract base class for auxiliary task heads."""

    @abstractmethod
    def compute_loss(
        self, hidden_states: torch.Tensor, batch: dict, mask: torch.Tensor
    ) -> HeadOutput:
        ...

    @abstractmethod
    def predict(
        self, hidden_states: torch.Tensor, batch: dict
    ) -> HeadOutput:
        ...

    def get_dummy_loss(self) -> torch.Tensor:
        """0.0 * params.sum() for DeepSpeed ZeRO-2 alignment."""
        device = next(self.parameters()).device
        dummy = torch.tensor(0.0, device=device)
        for p in self.parameters():
            if p.requires_grad:
                dummy = dummy + 0.0 * p.sum()
        return dummy
