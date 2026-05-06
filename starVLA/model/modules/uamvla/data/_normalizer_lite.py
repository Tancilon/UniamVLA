"""Lightweight, dependency-free copy of gr00t_lerobot.transform.state_action.Normalizer.

Why this file exists:
  StateNormalizer (this package's __init__-time peer) historically imports
  Normalizer from starVLA/dataloader/gr00t_lerobot/transform/state_action.py.
  That module pulls in the full training-side schema chain, which contains
  Python 3.9+ generic-builtin syntax (`tuple[float, float]`,
  `dict[str, X]`, ...) and heavyweight optional deps (numpydantic,
  pytorch3d). The CALVIN benchmark eval conda env runs Python 3.8 and
  ships only PyBullet + hydra + gym; it cannot import gr00t_lerobot.

  The Normalizer class itself is pure torch math with no schema or pydantic
  dependency. Re-implementing it here, syntactically compatible with Python
  3.8, lets the eval-side StateNormalizer fall back to a self-contained
  copy when the gr00t import chain is unavailable. Training environments
  still use the gr00t Normalizer as the single source of truth (see
  state_normalizer.py's try/except).

Behavior parity:
  Math is byte-identical to the gr00t copy as of this commit (q99 / mean_std
  / min_max / scale / binary modes). Any future change to the gr00t copy
  must be mirrored here, or the fallback path will silently diverge — that
  is the explicit cost of decoupling the eval client from training deps.
"""
from __future__ import annotations

import torch


class Normalizer:
    valid_modes = ["q99", "mean_std", "min_max", "binary"]

    def __init__(self, mode: str, statistics: dict, binary_threshold: float = 0.5):
        self.mode = mode
        self.statistics = statistics
        self.binary_threshold = binary_threshold
        for key, value in self.statistics.items():
            self.statistics[key] = torch.tensor(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        if self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            mask = q01 != q99
            normalized = torch.zeros_like(x)
            normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (
                q99[..., mask] - q01[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)
            normalized = torch.clamp(normalized, -1, 1)

        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            mask = std != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = (x[..., mask] - mean[..., mask]) / std[..., mask]
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

        elif self.mode == "min_max":
            min_ = self.statistics["min"].to(x.dtype)
            max_ = self.statistics["max"].to(x.dtype)
            mask = min_ != max_
            normalized = torch.zeros_like(x)
            normalized[..., mask] = (x[..., mask] - min_[..., mask]) / (
                max_[..., mask] - min_[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1
            normalized[..., ~mask] = 0

        elif self.mode == "scale":
            min_ = self.statistics["min"].to(x.dtype)
            max_ = self.statistics["max"].to(x.dtype)
            abs_max = torch.max(torch.abs(min_), torch.abs(max_))
            mask = abs_max != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = x[..., mask] / abs_max[..., mask]
            normalized[..., ~mask] = 0

        elif self.mode == "binary":
            normalized = (x > self.binary_threshold).to(x.dtype)

        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

        return normalized
