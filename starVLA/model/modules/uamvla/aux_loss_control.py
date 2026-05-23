from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", "0") or 0)


def _cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class AuxDenoisingSuite(nn.Module):
    """Run aux heads and apply global auxiliary loss budget control."""

    def __init__(
        self,
        heads: nn.ModuleDict,
        enabled: bool = True,
        aux_budget: float = 1.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.enabled = bool(enabled)
        self.aux_budget = float(aux_budget)

    @staticmethod
    def _metric_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().to(device=device, dtype=dtype)
        return torch.tensor(float(value), device=device, dtype=dtype)

    @staticmethod
    def _head_metric_key(head_name: str, metric_name: str) -> str:
        prefix = f"{head_name}_"
        metric_core = (
            metric_name[len(prefix):]
            if metric_name.startswith(prefix)
            else metric_name
        )
        return f"{head_name}_{metric_core}_raw"

    def forward(
        self,
        action_loss: torch.Tensor,
        hidden_states: torch.Tensor,
        batch: dict,
        masks: dict[str, torch.Tensor],
        global_step: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        if not self.enabled or not self.heads:
            return action_loss.new_zeros(()), {}

        device = action_loss.device
        dtype = action_loss.dtype
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor | float] = {}
        profile = _env_flag("UAMVLA_AUX_PROFILE")
        profile_rank = _rank()

        for name, head in self.heads.items():
            mask = masks[name]
            if profile:
                valid = int(mask.detach().sum().item()) if torch.is_tensor(mask) else -1
                _cuda_synchronize(device)
                head_start = time.perf_counter()
                print(
                    f"[uamvla-aux-profile][rank{profile_rank}] "
                    f"step={global_step} head={name} start valid={valid}/{len(mask)}",
                    flush=True,
                )
            out = head.compute_loss(hidden_states, batch, mask=mask)
            if profile:
                _cuda_synchronize(device)
                elapsed = time.perf_counter() - head_start
                print(
                    f"[uamvla-aux-profile][rank{profile_rank}] "
                    f"step={global_step} head={name} done elapsed={elapsed:.3f}s",
                    flush=True,
                )
            if out.loss is None:
                continue
            losses[name] = out.loss
            raw_value = out.metrics.get("loss_raw", out.metrics.get(f"{name}_loss", 0.0))
            metrics[f"{name}_loss_raw"] = self._metric_tensor(
                raw_value, device, dtype
            )
            metrics[f"{name}_valid_ratio"] = float(
                out.metrics.get("valid_ratio", mask.float().mean().item())
            )
            metrics[f"{name}_loss_weighted_pre_budget"] = out.loss.detach()
            metrics[f"{name}_loss_weighted"] = out.loss.detach()
            for metric_name, metric_value in out.metrics.items():
                if metric_name in {"loss_raw", f"{name}_loss", "valid_ratio"}:
                    continue
                metrics[self._head_metric_key(name, metric_name)] = (
                    metric_value.detach() if torch.is_tensor(metric_value) else metric_value
                )

        if not losses:
            return action_loss.new_zeros(()), metrics

        aux_total_pre = torch.stack([loss for loss in losses.values()]).sum()
        aux_total_post = self.aux_budget * aux_total_pre

        metrics["aux_total_pre_budget"] = aux_total_pre.detach()
        metrics["aux_total_post_budget"] = aux_total_post.detach()

        for name, loss in losses.items():
            metrics[f"{name}_loss_contribution_post_budget"] = (self.aux_budget * loss).detach()

        return aux_total_post, metrics
