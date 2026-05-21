from __future__ import annotations

import torch
import torch.nn as nn


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

        for name, head in self.heads.items():
            mask = masks[name]
            out = head.compute_loss(hidden_states, batch, mask=mask)
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
