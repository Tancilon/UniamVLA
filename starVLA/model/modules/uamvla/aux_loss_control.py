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
        warmup_steps: int = 2000,
        aux_ratio_cap: float = 0.5,
        action_loss_ema_beta: float = 0.99,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.enabled = bool(enabled)
        self.aux_budget = float(aux_budget)
        self.warmup_steps = int(warmup_steps)
        self.aux_ratio_cap = float(aux_ratio_cap)
        self.action_loss_ema_beta = float(action_loss_ema_beta)
        self.eps = float(eps)
        self.register_buffer("action_loss_ema", torch.tensor(0.0), persistent=True)
        self.register_buffer("_ema_initialized", torch.tensor(False), persistent=True)

    def _update_action_loss_ema(self, action_loss: torch.Tensor) -> torch.Tensor:
        value = action_loss.detach().float()
        if not bool(self._ema_initialized.item()):
            self.action_loss_ema.copy_(value)
            self._ema_initialized.fill_(True)
        else:
            beta = self.action_loss_ema_beta
            self.action_loss_ema.mul_(beta).add_(value, alpha=1.0 - beta)
        return self.action_loss_ema.to(device=action_loss.device, dtype=action_loss.dtype)

    def _warmup_scale(self, global_step: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.warmup_steps <= 0:
            value = 1.0
        else:
            value = min(1.0, max(0.0, float(global_step) / float(self.warmup_steps)))
        return torch.tensor(value, device=device, dtype=dtype)

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
        action_ema = self._update_action_loss_ema(action_loss)
        aux_cap = self.aux_ratio_cap * action_ema
        aux_scale = torch.clamp(aux_cap / (aux_total_pre.detach() + self.eps), max=1.0)
        warmup = self._warmup_scale(global_step, device, dtype)
        total_scale = warmup * self.aux_budget * aux_scale
        aux_total_post = total_scale * aux_total_pre

        metrics["aux_total_pre_budget"] = aux_total_pre.detach()
        metrics["aux_total_post_budget"] = aux_total_post.detach()
        metrics["aux_scale_budget"] = aux_scale.detach()
        metrics["aux_budget_warmup"] = warmup.detach()
        metrics["action_loss_ema"] = action_ema.detach()
        metrics["aux_cap"] = aux_cap.detach()

        for name, loss in losses.items():
            metrics[f"{name}_loss_contribution_post_budget"] = (total_scale * loss).detach()

        return aux_total_post, metrics
