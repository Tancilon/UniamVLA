"""Dynamic aux head weight balancing suite.

Implements a two-level dynamic weighting strategy:
1. Global annealing: all aux heads gradually fade out toward end of training
2. Relative balancing: maintain equilibrium between different aux heads via EMA

Training phases:
- Early (0% - 60%): Full aux head weight, balanced contributions
- Middle (60% - 80%): Gradual annealing begins
- Late (80% - 100%): Aggressive annealing, aux heads → 0, action loss dominates

Design rationale:
- Early training: aux heads help learn rich representations
- Late training: main task (action prediction) should dominate
- Final stage: pure action loss for task-specific fine-tuning
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class DynamicAuxSuite(nn.Module):
    """Dynamically balanced auxiliary loss suite with global annealing.

    Args:
        aux_heads: Dict of {name: AuxHead} instances to manage.
        total_steps: Total training steps for annealing schedule.
        base_weights: Static base weight for each head (optional override).
        warmup_steps: Steps before aux heads reach full weight (default: 5% of total).
        annealing_start_ratio: When to start annealing (default: 0.6 = 60% of training).
        annealing_end_ratio: When aux weights reach zero (default: 1.0 = 100%).
        enable_ema_balancing: Whether to use EMA-based relative balancing (default: True).
        ema_momentum: EMA momentum for loss tracking (default: 0.99).
        balance_warmup_steps: Steps to warm up EMA before balancing kicks in (default: 1000).
        target_ratios: Optional dict of {head_name: target_contribution_ratio}.
            If None, all heads are weighted equally in relative balancing.
        min_weight_floor: Minimum weight multiplier to prevent complete zero (default: 0.0).
        max_weight_cap: Maximum weight multiplier to prevent explosion (default: 5.0).
    """

    def __init__(
        self,
        aux_heads: Dict[str, nn.Module],
        total_steps: int,
        base_weights: Optional[Dict[str, float]] = None,
        warmup_steps: Optional[int] = None,
        annealing_start_ratio: float = 0.6,
        annealing_end_ratio: float = 1.0,
        enable_ema_balancing: bool = True,
        ema_momentum: float = 0.99,
        balance_warmup_steps: int = 1000,
        target_ratios: Optional[Dict[str, float]] = None,
        min_weight_floor: float = 0.0,
        max_weight_cap: float = 5.0,
    ):
        super().__init__()
        self.aux_heads = aux_heads
        self.total_steps = total_steps
        self.annealing_start_ratio = annealing_start_ratio
        self.annealing_end_ratio = annealing_end_ratio
        self.enable_ema_balancing = enable_ema_balancing
        self.ema_momentum = ema_momentum
        self.balance_warmup_steps = balance_warmup_steps
        self.min_weight_floor = min_weight_floor
        self.max_weight_cap = max_weight_cap

        # Warmup period: gradually ramp up aux weights from 0 to 1
        self.warmup_steps = warmup_steps if warmup_steps is not None else int(0.05 * total_steps)

        # Base static weights (fallback to head's own loss_weight)
        self.base_weights = {}
        for name, head in aux_heads.items():
            if base_weights and name in base_weights:
                self.base_weights[name] = base_weights[name]
            elif hasattr(head, "loss_weight"):
                self.base_weights[name] = head.loss_weight
            else:
                self.base_weights[name] = 1.0

        # Target ratios for relative balancing (equal by default)
        if target_ratios is None:
            n_heads = len(aux_heads)
            self.target_ratios = {name: 1.0 / n_heads for name in aux_heads}
        else:
            # Normalize to sum to 1.0
            total = sum(target_ratios.values())
            self.target_ratios = {k: v / total for k, v in target_ratios.items()}

        # EMA tracking for each head's raw loss
        self.register_buffer("ema_raw_loss", torch.zeros(len(aux_heads)))
        self.head_name_to_idx = {name: i for i, name in enumerate(aux_heads.keys())}

        # Statistics tracking
        self.register_buffer("ema_initialized", torch.tensor(False))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    # ──────────────────────────────────────────────────────────────────
    #  Phase 1: Global annealing multiplier
    # ──────────────────────────────────────────────────────────────────
    def _compute_global_multiplier(self, step: int) -> float:
        """Compute the global aux weight multiplier for the current step.

        Timeline:
          [0, warmup_steps)           : linear ramp 0 → 1.0
          [warmup, annealing_start)   : plateau at 1.0
          [annealing_start, anneal_end]: cosine decay 1.0 → 0.0
          [annealing_end, total_steps]: clamped at min_weight_floor

        Uses cosine decay for smooth gradient transition (avoids sudden
        loss landscape changes that can destabilize late-stage training).
        """
        # Warmup phase
        if step < self.warmup_steps and self.warmup_steps > 0:
            return float(step) / float(self.warmup_steps)

        annealing_start = int(self.annealing_start_ratio * self.total_steps)
        annealing_end   = int(self.annealing_end_ratio   * self.total_steps)

        # Plateau: full weight
        if step < annealing_start:
            return 1.0

        # After full annealing
        if step >= annealing_end:
            return self.min_weight_floor

        # Cosine decay from 1.0 → min_weight_floor
        progress = (step - annealing_start) / max(annealing_end - annealing_start, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_weight_floor + (1.0 - self.min_weight_floor) * cosine

    # ──────────────────────────────────────────────────────────────────
    #  Phase 2: EMA-based relative weight balancing
    # ──────────────────────────────────────────────────────────────────
    def _update_ema(self, raw_losses: Dict[str, float]) -> None:
        """Update EMA for each head's raw loss."""
        for name, val in raw_losses.items():
            idx = self.head_name_to_idx[name]
            if not self.ema_initialized.item():
                # Cold start: initialize EMA with first observed value
                self.ema_raw_loss[idx] = val
            else:
                self.ema_raw_loss[idx] = (
                    self.ema_momentum * self.ema_raw_loss[idx]
                    + (1.0 - self.ema_momentum) * val
                )
        if not self.ema_initialized.item():
            self.ema_initialized.fill_(True)

    def _compute_relative_weights(self, step: int) -> Dict[str, float]:
        """Compute per-head relative weight multipliers for EMA balancing.

        Goal: adjust each head's effective weight so its contribution to
        aux_total matches target_ratios. Returns a multiplier in
        [1/max_weight_cap, max_weight_cap] per head.

        Before balance_warmup_steps: returns 1.0 for all (EMA not warmed up).
        """
        if not self.enable_ema_balancing:
            return {name: 1.0 for name in self.aux_heads}

        if step < self.balance_warmup_steps or not self.ema_initialized.item():
            return {name: 1.0 for name in self.aux_heads}

        # Current weighted contribution estimate per head:
        #   contribution_i ≈ base_weight_i × ema_raw_loss_i
        contrib = {}
        for name in self.aux_heads:
            idx   = self.head_name_to_idx[name]
            ema_v = self.ema_raw_loss[idx].item()
            contrib[name] = self.base_weights[name] * max(ema_v, 1e-8)

        total_contrib = sum(contrib.values())
        if total_contrib < 1e-12:
            return {name: 1.0 for name in self.aux_heads}

        # Compute adjustment: if head_i contributes too little → upweight
        relative = {}
        for name in self.aux_heads:
            current_ratio  = contrib[name] / total_contrib
            target_ratio   = self.target_ratios.get(name, 1.0 / len(self.aux_heads))
            # multiplier = desired / actual contribution ratio
            multiplier = target_ratio / max(current_ratio, 1e-8)
            # Clamp to prevent explosion or over-suppression
            multiplier = min(max(multiplier, 1.0 / self.max_weight_cap), self.max_weight_cap)
            relative[name] = multiplier

        return relative

    # ──────────────────────────────────────────────────────────────────
    #  Main entry point
    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        action_loss: torch.Tensor,
        hidden_states: torch.Tensor,
        batch: dict,
        masks: Dict[str, torch.Tensor],
        global_step: int = 0,
    ):
        """Compute weighted aux loss and return (total_loss, metrics_dict).

        Steps:
          1. Run each head's compute_loss() to get raw losses.
          2. Update EMA with observed raw losses.
          3. Compute global annealing multiplier.
          4. Compute relative balancing multipliers.
          5. Apply effective weight = global_mult × relative_mult × base_weight.
          6. Accumulate into aux_total and update log_metrics.

        Args:
            action_loss: Scalar action loss tensor (not modified by aux losses
                         within this call — caller adds the returned aux_total).
            hidden_states: LLM hidden states (B, L, H).
            batch: Collated batch dict (must contain 'input_ids').
            masks: Per-head validity masks {head_name: BoolTensor(B)}.
            global_step: Current training step.

        Returns:
            total_loss: action_loss + aux_total (scalar Tensor).
            metrics: Dict with full breakdown of aux loss components.
        """
        step = global_step
        self.step_count.fill_(step)

        # ── 1. Compute raw losses for each head ──────────────────────
        raw_losses: Dict[str, float] = {}
        head_outputs = {}
        for name, head in self.aux_heads.items():
            mask = masks.get(name, torch.ones(hidden_states.shape[0],
                                              dtype=torch.bool,
                                              device=hidden_states.device))
            out = head.compute_loss(hidden_states, batch, mask=mask)
            head_outputs[name] = out
            # Collect raw loss for EMA (use weighted as raw if head exposes no raw)
            raw_val = 0.0
            if out.loss is not None:
                raw_metrics = out.metrics
                # Prefer head's own raw metric (e.g. "recon_loss", "depth_loss")
                for k, v in raw_metrics.items():
                    if "loss" in k:
                        raw_val = float(v) if not isinstance(v, torch.Tensor) else v.item()
                        break
                if raw_val == 0.0:
                    # Fallback: unweight by base weight to recover raw magnitude
                    bw = self.base_weights.get(name, 1.0)
                    raw_val = out.loss.item() / max(bw, 1e-8)
            raw_losses[name] = raw_val

        # ── 2. Update EMA ──────────────────────────────────────────────
        self._update_ema(raw_losses)

        # ── 3 & 4. Compute multipliers ─────────────────────────────────
        global_mult    = self._compute_global_multiplier(step)
        relative_mults = self._compute_relative_weights(step)

        # ── 5 & 6. Apply effective weights and accumulate ─────────────
        aux_total = action_loss.new_zeros(())
        metrics: Dict[str, float | torch.Tensor] = {}

        for name, out in head_outputs.items():
            # Forward aux head sub-metrics (raw loss, valid_ratio, etc.)
            for mk, mv in out.metrics.items():
                metrics[f"loss/{name}_{mk}"] = mv

            if out.loss is None:
                continue

            base_w    = self.base_weights.get(name, 1.0)
            rel_mult  = relative_mults.get(name, 1.0)
            eff_w     = global_mult * rel_mult * base_w

            weighted = eff_w * (out.loss / max(base_w, 1e-8))   # re-weight from raw
            aux_total = aux_total + weighted

            metrics[f"loss/{name}_eff_weight"]   = eff_w
            metrics[f"loss/{name}_contribution"] = weighted.detach()

        aux_total_val = aux_total.detach()
        metrics["loss/aux_total"]        = aux_total_val
        metrics["loss/global_aux_mult"]  = global_mult
        metrics["loss/action_loss_fm"]   = action_loss.detach()

        return action_loss + aux_total, metrics

    # ──────────────────────────────────────────────────────────────────
    #  Factory: build from YAML config + existing aux_heads dict
    # ──────────────────────────────────────────────────────────────────
    @classmethod
    def from_config(
        cls,
        aux_heads: Dict[str, "nn.Module"],
        config,               # omegaconf DictConfig or plain dict
    ) -> "DynamicAuxSuite":
        """Construct DynamicAuxSuite from the trainer/framework config.

        Expected YAML structure (under framework.dynamic_aux):

            framework:
              dynamic_aux:
                total_steps: 100000          # must match trainer.total_steps
                warmup_steps: 5000           # optional, default 5% of total
                annealing_start_ratio: 0.6   # plateau ends at 60% of training
                annealing_end_ratio:   1.0   # weights reach 0 at 100%
                enable_ema_balancing:  true
                ema_momentum:          0.99
                balance_warmup_steps:  1000
                min_weight_floor:      0.0   # 0.0 = full fade-out
                max_weight_cap:        5.0
                base_weights:          # override per-head static weight
                  recon:   0.1
                  depth:   1.0         # boosted to compensate raw magnitude
                  grounding: 0.5
                target_ratios:         # desired % contribution of each head
                  recon:   0.35
                  depth:   0.25
                  grounding: 0.20
                  affordance: 0.10
                  future:  0.10
        """
        # Support both omegaconf DictConfig and plain dict
        try:
            from omegaconf import OmegaConf
            if hasattr(config, "framework"):
                dyn_cfg = OmegaConf.to_container(
                    getattr(config.framework, "dynamic_aux", {}),
                    resolve=True,
                ) or {}
            else:
                dyn_cfg = {}
        except Exception:
            dyn_cfg = {}

        total_steps = int(dyn_cfg.get("total_steps", 100_000))

        return cls(
            aux_heads=aux_heads,
            total_steps=total_steps,
            base_weights=dyn_cfg.get("base_weights", None),
            warmup_steps=dyn_cfg.get("warmup_steps", None),
            annealing_start_ratio=float(dyn_cfg.get("annealing_start_ratio", 0.6)),
            annealing_end_ratio=float(dyn_cfg.get("annealing_end_ratio", 1.0)),
            enable_ema_balancing=bool(dyn_cfg.get("enable_ema_balancing", True)),
            ema_momentum=float(dyn_cfg.get("ema_momentum", 0.99)),
            balance_warmup_steps=int(dyn_cfg.get("balance_warmup_steps", 1000)),
            target_ratios=dyn_cfg.get("target_ratios", None),
            min_weight_floor=float(dyn_cfg.get("min_weight_floor", 0.0)),
            max_weight_cap=float(dyn_cfg.get("max_weight_cap", 5.0)),
        )
