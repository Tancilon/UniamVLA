# UamVLA Auxiliary Denoising Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the paper-ready UamVLA auxiliary denoising suite described in `docs/superpowers/specs/2026-05-21-uamvla-aux-denoising-suite-design.md`.

**Architecture:** Keep `UamVLAGR00T` action prediction unchanged, and insert an `AuxDenoisingSuite` after action loss computation to run and budget auxiliary losses. Add a shared `SpatialMapDenoisingHead` for depth, grounding, and affordance; add a separate `ActionConditionedFutureHead` with FiLM action modulation; extend sidecar loading and collation to expose the required batch keys.

**Tech Stack:** Python, PyTorch, OmegaConf configs, existing UamVLA aux-head interface, existing `ReconDenoiser`/DiT diffusion path, pytest.

---

## Scope Check

The spec spans model heads, framework wiring, sidecar loading, config, visualization, and preprocessing. These are coupled by one training contract: all branches must produce masked auxiliary losses and flow through the global budget. This plan keeps them in one implementation plan but commits after each independently testable layer.

## File Structure

New files:

- `starVLA/model/modules/uamvla/aux_loss_control.py`: owns `AuxDenoisingSuite`, EMA action-loss budget, warmup, and post-budget logging.
- `starVLA/model/modules/uamvla/aux_heads/spatial_map_denoising_head.py`: common 20x20 single-channel DDPM map denoising base.
- `starVLA/model/modules/uamvla/aux_heads/depth_head.py`: relative inverse depth target adapter plus map denoising wrapper.
- `starVLA/model/modules/uamvla/aux_heads/grounding_head.py`: grounding mask wrapper.
- `starVLA/model/modules/uamvla/aux_heads/affordance_head.py`: affordance heatmap wrapper.
- `starVLA/model/modules/uamvla/components/action_chunk_encoder.py`: normalized action chunk Transformer encoder.
- `starVLA/model/modules/uamvla/aux_heads/action_conditioned_future_head.py`: VAE latent denoising future head with action FiLM.
- `tests/test_aux_loss_control.py`: budget, EMA, warmup, and per-head contribution unit tests.
- `tests/test_spatial_map_denoising_head.py`: map target scaling, mask behavior, and dummy-loss tests.
- `tests/test_depth_grounding_affordance_heads.py`: target adapters and wrapper construction tests.
- `tests/test_action_conditioned_future_head.py`: action encoder, FiLM fusion, action dropout, and local future target tests.

Modified files:

- `starVLA/model/modules/uamvla/aux_heads/__init__.py`: export new heads.
- `starVLA/model/modules/uamvla/collator_helpers.py`: stack string metadata for `grounding_level` or keep it as a list in framework collation.
- `starVLA/model/framework/VLM4A/UamVLAOFT.py`: build heads, load sidecars, load action-conditioned future, collate batch keys, run suite in forward, collect visualizations.
- `starVLA/model/framework/VLM4A/UamVLAGR00T.py`: run suite in forward while keeping `predict_action()` clean.
- `starVLA/config/training/uamvla_gr00t_calvin_d.yaml`: add default disabled head configs and `aux_loss_control`.
- `starVLA/config/training/uamvla_oft_calvin_d.yaml`: mirror configs when useful for shared framework smoke tests.
- `tools/preprocess/calvin_preprocessor_lerobot.py`: write strict sidecar layout for depth, grounding, and affordance.
- `tools/preprocess/libero_preprocessor.py`: write the same strict sidecar layout for LIBERO.

---

### Task 1: Aux Loss Control Suite

**Files:**
- Create: `starVLA/model/modules/uamvla/aux_loss_control.py`
- Create: `tests/test_aux_loss_control.py`

- [ ] **Step 1: Write failing tests for warmup, EMA, cap, and contribution logs**

Add `tests/test_aux_loss_control.py`:

```python
import torch
import torch.nn as nn

from starVLA.model.modules.uamvla.aux_heads.base import HeadOutput
from starVLA.model.modules.uamvla.aux_loss_control import AuxDenoisingSuite


class _Head(nn.Module):
    def __init__(self, loss_value: float, valid_ratio: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.loss_value = loss_value
        self.valid_ratio = valid_ratio

    def compute_loss(self, hidden_states, batch, mask):
        raw = torch.tensor(self.loss_value, device=hidden_states.device)
        loss = raw * self.weight
        return HeadOutput(
            loss=loss,
            metrics={"loss_raw": float(self.loss_value), "valid_ratio": self.valid_ratio},
            predictions=None,
        )


def test_aux_suite_applies_warmup_and_ema_budget():
    heads = nn.ModuleDict({"depth": _Head(4.0, 0.5), "grounding": _Head(2.0, 1.0)})
    suite = AuxDenoisingSuite(
        heads=heads,
        aux_budget=1.0,
        warmup_steps=10,
        aux_ratio_cap=0.5,
        action_loss_ema_beta=0.0,
        eps=1.0e-8,
    )
    hidden = torch.zeros(2, 5, 3)
    batch = {}
    masks = {
        "depth": torch.tensor([True, False]),
        "grounding": torch.tensor([True, True]),
    }

    action_loss = torch.tensor(2.0)
    aux_loss, metrics = suite(
        action_loss=action_loss,
        hidden_states=hidden,
        batch=batch,
        masks=masks,
        global_step=5,
    )

    # raw aux = 6.0, cap = 1.0, budget scale = 1/6, warmup = 0.5
    assert torch.allclose(aux_loss, torch.tensor(0.5))
    assert metrics["aux_total_pre_budget"].item() == 6.0
    assert torch.allclose(metrics["aux_scale_budget"], torch.tensor(1.0 / 6.0))
    assert torch.allclose(metrics["aux_budget_warmup"], torch.tensor(0.5))
    assert torch.allclose(metrics["depth_loss_contribution_post_budget"], torch.tensor(4.0 / 12.0))
    assert torch.allclose(metrics["grounding_loss_contribution_post_budget"], torch.tensor(2.0 / 12.0))
    assert metrics["depth_valid_ratio"] == 0.5
    assert metrics["grounding_valid_ratio"] == 1.0


def test_aux_suite_disabled_returns_zero_loss():
    suite = AuxDenoisingSuite(heads=nn.ModuleDict({}), enabled=False)
    hidden = torch.zeros(1, 5, 3)
    aux_loss, metrics = suite(
        action_loss=torch.tensor(1.0),
        hidden_states=hidden,
        batch={},
        masks={},
        global_step=100,
    )
    assert aux_loss.item() == 0.0
    assert metrics == {}
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
pytest tests/test_aux_loss_control.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `starVLA.model.modules.uamvla.aux_loss_control`.

- [ ] **Step 3: Implement `AuxDenoisingSuite`**

Create `starVLA/model/modules/uamvla/aux_loss_control.py` with:

```python
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
            raw_key = f"{name}_loss_raw"
            valid_key = f"{name}_valid_ratio"
            metrics[raw_key] = self._metric_tensor(out.metrics.get("loss_raw", 0.0), device, dtype)
            metrics[valid_key] = float(out.metrics.get("valid_ratio", mask.float().mean().item()))
            metrics[f"{name}_loss_weighted_pre_budget"] = out.loss.detach()

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
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_aux_loss_control.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/modules/uamvla/aux_loss_control.py tests/test_aux_loss_control.py
git commit -m "feat(uamvla): add aux denoising loss control"
```

---

### Task 2: Collation Contract For New Aux Fields

**Files:**
- Modify: `starVLA/model/modules/uamvla/collator_helpers.py`
- Modify: `tests/test_collator_helpers.py`

- [ ] **Step 1: Write failing tests for tensor fields and `grounding_level` metadata**

Append to `tests/test_collator_helpers.py`:

```python
from starVLA.model.modules.uamvla.collator_helpers import stack_optional_string_fields


def test_stack_optional_tensor_fields_new_aux_maps():
    depth = torch.ones(1, 6, 6)
    grounding = torch.full((1, 20, 20), 0.25)
    affordance = torch.full((1, 20, 20), 0.75)
    action_future = torch.zeros(3, 64, 64)
    samples = [
        {
            "depth_target": depth,
            "grounding_mask": grounding,
            "affordance_heatmap": affordance,
            "image_action_future": action_future,
        },
        {"grounding_mask": grounding * 2},
    ]

    out = stack_optional_tensor_fields(
        samples,
        ["depth_target", "grounding_mask", "affordance_heatmap", "image_action_future"],
    )

    assert out["depth_target"].shape == (2, 1, 6, 6)
    assert torch.equal(out["depth_target_mask"], torch.tensor([True, False]))
    assert out["grounding_mask"].shape == (2, 1, 20, 20)
    assert torch.equal(out["grounding_mask_mask"], torch.tensor([True, True]))
    assert out["affordance_heatmap"].shape == (2, 1, 20, 20)
    assert torch.equal(out["affordance_heatmap_mask"], torch.tensor([True, False]))
    assert out["image_action_future"].shape == (2, 3, 64, 64)
    assert torch.equal(out["image_action_future_mask"], torch.tensor([True, False]))


def test_stack_optional_string_fields_for_grounding_level():
    out = stack_optional_string_fields(
        [{"grounding_level": "part"}, {}, {"grounding_level": "object"}],
        ["grounding_level"],
    )
    assert out["grounding_level"] == ["part", "", "object"]
    assert torch.equal(out["grounding_level_mask"], torch.tensor([True, False, True]))
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
pytest tests/test_collator_helpers.py::test_stack_optional_tensor_fields_new_aux_maps tests/test_collator_helpers.py::test_stack_optional_string_fields_for_grounding_level -q
```

Expected: FAIL with `ImportError` for `stack_optional_string_fields`.

- [ ] **Step 3: Implement string metadata helper**

Add to `starVLA/model/modules/uamvla/collator_helpers.py`:

```python
def stack_optional_string_fields(
    samples: Sequence[dict], field_names: Iterable[str]
) -> dict:
    """Stack optional string metadata fields as lists plus boolean masks."""
    out: dict = {}
    for field in field_names:
        if not any(field in s for s in samples):
            continue
        values = []
        mask = []
        for sample in samples:
            if field in sample:
                values.append(str(sample[field]))
                mask.append(True)
            else:
                values.append("")
                mask.append(False)
        out[field] = values
        out[f"{field}_mask"] = torch.tensor(mask, dtype=torch.bool)
    return out
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_collator_helpers.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/modules/uamvla/collator_helpers.py tests/test_collator_helpers.py
git commit -m "feat(uamvla): collate new aux sidecar fields"
```

---

### Task 3: Spatial Map Denoising Base

**Files:**
- Create: `starVLA/model/modules/uamvla/aux_heads/spatial_map_denoising_head.py`
- Modify: `starVLA/model/modules/uamvla/aux_heads/__init__.py`
- Create: `tests/test_spatial_map_denoising_head.py`

- [ ] **Step 1: Write failing tests for range conversion, mask behavior, and valid ratio**

Add `tests/test_spatial_map_denoising_head.py`:

```python
import torch
import torch.nn as nn

from starVLA.model.modules.uamvla.aux_heads.spatial_map_denoising_head import (
    SpatialMapDenoisingHead,
)


class _FakeDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.last_z_shape = None
        self.last_target = None

    def forward(self, z, target):
        self.last_z_shape = tuple(z.shape)
        self.last_target = target.detach().clone()
        return ((target * self.weight) ** 2).mean(dim=(1, 2, 3))

    def sample(self, z):
        return torch.zeros(z.shape[0], 1, z.shape[-2], z.shape[-1], device=z.device)


def test_spatial_map_head_scales_target_and_reports_metrics():
    head = SpatialMapDenoisingHead(
        hidden_size=4,
        image_token_id=99,
        patches_per_view=4,
        target_key="depth_target",
        mask_key="depth_mask",
        metric_prefix="depth",
        loss_weight=0.5,
        denoiser=_FakeDenoiser(),
    )
    hidden = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    input_ids = torch.full((2, 4), 99, dtype=torch.long)
    batch = {
        "input_ids": input_ids,
        "depth_target": torch.tensor(
            [
                [[[0.0, 0.5], [1.0, 0.25]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        ),
    }
    mask = torch.tensor([True, False])

    out = head.compute_loss(hidden, batch, mask)

    expected_target = torch.tensor([[[[-1.0, 0.0], [1.0, -0.5]]]])
    assert torch.equal(head.denoiser.last_target, expected_target)
    assert head.denoiser.last_z_shape == (1, 4, 2, 2)
    assert out.loss is not None
    assert out.metrics["loss_raw"] > 0.0
    assert out.metrics["valid_ratio"] == 0.5


def test_spatial_map_head_empty_mask_returns_dummy_loss():
    head = SpatialMapDenoisingHead(
        hidden_size=4,
        image_token_id=99,
        patches_per_view=4,
        target_key="grounding_mask",
        mask_key="grounding_mask_mask",
        metric_prefix="grounding",
        loss_weight=0.1,
        denoiser=_FakeDenoiser(),
    )
    hidden = torch.zeros(2, 4, 4)
    batch = {
        "input_ids": torch.full((2, 4), 99, dtype=torch.long),
        "grounding_mask": torch.zeros(2, 1, 2, 2),
    }
    out = head.compute_loss(hidden, batch, torch.tensor([False, False]))
    assert out.loss is not None
    assert out.loss.item() == 0.0
    assert out.metrics["loss_raw"] == 0.0
    assert out.metrics["valid_ratio"] == 0.0
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
pytest tests/test_spatial_map_denoising_head.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `spatial_map_denoising_head`.

- [ ] **Step 3: Implement `SpatialMapDenoisingHead`**

Create `starVLA/model/modules/uamvla/aux_heads/spatial_map_denoising_head.py` with a class that:

- subclasses `AuxHead`;
- accepts `hidden_size`, `image_token_id`, `patches_per_view`, `target_key`, `mask_key`, `metric_prefix`, `view_idx`, `target_size`, `loss_weight`, `denoiser_depth`, `denoiser_embed_dim`, `gen_timesteps`, and test-only `denoiser`;
- uses `slice_image_tokens` and `einops.rearrange` like `ReconHead`;
- resizes targets to `target_size` with bilinear interpolation when needed;
- clamps target to `[0, 1]`;
- maps target to `[-1, 1]`;
- returns `HeadOutput(loss=loss_weight * raw_loss, metrics={"loss_raw": float(raw_loss.detach().item()), "valid_ratio": float(mask.float().mean().item())}, predictions=None)`;
- implements `predict()` by sampling and mapping `[-1, 1]` back to `[0, 1]`.

The constructor must create `ReconDenoiser(x_channel=1, z_channel=hidden_size, embed_dim=denoiser_embed_dim, depth=denoiser_depth, n_patches=target_size * target_size, timesteps=gen_timesteps)` when `denoiser` is not passed.

- [ ] **Step 4: Export class**

Update `starVLA/model/modules/uamvla/aux_heads/__init__.py`:

```python
from starVLA.model.modules.uamvla.aux_heads.spatial_map_denoising_head import SpatialMapDenoisingHead
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_spatial_map_denoising_head.py tests/test_denoiser_smoke.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add starVLA/model/modules/uamvla/aux_heads/spatial_map_denoising_head.py starVLA/model/modules/uamvla/aux_heads/__init__.py tests/test_spatial_map_denoising_head.py
git commit -m "feat(uamvla): add spatial map denoising head"
```

---

### Task 4: Depth, Grounding, And Affordance Heads

**Files:**
- Create: `starVLA/model/modules/uamvla/aux_heads/depth_head.py`
- Create: `starVLA/model/modules/uamvla/aux_heads/grounding_head.py`
- Create: `starVLA/model/modules/uamvla/aux_heads/affordance_head.py`
- Modify: `starVLA/model/modules/uamvla/aux_heads/__init__.py`
- Create: `tests/test_depth_grounding_affordance_heads.py`

- [ ] **Step 1: Write failing tests for depth normalization and wrapper keys**

Add `tests/test_depth_grounding_affordance_heads.py`:

```python
import torch

from starVLA.model.modules.uamvla.aux_heads.depth_head import DepthDenoisingHead
from starVLA.model.modules.uamvla.aux_heads.grounding_head import GroundingMaskDenoisingHead
from starVLA.model.modules.uamvla.aux_heads.affordance_head import AffordanceHeatmapDenoisingHead


def test_depth_relative_inverse_depth_per_frame():
    depth = torch.tensor(
        [[[[1.0, 2.0], [4.0, 0.0]]]],
        dtype=torch.float32,
    )
    out = DepthDenoisingHead.relative_inverse_depth_per_frame(depth)
    assert out.shape == (1, 1, 2, 2)
    assert out[0, 0, 0, 0] > out[0, 0, 0, 1]
    assert out[0, 0, 0, 1] > out[0, 0, 1, 0]
    assert out[0, 0, 1, 1] == 0.0
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_wrapper_target_keys_are_stable():
    common = dict(hidden_size=4, image_token_id=99, patches_per_view=4, denoiser=None)
    depth = DepthDenoisingHead(**common)
    grounding = GroundingMaskDenoisingHead(**common)
    affordance = AffordanceHeatmapDenoisingHead(**common)
    assert depth.target_key == "depth_target"
    assert depth.mask_key == "depth_mask"
    assert grounding.target_key == "grounding_mask"
    assert grounding.mask_key == "grounding_mask_mask"
    assert affordance.target_key == "affordance_heatmap"
    assert affordance.mask_key == "affordance_mask"
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
pytest tests/test_depth_grounding_affordance_heads.py -q
```

Expected: FAIL with `ModuleNotFoundError` for the new head modules.

- [ ] **Step 3: Implement wrapper heads**

Implementation contract:

- `DepthDenoisingHead` subclasses `SpatialMapDenoisingHead`, sets `target_key="depth_target"`, `mask_key="depth_mask"`, `metric_prefix="depth"`, and overrides target preparation to call `relative_inverse_depth_per_frame()` before `[0, 1] -> [-1, 1]`.
- `GroundingMaskDenoisingHead` subclasses `SpatialMapDenoisingHead`, sets `target_key="grounding_mask"`, `mask_key="grounding_mask_mask"`, and `metric_prefix="grounding"`.
- `AffordanceHeatmapDenoisingHead` subclasses `SpatialMapDenoisingHead`, sets `target_key="affordance_heatmap"`, `mask_key="affordance_mask"`, and `metric_prefix="affordance"`.

Use this exact depth helper:

```python
@staticmethod
def relative_inverse_depth_per_frame(
    depth: torch.Tensor,
    eps: float = 1.0e-6,
    max_depth: float = 10.0,
) -> torch.Tensor:
    depth = depth.float()
    out = torch.zeros_like(depth)
    for i in range(depth.shape[0]):
        d = depth[i]
        valid = torch.isfinite(d) & (d > 0)
        if not valid.any():
            continue
        inv = torch.zeros_like(d)
        inv[valid] = 1.0 / torch.clamp(d[valid], min=eps, max=max_depth)
        vals = inv[valid]
        lo = torch.quantile(vals, 0.01)
        hi = torch.quantile(vals, 0.99)
        denom = torch.clamp(hi - lo, min=eps)
        rel = torch.clamp((inv - lo) / denom, 0.0, 1.0)
        rel = torch.where(valid, rel, torch.zeros_like(rel))
        out[i] = rel
    return out
```

- [ ] **Step 4: Export wrapper heads**

Update `starVLA/model/modules/uamvla/aux_heads/__init__.py`:

```python
from starVLA.model.modules.uamvla.aux_heads.depth_head import DepthDenoisingHead
from starVLA.model.modules.uamvla.aux_heads.grounding_head import GroundingMaskDenoisingHead
from starVLA.model.modules.uamvla.aux_heads.affordance_head import AffordanceHeatmapDenoisingHead
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_depth_grounding_affordance_heads.py tests/test_spatial_map_denoising_head.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add starVLA/model/modules/uamvla/aux_heads/depth_head.py starVLA/model/modules/uamvla/aux_heads/grounding_head.py starVLA/model/modules/uamvla/aux_heads/affordance_head.py starVLA/model/modules/uamvla/aux_heads/__init__.py tests/test_depth_grounding_affordance_heads.py
git commit -m "feat(uamvla): add spatial auxiliary map heads"
```

---

### Task 5: Action Chunk Encoder

**Files:**
- Create: `starVLA/model/modules/uamvla/components/action_chunk_encoder.py`
- Create: `tests/test_action_conditioned_future_head.py`

- [ ] **Step 1: Write failing tests for encoder shape**

Add the first test to `tests/test_action_conditioned_future_head.py`:

```python
import torch

from starVLA.model.modules.uamvla.components.action_chunk_encoder import ActionChunkEncoder


def test_action_chunk_encoder_outputs_hidden_size():
    encoder = ActionChunkEncoder(
        action_dim=7,
        hidden_size=16,
        action_embed_dim=8,
        num_layers=1,
        num_heads=2,
        max_horizon=8,
    )
    actions = torch.randn(3, 8, 7)
    out = encoder(actions)
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run failing test**

Run:

```bash
pytest tests/test_action_conditioned_future_head.py::test_action_chunk_encoder_outputs_hidden_size -q
```

Expected: FAIL with `ModuleNotFoundError` for `action_chunk_encoder`.

- [ ] **Step 3: Implement `ActionChunkEncoder`**

Create `starVLA/model/modules/uamvla/components/action_chunk_encoder.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn


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
        self.pos_embed = nn.Parameter(torch.zeros(1, max_horizon, action_embed_dim))
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
```

- [ ] **Step 4: Run test**

Run:

```bash
pytest tests/test_action_conditioned_future_head.py::test_action_chunk_encoder_outputs_hidden_size -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/modules/uamvla/components/action_chunk_encoder.py tests/test_action_conditioned_future_head.py
git commit -m "feat(uamvla): add action chunk encoder"
```

---

### Task 6: Action-Conditioned Future Head

**Files:**
- Create: `starVLA/model/modules/uamvla/aux_heads/action_conditioned_future_head.py`
- Modify: `starVLA/model/modules/uamvla/aux_heads/__init__.py`
- Modify: `tests/test_action_conditioned_future_head.py`

- [ ] **Step 1: Add failing tests for FiLM and action dropout**

Append to `tests/test_action_conditioned_future_head.py`:

```python
import torch.nn as nn

from starVLA.model.modules.uamvla.aux_heads.action_conditioned_future_head import (
    ActionConditionedFutureHead,
)


class _FakeVAE(nn.Module):
    latent_channels = 2
    scaling_factor = 1.0
    shift_factor = 0.0

    def encode(self, images):
        class _Posterior:
            def __init__(self, x):
                self.x = x

            def sample(self):
                return self.x[:, :2, ::8, ::8]

        return type("Encoded", (), {"latent_dist": _Posterior(images)})()

    def decode(self, latents):
        return torch.zeros(latents.shape[0], 3, latents.shape[-2] * 8, latents.shape[-1] * 8)


class _FakeDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.last_z = None

    def forward(self, z, target):
        self.last_z = z.detach().clone()
        return (target * self.weight).pow(2).mean(dim=(1, 2, 3))

    def sample(self, z):
        return torch.zeros(z.shape[0], 8, z.shape[-2], z.shape[-1], device=z.device)


def test_action_conditioned_future_uses_action_film_condition():
    head = ActionConditionedFutureHead(
        hidden_size=4,
        vae=_FakeVAE(),
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        image_token_id=99,
        patches_per_view=4,
        action_dim=7,
        action_horizon=8,
        target_resize=32,
        action_encoder_layers=1,
        action_embed_dim=8,
        action_encoder_heads=2,
        action_dropout=0.0,
        denoiser=_FakeDenoiser(),
    )
    hidden = torch.randn(2, 4, 4)
    batch = {
        "input_ids": torch.full((2, 4), 99, dtype=torch.long),
        "action": torch.randn(2, 8, 7),
        "image_action_future": torch.rand(2, 3, 32, 32),
    }
    mask = torch.tensor([True, True])
    out = head.compute_loss(hidden, batch, mask)
    assert out.loss is not None
    assert out.metrics["valid_ratio"] == 1.0
    assert head.denoiser.last_z.shape == (2, 4, 2, 2)


def test_action_conditioned_future_empty_mask_returns_dummy_loss():
    head = ActionConditionedFutureHead(
        hidden_size=4,
        vae=_FakeVAE(),
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        image_token_id=99,
        patches_per_view=4,
        action_dim=7,
        action_horizon=8,
        target_resize=32,
        action_encoder_layers=1,
        action_embed_dim=8,
        action_encoder_heads=2,
        denoiser=_FakeDenoiser(),
    )
    out = head.compute_loss(
        torch.zeros(2, 4, 4),
        {
            "input_ids": torch.full((2, 4), 99, dtype=torch.long),
            "action": torch.zeros(2, 8, 7),
            "image_action_future": torch.zeros(2, 3, 32, 32),
        },
        torch.tensor([False, False]),
    )
    assert out.loss is not None
    assert out.loss.item() == 0.0
    assert out.metrics["loss_raw"] == 0.0
    assert out.metrics["valid_ratio"] == 0.0
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
pytest tests/test_action_conditioned_future_head.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `action_conditioned_future_head`.

- [ ] **Step 3: Implement `ActionConditionedFutureHead`**

Implementation contract:

- subclass `FutureHead` where practical or copy its VAE normalization/latent conversion helpers exactly;
- add an `ActionChunkEncoder`;
- add `film = nn.Linear(hidden_size, 2 * hidden_size)`;
- in training, filter by mask first;
- encode `batch["action"][mask][:, -action_horizon:, :]`;
- if `self.training` and `action_dropout > 0`, zero action embeddings with per-sample Bernoulli mask;
- compute `fused_cond = spatial_cond * (1 + scale) + shift`;
- denoise `batch["image_action_future"][mask]`;
- return `metrics={"loss_raw": raw_loss.item(), "valid_ratio": mask.float().mean().item()}`.

- [ ] **Step 4: Export head**

Update `starVLA/model/modules/uamvla/aux_heads/__init__.py`:

```python
from starVLA.model.modules.uamvla.aux_heads.action_conditioned_future_head import ActionConditionedFutureHead
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_action_conditioned_future_head.py tests/test_aux_head_normalize_for_vae.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add starVLA/model/modules/uamvla/aux_heads/action_conditioned_future_head.py starVLA/model/modules/uamvla/aux_heads/__init__.py tests/test_action_conditioned_future_head.py
git commit -m "feat(uamvla): add action-conditioned future head"
```

---

### Task 7: Framework Head Construction And Config

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLAOFT.py`
- Modify: `starVLA/config/training/uamvla_gr00t_calvin_d.yaml`
- Modify: `starVLA/config/training/uamvla_oft_calvin_d.yaml`
- Modify: `tests/framework/test_uamvla_gr00t_static.py`
- Modify: `tests/test_uamvla_oft_config.py`

- [ ] **Step 1: Write config tests**

Append to `tests/framework/test_uamvla_gr00t_static.py`:

```python
def test_uamvla_gr00t_config_defines_aux_loss_control_and_new_heads():
    cfg = yaml.safe_load(Path("starVLA/config/training/uamvla_gr00t_calvin_d.yaml").read_text())
    assert cfg["framework"]["aux_loss_control"]["enabled"] is True
    assert cfg["framework"]["aux_loss_control"]["aux_ratio_cap"] == 0.5
    heads = cfg["framework"]["aux_heads"]
    for name in ["depth", "action_conditioned_future", "grounding", "affordance"]:
        assert name in heads
        assert heads[name]["enabled"] is False
```

- [ ] **Step 2: Run failing config test**

Run:

```bash
pytest tests/framework/test_uamvla_gr00t_static.py::test_uamvla_gr00t_config_defines_aux_loss_control_and_new_heads -q
```

Expected: FAIL with `KeyError` for `aux_loss_control` or missing head keys.

- [ ] **Step 3: Add config entries**

Add this to `framework` in `starVLA/config/training/uamvla_gr00t_calvin_d.yaml` and mirror the same block into `starVLA/config/training/uamvla_oft_calvin_d.yaml`:

```yaml
  aux_loss_control:
    enabled: true
    aux_budget: 1.0
    warmup_steps: 2000
    aux_ratio_cap: 0.5
    action_loss_ema_beta: 0.99
    eps: 1.0e-8
```

Add these under `framework.aux_heads`:

```yaml
    depth:
      enabled: false
      loss_weight: 0.1
      view_idx: 0
      target_kind: relative_inverse_depth_per_frame
      target_size: 20
      denoiser_depth: 3
      denoiser_embed_dim: 512
      gen_timesteps: "1000"
    action_conditioned_future:
      enabled: false
      loss_weight: 1.0
      view_idx: 0
      target_resize: 320
      denoiser_depth: 3
      denoiser_embed_dim: 1024
      repeat_factor: 4
      action_encoder_layers: 2
      action_embed_dim: 512
      action_encoder_heads: 8
      action_dropout: 0.1
      target_frame_offset: future_action_window_size
    grounding:
      enabled: false
      loss_weight: 0.1
      view_idx: 0
      target_size: 20
      denoiser_depth: 3
      denoiser_embed_dim: 512
    affordance:
      enabled: false
      loss_weight: 0.1
      view_idx: 0
      target_size: 20
      heatmap_sigma: 2.0
      denoiser_depth: 3
      denoiser_embed_dim: 512
```

- [ ] **Step 4: Wire head construction**

Modify `UamVLAOFT._maybe_build_aux_heads()`:

- construct `DepthDenoisingHead`, `GroundingMaskDenoisingHead`, and `AffordanceHeatmapDenoisingHead` with the same `vision_extra` fields used by `ReconHead`;
- construct `ActionConditionedFutureHead` with shared VAE, `action_dim`, and `action_horizon`;
- filter config keys `enabled` and `lr` before passing kwargs.

- [ ] **Step 5: Run config tests**

Run:

```bash
pytest tests/framework/test_uamvla_gr00t_static.py::test_uamvla_gr00t_config_defines_aux_loss_control_and_new_heads tests/test_uamvla_oft_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLAOFT.py starVLA/config/training/uamvla_gr00t_calvin_d.yaml starVLA/config/training/uamvla_oft_calvin_d.yaml tests/framework/test_uamvla_gr00t_static.py tests/test_uamvla_oft_config.py
git commit -m "feat(uamvla): configure new auxiliary heads"
```

---

### Task 8: Sidecar Loading And Batch Collation

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLAOFT.py`
- Modify: `tests/framework/test_uamvla_oft_future_video_path.py`
- Create: `tests/framework/test_uamvla_aux_sidecars.py`

- [ ] **Step 1: Write failing tests for sidecar paths and batch masks**

Add `tests/framework/test_uamvla_aux_sidecars.py` with a lightweight object created by `object.__new__(UamVLAOFT)` and temp files:

```python
from pathlib import Path

import numpy as np
import torch

from starVLA.model.framework.VLM4A.UamVLAOFT import UamVLAOFT


def test_aux_sidecar_paths_and_collate_masks(tmp_path):
    model = object.__new__(UamVLAOFT)
    model.sidecar_root = tmp_path
    model.aux_state_slice = {
        "target_pose_rot6d": [15, 21],
        "target_pose_trans": [21, 24],
        "static_cam_rot6d": [24, 30],
        "static_cam_trans": [30, 33],
    }
    model._image_target_cache = {}
    model._load_image_future = lambda traj, base: None
    model._load_image_action_future = lambda traj, base: torch.ones(3, 8, 8)

    for subdir, value in [
        ("depths/static/3", np.ones((4, 4), dtype=np.float32)),
        ("grounding_masks/static/3", np.ones((1, 20, 20), dtype=np.float32)),
        ("affordance_heatmaps/static/3", np.ones((1, 20, 20), dtype=np.float32)),
    ]:
        path = tmp_path / subdir
        path.mkdir(parents=True)
        np.save(path / "7.npy", value)

    sample = {
        "__trajectory_id": 3,
        "__base_index": 7,
        "image": [],
        "lang": "open drawer",
        "action": np.zeros((8, 7), dtype=np.float32),
        "state": np.zeros((1, 33), dtype=np.float32),
    }
    out = UamVLAOFT._unpack_lerobot_sample(model, sample)
    assert out["depth_target"].shape == (1, 4, 4)
    assert out["grounding_mask"].shape == (1, 20, 20)
    assert out["grounding_level"] == "object"
    assert out["affordance_heatmap"].shape == (1, 20, 20)
    assert out["image_action_future"].shape == (3, 8, 8)

    batch = UamVLAOFT._collate_aux(
        model,
        [out, {"action": torch.zeros(8, 7)}],
        {"input_ids": torch.zeros(2, 4, dtype=torch.long)},
    )
    assert torch.equal(batch["depth_mask"], torch.tensor([True, False]))
    assert torch.equal(batch["grounding_mask_mask"], torch.tensor([True, False]))
    assert torch.equal(batch["affordance_mask"], torch.tensor([True, False]))
    assert torch.equal(batch["action_conditioned_future_mask"], torch.tensor([True, False]))
    assert batch["grounding_level"] == ["object", ""]
```

- [ ] **Step 2: Run failing test**

Run:

```bash
pytest tests/framework/test_uamvla_aux_sidecars.py -q
```

Expected: FAIL because `_load_image_action_future` and new sidecar fields are absent.

- [ ] **Step 3: Implement sidecar loaders**

In `UamVLAOFT`, add helpers:

- `_load_npy_sidecar(path: Path, channel_first: bool = True) -> torch.Tensor | None`;
- `_load_depth_target(traj, base)`;
- `_load_grounding_mask(traj, base)` returns tensor and level, using `"object"` when no metadata file exists;
- `_load_affordance_heatmap(traj, base)`;
- `_image_action_future_frame_index(base_index)` returns `base_index + int(self.config.framework.action_model.future_action_window_size)`;
- `_load_image_action_future(traj, base)`.

Use the strict paths:

```text
depths/static/<traj>/<base>.npy
grounding_masks/static/<traj>/<base>.npy
affordance_heatmaps/static/<traj>/<base>.npy
```

- [ ] **Step 4: Extend `_unpack_lerobot_sample()`**

After existing `image_future` loading, add:

```python
depth = self._load_depth_target(traj, base)
if depth is not None:
    out["depth_target"] = depth

grounding = self._load_grounding_mask(traj, base)
if grounding is not None:
    out["grounding_mask"] = grounding["mask"]
    out["grounding_level"] = grounding["level"]

affordance = self._load_affordance_heatmap(traj, base)
if affordance is not None:
    out["affordance_heatmap"] = affordance

image_action_future = self._load_image_action_future(traj, base)
if image_action_future is not None:
    out["image_action_future"] = image_action_future
```

- [ ] **Step 5: Extend `_collate_aux()`**

Add new tensor fields to `stack_optional_tensor_fields`:

```python
[
    "image_target",
    "image_future",
    "point_cloud",
    "depth_target",
    "grounding_mask",
    "affordance_heatmap",
    "image_action_future",
    "action",
]
```

Add mask aliases:

```python
if "depth_target_mask" in batch_dict:
    batch_dict["depth_mask"] = batch_dict["depth_target_mask"]
if "affordance_heatmap_mask" in batch_dict:
    batch_dict["affordance_mask"] = batch_dict["affordance_heatmap_mask"]
if "image_action_future_mask" in batch_dict:
    batch_dict["action_conditioned_future_mask"] = batch_dict["image_action_future_mask"]
```

Add string metadata:

```python
batch_dict.update(stack_optional_string_fields(examples, ["grounding_level"]))
```

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/framework/test_uamvla_aux_sidecars.py tests/framework/test_uamvla_oft_future_video_path.py tests/test_collator_helpers.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLAOFT.py tests/framework/test_uamvla_aux_sidecars.py tests/framework/test_uamvla_oft_future_video_path.py
git commit -m "feat(uamvla): load auxiliary denoising sidecars"
```

---

### Task 9: Integrate AuxDenoisingSuite In Framework Forward Paths

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLAOFT.py`
- Modify: `starVLA/model/framework/VLM4A/UamVLAGR00T.py`
- Modify: `tests/framework/test_uamvla_gr00t_static.py`

- [ ] **Step 1: Write failing static test that suite receives action loss**

Append to `tests/framework/test_uamvla_gr00t_static.py`:

```python
def test_uamvla_gr00t_forward_uses_aux_suite_when_present(monkeypatch):
    module = _load_uamvla_gr00t_module(monkeypatch)
    monkeypatch.setattr(module.torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())

    class _FakeQwenInterface:
        image_token_id = 99

        def build_qwenvl_inputs(self, images, instructions):
            return {"input_ids": torch.full((len(images), 400), 99, dtype=torch.long)}

        def __call__(self, **kwargs):
            hidden = torch.ones((kwargs["input_ids"].shape[0], 400, 16), dtype=torch.float32)
            return types.SimpleNamespace(hidden_states=[hidden])

    class _FakeActionModel:
        def __call__(self, vl_embs, actions, state):
            return torch.tensor(2.0)

    class _FakeSuite:
        def __init__(self):
            self.calls = []

        def __call__(self, action_loss, hidden_states, batch, masks, global_step=0):
            self.calls.append((action_loss, hidden_states, batch, masks, global_step))
            return torch.tensor(0.25), {"aux_total_post_budget": torch.tensor(0.25)}

    model = object.__new__(module.UamVLAGR00T)
    model.qwen_vl_interface = _FakeQwenInterface()
    model.action_model = _FakeActionModel()
    model.action_horizon = 8
    model.aux_heads = {}
    model.aux_suite = _FakeSuite()
    model.config = _AttrDict(framework=_AttrDict(action_model=_AttrDict(repeated_diffusion_steps=1)))

    sample = {
        "image": [object()],
        "lang": "open",
        "action": np.zeros((8, 7), dtype=np.float32),
    }
    out = module.UamVLAGR00T.forward(model, [sample], global_step=3)
    assert out["action_loss"].item() == 2.25
    assert out["aux_total_post_budget"].item() == 0.25
    assert model.aux_suite.calls[0][4] == 3
```

- [ ] **Step 2: Run failing test**

Run:

```bash
pytest tests/framework/test_uamvla_gr00t_static.py::test_uamvla_gr00t_forward_uses_aux_suite_when_present -q
```

Expected: FAIL because `UamVLAGR00T.forward()` uses the old per-head loop.

- [ ] **Step 3: Instantiate `AuxDenoisingSuite` after head construction**

In `UamVLAOFT.__init__()` and `UamVLAGR00T.__init__()`, after `_maybe_build_aux_heads()`:

```python
self._maybe_build_aux_loss_control()
```

Add `_maybe_build_aux_loss_control()` to `UamVLAOFT`:

```python
def _maybe_build_aux_loss_control(self) -> None:
    from starVLA.model.modules.uamvla.aux_loss_control import AuxDenoisingSuite

    cfg = self.config.framework.get("aux_loss_control", {})
    self.aux_suite = AuxDenoisingSuite(
        heads=self.aux_heads,
        enabled=cfg.get("enabled", True),
        aux_budget=cfg.get("aux_budget", 1.0),
        warmup_steps=cfg.get("warmup_steps", 2000),
        aux_ratio_cap=cfg.get("aux_ratio_cap", 0.5),
        action_loss_ema_beta=cfg.get("action_loss_ema_beta", 0.99),
        eps=cfg.get("eps", 1.0e-8),
    )
```

- [ ] **Step 4: Replace old aux loop in `forward()` methods**

In both `UamVLAOFT.forward()` and `UamVLAGR00T.forward()`:

1. Build `batch_dict`.
2. Build `masks = {name: self._resolve_head_mask(name, batch_dict, hidden.shape[0], hidden.device) for name in self.aux_heads}`.
3. Call `aux_loss, aux_metrics = self.aux_suite(action_loss=total, hidden_states=hidden, batch=batch_dict, masks=masks, global_step=int(kwargs.get("global_step", 0)))`.
4. Set `total = total + aux_loss`.
5. Merge `aux_metrics` into `log_metrics`.

Keep legacy metric names only when tests require them; otherwise prefer the new names from the spec.

- [ ] **Step 5: Run framework tests**

Run:

```bash
pytest tests/framework/test_uamvla_gr00t_static.py tests/test_aux_loss_control.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLAOFT.py starVLA/model/framework/VLM4A/UamVLAGR00T.py tests/framework/test_uamvla_gr00t_static.py
git commit -m "feat(uamvla): apply auxiliary loss budget in frameworks"
```

---

### Task 10: Visualization For New Heads

**Files:**
- Modify: `starVLA/model/modules/uamvla/aux_heads/spatial_map_denoising_head.py`
- Modify: `starVLA/model/modules/uamvla/aux_heads/action_conditioned_future_head.py`
- Modify: `tests/test_spatial_map_denoising_head.py`
- Modify: `tests/test_action_conditioned_future_head.py`

- [ ] **Step 1: Write tests that `visualize()` returns GT-vs-pred objects without crashing**

Add assertions that use `num_samples=1`, mask one valid sample, and confirm list length is one for spatial map and action-conditioned future heads. Monkeypatch `wandb.Image` by importing `wandb` only inside the method; if `wandb` is not installed, return PIL images.

- [ ] **Step 2: Implement spatial-map GT-vs-pred visualization**

Implementation contract:

- call `predict()` on valid samples;
- upsample GT and prediction to 160x160 with nearest or bilinear interpolation;
- convert single-channel maps to RGB heatmap or grayscale PIL;
- concatenate GT and Pred horizontally;
- return `wandb.Image` when wandb imports, otherwise return PIL.

- [ ] **Step 3: Implement action-conditioned future GT-vs-pred visualization**

Implementation contract:

- mirror `FutureHead.visualize()`;
- use `image_action_future` as GT;
- label the output with the instruction when `batch["instruction"]` exists;
- return one image per requested valid sample.

- [ ] **Step 4: Run visualization tests**

Run:

```bash
pytest tests/test_spatial_map_denoising_head.py tests/test_action_conditioned_future_head.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/modules/uamvla/aux_heads/spatial_map_denoising_head.py starVLA/model/modules/uamvla/aux_heads/action_conditioned_future_head.py tests/test_spatial_map_denoising_head.py tests/test_action_conditioned_future_head.py
git commit -m "feat(uamvla): visualize auxiliary denoising heads"
```

---

### Task 11: Preprocessing Sidecar Writers

**Files:**
- Modify: `tools/preprocess/calvin_preprocessor_lerobot.py`
- Modify: `tools/preprocess/libero_preprocessor.py`
- Create or modify tests under `tests/` for preprocessing sidecar paths.

- [ ] **Step 1: Add tests for strict sidecar paths**

For CALVIN and LIBERO preprocessing tests, assert that the preprocessor writes or exposes these paths:

```text
depths/static/<trajectory_id>/<base_index>.npy
grounding_masks/static/<trajectory_id>/<base_index>.npy
affordance_heatmaps/static/<trajectory_id>/<base_index>.npy
```

Use tiny fake rendered arrays:

```python
rgb = np.zeros((8, 8, 3), dtype=np.uint8)
depth = np.ones((8, 8), dtype=np.float32)
seg = np.zeros((8, 8), dtype=np.int32)
seg[2:5, 3:6] = 7
```

Expected: FAIL until writer functions exist.

- [ ] **Step 2: Implement CALVIN sidecar writer helpers**

Add helpers that:

- save raw static depth to `depths/static`;
- save target part/object mask to `grounding_masks/static`;
- save target-point-cloud affordance heatmap to `affordance_heatmaps/static`;
- create parent directories with `mkdir(parents=True, exist_ok=True)`.

Use `np.save` for all three sidecars.

- [ ] **Step 3: Implement LIBERO sidecar writer helpers**

Mirror the strict layout in `tools/preprocess/libero_preprocessor.py`. If LIBERO already stores depth under a different layout, write the new strict path in addition to the old path during this migration.

- [ ] **Step 4: Run preprocessing tests**

Run:

```bash
pytest tests/test_calvin_lerobot_merger.py tests/dataloader/test_lerobot_sidecar_passthrough.py -q
```

Expected: PASS. If these tests do not cover the new writers, add the new specific preprocessing test file to the command and rerun.

- [ ] **Step 5: Commit**

```bash
git add tools/preprocess/calvin_preprocessor_lerobot.py tools/preprocess/libero_preprocessor.py tests
git commit -m "feat(uamvla): write auxiliary denoising sidecars"
```

---

### Task 12: End-To-End Static And Smoke Verification

**Files:**
- Modify: `tests/framework/test_uamvla_oft_aux_step_assert.py`
- Modify: `tests/framework/test_uamvla_gr00t_static.py`
- Modify: `docs/superpowers/specs/2026-05-21-uamvla-aux-denoising-suite-design.md` only if implementation reveals a spec correction.

- [ ] **Step 1: Add static checks that `predict_action()` does not require aux sidecars**

Append this test to `tests/framework/test_uamvla_gr00t_static.py`:

```python
def test_uamvla_gr00t_predict_action_ignores_aux_sidecars(monkeypatch):
    module = _load_uamvla_gr00t_module(monkeypatch)
    monkeypatch.setattr(module.torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())

    class _FakeQwenInterface:
        image_token_id = 99

        def build_qwenvl_inputs(self, images, instructions):
            return {"input_ids": torch.full((len(images), 400), 99, dtype=torch.long)}

        def __call__(self, **kwargs):
            hidden = torch.ones((kwargs["input_ids"].shape[0], 400, 16), dtype=torch.float32)
            return types.SimpleNamespace(hidden_states=[hidden])

    class _FakeActionModel:
        def predict_action(self, hidden, state):
            assert hidden.shape == (1, 400, 16)
            assert state is None
            return torch.zeros(1, 8, 7)

    def _forbid_aux_unpack(self, sample):
        forbidden = {
            "depth_target",
            "grounding_mask",
            "affordance_heatmap",
            "image_action_future",
        }
        assert not forbidden.intersection(sample)
        return sample

    model = object.__new__(module.UamVLAGR00T)
    model.qwen_vl_interface = _FakeQwenInterface()
    model.action_model = _FakeActionModel()
    model.action_horizon = 8
    model.config = _AttrDict(framework=_AttrDict(action_model=_AttrDict(state_dim=0)))
    model._prepare_examples = types.MethodType(lambda self, examples: examples, model)
    model._unpack_lerobot_sample = types.MethodType(_forbid_aux_unpack, model)

    out = module.UamVLAGR00T.predict_action(
        model,
        {"image": [object()], "lang": "open drawer"},
    )
    assert out["normalized_actions"].shape == (1, 8, 7)
```

- [ ] **Step 2: Add step assertions for new heads when fixture data exists**

Append these tests to `tests/framework/test_uamvla_oft_aux_step_assert.py`:

```python
@pytest.mark.parametrize(
    "head_name,metric_key",
    [
        ("depth", "depth_loss_weighted_pre_budget"),
        ("grounding", "grounding_loss_weighted_pre_budget"),
        ("affordance", "affordance_loss_weighted_pre_budget"),
        ("action_conditioned_future", "action_conditioned_future_loss_weighted_pre_budget"),
    ],
)
def test_new_aux_head_loss_above_dummy_when_labels_exist(configured_model_and_batch, head_name, metric_key):
    model, batch = configured_model_and_batch
    if head_name not in model.aux_heads:
        pytest.skip(f"{head_name} disabled in this smoke config")
    out = model.forward(batch)
    valid_ratio_key = f"{head_name}_valid_ratio"
    if valid_ratio_key not in out or float(out[valid_ratio_key]) == 0.0:
        pytest.skip(f"{head_name} labels absent from this smoke fixture")
    assert metric_key in out
    real_loss = out[metric_key].item()
    dummy_loss = model.aux_heads[head_name].get_dummy_loss().item()
    assert real_loss > dummy_loss
```

- [ ] **Step 3: Run unit and static tests**

Run:

```bash
pytest tests/test_aux_loss_control.py tests/test_spatial_map_denoising_head.py tests/test_depth_grounding_affordance_heads.py tests/test_action_conditioned_future_head.py tests/framework/test_uamvla_gr00t_static.py tests/test_collator_helpers.py -q
```

Expected: PASS.

- [ ] **Step 4: Run existing focused framework tests**

Run:

```bash
pytest tests/framework/test_uamvla_oft_resize.py tests/framework/test_uamvla_oft_future_video_path.py tests/test_uamvla_oft_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Run GPU smoke only when a free GPU and fixture dataset are available**

Run:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu_id> pytest tests/framework/test_uamvla_oft_aux_step_assert.py -q
```

Expected: PASS or SKIP with the test's existing dataset/GPU skip reason. Do not run this on CPU.

- [ ] **Step 6: Commit**

```bash
git add tests/framework/test_uamvla_oft_aux_step_assert.py tests/framework/test_uamvla_gr00t_static.py docs/superpowers/specs/2026-05-21-uamvla-aux-denoising-suite-design.md
git commit -m "test(uamvla): verify auxiliary denoising suite integration"
```

---

## Self-Review Against Spec

Spec coverage:

- Unified framework and plug-in task variables: Tasks 1, 7, 9.
- Existing `recon`, terminal `future`, and `pose` coexistence: Tasks 7 and 9 keep existing heads.
- Dense/latent epsilon-prediction and pose score matching split: Tasks 3, 4, 6 keep map/image DDPM and avoid pose changes.
- Static view only: Tasks 3, 4, 6 use `view_idx=0`.
- 20x20 token-grid map targets: Tasks 3 and 4.
- `[0, 1] -> [-1, 1]` inside heads: Task 3.
- Per-frame robust relative inverse depth: Task 4.
- Hierarchical grounding metadata: Tasks 2 and 8.
- Target point-cloud affordance sidecars: Task 11 writes strict sidecars and Task 8 loads them.
- Terminal future plus action-conditioned future: Tasks 6, 7, 8.
- Normalized action chunk and FiLM: Tasks 5 and 6.
- Action dropout: Task 6.
- Global aux budget with EMA action loss: Tasks 1 and 9.
- Missing-label masking: Tasks 2, 8, 9.
- GT-vs-pred visualization: Task 10.
- Predict-action independence: Task 12.

Red-flag scan:

- This plan contains no deferred-work markers or undefined function names.
- Steps that modify code include exact file paths, concrete behavior, and test commands.

Type consistency:

- Head masks use `depth_mask`, `grounding_mask_mask`, `affordance_mask`, and `action_conditioned_future_mask`.
- Head metrics use `loss_raw` and `valid_ratio`; suite logs `<head>_loss_raw`, `<head>_loss_weighted_pre_budget`, `<head>_loss_contribution_post_budget`, and `<head>_valid_ratio`.
- `HeadOutput.loss` is always interpreted as the pre-budget weighted loss.
