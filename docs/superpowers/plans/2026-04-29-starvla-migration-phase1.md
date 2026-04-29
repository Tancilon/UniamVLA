# starVLA Migration Implementation Plan (Phase 1: LIBERO end-to-end)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate UamVLA's research architecture (Qwen3-VL-8B + state encoder + 4 aux heads) into the UniamVLA fork of starVLA so that LIBERO Spatial training + eval runs end-to-end with SR ≥ 60%.

**Architecture:** Faithful port of UamVLA submodules (state_encoder, aux_heads, components, data utilities) into `starVLA/model/modules/uamvla/`, plus a new `UamVLA(baseframework)` class registered with starVLA's framework registry. Custom dataloader plugin reads UamVLA's existing JSONL format (skip LeRobot conversion). Two small additive patches to starVLA mainline (`base_framework.py`, `train_starvla.py`). Eval reuses starVLA's WebSocket pattern with zero Python changes — only 3 in-place wrapper bash scripts edited.

**Tech Stack:** PyTorch 2.x, transformers ≥ 4.57, accelerate + DeepSpeed ZeRO-2, OmegaConf, Qwen3-VL-8B-Instruct, libero, robosuite, mujoco. Conda envs: `uamvla` (training/server) and `libero_env` (eval client).

**Spec:** `docs/superpowers/specs/2026-04-29-starvla-migration-design.md` — read this first; it has all decisions, file layouts, and YAML/code stubs.

**Repo:** `/Users/tancilon/develop/localgit/UniamVLA` (GitHub `Tancilon/UniamVLA`).

**Working branch:** `feat/uamvla-migration` (created from `starVLA_dev` at Task 1).

**Commit attribution rule:** No Claude/Anthropic mentions in any commit. User identity only (`tancilon` / `tancilon1@gmail.com`). Apply to every commit in this plan.

---

## File Structure

| Path (relative to UniamVLA root) | Action | Purpose |
|---|---|---|
| `starVLA/utils/{geometry,rotation,point_cloud}.py` | NEW (port) | Math + IO utilities used by preprocessor, embodiment adapter, point cloud cleaning |
| `tools/preprocess/__init__.py` | NEW | Package marker |
| `tools/preprocess/{base,libero,target_object_resolver,calvin_*}_preprocessor.py` | NEW (port) | LIBERO HDF5 → UamVLA JSONL converter; CALVIN ports too (Phase 2 dormant) |
| `runners/preprocess_libero.py` | NEW (port) | CLI entry point for LIBERO preprocessing |
| `starVLA/model/framework/base_framework.py` | PATCH | +1 default no-op method `visualize_batch` |
| `starVLA/training/train_starvla.py` | PATCH | +`get_lr_groups` dispatch in `setup_optimizer_and_scheduler`; +viz hook in train loop |
| `starVLA/model/modules/uamvla/state_encoder/*.py` | NEW (port, 3 files) | ModularStateEncoder + LimbEncoders + special_tokens registration |
| `starVLA/model/modules/uamvla/aux_heads/*.py` | NEW (port, 5 files) | base + action/pose/future/recon heads |
| `starVLA/model/modules/uamvla/components/{pose,denoiser,pixel_decoder}/*.py` | NEW (port) | Pose components (PointNet2, ScoreNet, SDE, sampler), DiT denoiser, VAE pixel decoder |
| `starVLA/model/modules/uamvla/components/{query_reader,spatial_reader,task_adapter}.py` | NEW (port, 3 files) | Cross-attention readers + task adapter |
| `starVLA/model/modules/uamvla/data/{embodiment_adapter,embodiment_registry,action_tokenizer,chat_template}.py` | NEW (port, 4 files) | EmbodimentAdapter, ActionTokenizer, chat templates |
| `starVLA/model/modules/uamvla/backbone_wrapper.py` | NEW | Qwen3-VL backbone + state-token replacement (assembles state encoder + chat template + token replacement) |
| `starVLA/model/framework/VLM4A/UamVLA.py` | NEW | `@FRAMEWORK_REGISTRY.register("UamVLA")` class — implements forward / predict_action / visualize_batch / get_lr_groups |
| `starVLA/dataloader/uamvla_dataset.py` | NEW | Plugin: reads UamVLA JSONL → starVLA `examples` list; registers `DATASET_NAMED_MIXTURES["libero_uamvla"]` |
| `starVLA/config/training/uamvla_libero.yaml` | NEW | Phase 1 training config (single-file OmegaConf) |
| `examples/LIBERO/train_files/run_libero_train.sh` | EDIT IN-PLACE | UamVLA-specific paths/run_id/Framework_name |
| `examples/LIBERO/eval_files/run_policy_server.sh` | EDIT IN-PLACE | UamVLA-specific paths/CKPT |
| `examples/LIBERO/eval_files/eval_libero.sh` | EDIT IN-PLACE | UamVLA-specific paths/CKPT/task_suite |
| `tests/test_uamvla_*.py` | NEW | pytest unit + smoke tests (mirrors UamVLA's `tests/` layout) |

---

## Phase 1A: Setup & Foundation Ports (WP5 utilities + preprocessor)

### Task 1: Create work branch and bootstrap test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create feature branch from `starVLA_dev`**

```bash
cd /Users/tancilon/develop/localgit/UniamVLA
git checkout starVLA_dev
git pull origin starVLA_dev
git checkout -b feat/uamvla-migration
```

- [ ] **Step 2: Create `tests/__init__.py` and minimal `conftest.py`**

```python
# tests/conftest.py
"""pytest config for UniamVLA migration tests."""
import sys
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 3: Verify pytest discovers tests/**

Run: `pytest tests/ --collect-only -q`
Expected: `0 tests collected` (no tests yet, directory recognized)

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "[infra] Bootstrap tests/ directory for UamVLA migration"
```

---

### Task 2: Port utility modules (geometry, rotation, point_cloud)

**Files:**
- Create: `starVLA/utils/__init__.py` (empty)
- Create: `starVLA/utils/geometry.py` ← `UamVLA/uamvla/utils/geometry_utils.py`
- Create: `starVLA/utils/rotation.py` ← `UamVLA/uamvla/utils/rotation_utils.py`
- Create: `starVLA/utils/point_cloud.py` ← `UamVLA/uamvla/utils/point_cloud_utils.py`
- Create: `tests/test_utils_smoke.py`

- [ ] **Step 1: Write import smoke test**

```python
# tests/test_utils_smoke.py
"""Smoke test: ported utility modules import and expose expected callables."""
import numpy as np


def test_geometry_imports():
    from starVLA.utils.geometry import (
        crop_target_from_seg,
        depth_to_world_points,
        extract_instruction_from_filename,
        get_camera_intrinsic_from_fovy,
        linearize_depth,
        mat_to_6d,
    )
    # mat_to_6d basic correctness
    R = np.eye(3, dtype=np.float32)
    sixd = mat_to_6d(R)
    assert len(sixd) == 6


def test_rotation_imports():
    from starVLA.utils.rotation import quat_to_6d, euler_to_6d, axis_angle_to_6d
    aa = np.zeros(3, dtype=np.float32)
    sixd = axis_angle_to_6d(aa)
    assert sixd.shape == (6,)


def test_point_cloud_imports():
    from starVLA.utils.point_cloud import clean_point_cloud
    pts = np.random.randn(1024, 3).astype(np.float32)
    cleaned = clean_point_cloud(pts)
    assert cleaned.shape[1] == 3
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `pytest tests/test_utils_smoke.py -v`
Expected: 3 FAILED with `ModuleNotFoundError: No module named 'starVLA.utils'`

- [ ] **Step 3: Copy utility files**

```bash
mkdir -p starVLA/utils
touch starVLA/utils/__init__.py
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/utils/geometry_utils.py    starVLA/utils/geometry.py
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/utils/rotation_utils.py    starVLA/utils/rotation.py
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/utils/point_cloud_utils.py starVLA/utils/point_cloud.py
```

- [ ] **Step 4: Rewrite intra-package imports**

In each of the three new files, replace any `from uamvla.utils.*` imports with `from starVLA.utils.*`. Use:

```bash
grep -rn "from uamvla\." starVLA/utils/
# For each match, edit the file and change "from uamvla.X.Y" -> "from starVLA.X.Y" or "from starVLA.utils.Y" as appropriate
```

If a utility file imports from `uamvla.<other>` (not `uamvla.utils`), STOP and document the cross-dependency — port that module first. For Phase 1A these utilities should be self-contained or only reference each other.

- [ ] **Step 5: Run test to confirm pass**

Run: `pytest tests/test_utils_smoke.py -v`
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add starVLA/utils/ tests/test_utils_smoke.py
git commit -m "[utils] Port geometry/rotation/point_cloud utilities from UamVLA"
```

---

### Task 3: Port preprocessing base + LIBERO preprocessor (with image flip change)

**Files:**
- Create: `tools/__init__.py`, `tools/preprocess/__init__.py`
- Create: `tools/preprocess/base_preprocessor.py` ← `UamVLA/uamvla/data/preprocessing/base_preprocessor.py`
- Create: `tools/preprocess/target_object_resolver.py` ← same
- Create: `tools/preprocess/libero_preprocessor.py` ← same, **with `_render_rgb` flip change**
- Create: `tests/test_preprocessor_smoke.py`

- [ ] **Step 1: Copy preprocessor files**

```bash
mkdir -p tools/preprocess
touch tools/__init__.py tools/preprocess/__init__.py
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/data/preprocessing/base_preprocessor.py        tools/preprocess/
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/data/preprocessing/target_object_resolver.py  tools/preprocess/
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/data/preprocessing/libero_preprocessor.py     tools/preprocess/
```

- [ ] **Step 2: Rewrite imports in copied files**

In each of the three files, replace:
- `from uamvla.data.preprocessing.base_preprocessor` → `from tools.preprocess.base_preprocessor`
- `from uamvla.data.preprocessing.target_object_resolver` → `from tools.preprocess.target_object_resolver`
- `from uamvla.utils.geometry_utils` → `from starVLA.utils.geometry`
- `from uamvla.utils.rotation_utils` → `from starVLA.utils.rotation`
- `from uamvla.utils.point_cloud_utils` → `from starVLA.utils.point_cloud`

Verify with: `grep -rn "from uamvla\." tools/preprocess/`
Expected: no matches.

- [ ] **Step 3: Apply image flip change in `libero_preprocessor.py`**

Find this in `_render_rgb` (currently line 653 in UamVLA):

```python
return rgb[::-1].copy()
```

Change to:

```python
return rgb[::-1, ::-1].copy()
```

This aligns training data with starVLA eval client's `obs[::-1, ::-1]` (180° rotation, see spec §6.5).

- [ ] **Step 4: Write smoke test for libero_preprocessor import + class instantiation (no env)**

```python
# tests/test_preprocessor_smoke.py
"""Smoke test: libero_preprocessor module imports and class can be constructed."""
import pytest


def test_libero_preprocessor_imports():
    from tools.preprocess.libero_preprocessor import LiberoPreprocessor, MAX_ACTION_DIM, FRANKA_ACTION_DIM
    assert MAX_ACTION_DIM == 24
    assert FRANKA_ACTION_DIM == 7


def test_libero_preprocessor_construct_no_env():
    """Construct without target resolver — should not require LIBERO/MuJoCo at __init__."""
    from tools.preprocess.libero_preprocessor import LiberoPreprocessor
    p = LiberoPreprocessor(suite="libero_spatial", target_object_keyword=None, target_resolver=None)
    assert p.suite == "libero_spatial"
```

- [ ] **Step 5: Run test to confirm pass**

Run: `pytest tests/test_preprocessor_smoke.py -v`
Expected: 2 PASSED. (If `libero` import fails because LIBERO not installed locally, mark these tests with `pytest.importorskip("libero", reason="run on remote")` and skip locally; rerun to confirm skipped.)

- [ ] **Step 6: Commit**

```bash
git add tools/preprocess/ tests/test_preprocessor_smoke.py
git commit -m "[preprocess] Port LIBERO preprocessor; align image flip to [::-1, ::-1] for starVLA eval"
```

---

### Task 4: Port CALVIN preprocessor + map (Phase 2 dormant)

**Files:**
- Create: `tools/preprocess/calvin_preprocessor.py`
- Create: `tools/preprocess/calvin_env_adapter.py`
- Create: `tools/preprocess/calvin_task_map.py`

- [ ] **Step 1: Copy and import-rewrite**

```bash
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/data/preprocessing/calvin_preprocessor.py    tools/preprocess/
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/data/preprocessing/calvin_env_adapter.py     tools/preprocess/
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/data/preprocessing/calvin_task_map.py        tools/preprocess/
```

Rewrite imports same as Task 3 step 2. CALVIN-specific imports (`from calvin_env...`) leave alone — those resolve in `calvin_env` conda only.

- [ ] **Step 2: Smoke import check (skipped if calvin_env unavailable)**

Add to `tests/test_preprocessor_smoke.py`:

```python
def test_calvin_preprocessor_imports():
    pytest.importorskip("calvin_env", reason="CALVIN runs on remote in calvin_env conda")
    from tools.preprocess.calvin_preprocessor import CalvinPreprocessor
    assert CalvinPreprocessor is not None
```

- [ ] **Step 3: Run test**

Run: `pytest tests/test_preprocessor_smoke.py -v`
Expected: existing tests PASS; calvin test SKIPPED locally.

- [ ] **Step 4: Commit**

```bash
git add tools/preprocess/calvin_*.py tests/test_preprocessor_smoke.py
git commit -m "[preprocess] Port CALVIN preprocessor (Phase 2 dormant; runs in calvin_env)"
```

---

### Task 5: Port preprocess_libero.py runner

**Files:**
- Create: `runners/__init__.py`
- Create: `runners/preprocess_libero.py` ← `UamVLA/runners/preprocess_libero.py`

- [ ] **Step 1: Copy runner**

```bash
mkdir -p runners
touch runners/__init__.py
cp /Users/tancilon/develop/localgit/UamVLA/runners/preprocess_libero.py runners/
```

- [ ] **Step 2: Rewrite imports**

Replace any `from uamvla.data.preprocessing.*` → `from tools.preprocess.*`. Replace `from uamvla.utils.*` → `from starVLA.utils.*`. Verify:

```bash
grep -n "from uamvla\." runners/preprocess_libero.py
```
Expected: no matches.

- [ ] **Step 3: Verify --help works (smoke)**

Run: `python runners/preprocess_libero.py --help` (in `uamvla` conda env on Mac if libero available, else skip)
Expected: argparse help printout, no import error.

If `libero` import fails locally, defer to remote: this script ultimately runs on the GPU server. Mark as "verified on remote in Task 35".

- [ ] **Step 4: Commit**

```bash
git add runners/preprocess_libero.py runners/__init__.py
git commit -m "[runners] Port preprocess_libero entry point with starVLA import paths"
```

---

## Phase 1B: starVLA Mainline Patches (WP7)

### Task 6: Add default no-op `visualize_batch` to `baseframework`

**Files:**
- Modify: `starVLA/model/framework/base_framework.py:79-203` (class baseframework)
- Create: `tests/test_base_framework_patch.py`

- [ ] **Step 1: Write test that checks the new method exists with correct signature**

```python
# tests/test_base_framework_patch.py
import torch


def test_baseframework_has_visualize_batch_default():
    from starVLA.model.framework.base_framework import baseframework
    assert hasattr(baseframework, "visualize_batch")
    # Default returns empty dict (no behavior change for existing frameworks)
    fake = baseframework()
    out = fake.visualize_batch({}, n_samples=1)
    assert out == {}
```

- [ ] **Step 2: Run test — should fail**

Run: `pytest tests/test_base_framework_patch.py::test_baseframework_has_visualize_batch_default -v`
Expected: FAIL with `AttributeError` or fail on missing method.

- [ ] **Step 3: Patch base_framework.py — add method inside `class baseframework`**

Insert after the `forward_vlm` method (around line 204), before `from_pretrained`:

```python
    def visualize_batch(self, batch: dict, n_samples: int = 1) -> dict:
        """Default: no visualizations. Subclasses may override.

        Returns dict mapping log key (e.g. "viz/pose/0") to a wandb.Image
        or other loggable object. The trainer logs the dict via wandb.log()
        when cfg.trainer.visualization.enabled is True.
        """
        return {}
```

- [ ] **Step 4: Run test — should pass**

Run: `pytest tests/test_base_framework_patch.py -v`
Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/framework/base_framework.py tests/test_base_framework_patch.py
git commit -m "[framework] Add default no-op visualize_batch hook on baseframework"
```

---

### Task 7: Patch `train_starvla.py` for `get_lr_groups` dispatch

**Files:**
- Modify: `starVLA/training/train_starvla.py:78-101` (`setup_optimizer_and_scheduler`)
- Create: `tests/test_train_starvla_patches.py`

- [ ] **Step 1: Write test for dispatch logic**

```python
# tests/test_train_starvla_patches.py
"""Test that setup_optimizer_and_scheduler dispatches to model.get_lr_groups when present."""
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def get_lr_groups(self, lr_cfg):
        return [{"name": "linear", "params": list(self.linear.parameters()), "lr": 1e-3}]


def test_setup_optimizer_uses_get_lr_groups_when_present():
    from starVLA.training.train_starvla import setup_optimizer_and_scheduler
    cfg = MagicMock()
    cfg.trainer.learning_rate.base = 1e-4
    cfg.trainer.optimizer.betas = [0.9, 0.95]
    cfg.trainer.optimizer.eps = 1e-8
    cfg.trainer.optimizer.weight_decay = 1e-8
    cfg.trainer.lr_scheduler_type = "cosine"
    cfg.trainer.num_warmup_steps = 0
    cfg.trainer.max_train_steps = 100
    cfg.trainer.scheduler_specific_kwargs = {}

    model = _FakeModel()
    opt, _ = setup_optimizer_and_scheduler(model, cfg)
    # Should have one param group with name "linear" (from get_lr_groups, not legacy path)
    names = [g.get("name") for g in opt.param_groups]
    assert "linear" in names
```

- [ ] **Step 2: Run test — should fail (existing code calls `build_param_lr_groups` unconditionally)**

Run: `pytest tests/test_train_starvla_patches.py::test_setup_optimizer_uses_get_lr_groups_when_present -v`
Expected: FAIL or PASS-by-accident depending on `build_param_lr_groups` behavior with fake model. The KEY test is that `get_lr_groups` is called when present — adjust assertion to spy on the model:

```python
def test_setup_optimizer_uses_get_lr_groups_when_present():
    from starVLA.training.train_starvla import setup_optimizer_and_scheduler
    cfg = ...  # same as above
    model = _FakeModel()
    spy = MagicMock(side_effect=model.get_lr_groups)
    model.get_lr_groups = spy
    opt, _ = setup_optimizer_and_scheduler(model, cfg)
    spy.assert_called_once()
```

- [ ] **Step 3: Modify `setup_optimizer_and_scheduler` in `train_starvla.py:78-101`**

Replace the body of `setup_optimizer_and_scheduler` (the line `param_groups = build_param_lr_groups(model=model, cfg=cfg)` near line 80):

```python
def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler.

    Frameworks may declare their own LR group routing via a get_lr_groups(lr_cfg)
    method. If absent, fall back to the legacy module-name-based grouping.
    """
    if hasattr(model, "get_lr_groups"):
        param_groups = model.get_lr_groups(cfg.trainer.learning_rate)
    else:
        param_groups = build_param_lr_groups(model=model, cfg=cfg)

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    # ... rest unchanged
```

- [ ] **Step 4: Run test — should pass**

Run: `pytest tests/test_train_starvla_patches.py -v`
Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
git add starVLA/training/train_starvla.py tests/test_train_starvla_patches.py
git commit -m "[trainer] Dispatch to model.get_lr_groups when present (backwards-compatible)"
```

---

### Task 8: Patch `train_starvla.py` for visualization hook in train loop

**Files:**
- Modify: `starVLA/training/train_starvla.py` — find the per-step training loop body (search for `accelerator.backward` near line 350)

- [ ] **Step 1: Locate the training step body**

Run: `grep -n "self.completed_steps" starVLA/training/train_starvla.py`
Identify the train step where `self.completed_steps += 1` happens (typically right after optimizer.step). The viz hook goes after the step, on main process, conditional on YAML.

- [ ] **Step 2: Add hook logic after `self.completed_steps += 1`**

```python
# Insert after the completed_steps increment, INSIDE the `if self.accelerator.sync_gradients:` block
# so it fires exactly once per completed step under gradient_accumulation_steps > 1.
# The local variable in the train loop is `batch_vla` (not `batch`).
viz_cfg = getattr(self.config.trainer, "visualization", None) or {}
if viz_cfg.get("enabled", False):
    every = int(viz_cfg.get("train_every_n_steps", 1000))
    if self.completed_steps % every == 0 and self.accelerator.is_main_process:
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, "visualize_batch"):
            try:
                viz_imgs = unwrapped.visualize_batch(
                    batch_vla, n_samples=int(viz_cfg.get("num_samples", 1)),
                )
                if viz_imgs:
                    wandb.log(viz_imgs, step=self.completed_steps)
            except Exception as e:
                logger.warning(f"visualize_batch failed at step {self.completed_steps}: {e}")
```

The `try/except` is intentional — viz should never crash training.

- [ ] **Step 3: Smoke verify other framework classes still train (no behavior change)**

Run: `python -c "from starVLA.training.train_starvla import VLATrainer; print('import OK')"`
Expected: import succeeds.

(Full integration verification happens at Task 36 L2 1-step training.)

- [ ] **Step 4: Commit**

```bash
git add starVLA/training/train_starvla.py
git commit -m "[trainer] Add visualization hook in training loop (main-rank only, exception-safe)"
```

---

## Phase 1C: UamVLA Submodules Port (WP2 part 1)

### Task 9: Port state_encoder/ submodule

**Files:**
- Create: `starVLA/model/modules/uamvla/__init__.py`
- Create: `starVLA/model/modules/uamvla/state_encoder/__init__.py`
- Create: `starVLA/model/modules/uamvla/state_encoder/{modular_state_encoder,limb_encoders,special_tokens}.py`

- [ ] **Step 1: Copy directory**

```bash
mkdir -p starVLA/model/modules/uamvla
touch starVLA/model/modules/uamvla/__init__.py
cp -r /Users/tancilon/develop/localgit/UamVLA/uamvla/models/state_encoder starVLA/model/modules/uamvla/
# remove __pycache__
find starVLA/model/modules/uamvla/state_encoder -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

- [ ] **Step 2: Rewrite imports in all 3 files**

For each `.py` file in `state_encoder/`, replace:
- `from uamvla.data.embodiment_registry` → defer the import (use a lazy import inside `ModularStateEncoder.__init__`). Will be re-hoisted to module top in Task 14 once the data submodule lands.
- `from uamvla.utils.*` → `from starVLA.utils.*` (only `limb_encoders.py` and `special_tokens.py` need this if at all; current source has none).

Verified by grep: the source `state_encoder/` does NOT import from `uamvla.core.registry`, and `ModularStateEncoder` is NOT decorated with `@MODALITY_ENCODER_REGISTRY.register(...)` despite what earlier plan drafts assumed. No registry-decorator removal is needed. Direct construction by `UamVLAFramework` (Phase 1D) works without any registry indirection.

- [ ] **Step 3: Add smoke import test**

```python
# tests/test_state_encoder_smoke.py
def test_modular_state_encoder_imports():
    from starVLA.model.modules.uamvla.state_encoder.modular_state_encoder import ModularStateEncoder
    # Cannot fully construct without embodiment registry yet — just verify class exists
    assert ModularStateEncoder is not None


def test_special_tokens_helpers_import():
    from starVLA.model.modules.uamvla.state_encoder.special_tokens import register_state_tokens
    assert callable(register_state_tokens)
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_state_encoder_smoke.py -v`
Expected: 2 PASSED (if embodiment_registry import deferred / commented).

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/modules/uamvla/state_encoder/ starVLA/model/modules/uamvla/__init__.py tests/test_state_encoder_smoke.py
git commit -m "[modules/uamvla] Port state_encoder submodule"
```

---

### Task 10: Port aux_heads/ submodule (base + 4 heads)

**Files:**
- Create: `starVLA/model/modules/uamvla/aux_heads/__init__.py`
- Create: `starVLA/model/modules/uamvla/aux_heads/{base,action_head,pose_head,future_head,recon_head}.py`

- [ ] **Step 1: Copy directory**

```bash
cp -r /Users/tancilon/develop/localgit/UamVLA/uamvla/models/aux_heads starVLA/model/modules/uamvla/
find starVLA/model/modules/uamvla/aux_heads -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

- [ ] **Step 2: Rewrite imports across all 5 files**

Patterns to replace:
- `from uamvla.core.registry import AUX_HEAD_REGISTRY` → drop the import (heads will be constructed directly by UamVLAFramework, no registry needed)
- `@AUX_HEAD_REGISTRY.register("...")` → remove decorator line entirely, OR replace with no-op
- `from uamvla.models.aux_heads.base` → `from starVLA.model.modules.uamvla.aux_heads.base`
- `from uamvla.models.components.pose.*` → `from starVLA.model.modules.uamvla.components.pose.*` (Task 11 will satisfy)
- `from uamvla.models.components.denoiser.*` → `from starVLA.model.modules.uamvla.components.denoiser.*` (Task 12 satisfies)
- `from uamvla.models.components.spatial_reader` → `from starVLA.model.modules.uamvla.components.spatial_reader` (Task 13 satisfies)
- `from uamvla.models.components.query_reader` → same direction

Verify: `grep -rn "from uamvla\." starVLA/model/modules/uamvla/aux_heads/`
Expected: no matches.

- [ ] **Step 3: Smoke test (only `base` and `action_head` will fully import without component deps)**

```python
# tests/test_aux_heads_smoke.py
def test_aux_heads_base_imports():
    from starVLA.model.modules.uamvla.aux_heads.base import AuxHead, HeadOutput
    assert AuxHead is not None
    assert HeadOutput is not None


def test_action_head_imports():
    from starVLA.model.modules.uamvla.aux_heads.action_head import ActionHead
    assert ActionHead is not None
```

Skip pose/future/recon import tests — they need components ported (Tasks 11-13).

- [ ] **Step 4: Run test**

Run: `pytest tests/test_aux_heads_smoke.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/modules/uamvla/aux_heads/ tests/test_aux_heads_smoke.py
git commit -m "[modules/uamvla] Port aux_heads submodule (base + action/pose/future/recon)"
```

---

### Task 11: Port components/pose/ submodule

**Files:**
- Create: `starVLA/model/modules/uamvla/components/__init__.py`
- Create: `starVLA/model/modules/uamvla/components/pose/{__init__,pts_encoder,score_net,sde,sampler,pose_utils}.py`
- Create: `starVLA/model/modules/uamvla/components/pose/pointnet2/` (entire subdir)

- [ ] **Step 1: Copy components/pose/**

```bash
mkdir -p starVLA/model/modules/uamvla/components
touch starVLA/model/modules/uamvla/components/__init__.py
cp -r /Users/tancilon/develop/localgit/UamVLA/uamvla/models/components/pose starVLA/model/modules/uamvla/components/
find starVLA/model/modules/uamvla/components/pose -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

- [ ] **Step 2: Rewrite imports**

Patterns:
- `from uamvla.models.components.pose.*` → `from starVLA.model.modules.uamvla.components.pose.*`
- `from uamvla.utils.*` → `from starVLA.utils.*`

Note: `pointnet2/` may have its own internal imports — leave intra-package imports alone (e.g., `from .pointnet2_modules`).

- [ ] **Step 3: Smoke test**

```python
# tests/test_pose_components_smoke.py
def test_pose_components_imports():
    from starVLA.model.modules.uamvla.components.pose.pose_utils import get_pose_dim, rotation_6d_to_matrix
    from starVLA.model.modules.uamvla.components.pose.sde import init_sde
    # PointNet2Wrapper.__init__ lazy-imports the CUDA modules only at instantiation,
    # so module-level import always succeeds on Mac (without CUDA). The try/except
    # below is defensive only — it would fire only if `pts_encoder.py` itself
    # introduced a top-level CUDA-dependent import in the future.
    import pytest
    try:
        from starVLA.model.modules.uamvla.components.pose.pts_encoder import PointNet2Wrapper
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"PointNet2Wrapper module-level import failed: {e}")
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_pose_components_smoke.py -v`
Expected: PASSED on Mac (PointNet2 may skip).

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/modules/uamvla/components/ tests/test_pose_components_smoke.py
git commit -m "[modules/uamvla] Port components/pose (PointNet2, ScoreNet, SDE, sampler, pose_utils)"
```

---

### Task 12: Port components/denoiser, components/pixel_decoder

**Files:**
- Create: `starVLA/utils/diffusion_utils/{__init__,diffusion_utils,gaussian_diffusion,respace}.py` (Step 0 prereq — Phase 1A omission)
- Create: `starVLA/model/modules/uamvla/components/denoiser/{__init__,common,dit,scheduler}.py`
- Create: `starVLA/model/modules/uamvla/components/pixel_decoder/{__init__,vae}.py`

> **Phase 1A omission (commit separately as Step 0):** `uamvla/utils/diffusion_utils/` was missed during Phase 1A's util port (Task 2 covered geometry/rotation/point_cloud only). `denoiser/scheduler.py:9` imports `from uamvla.utils.diffusion_utils import create_diffusion`, so it must be ported first. The 4 files are verbatim — no `uamvla.*` imports inside (all internal imports are relative). Land as a separate commit titled `[utils] Port diffusion_utils package (deferred Phase 1A dependency for denoiser scheduler)` before Step 1.

- [ ] **Step 0 (prereq): Port diffusion_utils**

```bash
mkdir -p starVLA/utils/diffusion_utils
cp -r /Users/tancilon/develop/localgit/UamVLA/uamvla/utils/diffusion_utils/. starVLA/utils/diffusion_utils/
find starVLA/utils/diffusion_utils -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

Add `tests/test_diffusion_utils_smoke.py` with `test_create_diffusion_imports`. Commit with the message above. Then proceed to Step 1.

- [ ] **Step 1: Copy directories**

```bash
cp -r /Users/tancilon/develop/localgit/UamVLA/uamvla/models/components/denoiser       starVLA/model/modules/uamvla/components/
cp -r /Users/tancilon/develop/localgit/UamVLA/uamvla/models/components/pixel_decoder  starVLA/model/modules/uamvla/components/
find starVLA/model/modules/uamvla/components/denoiser starVLA/model/modules/uamvla/components/pixel_decoder -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

- [ ] **Step 2: Rewrite imports**

Replace `from uamvla.models.components.*` → `from starVLA.model.modules.uamvla.components.*`. Verify clean.

- [ ] **Step 3: Smoke test**

```python
# tests/test_denoiser_smoke.py
def test_denoiser_imports():
    from starVLA.model.modules.uamvla.components.denoiser.scheduler import ReconDenoiser
    assert ReconDenoiser is not None


def test_vae_imports():
    from starVLA.model.modules.uamvla.components.pixel_decoder.vae import VAEPixelDecoder
    assert VAEPixelDecoder is not None
```

- [ ] **Step 4: Run + commit**

Run: `pytest tests/test_denoiser_smoke.py -v` → PASSED

```bash
git add starVLA/model/modules/uamvla/components/denoiser/ starVLA/model/modules/uamvla/components/pixel_decoder/ tests/test_denoiser_smoke.py
git commit -m "[modules/uamvla] Port components/denoiser and components/pixel_decoder"
```

---

### Task 13: Port components/{query_reader, spatial_reader, task_adapter}

- [ ] **Step 1: Copy and rewrite**

```bash
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/models/components/query_reader.py    starVLA/model/modules/uamvla/components/
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/models/components/spatial_reader.py  starVLA/model/modules/uamvla/components/
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/models/components/task_adapter.py    starVLA/model/modules/uamvla/components/
```

Rewrite `from uamvla.*` imports same as Task 12.

- [ ] **Step 2: Add to smoke test**

Append to `tests/test_denoiser_smoke.py`:

```python
def test_readers_import():
    from starVLA.model.modules.uamvla.components.query_reader import TaskQueryReader
    from starVLA.model.modules.uamvla.components.spatial_reader import slice_image_tokens
    assert TaskQueryReader is not None
    assert callable(slice_image_tokens)
```

- [ ] **Step 3: Run + commit**

Run: `pytest tests/test_denoiser_smoke.py -v` → all PASSED

```bash
git add starVLA/model/modules/uamvla/components/{query_reader,spatial_reader,task_adapter}.py tests/test_denoiser_smoke.py
git commit -m "[modules/uamvla] Port components/query_reader, spatial_reader, task_adapter"
```

- [ ] **Step 4: Now retry pose/future/recon head imports (depends on Tasks 11-13)**

```python
# Append to tests/test_aux_heads_smoke.py
def test_pose_head_imports():
    import pytest
    try:
        from starVLA.model.modules.uamvla.aux_heads.pose_head import PoseHead
        assert PoseHead is not None
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"PoseHead requires CUDA-built PointNet2: {e}")


def test_future_head_imports():
    from starVLA.model.modules.uamvla.aux_heads.future_head import FutureHead
    assert FutureHead is not None


def test_recon_head_imports():
    from starVLA.model.modules.uamvla.aux_heads.recon_head import ReconHead
    assert ReconHead is not None
```

Run: `pytest tests/test_aux_heads_smoke.py -v` → all PASSED (pose may skip on Mac).

```bash
git add tests/test_aux_heads_smoke.py
git commit -m "[modules/uamvla] Verify pose/future/recon heads import after components ported"
```

---

### Task 14: Port data/ submodule (embodiment_adapter, registry, action_tokenizer, chat_template)

**Files:**
- Create: `starVLA/model/modules/uamvla/data/__init__.py`
- Create: `starVLA/model/modules/uamvla/data/{embodiment_adapter,embodiment_registry,action_tokenizer,chat_template}.py`

- [ ] **Step 1: Copy specific files**

```bash
mkdir -p starVLA/model/modules/uamvla/data
touch starVLA/model/modules/uamvla/data/__init__.py
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/data/embodiment_adapters.py     starVLA/model/modules/uamvla/data/embodiment_adapter.py
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/data/embodiment_registry.py     starVLA/model/modules/uamvla/data/embodiment_registry.py
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/data/action_tokenizer.py        starVLA/model/modules/uamvla/data/action_tokenizer.py
cp /Users/tancilon/develop/localgit/UamVLA/uamvla/data/chat_template.py           starVLA/model/modules/uamvla/data/chat_template.py
```

- [ ] **Step 2: Rewrite imports across all 4 files**

- `from uamvla.utils.rotation_utils` → `from starVLA.utils.rotation`
- `from uamvla.data.embodiment_adapters` → `from starVLA.model.modules.uamvla.data.embodiment_adapter`
- `from uamvla.data.embodiment_registry` → `from starVLA.model.modules.uamvla.data.embodiment_registry`
- `from uamvla.data.action_tokenizer` → `from starVLA.model.modules.uamvla.data.action_tokenizer`
- `from uamvla.data.chat_template` → `from starVLA.model.modules.uamvla.data.chat_template`

- [ ] **Step 3: Now go back to state_encoder/ and re-hoist the embodiment_registry import that was deferred in Task 9**

Only `state_encoder/modular_state_encoder.py` actually imports `embodiment_registry` (verified by grep — `special_tokens.py` does not, despite earlier plan drafts). In Task 9, the import was placed inside `ModularStateEncoder.__init__` as a lazy import. Now that the data submodule exists, hoist it back to module top:

```python
# At module top, alongside the other imports:
from starVLA.model.modules.uamvla.data.embodiment_registry import get_embodiment_config
```

Remove the lazy-import line + its explanatory comment from inside `__init__`.

- [ ] **Step 4: Smoke test**

```python
# tests/test_data_submodule_smoke.py
def test_embodiment_adapter_imports():
    from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter, EmbodimentAdapter
    adapter = LiberoAdapter()
    assert adapter.embodiment_name == "franka_libero"


def test_embodiment_registry_get_config():
    from starVLA.model.modules.uamvla.data.embodiment_registry import get_embodiment_config
    cfg = get_embodiment_config("franka_libero")
    assert "adapter" in cfg
    assert "special_tokens" in cfg


def test_action_tokenizer_imports():
    from starVLA.model.modules.uamvla.data.action_tokenizer import ActionTokenizer
    assert ActionTokenizer is not None


def test_chat_template_imports():
    from starVLA.model.modules.uamvla.data.chat_template import ChatTemplate, Qwen3VLChatTemplate
    assert Qwen3VLChatTemplate is not None
```

- [ ] **Step 5: Run + commit**

Run: `pytest tests/test_data_submodule_smoke.py tests/test_state_encoder_smoke.py -v` → all PASSED

```bash
git add starVLA/model/modules/uamvla/data/ starVLA/model/modules/uamvla/state_encoder/ tests/test_data_submodule_smoke.py
git commit -m "[modules/uamvla] Port data submodule (embodiment_adapter/registry, action_tokenizer, chat_template)"
```

---

### Task 15: Cross-import sanity test

- [ ] **Step 1: Add aggregate import test**

```python
# tests/test_uamvla_modules_full_import.py
"""All UamVLA submodules should import together without circular reference."""
def test_full_import():
    import starVLA.model.modules.uamvla.state_encoder.modular_state_encoder
    import starVLA.model.modules.uamvla.aux_heads.action_head
    import starVLA.model.modules.uamvla.aux_heads.future_head
    import starVLA.model.modules.uamvla.aux_heads.recon_head
    import starVLA.model.modules.uamvla.components.pose.pose_utils
    import starVLA.model.modules.uamvla.components.denoiser.scheduler
    import starVLA.model.modules.uamvla.components.pixel_decoder.vae
    import starVLA.model.modules.uamvla.components.spatial_reader
    import starVLA.model.modules.uamvla.data.embodiment_adapter
    import starVLA.model.modules.uamvla.data.action_tokenizer
    import starVLA.model.modules.uamvla.data.chat_template
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_uamvla_modules_full_import.py -v`
Expected: PASSED. If any circular imports surface, fix them now (before framework class assembly).

- [ ] **Step 3: Commit**

```bash
git add tests/test_uamvla_modules_full_import.py
git commit -m "[tests] Add aggregate UamVLA submodule import sanity test"
```

---

## Phase 1D: UamVLA Framework Class (WP2 main + WP3)

### Task 16: Create `backbone_wrapper.py`

**Files:**
- Create: `starVLA/model/modules/uamvla/backbone_wrapper.py`
- Create: `tests/test_backbone_wrapper.py`

The wrapper is the only NEW assembly module — it composes:
- Qwen3-VL-8B-Instruct (HF transformers)
- ModularStateEncoder (consumes `canonical_state`)
- chat_template scaffolding (`<|state_ee|>` etc. placeholders)
- token-position replacement (`_replace_state_tokens`, see UamVLA `models/uamvla_model.py:19-71`)

- [ ] **Step 1: Write test for the wrapper's `build_inputs` contract**

```python
# tests/test_backbone_wrapper.py
import pytest
import torch
from PIL import Image
import numpy as np


def test_build_inputs_returns_qwen_kwargs():
    pytest.importorskip("transformers", minversion="4.57.0")
    from starVLA.model.modules.uamvla.backbone_wrapper import build_uamvla_backbone

    cfg = type("Cfg", (), {})()
    cfg.framework = type("F", (), {})()
    cfg.framework.qwenvl = type("Q", (), {"base_vlm": "Qwen/Qwen3-VL-8B-Instruct", "attn_implementation": "eager"})()
    cfg.framework.embodiment = type("E", (), {"name": "franka_libero"})()

    pytest.importorskip("Qwen3VLForConditionalGeneration", reason="full backbone build needs ckpt — gate this test")
    # The detailed test runs only on remote with ckpt available
```

For Mac local, just verify the module imports (full functional test on remote).

```python
def test_backbone_wrapper_module_imports():
    from starVLA.model.modules.uamvla.backbone_wrapper import build_uamvla_backbone
    assert callable(build_uamvla_backbone)
```

- [ ] **Step 2: Implement `backbone_wrapper.py`**

The implementation has three parts:
1. `class UamVLABackboneInterface(nn.Module)`: holds `Qwen3VLForConditionalGeneration` + tokenizer + `ModularStateEncoder`. Provides `.build_inputs(images, instructions, canonical_state)` and `.forward(**qwen_inputs)`. State token replacement happens inside `forward` when `inputs_embeds` is built from `input_ids`.
2. `_replace_state_tokens(...)`: copy from UamVLA `models/uamvla_model.py:19-71` (the helper).
3. `build_uamvla_backbone(cfg)`: factory that loads Qwen3-VL from HF, instantiates `ModularStateEncoder`, calls `register_state_tokens` to extend tokenizer, resizes embeddings.

Detailed code is in spec §4.1 + §5 + UamVLA `models/uamvla_model.py:74-339` (port `prepare_multimodal_inputs` + `_backbone_forward` logic into the wrapper class).

Key contract:
- `build_inputs(images, instructions, canonical_state) → dict` returns kwargs ready to pass to Qwen3-VL forward (has `input_ids`, `pixel_values`, `image_grid_thw`, etc.)
- `__call__(**qwen_inputs)` runs forward and returns `output.hidden_states[-1]` consumer-friendly. State token replacement inserted between embed lookup and main forward when canonical_state is in batch.

- [ ] **Step 3: Smoke import test**

Run: `pytest tests/test_backbone_wrapper.py::test_backbone_wrapper_module_imports -v`
Expected: PASSED.

- [ ] **Step 4: Commit**

```bash
git add starVLA/model/modules/uamvla/backbone_wrapper.py tests/test_backbone_wrapper.py
git commit -m "[modules/uamvla] Add backbone_wrapper: Qwen3-VL + state token replacement"
```

---

### Task 17: Create `UamVLA` framework class skeleton

**Files:**
- Create: `starVLA/model/framework/VLM4A/UamVLA.py`
- Create: `tests/test_uamvla_framework_init.py`

- [ ] **Step 1: Write test for class registration and basic init**

```python
# tests/test_uamvla_framework_init.py
import pytest


def test_uamvla_class_registered():
    # Trigger framework auto-import
    from starVLA.model.framework.base_framework import _auto_import_framework_modules
    _auto_import_framework_modules()
    from starVLA.model.tools import FRAMEWORK_REGISTRY
    assert "UamVLA" in FRAMEWORK_REGISTRY._registry


def test_uamvla_class_minimal_construct_no_backbone(monkeypatch):
    """Construct UamVLA with backbone_wrapper mocked — verifies __init__ glue without HF download."""
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA, UamVLADefaultConfig
    pytest.importorskip("torch")
    # Mock build_uamvla_backbone to avoid loading Qwen3-VL
    import starVLA.model.modules.uamvla.backbone_wrapper as bw

    class _FakeBackbone:
        class _FakeConfig:
            hidden_size = 4096
        config = _FakeConfig()
        tokenizer = None  # ModularStateEncoder construction may need real tokenizer; test what we can

        def __call__(self, *args, **kwargs):
            raise RuntimeError("not used in init test")

    monkeypatch.setattr(bw, "build_uamvla_backbone", lambda cfg: _FakeBackbone())
    # Not a full test — verify class attributes accessible
    assert UamVLA.__name__ == "UamVLA"
```

- [ ] **Step 2: Run test — should FAIL (class doesn't exist)**

Run: `pytest tests/test_uamvla_framework_init.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `UamVLA.py`**

Skeleton (full body filled in Tasks 18-22):

```python
# starVLA/model/framework/VLM4A/UamVLA.py
"""UamVLA framework: Qwen3-VL-8B + ModularStateEncoder + 4 aux heads.

@FRAMEWORK_REGISTRY.register("UamVLA")

See: docs/superpowers/specs/2026-04-29-starvla-migration-design.md
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.uamvla.backbone_wrapper import build_uamvla_backbone
from starVLA.model.modules.uamvla.state_encoder.modular_state_encoder import ModularStateEncoder
from starVLA.model.modules.uamvla.state_encoder.special_tokens import register_state_tokens
from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter
from starVLA.model.modules.uamvla.aux_heads.action_head import ActionHead
from starVLA.model.modules.uamvla.aux_heads.pose_head import PoseHead
from starVLA.model.modules.uamvla.aux_heads.future_head import FutureHead
from starVLA.model.modules.uamvla.aux_heads.recon_head import ReconHead


@dataclass
class UamVLADefaultConfig:
    name: str = "UamVLA"
    embodiment: dict = field(default_factory=lambda: {"name": "franka_libero", "action_dim": 7})
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "ckpt/Qwen3-VL-8B-Instruct",
        "attn_implementation": "flash_attention_2",
    })
    action_model: dict = field(default_factory=lambda: {
        "future_action_window_size": 7,
        "num_bins": 256,
        "action_dim": 7,
    })
    state_encoder: dict = field(default_factory=lambda: {"type": "modular", "register_special_tokens": True})
    vae: dict = field(default_factory=lambda: {"path": "ckpt/pretrained_vae"})
    aux_heads: dict = field(default_factory=lambda: {
        "action": {"enabled": True, "loss_weight": 1.0, "lr": 1.0e-4},
        "pose":   {"enabled": True, "loss_weight": 0.5, "lr": 1.0e-4},
        "future": {"enabled": True, "loss_weight": 0.1, "lr": 1.0e-4},
        "recon":  {"enabled": True, "loss_weight": 0.1, "lr": 1.0e-4},
    })


def _build_aux_heads(framework_cfg, hidden_size, vae=None, vision_extra=None, lm_head=None, action_token_begin_id=0):
    """Construct enabled aux heads. Mirrors UamVLA models/uamvla_model.py:197-218 logic."""
    heads = {}
    cfg_heads = framework_cfg.aux_heads
    if cfg_heads.action.get("enabled", True):
        heads["action"] = ActionHead(
            hidden_size=hidden_size, lm_head=lm_head,
            action_token_begin_id=action_token_begin_id,
            loss_weight=float(cfg_heads.action.get("loss_weight", 1.0)),
        )
    if cfg_heads.pose.get("enabled", True):
        heads["pose"] = PoseHead(
            hidden_size=hidden_size,
            **{k: v for k, v in cfg_heads.pose.items() if k not in ("enabled", "lr")},
        )
    if cfg_heads.future.get("enabled", True) and vae is not None:
        heads["future"] = FutureHead(
            hidden_size=hidden_size, vae=vae, **(vision_extra or {}),
            **{k: v for k, v in cfg_heads.future.items() if k not in ("enabled", "lr")},
        )
    if cfg_heads.recon.get("enabled", True) and vae is not None:
        heads["recon"] = ReconHead(
            hidden_size=hidden_size, vae=vae, **(vision_extra or {}),
            **{k: v for k, v in cfg_heads.recon.items() if k not in ("enabled", "lr")},
        )
    return heads


@FRAMEWORK_REGISTRY.register("UamVLA")
class UamVLA(baseframework):
    """UamVLA: Qwen3-VL backbone + state encoder + multi-aux-head VLA model.

    Single-loss contract with starVLA trainer: forward returns
    {"action_loss": <sum of head losses>, ...per-head metrics}.
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(UamVLADefaultConfig, config)

        self.qwen_vl_interface = build_uamvla_backbone(self.config)
        hidden_size = self.qwen_vl_interface.config.hidden_size

        # Aux heads
        from starVLA.model.modules.uamvla.components.pixel_decoder.vae import VAEPixelDecoder
        head_names = [n for n in ("action", "pose", "future", "recon")
                      if self.config.framework.aux_heads.get(n, {}).get("enabled", True)]
        needs_vae = any(n in ("future", "recon") for n in head_names)
        self.vae = VAEPixelDecoder(self.config.framework.vae.path) if needs_vae else None

        # vision_extra: inferred from backbone config (Qwen3-VL ppv=400 for 640px)
        vision_extra = self._derive_vision_extra(hidden_size, needs_vae)

        self.aux_heads = nn.ModuleDict(_build_aux_heads(
            self.config.framework,
            hidden_size=hidden_size,
            vae=self.vae,
            vision_extra=vision_extra,
            lm_head=self.qwen_vl_interface.get_lm_head(),
            action_token_begin_id=self.config.framework.get("action_token_begin_id", 0),
        ))

        self.action_horizon = int(self.config.framework.action_model.future_action_window_size) + 1

    def _derive_vision_extra(self, hidden_size, needs_vae):
        if not needs_vae:
            return {}
        # Qwen3-VL @ 640px / patch16 / merge2 → ppv = 400 (20x20 grid)
        ppv = 400
        side = int(ppv ** 0.5)
        return {
            "image_mean": [0.5, 0.5, 0.5],
            "image_std":  [0.5, 0.5, 0.5],
            "image_token_id": self.qwen_vl_interface.image_token_id,  # backbone wrapper exposes this
            "patches_per_view": ppv,
            "n_patches": ppv,
            "target_resize": side * 16,  # 320 for ppv=400
        }

    # forward / predict_action / visualize_batch / get_lr_groups / supports_training_tag
    # filled in Tasks 18-22
```

- [ ] **Step 4: Run test — registration + import should pass**

Run: `pytest tests/test_uamvla_framework_init.py::test_uamvla_class_registered -v`
Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py tests/test_uamvla_framework_init.py
git commit -m "[framework] Register UamVLA framework class with init skeleton"
```

---

### Task 18: Implement `UamVLA.forward` (multi-head loss summing)

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLA.py` — add `forward` method to class

- [ ] **Step 1: Write test for forward output contract**

```python
# tests/test_uamvla_forward.py
import pytest
import torch
from unittest.mock import MagicMock


def test_forward_returns_action_loss_key(monkeypatch):
    """With mocked backbone+heads, forward() returns dict with action_loss as the only backprop key."""
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA, _build_aux_heads
    pytest.skip("Full forward test requires backbone + dataloader fixtures — defer to L2 1-step training smoke")
```

For Mac local, full forward is not testable without backbone weights. Defer integration to remote L2 (Task 36). Add the placeholder above as documentation.

- [ ] **Step 2: Add `forward` method to `UamVLA` class**

Insert into `UamVLA.py`:

```python
    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Training forward. Aux head losses summed into single 'action_loss' for trainer backprop."""
        qwen_inputs = self.qwen_vl_interface.build_inputs(
            images=[e["image"] for e in examples],
            instructions=[e["lang"] for e in examples],
            canonical_state=[e["canonical_state"] for e in examples],
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
            hidden = backbone_out.hidden_states[-1]  # (B, L, H)

        # Stack collated batch fields the heads consume
        batch_dict = self._collate_for_heads(examples, qwen_inputs)

        total = torch.tensor(0.0, device=hidden.device, requires_grad=True)
        log_metrics = {}
        for name, head in self.aux_heads.items():
            mask = batch_dict.get(f"{name}_mask")
            if mask is None:
                mask = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
            out = head.compute_loss(hidden, batch_dict, mask=mask)
            if out.loss is not None:
                total = total + out.loss
                log_metrics[f"{name}_loss"] = out.loss.detach()
            for mk, mv in out.metrics.items():
                log_metrics[f"{name}_{mk}"] = mv

        return {"action_loss": total, **log_metrics}

    def _collate_for_heads(self, examples, qwen_inputs):
        """Build the batch_dict aux heads read from. Includes input_ids, labels, masks, and per-head targets."""
        batch_dict = {
            "input_ids":     qwen_inputs["input_ids"],
            "labels":        qwen_inputs.get("labels"),
            "image":         qwen_inputs.get("pixel_values"),
            "instruction":   [e["lang"] for e in examples],
        }
        # Stack pose / future / recon targets from examples + build masks
        # (Detailed stacking logic mirrors UamVLA collator.py:213-281)
        from starVLA.model.modules.uamvla.data.chat_template import _stack_optional
        # ... see UamVLA uamvla/data/collator.py for the exact stack pattern
        # Simplified Phase 1: assume dataloader's collate_fn already produced these
        for k in ("pose_gt", "image_target", "image_future", "point_cloud", "static_cam_extrinsic"):
            present = [e for e in examples if k in e]
            if present:
                # Stack with batch-level zero-padding for missing samples
                batch_dict[k] = self._stack_optional_field(examples, k)
                batch_dict[f"{self._mask_key(k)}_mask"] = torch.tensor(
                    [k in e for e in examples], dtype=torch.bool,
                )
        return batch_dict

    @staticmethod
    def _mask_key(field_name):
        return {"pose_gt": "pose", "image_target": "recon", "image_future": "future"}.get(field_name, field_name)

    # `_collate_for_heads` body and helper functions live in the sibling module
    # `starVLA/model/modules/uamvla/collator_helpers.py` (see Step 3).
```

**Decision (resolved)**: dataloader returns `examples: List[dict]` where each dict has all per-sample fields (`canonical_state`, `pose_gt`, `image_future`, etc.). Framework's `forward` does the stacking inline via `_collate_for_heads`. This keeps the starVLA `examples` contract and matches `QwenGR00T.forward`. The 4 stacking helpers (`stack_canonical`, `stack_optional_tensor_fields`, `stack_pose_gt`, `stack_static_cam_extrinsic`) are ported from UamVLA `uamvla/data/collator.py:17-281` into a new sibling module `starVLA/model/modules/uamvla/collator_helpers.py` (see Step 3) so the framework class stays clean.

- [ ] **Step 3: Port collator helpers to a sibling module**

```bash
# Extract the _stack_canonical and optional-field stacking logic from UamVLA collator.py
# into a new file: starVLA/model/modules/uamvla/collator_helpers.py
```

Functions to port (from UamVLA `uamvla/data/collator.py`):
- `_stack_canonical(state_list)` — `:17-39`
- Optional-tensor stacking pattern — `:213-281` (refactor into `stack_optional_fields(samples, field_names) -> dict`)
- `f"{name}_mask"` derivation — `:233-236`

Run: `grep -n "^def\|^class" /Users/tancilon/develop/localgit/UamVLA/uamvla/data/collator.py` to enumerate.

- [ ] **Step 4: Use helpers in `UamVLA._collate_for_heads`**

Replace the `NotImplementedError` placeholder with calls to the new helpers.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py starVLA/model/modules/uamvla/collator_helpers.py tests/test_uamvla_forward.py
git commit -m "[framework] Implement UamVLA.forward with multi-aux-head loss summing"
```

---

### Task 19: Implement `UamVLA.predict_action`

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLA.py` — add `predict_action`

- [ ] **Step 1: Write contract test**

```python
# tests/test_uamvla_predict_action.py
def test_predict_action_returns_normalized_actions_key():
    """The output dict must have 'normalized_actions' as np.ndarray with shape (B, T, 7) for Franka."""
    # Full test deferred to L5 eval dry-run. Document the contract here.
    expected_keys = {"normalized_actions"}
    expected_shape_suffix = (7,)  # last dim must be 7 for franka_libero
    # Placeholder — verified at L5
```

- [ ] **Step 2: Implement `predict_action`**

```python
    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]

        from deployment.model_server.tools.image_tools import to_pil_preserve

        qwen_inputs = self.qwen_vl_interface.build_inputs(
            images=[[to_pil_preserve(img) for img in e["image"]] for e in examples],
            instructions=[e["lang"] for e in examples],
            canonical_state=stack_canonical([e["canonical_state"] for e in examples]),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
            hidden = backbone_out.hidden_states[-1]

        # Action head only — pose/future/recon don't run at eval
        pred_actions = self.aux_heads["action"].predict(hidden, batch=qwen_inputs)
        normalized_actions = self._decode_action_tokens(pred_actions, batch_size=len(examples))
        return {"normalized_actions": normalized_actions}

    def _decode_action_tokens(self, head_output, batch_size: int) -> np.ndarray:
        """Decode (B, L) argmax token ids → (B, T=action_horizon, 7) normalized actions.

        head_output is HeadOutput(predictions={"token_ids": Tensor(B, L)}) from ActionHead.predict.
        For Phase 1, ship a structure-correct stub that raises NotImplementedError pointing at
        Task 24 (dataloader must produce `labels` mask) and Task 33 (L5 eval dry-run will provide
        a real fixture and complete this method).
        """
        action_tokenizer = ActionTokenizer(self.qwen_vl_interface.tokenizer)
        pred_ids = head_output.predictions["token_ids"]  # (B, L)
        H = self.action_horizon  # T

        decoded = np.zeros((batch_size, H, 7), dtype=np.float32)
        for i in range(batch_size):
            token_ids_i = self._extract_action_tokens_for_sample(pred_ids[i], H)
            chunk = action_tokenizer.decode(token_ids_i).reshape(H, 7)
            decoded[i] = chunk
        return decoded

    def _extract_action_tokens_for_sample(self, pred_ids_row, action_horizon: int):
        """Find the H*7 action token positions in pred_ids_row (a (L,) tensor).

        Phase 1: requires upstream batch to provide either `labels` mask (preferred) or a
        known `action_token_begin_id`. Wired up via Task 24 (dataloader plugin) and
        completed via Task 33 (L5 eval dry-run).
        """
        raise NotImplementedError(
            "predict_action decoding requires labels mask or action_token_begin_id "
            "from the dataloader. See spec §4.3 — completed at Task 33 (L5 eval)."
        )
```

- [ ] **Step 3: Port `_decode_action_tokens` from `UamVLA/uamvla/eval_runners/base_runner.py`**

Read the relevant decoding section in `base_runner.py` (search for `action_tokenizer.decode` and `decode_chunk`). Adapt to operate on raw model output rather than UamVLA's HF generate path.

For Phase 1, the simplest decoder: take `action_head.predict(hidden)` predictions which already contain `token_ids`, then for each batch sample group consecutive 7-dim chunks, run `ActionTokenizer.decode(chunk_tokens) → (7,)` per timestep.

- [ ] **Step 4: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py tests/test_uamvla_predict_action.py
git commit -m "[framework] Implement UamVLA.predict_action returning normalized_actions"
```

---

### Task 20: Implement `UamVLA.visualize_batch`

- [ ] **Step 1: Add method**

Insert into `UamVLA.py`:

```python
    def visualize_batch(self, batch: dict, n_samples: int = 1) -> dict:
        """Iterate aux heads and call .visualize() on each, gathering wandb.Image entries."""
        out = {}
        # Recompute hidden states (the trainer doesn't always persist them)
        hidden = self._backbone_forward_for_viz(batch)
        for name, head in self.aux_heads.items():
            if not hasattr(head, "visualize"):
                continue
            mask = batch.get(f"{name}_mask")
            if mask is None or mask.any():
                imgs = head.visualize(hidden, batch, mask, num_samples=n_samples)
                for i, img in enumerate(imgs):
                    out[f"viz/{name}/{i}"] = img
        return out

    def _backbone_forward_for_viz(self, batch):
        """Run backbone forward on a viz batch — returns hidden states only."""
        # If batch has 'examples' list, use it; else assume batch is already-stacked dict
        if "examples" in batch:
            qwen_inputs = self.qwen_vl_interface.build_inputs(
                images=[e["image"] for e in batch["examples"]],
                instructions=[e["lang"] for e in batch["examples"]],
                canonical_state=[e["canonical_state"] for e in batch["examples"]],
            )
        else:
            qwen_inputs = batch
        with torch.no_grad():
            backbone_out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
        return backbone_out.hidden_states[-1]
```

- [ ] **Step 2: Smoke test (no real backbone)**

The visualize hook is exception-safe in the trainer (Task 8 step 2). Phase 1 verification: trainer runs without crashing if visualize_batch raises. Defer real viz testing to L4.

- [ ] **Step 3: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py
git commit -m "[framework] Implement UamVLA.visualize_batch dispatching to per-head .visualize"
```

---

### Task 21: Implement `UamVLA.get_lr_groups` and `supports_training_tag`

- [ ] **Step 1: Add both methods**

```python
    def get_lr_groups(self, lr_cfg) -> list:
        # Exclude state_encoder params from the qwen_vl_interface group to avoid
        # double-counting (state_encoder is a submodule of qwen_vl_interface).
        state_enc_params = set(id(p) for p in self.qwen_vl_interface.state_encoder.parameters())
        qwen_params = [p for p in self.qwen_vl_interface.parameters() if id(p) not in state_enc_params]
        groups = [
            {"name": "qwen_vl_interface",
             "params": qwen_params,
             "lr": float(lr_cfg.qwen_vl_interface)},
            {"name": "state_encoder",
             "params": list(self.qwen_vl_interface.state_encoder.parameters()),
             "lr": float(lr_cfg.state_encoder)},
        ]
        for name, head in self.aux_heads.items():
            head_cfg = self.config.framework.aux_heads[name]
            groups.append({
                "name": f"aux_head_{name}",
                "params": list(head.parameters()),
                "lr": float(head_cfg.get("lr", lr_cfg.base)),
            })
        return groups

    def supports_training_tag(self, tag: str) -> bool:
        # Phase 1: VLA only, no VLM co-training
        return tag == "vla"
```

- [ ] **Step 2: Add unit test for `get_lr_groups` returning correct group names**

```python
# tests/test_uamvla_lr_groups.py
import torch.nn as nn
from unittest.mock import MagicMock


def test_get_lr_groups_returns_expected_names(monkeypatch):
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA
    # Mock backbone build to avoid HF download
    import starVLA.model.modules.uamvla.backbone_wrapper as bw

    fake_backbone = MagicMock()
    fake_backbone.config.hidden_size = 4096
    fake_backbone.parameters = lambda: iter([nn.Parameter()])
    fake_backbone.state_encoder.parameters = lambda: iter([nn.Parameter()])
    monkeypatch.setattr(bw, "build_uamvla_backbone", lambda cfg: fake_backbone)
    # Disable aux heads needing VAE
    cfg = {"framework": {"aux_heads": {"action": {"enabled": True, "lr": 1e-4},
                                        "pose": {"enabled": False},
                                        "future": {"enabled": False},
                                        "recon": {"enabled": False}}}}
    model = UamVLA(config=cfg)
    lr_cfg = MagicMock()
    lr_cfg.qwen_vl_interface = 2e-5
    lr_cfg.state_encoder = 1e-4
    lr_cfg.base = 1e-4
    groups = model.get_lr_groups(lr_cfg)
    names = {g["name"] for g in groups}
    assert "qwen_vl_interface" in names
    assert "state_encoder" in names
    assert "aux_head_action" in names
```

- [ ] **Step 3: Run + commit**

Run: `pytest tests/test_uamvla_lr_groups.py -v`

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py tests/test_uamvla_lr_groups.py
git commit -m "[framework] Implement UamVLA.get_lr_groups and supports_training_tag"
```

---

### Task 22: L1 Smoke — full UamVLA framework init (mocked backbone)

**Files:**
- Create: `tests/test_uamvla_l1_init.py`

- [ ] **Step 1: Write end-to-end init smoke test**

```python
# tests/test_uamvla_l1_init.py
"""L1 smoke (Mac local): UamVLA framework init succeeds with mocked backbone.

Full backbone load happens only on remote with checkpoint available.
"""
import pytest
from unittest.mock import MagicMock


def test_uamvla_init_with_full_aux_heads_mocked(monkeypatch):
    pytest.importorskip("torch")
    pytest.skip("L1 init with all aux heads requires VAE ckpt — run on remote in WP8 L1 task")
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_uamvla_l1_init.py
git commit -m "[tests] Document L1 init smoke (deferred to remote with full ckpt)"
```

---

## Phase 1E: Data Pipeline (WP1 + WP5 mixture)

### Task 23: Implement `uamvla_dataset.py` UamVLADataset class

**Files:**
- Create: `starVLA/dataloader/uamvla_dataset.py`
- Create: `tests/test_uamvla_dataset.py`

- [ ] **Step 1: Write test for `UamVLADataset.__getitem__` shape contract using mock JSONL**

```python
# tests/test_uamvla_dataset.py
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def mock_jsonl_dir(tmp_path):
    """Create a minimal preprocessor-output-shaped directory."""
    sample = {
        "id": "test_00_ep0000_step0000",
        "episode_id": "test_00_ep0000",
        "step_idx": 0,
        "total_steps": 5,
        "image": ["images/obs/static/test_00_ep0000_step0000.jpg",
                  "images/obs/wrist/test_00_ep0000_step0000.jpg"],
        "instruction": "test task",
        "embodiment": "franka_libero",
        "action_dim": 7,
        "action": [0.1] * 7 + [0.0] * 17,
        "action_mask": [1] * 7 + [0] * 17,
        "ee_pos": [0.0, 0.0, 0.0],
        "ee_axis_angle": [0.0, 0.0, 0.0],
        "joint_pos": [0.0] * 7,
        "gripper_qpos": [0.04, 0.04],
        "dataset_source": "test",
    }
    (tmp_path / "data.jsonl").write_text(json.dumps(sample) + "\n")
    stats = {
        "view_names": ["static", "wrist"],
        "max_action_dim": 24,
        "embodiment_stats": {
            "franka_libero": {
                "action_dim": 7,
                "action_min_bound": [-1.0] * 7,
                "action_max_bound": [1.0] * 7,
            },
        },
        "robot_obs_mean": [0.0, 0.0, 0.0],
        "robot_obs_std": [1.0, 1.0, 1.0],
    }
    import yaml
    (tmp_path / "statistics.yaml").write_text(yaml.dump(stats))
    return tmp_path


def test_dataset_getitem_yields_expected_fields(mock_jsonl_dir, monkeypatch):
    from starVLA.dataloader.uamvla_dataset import UamVLADataset

    # Mock image loading (real .jpg fixtures not present)
    from PIL import Image
    monkeypatch.setattr(Image, "open", lambda p: Image.new("RGB", (640, 640)))

    ds = UamVLADataset(data_root=mock_jsonl_dir, embodiment="franka_libero", action_horizon=8)
    sample = ds[0]
    # Required fields per spec §4.6
    assert "image" in sample and len(sample["image"]) == 2
    assert "lang" in sample
    assert "action" in sample and sample["action"].shape == (8, 7)
    assert "canonical_state" in sample
    assert "arm_0" in sample["canonical_state"]
    assert "ee_pose" in sample["canonical_state"]["arm_0"]
    # Action normalization: with min=-1, max=1, normalized = action (no change in [-1,1] target range)
    np.testing.assert_allclose(sample["action"][0], [0.1] * 7, atol=1e-5)
```

- [ ] **Step 2: Run test — should fail**

Run: `pytest tests/test_uamvla_dataset.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement `UamVLADataset`**

```python
# starVLA/dataloader/uamvla_dataset.py
"""UamVLA dataloader plugin: reads UamVLA preprocessor JSONL → starVLA examples list.

Plugin entry: get_vla_dataset(data_cfg) — invoked by starVLA trainer when
datasets.vla_data.dataset_py == "uamvla_dataset".

Spec: docs/superpowers/specs/2026-04-29-starvla-migration-design.md §5
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from starVLA.model.modules.uamvla.data.embodiment_registry import get_embodiment_config
from starVLA.utils.point_cloud import clean_point_cloud


REQUIRED_OPTIONAL_FIELDS = ("image_target", "image_future", "point_cloud")


DATASET_NAMED_MIXTURES = {
    "libero_uamvla": [
        # (data_subdir, weight, embodiment_tag)
        ("libero_spatial", 1.0, "franka_libero"),
        # Phase 2: + libero_object, libero_goal, libero_10
    ],
}


class UamVLADataset(Dataset):
    """Read UamVLA JSONL → produce starVLA-compatible examples dict per __getitem__."""

    def __init__(
        self,
        data_root: Path | str,
        embodiment: str = "franka_libero",
        action_horizon: int = 8,
        max_samples: Optional[int] = None,
        transforms=None,
    ):
        self.data_root = Path(data_root)
        self.embodiment = embodiment
        self.action_horizon = action_horizon
        self.transforms = transforms

        # Load samples
        with open(self.data_root / "data.jsonl") as f:
            all_samples = [json.loads(l) for l in f if l.strip()]
        # Filter to those with required optional fields (mirrors UamVLA base_dataset.py:46-55)
        self.samples = [s for s in all_samples
                        if all(k in s for k in REQUIRED_OPTIONAL_FIELDS)] or all_samples
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Load stats
        with open(self.data_root / "statistics.yaml") as f:
            self.stats = yaml.safe_load(f)
        emb_stats = self.stats["embodiment_stats"][embodiment]
        self.action_min = np.array(emb_stats["action_min_bound"], dtype=np.float32)
        self.action_max = np.array(emb_stats["action_max_bound"], dtype=np.float32)
        self.view_names = list(self.stats.get("view_names", ["static", "wrist"]))

        # Embodiment adapter (stateless singleton)
        self.adapter = get_embodiment_config(embodiment)["adapter"]

        # Episode index for action chunk slicing
        self._episode_index = {(s["episode_id"], int(s["step_idx"])): i
                               for i, s in enumerate(self.samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        raw = self.samples[idx]

        # Load multi-view images
        images = []
        for img_path in raw["image"]:
            img = Image.open(self.data_root / img_path).convert("RGB")
            if self.transforms:
                img = self.transforms(img)
            images.append(img)

        # Action chunk (H, 7), normalized
        action, action_mask = self._slice_action_chunk(raw)

        # Canonical state via adapter
        canonical_state = self.adapter.to_canonical(raw)

        sample = {
            "image": images,
            "lang": raw["instruction"],
            "action": action,
            "action_mask": action_mask,
            "canonical_state": canonical_state,
            "embodiment": self.embodiment,
            "view_names": self.view_names,
        }

        # Optional aux head targets (loaded if present in JSONL)
        self._load_aux_targets(raw, sample)
        return sample

    def _slice_action_chunk(self, raw):
        H = self.action_horizon
        eid = raw["episode_id"]
        t = int(raw["step_idx"])
        T = int(raw["total_steps"])
        action = np.zeros((H, 7), dtype=np.float32)
        mask = np.zeros((H, 7), dtype=np.int64)

        for k in range(H):
            future_t = t + k
            if future_t >= T:
                break
            row_idx = self._episode_index.get((eid, future_t))
            if row_idx is None:
                break
            future_raw = self.samples[row_idx]
            # Trim 24D → 7D, then min-max normalize to [-1, 1]
            raw_action = np.array(future_raw["action"][:7], dtype=np.float32)
            normalized = self._normalize_action(raw_action)
            action[k] = normalized
            mask[k] = np.array(future_raw["action_mask"][:7], dtype=np.int64)
        return torch.tensor(action, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

    def _normalize_action(self, action):
        """min-max → [-1, 1]."""
        denom = (self.action_max - self.action_min) + 1e-8
        return 2.0 * (action - self.action_min) / denom - 1.0

    def _load_aux_targets(self, raw, sample):
        """Load image_target / image_future / point_cloud / pose_gt / static_cam_extrinsic if present."""
        if "image_target" in raw and raw["image_target"]:
            try:
                img = Image.open(self.data_root / raw["image_target"]).convert("RGB")
                if self.transforms:
                    img = self.transforms(img)
                sample["image_target"] = img
            except FileNotFoundError:
                pass
        if "image_future" in raw and raw["image_future"]:
            try:
                img = Image.open(self.data_root / raw["image_future"]).convert("RGB")
                if self.transforms:
                    img = self.transforms(img)
                sample["image_future"] = img
            except FileNotFoundError:
                pass
        if "pose_6d" in raw:
            sample["pose_gt"] = {
                "rotation":    torch.tensor(raw["pose_6d"]["rotation"], dtype=torch.float32),
                "translation": torch.tensor(raw["pose_6d"]["translation"], dtype=torch.float32),
            }
        if "static_cam_extrinsic" in raw:
            rot_flat = torch.tensor(raw["static_cam_extrinsic"]["rotation"], dtype=torch.float32)
            sample["static_cam_extrinsic"] = {
                "rotation": rot_flat.view(3, 3),
                "translation": torch.tensor(raw["static_cam_extrinsic"]["translation"], dtype=torch.float32),
            }
        if "point_cloud" in raw and raw["point_cloud"]:
            try:
                pts = np.load(self.data_root / raw["point_cloud"])
                pts = clean_point_cloud(pts)
                sample["point_cloud"] = torch.tensor(pts, dtype=torch.float32)
            except FileNotFoundError:
                pass


def get_vla_dataset(data_cfg, mode: str = "train", **kwargs) -> Dataset:
    """starVLA plugin entry point. Returns a Dataset that yields starVLA examples dicts."""
    # data_cfg.data_mix is the mixture name; resolve to subdirs
    data_root = Path(data_cfg.data_root_dir)
    mixture = DATASET_NAMED_MIXTURES.get(data_cfg.data_mix)
    if mixture is None:
        raise ValueError(f"Unknown data_mix '{data_cfg.data_mix}'. Available: {list(DATASET_NAMED_MIXTURES)}")

    # Phase 1: single embodiment, single subdir; multi-subdir support deferred
    if len(mixture) == 1:
        subdir, _weight, embodiment = mixture[0]
        return UamVLADataset(
            data_root=data_root / subdir,
            embodiment=embodiment,
            action_horizon=int(data_cfg.get("action_horizon", 8)),
        )
    # Phase 2: ConcatDataset across multiple subdirs with weights
    raise NotImplementedError("Multi-subdir mixture deferred to Phase 2")
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_uamvla_dataset.py -v`
Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
git add starVLA/dataloader/uamvla_dataset.py tests/test_uamvla_dataset.py
git commit -m "[dataloader] Add UamVLA JSONL dataloader plugin"
```

---

### Task 24: L3 dataloader smoke — collate function emits expected batch dict

**Files:**
- Modify: `starVLA/dataloader/uamvla_dataset.py` — add `collate_fn`
- Add to: `tests/test_uamvla_dataset.py`

- [ ] **Step 1: Write test for collate output**

```python
def test_uamvla_collate_fn_stacks_correctly(mock_jsonl_dir, monkeypatch):
    from starVLA.dataloader.uamvla_dataset import UamVLADataset, collate_fn
    from PIL import Image
    monkeypatch.setattr(Image, "open", lambda p: Image.new("RGB", (640, 640)))

    ds = UamVLADataset(mock_jsonl_dir, embodiment="franka_libero", action_horizon=8)
    samples = [ds[0]]  # batch of 1
    batch = collate_fn(samples)
    # Phase 1: collate returns the list (framework handles stacking) — verify it's a list
    assert isinstance(batch, list)
    assert len(batch) == 1
```

- [ ] **Step 2: Add `collate_fn` (passthrough — framework does stacking)**

In `uamvla_dataset.py`:

```python
def collate_fn(batch):
    """Pass through. The framework's forward() handles stacking via collator_helpers.

    starVLA convention (see lerobot_datasets.py:19-20): trivial list passthrough.
    """
    return batch
```

- [ ] **Step 3: Run + commit**

Run: `pytest tests/test_uamvla_dataset.py -v`

```bash
git add starVLA/dataloader/uamvla_dataset.py tests/test_uamvla_dataset.py
git commit -m "[dataloader] Add collate_fn passthrough (stacking lives in framework)"
```

---

### Task 25: Add `dataset_statistics.json` export at trainer startup

**Files:**
- Modify: `starVLA/training/train_starvla.py` — `setup_directories` or `_save_initial_configs`

- [ ] **Step 1: Identify export point**

Find `_save_initial_configs` in `train_starvla.py` (around line 165). Add stats export after the YAML config save.

- [ ] **Step 2: Add export logic**

```python
    def _save_dataset_statistics_json(self):
        """Convert UamVLA-style statistics.yaml → starVLA dataset_statistics.json schema."""
        if not self.accelerator.is_main_process:
            return
        # Read source statistics.yaml from the dataset's data_root_dir
        from pathlib import Path
        import yaml, json

        data_root = Path(self.config.datasets.vla_data.data_root_dir)
        # Pick first subdir matching mixture (Phase 1: single subdir)
        from starVLA.dataloader.uamvla_dataset import DATASET_NAMED_MIXTURES
        mixture = DATASET_NAMED_MIXTURES.get(self.config.datasets.vla_data.data_mix)
        if mixture is None:
            return  # Skip silently for non-UamVLA datasets
        subdir, _w, embodiment = mixture[0]
        stats_yaml = data_root / subdir / "statistics.yaml"
        if not stats_yaml.exists():
            return

        with open(stats_yaml) as f:
            src = yaml.safe_load(f)
        emb_stats = src["embodiment_stats"][embodiment]
        # mask: True for min-max dims, False for binarized (gripper at dim 6)
        mask = [True] * 6 + [False]
        out = {
            embodiment: {
                "action": {
                    "min":  emb_stats["action_min_bound"],
                    "max":  emb_stats["action_max_bound"],
                    "mask": mask,
                }
            }
        }
        out_path = Path(self.config.output_dir) / "dataset_statistics.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Wrote dataset_statistics.json to {out_path}")
```

Call from `_save_initial_configs`:

```python
    def _save_initial_configs(self):
        # ... existing config save ...
        self._save_dataset_statistics_json()
```

- [ ] **Step 3: Smoke test (skipped — runs on remote)**

Document this is verified at L4 50-step run (after first save_interval, ckpt dir contains the JSON).

- [ ] **Step 4: Commit**

```bash
git add starVLA/training/train_starvla.py
git commit -m "[trainer] Export dataset_statistics.json on init for eval client un-normalize"
```

---

## Phase 1F: Configuration (WP4)

### Task 26: Create `uamvla_libero.yaml` training config

**Files:**
- Create: `starVLA/config/training/uamvla_libero.yaml`

- [ ] **Step 1: Write the YAML**

Copy the full YAML from spec §6 verbatim into `starVLA/config/training/uamvla_libero.yaml`. Verify structure with:

```bash
python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('starVLA/config/training/uamvla_libero.yaml'); print(cfg.framework.name, cfg.datasets.vla_data.dataset_py, cfg.trainer.max_train_steps)"
```
Expected: `UamVLA uamvla_dataset 100000`

- [ ] **Step 2: Commit**

```bash
git add starVLA/config/training/uamvla_libero.yaml
git commit -m "[config] Add uamvla_libero.yaml Phase 1 training config"
```

---

### Task 27: Edit `run_libero_train.sh` in-place

**Files:**
- Modify: `examples/LIBERO/train_files/run_libero_train.sh`

- [ ] **Step 1: Edit the marked block**

Apply the diff from spec §8.3 (Framework_name=UamVLA, base_vlm, config_yaml, libero_data_root, data_mix, run_id, per_device_batch_size, wandb_project, wandb_entity).

- [ ] **Step 2: Smoke check the script parses**

Run: `bash -n examples/LIBERO/train_files/run_libero_train.sh`
Expected: no syntax error.

- [ ] **Step 3: Commit**

```bash
git add examples/LIBERO/train_files/run_libero_train.sh
git commit -m "[examples] Configure run_libero_train.sh for UamVLA framework"
```

---

### Task 28: Edit `run_policy_server.sh` and `eval_libero.sh` in-place

**Files:**
- Modify: `examples/LIBERO/eval_files/run_policy_server.sh`
- Modify: `examples/LIBERO/eval_files/eval_libero.sh`

- [ ] **Step 1: Apply diffs from spec §8.3**

For server: STARVLA_DIR, LIBERO_HOME, STARVLA_PYTHON, LIBERO_PYTHON, CKPT.
For client: STARVLA_DIR, CKPT, LIBERO_HOME, LIBERO_Python, task_suite_name=libero_spatial.

- [ ] **Step 2: Smoke check**

Run: `bash -n examples/LIBERO/eval_files/run_policy_server.sh examples/LIBERO/eval_files/eval_libero.sh`
Expected: no syntax errors.

- [ ] **Step 3: Commit**

```bash
git add examples/LIBERO/eval_files/run_policy_server.sh examples/LIBERO/eval_files/eval_libero.sh
git commit -m "[examples] Configure eval wrappers for UamVLA (server + client)"
```

---

## Phase 1G: Remote Preprocessing & Training (WP8 L2 / L4)

### Task 29: Push branch and sync remote workspace

- [ ] **Step 1: Push branch**

```bash
git push -u origin feat/uamvla-migration
```

- [ ] **Step 2: SSH to remote, clone or pull UniamVLA**

(User executes — Claude can not SSH.)

```bash
# On remote server (per UamVLA CLAUDE.md path convention):
ssh <remote_host>
cd /inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/
git clone https://github.com/Tancilon/UniamVLA.git    # if first time
cd UniamVLA
git fetch && git checkout feat/uamvla-migration && git pull
```

- [ ] **Step 3: Verify remote env has uamvla conda + dependencies**

```bash
conda activate uamvla
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
# Expected: PyTorch 2.x, transformers >= 4.57.0
```

- [ ] **Step 4: Confirm GPU availability**

```bash
nvidia-smi
```
Expected: 8 GPUs (H100 or H200) free.

---

### Task 30: Run preprocessing on remote (libero_spatial)

- [ ] **Step 1: Activate uamvla env on remote**

```bash
conda activate uamvla
```

- [ ] **Step 2: Run preprocessor**

```bash
cd /inspire/.../UniamVLA
python runners/preprocess_libero.py \
    --suite libero_spatial \
    --input_dir /inspire/.../LIBERO/datasets/libero_spatial \
    --output_dir /inspire/.../UniamVLA/datasets/uamvla_libero/libero_spatial
```

(Adapt CLI flags to match the runner's argparse — `python runners/preprocess_libero.py --help` first.)

Expected output: `data.jsonl` + `images/` + `point_clouds/` + `depth/` + `statistics.yaml` in output_dir.

- [ ] **Step 3: Sanity-check the preprocessor output**

```bash
ls /inspire/.../UniamVLA/datasets/uamvla_libero/libero_spatial/
wc -l /inspire/.../UniamVLA/datasets/uamvla_libero/libero_spatial/data.jsonl
head -1 /inspire/.../UniamVLA/datasets/uamvla_libero/libero_spatial/data.jsonl | python -c "import sys, json; s = json.loads(sys.stdin.read()); print('keys:', sorted(s.keys()))"
```

Expected fields per spec §6.3 + libero_preprocessor: id, episode_id, image, instruction, action, action_mask, ee_pos, ee_axis_angle, joint_pos, gripper_qpos, pose_6d, point_cloud, image_target, image_future, static_cam_extrinsic.

- [ ] **Step 4: Verify image flip applied (visual check on one .jpg)**

```bash
# Compare to one UamVLA-preprocessed image with old flip — should be 180° rotated relative to old data
```

- [ ] **Step 5: Commit a tag/log of preprocessing run**

(No code commit — preprocessing artifacts live in datasets/, which is .gitignored.)

---

### Task 31: L2 smoke — 1-step training on remote

- [ ] **Step 1: Run training with `--max_train_steps 1`**

```bash
cd /inspire/.../UniamVLA
bash examples/LIBERO/train_files/run_libero_train.sh
```

But override `max_train_steps` to 1 by editing the script's accelerate args temporarily, OR pass an extra CLI override:

```bash
# Edit run_libero_train.sh to add: --trainer.max_train_steps 1
# Run, observe logs.
```

- [ ] **Step 2: Verify forward succeeds, no NaN**

Watch for:
- `action_loss: <finite>` in stdout
- `pose_loss / future_loss / recon_loss` all logged as finite values (per spec §10.2)
- No DDP / DeepSpeed init errors
- Wandb run started

If NaN at step 1: see spec §10.3 first-check column.

- [ ] **Step 3: Verify checkpoint structure if save fires**

If `save_interval` triggered (set to small value for smoke), check output dir:

```bash
ls /inspire/.../UniamVLA/playground/Checkpoints/uamvla_libero_phase1/
# Expected: config.full.yaml, config.yaml, dataset_statistics.json, copied launcher script, checkpoints/
```

- [ ] **Step 4: Commit (no code change; log only)**

If issues found: fix them in earlier tasks (this is the integration stage).

---

### Task 32: L4 smoke — 50-step training

- [ ] **Step 1: Run with `--max_train_steps 50`**

Same script, override max_train_steps=50. Monitor wandb for ~5-15 min.

- [ ] **Step 2: Verify loss trajectory**

- `action_loss` decreases ≥ 5% from step 1 to step 50
- per-head losses bounded, none NaN
- viz hook fires (if `train_every_n_steps: 1000`, won't fire in 50 steps — OK; reduce to 25 for this smoke)

- [ ] **Step 3: Re-run with max_train_steps=2000 to trigger viz at step 1000**

Verify wandb shows `viz/pose/0`, `viz/future/0`, `viz/recon/0` images.

- [ ] **Step 4: Commit fixes if any**

---

## Phase 1H: Eval & Acceptance (WP8 L5-L7)

### Task 33: L5 — 1-task eval dry-run

- [ ] **Step 1: Train a small checkpoint (~10k steps) on remote**

```bash
# Override max_train_steps=10000 in run_libero_train.sh
bash examples/LIBERO/train_files/run_libero_train.sh
# Wait ~1-2 hours
```

- [ ] **Step 2: Update eval script with the smoke ckpt path**

Edit `run_policy_server.sh` and `eval_libero.sh` to point CKPT at the 10k step ckpt produced.

- [ ] **Step 3: Start server in `uamvla` env**

Terminal A:

```bash
conda activate uamvla
bash examples/LIBERO/eval_files/run_policy_server.sh
```
Wait for "Server listening on port 6694".

- [ ] **Step 4: Start client in `libero_env` (separate terminal)**

Terminal B:

```bash
conda activate libero_env
# Edit eval_libero.sh: --num_trials_per_task 1
bash examples/LIBERO/eval_files/eval_libero.sh
```

- [ ] **Step 5: Verify**

- 1 episode runs to completion (success or failure both OK at this stage)
- Replay video saved to results dir
- No WebSocket / dtype / chunk_size errors

If SR=0 across all attempts: see spec §10.3 "Eval SR=0" row.

---

### Task 34: L6 — partial 10-task eval

- [ ] **Step 1: Edit `eval_libero.sh` for 5 trials per task**

```bash
# --num_trials_per_task 5
```

- [ ] **Step 2: Run eval (server + client as in Task 33)**

Wait ~1-2 hours.

- [ ] **Step 3: Inspect partial SR**

```bash
grep "Total success rate" /path/to/eval/log
```

If partial SR ≥ ~50%, full L7 should achieve ≥ 60%. If <30%, root-cause via failure-mode checklist before launching expensive L7.

---

### Task 35: L7 — full Phase 1 acceptance run

- [ ] **Step 1: Full training run (100k steps)**

```bash
# run_libero_train.sh with max_train_steps=100000 (default in YAML)
bash examples/LIBERO/train_files/run_libero_train.sh
```

Wall-clock target: ~12-24 hours on 8×H100.

- [ ] **Step 2: Pick best checkpoint**

Eval at multiple ckpts (50k, 75k, 100k) — typically later is better but check loss flatness.

- [ ] **Step 3: Run full L7 eval**

```bash
# eval_libero.sh with --num_trials_per_task 50
bash examples/LIBERO/eval_files/eval_libero.sh
```

Wait ~6-12 hours (10 task × 50 trial × 220 max steps).

- [ ] **Step 4: Acceptance check**

| Metric | Threshold | Result |
|---|---|---|
| LIBERO Spatial 10×50 SR | ≥ 60% | (record) |
| All per-head losses converge | yes | (record) |
| Reproducibility — `from_pretrained(ckpt)` reload works | yes | (verify) |

If SR < 60%: root-cause via spec §10.3, fix, re-train relevant ckpt range or full re-run.

- [ ] **Step 5: Commit final results summary**

Write a brief markdown to `docs/superpowers/results/2026-XX-XX-phase1-libero-spatial-results.md` with the final SR, training time, ckpt path, eval video paths.

```bash
git add docs/superpowers/results/
git commit -m "[results] Record Phase 1 LIBERO Spatial acceptance run results"
```

- [ ] **Step 6: Tag release**

```bash
git tag phase1-libero-acceptance
git push origin feat/uamvla-migration --tags
```

---

## Self-Review Checklist

After implementation:

- [ ] All commits use `tancilon` git identity, no Claude/Anthropic mentions anywhere
- [ ] starVLA mainline patches (`base_framework.py`, `train_starvla.py`) are minimal-additive — existing framework classes (e.g., QwenAdapter) still train successfully if invoked
- [ ] `model2libero_interface.py` and `eval_libero.py` Python files have ZERO modifications
- [ ] All tests in `tests/` pass on Mac local (skipping CUDA-bound tests)
- [ ] L7 LIBERO Spatial SR ≥ 60% on full 10×50 trial run
- [ ] `dataset_statistics.json` is auto-saved to ckpt dir at trainer init
- [ ] Run dir from L7 contains: `config.full.yaml`, `config.yaml`, `dataset_statistics.json`, copied `run_libero_train.sh`
- [ ] `from_pretrained(<ckpt>)` reload succeeds end-to-end (round-trip test on Mac with mocked backbone)

---

## Out-of-Scope Reminders

These are Phase 2+ — explicitly NOT in this plan:

- CALVIN preprocessing/eval (CALVIN preprocessor files ported but unused)
- SimplerEnv / RoboTwin / VLA-Arena
- True 24D unified-action runtime
- Cross-embodiment heterogeneous-batch co-training
- VLM co-training (`forward_vlm`)
- Action ensemble enabling (Phase 1.5)
