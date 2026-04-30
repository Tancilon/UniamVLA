# State Input Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenVLA-OFT-style `BOUNDS_Q99` input normalization to UniamVLA's canonical_state pipeline so per-dimension state values land in `[-1, 1]` before reaching the state encoder. Encoder structure stays untouched.

**Architecture:** Per-field statistics (q01/q99/min/max/mean/std) computed in `LiberoPreprocessor` over the canonical representation, written to `statistics.yaml` under a new `state_stats` block. A new `StateNormalizer` (composing the existing `Normalizer` class from `state_action.py`) walks the canonical_state dict and applies per-field q99 normalization with clip. Wired into `UamVLADataset.__getitem__` after the embodiment adapter.

**Tech Stack:** PyTorch, NumPy, PyYAML, pytest. Follows existing UamVLA conventions (Pydantic transforms in `gr00t_lerobot`, embodiment registry, dotted-path config).

**Spec:** `docs/superpowers/specs/2026-04-30-state-input-normalization-design.md`

**Reference:** OpenVLA-OFT clone at `/Users/tancilon/develop/localgit/openvla-oft` — see `prismatic/models/projectors.py` and `prismatic/vla/datasets/rlds/utils/data_utils.py`.

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `tools/preprocess/libero_preprocessor.py` | Modify (lines 665-715) | Compute per-field canonical-space stats via `LiberoAdapter`; write `state_stats` block; remove `robot_obs_mean/std` |
| `starVLA/model/modules/uamvla/data/state_normalizer.py` | **Create** | `StateNormalizer` class: load stats, instantiate per-field `Normalizer`, walk canonical dict |
| `starVLA/dataloader/uamvla_dataset.py` | Modify | Construct `StateNormalizer` in `__init__`, call in `__getitem__`, plumb config |
| `starVLA/config/training/uamvla_libero.yaml` | Modify | Add `framework.state_encoder.normalization` block |
| `starVLA/model/modules/uamvla/data/statistics.py` | Modify | Drop deprecated `robot_obs_mean/std` defaults |
| `tests/conftest.py` | Modify | Drop 3-dim `robot_obs_mean/std` from `sample_dataset_dir` fixture |
| `tests/test_data.py` | Modify | Drop `robot_obs_mean` fixture entries |
| `tests/test_state_normalizer.py` | **Create** | Unit tests for `StateNormalizer` |
| `tests/test_preprocessor_smoke.py` | Modify | Assert new `state_stats` schema after preprocess |
| `tests/test_uamvla_dataset.py` | Modify | Assert post-norm canonical_state values within `[-1, 1]` |
| `datasets/uamvla_test/libero_spatial/statistics.yaml` | Regenerate | Re-run preprocessor to produce schema-compliant stats |

---

### Task 1: Verify Normalizer.q99 already clips, sync spec

**Why:** Spec section 4.3 claims `Normalizer.mode == "q99"` does not clip and proposes adding a `clip` flag. Reading `state_action.py:114-135` shows it **already** clips at line 135 (`torch.clamp(normalized, -1, 1)`). No code change needed; just update spec to record the finding.

**Files:**
- Read: `starVLA/dataloader/gr00t_lerobot/transform/state_action.py:99-135`
- Modify: `docs/superpowers/specs/2026-04-30-state-input-normalization-design.md` (section 4.3 + section 8 entry)

- [ ] **Step 1: Confirm clip is present**

Run:
```bash
sed -n '99,140p' starVLA/dataloader/gr00t_lerobot/transform/state_action.py
```
Expected: line 135 contains `normalized = torch.clamp(normalized, -1, 1)`.

- [ ] **Step 2: Audit existing q99 callers**

Run:
```bash
grep -rn '"q99"\|q99' starVLA/ tools/ tests/ --include="*.py" --include="*.yaml" | grep -v state_action.py | grep -v ".pyc"
```
Expected: zero or only test/spec references — no production caller depends on un-clipped behavior. If a caller is found that depends on `q99` without clip, halt and flag for human review.

- [ ] **Step 3: Edit spec section 4.3 to record finding**

Replace the "Two options... Decision: option (A)" paragraph in `docs/superpowers/specs/2026-04-30-state-input-normalization-design.md` with:

```markdown
**Finding (2026-04-30, during plan):** `Normalizer.q99` at `state_action.py:114-135` **already clips** to `[-1, 1]` via `torch.clamp` at line 135. The full sequence is: linear map → clamp. This matches OpenVLA-OFT's `BOUNDS_Q99` recipe end-to-end. No change to `state_action.py` is required.
```

Also remove the row `| starVLA/dataloader/gr00t_lerobot/transform/state_action.py | Add clip=True default ... |` from section 8.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-04-30-state-input-normalization-design.md
git commit -m "[spec] Record Normalizer.q99 already clips; drop state_action.py edit"
```

---

### Task 2: Implement StateNormalizer (TDD)

**Why:** Need a single object that loads `state_stats` from `statistics.yaml`, instantiates a per-field `Normalizer` from existing infrastructure, and walks the canonical_state dict applying each one.

**Files:**
- Create: `starVLA/model/modules/uamvla/data/state_normalizer.py`
- Create: `tests/test_state_normalizer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_state_normalizer.py`:

```python
"""Unit tests for StateNormalizer."""
from __future__ import annotations

import pytest
import torch

from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer


def _make_stats_dict():
    """Synthetic state_stats covering franka_libero canonical layout."""
    return {
        "state_stats": {
            "franka_libero": {
                "arm_0.ee_pose": {
                    "q01": [-1.0] * 9,
                    "q99": [ 1.0] * 9,
                    "min": [-2.0] * 9,
                    "max": [ 2.0] * 9,
                    "mean": [0.0] * 9,
                    "std": [1.0] * 9,
                },
                "arm_0.joint_pos": {
                    "q01": [-3.14] * 7,
                    "q99": [ 3.14] * 7,
                    "min": [-3.14] * 7,
                    "max": [ 3.14] * 7,
                    "mean": [0.0] * 7,
                    "std": [1.0] * 7,
                },
                "gripper_0": {
                    "q01": [0.0],
                    "q99": [1.0],
                    "min": [0.0],
                    "max": [1.0],
                    "mean": [0.5],
                    "std": [0.3],
                },
            }
        }
    }


def _make_canonical():
    return {
        "arm_0": {
            "ee_pose":   torch.zeros(9, dtype=torch.float32),
            "joint_pos": torch.zeros(7, dtype=torch.float32),
        },
        "gripper_0": torch.tensor([0.5], dtype=torch.float32),
    }


def test_q99_normalizes_zero_input_to_zero_when_symmetric():
    """With q01=-1, q99=+1, input 0 → normalized 0 (midpoint)."""
    norm = StateNormalizer(
        stats_dict=_make_stats_dict(), embodiment="franka_libero", mode="q99"
    )
    out = norm(_make_canonical())
    assert torch.allclose(out["arm_0"]["ee_pose"], torch.zeros(9), atol=1e-6)
    assert torch.allclose(out["arm_0"]["joint_pos"], torch.zeros(7), atol=1e-6)


def test_q99_clips_out_of_range_input_to_neg_one_or_one():
    """Values past q99 must be clipped to +1; values past q01 must be clipped to -1."""
    norm = StateNormalizer(
        stats_dict=_make_stats_dict(), embodiment="franka_libero", mode="q99"
    )
    canonical = _make_canonical()
    canonical["arm_0"]["ee_pose"] = torch.full((9,), 10.0)   # well past q99=+1
    canonical["arm_0"]["joint_pos"] = torch.full((7,), -10.0)  # well past q01=-3.14
    out = norm(canonical)
    assert torch.allclose(out["arm_0"]["ee_pose"], torch.ones(9))
    assert torch.allclose(out["arm_0"]["joint_pos"], -torch.ones(7))


def test_mode_none_is_identity():
    norm = StateNormalizer(
        stats_dict=_make_stats_dict(), embodiment="franka_libero", mode="none"
    )
    inp = _make_canonical()
    inp["arm_0"]["ee_pose"] = torch.full((9,), 7.0)
    out = norm(inp)
    assert torch.equal(out["arm_0"]["ee_pose"], torch.full((9,), 7.0))


def test_apply_to_filters_fields():
    """Only fields in apply_to are normalized; others pass through unchanged."""
    norm = StateNormalizer(
        stats_dict=_make_stats_dict(),
        embodiment="franka_libero",
        mode="q99",
        apply_to=["arm_0.ee_pose"],   # joint_pos and gripper_0 NOT normalized
    )
    inp = _make_canonical()
    inp["arm_0"]["joint_pos"] = torch.full((7,), 10.0)
    out = norm(inp)
    # ee_pose normalized: input 0 with q01=-1, q99=+1 → 0
    assert torch.allclose(out["arm_0"]["ee_pose"], torch.zeros(9), atol=1e-6)
    # joint_pos untouched
    assert torch.allclose(out["arm_0"]["joint_pos"], torch.full((7,), 10.0))


def test_q01_equals_q99_passes_through():
    """Constant feature (q01==q99) must be passed through unchanged then clamped to [-1,1]."""
    stats = _make_stats_dict()
    stats["state_stats"]["franka_libero"]["gripper_0"]["q01"] = [0.5]
    stats["state_stats"]["franka_libero"]["gripper_0"]["q99"] = [0.5]
    norm = StateNormalizer(stats_dict=stats, embodiment="franka_libero", mode="q99")
    inp = _make_canonical()
    inp["gripper_0"] = torch.tensor([0.5], dtype=torch.float32)
    out = norm(inp)
    # Existing Normalizer behavior: passthrough then clamp; 0.5 in range so stays 0.5
    assert torch.allclose(out["gripper_0"], torch.tensor([0.5]))


def test_missing_state_stats_raises_when_mode_not_none():
    with pytest.raises(KeyError, match="state_stats"):
        StateNormalizer(
            stats_dict={"embodiment_stats": {}}, embodiment="franka_libero", mode="q99"
        )


def test_missing_state_stats_ok_when_mode_none():
    norm = StateNormalizer(
        stats_dict={"embodiment_stats": {}}, embodiment="franka_libero", mode="none"
    )
    out = norm(_make_canonical())
    assert torch.allclose(out["arm_0"]["ee_pose"], torch.zeros(9))


def test_missing_embodiment_in_state_stats_raises():
    with pytest.raises(KeyError, match="franka_other"):
        StateNormalizer(
            stats_dict=_make_stats_dict(), embodiment="franka_other", mode="q99"
        )
```

- [ ] **Step 2: Run tests, confirm they fail with import error**

Run: `pytest tests/test_state_normalizer.py -v`
Expected: `ModuleNotFoundError: No module named 'starVLA.model.modules.uamvla.data.state_normalizer'`.

- [ ] **Step 3: Implement StateNormalizer**

Create `starVLA/model/modules/uamvla/data/state_normalizer.py`:

```python
"""Per-field state normalization for canonical_state dicts.

Composes existing `Normalizer` (q99 / mean_std / min_max) one instance per field,
matching OpenVLA-OFT's `normalize_action_and_proprio` recipe for the proprio half.
Spec: docs/superpowers/specs/2026-04-30-state-input-normalization-design.md
"""
from __future__ import annotations

import copy
from typing import Optional

import torch

from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer


_VALID_MODES = ("q99", "mean_std", "min_max", "none")


class StateNormalizer:
    """
    Apply per-field normalization to a canonical_state dict.

    Args:
        stats_dict: Loaded statistics.yaml as a dict. Must contain
            stats_dict["state_stats"][embodiment] when mode != "none".
        embodiment: Embodiment key, e.g. "franka_libero".
        mode: One of "q99" | "mean_std" | "min_max" | "none". Default "q99".
        apply_to: Optional list of dotted field paths (e.g. ["arm_0.ee_pose"]).
            If None, every field present in state_stats[embodiment] is normalized.
            Fields not in this list pass through unchanged.
    """

    def __init__(
        self,
        stats_dict: dict,
        embodiment: str,
        mode: str = "q99",
        apply_to: Optional[list[str]] = None,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Valid: {_VALID_MODES}")
        self.mode = mode
        self.embodiment = embodiment
        self._normalizers: dict[str, Normalizer] = {}

        if mode == "none":
            return

        if "state_stats" not in stats_dict:
            raise KeyError(
                "statistics.yaml is missing 'state_stats' block. Re-run the "
                "preprocessor to regenerate it, or set normalization mode to 'none'."
            )
        if embodiment not in stats_dict["state_stats"]:
            raise KeyError(
                f"state_stats has no entry for embodiment '{embodiment}'. "
                f"Available: {list(stats_dict['state_stats'].keys())}"
            )

        emb_stats = stats_dict["state_stats"][embodiment]
        for field_path, field_stats in emb_stats.items():
            if apply_to is not None and field_path not in apply_to:
                continue
            self._normalizers[field_path] = Normalizer(
                mode=mode, statistics=dict(field_stats)
            )

    def __call__(self, canonical_state: dict) -> dict:
        if self.mode == "none" or not self._normalizers:
            return canonical_state

        out = copy.deepcopy(canonical_state)
        for field_path, normalizer in self._normalizers.items():
            keys = field_path.split(".")
            ref = out
            for k in keys[:-1]:
                ref = ref[k]
            tensor = ref[keys[-1]]
            assert isinstance(tensor, torch.Tensor), (
                f"Field {field_path} must be a torch.Tensor; got {type(tensor)}"
            )
            ref[keys[-1]] = normalizer.forward(tensor)
        return out
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_state_normalizer.py -v`
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/modules/uamvla/data/state_normalizer.py tests/test_state_normalizer.py
git commit -m "[state-norm] Add StateNormalizer with per-field q99 + clip"
```

---

### Task 3: Compute canonical-space state_stats in LiberoPreprocessor (TDD)

**Why:** `_write_statistics` currently writes only `action_min/max_bound` plus deprecated 3-dim `robot_obs_mean/std`. We need per-field q01/q99/min/max/mean/std over the **canonical** representation, computed by running each sample through `LiberoAdapter.to_canonical`.

**Files:**
- Modify: `tools/preprocess/libero_preprocessor.py:665-715`
- Modify: `tests/test_preprocessor_smoke.py` (extend with state_stats assertions)

- [ ] **Step 1: Write the failing assertion**

Open `tests/test_preprocessor_smoke.py` and locate the test that runs `LiberoPreprocessor.process()` end-to-end on a tiny fixture and reads `statistics.yaml`. Add a new test (or extend the existing post-process test) with these assertions:

```python
def test_state_stats_block_written_with_canonical_layout(tmp_path, ...):
    """After preprocess, statistics.yaml must contain state_stats with canonical-shape per-field stats."""
    # ... existing setup that runs preprocessor on a small fixture ...
    import yaml
    with open(tmp_path / "statistics.yaml") as f:
        stats = yaml.safe_load(f)

    assert "state_stats" in stats, "statistics.yaml missing 'state_stats' block"
    franka = stats["state_stats"]["franka_libero"]

    # Canonical layout per LiberoAdapter.to_canonical()
    EXPECTED = {
        "arm_0.ee_pose":   9,
        "arm_0.joint_pos": 7,
        "gripper_0":       1,
    }
    for field_path, dim in EXPECTED.items():
        assert field_path in franka, f"missing {field_path}"
        per_field = franka[field_path]
        for stat_key in ("q01", "q99", "min", "max", "mean", "std"):
            assert stat_key in per_field, f"{field_path} missing {stat_key}"
            assert len(per_field[stat_key]) == dim, (
                f"{field_path}.{stat_key} expected dim {dim}, got {len(per_field[stat_key])}"
            )
        # Sanity: q01 <= q99 element-wise
        for q1, q9 in zip(per_field["q01"], per_field["q99"]):
            assert q1 <= q9 + 1e-6, f"{field_path}: q01 ({q1}) > q99 ({q9})"
```

If `tests/test_preprocessor_smoke.py` doesn't already have a fixture that runs the preprocessor end-to-end on a tiny LIBERO subset, reuse the same setup as the existing test that asserts `statistics.yaml` exists. Look for `LiberoPreprocessor` invocation in the file and copy the pattern.

- [ ] **Step 2: Run test, confirm it fails**

Run: `pytest tests/test_preprocessor_smoke.py::test_state_stats_block_written_with_canonical_layout -v`
Expected: AssertionError "statistics.yaml missing 'state_stats' block".

- [ ] **Step 3: Modify _write_statistics**

In `tools/preprocess/libero_preprocessor.py`, replace the body of `_write_statistics` (lines 665-715) with:

```python
def _write_statistics(
    self,
    samples: list[dict],
    output_path: Path,
    camera_intrinsics: dict,
):
    """Compute normalization statistics and write statistics.yaml.

    Action stats: min/max bounds (unchanged).
    State stats (NEW): per-field q01/q99/min/max/mean/std over the CANONICAL
        representation produced by LiberoAdapter.to_canonical(), keyed by
        dotted path matching the canonical_state nested dict.
    """
    import yaml
    import numpy as np

    from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter

    actions_7d = np.array(
        [s["action"][:FRANKA_ACTION_DIM] for s in samples]
    )

    # Canonical-space state stats: run adapter per sample, accumulate per field.
    adapter = LiberoAdapter()
    field_buffers: dict[str, list[np.ndarray]] = {
        "arm_0.ee_pose": [],
        "arm_0.joint_pos": [],
        "gripper_0": [],
    }
    for s in samples:
        canonical = adapter.to_canonical(s)
        field_buffers["arm_0.ee_pose"].append(
            canonical["arm_0"]["ee_pose"].numpy()
        )
        field_buffers["arm_0.joint_pos"].append(
            canonical["arm_0"]["joint_pos"].numpy()
        )
        field_buffers["gripper_0"].append(canonical["gripper_0"].numpy())

    franka_state_stats: dict[str, dict] = {}
    for field_path, vals in field_buffers.items():
        arr = np.stack(vals).astype(np.float64)  # (N, D)
        franka_state_stats[field_path] = {
            "q01":  np.quantile(arr, 0.01, axis=0).tolist(),
            "q99":  np.quantile(arr, 0.99, axis=0).tolist(),
            "min":  arr.min(axis=0).tolist(),
            "max":  arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).tolist(),
        }

    stats = {
        "view_names": ["static", "wrist"],
        "max_action_dim": MAX_ACTION_DIM,
        "embodiment_stats": {
            "franka_libero": {
                "action_dim": FRANKA_ACTION_DIM,
                "action_min_bound": actions_7d.min(axis=0).tolist(),
                "action_max_bound": actions_7d.max(axis=0).tolist(),
            },
        },
        "state_stats": {"franka_libero": franka_state_stats},
        "cameras": {
            "static": {"intrinsic": camera_intrinsics[STATIC_CAM]},
            "wrist":  {"intrinsic": camera_intrinsics[WRIST_CAM]},
        },
        "point_cloud": {
            "num_points": NUM_POINTS,
            "frame": "world",
        },
    }

    with open(output_path / "statistics.yaml", "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Wrote statistics.yaml ({len(samples)} samples)")
```

Key changes:
- Drops `robot_obs` ndarray construction (was lines 684).
- Drops `robot_obs_mean` / `robot_obs_std` keys from output stats dict.
- Adds `state_stats.franka_libero.<field_path>.{q01,q99,min,max,mean,std}` block.

- [ ] **Step 4: Run test, confirm pass**

Run: `pytest tests/test_preprocessor_smoke.py::test_state_stats_block_written_with_canonical_layout -v`
Expected: PASS.

- [ ] **Step 5: Run full preprocessor smoke test**

Run: `pytest tests/test_preprocessor_smoke.py -v`
Expected: all tests pass. If a pre-existing assertion now fails because `robot_obs_mean` is gone, **do not fix it here** — it gets fixed in Task 4.

- [ ] **Step 6: Commit (preprocessor change only, leaves robot_obs cleanup for next task)**

```bash
git add tools/preprocess/libero_preprocessor.py tests/test_preprocessor_smoke.py
git commit -m "[preprocess] Compute canonical-space state_stats per field"
```

---

### Task 4: Drop deprecated robot_obs_mean/std from callers

**Why:** Task 3 stopped writing `robot_obs_mean/std`; readers and fixtures still reference them. The TODO at `libero_preprocessor.py:680-683` enumerates: `uamvla/data/statistics.py`, `tests/conftest.py`, `tests/test_data.py`. There may be more — grep first.

**Files:**
- Find via grep, then modify all matches.
- Likely: `starVLA/model/modules/uamvla/data/statistics.py`, `tests/conftest.py`, `tests/test_data.py`.

- [ ] **Step 1: Enumerate all robot_obs_mean / robot_obs_std references**

Run:
```bash
grep -rn "robot_obs_mean\|robot_obs_std" starVLA/ tests/ tools/ --include="*.py" --include="*.yaml"
```
Record every match. Each match must either be deleted or have its dependency on the field removed.

- [ ] **Step 2: For each match, modify**

For production code (`starVLA/.../statistics.py`): if it provides a default `robot_obs_mean=[0,0,0]`, delete the default; if anything reads `stats["robot_obs_mean"]`, remove that read and any downstream usage. The state encoder no longer needs this — `StateNormalizer` covers what `robot_obs_mean/std` partially attempted.

For test fixtures (`tests/conftest.py`, `tests/test_data.py`): remove `robot_obs_mean` and `robot_obs_std` keys from any synthetic stats dicts. If a test asserted their presence, delete the assertion (the schema no longer guarantees them).

For each file, after modifying:
- Run the file's pytest module: `pytest tests/<file>.py -v`
- Expected: green or only failures unrelated to robot_obs.

- [ ] **Step 3: Drop the obsolete TODO comment**

In `tools/preprocess/libero_preprocessor.py`, remove lines 677-683 (the `TODO(state-encoder PR#2)` comment block) since it's now done.

- [ ] **Step 4: Run full test suite to confirm no robot_obs regressions**

Run: `pytest tests/ -v -x`
Expected: all green. If a test fails on something unrelated, leave it for separate triage but verify the failure does not mention `robot_obs`.

- [ ] **Step 5: Commit**

```bash
git add starVLA/ tests/ tools/preprocess/libero_preprocessor.py
git commit -m "[cleanup] Drop deprecated robot_obs_mean/std references"
```

---

### Task 5: Re-preprocess test dataset

**Why:** `datasets/uamvla_test/libero_spatial/statistics.yaml` was written with the old schema. Need to regenerate it with `state_stats` block to make any downstream test that loads this dataset work.

**Files:**
- Regenerate: `datasets/uamvla_test/libero_spatial/statistics.yaml`
- (May regenerate `data.jsonl` too if pipeline mandates; check the preprocessor's idempotency.)

- [ ] **Step 1: Locate the preprocess invocation that produced the test dataset**

Run:
```bash
grep -rn "uamvla_test\|libero_spatial" examples/ scripts/ tools/ Makefile 2>/dev/null | head
```
You're looking for a script or make target that runs `LiberoPreprocessor.process(...)` on a small LIBERO subset producing `datasets/uamvla_test/libero_spatial/`. If none exists, fall back to writing a one-off Python script: instantiate `LiberoPreprocessor`, call `.process(input_dir, output_dir)` with the LIBERO-spatial test HDF5 and `datasets/uamvla_test/libero_spatial/`. The HDF5 source path is documented in memory id 3475 ("Local LIBERO test datasets confirmed").

- [ ] **Step 2: Re-run preprocessing**

Use the conda env confirmed at memory id 3478 (`libero_env` with mujoco + libero + robosuite). Run the script.
Expected: `datasets/uamvla_test/libero_spatial/statistics.yaml` regenerated. No errors.

- [ ] **Step 3: Verify regenerated file has the new schema**

Run:
```bash
python -c "
import yaml
s = yaml.safe_load(open('datasets/uamvla_test/libero_spatial/statistics.yaml'))
assert 'state_stats' in s, 'state_stats missing'
franka = s['state_stats']['franka_libero']
for fp, dim in [('arm_0.ee_pose', 9), ('arm_0.joint_pos', 7), ('gripper_0', 1)]:
    assert fp in franka, f'missing {fp}'
    for k in ['q01', 'q99', 'min', 'max', 'mean', 'std']:
        assert len(franka[fp][k]) == dim, f'{fp}.{k} wrong dim'
assert 'robot_obs_mean' not in s, 'robot_obs_mean should be gone'
print('OK')
"
```
Expected output: `OK`.

- [ ] **Step 4: Commit regenerated test data**

```bash
git add datasets/uamvla_test/libero_spatial/statistics.yaml
git commit -m "[data] Regenerate libero_spatial test stats with state_stats block"
```

---

### Task 6: Add normalization config block to uamvla_libero.yaml

**Why:** `StateNormalizer` needs to be configurable from YAML. Add the block; defaults match OpenVLA-OFT.

**Files:**
- Modify: `starVLA/config/training/uamvla_libero.yaml`

- [ ] **Step 1: Locate the state_encoder block**

Run: `grep -n "state_encoder" starVLA/config/training/uamvla_libero.yaml`
Expected: a block under `framework.state_encoder` with `type:` and `register_special_tokens:`.

- [ ] **Step 2: Add normalization sub-block**

Edit `starVLA/config/training/uamvla_libero.yaml`. Under `framework.state_encoder`, immediately after the existing keys, add:

```yaml
    normalization:
      mode: q99                # one of: q99 | mean_std | min_max | none
      apply_to:                # canonical field paths to normalize
        - arm_0.ee_pose
        - arm_0.joint_pos
        - gripper_0
```

Indentation must match neighbors (typically 4 spaces under `state_encoder:`).

- [ ] **Step 3: Sanity-check YAML parses**

Run:
```bash
python -c "
import yaml
c = yaml.safe_load(open('starVLA/config/training/uamvla_libero.yaml'))
n = c['framework']['state_encoder']['normalization']
assert n['mode'] == 'q99'
assert n['apply_to'] == ['arm_0.ee_pose', 'arm_0.joint_pos', 'gripper_0']
print('OK')
"
```
Expected output: `OK`.

- [ ] **Step 4: Commit**

```bash
git add starVLA/config/training/uamvla_libero.yaml
git commit -m "[config] Add state encoder normalization block (q99 default)"
```

---

### Task 7: Wire StateNormalizer into UamVLADataset (TDD)

**Why:** Now that `StateNormalizer` exists, stats are written, and config is in place, plumb everything through. Construct in `__init__`, call in `__getitem__` after the adapter.

**Files:**
- Modify: `starVLA/dataloader/uamvla_dataset.py`
- Modify: `tests/test_uamvla_dataset.py`

- [ ] **Step 1: Write failing test asserting post-norm range**

Open `tests/test_uamvla_dataset.py`, add:

```python
def test_canonical_state_normalized_to_unit_range(sample_dataset_dir):
    """After normalization, every field in canonical_state should be in [-1, 1]."""
    import torch
    from starVLA.dataloader.uamvla_dataset import UamVLADataset

    ds = UamVLADataset(
        data_root=sample_dataset_dir,
        embodiment="franka_libero",
        action_horizon=8,
        normalization={
            "mode": "q99",
            "apply_to": ["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"],
        },
    )
    sample = ds[0]
    cs = sample["canonical_state"]
    for field in (cs["arm_0"]["ee_pose"], cs["arm_0"]["joint_pos"], cs["gripper_0"]):
        assert torch.all(field >= -1.0 - 1e-6), f"value < -1: {field}"
        assert torch.all(field <= 1.0 + 1e-6), f"value > 1: {field}"


def test_canonical_state_mode_none_passes_raw_values(sample_dataset_dir):
    """mode=none must NOT modify canonical_state values."""
    import torch
    from starVLA.dataloader.uamvla_dataset import UamVLADataset
    from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter

    ds = UamVLADataset(
        data_root=sample_dataset_dir,
        embodiment="franka_libero",
        action_horizon=8,
        normalization={"mode": "none", "apply_to": []},
    )
    sample = ds[0]
    raw = ds.samples[0]
    expected = LiberoAdapter().to_canonical(raw)
    assert torch.allclose(sample["canonical_state"]["arm_0"]["ee_pose"],
                          expected["arm_0"]["ee_pose"])
```

This test depends on a `sample_dataset_dir` fixture. Look in `tests/conftest.py` to see if it exists; if it points to `datasets/uamvla_test/libero_spatial/`, you're fine. If not, follow the existing pattern (other tests in the file already use the same fixture).

- [ ] **Step 2: Run, confirm failure**

Run: `pytest tests/test_uamvla_dataset.py::test_canonical_state_normalized_to_unit_range -v`
Expected: TypeError or AssertionError — `UamVLADataset.__init__` does not accept `normalization=` kwarg yet.

- [ ] **Step 3: Modify UamVLADataset.__init__ signature**

Edit `starVLA/dataloader/uamvla_dataset.py`.

Change the constructor signature (line 40-47) to add a `normalization` kwarg:

```python
def __init__(
    self,
    data_root: Path | str,
    embodiment: str = "franka_libero",
    action_horizon: int = 8,
    max_samples: Optional[int] = None,
    transforms=None,
    normalization: Optional[dict] = None,
):
```

After `self.adapter = get_embodiment_config(embodiment)["adapter"]` (line 71), add:

```python
    # State normalizer: build from stats already loaded into self.stats.
    from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer
    norm_cfg = normalization or {"mode": "q99"}
    self.state_normalizer = StateNormalizer(
        stats_dict=self.stats,
        embodiment=embodiment,
        mode=norm_cfg.get("mode", "q99"),
        apply_to=norm_cfg.get("apply_to"),
    )
```

In `__getitem__` (line 80-109), change line 95 from:

```python
        canonical_state = self.adapter.to_canonical(raw)
```

to:

```python
        canonical_state = self.state_normalizer(self.adapter.to_canonical(raw))
```

- [ ] **Step 4: Plumb config through get_vla_dataset**

In `starVLA/dataloader/uamvla_dataset.py`, modify `get_vla_dataset` (line 185-202) to pass normalization config:

```python
def get_vla_dataset(data_cfg, mode: str = "train", **kwargs) -> Dataset:
    """starVLA plugin entry point. Returns a Dataset that yields starVLA examples dicts."""
    data_root = Path(data_cfg.data_root_dir)
    mixture = DATASET_NAMED_MIXTURES.get(data_cfg.data_mix)
    if mixture is None:
        raise ValueError(f"Unknown data_mix '{data_cfg.data_mix}'. Available: {list(DATASET_NAMED_MIXTURES)}")

    # Pull state encoder normalization config if present
    framework = getattr(data_cfg, "framework", None) or kwargs.get("framework")
    norm_cfg = None
    if framework is not None:
        state_enc = getattr(framework, "state_encoder", None) or framework.get("state_encoder", {})
        if hasattr(state_enc, "get"):
            norm_cfg = state_enc.get("normalization")
        else:
            norm_cfg = getattr(state_enc, "normalization", None)

    if len(mixture) == 1:
        subdir, _weight, embodiment = mixture[0]
        return UamVLADataset(
            data_root=data_root / subdir,
            embodiment=embodiment,
            action_horizon=int(data_cfg.get("action_horizon", 8)),
            normalization=dict(norm_cfg) if norm_cfg else None,
        )
    raise NotImplementedError("Multi-subdir mixture deferred to Phase 2")
```

If `data_cfg` and `framework` plumbing in this codebase uses OmegaConf (likely, given Hydra-style YAML), the `getattr` / `.get` pattern above handles both DictConfig and plain dict. If neither works in your codebase pattern, follow whatever the surrounding starVLA loader code does (e.g., look at the gr00t dataloader's plugin entry for reference).

- [ ] **Step 5: Run tests, confirm both pass**

Run: `pytest tests/test_uamvla_dataset.py::test_canonical_state_normalized_to_unit_range tests/test_uamvla_dataset.py::test_canonical_state_mode_none_passes_raw_values -v`
Expected: both PASS.

- [ ] **Step 6: Run full uamvla_dataset test module**

Run: `pytest tests/test_uamvla_dataset.py -v`
Expected: all tests pass. Pre-existing tests should still hold because default normalization (`q99` + all fields) does not break the dataset's external contract — canonical_state is still a dict of tensors with the same shapes.

- [ ] **Step 7: Commit**

```bash
git add starVLA/dataloader/uamvla_dataset.py tests/test_uamvla_dataset.py
git commit -m "[dataset] Wire StateNormalizer into UamVLADataset.__getitem__"
```

---

### Task 8: End-to-end smoke validation

**Why:** Confirm normalized canonical_state runs cleanly through `ModularStateEncoder` and produces the expected `(B, 3, hidden_dim)` token shape. Catches any contract drift.

**Files:**
- Modify: `tests/test_state_encoder_smoke.py` (or create a new e2e test file).

- [ ] **Step 1: Add smoke test**

Append to `tests/test_state_encoder_smoke.py`:

```python
def test_normalized_canonical_state_passes_through_modular_encoder(sample_dataset_dir):
    """Full pipeline: dataset → StateNormalizer → ModularStateEncoder → 3-token output."""
    import torch
    from starVLA.dataloader.uamvla_dataset import UamVLADataset
    from starVLA.model.modules.uamvla.state_encoder.modular_state_encoder import (
        ModularStateEncoder,
    )

    HIDDEN_DIM = 1024  # arbitrary for smoke; not the production 3584
    encoder = ModularStateEncoder(embodiment="franka_libero", hidden_dim=HIDDEN_DIM)
    encoder.eval()

    ds = UamVLADataset(
        data_root=sample_dataset_dir,
        embodiment="franka_libero",
        action_horizon=8,
        normalization={"mode": "q99",
                       "apply_to": ["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"]},
    )
    sample = ds[0]
    cs = {k: ({k2: v2.unsqueeze(0) for k2, v2 in v.items()}
              if isinstance(v, dict) else v.unsqueeze(0))
          for k, v in sample["canonical_state"].items()}

    with torch.no_grad():
        out = encoder(cs)

    assert out.shape == (1, 3, HIDDEN_DIM), f"unexpected encoder output shape: {out.shape}"
    assert torch.isfinite(out).all(), "non-finite values in encoder output"
```

If `ModularStateEncoder.__init__` signature differs from `(embodiment, hidden_dim)`, look at `starVLA/model/modules/uamvla/state_encoder/modular_state_encoder.py:21-43` and adapt.

- [ ] **Step 2: Run smoke test**

Run: `pytest tests/test_state_encoder_smoke.py::test_normalized_canonical_state_passes_through_modular_encoder -v`
Expected: PASS.

- [ ] **Step 3: Run the entire test suite once for regression sanity**

Run: `pytest tests/ -v`
Expected: green. If anything fails, investigate before declaring done.

- [ ] **Step 4: Commit**

```bash
git add tests/test_state_encoder_smoke.py
git commit -m "[test] E2E smoke: dataset → StateNormalizer → ModularStateEncoder"
```

---

## Done definition

- All 8 tasks committed.
- Test suite passes: `pytest tests/ -v`.
- `datasets/uamvla_test/libero_spatial/statistics.yaml` contains `state_stats.franka_libero.{arm_0.ee_pose, arm_0.joint_pos, gripper_0}` with q01/q99/min/max/mean/std arrays of shapes 9/7/1.
- `robot_obs_mean` / `robot_obs_std` are gone from production code, fixtures, and the regenerated stats file.
- A fresh dataset load with default normalization yields canonical_state values within `[-1, 1]`.
- Spec self-corrected (Task 1) to reflect that `Normalizer.q99` already clips.

## Out of scope (explicit non-goals from spec)

- Encoder architecture changes (LayerNorm / depth / bottleneck / token count).
- Other embodiments (Bridge, ALOHA, CALVIN).
- Backbone changes.
- Action normalization (already separate; uses min_max).
