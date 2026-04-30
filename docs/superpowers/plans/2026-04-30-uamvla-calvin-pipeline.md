# UamVLA CALVIN Preprocessing + Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the UamVLA-line CALVIN preprocessing + training pipeline (CLI runner, statistics migration, mixture entry, training yaml/launcher) so that a freshly trained UamVLA-CALVIN checkpoint loads cleanly into the existing eval policy server.

**Architecture:** Mirror the existing UamVLA-LIBERO pipeline. Reuse the already-built `CalvinPreprocessor` + `CalvinAdapter` + `franka_calvin` embodiment; fix the one schema gap (`statistics.yaml` missing `state_stats[franka_calvin]`); add the missing CLI / mixture / yaml / launcher; rename one `unnorm_key` literal in the eval shell so the checkpoint can be loaded. Eval-time state passthrough (CALVIN parallel of the LIBERO fix) is **explicitly out of scope** — known limitation, deferred to a follow-up spec.

**Tech Stack:** Python 3 (numpy, yaml, torch, PIL, pytest, OmegaConf), bash, CALVIN dataset format (npz episodes + lang_annotations + scene_info), CALVIN env (PyBullet — only for remote e2e, not for unit tests).

---

## Spec

Source spec: [`docs/superpowers/specs/2026-04-30-uamvla-calvin-pipeline-design.md`](../specs/2026-04-30-uamvla-calvin-pipeline-design.md). Read it first if you have not.

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `tools/preprocess/calvin_preprocessor.py` (function `_write_statistics`, lines ~406-440) | Replace legacy `robot_obs_mean/std` writer with per-field q01/q99/min/max/mean/std over canonical state, mirroring `LiberoPreprocessor.write_statistics`. |
| Modify | `starVLA/dataloader/uamvla_dataset.py` (`DATASET_NAMED_MIXTURES`, line ~28) | Register `calvin_uamvla` mixture entry pointing at `("task_D_D", 1.0, "franka_calvin")`. |
| Create | `tests/test_uamvla_calvin_pipeline_local.py` | Local unit tests (no `calvin_env`) for stats schema, mixture, dataset chain, CLI args, yaml load. |
| Create | `runners/preprocess_calvin.py` | CLI entry point wrapping `CalvinPreprocessor.process(...)`, structurally identical to `runners/preprocess_libero.py`. |
| Create | `starVLA/config/training/uamvla_calvin.yaml` | UamVLA training config; clone of `uamvla_libero.yaml` with embodiment / data_root / data_mix / run_id substitutions. |
| Create | `examples/calvin/train_files/run_uamvla_calvin_train.sh` | Training launcher; clone of `examples/LIBERO/train_files/run_libero_train.sh` with calvin paths/run_id. |
| Modify | `examples/calvin/eval_files/eval_calvin.sh` (line 10, single literal) | `unnorm_key="franka"` → `unnorm_key="franka_calvin"`. |
| Modify | `examples/calvin/README.md` (insert near top) | Add "**UamVLA path**" quickstart section + known-limitation callout. |

The new test file is a single self-contained module. Each task adds its own test cases to it (cumulative). The fixture builder for the synthetic CALVIN dataset is also defined in this file (private helper, no separate fixture module).

## Verification Notes

- Tasks 1–8 run on macOS (no `calvin_env` needed). Task 9 is a manual remote acceptance step run on a machine with `calvin_env` installed; it produces no auto-test output.
- `tests/test_preprocessor_smoke.py:21` (`test_calvin_preprocessor_imports`, `pytest.importorskip("calvin_env")`) is left untouched — it covers an orthogonal axis (import-time regressions on machines that DO have calvin_env).

---

## Task 1: Migrate `_write_statistics` to new state_stats schema (G2)

**Files:**
- Modify: `tools/preprocess/calvin_preprocessor.py:406-440` (function `_write_statistics`)
- Test: `tests/test_uamvla_calvin_pipeline_local.py` (new file)

**Why first:** Every downstream task assumes the new schema. If this is broken, the synthetic-fixture test in Task 3 cannot have a self-consistent statistics.yaml, and a real preprocessor run in Task 9 would emit unusable stats.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uamvla_calvin_pipeline_local.py` with the following content:

```python
"""Local unit tests for the UamVLA CALVIN preprocessing + training pipeline.

These tests run on any machine without requiring calvin_env / PyBullet.
Coverage:
  - statistics.yaml schema migration (Task 1)
  - mixture registry entry (Task 2)
  - synthetic dataset → UamVLADataset chain (Task 3)
  - CLI arg parsing (Task 4)
  - training yaml loadable (Task 5)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# Task 1: _write_statistics schema migration
# ============================================================================

def _make_synthetic_calvin_samples(n: int = 6) -> list[dict]:
    """Build n CALVIN-shape sample dicts. Values are deterministic but varied
    across samples so q01 != q99 (otherwise q99 normalization would degenerate
    to identity and downstream tests can't tell stats were applied).
    """
    rng = np.random.default_rng(seed=42)
    samples = []
    for i in range(n):
        # 15-dim robot_obs: tcp(3) + euler(3) + gripper_width(1) + joints(7) + gripper_cmd(1)
        # Ranges chosen to match the values we probed from a real CALVIN frame:
        #   tcp ≈ [0.10, -0.09, 0.57], euler ≈ [3.10, 0.02, 1.50], gripper_width ≈ 0.08,
        #   joints ≈ 7-tuple within [-π, π], gripper_cmd ∈ {-1, 1}
        tcp = np.array([0.10, -0.09, 0.57], dtype=np.float64) + 0.05 * rng.standard_normal(3)
        euler = np.array([3.10, 0.02, 1.50], dtype=np.float64) + 0.05 * rng.standard_normal(3)
        gripper_width = 0.08 + 0.005 * rng.standard_normal()
        joints = np.array([-0.59, 0.90, 1.74, -1.84, -0.92, 1.60, 0.59], dtype=np.float64) \
                 + 0.1 * rng.standard_normal(7)
        gripper_cmd = 1.0
        robot_obs_15 = np.concatenate([
            tcp, euler, [gripper_width], joints, [gripper_cmd],
        ]).tolist()

        rel_actions_7 = (0.01 * rng.standard_normal(7)).tolist()
        action_24 = rel_actions_7 + [0.0] * 17

        samples.append({
            "id": f"calvin_debug_ep00000_step{i:04d}",
            "episode_id": "calvin_debug_ep00000",
            "step_idx": i,
            "total_steps": n,
            "image": [
                f"images/obs/static/calvin_debug_ep00000_step{i:04d}.jpg",
                f"images/obs/wrist/calvin_debug_ep00000_step{i:04d}.jpg",
            ],
            "instruction": "test instruction",
            "embodiment": "franka_calvin",
            "action_dim": 7,
            "action": action_24,
            "action_mask": [1] * 7 + [0] * 17,
            "robot_obs": robot_obs_15,
            "dataset_source": "calvin_debug",
        })
    return samples


def test_write_statistics_schema(tmp_path):
    """`_write_statistics` emits the new state_stats[franka_calvin] block."""
    from tools.preprocess.calvin_preprocessor import _write_statistics

    samples = _make_synthetic_calvin_samples(n=6)
    intrinsics = {
        "static": {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
        "wrist":  {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
    }

    _write_statistics(samples, tmp_path, intrinsics)

    with open(tmp_path / "statistics.yaml") as f:
        stats = yaml.safe_load(f)

    # Top-level shape
    assert set(stats.keys()) == {
        "view_names", "max_action_dim", "embodiment_stats",
        "state_stats", "cameras", "point_cloud",
    }, f"unexpected top-level keys: {sorted(stats.keys())}"

    # state_stats[franka_calvin] presence + shape
    assert "franka_calvin" in stats["state_stats"]
    fs = stats["state_stats"]["franka_calvin"]
    assert set(fs.keys()) == {"arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"}, \
        f"unexpected state field keys: {sorted(fs.keys())}"

    expected_lengths = {"arm_0.ee_pose": 9, "arm_0.joint_pos": 7, "gripper_0": 1}
    expected_substats = {"q01", "q99", "min", "max", "mean", "std"}
    for path, length in expected_lengths.items():
        assert set(fs[path].keys()) == expected_substats, (
            f"{path}: substats {sorted(fs[path].keys())} != {sorted(expected_substats)}"
        )
        for k in expected_substats:
            assert len(fs[path][k]) == length, (
                f"{path}.{k}: len={len(fs[path][k])}, expected {length}"
            )


def test_write_statistics_no_legacy_keys(tmp_path):
    """`robot_obs_mean` / `robot_obs_std` are removed from the new schema."""
    from tools.preprocess.calvin_preprocessor import _write_statistics

    samples = _make_synthetic_calvin_samples(n=4)
    intrinsics = {
        "static": {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
        "wrist":  {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
    }

    _write_statistics(samples, tmp_path, intrinsics)

    with open(tmp_path / "statistics.yaml") as f:
        stats = yaml.safe_load(f)

    assert "robot_obs_mean" not in stats, \
        "Legacy key 'robot_obs_mean' must be removed in the new schema."
    assert "robot_obs_std" not in stats, \
        "Legacy key 'robot_obs_std' must be removed in the new schema."
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_write_statistics_schema tests/test_uamvla_calvin_pipeline_local.py::test_write_statistics_no_legacy_keys -v
```

Expected: both tests **FAIL**. `test_write_statistics_schema` fails on the assertion that top-level keys exclude `robot_obs_mean`/`robot_obs_std`; `test_write_statistics_no_legacy_keys` fails for the same reason. (The current implementation at `calvin_preprocessor.py:431-432` writes these legacy keys.)

- [ ] **Step 3: Replace `_write_statistics` implementation**

In `tools/preprocess/calvin_preprocessor.py`, locate the existing `_write_statistics` function (currently at lines 406-440). Replace its body to compute per-field canonical-space stats. The full replacement function (drop in verbatim, preserving the existing function signature):

```python
def _write_statistics(
    samples: list[dict],
    output_dir: Path,
    camera_intrinsics: dict,
) -> None:
    from starVLA.model.modules.uamvla.data.embodiment_adapter import CalvinAdapter

    actions_7d = np.array(
        [s["action"][:FRANKA_ACTION_DIM] for s in samples], dtype=np.float64,
    )

    # Per-field stats over the CANONICAL representation produced by CalvinAdapter,
    # keyed by dotted path matching the canonical_state nested dict.
    adapter = CalvinAdapter()
    field_buffers: dict[str, list[np.ndarray]] = {
        "arm_0.ee_pose":   [],
        "arm_0.joint_pos": [],
        "gripper_0":       [],
    }
    for s in samples:
        canonical = adapter.to_canonical(s)  # reads s["robot_obs"] (15-dim)
        field_buffers["arm_0.ee_pose"].append(canonical["arm_0"]["ee_pose"].numpy())
        field_buffers["arm_0.joint_pos"].append(canonical["arm_0"]["joint_pos"].numpy())
        field_buffers["gripper_0"].append(canonical["gripper_0"].numpy())

    franka_state_stats: dict[str, dict] = {}
    for path, vals in field_buffers.items():
        arr = np.stack(vals).astype(np.float64)  # (N, D)
        franka_state_stats[path] = {
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
            EMBODIMENT: {
                "action_dim": FRANKA_ACTION_DIM,
                "action_min_bound": actions_7d.min(axis=0).tolist(),
                "action_max_bound": actions_7d.max(axis=0).tolist(),
            },
        },
        "state_stats": {EMBODIMENT: franka_state_stats},
        "cameras": {
            "static": {"intrinsic": camera_intrinsics["static"]},
            "wrist":  {"intrinsic": camera_intrinsics["wrist"]},
        },
        "point_cloud": {"num_points": NUM_POINTS, "frame": "world"},
    }
    with open(output_dir / "statistics.yaml", "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
```

The existing 5-line legacy NOTE comment block (`# NOTE: CALVIN preprocessor still writes legacy robot_obs_mean/std until a future PR migrates it...`) at the top of the old function body should also be removed — the migration is no longer pending.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_write_statistics_schema tests/test_uamvla_calvin_pipeline_local.py::test_write_statistics_no_legacy_keys -v
```

Expected: both **PASS**.

- [ ] **Step 5: Commit**

```bash
git add tools/preprocess/calvin_preprocessor.py tests/test_uamvla_calvin_pipeline_local.py
git commit -m "[calvin] migrate _write_statistics to state_stats[franka_calvin] schema

Replace legacy robot_obs_mean/std (15-dim aggregate) with per-field
q01/q99/min/max/mean/std over the canonical state produced by
CalvinAdapter. Mirrors the LIBERO migration in libero_preprocessor.py.
Without this, StateNormalizer.__init__ raises KeyError on every
CALVIN-trained run."
```

---

## Task 2: Register `calvin_uamvla` mixture entry (G3)

**Files:**
- Modify: `starVLA/dataloader/uamvla_dataset.py:28-34` (`DATASET_NAMED_MIXTURES`)
- Test: `tests/test_uamvla_calvin_pipeline_local.py` (append)

- [ ] **Step 1: Append failing test**

Append to `tests/test_uamvla_calvin_pipeline_local.py` (after the Task 1 section):

```python
# ============================================================================
# Task 2: mixture registry entry
# ============================================================================

def test_calvin_uamvla_mixture_registered():
    from starVLA.dataloader.uamvla_dataset import DATASET_NAMED_MIXTURES

    assert "calvin_uamvla" in DATASET_NAMED_MIXTURES, (
        f"Missing 'calvin_uamvla' mixture; available: {list(DATASET_NAMED_MIXTURES)}"
    )
    assert DATASET_NAMED_MIXTURES["calvin_uamvla"] == [
        ("task_D_D", 1.0, "franka_calvin"),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_calvin_uamvla_mixture_registered -v
```

Expected: FAIL with `AssertionError: Missing 'calvin_uamvla' mixture; available: ['libero_uamvla']`.

- [ ] **Step 3: Add the mixture entry**

In `starVLA/dataloader/uamvla_dataset.py`, replace the existing `DATASET_NAMED_MIXTURES` block (lines 28-34) with:

```python
DATASET_NAMED_MIXTURES = {
    "libero_uamvla": [
        # (data_subdir, weight, embodiment_tag)
        ("libero_spatial", 1.0, "franka_libero"),
        # Phase 2: + libero_object, libero_goal, libero_10
    ],
    "calvin_uamvla": [
        ("task_D_D", 1.0, "franka_calvin"),
    ],
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_calvin_uamvla_mixture_registered -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add starVLA/dataloader/uamvla_dataset.py tests/test_uamvla_calvin_pipeline_local.py
git commit -m "[calvin] register calvin_uamvla mixture entry"
```

---

## Task 3: Synthetic-fixture dataset chain test

**Files:**
- Test only: `tests/test_uamvla_calvin_pipeline_local.py` (append)

This task adds an end-to-end test that builds a synthetic CALVIN-shape dataset on disk, instantiates `UamVLADataset(embodiment="franka_calvin")`, and verifies the full chain (`CalvinAdapter.to_canonical` + `StateNormalizer` + action chunk slicing). It depends on Task 1's stats schema and Task 2's mixture entry, so it lives downstream.

- [ ] **Step 1: Append the fixture builder helper and the failing test**

Append to `tests/test_uamvla_calvin_pipeline_local.py`:

```python
# ============================================================================
# Task 3: synthetic-fixture dataset chain test
# ============================================================================

def _build_synthetic_calvin_dataset(tmp_path: Path, n_samples: int = 6) -> Path:
    """Build a complete CALVIN-shape UAM dataset under tmp_path.

    Produces:
      - data.jsonl with `n_samples` rows (one episode of length n_samples)
      - statistics.yaml with embodiment_stats[franka_calvin] + state_stats[franka_calvin]
      - dummy 256x256 RGB jpgs at every referenced image path
      - 1024×3 zero point clouds at every referenced point_cloud path
      - dummy 256×256 float32 depth npys at every referenced depth path

    Stats are computed from the same fixture data run through CalvinAdapter,
    keeping the fixture self-consistent (q99 != q01 → StateNormalizer
    actually transforms values, so downstream tests can detect it).
    """
    from PIL import Image

    samples = _make_synthetic_calvin_samples(n=n_samples)

    # Augment each sample with the aux fields a real CALVIN preprocessor would write.
    for s in samples:
        sid = s["id"]
        s["image_target"] = f"images/target/{sid}.jpg"
        s["image_future"] = f"images/future/{s['episode_id']}.jpg"
        s["depth_static"] = f"depth/static/{sid}.npy"
        s["depth_wrist"]  = f"depth/wrist/{sid}.npy"
        s["point_cloud"]  = f"point_clouds/{sid}.npy"
        s["pose_6d"] = {
            "rotation":    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # identity 6D rot
            "translation": [0.0, 0.0, 0.0],
            "object_id":   "test_object",
        }
        s["static_cam_extrinsic"] = {
            "rotation":    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
        }
        s["wrist_cam_extrinsic"] = {
            "rotation":    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
        }

    # Create directory tree.
    for sub in (
        "images/obs/static", "images/obs/wrist",
        "images/target", "images/future",
        "depth/static", "depth/wrist",
        "point_clouds",
    ):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    # Write data.jsonl.
    with open(tmp_path / "data.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Write dummy media files.
    blank_rgb = Image.new("RGB", (256, 256), color=(127, 127, 127))
    pc_zeros = np.zeros((1024, 3), dtype=np.float32)
    depth_zeros = np.zeros((256, 256), dtype=np.float32)
    futures_written: set[str] = set()
    for s in samples:
        for img_rel in s["image"] + [s["image_target"]]:
            blank_rgb.save(tmp_path / img_rel, quality=95)
        if s["image_future"] not in futures_written:
            blank_rgb.save(tmp_path / s["image_future"], quality=95)
            futures_written.add(s["image_future"])
        np.save(tmp_path / s["point_cloud"], pc_zeros)
        np.save(tmp_path / s["depth_static"], depth_zeros)
        np.save(tmp_path / s["depth_wrist"], depth_zeros)

    # Compute stats by running samples through CalvinAdapter (self-consistent).
    from tools.preprocess.calvin_preprocessor import _write_statistics
    intrinsics = {
        "static": {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
        "wrist":  {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
    }
    _write_statistics(samples, tmp_path, intrinsics)

    return tmp_path


def test_synthetic_dataset_loads(tmp_path):
    """End-to-end: synthetic CALVIN fixture → UamVLADataset → canonical_state w/ q99."""
    import torch
    from starVLA.dataloader.uamvla_dataset import UamVLADataset
    from starVLA.model.modules.uamvla.data.embodiment_adapter import CalvinAdapter

    data_root = _build_synthetic_calvin_dataset(tmp_path, n_samples=6)

    ds = UamVLADataset(
        data_root=data_root,
        embodiment="franka_calvin",
        action_horizon=8,
        normalization={
            "mode": "q99",
            "apply_to": ["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"],
        },
    )

    assert len(ds) == 6, f"expected 6 samples, got {len(ds)}"

    sample = ds[0]
    cs = sample["canonical_state"]
    assert cs["arm_0"]["ee_pose"].shape == (9,), cs["arm_0"]["ee_pose"].shape
    assert cs["arm_0"]["joint_pos"].shape == (7,), cs["arm_0"]["joint_pos"].shape
    assert cs["gripper_0"].shape == (1,), cs["gripper_0"].shape

    assert sample["action"].shape == (8, 7), sample["action"].shape
    assert sample["action_mask"].shape == (8, 7), sample["action_mask"].shape

    # State normalizer should have actually transformed the values. Recompute
    # the raw (pre-normalizer) canonical state independently and verify the
    # post-normalizer tensor differs. If they match, StateNormalizer silently
    # no-oped (e.g., mode='none' fell through, or stats were degenerate).
    raw_canonical = CalvinAdapter().to_canonical(ds.samples[0])
    raw_ee_pose = raw_canonical["arm_0"]["ee_pose"]
    assert not torch.allclose(cs["arm_0"]["ee_pose"], raw_ee_pose), (
        "Post-normalizer ee_pose equals raw canonical — StateNormalizer no-oped. "
        "Check that mode='q99' is wired and state_stats has non-degenerate q01/q99."
    )
```

- [ ] **Step 2: Run test to verify it passes**

Run:
```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_synthetic_dataset_loads -v
```

Expected: **PASS**. Tasks 1 and 2 are already merged, so the chain is complete; this test merely composes them.

If it fails: the most likely cause is a misalignment between CalvinAdapter's required input fields (`robot_obs`) and what the synthetic samples carry. Inspect the failing assertion message — `CalvinAdapter._check_required_fields` raises a clear KeyError if `robot_obs` is missing or wrong-shaped.

- [ ] **Step 3: Commit**

```bash
git add tests/test_uamvla_calvin_pipeline_local.py
git commit -m "[calvin] add synthetic-fixture dataset chain test"
```

---

## Task 4: CLI runner `runners/preprocess_calvin.py` (G1)

**Files:**
- Create: `runners/preprocess_calvin.py`
- Test: `tests/test_uamvla_calvin_pipeline_local.py` (append)

The CLI is structurally identical to `runners/preprocess_libero.py`. It does **not** need calvin_env to be importable at the module level — only when `.process(...)` is actually called (which the tests do not do).

- [ ] **Step 1: Append failing CLI tests**

Append to `tests/test_uamvla_calvin_pipeline_local.py`:

```python
# ============================================================================
# Task 4: CLI runner argument parsing
# ============================================================================

CLI_PATH = ROOT / "runners" / "preprocess_calvin.py"


def test_cli_help():
    """`--help` exits 0 and lists all six expected args."""
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    for arg in (
        "--input_dir", "--output_dir", "--dataset_source",
        "--num_workers", "--default_scene",
        "--on_resolve_failure", "--on_missing_target",
    ):
        assert arg in result.stdout, f"--help output missing {arg}\n{result.stdout}"


def test_cli_required_args_missing_input():
    """Omitting --input_dir fails with non-zero exit."""
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--output_dir", "/tmp/x"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0, "Missing --input_dir should fail"
    assert "input_dir" in result.stderr.lower() or "input_dir" in result.stdout.lower()


def test_cli_required_args_missing_output():
    """Omitting --output_dir fails with non-zero exit."""
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--input_dir", "/tmp/x"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0, "Missing --output_dir should fail"
    assert "output_dir" in result.stderr.lower() or "output_dir" in result.stdout.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_cli_help tests/test_uamvla_calvin_pipeline_local.py::test_cli_required_args_missing_input tests/test_uamvla_calvin_pipeline_local.py::test_cli_required_args_missing_output -v
```

Expected: all three **FAIL** because `runners/preprocess_calvin.py` does not exist (subprocess returns non-zero with `No such file or directory`-equivalent message in stderr — the assertion on returncode fails for `test_cli_help`).

- [ ] **Step 3: Create `runners/preprocess_calvin.py`**

Write the new file with this exact content:

```python
"""CLI entry point for CALVIN → UamVLA dataset preprocessing.

Usage:
    # Process the debug dataset (the path on the spec author's local box):
    python runners/preprocess_calvin.py \\
        --input_dir /Users/tancilon/develop/localgit/UamVLA/datasets/calvin_debug_dataset/training \\
        --output_dir datasets/uamvla_calvin/task_D_D \\
        --dataset_source calvin_debug

    # Process a real task_D_D split:
    python runners/preprocess_calvin.py \\
        --input_dir /path/to/calvin/task_D_D/training \\
        --output_dir datasets/uamvla_calvin/task_D_D \\
        --dataset_source task_D_D

This driver requires `calvin_env` (PyBullet) at runtime — the worker calls
env.reset(robot_obs, scene_obs) + env.render_cameras for each frame. On
machines without calvin_env, the CLI still loads (no module-level import),
but `--input_dir` actually invoking the worker will fail at make_calvin_env_adapter.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so tools/starVLA packages are importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.preprocess.calvin_preprocessor import CalvinPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a CALVIN split to UamVLA unified format.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="A CALVIN split dir containing episode_*.npz, ep_start_end_ids.npy, "
             "lang_annotations/auto_lang_ann.npy, scene_info.npy.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Target UAM dataset dir; created if missing. "
             "Recommended layout: datasets/uamvla_calvin/<split>/.",
    )
    parser.add_argument(
        "--dataset_source",
        type=str,
        default="calvin",
        help="String prefix for episode_id and JSONL dataset_source field. "
             "Override with the split name (task_D_D, calvin_debug) for clarity. "
             "Default: 'calvin'.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="multiprocessing.Pool size. Use >1 only on machines with reliable "
             "PyBullet/EGL. Default: 1.",
    )
    parser.add_argument(
        "--default_scene",
        type=str,
        default=None,
        help="Forwarded to SceneResolver. Set when the split's scene_info.npy "
             "does not cover all windows. Default: None.",
    )
    parser.add_argument(
        "--on_resolve_failure",
        choices=("skip", "abort"),
        default="abort",
        help="What to do if a window's task_label is unknown to calvin_task_map. "
             "Default: abort.",
    )
    parser.add_argument(
        "--on_missing_target",
        choices=("skip", "abort"),
        default="abort",
        help="What to do if a frame's target object is not visible in the "
             "static-camera seg mask. Default: abort.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    preprocessor = CalvinPreprocessor(
        dataset_source=args.dataset_source,
        default_scene=args.default_scene,
        num_workers=args.num_workers,
        on_resolve_failure=args.on_resolve_failure,
        on_missing_target=args.on_missing_target,
    )

    preprocessor.process(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_cli_help tests/test_uamvla_calvin_pipeline_local.py::test_cli_required_args_missing_input tests/test_uamvla_calvin_pipeline_local.py::test_cli_required_args_missing_output -v
```

Expected: all three **PASS**.

- [ ] **Step 5: Commit**

```bash
git add runners/preprocess_calvin.py tests/test_uamvla_calvin_pipeline_local.py
git commit -m "[calvin] add runners/preprocess_calvin.py CLI driver"
```

---

## Task 5: Training yaml `starVLA/config/training/uamvla_calvin.yaml` (G4)

**Files:**
- Create: `starVLA/config/training/uamvla_calvin.yaml`
- Test: `tests/test_uamvla_calvin_pipeline_local.py` (append)

- [ ] **Step 1: Append failing yaml-loadability test**

Append to `tests/test_uamvla_calvin_pipeline_local.py`:

```python
# ============================================================================
# Task 5: training yaml loadable
# ============================================================================

def test_yaml_loadable():
    """uamvla_calvin.yaml exists, parses, and has the expected substitutions."""
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    yaml_path = ROOT / "starVLA" / "config" / "training" / "uamvla_calvin.yaml"
    assert yaml_path.exists(), f"missing {yaml_path}"

    cfg = OmegaConf.load(str(yaml_path))

    # Substitutions vs the LIBERO clone
    assert cfg.framework.embodiment.name == "franka_calvin", \
        f"framework.embodiment.name = {cfg.framework.embodiment.name!r}"
    assert cfg.datasets.vla_data.data_mix == "calvin_uamvla", \
        f"datasets.vla_data.data_mix = {cfg.datasets.vla_data.data_mix!r}"
    assert cfg.datasets.vla_data.data_root_dir == "datasets/uamvla_calvin", \
        f"datasets.vla_data.data_root_dir = {cfg.datasets.vla_data.data_root_dir!r}"

    # Identical-to-LIBERO contract checks (catch yaml drift):
    assert list(cfg.framework.state_encoder.normalization.apply_to) == [
        "arm_0.ee_pose", "arm_0.joint_pos", "gripper_0",
    ], f"unexpected state normalization apply_to: {cfg.framework.state_encoder.normalization.apply_to}"
    assert cfg.framework.name == "UamVLA"
    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.future_action_window_size == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_yaml_loadable -v
```

Expected: FAIL with `AssertionError: missing .../uamvla_calvin.yaml`.

- [ ] **Step 3: Create the yaml as a 4-line-substitution clone of `uamvla_libero.yaml`**

Create `starVLA/config/training/uamvla_calvin.yaml` with this exact content (this is `uamvla_libero.yaml` with only the four substitutions called out in spec §5.3):

```yaml
run_id: uamvla_calvin_phase1
run_root_dir: playground/Checkpoints
seed: 42
trackers: [jsonl, wandb]
wandb_project: uamvla
wandb_entity: tancilon
is_debug: false
version_id: "0.1"

# ───────────────────────────────────────────────────────
# framework: model construction
# ───────────────────────────────────────────────────────
framework:
  name: UamVLA
  embodiment:
    name: franka_calvin
    action_dim: 7
  qwenvl:
    base_vlm: ckpt/Qwen3-VL-8B-Instruct
    attn_implementation: flash_attention_2
  action_model:
    future_action_window_size: 7        # = action_horizon - 1; starVLA convention (read by ModelClient)
    num_bins: 256                        # ActionTokenizer bin count
    action_dim: 7
  state_encoder:
    type: modular
    register_special_tokens: true
    normalization:
      mode: q99                # one of: q99 | mean_std | min_max | none
      apply_to:                # canonical field paths to normalize
        - arm_0.ee_pose
        - arm_0.joint_pos
        - gripper_0
  vae:
    path: ckpt/pretrained_vae
  aux_heads:
    action:
      enabled: true
      loss_weight: 1.0
      lr: 1.0e-4
    pose:
      enabled: true
      loss_weight: 0.5
      lr: 1.0e-4
      pose_mode: rot_matrix
      sde_mode: ve
      num_queries: 4
      semantic_dim: 512
      sampling_steps: 500
    future:
      enabled: true
      loss_weight: 0.1
      lr: 1.0e-4
      view_idx: 0
      target_resize: 320                 # ppv=400 → 20*16
      denoiser_depth: 3
      denoiser_embed_dim: 1024
      repeat_factor: 4
    recon:
      enabled: true
      loss_weight: 0.1
      lr: 1.0e-4
      view_idx: 0
      target_resize: 320
      denoiser_depth: 3
      denoiser_embed_dim: 1024
      repeat_factor: 4

# ───────────────────────────────────────────────────────
# datasets: data loader plugin selection
# ───────────────────────────────────────────────────────
datasets:
  vla_data:
    dataset_py: uamvla_dataset           # starVLA/dataloader/uamvla_dataset.py
    data_root_dir: datasets/uamvla_calvin
    data_mix: calvin_uamvla
    image_size: 640                       # Qwen3-VL ppv=400 ⇒ 640x640 (UamVLA path: currently unused — see spec §5.3)
    per_device_batch_size: 2              # 8B backbone is tighter than 4B (starvla_cotrain_libero.yaml uses 16)
    delete_pause_frame: false

# ───────────────────────────────────────────────────────
# trainer: optimization / logging
# ───────────────────────────────────────────────────────
trainer:
  epochs: 50
  max_train_steps: 100000
  num_warmup_steps: 500
  save_interval: 2000
  eval_interval: 5000

  learning_rate:
    base:                  1.0e-4         # fallback for unmatched param groups
    qwen_vl_interface:     2.0e-5         # backbone (UamVLA's backbone_lr equivalent)
    state_encoder:         1.0e-4
    # per-head LR is read from framework.aux_heads.{name}.lr by UamVLA.get_lr_groups()

  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 1.0e-6

  freeze_modules: null                    # Phase 1 from-scratch — no freeze
  loss_scale:
    vla: 1.0                              # vlm not used (supports_training_tag returns False)

  max_grad_norm: 1.0
  logging_frequency: 10
  gradient_clipping: 1.0
  gradient_accumulation_steps: 16
  gradient_checkpointing: true

  optimizer:
    name: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-8
    weight_decay: 1.0e-8

  visualization:
    enabled: true
    train_every_n_steps: 1000
    num_samples: 1

  is_resume: false
  resume_epoch: null
  resume_step: null
  enable_gradient_checkpointing: true
  enable_mixed_precision_training: true
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_yaml_loadable -v
```

Expected: **PASS**.

- [ ] **Step 5: Commit**

```bash
git add starVLA/config/training/uamvla_calvin.yaml tests/test_uamvla_calvin_pipeline_local.py
git commit -m "[calvin] add UamVLA training config uamvla_calvin.yaml"
```

---

## Task 6: Training launcher `examples/calvin/train_files/run_uamvla_calvin_train.sh` (G5)

**Files:**
- Create: `examples/calvin/train_files/run_uamvla_calvin_train.sh`

This is a shell script. The local "test" is a `bash -n` syntax check; the launcher is exercised end-to-end only in Task 9 (remote).

The closest precedent is `examples/LIBERO/train_files/run_libero_train.sh` (also a UamVLA launcher) — clone from it, swap libero→calvin paths. The result is equivalent to spec §5.5's prescribed substitutions over the QwenPI launcher; using the LIBERO UamVLA precedent reduces structural drift between the two UamVLA launchers.

- [ ] **Step 1: Create the launcher**

Create `examples/calvin/train_files/run_uamvla_calvin_train.sh` with this content:

```bash


export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=UamVLA
freeze_module_list=''
base_vlm=ckpt/Qwen3-VL-8B-Instruct
config_yaml=./starVLA/config/training/uamvla_calvin.yaml
calvin_data_root=datasets/uamvla_calvin
data_mix=calvin_uamvla
run_root_dir=./playground/Checkpoints
run_id=uamvla_calvin_phase1
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${calvin_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 2 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.save_interval 2000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 5000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project uamvla_calvin \
  --wandb_entity tancilon \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
```

- [ ] **Step 2: Bash syntax check**

Run:
```bash
bash -n examples/calvin/train_files/run_uamvla_calvin_train.sh && echo "OK"
```

Expected: prints `OK` (no syntax errors).

- [ ] **Step 3: Make executable + commit**

```bash
chmod +x examples/calvin/train_files/run_uamvla_calvin_train.sh
git add examples/calvin/train_files/run_uamvla_calvin_train.sh
git commit -m "[calvin] add UamVLA training launcher run_uamvla_calvin_train.sh"
```

---

## Task 7: Eval `unnorm_key` rename (G7)

**Files:**
- Modify: `examples/calvin/eval_files/eval_calvin.sh:10` (single line)

This is a one-line literal change. The remote eval-load smoke (§7.2 step 5 in the spec, exercised in Task 9) is the actual verification. Locally we only verify the file change took effect.

- [ ] **Step 1: Apply the rename**

In `examples/calvin/eval_files/eval_calvin.sh`, change:

```bash
unnorm_key="franka"
```

to:

```bash
unnorm_key="franka_calvin"
```

- [ ] **Step 2: Verify the change with a grep + bash syntax check**

Run:
```bash
grep -n 'unnorm_key=' examples/calvin/eval_files/eval_calvin.sh
bash -n examples/calvin/eval_files/eval_calvin.sh && echo "OK"
```

Expected output:
```
10:unnorm_key="franka_calvin"
OK
```

If the grep returns more than one matching line, the line numbers may have shifted slightly — confirm by reading the file. The single occurrence should be the assignment, not a comment or reference.

- [ ] **Step 3: Commit**

```bash
git add examples/calvin/eval_files/eval_calvin.sh
git commit -m "[calvin] eval_calvin.sh: unnorm_key franka -> franka_calvin

Without this rename, ModelClient._check_unnorm_key fires an
AssertionError on every UamVLA-CALVIN checkpoint load — train
writes dataset_statistics.json keyed by embodiment, which for
CALVIN is 'franka_calvin', not 'franka'."
```

---

## Task 8: README addendum

**Files:**
- Modify: `examples/calvin/README.md` (insert near top, before existing QwenPI sections)

The existing README is QwenPI-focused. Add a "UamVLA path" section that distinguishes the two pipelines and gives a UamVLA quickstart, plus a known-limitation callout about state passthrough (per spec §5.6).

- [ ] **Step 1: Insert the UamVLA path section**

In `examples/calvin/README.md`, find this header (currently the document title, lines 1-2):

```markdown
# 🚀 Calvin Training and Evaluation
```

Replace the early frontmatter (lines 1-7, ending at the `> **Note:**` block about UNT team — keep that line) by inserting a new "Two pipelines" section followed by a UamVLA quickstart **before** the existing `## 📊 Benchmark Results (Calvin)` section. Concretely, edit the file so its top reads:

```markdown
# 🚀 Calvin Training and Evaluation

This document describes how to **train and evaluate StarVLA models on the Calvin benchmark**, including dataset preparation, training configuration, and evaluation procedures.

> **Note:** Calvin benchmark experiments were conducted by the UNT team. For inquiries, please contact Zhijie Song (1600013008@pku.edu.cn) or Feng Yan (bphengyan@163.com).


---

## 🛣️ Two pipelines

This directory hosts **two independent training pipelines** that share only the eval scripts in `eval_files/`:

| Pipeline | Format | Training entry | Training config |
|----------|--------|----------------|-----------------|
| **QwenPI** (existing) | LeRobot v3.0 | `run_calvin_train.sh` | `starvla_train_calvin.yaml` |
| **UamVLA** (this section) | UAM unified format (`data.jsonl` + `statistics.yaml`) | `run_uamvla_calvin_train.sh` | `starVLA/config/training/uamvla_calvin.yaml` |

The two pipelines do not share preprocessing — pick one and stay in lane.

### UamVLA quickstart

1. **Preprocess** a CALVIN split into UAM format (requires the `calvin_env` conda environment):

   ```bash
   python runners/preprocess_calvin.py \
       --input_dir /path/to/calvin/task_D_D/training \
       --output_dir datasets/uamvla_calvin/task_D_D \
       --dataset_source task_D_D
   ```

   The preprocessor regenerates `statistics.yaml` from the merged `data.jsonl` on every run, so re-running refreshes stats if `CalvinAdapter` ever changes.

2. **Train**:

   ```bash
   bash examples/calvin/train_files/run_uamvla_calvin_train.sh
   ```

   Verify the paths inside the script first (`base_vlm`, `calvin_data_root`, `run_root_dir`).

3. **Evaluate** — uses the same eval scripts as QwenPI (`run_policy_server.sh` + `eval_calvin.sh`); see §3 below.

> **Known limitation (CALVIN eval state distribution gap).** UamVLA training feeds `canonical_state` to the model every step, but the current CALVIN eval client (`eval_calvin.py`) sends only `image` + `lang` to the policy server. A trained UamVLA-CALVIN checkpoint loads cleanly into the eval pipeline (after the `unnorm_key` rename in `eval_calvin.sh`) but inference uses the state-less branch — a real distribution mismatch from training. Expect avoidable success-rate loss until a CALVIN parallel of `docs/superpowers/specs/2026-04-30-uamvla-eval-state-passthrough-design.md` lands. Tracked as a follow-up spec.

---

```

(Then leave the existing `## 📊 Benchmark Results (Calvin)` and the rest of the file unchanged.)

- [ ] **Step 2: Manual review**

Run:
```bash
head -50 examples/calvin/README.md
```

Expected: the new "Two pipelines" + "UamVLA quickstart" + "Known limitation" content appears at the top, immediately after the introductory `> **Note:**` block. The original content from `## 📊 Benchmark Results (Calvin)` onwards is unchanged.

- [ ] **Step 3: Commit**

```bash
git add examples/calvin/README.md
git commit -m "[calvin] README: add UamVLA path quickstart + known-limitation callout"
```

---

## Task 9: Remote e2e (manual acceptance — not auto-run by CI)

**Goal:** On a machine with `calvin_env` conda environment installed, run the full preprocess → train → eval-load chain on the debug dataset and verify each stage produces the expected artifacts. This task does not commit anything by itself — it is the gating manual check before declaring the spec done.

**Pre-conditions:**
- Machine has `calvin_env` conda environment with PyBullet/EGL working.
- `starVLA` conda environment for training/server side.
- Local clone of the debug dataset at `/Users/tancilon/develop/localgit/UamVLA/datasets/calvin_debug_dataset` (or pass an alternate path).

- [ ] **Step 1: Preprocess the debug dataset**

In `calvin_env`:

```bash
python runners/preprocess_calvin.py \
    --input_dir /Users/tancilon/develop/localgit/UamVLA/datasets/calvin_debug_dataset/training \
    --output_dir datasets/uamvla_calvin/task_D_D \
    --dataset_source calvin_debug
```

Expected:
- Exit code 0.
- `datasets/uamvla_calvin/task_D_D/data.jsonl` exists, with ≈576 rows (9 windows × ≤64 frames; some frames may be skipped if `--on_missing_target=abort` fires — switch to `skip` if so).
- `datasets/uamvla_calvin/task_D_D/statistics.yaml` exists and contains `state_stats: {franka_calvin: ...}` with no `robot_obs_mean`/`robot_obs_std` keys (verify by `grep state_stats datasets/uamvla_calvin/task_D_D/statistics.yaml`).
- At least one image in each of `images/obs/static/`, `images/obs/wrist/`, `images/target/`, `images/future/`.
- At least one `.npy` in `point_clouds/`.

If `make_calvin_env_adapter` raises `FileNotFoundError` for a `.hydra/config.yaml`, see spec §9 risks — likely fix is pointing `--input_dir` at the parent directory that contains `.hydra/`, or the debug dataset's training subdir genuinely lacks it (in which case verify on a real `task_D_D` split before declaring this step blocking).

- [ ] **Step 2: Insert one-time vision-token-count probe**

Before running training, add a temporary log line to `starVLA/model/modules/uamvla/backbone_wrapper.py` immediately after `proc_out = self.processor(...)` (line 278 area). The change:

```python
        proc_out = self.processor(
            text=texts,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        )

        # TEMP probe (revert before commit) — verify Qwen3VLProcessor produces
        # 400 tokens per view for 256x256 input. UamVLA.py:181 hardcodes
        # patches_per_view=400; mismatches surface as slice_image_tokens errors.
        try:
            _input_ids = proc_out.get("input_ids", None)
            _image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if _input_ids is not None and _image_token_id is not None:
                _per_view_count = int((_input_ids[0] == _image_token_id).sum())
                # If the batch has multiple images per sample, divide by len(images[0]).
                _n_views = max(1, len(images[0]) if images else 1)
                logger.info(
                    "[ppv-probe] vision <|image_pad|> tokens in row 0: %d total, "
                    "%d per view (n_views=%d). Expected 400 to match UamVLA.py:181 ppv.",
                    _per_view_count, _per_view_count // _n_views, _n_views,
                )
        except Exception as _e:  # pragma: no cover
            logger.warning("[ppv-probe] failed: %s", _e)
```

This block goes **only** in this remote-test branch. It will be reverted in step 4.

- [ ] **Step 3: 2-step training smoke**

In `starVLA` env:

```bash
python starVLA/training/train_starvla.py \
    --config_yaml starVLA/config/training/uamvla_calvin.yaml \
    --datasets.vla_data.data_root_dir datasets/uamvla_calvin \
    --is_debug True \
    --trainer.max_train_steps 2
```

Expected:
- Completes 2 training steps without exception; loss is finite.
- The probe log fires at least once with `per view = 400`. If it logs a different number, `slice_image_tokens` will eventually fail in future/recon heads — investigate `Qwen3VLProcessor`'s `min_pixels`/`max_pixels` defaults vs the loaded checkpoint's processor config. Resolution options (in order of preference): (a) override `min_pixels`/`max_pixels` at processor instantiation in `backbone_wrapper.py`, (b) pre-resize the PIL images before `self.processor(...)`. **Do not declare this step green until the probe shows 400.**

- [ ] **Step 4: Revert the probe**

After the smoke is green, remove the TEMP probe block added in step 2:

```bash
git checkout -- starVLA/model/modules/uamvla/backbone_wrapper.py
```

Verify with `git diff starVLA/model/modules/uamvla/backbone_wrapper.py` that the file matches HEAD.

- [ ] **Step 5: Eval-load smoke**

This proves Task 7 (G7 unnorm_key rename) is correctly applied.

In `starVLA` env (terminal 1):

```bash
bash examples/calvin/eval_files/run_policy_server.sh
```

Expected: server boots, prints listening port, no errors.

In `calvin_env` (terminal 2):

```bash
bash examples/calvin/eval_files/eval_calvin.sh \
    --args.pretrained-path playground/Checkpoints/uamvla_calvin_phase1/checkpoints/<2-step-ckpt-file> \
    --args.num_sequences 1
```

Expected: ModelClient loads the checkpoint **without** firing `AssertionError: The 'unnorm_key' you chose is not in the set of available dataset statistics`. Single eval episode runs to completion (success or failure of the rollout itself is irrelevant — we are testing the load + inference path, not benchmark accuracy).

If the assertion fires: re-confirm Task 7 was applied (`grep unnorm_key examples/calvin/eval_files/eval_calvin.sh` should show `franka_calvin`). If it shows `franka_calvin` and the assertion still fires, inspect the checkpoint's `dataset_statistics.json` keys — `train_starvla.py` should have keyed them by `franka_calvin`; if they are keyed differently, this points at a `_save_dataset_statistics_json` regression worth a separate bug.

- [ ] **Step 6: Sign off**

If steps 1–5 all pass: the spec is delivered. Manually annotate the spec PR / branch description with:

```
Task 9 (remote e2e) verified on <hostname>, <date>:
  - preprocess on calvin_debug_dataset/training: <N> rows, statistics.yaml has state_stats[franka_calvin]
  - 2-step training smoke: completed; ppv-probe logged 400 per view
  - eval-load smoke: ModelClient loaded uamvla_calvin_phase1/<ckpt> with unnorm_key=franka_calvin
```

Nothing to commit from this task.

---

## Spec Coverage Map

| Spec section / requirement | Task |
|----------------------------|------|
| §3.2 G1: missing CLI runner | Task 4 |
| §3.2 G2: stats migration | Task 1 |
| §3.2 G3: missing mixture entry | Task 2 |
| §3.2 G4: missing training yaml | Task 5 |
| §3.2 G5: missing launcher | Task 6 |
| §3.2 G6: missing local pipeline test | Tasks 1, 2, 3, 4, 5 (cumulative in `tests/test_uamvla_calvin_pipeline_local.py`) |
| §3.2 G7: `unnorm_key` rename | Task 7 |
| §5.1 stats migration body | Task 1 step 3 |
| §5.2 mixture entry | Task 2 step 3 |
| §5.3 training yaml clone | Task 5 step 3 |
| §5.3 image_size dead-field caveat (yaml comment) | Task 5 step 3 (inline comment in yaml) |
| §5.4 CLI args, defaults, header comment | Task 4 step 3 |
| §5.5 launcher clone | Task 6 step 1 |
| §5.6 README addendum (UamVLA path + known-limitation callout) | Task 8 |
| §5.7 eval `unnorm_key` rename | Task 7 |
| §6.1–6.4 data contract | Implicit (covered by tests in Tasks 1, 3, 5) |
| §7.1 local unit tests (7 cases) | Tasks 1, 2, 3, 4, 5 |
| §7.2 remote e2e (preprocess, ppv-probe, train smoke, eval-load smoke) | Task 9 |
| §7.3 existing `test_preprocessor_smoke.py` left in place | (no action needed) |
| §9 risk: ppv != 400 | Task 9 step 2-3 (probe + resolution path) |
| §9 risk: state distribution gap | Documented in Task 8 README callout |
| §10 naming convention | Embedded in all task code |
