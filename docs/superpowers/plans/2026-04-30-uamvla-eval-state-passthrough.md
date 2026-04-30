# UamVLA Eval-Time State Passthrough — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the LIBERO eval train/eval distribution gap by sending the same `canonical_state` the training pipeline produces, via client-side adapter+normalizer in `model2libero_interface.py` keyed by a `statistics.yaml` shipped with the checkpoint.

**Architecture:** Client-side conversion (libero_env). Trainer copies `statistics.yaml` to the run dir. `ModelClient.__init__` resolves `unnorm_key`, locates `run_dir = ckpt.parents[1]`, opt-in loads adapter+normalizer if `state_stats[unnorm_key]` is present. `ModelClient.step()` always pops the new key `uamvla_raw_state`, gates conversion on a public `uamvla_state_enabled` flag. `eval_libero.py` only assembles raw state when the flag is true. Server-side `stack_canonical` gains a `torch.as_tensor` wrap to tolerate WebSocket-deserialized numpy leaves.

**Tech Stack:** Python 3.10 (training env) / Python 3.8 (libero_env), PyTorch, OmegaConf, msgpack-numpy, pytest, accelerate.

**Spec reference:** [docs/superpowers/specs/2026-04-30-uamvla-eval-state-passthrough-design.md](../specs/2026-04-30-uamvla-eval-state-passthrough-design.md) (commit `b39a255`).

---

## File Structure

**New files:**
- `tools/probes/probe_libero_obs_keys.py` — P1 probe; standalone script run inside `libero_env`.
- `tests/test_libero_quat_axisangle_alignment.py` — P2 alignment test (skipped without LIBERO env).
- `tests/test_libero_env_imports.py` — P3 import smoke (skipped without `libero_env`).
- `tests/test_to_numpy_leaves.py` — unit tests for the dict→numpy walk helper.
- `tests/test_stack_canonical_numpy_input.py` — robustness test for §6.4 wrap.
- `tests/test_train_starvla_persist_stats_yaml.py` — covers §6.1 trainer copy.
- `tests/test_eval_state_passthrough_wiring.py` — fake-client wiring test for §6.3.1/§6.3.2.
- `tests/test_train_eval_state_alignment.py` — §8.2 train/eval canonical_state contract.
- `tests/test_eval_state_passthrough_integration.py` — §8.4 server boot + single infer (marked `slow`).

**Modified files:**
- `starVLA/training/train_starvla.py:195-231` — `_save_dataset_statistics_json` extended; new `_copy_stats_yaml_to_run_dir` static helper.
- `starVLA/model/modules/uamvla/collator_helpers.py:28-50` — `stack_canonical` wraps leaves with `torch.as_tensor`.
- `examples/LIBERO/eval_files/model2libero_interface.py` — `__init__` rewrite (B1+B2 fixes + UamVLA opt-in load); `step()` extension; module-level `_to_numpy_leaves` helper.
- `examples/LIBERO/eval_files/eval_libero.py` — gated `uamvla_raw_state` assembly into `example_dict`.

**Untouched:** `UamVLA.predict_action`, the WebSocket transport, the dataloader, the preprocessor, `dataset_statistics.json` schema.

---

## Phase 0 — Pre-flight Probes

These artifacts validate design assumptions §4 R1/R2/R3. They must be created in the same PR as the rest of the work; they only need to *run successfully* on the engineer's libero_env machine before we declare the design honored.

### Task 1: Create P1 probe — `robot0_joint_pos` existence

**Files:**
- Create: `tools/probes/probe_libero_obs_keys.py`

- [ ] **Step 1: Create the probe script**

```python
"""P1 probe (spec §8.1): verify LIBERO env exposes robot0_joint_pos.

Run inside libero_env with a valid LIBERO_HOME / MUJOCO_GL setup:

    python tools/probes/probe_libero_obs_keys.py

Exit code 0 ⇒ key present with shape (7,). Non-zero ⇒ design must be revisited.
"""
from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:
        print(f"FAIL: cannot import LIBERO ({e}); run inside libero_env.", file=sys.stderr)
        return 2

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = suite.get_task_bddl_file_path(0)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    obs = env.reset()

    print(f"obs.keys() = {sorted(obs.keys())}")
    if "robot0_joint_pos" not in obs:
        print("FAIL: 'robot0_joint_pos' missing from obs dict.", file=sys.stderr)
        return 1

    arr = np.asarray(obs["robot0_joint_pos"])
    print(f"robot0_joint_pos.shape = {arr.shape}, dtype = {arr.dtype}")
    if arr.shape != (7,):
        print(f"FAIL: expected shape (7,), got {arr.shape}.", file=sys.stderr)
        return 1

    print("OK: robot0_joint_pos present with shape (7,).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Commit**

```bash
git add tools/probes/probe_libero_obs_keys.py
git commit -m "probes: add libero obs-keys probe for joint_pos"
```

- [ ] **Step 3: Document** (no code change) — note that this probe must be run inside `libero_env` before merging the rest of the PR.

### Task 2: Create P2 probe — quat→axisangle alignment

**Files:**
- Create: `tests/test_libero_quat_axisangle_alignment.py`

- [ ] **Step 1: Write the probe test (`pytest`-runnable, skips if LIBERO not importable)**

```python
"""P2 probe (spec §8.1): verify _quat2axisangle(robot0_eef_quat) matches HDF5 obs/ee_ori.

Skipped automatically if libero_env deps are missing. Run inside libero_env when
testing the design end-to-end:

    pytest tests/test_libero_quat_axisangle_alignment.py -v
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

libero = pytest.importorskip("libero.libero")
h5py = pytest.importorskip("h5py")


# Canonical robosuite quat→axisangle (mirrors examples/LIBERO/eval_files/eval_libero.py:_quat2axisangle).
def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


HDF5_GLOB_HINT = "dataset/libero2uam/raw/libero_spatial/**/*.hdf5"


@pytest.mark.libero_env
def test_eval_quat2axisangle_matches_hdf5_ee_ori(tmp_path):
    candidates = list(Path(".").glob(HDF5_GLOB_HINT))
    if not candidates:
        pytest.skip(f"No LIBERO HDF5 files matching {HDF5_GLOB_HINT}")
    h5_path = candidates[0]

    with h5py.File(h5_path, "r") as f:
        demo = next(iter(f["data"].values()))
        ee_ori = demo["obs/ee_ori"][:]   # (T, 3) axis-angle from HDF5
        ee_quat = demo["obs/ee_states"][:, 3:7] if "ee_states" in demo["obs"] else demo["obs/ee_quat"][:]

    n = min(10, ee_ori.shape[0])
    diffs = []
    for t in range(n):
        recovered = _quat2axisangle(ee_quat[t])
        diffs.append(np.abs(recovered - ee_ori[t]).max())

    max_diff = max(diffs)
    assert max_diff < 1e-5, (
        f"_quat2axisangle output diverges from HDF5 obs/ee_ori "
        f"(max abs diff over {n} timesteps = {max_diff:.3e}; tol 1e-5)"
    )
```

- [ ] **Step 2: Run the probe (locally if libero_env is set up; otherwise document expected behavior)**

Run: `pytest tests/test_libero_quat_axisangle_alignment.py -v`
Expected: PASS in libero_env, SKIP elsewhere.

- [ ] **Step 3: Commit**

```bash
git add tests/test_libero_quat_axisangle_alignment.py
git commit -m "probes: P2 quat→axisangle alignment test"
```

### Task 3: Create P3 probe — libero_env import smoke

**Files:**
- Create: `tests/test_libero_env_imports.py`

- [ ] **Step 1: Write the import smoke test**

```python
"""P3 probe (spec §8.1): verify libero_env can import StateNormalizer + LiberoAdapter.

If this test fails inside libero_env, R3 fallback (inline q99 in client) activates.
"""
import importlib

import pytest


@pytest.mark.libero_env
def test_libero_env_can_import_state_normalizer():
    """The two modules client-side conversion depends on must be importable."""
    importlib.import_module(
        "starVLA.model.modules.uamvla.data.embodiment_adapter"
    )
    importlib.import_module(
        "starVLA.model.modules.uamvla.data.state_normalizer"
    )
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_libero_env_imports.py -v`
Expected: PASS in libero_env. If FAIL, switch §6.3.1 to inline q99 fallback (deferred until that phase).

- [ ] **Step 3: Commit**

```bash
git add tests/test_libero_env_imports.py
git commit -m "probes: P3 libero_env import smoke for StateNormalizer"
```

---

## Phase 1 — Server-Side Robustness

### Task 4: `stack_canonical` accepts numpy leaves (§6.4)

**Files:**
- Test: `tests/test_stack_canonical_numpy_input.py`
- Modify: `starVLA/model/modules/uamvla/collator_helpers.py:28-50`

- [ ] **Step 1: Write the failing test**

```python
"""Spec §6.4 — stack_canonical must tolerate numpy-leaf canonical_state dicts
arriving via msgpack-numpy WebSocket deserialization."""
import numpy as np
import torch

from starVLA.model.modules.uamvla.collator_helpers import stack_canonical


def test_stack_canonical_accepts_numpy_leaves():
    a = {
        "arm_0": {
            "ee_pose":   np.zeros(9, dtype=np.float32),
            "joint_pos": np.zeros(7, dtype=np.float32),
        },
        "gripper_0": np.zeros(1, dtype=np.float32),
    }
    b = {
        "arm_0": {
            "ee_pose":   np.ones(9, dtype=np.float32),
            "joint_pos": np.ones(7, dtype=np.float32),
        },
        "gripper_0": np.ones(1, dtype=np.float32),
    }
    out = stack_canonical([a, b])

    assert isinstance(out["arm_0"]["ee_pose"], torch.Tensor)
    assert out["arm_0"]["ee_pose"].shape == (2, 9)
    assert out["arm_0"]["joint_pos"].shape == (2, 7)
    assert out["gripper_0"].shape == (2, 1)


def test_stack_canonical_still_accepts_torch_leaves():
    """Regression: training-side path must not break."""
    a = {
        "arm_0": {"ee_pose": torch.zeros(9), "joint_pos": torch.zeros(7)},
        "gripper_0": torch.zeros(1),
    }
    b = {
        "arm_0": {"ee_pose": torch.ones(9), "joint_pos": torch.ones(7)},
        "gripper_0": torch.ones(1),
    }
    out = stack_canonical([a, b])
    assert out["arm_0"]["ee_pose"].shape == (2, 9)
    assert torch.equal(out["gripper_0"], torch.stack([torch.zeros(1), torch.ones(1)]))
```

- [ ] **Step 2: Run; expect failure**

Run: `pytest tests/test_stack_canonical_numpy_input.py -v`
Expected: FAIL on `test_stack_canonical_accepts_numpy_leaves` with `TypeError: stack(): argument 'tensors' (position 1) must be tuple of Tensors, not ndarray`.

- [ ] **Step 3: Patch `stack_canonical`**

Replace the body of `stack_canonical` in `starVLA/model/modules/uamvla/collator_helpers.py`:

```python
def stack_canonical(state_list: list[dict]) -> dict:
    """Stack list of canonical_state dicts along batch dim 0.

    Phase 1 contract: at most ONE level of nesting.
    Supported shapes:
        {"limb_id": tensor}                          # flat
        {"limb_id": {"key": tensor, "key2": tensor}} # nested (1 level)
    Each leaf is wrapped in ``torch.as_tensor`` so callers may pass numpy
    arrays (e.g. WebSocket-deserialized canonical_state from the eval
    client) without manual conversion.
    """
    template = state_list[0]
    out: dict = {}
    for limb_id, val in template.items():
        if isinstance(val, dict):
            out[limb_id] = {
                k: torch.stack(
                    [torch.as_tensor(s[limb_id][k]) for s in state_list], dim=0
                )
                for k in val.keys()
            }
        else:
            out[limb_id] = torch.stack(
                [torch.as_tensor(s[limb_id]) for s in state_list], dim=0
            )
    return out
```

- [ ] **Step 4: Run; expect pass**

Run: `pytest tests/test_stack_canonical_numpy_input.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the existing collator_helpers tests for regression**

Run: `pytest tests/test_collator_helpers.py -v`
Expected: all pre-existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add starVLA/model/modules/uamvla/collator_helpers.py tests/test_stack_canonical_numpy_input.py
git commit -m "[uamvla] stack_canonical: torch.as_tensor leaf wrap to tolerate numpy input"
```

---

## Phase 2 — Trainer-Side Stats Persistence (§6.1)

### Task 5: Trainer copies `statistics.yaml` to run dir

**Files:**
- Test: `tests/test_train_starvla_persist_stats_yaml.py`
- Modify: `starVLA/training/train_starvla.py` (imports + `VLATrainer._save_dataset_statistics_json` + new static helper `_copy_stats_yaml_to_run_dir`)

- [ ] **Step 1: Write the failing test**

```python
"""Spec §6.1 — verify trainer copies the source statistics.yaml verbatim
into output_dir alongside dataset_statistics.json."""
from pathlib import Path

import pytest


def test_copy_stats_yaml_to_run_dir_copies_file(tmp_path: Path):
    from starVLA.training.train_starvla import VLATrainer

    src = tmp_path / "statistics.yaml"
    src.write_text("view_names:\n- static\n- wrist\n")
    out = tmp_path / "out"
    out.mkdir()

    VLATrainer._copy_stats_yaml_to_run_dir(src, out)

    dst = out / "statistics.yaml"
    assert dst.exists()
    assert dst.read_text() == src.read_text()


def test_copy_stats_yaml_to_run_dir_noop_when_source_missing(tmp_path: Path):
    from starVLA.training.train_starvla import VLATrainer

    src = tmp_path / "absent.yaml"
    out = tmp_path / "out"
    out.mkdir()

    # Should not raise.
    VLATrainer._copy_stats_yaml_to_run_dir(src, out)
    assert not (out / "statistics.yaml").exists()
```

- [ ] **Step 2: Run; expect failure**

Run: `pytest tests/test_train_starvla_persist_stats_yaml.py -v`
Expected: FAIL — `AttributeError: type object 'VLATrainer' has no attribute '_copy_stats_yaml_to_run_dir'`.

- [ ] **Step 3: Add `shutil` import and the helper to `train_starvla.py`**

In `starVLA/training/train_starvla.py`, add `import shutil` after `import os` (around line 17). Inside `class VLATrainer(...)`, just below the existing `_save_dataset_statistics_json` method, add:

```python
    @staticmethod
    def _copy_stats_yaml_to_run_dir(stats_yaml_src: Path, output_dir: Path) -> None:
        """Copy UamVLA statistics.yaml into the run dir if the source exists.

        Mirrors the dataset_statistics.json placement so read_mode_config's
        run_dir = checkpoint_pt.parents[1] resolution finds both files.
        """
        if stats_yaml_src.exists():
            shutil.copy(stats_yaml_src, output_dir / "statistics.yaml")
            logger.info(f"Copied statistics.yaml to {output_dir}")
```

- [ ] **Step 4: Wire helper into `_save_dataset_statistics_json`**

In `_save_dataset_statistics_json`, immediately after the existing `logger.info(f"Wrote dataset_statistics.json to {out_path}")` line (currently around line 231), add:

```python
        self._copy_stats_yaml_to_run_dir(stats_yaml, Path(self.config.output_dir))
```

(`stats_yaml` is already a local in scope from earlier in the method.)

- [ ] **Step 5: Run tests; expect pass**

Run: `pytest tests/test_train_starvla_persist_stats_yaml.py -v`
Expected: 2 passed.

- [ ] **Step 6: Run trainer-related regression tests**

Run: `pytest tests/test_train_starvla_patches.py -v`
Expected: pre-existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add starVLA/training/train_starvla.py tests/test_train_starvla_persist_stats_yaml.py
git commit -m "[trainer] persist statistics.yaml to run dir for eval-time state passthrough"
```

---

## Phase 3 — Eval-Client Internal Helper

### Task 6: `_to_numpy_leaves` helper (§6.3.3)

**Files:**
- Test: `tests/test_to_numpy_leaves.py`
- Modify: `examples/LIBERO/eval_files/model2libero_interface.py` (add module-level helper above `class ModelClient`).

- [ ] **Step 1: Write the failing test**

```python
"""Spec §6.3.3 — module-level helper that walks a 1-level nested dict and
converts torch.Tensor leaves to numpy.ndarray, leaving other types untouched."""
import numpy as np
import torch

from examples.LIBERO.eval_files.model2libero_interface import _to_numpy_leaves


def test_to_numpy_leaves_converts_torch_in_nested_dict():
    src = {
        "arm_0": {
            "ee_pose":   torch.tensor([1.0, 2.0, 3.0]),
            "joint_pos": torch.zeros(7),
        },
        "gripper_0": torch.tensor([0.5]),
    }

    out = _to_numpy_leaves(src)

    assert isinstance(out["arm_0"]["ee_pose"], np.ndarray)
    assert isinstance(out["arm_0"]["joint_pos"], np.ndarray)
    assert isinstance(out["gripper_0"], np.ndarray)
    np.testing.assert_array_equal(out["arm_0"]["ee_pose"], np.array([1.0, 2.0, 3.0]))


def test_to_numpy_leaves_passes_through_non_torch():
    src = {
        "a": np.array([1, 2, 3], dtype=np.float32),
        "b": {"c": [1, 2], "d": "hello"},
    }
    out = _to_numpy_leaves(src)

    assert isinstance(out["a"], np.ndarray)
    assert out["b"]["c"] == [1, 2]
    assert out["b"]["d"] == "hello"
```

- [ ] **Step 2: Run; expect failure**

Run: `pytest tests/test_to_numpy_leaves.py -v`
Expected: FAIL — `ImportError: cannot import name '_to_numpy_leaves' ...`.

- [ ] **Step 3: Add helper to `model2libero_interface.py`**

At the top of `examples/LIBERO/eval_files/model2libero_interface.py`, just below the existing imports and above `class ModelClient:`, add:

```python
def _to_numpy_leaves(d: dict) -> dict:
    """Walk a 1-level nested dict; convert torch.Tensor leaves to numpy.

    Used at the WebSocket boundary: msgpack-numpy serializes numpy.ndarray
    natively but not torch.Tensor. Server-side stack_canonical accepts numpy
    after the §6.4 torch.as_tensor wrap.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {
                kk: vv.detach().cpu().numpy() if hasattr(vv, "detach") else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v.detach().cpu().numpy() if hasattr(v, "detach") else v
    return out
```

- [ ] **Step 4: Run; expect pass**

Run: `pytest tests/test_to_numpy_leaves.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/LIBERO/eval_files/model2libero_interface.py tests/test_to_numpy_leaves.py
git commit -m "[eval] _to_numpy_leaves helper for canonical_state wire serialization"
```

---

## Phase 4 — `ModelClient.__init__` Rewrite

### Task 7: Fake-client wiring tests for `__init__` (§8.3 part 1)

This task writes the **failing** tests that will drive Tasks 8 and 9. The same test file gets extended with step()-related cases in Task 9.

**Files:**
- Create: `tests/test_eval_state_passthrough_wiring.py`

- [ ] **Step 1: Write the wiring test scaffolding**

```python
"""Spec §8.3 — ModelClient + step() wiring tests against a stubbed
WebSocket client. Catches the contract bugs §8.2 alignment test cannot:

- B1 (unnorm_key=None default) → must resolve via _check_unnorm_key in __init__.
- B2 (policy_ckpt_path is a .pt file) → must read run_dir/statistics.yaml,
  not Path(.pt) / "statistics.yaml".
- step() must always pop uamvla_raw_state and inject canonical_state when
  state passthrough is enabled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def stats_yaml_dict() -> dict:
    """Minimal statistics.yaml content with both action and state stats for franka_libero."""
    return {
        "view_names": ["static", "wrist"],
        "max_action_dim": 24,
        "embodiment_stats": {
            "franka_libero": {
                "action_dim": 7,
                "action_min_bound": [-1.0] * 7,
                "action_max_bound": [1.0] * 7,
            }
        },
        "state_stats": {
            "franka_libero": {
                "arm_0.ee_pose": {
                    "q01": [-0.5] * 9,
                    "q99": [0.5] * 9,
                    "min": [-1.0] * 9, "max": [1.0] * 9,
                    "mean": [0.0] * 9, "std": [0.3] * 9,
                },
                "arm_0.joint_pos": {
                    "q01": [-1.0] * 7,
                    "q99": [1.0] * 7,
                    "min": [-2.0] * 7, "max": [2.0] * 7,
                    "mean": [0.0] * 7, "std": [0.5] * 7,
                },
                "gripper_0": {
                    "q01": [0.0],
                    "q99": [1.0],
                    "min": [0.0], "max": [1.0],
                    "mean": [0.5], "std": [0.3],
                },
            }
        },
    }


@pytest.fixture
def fake_run_dir(tmp_path: Path, stats_yaml_dict: dict) -> Path:
    """Build a fake run dir with the .pt path layout read_mode_config expects."""
    run_dir = tmp_path / "fake_run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "fake.pt").write_bytes(b"")  # empty file is fine
    with open(run_dir / "statistics.yaml", "w") as f:
        yaml.safe_dump(stats_yaml_dict, f)
    return run_dir


@pytest.fixture
def patched_read_mode_config(monkeypatch, stats_yaml_dict):
    """Stub read_mode_config so __init__ can succeed without a real config.yaml."""
    fake_norm_stats = {
        "franka_libero": {
            "action": {
                "min": stats_yaml_dict["embodiment_stats"]["franka_libero"]["action_min_bound"],
                "max": stats_yaml_dict["embodiment_stats"]["franka_libero"]["action_max_bound"],
                "mask": [True] * 6 + [False],
            }
        }
    }
    fake_model_config = {
        "framework": {"action_model": {"future_action_window_size": 7}}
    }

    def fake(_path):
        return fake_model_config, fake_norm_stats

    monkeypatch.setattr(
        "examples.LIBERO.eval_files.model2libero_interface.read_mode_config",
        fake,
    )


class StubWebsocketClient:
    """Records the last predict_action payload; never opens a socket."""
    def __init__(self):
        self.last_payload = None
    def predict_action(self, query_info: dict) -> dict:
        self.last_payload = query_info
        # Return a syntactically valid response.
        return {
            "data": {
                "normalized_actions": np.zeros((1, 8, 7), dtype=np.float32)
            }
        }


@pytest.fixture
def make_client(fake_run_dir, patched_read_mode_config, monkeypatch):
    """Build a ModelClient connected to a stub websocket."""
    from examples.LIBERO.eval_files import model2libero_interface

    # Prevent the real WebsocketClientPolicy from trying to open a socket.
    monkeypatch.setattr(
        model2libero_interface, "WebsocketClientPolicy",
        lambda *a, **kw: StubWebsocketClient(),
    )

    def _make(unnorm_key=None, with_stats_yaml=True):
        if not with_stats_yaml:
            (fake_run_dir / "statistics.yaml").unlink(missing_ok=True)
        client = model2libero_interface.ModelClient(
            policy_ckpt_path=fake_run_dir / "checkpoints" / "fake.pt",
            unnorm_key=unnorm_key,
            action_ensemble=False,
        )
        return client

    return _make


# ---------------------------------------------------------------------------
# B1 + B2: __init__ resolves unnorm_key and locates statistics.yaml correctly.
# ---------------------------------------------------------------------------
def test_unnorm_key_is_resolved_when_none_passed(make_client):
    """B1: default unnorm_key=None must be resolved via _check_unnorm_key."""
    client = make_client(unnorm_key=None)
    assert client.unnorm_key == "franka_libero"


def test_state_passthrough_enabled_when_yaml_present(make_client):
    """B2 + state-passthrough init: statistics.yaml at run_dir/ engages the path."""
    client = make_client(unnorm_key=None)
    assert client.uamvla_state_enabled is True
    assert hasattr(client, "_adapter")
    assert hasattr(client, "_state_normalizer")


def test_state_passthrough_disabled_when_yaml_absent(make_client):
    """statistics.yaml absent ⇒ flag stays False; non-UamVLA models unaffected."""
    client = make_client(unnorm_key=None, with_stats_yaml=False)
    assert client.uamvla_state_enabled is False
```

- [ ] **Step 2: Run; expect failure**

Run: `pytest tests/test_eval_state_passthrough_wiring.py -v`
Expected: 3 failures (init does not yet resolve unnorm_key, has no `uamvla_state_enabled`, etc.).

- [ ] **Step 3: Commit (red TDD checkpoint)**

```bash
git add tests/test_eval_state_passthrough_wiring.py
git commit -m "[eval] test scaffolding for state-passthrough wiring (red)"
```

### Task 8: `ModelClient.__init__` rewrite (§6.3.1)

**Files:**
- Modify: `examples/LIBERO/eval_files/model2libero_interface.py:1-58`

- [ ] **Step 1: Add necessary top-of-file imports**

At the top of `examples/LIBERO/eval_files/model2libero_interface.py`, ensure these imports exist (add the missing ones):

```python
import yaml
```

`Path`, `read_mode_config`, and the rest are already imported.

- [ ] **Step 2: Replace the `__init__` body (lines ~30-58)**

Replace the section starting with `# build client to connect server policy` and ending with `self.action_chunk_size = self.get_action_chunk_size(...)` (i.e. roughly lines 30-58) with:

```python
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup

        # ----- Resolve unnorm_key once and write back to self (fixes B1).
        # eval_libero.py does not pass unnorm_key; the staticmethod in
        # get_action_stats resolved it locally but did NOT update self.
        # Doing it here makes self.unnorm_key authoritative for both
        # action stats and state stats lookups.
        _, _norm_stats = read_mode_config(policy_ckpt_path)
        self.unnorm_key = self._check_unnorm_key(_norm_stats, unnorm_key)
        self.action_norm_stats = _norm_stats[self.unnorm_key]["action"]

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {self.unnorm_key} ***")

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon  # 0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)

        # ----- UamVLA opt-in state-passthrough setup (fixes B2 + spec §6.3.1).
        # statistics.yaml lives at the run dir level (mirrors dataset_statistics.json).
        # read_mode_config resolves run_dir = checkpoint_pt.parents[1] so we use the
        # same logic here.
        self.uamvla_state_enabled = False  # public; gates eval_libero.py assembly
        run_dir = Path(policy_ckpt_path).parents[1]
        stats_yaml_path = run_dir / "statistics.yaml"
        if stats_yaml_path.exists():
            with open(stats_yaml_path) as f:
                stats_dict = yaml.safe_load(f)
            if (
                isinstance(stats_dict, dict)
                and "state_stats" in stats_dict
                and self.unnorm_key in stats_dict["state_stats"]
            ):
                from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter
                from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer
                self._adapter = LiberoAdapter()
                self._state_normalizer = StateNormalizer(
                    stats_dict=stats_dict,
                    embodiment=self.unnorm_key,
                    mode="q99",
                )
                self.uamvla_state_enabled = True
                print(f"*** UamVLA state passthrough enabled (stats: {stats_yaml_path}) ***")
```

- [ ] **Step 3: Run the wiring tests; expect 3/3 of the init-related cases to pass**

Run: `pytest tests/test_eval_state_passthrough_wiring.py::test_unnorm_key_is_resolved_when_none_passed tests/test_eval_state_passthrough_wiring.py::test_state_passthrough_enabled_when_yaml_present tests/test_eval_state_passthrough_wiring.py::test_state_passthrough_disabled_when_yaml_absent -v`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add examples/LIBERO/eval_files/model2libero_interface.py
git commit -m "[eval] ModelClient.__init__: resolve unnorm_key, opt-in state passthrough"
```

---

## Phase 5 — `ModelClient.step()` Extension

### Task 9: `step()` pop+convert + extension of wiring tests (§6.3.2 + §8.3 part 2)

**Files:**
- Modify: `examples/LIBERO/eval_files/model2libero_interface.py:76-98`
- Modify: `tests/test_eval_state_passthrough_wiring.py` (extend with step()-related tests)

- [ ] **Step 1: Append the new tests to `tests/test_eval_state_passthrough_wiring.py`**

```python
# ---------------------------------------------------------------------------
# step() pop + convert behavior.
# ---------------------------------------------------------------------------
def _build_example_with_raw_state():
    return {
        "image": [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)],
        "lang": "pick up the bowl",
        "uamvla_raw_state": {
            "ee_pos":        np.array([0.1, 0.0, 0.5], dtype=np.float32),
            "ee_axis_angle": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "joint_pos":     np.zeros(7, dtype=np.float32),
            "gripper_qpos":  np.array([0.02, 0.02], dtype=np.float32),
        },
    }


def test_step_pops_raw_state_and_injects_canonical_state(make_client):
    client = make_client(unnorm_key=None)
    example = _build_example_with_raw_state()
    client.step(example, step=0)

    sent = client.client.last_payload
    assert "examples" in sent
    sent_example = sent["examples"][0]

    # Raw state must be popped (cleanliness: never on the wire).
    assert "uamvla_raw_state" not in sent_example
    # Canonical state must be injected with the expected nested shape.
    assert "canonical_state" in sent_example
    canonical = sent_example["canonical_state"]
    assert canonical["arm_0"]["ee_pose"].shape == (9,)
    assert canonical["arm_0"]["joint_pos"].shape == (7,)
    assert canonical["gripper_0"].shape == (1,)
    # Leaves must be numpy (msgpack-numpy can serialize them).
    assert isinstance(canonical["arm_0"]["ee_pose"], np.ndarray)


def test_step_pops_raw_state_even_when_passthrough_disabled(make_client):
    """If statistics.yaml is absent the gate is False, but step() must still
    pop uamvla_raw_state so it never reaches the wire."""
    client = make_client(unnorm_key=None, with_stats_yaml=False)
    example = _build_example_with_raw_state()
    client.step(example, step=0)

    sent = client.client.last_payload
    sent_example = sent["examples"][0]
    assert "uamvla_raw_state" not in sent_example
    assert "canonical_state" not in sent_example
```

- [ ] **Step 2: Run the new tests; expect failure**

Run: `pytest tests/test_eval_state_passthrough_wiring.py::test_step_pops_raw_state_and_injects_canonical_state tests/test_eval_state_passthrough_wiring.py::test_step_pops_raw_state_even_when_passthrough_disabled -v`
Expected: FAIL — `step()` does not yet pop or inject anything.

- [ ] **Step 3: Patch `step()` in `model2libero_interface.py`**

In `step()`, find the block:

```python
        images = [self._resize_image(image) for image in images]
        example["image"] = images
        vla_input = {
            "examples": [example],
            ...
```

Insert immediately **after** `example["image"] = images` and **before** `vla_input = {...}`:

```python
        # Spec §6.3.2: always pop the UamVLA raw-state key so it never reaches
        # the wire. Conversion to canonical_state is gated on
        # uamvla_state_enabled (see __init__).
        raw = example.pop("uamvla_raw_state", None)
        if self.uamvla_state_enabled and raw is not None:
            canonical = self._adapter.to_canonical(raw)
            canonical = self._state_normalizer(canonical)
            example["canonical_state"] = _to_numpy_leaves(canonical)
```

- [ ] **Step 4: Run wiring tests; expect all to pass**

Run: `pytest tests/test_eval_state_passthrough_wiring.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/LIBERO/eval_files/model2libero_interface.py tests/test_eval_state_passthrough_wiring.py
git commit -m "[eval] ModelClient.step: pop uamvla_raw_state, inject canonical_state"
```

---

## Phase 6 — Eval Driver Wiring

### Task 10: `eval_libero.py` gated raw-state assembly (§6.2)

**Files:**
- Modify: `examples/LIBERO/eval_files/eval_libero.py:140-165`

This is glue code; the wiring test (Task 9) already covers the contract that `example_dict["uamvla_raw_state"]` is consumed properly by ModelClient. There is no easy unit test here without a real LIBERO env, so this task is implementation-only and verified through the §8.4 integration test (Task 12).

- [ ] **Step 1: Locate the `example_dict` block**

Around `examples/LIBERO/eval_files/eval_libero.py:161-165`:

```python
                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                }
```

- [ ] **Step 2: Insert the gated raw-state assembly directly after `example_dict = {...}`**

```python
                # Spec §6.2: when ModelClient is in UamVLA state-passthrough
                # mode, hand it the four raw fields LiberoAdapter expects.
                # robot0_joint_pos is fetched here for the first time (the
                # 8-D `state` concat above doesn't include it). Validated by
                # the P1 probe (tools/probes/probe_libero_obs_keys.py).
                if getattr(client_model, "uamvla_state_enabled", False):
                    example_dict["uamvla_raw_state"] = {
                        "ee_pos":        np.asarray(obs["robot0_eef_pos"],        dtype=np.float32),
                        "ee_axis_angle": _quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
                        "joint_pos":     np.asarray(obs["robot0_joint_pos"],      dtype=np.float32),
                        "gripper_qpos":  np.asarray(obs["robot0_gripper_qpos"],   dtype=np.float32),
                    }
```

- [ ] **Step 3: Sanity-import**

Run: `python -c "import examples.LIBERO.eval_files.eval_libero"`
Expected: exit 0 (import-time syntax/name errors caught).

- [ ] **Step 4: Commit**

```bash
git add examples/LIBERO/eval_files/eval_libero.py
git commit -m "[eval] eval_libero.py: gate uamvla_raw_state assembly on client flag"
```

---

## Phase 7 — Train/Eval Contract Test (§8.2)

### Task 11: Canonical-state alignment test

**Files:**
- Create: `tests/test_train_eval_state_alignment.py`

Both pipelines instantiate the **same** `LiberoAdapter` + `StateNormalizer(mode="q99")` constructors with the **same** `statistics.yaml`. The test runs both on a hand-written raw obs dict and asserts every leaf is `torch.allclose(atol=1e-6)`. It is intentionally somewhat tautological today; its job is to lock the contract so any future divergence in either pipeline trips a clear failure.

- [ ] **Step 1: Write the test**

```python
"""Spec §8.2 — canonical_state produced by the training pipeline must equal
the canonical_state produced by the eval client for the same raw observation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml


def _build_stats_yaml(tmp_path: Path) -> Path:
    """Same fixture content as tests/test_eval_state_passthrough_wiring.py.
    Re-stated locally to keep this test self-contained."""
    stats = {
        "view_names": ["static", "wrist"],
        "max_action_dim": 24,
        "embodiment_stats": {
            "franka_libero": {
                "action_dim": 7,
                "action_min_bound": [-1.0] * 7,
                "action_max_bound": [1.0] * 7,
            }
        },
        "state_stats": {
            "franka_libero": {
                "arm_0.ee_pose": {
                    "q01": [-0.5] * 9, "q99": [0.5] * 9,
                    "min": [-1.0] * 9, "max": [1.0] * 9,
                    "mean": [0.0] * 9, "std": [0.3] * 9,
                },
                "arm_0.joint_pos": {
                    "q01": [-1.0] * 7, "q99": [1.0] * 7,
                    "min": [-2.0] * 7, "max": [2.0] * 7,
                    "mean": [0.0] * 7, "std": [0.5] * 7,
                },
                "gripper_0": {
                    "q01": [0.0], "q99": [1.0],
                    "min": [0.0], "max": [1.0],
                    "mean": [0.5], "std": [0.3],
                },
            }
        },
    }
    path = tmp_path / "statistics.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(stats, f)
    return path


def _hand_raw_obs() -> dict:
    return {
        "ee_pos":        np.array([0.1, 0.0, 0.5], dtype=np.float32),
        "ee_axis_angle": np.array([0.1, -0.2, 0.3], dtype=np.float32),
        "joint_pos":     np.array([0.0, -0.5, 0.0, 1.5, 0.0, 1.0, 0.0], dtype=np.float32),
        "gripper_qpos":  np.array([0.02, 0.02], dtype=np.float32),
    }


def _train_side(stats_yaml: Path, raw: dict) -> dict:
    """Mirror UamVLADataset.__getitem__'s normalization sub-pipeline
    (see uamvla_dataset.py:71-82, 105-106)."""
    from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter
    from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer

    with open(stats_yaml) as f:
        stats_dict = yaml.safe_load(f)
    adapter = LiberoAdapter()
    normalizer = StateNormalizer(stats_dict=stats_dict, embodiment="franka_libero", mode="q99")
    return normalizer(adapter.to_canonical(raw))


def _eval_side(stats_yaml: Path, raw: dict) -> dict:
    """Mirror ModelClient's _adapter / _state_normalizer (see model2libero_interface.py
    state-passthrough block in __init__)."""
    from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter
    from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer

    with open(stats_yaml) as f:
        stats_dict = yaml.safe_load(f)
    adapter = LiberoAdapter()
    normalizer = StateNormalizer(stats_dict=stats_dict, embodiment="franka_libero", mode="q99")
    return normalizer(adapter.to_canonical(raw))


def test_train_and_eval_canonical_state_match(tmp_path: Path):
    stats_yaml = _build_stats_yaml(tmp_path)
    raw = _hand_raw_obs()

    train_out = _train_side(stats_yaml, raw)
    eval_out = _eval_side(stats_yaml, raw)

    assert set(train_out.keys()) == set(eval_out.keys())
    for limb in train_out.keys():
        if isinstance(train_out[limb], dict):
            assert set(train_out[limb].keys()) == set(eval_out[limb].keys())
            for k in train_out[limb]:
                t, e = train_out[limb][k], eval_out[limb][k]
                assert t.shape == e.shape, f"{limb}.{k}: shape mismatch"
                assert t.dtype == e.dtype, f"{limb}.{k}: dtype mismatch"
                assert torch.allclose(t, e, atol=1e-6), f"{limb}.{k}: value mismatch"
        else:
            t, e = train_out[limb], eval_out[limb]
            assert t.shape == e.shape
            assert torch.allclose(t, e, atol=1e-6)
```

- [ ] **Step 2: Run; expect pass**

Run: `pytest tests/test_train_eval_state_alignment.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_train_eval_state_alignment.py
git commit -m "[test] §8.2 train/eval canonical_state alignment contract"
```

---

## Phase 8 — Integration Test (§8.4)

### Task 12: Server boot + single infer integration

This test boots a minimal in-process server-side pipeline and verifies the full flow from `ModelClient.step()` through msgpack-numpy serialization through `stack_canonical` works on numpy leaves. To stay self-contained (no real model), the test stubs the framework's `predict_action` to inspect the received `canonical_state` and return a syntactically valid response.

**Files:**
- Create: `tests/test_eval_state_passthrough_integration.py`

- [ ] **Step 1: Write the integration test**

```python
"""Spec §8.4 — server-side stack_canonical receives the numpy-leaved
canonical_state coming from ModelClient.step() (no real WebSocket; we
exercise the deserialization → stack_canonical path directly)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

pytestmark = pytest.mark.slow


def _stats_yaml(tmp_path):
    stats = {
        "view_names": ["static", "wrist"],
        "max_action_dim": 24,
        "embodiment_stats": {"franka_libero": {
            "action_dim": 7,
            "action_min_bound": [-1.0] * 7,
            "action_max_bound": [1.0] * 7,
        }},
        "state_stats": {"franka_libero": {
            "arm_0.ee_pose":   {"q01": [-0.5] * 9, "q99": [0.5] * 9, "min": [-1.] * 9, "max": [1.] * 9, "mean": [0.] * 9, "std": [0.3] * 9},
            "arm_0.joint_pos": {"q01": [-1.0] * 7, "q99": [1.0] * 7, "min": [-2.] * 7, "max": [2.] * 7, "mean": [0.] * 7, "std": [0.5] * 7},
            "gripper_0":       {"q01": [0.0],      "q99": [1.0],      "min": [0.],     "max": [1.],     "mean": [0.5],    "std": [0.3]},
        }},
    }
    path = tmp_path / "statistics.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(stats, f)
    return path


def test_canonical_state_survives_msgpack_and_stack(tmp_path, monkeypatch):
    """ModelClient.step → msgpack-numpy → unpack → stack_canonical(numpy leaves)."""
    from examples.LIBERO.eval_files import model2libero_interface
    from deployment.model_server.tools import msgpack_numpy
    from starVLA.model.modules.uamvla.collator_helpers import stack_canonical

    # ----- Fake run dir
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "fake.pt").write_bytes(b"")
    _stats_yaml_path = _stats_yaml(run_dir)

    # ----- Stub read_mode_config
    monkeypatch.setattr(
        model2libero_interface, "read_mode_config",
        lambda _p: (
            {"framework": {"action_model": {"future_action_window_size": 7}}},
            {"franka_libero": {"action": {"min": [-1.] * 7, "max": [1.] * 7, "mask": [True] * 6 + [False]}}},
        ),
    )

    # ----- Stub websocket: capture payload, route through msgpack-numpy roundtrip,
    # then exercise stack_canonical on the deserialized canonical_state.
    captured = {}

    class RoundtripStub:
        def predict_action(self, query_info):
            data = msgpack_numpy.packb(query_info)
            payload = msgpack_numpy.unpackb(data)
            captured["payload"] = payload
            example = payload["examples"][0]
            # This is the contract: server-side stack_canonical must accept whatever
            # arrives over the wire. After the §6.4 fix it accepts numpy leaves.
            stacked = stack_canonical([example["canonical_state"]])
            captured["stacked"] = stacked
            return {"data": {"normalized_actions": np.zeros((1, 8, 7), dtype=np.float32)}}

    monkeypatch.setattr(
        model2libero_interface, "WebsocketClientPolicy",
        lambda *a, **kw: RoundtripStub(),
    )

    client = model2libero_interface.ModelClient(
        policy_ckpt_path=run_dir / "checkpoints" / "fake.pt",
        unnorm_key=None,
        action_ensemble=False,
    )
    assert client.uamvla_state_enabled

    example = {
        "image": [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)],
        "lang": "task",
        "uamvla_raw_state": {
            "ee_pos":        np.array([0.1, 0.0, 0.5], dtype=np.float32),
            "ee_axis_angle": np.zeros(3, dtype=np.float32),
            "joint_pos":     np.zeros(7, dtype=np.float32),
            "gripper_qpos":  np.array([0.02, 0.02], dtype=np.float32),
        },
    }
    client.step(example, step=0)

    # Wire-format assertions
    sent_example = captured["payload"]["examples"][0]
    assert isinstance(sent_example["canonical_state"]["arm_0"]["ee_pose"], np.ndarray)
    assert "uamvla_raw_state" not in sent_example

    # Server-side stack assertions
    stacked = captured["stacked"]
    assert isinstance(stacked["arm_0"]["ee_pose"], torch.Tensor)
    assert stacked["arm_0"]["ee_pose"].shape == (1, 9)
    assert stacked["arm_0"]["joint_pos"].shape == (1, 7)
    assert stacked["gripper_0"].shape == (1, 1)
```

- [ ] **Step 2: Run; expect pass**

Run: `pytest tests/test_eval_state_passthrough_integration.py -v`
Expected: PASS (exercises every server-side wire-format invariant in scope).

- [ ] **Step 3: Commit**

```bash
git add tests/test_eval_state_passthrough_integration.py
git commit -m "[test] §8.4 integration: ModelClient → msgpack → stack_canonical roundtrip"
```

---

## Phase 9 — Final Verification

### Task 13: Full test suite + manual probe checklist

- [ ] **Step 1: Run the full `tests/` suite**

Run: `pytest tests/ -x -v --ignore=tests/test_uamvla_predict_action_qwen3vl_generate.py`
Expected: all green except tests requiring optional external assets (which should skip cleanly).

If any pre-existing test fails because of these changes, stop and fix the regression.

- [ ] **Step 2: (libero_env machine only) Run the three probes**

```bash
# P3 import smoke
pytest tests/test_libero_env_imports.py -v

# P2 quat alignment
pytest tests/test_libero_quat_axisangle_alignment.py -v

# P1 obs-keys probe
python tools/probes/probe_libero_obs_keys.py
```

Expected: all PASS / exit 0. If P1 or P2 fails ⇒ design assumption violated; pause and revisit spec §4. If P3 fails ⇒ activate inline-q99 fallback per spec §6.3.1 R3 (separate follow-up).

- [ ] **Step 3: Manual end-to-end smoke (opt-in, requires real ckpt + LIBERO env)**

This step is operator-driven, not a CI gate. Run a few episodes of LIBERO Spatial with a UamVLA ckpt that has `statistics.yaml` shipped, and observe that:
- Server logs `*** UamVLA state passthrough enabled (stats: ...) ***` once at startup.
- A few episode SRs are non-zero (sanity).
- No exceptions related to `canonical_state`, `unnorm_key`, or `joint_pos`.

- [ ] **Step 4: Commit anything left (none expected) and push**

```bash
git status   # confirm clean
git push -u origin starVLA_dev   # only if requested by user
```

Do **not** push without explicit user permission.

---

## Self-Review Notes

**Spec coverage check:**

| Spec section | Task |
|---|---|
| §6.1 trainer copy | Task 5 |
| §6.2 eval driver gating | Task 10 |
| §6.3.1 ModelClient init rewrite | Tasks 7 (red) + 8 (green) |
| §6.3.2 step extension | Task 9 |
| §6.3.3 `_to_numpy_leaves` | Task 6 |
| §6.4 stack_canonical wrap | Task 4 |
| §8.1 P1/P2/P3 probes | Tasks 1 / 2 / 3 + Task 13 step 2 |
| §8.2 train/eval alignment | Task 11 |
| §8.3 fake-client wiring | Tasks 7 + 9 |
| §8.4 integration | Task 12 |
| §8.5 E2E SR (not gating) | Task 13 step 3 (operator-driven) |

**Type / signature consistency:** `uamvla_state_enabled` (public), `_adapter` / `_state_normalizer` (private) introduced in Task 8 are consumed in Tasks 9, 10, 11, 12 with matching names. `_to_numpy_leaves` introduced in Task 6 is consumed in Task 9. `_copy_stats_yaml_to_run_dir` introduced in Task 5.

**Placeholder scan:** No "TODO", "TBD", "implement later". Every step has runnable code or a runnable command.
