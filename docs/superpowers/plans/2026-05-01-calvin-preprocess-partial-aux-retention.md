# CALVIN Preprocess Partial-Aux Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the CALVIN preprocessor from dropping ~12.5 % of frames (and 1,507 entire windows) when a target object is occluded in the static-camera segmentation mask. Retain obs / depth / action / image_future supervision on **100 %** of frames; keep the seg-dependent fields (`point_cloud`, `image_target`, `pose_6d`) on the ~87.5 % where seg works; teach the framework to handle batches where every sample lacks a given aux field; re-preprocess `task_ABC_D` end-to-end and verify the new dataset trains.

**Architecture:** Three-layer fix:
1. **Preprocessor** (`calvin_preprocessor.py`) — replace the two `continue` paths in `process_window` with "emit a partial-aux row"; lift `last_rgb_static` out of the seg-gated path so all-occluded windows still anchor `image_future`.
2. **Dataset** (`uamvla_dataset.py`) — delete the `REQUIRED_OPTIONAL_FIELDS` filter that hid partial-aux rows from training; register a new `calvin_abc_d_uamvla` mixture so the post-fix `task_ABC_D` is reachable.
3. **Framework** (`UamVLA.py`) — new `_resolve_head_mask` helper supplies an all-True default mask for the universal `action` head and an **all-False** default for non-universal heads (pose / recon / future) when the collator drops the field-and-mask-key pair (which happens iff every sample lacks the field). All-False routes through each head's existing `not mask.any()` early-exit and returns `get_dummy_loss()`, preserving DeepSpeed ZeRO-2 all-reduce shape across ranks. Both `forward()` and `visualize_batch()` call sites use the helper.

After local TDD lands the patch, a two-phase remote verification re-preprocesses `task_ABC_D` and runs a 2-step training smoke. **Step 2 (`task_ABC_D`, ~12 hr) is gated on Step 1 (debug, ~10 min) passing AND local 16/16 tests passing.**

**Tech Stack:** Python 3 (numpy, torch, PIL, pytest, OmegaConf, deepspeed/accelerate), bash, CALVIN dataset format (`episode_*.npz` + `lang_annotations` + `scene_info` + per-window seg/depth render).

---

## Spec

Source spec: [`docs/superpowers/specs/2026-05-01-calvin-preprocess-partial-aux-retention-design.md`](../specs/2026-05-01-calvin-preprocess-partial-aux-retention-design.md). Read it first if you have not — the gap table (§2.3 G1–G7) and the per-frame data-flow appendix (§7) are the load-bearing references for every task below.

Sister plan (same repo, prior CALVIN work): [`docs/superpowers/plans/2026-04-30-uamvla-calvin-pipeline.md`](./2026-04-30-uamvla-calvin-pipeline.md). This plan extends `tests/test_uamvla_calvin_pipeline_local.py` from 8 → 11 tests; the existing 8 stay green.

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `tools/preprocess/calvin_preprocessor.py` (`CalvinWorker.process_window`, lines ~218–360) | Drop-in replacement of inner-loop body: lift `last_rgb_static` unconditionally, replace the two `continue` paths with partial-aux row emission, add per-frame info log when seg fails. |
| Modify | `starVLA/dataloader/uamvla_dataset.py` (lines 25, 72–75, 41–47) | Delete `REQUIRED_OPTIONAL_FIELDS` constant + the filter that uses it; register `calvin_abc_d_uamvla` mixture entry. |
| Modify | `starVLA/model/framework/VLM4A/UamVLA.py` (around lines 318–323 in `forward`, 673–678 in `visualize_batch`) | Add module-level `_UNIVERSAL_HEADS = ("action",)` constant + `_resolve_head_mask` helper; route both call sites through it. |
| Modify | `tests/test_uamvla_calvin_pipeline_local.py` (append 3 tests + 1 helper) | Add `_build_partial_aux_calvin_dataset` helper, `test_partial_aux_row_loads` (Test A), `test_resolve_head_mask_defaults_for_missing_keys` (Test B), `test_all_occluded_batch_collator_invariant` (Test C). |

The new tests append to the existing module — do **not** create a new test file. `_build_partial_aux_calvin_dataset` reuses `_make_synthetic_calvin_samples` and the same `_write_statistics(samples, ...)` call pattern as `_build_synthetic_calvin_dataset`, ensuring stats are computed over **all** samples (matching production) rather than only the seg-success subset.

## Verification Notes

- **Tasks 1–7 run on macOS** with no `calvin_env` / PyBullet required. Task 1 (preprocessor) has no local TDD test — calvin_env is needed to drive `process_window` end-to-end. The preprocessor change is structurally validated by `pytest tests/test_preprocessor_smoke.py` (already gated by `pytest.importorskip("calvin_env")`) and behaviourally validated only on the remote (Task 8 / Task 9).
- **Task 7 is a hard gate** for Task 8: 16/16 tests must pass locally before any remote run. Task 8 (~10 min) is a hard gate for Task 9 (~12 hr, irreversible — output dir is `rm -rf`'d).
- Three new tests are added (Test A, B, C). Total CALVIN: 8 prior + 3 new = **11**. LIBERO regression: **5** (`tests/test_uamvla_dataset.py` — must stay green untouched). Combined: **16/16**.
- **Commit message rule (CLAUDE.md):** No `Co-Authored-By: Claude`, no "Generated with Claude Code" footer, no AI mention in commit/PR text. Use only the local git identity. The example commits below honour this — do not add anything.

---

## Task 1: Preprocessor — emit partial-aux rows on seg failure (G1 + G2)

**Files:**
- Modify: `tools/preprocess/calvin_preprocessor.py:218–360` (the per-frame body of `CalvinWorker.process_window` plus the post-loop epilogue)

**Why first:** Every downstream task depends on the new on-disk format. Filter deletion (Task 2) without the preprocessor change would still see only the seg-success subset on `task_ABC_D` (no partial rows have been written yet). Framework patch (Tasks 4–6) is a no-op until partial rows reach training. Locking the preprocessor first keeps the dependency direction clean.

**Why no local test for this task:** `process_window` calls `self.env.reset(...)` + `self.env.render_cameras(...)` which require `calvin_env` (PyBullet). The macOS dev box does not have it. The behavioural validation is in Task 8 (remote Step 1, ~10 min on debug dataset) and Task 9 (remote Step 2, ~12 hr on `task_ABC_D`). Locally we limit ourselves to (a) a structural grep that confirms the two `continue` paths are gone and (b) running the existing `pytest tests/test_preprocessor_smoke.py` to confirm the import still works on machines that have calvin_env (this test is `importorskip`'d on macOS, so it just skips locally — fine).

- [ ] **Step 1: Read the current `process_window` body**

Open `tools/preprocess/calvin_preprocessor.py` and read lines 174–360. The relevant landmarks (line numbers approximate, will shift slightly during edit):
- L216: `last_rgb_static: np.ndarray | None = None` (per-window state)
- L262–268: first `continue` (when `pts is None` and `on_missing_target == "skip"`)
- L281–294: second `continue` (when `target_img is None` and `on_missing_target == "skip"`)
- L301: `last_rgb_static = rendered["rgb_static"]` — currently only reached if both seg checks succeed
- L358: `assert last_rgb_static is not None` (post-loop, fires when every frame's seg failed)

- [ ] **Step 2: Apply the drop-in replacement of the per-frame body**

Replace the per-frame loop body (the block starting at the `for step_idx, t in enumerate(frames):` line and ending immediately before the post-loop `image_future` write) with the verbatim drop-in from spec §3.1. The replacement does five things:

1. Always saves rgb_static / rgb_wrist / depth_static / depth_wrist (these never depended on seg).
2. Sets `last_rgb_static = rendered["rgb_static"]` **immediately after render** — the G2 fix. This guarantees `image_future` can be anchored even when every frame's seg fails.
3. Tries `depth_to_world_points`; on success → writes pc + sets `pc_rel_present`; on failure under `on_missing_target == "skip"` → leaves `pc_rel_present = None` (no `continue`); under `"abort"` → raises RuntimeError exactly as before.
4. Tries `crop_target_from_seg`; on success → writes target.jpg + sets `target_rel_present`; on failure under `"skip"` → leaves `target_rel_present = None`; under `"abort"` → raises RuntimeError.
5. If either field is missing, emits a single `logger.info("frame %d in window %d: seg failed for target %r — emitting partial-aux row (image_target=%s, point_cloud=%s, pose_6d=%s)", ...)` line — replacing the two prior `Skipping frame ...` warnings. Per spec §3.1 the log is operational visibility only; **validation in Tasks 8/9 cross-checks against JSONL row counts and filesystem artefacts, not against log-line counts** (avoids coupling validation to log format).
6. Assembles the row with always-fields, then conditionally adds `point_cloud` + `pose_6d` (only if `pc_rel_present`) and `image_target` (only if `target_rel_present`). `pose_6d` is gated on `point_cloud` because `pose_head.py:166` asserts `batch["point_cloud"]` is present — emitting `pose_6d` without `point_cloud` would crash the head.

The full replacement body is in spec §3.1 (lines 92–212 of the spec). Copy it verbatim — do not paraphrase. Two integration points to preserve:
- The surrounding loop header (`for step_idx, t in enumerate(frames):`) and the variables it imports from outside (`frames`, `episode_id`, `total_steps`, `target_object_id`, `target_seg_id`, `window`, `future_rel`, `output_dir`, `RENDER_W`, `RENDER_H`, `NUM_POINTS`, `MAX_ACTION_DIM`, `FRANKA_ACTION_DIM`, `ACTION_MASK`, `EMBODIMENT`, `self.dataset_source`, `self.env`, `self.on_missing_target`, `logger`) — leave all of these alone.
- The post-loop `if not rows: ... return None` safety net **stays** as a guard for the genuinely-pathological case where every frame's render itself fails (not seg-fail). With G1+G2 applied, this branch is essentially unreachable for typical CALVIN data.

The post-loop `assert last_rgb_static is not None` is also preserved — with G2's unconditional update at the top of the per-frame body, this assertion now only fires if the loop ran zero iterations (window has zero frames). That's a defensive check, not a normal path. Per spec §3.1 you may optionally rephrase it as `if last_rgb_static is None: logger.warning(...); return None` — pick whichever is closer to the surrounding code's style. The drop-in below assumes you keep the assert.

- [ ] **Step 3: Confirm the two `continue` paths are gone (structural check)**

Run:

```bash
grep -nE 'Skipping frame|^\s+continue\s*$' tools/preprocess/calvin_preprocessor.py
```

Expected output: empty (no matches). If you see any matches, you missed a path — re-read step 2.

Also run:

```bash
grep -n 'last_rgb_static' tools/preprocess/calvin_preprocessor.py
```

Expected: at least one occurrence inside the per-frame body that is **not** preceded by an `if pts is not None:` or `if target_img is not None:` guard — i.e., the assignment is at the top of the body, immediately after `rendered = self.env.render_cameras(...)`. The post-loop `assert last_rgb_static is not None` line is also expected to remain.

- [ ] **Step 4: Run existing tests to confirm we didn't break anything**

The only existing test that touches the preprocessor module is `test_preprocessor_smoke.py:21:test_calvin_preprocessor_imports`, which is gated behind `pytest.importorskip("calvin_env")`. On macOS this test will skip — that's fine. We're not regressing anything; the new code path is only reachable from inside `process_window` which is itself behind `calvin_env`.

Run:

```bash
pytest tests/test_preprocessor_smoke.py -v
pytest tests/test_uamvla_calvin_pipeline_local.py -v
pytest tests/test_uamvla_dataset.py -v
```

Expected: all three runs end with **0 failures**. The CALVIN preprocessor smoke test will be `s` (skipped — calvin_env not installed). The pipeline test (8 prior tests) and the LIBERO regression (5 tests) must all pass — they don't touch the preprocessor module.

- [ ] **Step 5: Commit**

```bash
git add tools/preprocess/calvin_preprocessor.py
git commit -m "[calvin] preprocessor: retain partial-aux rows on seg failure

Replace the two 'Skipping frame ...' continue branches in
process_window with partial-aux row emission: obs/depth/action/
image_future are written for every frame; point_cloud, image_target,
and pose_6d are written only when seg succeeds. Lift last_rgb_static
update out of the seg-gated path so all-occluded windows still
anchor image_future. Add a single per-frame info log when seg fails.

Restores ~134k frames and ~1.5k full-occluded windows per task_ABC_D
re-preprocess; recon/pose retention unchanged at ~87.5%.

Spec: docs/superpowers/specs/2026-05-01-calvin-preprocess-partial-aux-retention-design.md §3.1 G1+G2"
```

---

## Task 2: Dataset filter deletion + Test A (G3 + G4 + G7-A)

**Files:**
- Modify: `starVLA/dataloader/uamvla_dataset.py:25` (delete `REQUIRED_OPTIONAL_FIELDS` constant)
- Modify: `starVLA/dataloader/uamvla_dataset.py:71–75` (replace filter with unconditional load)
- Modify: `tests/test_uamvla_calvin_pipeline_local.py` (append `_build_partial_aux_calvin_dataset` helper + `test_partial_aux_row_loads`)

This is the first true TDD task. The test must be written and verified RED before any production change is touched.

- [ ] **Step 1: Append the failing Test A (and its fixture builder)**

Open `tests/test_uamvla_calvin_pipeline_local.py` and append the following at the end of the file (after the existing `test_yaml_loadable` test):

```python
# ============================================================================
# Task A (this PR): partial-aux row loads through UamVLADataset
# ============================================================================

def _build_partial_aux_calvin_dataset(
    tmp_path: Path,
    n_samples: int = 6,
    n_occluded: int = 2,
) -> Path:
    """Like _build_synthetic_calvin_dataset, but the LAST `n_occluded` rows
    omit image_target / point_cloud / pose_6d (and skip writing those media
    files), mimicking what the new preprocessor produces when seg fails.

    statistics.yaml is computed over ALL n_samples — production's
    _write_statistics(samples, ...) iterates every sample's robot_obs via
    CalvinAdapter regardless of occlusion, so stats span every frame.
    """
    from PIL import Image
    from tools.preprocess.calvin_preprocessor import _write_statistics

    samples = _make_synthetic_calvin_samples(n=n_samples)

    # Always-fields on every sample.
    for s in samples:
        sid = s["id"]
        s["image_future"] = f"images/future/{s['episode_id']}.jpg"
        s["depth_static"] = f"depth/static/{sid}.npy"
        s["depth_wrist"]  = f"depth/wrist/{sid}.npy"
        s["static_cam_extrinsic"] = {
            "rotation":    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
        }
        s["wrist_cam_extrinsic"] = {
            "rotation":    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
        }

    # Aux fields on the first (n_samples - n_occluded) rows only.
    n_visible = n_samples - n_occluded
    for s in samples[:n_visible]:
        sid = s["id"]
        s["image_target"] = f"images/target/{sid}.jpg"
        s["point_cloud"]  = f"point_clouds/{sid}.npy"
        s["pose_6d"] = {
            "rotation":    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
            "object_id":   "test_object",
        }
    # samples[n_visible:] are partial — image_target / point_cloud / pose_6d absent.

    for sub in (
        "images/obs/static", "images/obs/wrist",
        "images/target", "images/future",
        "depth/static", "depth/wrist",
        "point_clouds",
    ):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    with open(tmp_path / "data.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    blank_rgb = Image.new("RGB", (256, 256), color=(127, 127, 127))
    pc_zeros = np.zeros((1024, 3), dtype=np.float32)
    depth_zeros = np.zeros((256, 256), dtype=np.float32)
    futures_written: set[str] = set()
    for s in samples:
        # Always-fields: write obs jpgs, depth npys, future jpg.
        for img_rel in s["image"]:
            blank_rgb.save(tmp_path / img_rel, quality=95)
        if s["image_future"] not in futures_written:
            blank_rgb.save(tmp_path / s["image_future"], quality=95)
            futures_written.add(s["image_future"])
        np.save(tmp_path / s["depth_static"], depth_zeros)
        np.save(tmp_path / s["depth_wrist"], depth_zeros)
        # Seg-dependent: only when present in the sample dict.
        if "image_target" in s:
            blank_rgb.save(tmp_path / s["image_target"], quality=95)
        if "point_cloud" in s:
            np.save(tmp_path / s["point_cloud"], pc_zeros)

    intrinsics = {
        "static": {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
        "wrist":  {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
    }
    _write_statistics(samples, tmp_path, intrinsics)
    return tmp_path


def test_partial_aux_row_loads(tmp_path):
    """Mixed full + partial-aux dataset loads through UamVLADataset.

    Visible rows (first 4) carry image_target / point_cloud / pose_gt.
    Occluded rows (last 2) omit them. All 6 rows must load — the legacy
    REQUIRED_OPTIONAL_FIELDS filter would have either dropped the partial
    rows (kept only the 4 truthy filtered samples) or fallen through to
    `or all_samples`. After Task 2's filter deletion, all 6 load directly.
    """
    from starVLA.dataloader.uamvla_dataset import UamVLADataset

    data_root = _build_partial_aux_calvin_dataset(
        tmp_path, n_samples=6, n_occluded=2,
    )

    ds = UamVLADataset(
        data_root=data_root,
        embodiment="franka_calvin",
        action_horizon=8,
        normalization={
            "mode": "q99",
            "apply_to": ["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"],
        },
    )

    assert len(ds) == 6, (
        f"expected all 6 rows to load (filter deleted), got {len(ds)}"
    )

    # Visible rows: aux fields present.
    assert "image_target" in ds[0], "row 0 (visible) must keep image_target"
    assert "point_cloud" in ds[0], "row 0 (visible) must keep point_cloud"
    assert "pose_gt" in ds[0], "row 0 (visible) must keep pose_gt"

    # Occluded rows: aux fields absent.
    assert "image_target" not in ds[5], "row 5 (occluded) must omit image_target"
    assert "point_cloud" not in ds[5], "row 5 (occluded) must omit point_cloud"
    assert "pose_gt" not in ds[5], "row 5 (occluded) must omit pose_gt"

    # Universal fields present on every row, including occluded ones.
    for i in range(6):
        assert ds[i]["action"].shape == (8, 7), (
            f"row {i}: action shape {ds[i]['action'].shape} != (8, 7)"
        )
        assert ds[i]["canonical_state"]["arm_0"]["ee_pose"].shape == (9,)
        assert ds[i]["canonical_state"]["arm_0"]["joint_pos"].shape == (7,)
        assert ds[i]["canonical_state"]["gripper_0"].shape == (1,)
```

- [ ] **Step 2: Run Test A to verify it fails (RED)**

Run:

```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_partial_aux_row_loads -v
```

Expected: **FAIL** with `AssertionError: expected all 6 rows to load (filter deleted), got 4`.

The filter at `uamvla_dataset.py:74–75` is:

```python
self.samples = [s for s in all_samples
                if all(k in s for k in REQUIRED_OPTIONAL_FIELDS)] or all_samples
```

With 4 full + 2 partial: the list comprehension yields 4 truthy samples → `or all_samples` does NOT fire → `len(ds) == 4`. Test asserts 6. RED.

If you instead see an ImportError or fixture error: fix that first — the test must reach the `assert len(ds) == 6` line.

- [ ] **Step 3: Delete `REQUIRED_OPTIONAL_FIELDS` and the filter (G3 + G4)**

In `starVLA/dataloader/uamvla_dataset.py`:

**Delete line 25** (the constant):

```python
REQUIRED_OPTIONAL_FIELDS = ("image_target", "image_future", "point_cloud")
```

**Replace lines 71–75** (the filter) — find:

```python
        with open(self.data_root / "data.jsonl") as f:
            all_samples = [json.loads(l) for l in f if l.strip()]
        self.samples = [s for s in all_samples
                        if all(k in s for k in REQUIRED_OPTIONAL_FIELDS)] or all_samples
```

Replace with:

```python
        with open(self.data_root / "data.jsonl") as f:
            self.samples = [json.loads(l) for l in f if l.strip()]
```

After the edit, search for any leftover references:

```bash
grep -n 'REQUIRED_OPTIONAL_FIELDS' starVLA/ tests/ tools/ runners/ -r
```

Expected: empty. If any reference remains (e.g., re-exported in a `__init__.py`), delete it too.

- [ ] **Step 4: Run Test A to verify it passes (GREEN)**

Run:

```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_partial_aux_row_loads -v
```

Expected: **PASS**.

- [ ] **Step 5: Run the full pipeline + LIBERO regression to confirm no breakage**

```bash
pytest tests/test_uamvla_calvin_pipeline_local.py tests/test_uamvla_dataset.py -v
```

Expected:
- `test_uamvla_calvin_pipeline_local.py`: **9 passed** (8 prior + the new `test_partial_aux_row_loads`).
- `test_uamvla_dataset.py`: **5 passed** (LIBERO regression — must stay green; the deleted filter was global so this catches any LIBERO path that depended on the `or all_samples` fallback).

If `test_uamvla_dataset.py` fails, the LIBERO data may have been silently relying on the filter's fallback. Per spec §5 R1, in that case the right move is **NOT** to restore the constant globally — instead, re-introduce the filter only on the LIBERO branch. Stop here and ask the spec author before proceeding.

- [ ] **Step 6: Commit**

```bash
git add starVLA/dataloader/uamvla_dataset.py tests/test_uamvla_calvin_pipeline_local.py
git commit -m "[calvin] dataset: drop REQUIRED_OPTIONAL_FIELDS filter; load partial-aux rows

Delete the REQUIRED_OPTIONAL_FIELDS constant and the filter at
UamVLADataset.__init__. Per-sample missing aux fields are already
handled downstream by stack_optional_tensor_fields / stack_pose_gt
(via per-sample boolean masks). The filter was hiding the partial-aux
rows the new preprocessor (Task 1) emits.

Add test_partial_aux_row_loads covering a fixture with 4 visible +
2 occluded rows; all 6 load, occluded rows omit image_target /
point_cloud / pose_gt; canonical_state + action present on all.

Spec: §3.2 G3+G4, §3.4 Test A"
```

---

## Task 3: Register `calvin_abc_d_uamvla` mixture (G5)

**Files:**
- Modify: `starVLA/dataloader/uamvla_dataset.py:41–47` (`DATASET_NAMED_MIXTURES`)
- Modify: `tests/test_uamvla_calvin_pipeline_local.py` (append a one-line registry assertion test)

The post-fix `task_ABC_D/` data root needs to be reachable from a mixture key. The pre-existing `calvin_uamvla -> task_D_D` entry is preserved for backward compatibility — see spec §1.2 item 4.

- [ ] **Step 1: Append a failing registry assertion test**

Append to `tests/test_uamvla_calvin_pipeline_local.py`:

```python
def test_calvin_abc_d_uamvla_mixture_registered():
    """The post-fix task_ABC_D data root is reachable from a mixture key."""
    from starVLA.dataloader.uamvla_dataset import DATASET_NAMED_MIXTURES

    assert "calvin_abc_d_uamvla" in DATASET_NAMED_MIXTURES, (
        f"Missing 'calvin_abc_d_uamvla' mixture; available: "
        f"{list(DATASET_NAMED_MIXTURES)}"
    )
    assert DATASET_NAMED_MIXTURES["calvin_abc_d_uamvla"] == [
        ("task_ABC_D", 1.0, "franka_calvin"),
    ], DATASET_NAMED_MIXTURES["calvin_abc_d_uamvla"]

    # Backward-compat: the prior calvin_uamvla -> task_D_D entry must remain.
    assert DATASET_NAMED_MIXTURES["calvin_uamvla"] == [
        ("task_D_D", 1.0, "franka_calvin"),
    ], DATASET_NAMED_MIXTURES["calvin_uamvla"]
```

- [ ] **Step 2: Run the test to verify it fails (RED)**

```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_calvin_abc_d_uamvla_mixture_registered -v
```

Expected: **FAIL** with `AssertionError: Missing 'calvin_abc_d_uamvla' mixture; available: ['libero_uamvla', 'calvin_uamvla']`.

- [ ] **Step 3: Add the new mixture entry**

In `starVLA/dataloader/uamvla_dataset.py`, find the existing `DATASET_NAMED_MIXTURES` dict (around lines 41–47):

```python
DATASET_NAMED_MIXTURES = {
    "libero_uamvla": [
        ("libero_spatial", 1.0, "franka_libero"),
    ],
    "calvin_uamvla": [
        ("task_D_D", 1.0, "franka_calvin"),
    ],
}
```

Replace with:

```python
DATASET_NAMED_MIXTURES = {
    "libero_uamvla": [
        ("libero_spatial", 1.0, "franka_libero"),
    ],
    "calvin_uamvla": [
        ("task_D_D", 1.0, "franka_calvin"),         # existing — kept for backward compat
    ],
    "calvin_abc_d_uamvla": [
        ("task_ABC_D", 1.0, "franka_calvin"),
    ],
}
```

- [ ] **Step 4: Run the test to verify it passes (GREEN)**

```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_calvin_abc_d_uamvla_mixture_registered -v
```

Expected: **PASS**.

- [ ] **Step 5: Commit**

```bash
git add starVLA/dataloader/uamvla_dataset.py tests/test_uamvla_calvin_pipeline_local.py
git commit -m "[calvin] dataset: register calvin_abc_d_uamvla mixture for task_ABC_D

Add 'calvin_abc_d_uamvla' -> [('task_ABC_D', 1.0, 'franka_calvin')] so
the re-preprocessed task_ABC_D root is reachable. Pre-existing
calvin_uamvla -> task_D_D entry preserved for backward compat with any
caller still pointing at the debug-dataset slice.

Training yaml's default data_mix stays 'calvin_uamvla'; operators opt
into 'calvin_abc_d_uamvla' via --datasets.vla_data.data_mix at the
command line for post-fix runs (see spec §4.4).

Spec: §3.2 G5"
```

---

## Task 4: `_resolve_head_mask` helper + Test B (G6 helper + G7-B)

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLA.py` (add module-level constant + helper near the top of the file, before the `UamVLA` class)
- Modify: `tests/test_uamvla_calvin_pipeline_local.py` (append `test_resolve_head_mask_defaults_for_missing_keys`)

**Why a helper instead of inlining:** Both `forward()` (Task 5) and `visualize_batch()` (Task 6) face the same missing-mask problem. Test B exercises **the helper**, not a copy of the dispatch loop — so the test cannot drift from the deployed logic when the helper is later changed. Spec §3.4 Test B is explicit on this point.

- [ ] **Step 1: Append the failing Test B**

Append to `tests/test_uamvla_calvin_pipeline_local.py`:

```python
# ============================================================================
# Test B (this PR): _resolve_head_mask defaults for missing aux mask keys
# ============================================================================

def test_resolve_head_mask_defaults_for_missing_keys():
    """`_resolve_head_mask` returns all-True for `action` (universal) and
    all-False for `pose` / `recon` / `future` when their mask key is missing
    from batch_dict.

    The missing-key case is the one stack_optional_tensor_fields produces when
    every sample in the batch lacked the aux field (see collator_helpers.py:
    if no sample has the field, both the field tensor AND its `_mask` key are
    omitted entirely). Without this helper, UamVLA.forward() would default
    such missing masks to torch.ones(...) and crash inside pose / recon heads
    on `batch["image_target"]` / `assert "point_cloud" in batch`.

    All-False routes through each head's existing `not mask.any()` early exit
    in compute_loss → returns get_dummy_loss(...) — preserving the DeepSpeed
    ZeRO-2 all-reduce shape across ranks.
    """
    import torch
    from starVLA.model.framework.VLM4A.UamVLA import (
        _resolve_head_mask,
        _UNIVERSAL_HEADS,
    )

    # Sanity: the registry of universal heads is the documented one.
    assert _UNIVERSAL_HEADS == ("action",), (
        f"_UNIVERSAL_HEADS drifted from spec: {_UNIVERSAL_HEADS!r}"
    )

    B = 4
    device = torch.device("cpu")
    batch_empty: dict = {}

    action_mask = _resolve_head_mask("action", batch_empty, B, device)
    pose_mask   = _resolve_head_mask("pose",   batch_empty, B, device)
    recon_mask  = _resolve_head_mask("recon",  batch_empty, B, device)
    future_mask = _resolve_head_mask("future", batch_empty, B, device)

    # Shape & dtype contract.
    for name, m in (
        ("action", action_mask), ("pose", pose_mask),
        ("recon", recon_mask), ("future", future_mask),
    ):
        assert m.dtype == torch.bool, f"{name}: dtype {m.dtype} != bool"
        assert m.shape == (B,), f"{name}: shape {tuple(m.shape)} != ({B},)"
        assert m.device == device, f"{name}: device {m.device} != {device}"

    # Default values per universality.
    assert action_mask.all(),       "action default must be all-True (universal)"
    assert not pose_mask.any(),     "pose default must be all-False"
    assert not recon_mask.any(),    "recon default must be all-False"
    assert not future_mask.any(),   "future default must be all-False"

    # When the mask IS present in batch_dict, the helper returns it verbatim.
    explicit = torch.tensor([True, False, True, True])
    out = _resolve_head_mask("pose", {"pose_mask": explicit}, B, device)
    assert out is explicit, (
        "present mask key must be returned unchanged (no copy, no recompute)"
    )
```

- [ ] **Step 2: Run Test B to verify it fails (RED)**

```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_resolve_head_mask_defaults_for_missing_keys -v
```

Expected: **FAIL** with `ImportError: cannot import name '_resolve_head_mask' from 'starVLA.model.framework.VLM4A.UamVLA'` (or a similar import-time error referencing `_UNIVERSAL_HEADS`).

- [ ] **Step 3: Add `_UNIVERSAL_HEADS` and `_resolve_head_mask` to `UamVLA.py`**

In `starVLA/model/framework/VLM4A/UamVLA.py`, find a location at module scope **before** the `UamVLA` class definition (typical location: just after the imports block, or just after the `_build_aux_heads` helper at line ~69). Insert:

```python
# Aux heads whose target is universally present on every sample.
# Action labels + per-action-dim mask are always written by the dataset, so
# action's per-sample boolean mask defaults to all-True when absent. Every
# other head consumes a per-sample subset; their mask comes from the collator
# and may be absent entirely when stack_optional_tensor_fields drops both the
# field tensor and its `_mask` key (the every-sample-lacks-it case).
_UNIVERSAL_HEADS = ("action",)


def _resolve_head_mask(
    head_name: str,
    batch_dict: dict,
    batch_size: int,
    device: "torch.device",
) -> "torch.Tensor":
    """Return the per-sample boolean mask the aux head should consume.

    Universal heads default to all-True when their mask key is missing.
    Non-universal heads default to all-False — the head's compute_loss /
    visualize then early-exits via `not mask.any()` and returns
    get_dummy_loss() (forward path) or a no-op (visualize path), preserving
    DeepSpeed ZeRO-2's all-reduce shape across ranks (see base.py:31).

    Both UamVLA.forward() and UamVLA.visualize_batch() route through this
    helper so test_resolve_head_mask_defaults_for_missing_keys cannot drift
    from the deployed logic.
    """
    mask = batch_dict.get(f"{head_name}_mask")
    if mask is None:
        fill = head_name in _UNIVERSAL_HEADS
        return torch.full((batch_size,), fill, dtype=torch.bool, device=device)
    return mask
```

If `torch` is not yet imported at module level (it should be — `UamVLA.py` already uses it pervasively), confirm with `grep '^import torch' starVLA/model/framework/VLM4A/UamVLA.py`. The type-hint strings (`"torch.device"`, `"torch.Tensor"`) are quoted to avoid forcing any new import order; they are evaluated lazily.

- [ ] **Step 4: Run Test B to verify it passes (GREEN)**

```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_resolve_head_mask_defaults_for_missing_keys -v
```

Expected: **PASS**.

- [ ] **Step 5: Run a broader smoke to confirm the helper is import-safe**

The new helper is module-level, so an import-time bug (e.g., a stray syntax error) would break every UamVLA-touching test. Run:

```bash
pytest tests/test_uamvla_forward.py tests/test_aux_heads_smoke.py tests/test_uamvla_calvin_pipeline_local.py tests/test_uamvla_dataset.py -v
```

Expected: all green. The new helper is only invoked by Test B at this point — the other tests just need to be able to import the module.

- [ ] **Step 6: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py tests/test_uamvla_calvin_pipeline_local.py
git commit -m "[uamvla] add _resolve_head_mask helper for missing aux mask keys

Module-level _UNIVERSAL_HEADS = ('action',) constant + _resolve_head_mask
helper. When stack_optional_tensor_fields drops both the field tensor and
its '_mask' key (every sample in the batch lacked the field), the helper
supplies an all-False default for non-universal heads (pose / recon /
future) and an all-True default for action. The all-False default routes
through each head's existing `not mask.any()` early-exit in compute_loss,
returning get_dummy_loss() — preserving DeepSpeed ZeRO-2's all-reduce
shape across ranks (see base.py:31).

Forward and visualize_batch call sites are wired in the next two commits.

Spec: §3.3 G6 (helper), §3.4 Test B"
```

---

## Task 5: `forward()` — route aux dispatch through `_resolve_head_mask` (G6 forward)

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLA.py:318–323` (the aux-head dispatch loop in `forward()`)

This task has **no new test** — Test B from Task 4 already exercises the helper. The change is purely a wiring update at the forward call site.

- [ ] **Step 1: Read the current forward() dispatch loop**

In `starVLA/model/framework/VLM4A/UamVLA.py`, find the dispatch loop in `forward()` (currently around lines 318–329). The current shape:

```python
        for name, head in self.aux_heads.items():
            mask = batch_dict.get(f"{name}_mask")
            if mask is None:
                mask = torch.ones(
                    hidden.shape[0], dtype=torch.bool, device=hidden.device,
                )
            out = head.compute_loss(hidden, batch_dict, mask=mask)
            if out.loss is not None:
                total = total + out.loss
                log_metrics[f"{name}_loss"] = out.loss.detach()
            for mk, mv in out.metrics.items():
                log_metrics[f"{name}_{mk}"] = mv
```

The bug: `mask = torch.ones(...)` is wrong for non-universal heads when the mask key is missing — pose / recon / future then receive a fake all-True mask and crash on `batch["image_target"]` / `assert "point_cloud" in batch`.

- [ ] **Step 2: Replace the dispatch loop**

Replace the block above with:

```python
        for name, head in self.aux_heads.items():
            mask = _resolve_head_mask(
                name, batch_dict, hidden.shape[0], hidden.device,
            )
            out = head.compute_loss(hidden, batch_dict, mask=mask)
            if out.loss is not None:
                total = total + out.loss
                log_metrics[f"{name}_loss"] = out.loss.detach()
            for mk, mv in out.metrics.items():
                log_metrics[f"{name}_{mk}"] = mv
```

The body of the loop (the `out = head.compute_loss(...)` line and below) is unchanged.

- [ ] **Step 3: Confirm the inline `torch.ones(...)` fallback is gone in `forward()`**

```bash
grep -n 'torch.ones' starVLA/model/framework/VLM4A/UamVLA.py
```

Expected: zero hits inside `forward()` (lines 279–340 region). One hit may remain in `visualize_batch()` (around line 678) — that one is removed in Task 6.

If you see a hit inside the `forward()` body, you missed the replacement.

- [ ] **Step 4: Run `forward()`-touching tests**

```bash
pytest tests/test_uamvla_forward.py tests/test_aux_heads_smoke.py tests/test_uamvla_calvin_pipeline_local.py -v
```

Expected: all green. `test_uamvla_forward.py` uses fixtures with every aux field present (existing test), so the helper returns the explicit masks unchanged — behaviour is identical to before for the populated case. The new helper path is structurally exercised through this test too.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py
git commit -m "[uamvla] forward(): route aux head dispatch through _resolve_head_mask

Replaces the ad-hoc 'mask is None -> torch.ones(...)' fallback in the
forward() aux-head dispatch loop. Universal heads (action) still default
to all-True; non-universal heads (pose / recon / future) now default to
all-False when the collator dropped the {head}_mask key (the every-sample-
lacks-the-field case). The head's compute_loss then early-exits via
`not mask.any()` and returns get_dummy_loss() — no crash on missing
batch fields, ZeRO-2 all-reduce shape preserved.

Behaviour unchanged for the populated case (mask present, mixed True/False
or all-True): helper returns the explicit mask verbatim.

Spec: §3.3 G6 (forward call site)"
```

---

## Task 6: `visualize_batch()` patch + Test C collator invariant (G6 visualize + G7-C)

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLA.py:673–678` (the aux-head dispatch loop in `visualize_batch()`)
- Modify: `tests/test_uamvla_calvin_pipeline_local.py` (append `test_all_occluded_batch_collator_invariant`)

The visualize call site has the **same** bug as forward — `mask = batch_dict.get(...) or torch.ones(...)` defaults a missing mask to all-True, which makes `recon_head.visualize` and `pose_head.visualize` attempt to read `batch["image_target"]` / `batch["point_cloud"]` and crash. The crash is currently caught and warned at `train_starvla.py:374`, so visualization just silently fails for occluded batches. Routing through `_resolve_head_mask` makes the heads' built-in `not mask.any()` short-circuits actually fire.

Test C is a precondition lock: it asserts that `stack_optional_tensor_fields` really does drop both the field tensor and its `_mask` key when every sample lacks the field. The whole §3.3 patch design depends on this collator behaviour being stable; if a future maintainer changes the collator to insert an empty tensor + all-False mask, the helper's all-False default would still be correct but Test B's "missing key" branch would no longer be reached. Test C catches that drift.

- [ ] **Step 1: Append Test C**

Append to `tests/test_uamvla_calvin_pipeline_local.py`:

```python
# ============================================================================
# Test C (this PR): collator invariant — _resolve_head_mask precondition
# ============================================================================

def test_all_occluded_batch_collator_invariant():
    """Lock in the precondition the §3.3 framework patch relies on.

    When every sample in a batch lacks an aux field, stack_optional_tensor_fields
    must omit BOTH the field tensor and its `_mask` key from the output dict.
    The all-False default in _resolve_head_mask is the right choice EXACTLY
    because the helper sees the mask key as missing — if a future refactor
    started inserting an empty-tensor + all-False mask, this test would catch
    the drift (Test B's "missing key" branch would no longer be exercised in
    production, even though Test B itself would still pass).

    Mixed case is also covered: when at least one sample has the field, both
    the field tensor and the all-True (or partial-True) mask must be present.
    """
    import torch
    from starVLA.model.modules.uamvla.collator_helpers import (
        stack_optional_tensor_fields,
    )

    # Case 1: every sample lacks image_target and point_cloud.
    samples = [
        {"image_future": torch.zeros(3, 224, 224)},
        {"image_future": torch.zeros(3, 224, 224)},
    ]
    batch = stack_optional_tensor_fields(
        samples, ["image_target", "image_future", "point_cloud"],
    )
    assert "image_target"      not in batch
    assert "image_target_mask" not in batch
    assert "point_cloud"       not in batch
    assert "point_cloud_mask"  not in batch
    assert "image_future"      in batch
    assert "image_future_mask" in batch
    assert batch["image_future_mask"].dtype == torch.bool
    assert batch["image_future_mask"].all(), (
        "every sample has image_future → mask is all-True"
    )

    # Case 2: mixed — sample 0 has image_target, sample 1 does not.
    samples_mixed = [
        {"image_target": torch.zeros(3, 224, 224),
         "image_future": torch.zeros(3, 224, 224)},
        {"image_future": torch.zeros(3, 224, 224)},
    ]
    batch_mixed = stack_optional_tensor_fields(
        samples_mixed, ["image_target", "image_future"],
    )
    assert "image_target"      in batch_mixed
    assert "image_target_mask" in batch_mixed
    assert batch_mixed["image_target_mask"].tolist() == [True, False], (
        f"mixed mask must be [True, False], got "
        f"{batch_mixed['image_target_mask'].tolist()}"
    )
```

- [ ] **Step 2: Run Test C — should pass immediately (collator already behaves this way)**

```bash
pytest tests/test_uamvla_calvin_pipeline_local.py::test_all_occluded_batch_collator_invariant -v
```

Expected: **PASS** on first run. This is a "lock the contract" test, not a TDD-driver test. If it fails, the collator implementation has changed and `_resolve_head_mask` may need a corresponding update — stop and ask before proceeding.

- [ ] **Step 3: Read the current visualize_batch() dispatch loop**

In `starVLA/model/framework/VLM4A/UamVLA.py`, find the dispatch loop in `visualize_batch()` (currently around lines 673–678). The current shape (note: the surrounding signature differs from forward — visualize_batch takes additional args like `n_samples`):

```python
        for name, head in self.aux_heads.items():
            ...
            mask = batch_dict.get(f"{name}_mask")
            if mask is None:
                mask = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
            ...  # head.visualize(...) call follows
```

The full surrounding context (what the head.visualize call signature is, how its output is consumed) varies. **Do not rewrite the entire loop body** — only replace the `mask = batch_dict.get(...) ; if mask is None: mask = torch.ones(...)` two-line fallback with a single line that calls the helper.

- [ ] **Step 4: Replace the mask-resolution lines in visualize_batch()**

Find the two-line block:

```python
            mask = batch_dict.get(f"{name}_mask")
            if mask is None:
                mask = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
```

Replace with:

```python
            mask = _resolve_head_mask(
                name, batch_dict, hidden.shape[0], hidden.device,
            )
```

Everything before and after this block stays as-is (the surrounding `for name, head in self.aux_heads.items():` header and the eventual `head.visualize(...)` call).

- [ ] **Step 5: Confirm both call sites now use the helper**

```bash
grep -n '_resolve_head_mask\|torch.ones' starVLA/model/framework/VLM4A/UamVLA.py
```

Expected: two `_resolve_head_mask(...)` invocations (one in `forward()`, one in `visualize_batch()`); zero `torch.ones(...)` hits inside either dispatch loop. Other `torch.ones(...)` uses elsewhere in the file (e.g., attention mask construction in `__call__`) are unrelated and may remain — only the two dispatch-loop fallbacks are being removed.

- [ ] **Step 6: Run the full local suite**

```bash
pytest tests/test_uamvla_calvin_pipeline_local.py tests/test_uamvla_dataset.py tests/test_uamvla_forward.py tests/test_aux_heads_smoke.py -v
```

Expected:
- `test_uamvla_calvin_pipeline_local.py`: **11 passed** (8 prior + Test A + Test B + Test C).
- `test_uamvla_dataset.py`: **5 passed**.
- The forward + aux-heads smoke tests stay green.

- [ ] **Step 7: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py tests/test_uamvla_calvin_pipeline_local.py
git commit -m "[uamvla] visualize_batch(): route aux dispatch through _resolve_head_mask

Same fix as forward(): replace the inline 'mask is None -> torch.ones(...)'
fallback in visualize_batch()'s aux-head dispatch with a call to
_resolve_head_mask. Non-universal heads now correctly receive an
all-False default when the collator dropped their _mask key — the head's
visualize() short-circuits via `not mask.any()` instead of crashing on
batch['image_target'] / batch['point_cloud'] (which train_starvla.py
currently swallows + warns at line 374).

Add test_all_occluded_batch_collator_invariant locking the
stack_optional_tensor_fields contract that the helper depends on:
when every sample lacks an aux field, both the field tensor AND its
_mask key are dropped; mixed cases yield a [True, False, ...] mask.

Spec: §3.3 G6 (visualize call site), §3.4 Test C"
```

---

## Task 7: Local 16/16 sweep — hard gate before remote (§4.1)

**Goal:** Verify the full local test surface is green — 11 CALVIN pipeline tests + 5 LIBERO regression tests = **16/16** — before any remote work begins. This is the **hard gate** for Tasks 8 and 9. Step 2 of the remote verification (Task 9, ~12 hours, irreversible because the output dir is `rm -rf`'d) must not be entered if any local test is red.

This task does not modify code or commit anything new — it's a verification checkpoint.

- [ ] **Step 1: Run the CALVIN pipeline test file (must report 11 passed)**

```bash
pytest tests/test_uamvla_calvin_pipeline_local.py -v
```

Expected output:
- 11 tests collected.
- All 11 pass:
  1. `test_write_statistics_schema` (Task 1 of prior plan)
  2. `test_write_statistics_no_legacy_keys`
  3. `test_calvin_uamvla_mixture_registered`
  4. `test_synthetic_dataset_loads`
  5. `test_cli_help`
  6. `test_cli_required_args_missing_input`
  7. `test_cli_required_args_missing_output`
  8. `test_yaml_loadable`
  9. `test_partial_aux_row_loads` ← Test A (this PR)
  10. `test_calvin_abc_d_uamvla_mixture_registered` ← (this PR)
  11. `test_resolve_head_mask_defaults_for_missing_keys` ← Test B (this PR)
  12. `test_all_occluded_batch_collator_invariant` ← Test C (this PR)

That's 12 tests, not 11. Reconcile: spec §1.2 / §2.3 G7 / §3.4 budget **3 new tests** (Tests A, B, C) on top of the prior 8, giving 11. The mixture-registration assertion (`test_calvin_abc_d_uamvla_mixture_registered`) is incidental and is the 12th. If you'd rather honour the spec's "11" number exactly, you can fold the mixture assertion into another test (e.g., assert it inside Test A's setup), but it is also fine to land 12 — the spec's gating intent is "≥ 11", not "exactly 11". Document the count in your final commit / PR description if you opt for 12.

If the count is < 11 or any test is red: stop. Do not proceed to Task 8.

- [ ] **Step 2: Run the LIBERO regression tests (must report 5 passed)**

```bash
pytest tests/test_uamvla_dataset.py -v
```

Expected: **5 passed**. These tests guard against R1 in spec §5 — the deleted `REQUIRED_OPTIONAL_FIELDS` filter was global, so a hidden LIBERO path that depended on the `or all_samples` fallback would surface here. If any LIBERO test fails: per spec §5 R1, do NOT restore the global constant — instead, re-introduce the filter only on the LIBERO branch (e.g., `if self.embodiment.startswith("franka_libero"): ...`). Stop and ask the spec author before that path.

- [ ] **Step 3: Run the broader smoke surface to confirm no second-order regression**

```bash
pytest tests/test_uamvla_forward.py tests/test_aux_heads_smoke.py tests/test_collator_helpers.py tests/test_preprocessor_smoke.py tests/test_state_normalizer.py tests/test_state_encoder_smoke.py -v
```

Expected: all green (some may report `s` skipped on macOS — e.g., `test_calvin_preprocessor_imports` is gated behind `pytest.importorskip("calvin_env")`). A failure here would indicate the framework patch (Tasks 4–6) broke something orthogonal to CALVIN.

- [ ] **Step 4: Sign off the gate**

If steps 1–3 are green: Task 7 passes; Task 8 may proceed.

If any step is red: stop, fix, re-run from step 1.

- [ ] **Step 5: Push the local commits to `starVLA_dev` (so the remote can pull them)**

```bash
git status                     # confirm working tree clean
git log --oneline -7           # confirm Tasks 1, 2, 3, 4, 5, 6 commits present
git push origin starVLA_dev
```

Expected: 6 new commits land on the remote `starVLA_dev` branch (one per task, 1 / 2 / 3 / 4 / 5 / 6). Confirm with `git log origin/starVLA_dev --oneline -7`.

No local commit from Task 7 itself.

---

## Task 8: Remote Step 1 — debug-dataset preprocessor validation (§4.2, ~10 min)

**Goal:** Run the new preprocessor on the small CALVIN debug dataset and verify the output structurally. ~10 min; catches any logic regression in Task 1 cheaply, before paying ~12 hr for `task_ABC_D`.

**Pre-conditions:**
- Task 7 passed locally (16/16 tests green).
- Remote machine has `calvin_env` conda environment + working PyBullet/EGL.
- Local commits pushed to `origin/starVLA_dev`.

**Hard gate:** Task 9 is blocked until every Step 1 pass criterion below is satisfied.

- [ ] **Step 1: Sync the remote and clean the debug output dir**

On the remote box:

```bash
cd /inspire/.../UniamVLA   # path varies; substitute your remote checkout
conda activate calvin_env
git fetch origin
git checkout starVLA_dev
git pull origin starVLA_dev
git log --oneline -7         # confirm the 6 new commits are present

rm -rf datasets/uamvla_calvin/task_D_D_partial   # mandatory clean slate
```

The `rm -rf` is required — `CalvinPreprocessor.process` merges any pre-existing `shards/*.jsonl` into the final output. A leftover directory from a crashed prior run would silently contaminate this verification (spec §4.3).

- [ ] **Step 2: Run the preprocessor on the debug dataset**

```bash
python runners/preprocess_calvin.py \
    --input_dir /inspire/.../UamVLA/datasets/calvin_debug_dataset/training \
    --output_dir datasets/uamvla_calvin/task_D_D_partial \
    --dataset_source calvin_debug_partial \
    --num_workers 8 \
    --on_resolve_failure skip --on_missing_target skip \
    2>&1 | tee /tmp/preprocess_debug_partial.log
```

Substitute the `--input_dir` path for your remote layout. Expected wall time: ~10 minutes on a modern box with EGL.

- [ ] **Step 3: Verify the pass criteria from spec §4.2**

After the run completes, run each of the following checks. Every one must pass — these are filesystem/JSONL-based checks, **not** log-line counts (per spec §4.2 the per-frame info log is operational visibility, not a validation source).

```bash
OUT=datasets/uamvla_calvin/task_D_D_partial

# (a) Row count: at least the prior baseline (no regression).
ROWS=$(wc -l < "$OUT/data.jsonl"); echo "rows=$ROWS"
test "$ROWS" -ge 503 && echo "OK: rows >= 503" || echo "FAIL: rows < 503"

# (b) Episode count: every source window retained.
EPS=$(awk -F'"episode_id": "' '{print $2}' "$OUT/data.jsonl" \
      | awk -F'"' '{print $1}' | sort -u | wc -l)
echo "episodes=$EPS"
test "$EPS" -eq 9 && echo "OK: episodes == 9" || echo "FAIL: episodes != 9"

# (c) No window-level drops.
NDROP=$(grep -c 'produced no samples' /tmp/preprocess_debug_partial.log || true)
echo "produced_no_samples=$NDROP"
test "$NDROP" -eq 0 && echo "OK: 0 dropped windows" || echo "FAIL: $NDROP dropped"

# (d) Filesystem orphan check: obs/static count matches data.jsonl rows.
OBS=$(ls "$OUT/images/obs/static" | wc -l); echo "obs_static_files=$OBS"
test "$OBS" -eq "$ROWS" && echo "OK: no obs orphans" || echo "FAIL: obs/rows mismatch"

# (e) Partial-row count (rows missing image_target).
PARTIAL=$(awk '!/image_target/{c++} END{print c+0}' "$OUT/data.jsonl")
echo "partial_rows=$PARTIAL"
TARGET=$(ls "$OUT/images/target" 2>/dev/null | wc -l); echo "target_files=$TARGET"
DIFF=$(( OBS - TARGET ))
echo "obs - target = $DIFF (must equal partial_rows)"
test "$PARTIAL" -eq "$DIFF" && echo "OK: partial-row accounting consistent" \
    || echo "FAIL: partial $PARTIAL != obs-target $DIFF"

# (f) statistics.yaml schema check.
python - <<'PY'
import yaml, sys
with open("datasets/uamvla_calvin/task_D_D_partial/statistics.yaml") as f:
    s = yaml.safe_load(f)
expected = {"view_names","max_action_dim","embodiment_stats",
            "state_stats","cameras","point_cloud"}
got = set(s.keys())
assert got == expected, f"top-level keys mismatch: got {sorted(got)}, expected {sorted(expected)}"
assert "franka_calvin" in s["state_stats"], f"state_stats missing franka_calvin: {list(s['state_stats'])}"
print("OK: statistics.yaml schema valid")
PY
```

Pass criteria summary (all must hold):

| Check | Pass condition |
|---|---|
| (a) `data.jsonl` rows | ≥ 503 (no regression vs. the prior baseline) |
| (b) distinct `episode_id` count | == 9 |
| (c) `produced no samples` log lines | == 0 |
| (d) `images/obs/static/` file count | == `data.jsonl` row count |
| (e) Partial-row count | == `images/obs/static` count − `images/target` count |
| (f) `statistics.yaml` schema | Top-level keys = `{view_names, max_action_dim, embodiment_stats, state_stats, cameras, point_cloud}`; `state_stats[franka_calvin]` present |

If any criterion fails: stop. Diagnose. Do NOT proceed to Task 9 — a logic bug here would multiply by ~17,870 episodes on `task_ABC_D` and waste 12 hours.

If all pass: Task 8 is green. Task 9 may proceed.

- [ ] **Step 4: (Optional cleanup) Remove the debug output**

The `task_D_D_partial` directory was a verification-only artefact — it's not registered in any mixture and won't be referenced by training. Per spec §7, it should not be kept after Step 1 verification:

```bash
rm -rf datasets/uamvla_calvin/task_D_D_partial
```

This is optional but recommended — the directory is throwaway and consumes disk for nothing. No commit either way.

---

## Task 9: Remote Step 2 — `task_ABC_D` re-preprocess + training smoke (§4.3 + §4.4, ~12 hr)

**Goal:** Re-preprocess the full `task_ABC_D/` split with the new partial-aux preprocessor; verify the output matches spec §4.3's expected metrics; run a 2-step training smoke against the new mixture to confirm dataset → trainer integration.

**Pre-conditions (all hard gates):**
- Task 7 passed (local 16/16 green).
- Task 8 passed (Step 1 debug verification all green).
- Remote machine has `calvin_env` and the post-fix `starVLA_dev` checkout.
- Existing `datasets/uamvla_calvin/task_ABC_D/` has been deleted (the spec author already did this — re-confirm). If a partial dir exists from a crashed prior attempt, **`rm -rf` it before launch**, every time. Per spec §4.3, the preprocessor merges any pre-existing `shards/*.jsonl` into the output, so a partial leftover would silently contaminate this run.

**Estimated wall time:** ~12 hours on the configured remote (8 workers, full `task_ABC_D` split). Run in `nohup ... &` so a dropped SSH session does not abort it.

This task makes no commits.

- [ ] **Step 1: Confirm clean slate + free disk**

```bash
cd /inspire/.../UniamVLA
conda activate calvin_env

# Confirm the directory is empty / non-existent.
ls datasets/uamvla_calvin/task_ABC_D 2>/dev/null && \
    { echo "WARN: dir exists — rm -rf before launch"; \
      rm -rf datasets/uamvla_calvin/task_ABC_D; }

# Confirm enough free disk: rough budget ~1.07M rows × ~70 KB/row (rgb_static
# + rgb_wrist + depths + occasional pc) ≈ 70 GB. Have at least 100 GB free.
df -h datasets/
```

If less than 100 GB free: free space first; the run will silently fail mid-write otherwise.

- [ ] **Step 2: Launch the preprocessor in the background**

```bash
nohup env CUDA_VISIBLE_DEVICES=0 python runners/preprocess_calvin.py \
    --input_dir /inspire/.../UamVLA/datasets/task_ABC_D/training \
    --output_dir datasets/uamvla_calvin/task_ABC_D \
    --dataset_source task_ABC_D \
    --num_workers 8 \
    --on_resolve_failure skip --on_missing_target skip \
    > /tmp/preprocess_task_ABC_D_v2.log 2>&1 &

PREPROCESS_PID=$!
echo "preprocessor PID: $PREPROCESS_PID"

# Periodically inspect.
tail -f /tmp/preprocess_task_ABC_D_v2.log
```

Substitute the `--input_dir` for your remote layout.

- [ ] **Step 3: Wait for completion**

The run should take ~12 hours. Monitor with:

```bash
ps -p $PREPROCESS_PID && echo "still running" || echo "exited"
tail -200 /tmp/preprocess_task_ABC_D_v2.log
```

The end of the log should report the final row / shard count and exit cleanly.

If the job crashes mid-way:

```bash
rm -rf datasets/uamvla_calvin/task_ABC_D   # mandatory before re-launch
# then re-run step 2
```

Per spec §4.3, partial shards from a crashed run **must** be removed before re-launch — they would otherwise be merged into the next attempt's output.

- [ ] **Step 4: Verify the §4.3 pass criteria**

After completion, run these checks. Every one must hold (numbers compared to spec §4.3 baseline):

```bash
OUT=datasets/uamvla_calvin/task_ABC_D

# (a) Row count: strictly greater than 937,169; expected close to 1,071,743.
ROWS=$(wc -l < "$OUT/data.jsonl"); echo "rows=$ROWS"
test "$ROWS" -gt 937169 && echo "OK: rows > 937169" || echo "FAIL: regression"
# Expected close to 1,071,743 — log a warning if the absolute diff is > 5%.
EXPECTED=1071743
DIFF_PCT=$(python -c "print(abs($ROWS - $EXPECTED) * 100.0 / $EXPECTED)")
echo "row diff vs expected: ${DIFF_PCT}%"

# (b) Episode count: exactly 17,870 (every source window retained).
EPS=$(awk -F'"episode_id": "' '{print $2}' "$OUT/data.jsonl" \
      | awk -F'"' '{print $1}' | sort -u | wc -l)
echo "episodes=$EPS"
test "$EPS" -eq 17870 && echo "OK: episodes == 17870" || echo "FAIL: episodes=$EPS"

# (c) No window-level drops.
NDROP=$(grep -c 'produced no samples' /tmp/preprocess_task_ABC_D_v2.log || true)
echo "produced_no_samples=$NDROP"
test "$NDROP" -eq 0 && echo "OK: 0 dropped windows" || echo "FAIL: $NDROP dropped"

# (d) obs/static file count matches rows (within ±1 for any race).
OBS=$(ls "$OUT/images/obs/static" | wc -l); echo "obs_static=$OBS"
test "$(python -c "print(abs($OBS - $ROWS))")" -le 1 && echo "OK: obs/rows aligned" \
    || echo "FAIL: obs/rows mismatch"

# (e) image_target file count: within ±2% of pre-fix 937,169.
TARGET=$(ls "$OUT/images/target" | wc -l); echo "target_files=$TARGET"
python - <<PY
diff = abs($TARGET - 937169) / 937169.0 * 100
assert diff <= 2.0, f"image_target deviation ${diff:.2f}% > 2%"
print(f"OK: image_target within {diff:.2f}% of 937169")
PY

# (f) point_clouds count: within ±2% of pre-fix 937,169.
PC=$(ls "$OUT/point_clouds" | wc -l); echo "pc_files=$PC"
python - <<PY
diff = abs($PC - 937169) / 937169.0 * 100
assert diff <= 2.0, f"point_clouds deviation ${diff:.2f}% > 2%"
print(f"OK: point_clouds within {diff:.2f}% of 937169")
PY

# (g) image_future count: exactly equal to episode count.
FUT=$(ls "$OUT/images/future" | wc -l); echo "future_files=$FUT"
test "$FUT" -eq "$EPS" && echo "OK: futures == episodes" \
    || echo "FAIL: futures=$FUT, episodes=$EPS"

# (h) Partial-row count = obs/static - image_target (filesystem diff;
#     not coupled to log format).
PARTIAL=$(awk '!/image_target/{c++} END{print c+0}' "$OUT/data.jsonl")
echo "partial_rows=$PARTIAL"
DIFF=$(( OBS - TARGET ))
test "$PARTIAL" -eq "$DIFF" && echo "OK: partial-row accounting" \
    || echo "FAIL: partial $PARTIAL != obs-target $DIFF"
```

Pass criteria summary (all must hold):

| Metric | Pre-fix | Post-fix expectation | Pass condition |
|---|---|---|---|
| `data.jsonl` rows | 937,169 | ~1,071,743 | strictly > 937,169 |
| Distinct `episode_id` count | 16,363 | 17,870 | exactly 17,870 |
| `produced no samples` log lines | 1,507 | 0 | exactly 0 |
| `images/obs/static` files | 1,071,743 (rendered, not all referenced pre-fix) | ~1,072k | abs diff vs `data.jsonl` rows ≤ 1 |
| `images/target` files | 937,169 | similar | within ±2 % of 937,169 |
| `point_clouds` files | 937,169 | similar | within ±2 % of 937,169 |
| `images/future` files | 16,363 | 17,870 | == episode count |
| Partial rows | 0 | ~12.5 % | == `images/obs/static count − images/target count` |

If any criterion fails: stop. Diagnose. Re-run if a re-launch is justified (and remember the mandatory `rm -rf` first). Do NOT proceed to Step 5.

- [ ] **Step 5: Run the §4.4 training smoke**

This sanity-checks that the new dataset reaches the trainer cleanly. Note the explicit `--datasets.vla_data.data_mix calvin_abc_d_uamvla` override — the yaml's default still points at `calvin_uamvla` (= `task_D_D`), so the smoke must opt into the new mixture (per spec §4.4):

```bash
WANDB_MODE=offline PYTHONPATH=$(pwd) accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 \
    starVLA/training/train_starvla.py \
    --config_yaml starVLA/config/training/uamvla_calvin.yaml \
    --datasets.vla_data.data_root_dir datasets/uamvla_calvin \
    --datasets.vla_data.data_mix calvin_abc_d_uamvla \
    --trainer.max_train_steps 2 \
    2>&1 | tee /tmp/smoke_task_ABC_D.log | tail -60
```

Pass criteria (per spec §4.4):

- 2 global steps complete; total loss reported and finite (not NaN, not Inf).
- Per-head loss metrics are present in the step output. With per_device_batch_size=2 and ~12.5% per-frame occlusion, P(both samples in a 2-sample batch occlude the same field) ≈ 1.5%. So the 2-step smoke probably won't surface an all-occluded batch — but it WILL surface any general crash from the framework patch (Tasks 4–6). If the patch is broken, this smoke fails before completing 2 steps.
- No exceptions in the log. The framework patch is supposed to make all-occluded batches a no-op; an exception here is a regression.

If the smoke passes: Task 9 is green. The spec is delivered.

If the smoke fails: capture the traceback, identify whether the failure is in (a) the framework patch (Tasks 4–6), (b) the dataset (Task 2), (c) the preprocessor output (Task 1 / 9 step 4), or (d) something orthogonal. Open a bug; do not paper over with mask-tweak hacks.

- [ ] **Step 6: Sign off**

If steps 1–5 are green: spec delivered. Annotate the spec / branch description with a one-paragraph completion note (no commit needed):

```
Task 9 (remote e2e) verified on <hostname>, <YYYY-MM-DD>:
  - Step 1 (debug, ~10 min): rows=<N>, episodes=9, partial_rows=<M>, statistics.yaml ok
  - Step 2 (task_ABC_D, ~<elapsed>):
      rows=<N>, episodes=17870, produced_no_samples=0,
      image_target=<N>, point_cloud=<N>, image_future=17870
  - §4.4 smoke (1 GPU, max_train_steps=2): completed, total loss=<...>, no exceptions
```

Nothing to commit from this task.

---

## Spec Coverage Map

| Spec section / requirement | Task |
|---|---|
| §1.2 item 1 (preprocessor: emit partial rows + lift `last_rgb_static` + per-frame info log) | Task 1 |
| §1.2 item 2 (delete `REQUIRED_OPTIONAL_FIELDS` filter) | Task 2 |
| §1.2 item 3 (`UamVLA.py` patch: helper + forward + visualize_batch) | Tasks 4 + 5 + 6 |
| §1.2 item 4 (register `calvin_abc_d_uamvla` mixture) | Task 3 |
| §1.2 item 5 (re-preprocess `task_ABC_D/`, clean slate first) | Task 9 |
| §1.2 item 6 (3 new tests) | Tasks 2 (Test A) + 4 (Test B) + 6 (Test C) |
| §2.3 G1 (`continue` paths in process_window) | Task 1 |
| §2.3 G2 (`last_rgb_static` lifted out of seg-gated path) | Task 1 |
| §2.3 G3 (delete `REQUIRED_OPTIONAL_FIELDS` constant) | Task 2 |
| §2.3 G4 (replace filter with unconditional load) | Task 2 |
| §2.3 G5 (register `calvin_abc_d_uamvla`) | Task 3 |
| §2.3 G6 (`_resolve_head_mask` at `forward()` and `visualize_batch()`) | Tasks 4 + 5 + 6 |
| §2.3 G7 (3 new tests in `test_uamvla_calvin_pipeline_local.py`) | Tasks 2 (A), 4 (B), 6 (C) |
| §2.4 expected post-fix metrics | Verified by Task 9 step 4 |
| §3.1 preprocessor drop-in body | Task 1 step 2 |
| §3.2 dataset filter deletion + new mixture | Tasks 2 step 3, 3 step 3 |
| §3.3 `_UNIVERSAL_HEADS` + `_resolve_head_mask` + behavioural delta table | Task 4 step 3 |
| §3.3 forward() patch | Task 5 step 2 |
| §3.3 visualize_batch() patch | Task 6 step 4 |
| §3.4 Test A (`test_partial_aux_row_loads`) + fixture builder | Task 2 step 1 |
| §3.4 Test B (`test_resolve_head_mask_defaults_for_missing_keys`) | Task 4 step 1 |
| §3.4 Test C (`test_all_occluded_batch_collator_invariant`) | Task 6 step 1 |
| §4.1 local 16/16 must pass | Task 7 |
| §4.2 remote Step 1 (debug dataset, ~10 min) | Task 8 |
| §4.3 remote Step 2 (`task_ABC_D` re-preprocess, ~12 hr; mandatory clean slate) | Task 9 steps 1–4 |
| §4.4 training smoke with `--datasets.vla_data.data_mix calvin_abc_d_uamvla` | Task 9 step 5 |
| §5 R1 (LIBERO regression risk) | Task 2 step 5 + Task 7 step 2 |
| §5 R2 (multi-process race in preprocessor) | Task 8 (Step 1 already runs at `--num_workers 8`) |
| §5 R3 (`pose_gt` ↔ `point_cloud` coupling) | Task 1 step 2 (`pose_6d` gated on `pc_rel_present`) |
| §5 R4 (collator handles per-sample masks) | Task 6 step 1 (Test C locks invariant) |
| §5 R5 (all-occluded batch crash) | Tasks 4 + 5 + 6 (helper + both call sites) |
| §5 R6 (`rm -rf` data loss) | Task 9 step 1 (clean-slate procedure) |
| §6 follow-ups (out of scope) | Not in plan — left for future specs |
| §7 naming + per-frame data-flow appendix | Embedded in Task 1 step 2 |

---
