# CALVIN Preprocess Partial-Aux Retention — Design Spec

**Date**: 2026-05-01
**Branch**: starVLA_dev
**Status**: Draft
**Spec author follow-up to**: [`2026-04-30-uamvla-calvin-pipeline-design.md`](./2026-04-30-uamvla-calvin-pipeline-design.md)

## 1. Goal & Scope

### 1.1 Goal

Stop the CALVIN preprocessor from dropping entire frames when the target object is occluded in the static-camera segmentation mask. Retain the obs / depth / action / image_future supervision (which never depended on seg) so the main backbone, the action head, and the future head can train on **100%** of rendered frames, while the seg-dependent fields (`point_cloud`, `image_target`, and the `pose_6d` that feeds pose head's required input) remain on the ~87.5% of frames where seg works.

### 1.2 Scope

**In scope:**

1. Modify `tools/preprocess/calvin_preprocessor.py:CalvinWorker.process_window` so seg failures do not skip the frame; instead, emit a row that omits the seg-dependent fields. Lift `last_rgb_static` updates out of the seg-gated path so windows whose every frame is occluded are no longer dropped. Add a per-frame info log when seg fails (replacing the now-irrelevant "Skipping frame" log) so operators retain live visibility.
2. Delete `REQUIRED_OPTIONAL_FIELDS` and the corresponding pre-load filter in `starVLA/dataloader/uamvla_dataset.py`, deferring per-sample aux-field handling to the framework's `stack_optional_tensor_fields` / `stack_pose_gt` mask machinery.
3. **Patch `starVLA/model/framework/VLM4A/UamVLA.py:forward()`** to handle the case where an aux field is absent from every sample in the batch (the collator omits both the field tensor *and* its `_mask` key in that case — see `collator_helpers.py:93-94`). The current code defaults missing masks to all-True, which crashes pose / recon heads that assert `batch["point_cloud"]` / read `batch["image_target"]` unconditionally. Convention: action head defaults to all-True (action is universal); other heads skip the batch when their per-sample mask is missing or all-False.
4. **Register a new mixture** `calvin_abc_d_uamvla -> [("task_ABC_D", 1.0, "franka_calvin")]` in `starVLA/dataloader/uamvla_dataset.py`. The pre-existing `calvin_uamvla -> task_D_D` entry is left intact for backward compatibility with any caller pointing at that path.
5. Re-preprocess `datasets/uamvla_calvin/task_ABC_D/` in place (overwrite — the prior output has been deleted by the spec author; if a re-run is ever needed, the procedure starts with an explicit `rm -rf` so old shards do not get merged in).
6. Add **three** new local unit tests: (a) `test_partial_aux_row_loads` — covers §3.1 + §3.2 (dataset side); (b) `test_resolve_head_mask_defaults_for_missing_keys` — covers §3.3 framework patch directly; (c) `test_all_occluded_batch_collator_invariant` — covers the collator precondition the patch relies on. After this work: **11 CALVIN tests + 5 LIBERO tests = 16/16 must pass**.

**Out of scope:**

- LIBERO preprocessor — no rendering changes, no re-preprocess of `datasets/uamvla_libero/`. Filter-relaxation behaviour is verified to be a no-op for LIBERO via the regression tests.
- Re-preprocess of `task_D_D` (the prior debug-dataset slice) — too small to matter; real `task_D_D` data is a separate future task.
- Training yaml / launcher / aux-head loss-weight tuning — the mask infrastructure already handles partial samples; loss-weight retuning is observed-and-react.
- Optional "action-only" training mode — over-engineering; per-sample masks already disable a head for samples that lack its targets.
- Filter-with-configurable-fallback (option C from brainstorming) — go straight to deletion; revisit only if regression tests fail.
- Orphan-cleanup utility — old orphans are gone with the `rm -rf`; new preprocessor doesn't produce orphans.

## 2. Current State & Gaps

### 2.1 Measured baseline (task_ABC_D under the previous preprocessor)

| Metric | Value |
|---|---|
| Source lang windows | 17,870 |
| Output episodes | 16,363 |
| Output rows in `data.jsonl` | 937,169 |
| Frame retention | 87.5 % |
| Frames dropped (`Skipping frame ... target ... not visible`) | 134,574 |
| Windows silently dropped (`produced no samples; dropping window`) | 1,507 |
| Orphan obs / depth files left on disk | 134,574 (~70 GB) |
| Frames in episodes shorter than 20 (p1 = 7) | 666 episodes |

The frame loss is systematically biased — the ten task labels with the worst retention are all "pick / lift / grasp pink block in cabinet / slider / shelf"; the five with the least loss are all "open / pull / slide drawer / door". The bias is a calvin_env rendering property of enclosed-space targets, not a preprocessor bug, but the action / future / main-backbone supervision **does not** need to suffer the same loss.

### 2.2 Per-head supervision dependencies

| Head | Required input field | Required label field | Depends on seg? |
|---|---|---|---|
| **Action** | `image[obs]`, `lang`, `state` | `action` | ❌ |
| **Future** | hidden_states | `image_future` (per-episode last frame) | ❌ |
| **Recon** | hidden_states + `image_target` | `image_target` | ✅ |
| **Pose** | hidden_states + `point_cloud` | `pose_gt` | ✅ |

Pose head asserts `batch['point_cloud']` is present at `pose_head.py:166`. Even if `pose_gt` is provided on a seg-failed frame, pose head cannot consume it — the input feature is missing. We therefore couple `pose_gt` to `point_cloud`: emit `pose_6d` only when seg succeeded.

### 2.3 Identified gaps

| ID | Location | Current behaviour | Required behaviour |
|---|---|---|---|
| G1 | `calvin_preprocessor.py:251–297` (`process_window` seg checks) | `pts is None` or `target_img is None` → `continue`; the obs / depth files written earlier in the same iteration are leaked. | Don't `continue`. Emit a row that omits the failed fields; obs / depth files written are now legitimate references. |
| G2 | `calvin_preprocessor.py:303` (`last_rgb_static = rendered["rgb_static"]`) | Updated only after the target_img check passes. When **every** frame in a window fails seg, `last_rgb_static` stays `None`, the post-loop `assert last_rgb_static is not None` triggers, and `if not rows: ... return None` drops the entire window. | Update unconditionally at the top of the per-frame body, immediately after rendering. With G1 also applied, the window now produces rows even when every frame is occluded, and `image_future` is written. |
| G3 | `uamvla_dataset.py:25` `REQUIRED_OPTIONAL_FIELDS = ("image_target", "image_future", "point_cloud")` | Used by the filter at line 58 to gate samples. | Delete the constant. |
| G4 | `uamvla_dataset.py:55–59` (the filter) | `[s for s in all_samples if all(k in s for k in REQUIRED_OPTIONAL_FIELDS)] or all_samples` | Replace with the unfiltered load (`self.samples = all_samples`). |
| G5 | `uamvla_dataset.py:DATASET_NAMED_MIXTURES` | Only `calvin_uamvla -> task_D_D` is registered. Re-preprocessed `task_ABC_D/` is unreachable from any mixture. | Add `calvin_abc_d_uamvla -> [("task_ABC_D", 1.0, "franka_calvin")]`. |
| G6 | `UamVLA.py:319–323` (forward()'s aux-head dispatch) | `mask = batch_dict.get(f"{name}_mask")` falls back to `torch.ones(...)` (all-True) when missing. But when every sample lacks an aux field, the collator drops both the field and its mask key (see `collator_helpers.py:93-94`); pose/recon heads then crash on `batch["image_target"]` / `assert "point_cloud" in batch`. | Action head: missing mask → all-True (correct, action is universal). Pose / recon / future: missing or all-False mask → skip the head this batch (don't call `compute_loss`). |
| G7 | `tests/test_uamvla_calvin_pipeline_local.py` | All 8 existing tests build fixtures with every aux field present. None covers a "partial-aux row" path or an "all-occluded batch" path. | Add `test_partial_aux_row_loads` (dataset-side) and `test_all_occluded_batch_skips_aux_head` (framework-side, see §3.4). |

### 2.4 Expected post-fix metrics

| Metric | Pre-fix | Post-fix (predicted) |
|---|---|---|
| Episodes | 16,363 | **17,870** (every source window retained) |
| Rows in `data.jsonl` | 937,169 | **~1,072k** (= 937,169 prior rows + 134,574 previously-skipped frames; the 134,574 number already includes the frames inside the 1,507 fully-occluded windows that were silently dropped — those windows contributed roughly 1,507 × 64 ≈ 96k of the 134k skipped frames) |
| Action / Future / main-backbone frame retention | 87.5 % | **100 %** |
| Recon / Pose frame retention | 87.5 % | 87.5 % (unchanged — correct semantics) |
| Orphan files on disk | 134,574 | 0 |

## 3. Detailed Changes

### 3.1 Preprocessor — `tools/preprocess/calvin_preprocessor.py`

**Function**: `CalvinWorker.process_window` (the per-frame body of the loop starting at line ~218).

**Change shape** (drop-in replacement of the inner-loop body, keeping the surrounding loop structure):

```python
for step_idx, t in enumerate(frames):
    npz = np.load(input_dir / f"episode_{t:07d}.npz")
    rel_actions = np.asarray(npz["rel_actions"], dtype=np.float32)
    robot_obs   = np.asarray(npz["robot_obs"],   dtype=np.float32)
    scene_obs   = np.asarray(npz["scene_obs"],   dtype=np.float32)

    self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    rendered = self.env.render_cameras(width=RENDER_W, height=RENDER_H)
    obj_pos, obj_mat = self.env.get_object_pose(target_object_id)

    sample_id = f"{episode_id}_step{step_idx:04d}"

    # --- Always-write fields (no seg dependency) ---
    static_rel = f"images/obs/static/{sample_id}.jpg"
    wrist_rel  = f"images/obs/wrist/{sample_id}.jpg"
    Image.fromarray(rendered["rgb_static"]).save(output_dir / static_rel, quality=95)
    Image.fromarray(rendered["rgb_wrist"]).save(output_dir / wrist_rel, quality=95)

    depth_static_rel = f"depth/static/{sample_id}.npy"
    depth_wrist_rel  = f"depth/wrist/{sample_id}.npy"
    np.save(output_dir / depth_static_rel, rendered["depth_static"].astype(np.float32))
    np.save(output_dir / depth_wrist_rel,  rendered["depth_wrist"].astype(np.float32))

    # G2 fix: anchor image_future regardless of seg outcome.
    last_rgb_static = rendered["rgb_static"]

    # --- Seg-dependent fields: try, omit on failure ---
    pc_rel_present:     str | None = None
    target_rel_present: str | None = None

    pts = depth_to_world_points(
        depth=rendered["depth_static"],
        seg_mask=rendered["seg_static"],
        intrinsic=rendered["static_intrinsic"],
        cam_R=rendered["static_cam_R"],
        cam_t=rendered["static_cam_t"],
        target_id=target_seg_id,
        num_points=NUM_POINTS,
    )
    if pts is not None:
        pc_rel = f"point_clouds/{sample_id}.npy"
        np.save(output_dir / pc_rel, pts)
        pc_rel_present = pc_rel
    elif self.on_missing_target == "abort":
        raise RuntimeError(
            f"Target {target_object_id!r} not visible in frame {t} "
            f"(window {window.window_idx})"
        )
    # else (skip mode): leave pc_rel_present = None

    target_img = crop_target_from_seg(
        rendered["rgb_static"], rendered["seg_static"], target_id=target_seg_id,
    )
    if target_img is not None:
        target_rel = f"images/target/{sample_id}.jpg"
        target_img.save(output_dir / target_rel, quality=95)
        target_rel_present = target_rel
    elif self.on_missing_target == "abort":
        raise RuntimeError(
            f"Seg crop failed for frame {t} (window {window.window_idx})"
        )

    # Replace the old "Skipping frame ..." log with a wording that reflects
    # the new behaviour (we now retain the frame, just drop the seg-dependent
    # fields). Operators see one line per partial-aux frame; verification in
    # §4.2/§4.3 cross-checks this against JSONL counts but does not require
    # log-count to match (avoids coupling validation to log format).
    if pc_rel_present is None or target_rel_present is None:
        logger.info(
            "frame %d in window %d: seg failed for target %r — emitting "
            "partial-aux row (image_target=%s, point_cloud=%s, pose_6d=%s)",
            t, window.window_idx, target_object_id,
            "kept" if target_rel_present else "omitted",
            "kept" if pc_rel_present     else "omitted",
            "kept" if pc_rel_present     else "omitted",  # pose_6d gated on pc
        )

    # --- Assemble row (always-fields + present aux-fields) ---
    action_7d  = rel_actions.tolist()
    action_24d = action_7d + [0.0] * (MAX_ACTION_DIM - FRANKA_ACTION_DIM)
    row = {
        "id":           sample_id,
        "episode_id":   episode_id,
        "step_idx":     step_idx,
        "total_steps":  total_steps,
        "image":        [static_rel, wrist_rel],
        "instruction":  window.instruction,
        "embodiment":   EMBODIMENT,
        "action_dim":   FRANKA_ACTION_DIM,
        "action":       action_24d,
        "action_mask":  ACTION_MASK,
        "robot_obs":    robot_obs.tolist(),
        "dataset_source": self.dataset_source,
        "image_future": future_rel,
        "depth_static": depth_static_rel,
        "depth_wrist":  depth_wrist_rel,
        "static_cam_extrinsic": {
            "rotation":    np.asarray(rendered["static_cam_R"]).flatten().tolist(),
            "translation": np.asarray(rendered["static_cam_t"]).tolist(),
        },
        "wrist_cam_extrinsic": {
            "rotation":    np.asarray(rendered["wrist_cam_R"]).flatten().tolist(),
            "translation": np.asarray(rendered["wrist_cam_t"]).tolist(),
        },
    }

    # Seg-dependent. pose_6d is gated on point_cloud because pose head's
    # forward asserts batch["point_cloud"] is present (pose_head.py:166).
    if pc_rel_present is not None:
        row["point_cloud"] = pc_rel_present
        row["pose_6d"] = {
            "rotation":    mat_to_6d(np.asarray(obj_mat)),
            "translation": np.asarray(obj_pos).tolist(),
            "object_id":   target_object_id,
        }
    if target_rel_present is not None:
        row["image_target"] = target_rel_present

    rows.append(row)
```

**Removed semantics:**
- The two `continue` paths inside the seg checks.
- The post-loop `assert last_rgb_static is not None` is no longer reachable in the normal-occlusion path; keep it (or rephrase as a defensive `if last_rgb_static is None: ...` log + return None) for the genuinely-pathological case where every frame's render itself fails.

**Preserved semantics:**
- The `if not rows: ... return None` post-loop drop remains as a safety net for cases where every render fails (not seg-fail). With the new logic this is essentially unreachable for typical CALVIN data.
- `--on_missing_target abort` continues to raise RuntimeError exactly as before — only `--on_missing_target skip` (the recommended mode for production) sees the new behaviour.

### 3.2 Dataset filter + new mixture — `starVLA/dataloader/uamvla_dataset.py`

**Delete the filter:**

```python
# Delete (line 25):
# REQUIRED_OPTIONAL_FIELDS = ("image_target", "image_future", "point_cloud")
```

```python
# Replace lines 55–59:
with open(self.data_root / "data.jsonl") as f:
    all_samples = [json.loads(l) for l in f if l.strip()]
self.samples = [s for s in all_samples
                if all(k in s for k in REQUIRED_OPTIONAL_FIELDS)] or all_samples

# With:
with open(self.data_root / "data.jsonl") as f:
    self.samples = [json.loads(l) for l in f if l.strip()]
```

The framework's `stack_optional_tensor_fields(samples, ["image_target", "image_future", ...])` and `stack_pose_gt(samples)` produce per-sample masks for partially-present fields. Per-sample handling is done by aux heads via these masks (see `pose_head.py:185–186`, `future_head.py:112`, `recon_head.py:121`). The all-batch-occluded edge case is handled in §3.4 below.

**Add the new mixture:**

```python
# In DATASET_NAMED_MIXTURES (around line 28-37), add:
DATASET_NAMED_MIXTURES = {
    "libero_uamvla": [
        ("libero_spatial", 1.0, "franka_libero"),
    ],
    "calvin_uamvla": [
        ("task_D_D", 1.0, "franka_calvin"),       # existing — kept for backward compat
    ],
    "calvin_abc_d_uamvla": [                      # NEW
        ("task_ABC_D", 1.0, "franka_calvin"),
    ],
}
```

The new mixture is referenced by `--datasets.vla_data.data_mix calvin_abc_d_uamvla` at training time (training yaml's default stays `calvin_uamvla` to preserve current behaviour for any caller relying on it; the operator overrides at command line for the post-fix `task_ABC_D` runs — see §4.4).

### 3.3 Framework patch — `starVLA/model/framework/VLM4A/UamVLA.py`

**File**: `starVLA/model/framework/VLM4A/UamVLA.py`, the aux-head dispatch loop in `forward()` (around line 318-329).

**Current behaviour** (lines 319–323): `mask = batch_dict.get(f"{name}_mask")` falls back to `torch.ones(...)` (all-True) when missing. But when every sample in a batch lacks a particular aux field, `stack_optional_tensor_fields` drops *both* the field tensor and its `_mask` key from `batch_dict` (see `collator_helpers.py:92-94`). The all-True fallback then sends a now-fake mask into `head.compute_loss`, which crashes on the first `batch["image_target"]` / `assert "point_cloud" in batch` reference.

**Why we don't `continue`**: each aux head's `compute_loss` already early-exits via `get_dummy_loss(...)` when `not mask.any()` (see `recon_head.py:110`, `pose_head.py:159`, `future_head.py` similarly). The dummy loss returns a zero tensor wired into the optimizer graph — its docstring (`base.py:31`) is explicit that it exists for **DeepSpeed ZeRO-2 alignment**: every rank must contribute a tensor of the right shape to the all-reduce, even when this rank has no real samples for the head. If we `continue` and skip the head entirely, ranks that DO have samples will all-reduce a tensor that ranks-without-samples never produced, and the optimizer hangs / desyncs.

**Correct fix**: keep dispatching to every head every step; just provide a **valid default mask** (all-False for non-universal heads, all-True for action). Each head's internal `not mask.any()` early-exit then returns `get_dummy_loss(...)` — preserving the all-reduce shape across ranks.

**Replacement**:

```python
# Module-level constant: "action" is universally present (every sample has an
# action label + per-action-dim mask). All other heads operate on a subset of
# samples; their per-sample boolean mask comes from the collator and may be
# absent entirely when every sample in the batch lacked the aux field.
_UNIVERSAL_HEADS = ("action",)


def _resolve_head_mask(
    head_name: str, batch_dict: dict, batch_size: int, device: torch.device,
) -> torch.Tensor:
    """Return the per-sample boolean mask the aux head should consume.

    Universal heads default to all-True when their mask is absent. Non-universal
    heads default to all-False — the head's compute_loss then early-exits via
    get_dummy_loss(), preserving DeepSpeed ZeRO-2's all-reduce shape across
    ranks (see base.py:31 / get_dummy_loss docstring).

    Centralising this in a helper lets §3.4 Test B exercise the actual logic
    used by forward(), instead of duplicating it inside the test.
    """
    mask = batch_dict.get(f"{head_name}_mask")
    if mask is None:
        fill = head_name in _UNIVERSAL_HEADS  # True for action, False for others
        return torch.full((batch_size,), fill, dtype=torch.bool, device=device)
    return mask
```

```python
# Updated dispatch loop in forward() (replaces lines 318-329 in the current file):
for name, head in self.aux_heads.items():
    mask = _resolve_head_mask(name, batch_dict, hidden.shape[0], hidden.device)
    out = head.compute_loss(hidden, batch_dict, mask=mask)
    if out.loss is not None:
        total = total + out.loss
        log_metrics[f"{name}_loss"] = out.loss.detach()
    for mk, mv in out.metrics.items():
        log_metrics[f"{name}_{mk}"] = mv
```

**Behavioural delta:**

| Mask state | Before | After |
|---|---|---|
| Missing key, `action` head | `mask = ones(B)`, head runs | `mask = ones(B)`, head runs (unchanged) |
| Missing key, pose/recon/future | `mask = ones(B)`, head crashes on `batch[field]` | `mask = zeros(B)`, head's `not mask.any()` early-exit returns `get_dummy_loss(...)` — no crash, ZeRO-2 all-reduce shape preserved |
| Present, all-False | head runs, internally returns dummy loss | Same |
| Present, mixed | head processes True subset | Same |
| Present, all-True | head runs | Same |

The `_UNIVERSAL_HEADS` tuple is a module-level constant. If a future head is added that's universal (e.g., a "language modelling loss" on every sample), extend the tuple. Naming-by-string is fragile but better than per-head class attributes for a 4-head registry — revisit if the registry grows.

### 3.4 Tests — `tests/test_uamvla_calvin_pipeline_local.py`

**Test A — partial-aux row loads (dataset side):**

```python
def _build_partial_aux_calvin_dataset(tmp_path: Path, n_samples: int = 6, n_occluded: int = 2) -> Path:
    """Like _build_synthetic_calvin_dataset, but the last `n_occluded` rows
    omit image_target / point_cloud / pose_6d to mimic seg-failed frames
    under the new preprocessor.

    statistics.yaml is computed over ALL n_samples (matching production —
    `_write_statistics(samples, ...)` iterates every sample's `robot_obs`
    via CalvinAdapter regardless of occlusion state).
    """
    # ... build full samples via _make_synthetic_calvin_samples(n=n_samples) ...
    # ... for the last n_occluded rows, drop image_target/point_cloud/pose_6d keys
    #     (and skip writing those media files) ...
    # ... write all rows to data.jsonl ...
    # ... call _write_statistics(all_n_samples, tmp_path, intrinsics) — matches
    #     production. CalvinAdapter only reads robot_obs (always present) so
    #     stats include occluded samples like production does ...

def test_partial_aux_row_loads(tmp_path):
    """Mixed full/partial-aux dataset loads via UamVLADataset; missing fields
    are absent from per-sample dicts; core fields universally present."""
    from starVLA.dataloader.uamvla_dataset import UamVLADataset

    data_root = _build_partial_aux_calvin_dataset(tmp_path, n_samples=6, n_occluded=2)
    ds = UamVLADataset(
        data_root=data_root, embodiment="franka_calvin", action_horizon=8,
        normalization={"mode": "q99",
                       "apply_to": ["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"]},
    )

    assert len(ds) == 6, "all 6 rows must load — filter is gone"
    # Visible rows have aux fields:
    assert "image_target" in ds[0]
    assert "point_cloud" in ds[0]
    assert "pose_gt" in ds[0]
    # Occluded rows do not:
    assert "image_target" not in ds[5]
    assert "point_cloud" not in ds[5]
    assert "pose_gt" not in ds[5]
    # Universal fields present on all:
    for i in range(6):
        assert ds[i]["action"].shape == (8, 7)
        assert ds[i]["canonical_state"]["arm_0"]["ee_pose"].shape == (9,)
```

**Test B — `_resolve_head_mask` defaults are correct (framework side):**

This test exercises the **same `_resolve_head_mask` helper that `UamVLA.forward()` calls**, not a copy of the dispatch loop. If a future maintainer changes the helper without touching the test, or vice versa, the test fails — which is what we want.

```python
def test_resolve_head_mask_defaults_for_missing_keys(tmp_path):
    """`_resolve_head_mask` returns all-True for `action` (universal) and
    all-False for `pose` / `recon` / `future` when their mask key is missing
    from batch_dict (the case where every sample in the batch lacked the aux
    field — see collator_helpers.py:92-94). Verifies the §3.3 patch.

    Calling this helper from BOTH forward() and the test ensures the test
    can never diverge from the deployed logic.
    """
    import torch
    from starVLA.model.framework.VLM4A.UamVLA import (
        _resolve_head_mask, _UNIVERSAL_HEADS,
    )

    # Sanity: action is the only universal head in the current registry.
    assert _UNIVERSAL_HEADS == ("action",)

    B = 4
    device = torch.device("cpu")

    # Empty batch_dict: every aux mask key is missing.
    batch_empty: dict = {}
    action_mask = _resolve_head_mask("action", batch_empty, B, device)
    pose_mask   = _resolve_head_mask("pose",   batch_empty, B, device)
    recon_mask  = _resolve_head_mask("recon",  batch_empty, B, device)
    future_mask = _resolve_head_mask("future", batch_empty, B, device)

    assert action_mask.dtype == torch.bool and action_mask.shape == (B,)
    assert action_mask.all(),     "action default = all-True (universal)"
    assert not pose_mask.any(),   "pose default = all-False (non-universal)"
    assert not recon_mask.any(),  "recon default = all-False (non-universal)"
    assert not future_mask.any(), "future default = all-False (non-universal)"

    # When the mask key IS present, the helper returns it verbatim.
    explicit_mask = torch.tensor([True, False, True, True])
    batch_with = {"pose_mask": explicit_mask}
    out = _resolve_head_mask("pose", batch_with, B, device)
    assert out is explicit_mask, "present mask key is returned unchanged"


def test_all_occluded_batch_collator_invariant(tmp_path):
    """Sanity-check the precondition the §3.3 patch relies on:
    when every sample in a batch lacks an aux field, stack_optional_tensor_fields
    omits both the field tensor and its `_mask` key. The patch's all-False
    default kicks in exactly because of this collator behaviour.
    """
    import torch
    from starVLA.model.modules.uamvla.collator_helpers import stack_optional_tensor_fields

    # 2 samples, both lacking image_target / point_cloud, but with image_future
    samples = [
        {"image_future": torch.zeros(3, 224, 224)},
        {"image_future": torch.zeros(3, 224, 224)},
    ]
    batch = stack_optional_tensor_fields(
        samples, ["image_target", "image_future", "point_cloud"],
    )
    assert "image_target" not in batch and "image_target_mask" not in batch
    assert "point_cloud"  not in batch and "point_cloud_mask"  not in batch
    assert "image_future" in batch     and "image_future_mask" in batch
    # Both samples present → mask is all-True
    assert batch["image_future_mask"].all()
```

**Existing tests** (8 CALVIN + 5 LIBERO) must stay green with no modification. They use fixtures where every sample has every aux field, so neither filter deletion nor framework patch changes their behaviour.

## 4. Verification & Re-preprocess Procedure

### 4.1 Local unit verification (macOS, no `calvin_env`)

```bash
pytest tests/test_uamvla_calvin_pipeline_local.py -v   # 11 tests (8 prior + 3 new)
pytest tests/test_uamvla_dataset.py -v                  # 5 LIBERO regression
```

Required: **16/16 pass**.

### 4.2 Remote — Step 1: validate on debug dataset

Always run before Step 2. The debug dataset is small (~10 minutes), so a logic regression is caught before the multi-hour `task_ABC_D` run.

```bash
cd /inspire/.../UniamVLA && conda activate calvin_env
git pull origin starVLA_dev   # pull the new preprocessor
rm -rf datasets/uamvla_calvin/task_D_D_partial   # clean slate

python runners/preprocess_calvin.py \
    --input_dir /inspire/.../UamVLA/datasets/calvin_debug_dataset/training \
    --output_dir datasets/uamvla_calvin/task_D_D_partial \
    --dataset_source calvin_debug_partial \
    --num_workers 8 \
    --on_resolve_failure skip --on_missing_target skip \
    2>&1 | tee /tmp/preprocess_debug_partial.log
```

**Pass criteria for Step 1** (all checks based on the JSONL output and on-disk artefacts; the per-frame info log is operational visibility, not a validation source):

| Check | Pass condition |
|---|---|
| `wc -l datasets/uamvla_calvin/task_D_D_partial/data.jsonl` | `>= 503` (no fewer rows than the prior baseline) |
| `distinct episode_id` count | `== 9` (every source window retained) |
| `produced no samples` log lines | `== 0` (no window dropped) |
| Partial-row count (rows where `image_target` not in row) | matches the difference `images/obs/static_count - images/target_count` (i.e. seg-failed frames count from filesystem, not from logs) |
| `images/obs/static/` file count | `== data.jsonl rows` (no orphans) |
| `images/target/` file count | equals number of rows that DO have `image_target` (= rows - partial-rows) |
| `statistics.yaml` schema | top-level keys are `{view_names, max_action_dim, embodiment_stats, state_stats, cameras, point_cloud}`; `state_stats[franka_calvin]` present |

### 4.3 Remote — Step 2: re-preprocess `task_ABC_D`

**Precondition: Step 1 must pass.** The output directory must be a clean slate before Step 2 — the preprocessor merges any pre-existing `shards/*.jsonl` into the final output, so a partial leftover from a crashed prior run would silently contaminate the new dataset.

```bash
# Clean slate (mandatory — even if you think the directory is empty, do this).
rm -rf datasets/uamvla_calvin/task_ABC_D

nohup env CUDA_VISIBLE_DEVICES=0 python runners/preprocess_calvin.py \
    --input_dir /inspire/.../UamVLA/datasets/task_ABC_D/training \
    --output_dir datasets/uamvla_calvin/task_ABC_D \
    --dataset_source task_ABC_D \
    --num_workers 8 \
    --on_resolve_failure skip --on_missing_target skip \
    > /tmp/preprocess_task_ABC_D_v2.log 2>&1 &

tail -f /tmp/preprocess_task_ABC_D_v2.log
```

If Step 2 crashes mid-run, **always `rm -rf datasets/uamvla_calvin/task_ABC_D` before re-launching** — partial shards from the crashed run would otherwise be merged into the next attempt's output.

**Pass criteria for Step 2** (numbers compared to Step 2.1 baseline):

| Metric | Pre-fix | Post-fix expectation | Pass condition |
|---|---|---|---|
| `data.jsonl` rows | 937,169 | ~1,072k | strictly greater than 937,169; expected close to **1,071,743** (= 937,169 prior rows + 134,574 previously-skipped frames; the pre-fix obs-static file count happens to equal this expected post-fix row count because every obs file was rendered, just not always referenced) |
| `distinct episode_id` count | 16,363 | 17,870 | exactly 17,870 |
| `produced no samples` log lines | 1,507 | 0 | exactly 0 |
| `images/obs/static` files | 1,071,743 | matches rows (~1,072k) | abs(diff) ≤ 1 |
| `images/target` files | 937,169 | similar (~937k) | within ±2 % of pre-fix |
| `point_clouds` files | 937,169 | similar (~937k) | within ±2 % of pre-fix |
| `images/future` files | 16,363 | 17,870 | exactly equal to episode count |
| Partial rows (no `image_target`) | 0 % | ~12.5 % | matches `images/obs/static count - images/target count` (filesystem diff; not coupled to log format) |

### 4.4 Training-side smoke (recommended)

After Step 2, sanity-check the dataset → trainer integration with a 2-step training smoke. Note the explicit `--datasets.vla_data.data_mix calvin_abc_d_uamvla` override — the yaml's default still points at `calvin_uamvla` (= `task_D_D`), so the smoke must opt into the new mixture:

```bash
WANDB_MODE=offline PYTHONPATH=$(pwd) accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
    starVLA/training/train_starvla.py \
    --config_yaml starVLA/config/training/uamvla_calvin.yaml \
    --datasets.vla_data.data_root_dir datasets/uamvla_calvin \
    --datasets.vla_data.data_mix calvin_abc_d_uamvla \
    --trainer.max_train_steps 2 \
    2>&1 | tail -30
```

**Pass condition:** 2 global steps complete, finite total loss. Per-head losses may legitimately be missing from a step's metrics if every sample in that step's batch lacked the corresponding aux field (§3.3 patch skips the head in that case) — that's expected, not a failure. With per_device_batch_size=2 and ~12.5% per-frame occlusion, P(both samples in a batch occlude the same field) ≈ 1.5%, so a 2-step smoke might or might not surface this; it does, however, surface any general crash from the framework patch.

For full multi-GPU training, change `--num_processes 1` to your GPU count (e.g., 8 on H200) and remove `--trainer.max_train_steps 2`. Operator should also consider promoting `calvin_abc_d_uamvla` to the yaml default if `task_ABC_D` becomes the canonical training set.

## 5. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **R1**: Some hidden LIBERO training path relied on `REQUIRED_OPTIONAL_FIELDS`'s `or all_samples` fallback. | LIBERO sample count silently changes. | §4.1 LIBERO regression tests (`test_uamvla_dataset.py`) must pass unchanged. If they fail, restore the constant but rewrite the filter to be CALVIN-permissive only. |
| **R2**: Multi-process preprocessor (`--num_workers 8`) has a race condition unmasked by the new logic (e.g., row-write ordering across workers). | Step 2 partial output / row drops. | Step 1 already runs at `--num_workers 8` on the debug dataset; multi-process bugs surface in ~10 minutes, not multi-hour. |
| **R3**: `pose_gt` is hard-coupled to `point_cloud`. A future head consuming `pose_gt` independently would need this coupling lifted. | Future flexibility. | YAGNI. Note in §6 follow-ups; refactor when an actual consumer arrives. |
| **R4**: Existing CALVIN test fixtures assume every sample is fully populated. | Existing tests crash after filter deletion if collators have a hidden "every sample populated" assumption. | §4.1 verifies all 13 existing tests still pass. The framework's `stack_optional_tensor_fields` documentation and code at `collator_helpers.py` already supports per-sample mask, so this is verification, not exposure of new behaviour. |
| **R5**: A batch where every sample lacks a given aux field. | Without §3.3, recon/pose heads crash on `batch[field]` access (KeyError) because the collator omits both the field and its mask. | §3.3 framework patch skips the head dispatch when its mask is missing or all-False. §3.4 Test B (`test_all_occluded_batch_skips_aux_head`) directly exercises this case — pure-occluded batch is no longer an undocumented edge case. |
| **R6**: Step 2 `rm -rf` data loss while re-running. | Lose all CALVIN data. | The operator already ran `rm -rf` on the prior output. Step 2 simply produces fresh output; if Step 2 itself crashes mid-run, re-run from scratch (preprocessor is restartable from empty state). |

## 6. Out of Scope & Follow-ups

### 6.1 Explicitly out of scope

1. Re-preprocess `task_D_D` — debug-dataset slice; real-task_D_D run is a separate task.
2. LIBERO preprocessor / data — LIBERO renders fewer occluded scenes; benefit small. Open a separate spec when needed.
3. Training yaml / launcher / aux-head loss-weight tuning — observe-and-react.
4. "Action-only" training mode toggle.
5. Configurable filter fallback (option C) — go straight to deletion.
6. Orphan-cleanup utility — old orphans are gone, new preprocessor produces none.

### 6.2 Suggested follow-ups (not in this spec)

1. **Real `task_D_D` preprocessing** when the team obtains the full task_D_D dataset; trivially adds a mixture entry per the existing pattern.
2. **CALVIN eval state-passthrough** (carried over from the prior spec's known limitation) — independent of this work.
3. **Aux-loss numerical monitoring** — after the first epoch on the new dataset, dump pose / recon / future loss means and verify they remain finite and stable. Useful, but observation, not spec.
4. **Decouple `pose_gt` from `point_cloud`** — only when a new consumer of `pose_gt` (without point_cloud) emerges.
5. **Restartable / incremental preprocessing** — ability to resume from a partial output dir on crash. Currently it's full re-run.

## 7. Naming / data-flow appendix

For consistency with the prior CALVIN spec:

- `franka_calvin` — embodiment.
- `calvin_uamvla` — mixture key.
- `task_ABC_D` — split subdir under `datasets/uamvla_calvin/`.
- `task_D_D_partial` — temporary debug-dataset output for §4.2 only; **not** registered as a mixture, **not** kept after Step 2 verification.

Per-frame data flow under the new preprocessor (replaces the prior `produced_no_samples` drop and the orphan obs/depth files):

```
frame t (after env.reset + render_cameras)
  ├─ rgb_static, rgb_wrist, depth_static, depth_wrist  ── always saved + referenced ✅
  ├─ last_rgb_static ← rgb_static                       ── always updated (G2 fix)
  ├─ depth_to_world_points → pts
  │     └─ if pts: save pc → row["point_cloud"], row["pose_6d"]
  │     └─ else (skip-mode): omit fields, proceed                 ── no early continue
  ├─ crop_target_from_seg → target_img
  │     └─ if target_img: save → row["image_target"]
  │     └─ else (skip-mode): omit field, proceed                  ── no early continue
  ├─ if either of the above omitted a field: logger.info(...) for visibility
  └─ rows.append(row)                                   ── always

window epilogue:
  ├─ if rows is empty (every frame's render itself failed — pathological):
  │     log "produced no samples" + return None
  └─ else: write image_future from last_rgb_static, write shard
```
