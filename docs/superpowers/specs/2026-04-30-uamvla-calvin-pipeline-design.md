# UamVLA CALVIN Preprocessing + Training Pipeline — Design Spec

**Date**: 2026-04-30
**Branch**: starVLA_dev
**Status**: Draft

## 1. Goal

Stand up a complete CALVIN preprocessing + training pipeline on the **UamVLA** code line, mirroring the existing LIBERO pipeline so that:

1. `python runners/preprocess_calvin.py --input_dir <CALVIN-split> --output_dir <UAM-dataset>` produces a UAM-format dataset consumable by `UamVLADataset`.
2. `bash examples/calvin/train_files/run_uamvla_calvin_train.sh` trains the UamVLA model on that dataset.

The eval path stays untouched; the existing `examples/calvin/eval_files/` (policy server + `eval_calvin.py`) is reused as-is once a checkpoint is produced.

## 2. Out of Scope

- Any change to evaluation code (`examples/calvin/eval_files/`, `eval_calvin.py`, policy server scripts, `eval_libero` parallels).
- The QwenPI-on-CALVIN line (`examples/calvin/train_files/starvla_train_calvin.yaml`, `run_calvin_train.sh`) stays untouched. Both pipelines coexist under `examples/calvin/`.
- LIBERO preprocessor refactor (e.g., extracting `write_statistics` into `BasePreprocessor`). YAGNI for now.
- Incremental processing across multiple splits into one output dir. CALVIN preprocessor today doesn't support it; this spec doesn't add it.
- Training hyperparameter tuning specific to CALVIN. We clone LIBERO settings 1:1; tuning is a follow-up after the first baseline.
- Multi-mixture training (combining CALVIN + LIBERO into one run). One CALVIN mixture per training run.

## 3. Current State

### 3.1 Already in place

| Component | Location | Status |
|-----------|----------|--------|
| CALVIN preprocessor (per-window worker, multiprocess driver) | `tools/preprocess/calvin_preprocessor.py` (703 lines) | Functional via `calvin_env` (PyBullet) |
| PyBullet env wrapper | `tools/preprocess/calvin_env_adapter.py` | Functional |
| Task-label → target-object resolver | `tools/preprocess/calvin_task_map.py` | Functional |
| `CalvinAdapter`: 15-dim `robot_obs` → canonical state | `starVLA/model/modules/uamvla/data/embodiment_adapter.py:102` | Functional |
| `franka_calvin` embodiment registration | `starVLA/model/modules/uamvla/data/embodiment_registry.py:44` | Functional |
| Structural smoke test (skip without `calvin_env`) | `tests/test_preprocessor_smoke.py:21` | Functional |

### 3.2 Gaps

| ID | File / Concern | Status |
|----|----------------|--------|
| G1 | `runners/preprocess_calvin.py` | Missing — no CLI entry point. |
| G2 | `tools/preprocess/calvin_preprocessor.py:_write_statistics` (lines 406–440) | Wrong schema — emits legacy `robot_obs_mean/std`, missing `state_stats[franka_calvin]`. **Blocks training**: `StateNormalizer.__init__` raises `KeyError` when `state_stats` is absent (`state_normalizer.py:50–54`). |
| G3 | `DATASET_NAMED_MIXTURES["calvin_uamvla"]` in `starVLA/dataloader/uamvla_dataset.py:28` | Missing. |
| G4 | `starVLA/config/training/uamvla_calvin.yaml` | Missing. The existing `examples/calvin/train_files/starvla_train_calvin.yaml` is for the QwenPI framework. |
| G5 | `examples/calvin/train_files/run_uamvla_calvin_train.sh` | Missing. The existing `run_calvin_train.sh` launches QwenPI. |
| G6 | Local pipeline test (no `calvin_env` required) | Missing. The existing smoke test only verifies imports. |

## 4. Data Flow

```
CALVIN split dir (e.g. calvin_debug_dataset/training/)
  ├── episode_NNNNNNN.npz × N           ← per-frame state+action+RGB+depth (RGB unused by us)
  ├── ep_start_end_ids.npy
  ├── ep_lens.npy
  ├── scene_info.npy
  └── lang_annotations/auto_lang_ann.npy

         │  python runners/preprocess_calvin.py --input_dir <split> --output_dir <subdir>
         │    ├─ collect_lang_windows  → 64-frame windows (drop non-language frames)
         │    ├─ CalvinWorker.process_window per window
         │    │    └─ env.reset(robot_obs, scene_obs); env.render_cameras(256, 256)
         │    │    └─ write RGB / depth / point_cloud / target / pose_6d / shard JSONL
         │    └─ merge_shards_and_write_stats
         ▼

datasets/uamvla_calvin/<split>/         ← UAM format (mirrors datasets/uamvla_libero/<suite>/)
  ├── data.jsonl                        ← one row per sample, raw 15-dim robot_obs
  ├── statistics.yaml                   ← embodiment_stats + state_stats[franka_calvin] (NEW schema)
  ├── images/obs/{static,wrist}/*.jpg   ← 256×256 PyBullet renders
  ├── images/{target,future}/*.jpg
  ├── depth/{static,wrist}/*.npy
  ├── point_clouds/*.npy
  └── shards/                           ← intermediate per-window shards

         │  UamVLADataset(data_root=datasets/uamvla_calvin/<split>, embodiment="franka_calvin")
         │    ├─ load data.jsonl + statistics.yaml
         │    ├─ CalvinAdapter.to_canonical(raw)
         │    │   → {arm_0: {ee_pose:9, joint_pos:7}, gripper_0:1}
         │    └─ StateNormalizer(stats, "franka_calvin", mode="q99")
         ▼

starVLA training step (config = uamvla_calvin.yaml, framework = UamVLA)
```

## 5. Detailed Changes

### 5.1 Preprocessor statistics migration (G2)

**File**: `tools/preprocess/calvin_preprocessor.py:_write_statistics`

**Replace** the current implementation (lines 406–440) with one that mirrors `LiberoPreprocessor.write_statistics` (`libero_preprocessor.py:691–746`):

```python
def _write_statistics(samples, output_dir, camera_intrinsics):
    from starVLA.model.modules.uamvla.data.embodiment_adapter import CalvinAdapter

    actions_7d = np.array(
        [s["action"][:FRANKA_ACTION_DIM] for s in samples], dtype=np.float64,
    )

    # Compute per-field stats over canonical state.
    adapter = CalvinAdapter()
    field_buffers = {
        "arm_0.ee_pose":   [],
        "arm_0.joint_pos": [],
        "gripper_0":       [],
    }
    for s in samples:
        canonical = adapter.to_canonical(s)  # reads s["robot_obs"] (15-dim)
        field_buffers["arm_0.ee_pose"].append(canonical["arm_0"]["ee_pose"].numpy())
        field_buffers["arm_0.joint_pos"].append(canonical["arm_0"]["joint_pos"].numpy())
        field_buffers["gripper_0"].append(canonical["gripper_0"].numpy())

    franka_state_stats = {}
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

**Removed**: `robot_obs_mean`, `robot_obs_std` keys (legacy 15-dim aggregate, no production reader after the LIBERO migration). Repo-wide `grep -rn 'robot_obs_mean\|robot_obs_std'` was run during this design — only references are inside the calvin preprocessor itself and historical plan docs. Safe to delete.

The constants used above (`EMBODIMENT = "franka_calvin"`, `FRANKA_ACTION_DIM = 7`, `MAX_ACTION_DIM = 24`, `NUM_POINTS = 1024`) already exist at the top of `calvin_preprocessor.py` (lines 135–141) — no rename needed.

### 5.2 Mixture registration (G3)

**File**: `starVLA/dataloader/uamvla_dataset.py:28`

```python
DATASET_NAMED_MIXTURES = {
    "libero_uamvla": [
        # (data_subdir, weight, embodiment_tag)
        ("libero_spatial", 1.0, "franka_libero"),
    ],
    "calvin_uamvla": [
        ("task_D_D", 1.0, "franka_calvin"),   # NEW
    ],
}
```

The subdir name (`task_D_D`) follows CALVIN's standard split naming. The debug dataset is treated as a `task_D_D`-shaped slice and processed into `datasets/uamvla_calvin/task_D_D/`.

To run a different split (e.g., `task_ABC_D`), users add another one-line entry — out of scope for this spec, but the pattern is now established.

### 5.3 Training yaml (G4)

**File**: `starVLA/config/training/uamvla_calvin.yaml` (new)

Cloned from `starVLA/config/training/uamvla_libero.yaml` with these substitutions:

| Field | LIBERO value | CALVIN value |
|-------|--------------|--------------|
| `run_id` | `uamvla_libero_phase1` | `uamvla_calvin_phase1` |
| `framework.embodiment.name` | `franka_libero` | `franka_calvin` |
| `datasets.vla_data.data_root_dir` | `datasets/uamvla_libero` | `datasets/uamvla_calvin` |
| `datasets.vla_data.data_mix` | `libero_uamvla` | `calvin_uamvla` |

All other fields **identical**, including:

- Full aux_heads block (action / pose / future / recon all enabled).
- `state_encoder.normalization.apply_to: [arm_0.ee_pose, arm_0.joint_pos, gripper_0]` — the canonical schema CALVIN produces matches LIBERO byte-for-byte.
- `image_size: 640` (Qwen3-VL transform target; preprocessor renders at 256×256, dataloader transform handles the upscale).
- Trainer block: same LRs, scheduler, batch size, grad accumulation, optimizer settings.

### 5.4 Preprocessor CLI (G1)

**File**: `runners/preprocess_calvin.py` (new)

Thin wrapper around `CalvinPreprocessor(...).process(input_dir, output_dir)`, structurally identical to `runners/preprocess_libero.py`.

```
usage: preprocess_calvin.py [-h] --input_dir INPUT_DIR --output_dir OUTPUT_DIR
                            [--dataset_source DATASET_SOURCE]
                            [--num_workers NUM_WORKERS]
                            [--default_scene DEFAULT_SCENE]
                            [--on_resolve_failure {skip,abort}]
                            [--on_missing_target {skip,abort}]
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--input_dir` | (required) | A CALVIN split dir containing `episode_*.npz`, `ep_start_end_ids.npy`, `lang_annotations/auto_lang_ann.npy`, `scene_info.npy`. |
| `--output_dir` | (required) | Target UAM dataset dir; created if missing. Recommended layout: `datasets/uamvla_calvin/<split>/`. |
| `--dataset_source` | `"calvin"` | String prefix for `episode_id` and JSONL `dataset_source` field. Override with the split name (`task_D_D`, `calvin_debug`) for clarity in audit logs. |
| `--num_workers` | `1` | `multiprocessing.Pool` size. Use >1 only on machines with reliable PyBullet/EGL. The preprocessor uses `_POOL_CONTEXT="spawn"` to avoid fork+EGL crashes. |
| `--default_scene` | `None` | Forwarded to `SceneResolver`. Set when the split's `scene_info.npy` does not cover all windows. |
| `--on_resolve_failure` | `abort` | `skip` or `abort` if a window's `task_label` is unknown to `calvin_task_map`. |
| `--on_missing_target` | `abort` | `skip` or `abort` if a frame's target object is not visible in the static-camera seg mask. |

Header comment example:
```
# Process the debug dataset:
#   python runners/preprocess_calvin.py \
#       --input_dir /Users/tancilon/develop/localgit/UamVLA/datasets/calvin_debug_dataset/training \
#       --output_dir datasets/uamvla_calvin/task_D_D \
#       --dataset_source calvin_debug
```

### 5.5 Training launcher (G5)

**File**: `examples/calvin/train_files/run_uamvla_calvin_train.sh` (new)

Cloned from `examples/calvin/train_files/run_calvin_train.sh` (the QwenPI launcher). Substitutions:

- `Framework_name=UamVLA`  *(was `QwenPI`)*
- `config_yaml=./starVLA/config/training/uamvla_calvin.yaml`  *(was QwenPI yaml)*
- `calvin_data_root=datasets/uamvla_calvin`  *(was LeRobot-format path)*
- `data_mix=calvin_uamvla`  *(was `calvin_task_D_D`)*
- `run_id=0430_uamvla_calvin_task_D_D`
- `wandb_project=uamvla_calvin`

Drop QwenPI-only flags: `--framework.qwenvl.base_vlm`, `--datasets.vla_data.per_device_batch_size 4`, `--trainer.vla_data.video_backend torchvision_av`. UamVLA reads these from the yaml directly.

The two launchers (`run_calvin_train.sh` for QwenPI and `run_uamvla_calvin_train.sh` for UamVLA) coexist under `examples/calvin/train_files/`.

### 5.6 README addendum

**File**: `examples/calvin/README.md`

Add a "**UamVLA path**" section at the top distinguishing it from the existing QwenPI section. Required content:

- One-paragraph statement that the directory hosts two independent training pipelines (QwenPI via LeRobot format; UamVLA via UAM unified format) and they share only the eval scripts in `eval_files/`.
- Quickstart for the UamVLA path: preprocess command (referring to `runners/preprocess_calvin.py`), train command (referring to `run_uamvla_calvin_train.sh`).

## 6. Data Contract

### 6.1 `robot_obs` handling

| Stage | What happens |
|-------|--------------|
| Read from CALVIN npz (`calvin_preprocessor.py:221`) | All 15 dims loaded as float32. |
| Written to JSONL (`calvin_preprocessor.py:318`) | Full 15-dim array stored verbatim under `"robot_obs"`. Auditability preserved. |
| Used to drive PyBullet env replay (`calvin_preprocessor.py:224`) | Combined with `scene_obs` (24-dim) to call `env.reset`. `scene_obs` is **not** persisted to JSONL. |
| `CalvinAdapter.to_canonical` (`embodiment_adapter.py:115`) | Splits into: `arm_0.ee_pose` (9D = `tcp_xyz` + `euler_to_6d(euler)`), `arm_0.joint_pos` (7D = `robot_obs[7:14]`), `gripper_0` (1D = `robot_obs[6] / 0.077`). **`robot_obs[14]` (gripper command) is dropped** — it duplicates `rel_actions[6]` (the action label) and mixing a command into proprio creates train/eval distribution skew. |
| `StateNormalizer` (`state_normalizer.py:72`) | Per-field q99 affine using `state_stats[franka_calvin][<field>]` from `statistics.yaml`. |

### 6.2 Canonical state schema (matches LIBERO byte-for-byte)

```python
{
    "arm_0": {
        "ee_pose":   torch.Tensor (9,),  # tcp_xyz(3) + 6D-rot from euler(3)
        "joint_pos": torch.Tensor (7,),  # 7-DoF Franka joints
    },
    "gripper_0": torch.Tensor (1,),       # gripper width / 0.077, ~[0, 1]
}
```

### 6.3 Action contract (unchanged)

7-dim from `rel_actions` per timestep, padded to 24-dim with zeros, `action_mask = [1]*7 + [0]*17`. Min-max normalized to [-1, 1] at dataset time using `embodiment_stats[franka_calvin].action_{min,max}_bound`.

### 6.4 `statistics.yaml` schema (after migration)

```yaml
view_names: [static, wrist]
max_action_dim: 24
embodiment_stats:
  franka_calvin:
    action_dim: 7
    action_min_bound: [..7 floats..]
    action_max_bound: [..7 floats..]
state_stats:
  franka_calvin:
    arm_0.ee_pose:   {q01: [...9...], q99: [...9...], min: [...9...], max: [...9...], mean: [...9...], std: [...9...]}
    arm_0.joint_pos: {q01: [...7...], q99: [...7...], min: [...7...], max: [...7...], mean: [...7...], std: [...7...]}
    gripper_0:       {q01: [...1...], q99: [...1...], min: [...1...], max: [...1...], mean: [...1...], std: [...1...]}
cameras:
  static: {intrinsic: {fx, fy, cx, cy, ...}}
  wrist:  {intrinsic: {fx, fy, cx, cy, ...}}
point_cloud: {num_points: 1024, frame: world}
# NO robot_obs_mean / robot_obs_std
```

## 7. Verification Strategy

Two layers, mirroring the LIBERO precedent.

### 7.1 Local unit (any machine, no `calvin_env` required)

**File**: `tests/test_uamvla_calvin_pipeline_local.py` (new)

Test cases:

1. **`test_calvin_uamvla_mixture_registered`** — assert `DATASET_NAMED_MIXTURES["calvin_uamvla"]` equals `[("task_D_D", 1.0, "franka_calvin")]`.

2. **`test_cli_help`** — invoke `python runners/preprocess_calvin.py --help` via subprocess, assert exit code 0 and that the six expected args appear in stdout.

3. **`test_cli_required_args`** — invoke without `--input_dir` / `--output_dir`, assert non-zero exit.

4. **`test_synthetic_dataset_loads`** — given a synthetic CALVIN-shape fixture under `tmp_path`:
   - `data.jsonl` with 4 rows. Each row has:
     - `id`, `episode_id`, `step_idx`, `total_steps=4`
     - `image: ["images/obs/static/<id>.jpg", "images/obs/wrist/<id>.jpg"]`
     - `robot_obs`: 15-element list of plausible CALVIN values (matches the sample we probed: tcp ≈ [0.1, -0.09, 0.57], euler ≈ [3.1, 0.02, 1.5], gripper_width ≈ 0.08, joints ≈ 7×[-π, π], gripper_cmd = 1.0).
     - `action`: 24-dim (first 7 = `rel_actions`, remaining 17 = 0).
     - `action_mask`: `[1]*7 + [0]*17`.
     - `embodiment: "franka_calvin"`, `action_dim: 7`.
     - Aux fields populated: `image_target`, `image_future`, `point_cloud`, `pose_6d`, `static_cam_extrinsic`, `wrist_cam_extrinsic`, `depth_static`, `depth_wrist`.
   - Synthetic 256×256 RGB JPGs and 1024×3 point cloud `.npy` files at the referenced paths.
   - `statistics.yaml` with `embodiment_stats[franka_calvin]` and `state_stats[franka_calvin]` populated with realistic-magnitude values (use the same fixture data run through `CalvinAdapter` to compute the q99 stats — keeps the fixture self-consistent).

   Then instantiate `UamVLADataset(data_root=tmp_path, embodiment="franka_calvin", action_horizon=8, normalization={"mode": "q99", "apply_to": ["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"]})` and assert:
   - `len(ds) == 4`.
   - `ds[0]["canonical_state"]["arm_0"]["ee_pose"].shape == (9,)`.
   - `ds[0]["canonical_state"]["arm_0"]["joint_pos"].shape == (7,)`.
   - `ds[0]["canonical_state"]["gripper_0"].shape == (1,)`.
   - `ds[0]["action"].shape == (8, 7)` and `ds[0]["action_mask"].shape == (8, 7)`.
   - State normalizer is wired: the post-normalizer `ee_pose` differs from the raw 9-dim canonical pre-normalization (computed independently). This catches the case where `StateNormalizer` silently no-ops because of a mode mismatch.

5. **`test_write_statistics_schema`** — call `_write_statistics(samples, tmp_path, camera_intrinsics)` directly with synthetic samples (each having a 15-dim `robot_obs` and 24-dim `action`). Then load the YAML and assert:
   - Top-level keys are exactly: `{view_names, max_action_dim, embodiment_stats, state_stats, cameras, point_cloud}`.
   - **Absent**: `robot_obs_mean`, `robot_obs_std`.
   - `state_stats["franka_calvin"]` has exactly `{"arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"}` as keys.
   - Each field has all six sub-keys (`q01`, `q99`, `min`, `max`, `mean`, `std`).
   - Lengths: `arm_0.ee_pose` → 9, `arm_0.joint_pos` → 7, `gripper_0` → 1.

6. **`test_write_statistics_no_legacy_keys`** — explicit assertion that `"robot_obs_mean" not in stats and "robot_obs_std" not in stats`. Failing test guards the migration.

7. **`test_yaml_loadable`** — load `starVLA/config/training/uamvla_calvin.yaml` via OmegaConf and assert `framework.embodiment.name == "franka_calvin"`, `datasets.vla_data.data_mix == "calvin_uamvla"`, `datasets.vla_data.data_root_dir == "datasets/uamvla_calvin"`. Catches accidental yaml drift from the LIBERO clone.

### 7.2 Remote e2e (machine with `calvin_env` installed)

Procedure documented in `examples/calvin/README.md` and as a header comment in `run_uamvla_calvin_train.sh`. **Not** auto-run by CI — it requires PyBullet/EGL which is unavailable on macOS.

Steps:

1. Activate the conda environment that has `calvin_env`.
2. Run preprocessor on the debug dataset:
   ```bash
   python runners/preprocess_calvin.py \
       --input_dir /Users/tancilon/develop/localgit/UamVLA/datasets/calvin_debug_dataset/training \
       --output_dir datasets/uamvla_calvin/task_D_D \
       --dataset_source calvin_debug
   ```
   Expectation: exits 0; produces `data.jsonl` with ~576 rows (9 windows × ≤64 frames).

3. Verify output:
   - `datasets/uamvla_calvin/task_D_D/data.jsonl` has ≥ 1 row.
   - `statistics.yaml` has `state_stats[franka_calvin]` (no legacy `robot_obs_mean/std`).
   - At least one image present in each of `images/obs/static/`, `images/obs/wrist/`, `images/target/`, `images/future/`.
   - At least one `.npy` in `point_clouds/`.

4. 2-step training smoke:
   ```bash
   python starVLA/training/train_starvla.py \
       --config_yaml starVLA/config/training/uamvla_calvin.yaml \
       --datasets.vla_data.data_root_dir datasets/uamvla_calvin \
       --is_debug True \
       --trainer.max_train_steps 2
   ```
   Expectation: completes 2 training steps without exception; loss is finite.

### 7.3 What the existing smoke test still covers

`tests/test_preprocessor_smoke.py:21` (`test_calvin_preprocessor_imports`) is kept. It catches *import-time* regressions (e.g., a refactor that breaks `from tools.preprocess.calvin_preprocessor import CalvinPreprocessor` on a machine that has `calvin_env`). The new `test_uamvla_calvin_pipeline_local.py` covers schema and dataset wiring on machines that *don't* have `calvin_env`. Both layers are useful.

## 8. Rollout Sequence (TDD task order)

Each step is a TDD checkpoint: write failing test → implement → confirm green → commit.

| # | Task | New/Modified files | Test |
|---|------|-------------------|------|
| 1 | **Stats migration** (G2) | `tools/preprocess/calvin_preprocessor.py` — replace `_write_statistics` (lines 406–440) | `test_write_statistics_schema`, `test_write_statistics_no_legacy_keys` |
| 2 | **Mixture entry** (G3) | `starVLA/dataloader/uamvla_dataset.py:28` — one-line add | `test_calvin_uamvla_mixture_registered` |
| 3 | **Synthetic-fixture chain** (verifies 1+2 compose) | `tests/test_uamvla_calvin_pipeline_local.py` — add fixtures + dataset chain test | `test_synthetic_dataset_loads` |
| 4 | **CLI runner** (G1) | `runners/preprocess_calvin.py` (new) | `test_cli_help`, `test_cli_required_args` |
| 5 | **Training yaml** (G4) | `starVLA/config/training/uamvla_calvin.yaml` (new) | `test_yaml_loadable` |
| 6 | **Launcher script** (G5) | `examples/calvin/train_files/run_uamvla_calvin_train.sh` (new) | Manual: `bash -n run_uamvla_calvin_train.sh` (syntax check) |
| 7 | **README addendum** | `examples/calvin/README.md` — add UamVLA section | Manual review |
| 8 | **Remote e2e (manual, not auto)** | — | `bash run_uamvla_calvin_train.sh` after `preprocess_calvin.py` on the debug dataset, in a `calvin_env`-equipped environment |

Tasks 1–7 are runnable on macOS (the developer's local box) without `calvin_env`. Task 8 is the gating remote check before merging.

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| `calvin_env` install on the developer's local Mac fails (EGL is Linux-NVIDIA-specific). | Local unit tests (§7.1) bypass it entirely via the synthetic fixture. The remote e2e (§7.2) is the gating check; spec does not require local `calvin_env`. |
| `make_calvin_env_adapter(dataset_path=...)` may require `<dataset_path>/.hydra/config.yaml` (CALVIN's own metadata). The debug dataset's `training/` subdirectory may not have it. | Surfaces on first §7.2 run as a clear `FileNotFoundError`. If it fires: (a) point at the parent dir that does have `.hydra/`, or (b) add a `--hydra_config_dir` CLI override. Out of scope to fix preemptively — verify first. |
| Two training pipelines coexisting under `examples/calvin/` invites confusion. | Distinct launcher names (`run_calvin_train.sh` for QwenPI, `run_uamvla_calvin_train.sh` for UamVLA); README addendum explicitly separates the two paths. |
| Synthetic fixture in §7.1 drifts from real CALVIN data shape over time. | Fixture is built using the same `CalvinAdapter` and constants the real preprocessor uses (`FRANKA_ACTION_DIM`, `MAX_ACTION_DIM`, `EMBODIMENT`). Any future schema change forces both real + fixture to update together. |
| `state_stats` block written by the migration is computed using `CalvinAdapter`. If `CalvinAdapter` changes (e.g., gripper normalization constant), old preprocessed datasets will have stale stats. | `statistics.yaml` is regenerated from `data.jsonl` on every `merge_shards_and_write_stats` call. Re-running the preprocessor refreshes stats. Document this in the README. |
| `robot_obs_mean/std` deletion breaks something we missed in the grep. | The repo-wide grep at design time returned only the calvin preprocessor itself and historical plan docs. If a hidden consumer surfaces, restore the legacy block as a one-line fallback alongside the new schema (cheap reversal). |

## 10. Naming Convention Summary

To prevent confusion across the touched components:

- `franka_calvin` — embodiment name (registry key, statistics.yaml keys, embodiment adapter).
- `calvin_uamvla` — mixture name (`DATASET_NAMED_MIXTURES` key, `data_mix` yaml value).
- `task_D_D` — split name (mixture entry's subdir, suggested directory under `datasets/uamvla_calvin/`).
- `uamvla_calvin_phase1` — `run_id` in the training yaml.
- `datasets/uamvla_calvin/` — top-level data root directory; sibling of `datasets/uamvla_libero/`.

This is intentionally parallel to `franka_libero` / `libero_uamvla` / `libero_spatial` / `uamvla_libero_phase1` / `datasets/uamvla_libero/`.
