# CALVIN LeRobot Multi-Scene Merge Design

Date: 2026-05-14

## Goal

Regenerate `datasets/calvin2uam/lerobot_calvin_abcd` from the raw multi-scene
CALVIN `task_ABCD_D` split without rendering scene A/B/C/D samples under the
wrong PyBullet scene.

The formal pipeline must:

- split the raw multi-scene CALVIN split into per-scene virtual splits;
- run the existing LeRobot preprocessor once per scene with the right `.hydra`
  scene config;
- merge the per-scene LeRobot outputs into one valid LeRobot dataset;
- fail safely when the final output directory already exists unless the user
  explicitly requests overwrite;
- generate `meta/stats_gr00t.json` by default, with an option to skip stats for
  fast smoke/debug runs.

## Context

`runners/preprocess_calvin.py` already outputs the current LeRobot v2 layout:

```text
<dataset>/
  data/chunk-XXX/episode_NNNNNN.parquet
  videos/chunk-XXX/video.primary_image/episode_NNNNNN.mp4
  videos/chunk-XXX/video.wrist_image/episode_NNNNNN.mp4
  image_targets/<episode_index>.png
  point_clouds/<episode_index>/<base_index>.npy
  camera_params.json
  meta/modality.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/info.json
```

It creates one PyBullet environment for the provided split directory. That env
gets its static fixture layout from the split's `.hydra/merged_config.yaml`.
For a multi-scene split such as `task_ABCD_D`, resetting the env with per-frame
`scene_obs` is not enough to switch static fixtures between A/B/C/D. Therefore a
single direct run over `task_ABCD_D/training` can render frames with the wrong
URDF layout.

`tools/preprocess/calvin_scene_splitter.py` already solves the rendering issue
at the input side by creating per-scene virtual split directories and rewriting
each virtual split's `.hydra/merged_config.yaml`.

The blocker is the final merge step. `tools/preprocess/calvin_split_merger.py`
is a JSONL-era merger: it expects `data.jsonl`, JSONL artifact directories, and
`statistics.yaml`. It is incompatible with the current LeRobot parquet dataset
format and includes a `NotImplementedError` for the removed statistics path.

## Decision

Use a LeRobot-specific merger and remove the obsolete JSONL merger.

Specifically:

- add `tools/preprocess/calvin_lerobot_merger.py`;
- delete `tools/preprocess/calvin_split_merger.py`;
- update `runners/preprocess_calvin_multiscene.py` to import and call the new
  LeRobot merger;
- update tests that reference the old JSONL merger to target the new LeRobot
  merger behavior.

This keeps format-specific logic separate and prevents future accidental use of
the old JSONL/statistics.yaml path.

## Pipeline

The formal pipeline remains a three-stage process:

1. `calvin_scene_splitter.split_calvin_by_scene(...)`
   - reads `scene_info.npy`;
   - creates `work_dir/virtual/scene_A`, `scene_B`, etc.;
   - filters `lang_annotations/auto_lang_ann.npy`;
   - symlinks only `episode_*.npz` files in that scene range;
   - rewrites each virtual split's `.hydra/merged_config.yaml` to point at the
     matching `calvin_scene_<X>.yaml`.

2. `runners/preprocess_calvin.py`
   - runs once per virtual split;
   - each run gets a unique per-scene `dataset_source`;
   - each run writes a standalone per-scene LeRobot dataset under
     `work_dir/preproc/scene_X`.

3. `calvin_lerobot_merger.merge_lerobot_scene_outputs(...)`
   - reads all per-scene LeRobot dataset outputs;
   - renumbers episodes and rows into one global dataset;
   - rewrites parquet metadata columns;
   - copies or hard-links videos and sidecars to the new global episode ids;
   - rebuilds `meta/*`, `camera_params.json`, and stats.

## Output Safety

`runners/preprocess_calvin_multiscene.py` must be safe by default.

- If `--output_dir` exists and `--overwrite` is not passed, abort before any
  expensive preprocessing starts.
- If `--overwrite` is passed, remove `--output_dir` before the merge writes the
  final dataset.
- Per-scene preprocessor outputs under `work_dir/preproc/scene_X` are always
  removed and regenerated for each run. This avoids stale scene outputs leaking
  into a new merge.
- `--clean_work` removes the work directory only after all stages complete
  successfully.

Atomic final replacement is intentionally out of scope for this iteration. The
chosen behavior is explicit and predictable: no overwrite unless requested.

## LeRobot Merge Semantics

The new merger accepts only LeRobot-format per-scene dataset directories.
Each input directory must contain:

```text
data/chunk-*/episode_*.parquet
videos/chunk-*/video.primary_image/episode_*.mp4
videos/chunk-*/video.wrist_image/episode_*.mp4
image_targets/*.png
point_clouds/<episode_index>/<base_index>.npy
camera_params.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/info.json
meta/modality.json
```

Inputs are merged in the provided order, normally A, B, C, D. Original
per-scene episode ids are not preserved. The merger assigns new global episode
ids:

```text
scene_A local episodes 0..Na-1       -> global episodes 0..Na-1
scene_B local episodes 0..Nb-1       -> global episodes Na..Na+Nb-1
scene_C local episodes 0..Nc-1       -> next range
scene_D local episodes 0..Nd-1       -> next range
```

For every episode parquet, the merger rewrites:

- `episode_index`: the new global episode id;
- `trajectory_id`: the new global episode id;
- `frame_index`: contiguous `0..length-1` inside the episode;
- `base_index`: contiguous `0..length-1` inside the episode;
- `index`: global contiguous frame index across the merged dataset;
- `task_index`: index into the rebuilt global `meta/tasks.jsonl`.

The merger writes each episode to the standard chunk path:

```text
data/chunk-{global_episode_index // 1000:03d}/episode_{global_episode_index:06d}.parquet
```

It also maps files to the matching global episode id:

- `videos/chunk-*/video.primary_image/episode_<old>.mp4`
  -> `videos/chunk-*/video.primary_image/episode_<global>.mp4`;
- `videos/chunk-*/video.wrist_image/episode_<old>.mp4`
  -> `videos/chunk-*/video.wrist_image/episode_<global>.mp4`;
- `image_targets/<old>.png` -> `image_targets/<global>.png`;
- `point_clouds/<old>/<base>.npy` -> `point_clouds/<global>/<base>.npy`.

Files should be hard-linked where possible and copied when hard-linking fails,
so the final dataset is relocatable and self-contained.

## Metadata

The merger rebuilds metadata instead of concatenating per-scene metadata files.

### `meta/tasks.jsonl`

Deduplicate by instruction string in first-seen order across merged episodes.
Write one record per unique instruction:

```json
{"task_index": 0, "task": "move the door all the way to the right"}
```

### `meta/episodes.jsonl`

Write one record per global episode:

```json
{"episode_index": 0, "tasks": ["..."], "length": 65}
```

### `meta/info.json`

Recompute:

- `total_episodes`;
- `total_frames`;
- `total_tasks`;
- `splits.train`;
- `chunks_size`, fixed to `1000`;
- `data_path`;
- `video_path`;
- video feature schema for `video.primary_image` and `video.wrist_image`.

The schema should match `CalvinPreprocessorLeRobot._emit_meta`.

### `meta/modality.json`

Copy the canonical file from
`examples/calvin/train_files/data_registry/modality.json`, matching the
single-scene LeRobot preprocessor.

### `camera_params.json`

Copy from the first input dataset. Validate every other input has identical
camera params. If any input differs, abort with a clear error, because pose and
point-cloud consumers assume one dataset-level static camera model.

## Stats

By default, the formal pipeline generates `meta/stats_gr00t.json` after the
merge completes.

Implementation should reuse the same LeRobot stats logic used by the current
dataset reader. The existing reader stores stats in `meta/stats_gr00t.json`
with a cache payload containing `__format_version`, `__cache_config`, and
`statistics`. The merger must write that same cache format, not the removed
`statistics.yaml` schema.

If the stats logic is currently only triggered during first dataset load,
expose a small reusable helper such as:

```python
compute_lerobot_stats(
    dataset_root: Path,
    dataset_name: str,
    robot_type: str,
    action_mode: str,
) -> Path
```

The helper should write the same `meta/stats_gr00t.json` schema that the reader
expects. For the CALVIN ABCD dataset, `dataset_name` is normally
`lerobot_calvin_abcd`, `robot_type` is `uamvla_calvin_franka`, and
`action_mode` should match training config semantics, currently `delta_qpos` /
delta action statistics.

The multiscene runner and the merger CLI both expose `--skip_stats`. When
`--skip_stats` is passed, the merged dataset is still structurally valid, but
the first dataloader use may need to compute stats lazily.

## CLI

New standalone merger:

```bash
python -m tools.preprocess.calvin_lerobot_merger \
  --inputs work/preproc/scene_A work/preproc/scene_B work/preproc/scene_C work/preproc/scene_D \
  --output_dir datasets/calvin2uam/lerobot_calvin_abcd \
  --dataset_name lerobot_calvin_abcd \
  --robot_type uamvla_calvin_franka \
  --action_mode delta \
  --overwrite
```

The standalone merger must also accept `--skip_stats`.

Updated end-to-end runner:

```bash
python runners/preprocess_calvin_multiscene.py \
  --input_dir datasets/calvin/task_ABCD_D/training \
  --work_dir /tmp/calvin_abcd_lerobot_work \
  --output_dir datasets/calvin2uam/lerobot_calvin_abcd \
  --dataset_source calvin_abcd \
  --scenes A,B,C,D \
  --overwrite
```

The runner should forward `--skip_stats`, `--robot_type`, and `--action_mode`
to the merger. Defaults should match the existing CALVIN registry:
`robot_type=uamvla_calvin_franka`, `action_mode=delta`.

`--dataset_source` remains useful for logs and per-scene naming, but the
LeRobot merged output does not need JSONL-style sample ids.

## Error Handling

The merger must fail loudly when:

- an input directory is missing required LeRobot top-level paths;
- an episode listed in `meta/episodes.jsonl` has no parquet;
- required sidecars for an emitted episode are missing;
- primary or wrist video files are missing;
- a parquet file lacks expected columns;
- camera params differ across inputs;
- output already exists without explicit overwrite.

The merger should include the input path, scene directory, and local episode id
in errors where possible. The most useful failure is the one that lets the user
rerun only the broken preprocessing stage.

## Tests

Add tests around small synthetic LeRobot scene outputs rather than full CALVIN
data.

Required tests:

1. **Episode renumbering**
   - build two tiny input scene datasets;
   - merge them;
   - assert global episode ids are contiguous and chunk paths match.

2. **Parquet rewrite**
   - assert `episode_index`, `trajectory_id`, `frame_index`, `base_index`,
     `index`, and `task_index` are rewritten correctly.

3. **Sidecar and video remap**
   - assert image targets, point clouds, and both videos land under global
     episode ids.

4. **Meta rebuild**
   - assert `tasks.jsonl` deduplicates instructions;
   - assert `episodes.jsonl` lengths match parquet row counts;
   - assert `info.json` totals and `chunks_size` are correct.

5. **Output safety**
   - assert existing output without `--overwrite` fails;
   - assert `--overwrite` clears and recreates the output.

6. **Camera params validation**
   - identical camera params pass;
   - mismatched camera params fail.

7. **Multiscene runner wiring**
   - monkeypatch subprocess preprocessing and splitter outputs;
   - assert the runner clears per-scene preproc dirs and calls the new merger.

8. **Stats switch**
   - `--skip_stats` skips stats generation;
   - default path calls the stats helper. If the real stats helper is too heavy
     for CI, mock it and assert the call contract.

Existing tests that validate `calvin_split_merger.py` JSONL behavior should be
removed or rewritten for `calvin_lerobot_merger.py`.

## Out Of Scope

- Changing how `runners/preprocess_calvin.py` renders a single scene.
- Parallelizing per-scene preprocessing beyond the existing subprocess-per-scene
  structure.
- Supporting incremental reuse of existing `work_dir/preproc/scene_X` outputs.
- Atomic replacement of an existing final output directory.
- Preserving original local per-scene episode ids in the merged dataset.

## Acceptance Criteria

The implementation is acceptable when:

- `runners/preprocess_calvin_multiscene.py` no longer imports the old JSONL
  merger;
- `tools/preprocess/calvin_split_merger.py` is deleted;
- `tools/preprocess/calvin_lerobot_merger.py` can merge synthetic LeRobot
  per-scene datasets;
- merged parquet rows and sidecars use contiguous global episode/frame/index
  numbering;
- merged `meta/*` files match the layout expected by the current LeRobot reader;
- output overwrite behavior is safe by default;
- stats generation is available by default and skippable with `--skip_stats`;
- targeted unit tests pass.
