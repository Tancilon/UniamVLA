# LIBERO Preprocessing Throughput Optimization

Date: 2026-05-22
Status: implemented locally
Related design: `docs/superpowers/specs/2026-05-21-uamvla-libero-aux-preprocessing-design.md`

## 1. Purpose

The current LIBERO preprocessing path can generate the full UamVLA-GR00T
auxiliary sidecar layout, but full-suite preprocessing on a single RTX 4090 is
slow and fragile: a late task failure can waste many hours of completed replay
work, and every frame performs several expensive replay/render/write operations.

This design improves throughput and recoverability while preserving the exact
final dataset contract:

```text
lerobot_libero_<suite>/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/video.primary_image/episode_000000.mp4
  videos/chunk-000/video.wrist_image/episode_000000.mp4
  image_targets/<episode>/<frame>.png
  point_clouds/<episode>/<frame>.npy
  depths/static/<episode>/<frame>.npy
  grounding_masks/static/<episode>/<frame>.npy
  grounding_masks/static/<episode>/<frame>.json
  affordance_heatmaps/static/<episode>/<frame>.npy
  meta/*.json*
  camera_params.json
```

Training code and dataloader sidecar lookup semantics must not change.

## 2. Scope

In scope:

- single-machine, single-GPU preprocessing, especially one RTX 4090;
- task-level parallel replay with stable GPU binding;
- resumable suite processing without renumbering episodes or rows;
- task-level failure reports and optional fail-fast behavior;
- low-risk per-frame render-path optimization;
- lightweight task-level throughput profiling;
- writer optimizations that keep the on-disk layout byte-for-byte compatible in
  path names and array schemas.

Out of scope:

- changing final sidecar storage to episode-level `.npz`, zarr, tar shards, or
  other packed formats;
- changing model or dataloader code;
- demo-level or frame-level distributed sharding;
- multi-node merge workflows;
- changing active-target, affordance, depth, grounding, or pose semantics.

## 3. Existing Bottlenecks

The current pipeline is:

```text
runners/preprocess_libero.py
  -> LiberoPreprocessor.process()
    -> one TaskJob per HDF5 task file
      -> _TaskReplayWorker.run()
        -> replay every demo frame
        -> write every episode immediately
```

The main bottlenecks are:

1. Per-frame replay/render cost. `_extract_frame_payload()` currently renders
   static RGB, aligned static RGB, wrist RGB, static depth, instance
   segmentation, and geom segmentation. `rgb_static_aligned` is currently just a
   second call to `_render_rgb(STATIC_CAM)`, so static RGB is rendered twice.
2. Small-file I/O. The writer emits per-frame `.npy`, `.json`, and `.png`
   sidecars, plus two videos and a parquet per episode. This layout is required,
   but repeated directory creation and scattered writes still add overhead.
3. All-or-nothing pool behavior. A task failure raised through
   `multiprocessing.Pool.map()` aborts the whole suite. Completed outputs remain
   on disk, but there is no authoritative marker that lets the next run skip
   them safely.
4. Limited observability. Logs show high-level processing, but they do not
   clearly report per-task elapsed time, frames per second, retry count, or
   failed task summaries.

## 4. Design Overview

The optimized pipeline keeps task-level jobs but adds three layers:

```text
stable preprocessing plan
  -> resumable task scheduler
    -> optimized TaskReplayWorker
      -> compatible LiberoLerobotWriter
        -> final meta rebuild from plan + done markers
```

The implementation should favor robust production preprocessing over maximal
theoretical speed. On a single RTX 4090, the recommended operating point remains
`--num-workers 4` initially, with support for trying `5` or `6` if the machine
has enough CPU, memory, and I/O bandwidth.

## 5. Stable Preprocessing Plan

Before launching worker jobs, the preprocessor scans all selected HDF5 files and
computes the same `episode_plan` it uses today:

```text
(hdf5 filename, demo index)
  -> episode_index
  -> row_start
  -> frame_count
```

The plan is written to:

```text
<suite_output>/meta/preprocess_plan.json
```

The plan is the stable anchor for resume. Episode indices and global row indices
must never depend on which tasks are still incomplete. A resumed run must either
reuse the existing plan or fail with a clear error.

The plan records:

- suite name;
- sorted task filenames;
- demo counts and frame counts after any smoke-run limits;
- `episode_index` and `row_start` for every demo;
- preprocessing options that affect output shape or semantics, including
  `min_segment_len`, `active_target_score_window`, `max_tasks`,
  `max_demos_per_task`, and `max_frames_per_demo`;
- output format version;
- creation timestamp.

Plan validation is strict. If the input HDF5 set, demo counts, frame counts, or
shape-affecting options differ from the saved plan, `--resume` fails and asks
the user to run with `--overwrite` or a separate output directory.

## 6. Done Markers And Failed Reports

Each successful task writes a done marker:

```text
<suite_output>/meta/preprocess_tasks/<task_stem>.done.json
```

The marker records:

- task filename and task name;
- task index;
- processed demo indices;
- emitted episode indices;
- emitted frame count;
- coverage counts;
- camera params availability;
- elapsed seconds and frames per second;
- retry count;
- relevant preprocessing options.

If a task fails and fail-fast is disabled, the scheduler records the failure in:

```text
<suite_output>/meta/preprocess_failed_tasks.json
```

The failure report records:

- task filename and task name;
- exception type and message;
- traceback summary;
- retry count;
- worker process id if available;
- assigned render GPU id.

Failure reports are overwritten on each run with the current incomplete set, so
they describe the latest resume state rather than accumulating stale failures.

## 7. Resume Scheduler

The CLI gains:

```text
--resume
--force-task <task_stem>        # repeatable
--fail-fast
--profile
--max-retries <int>
```

`--overwrite` and `--resume` are mutually exclusive.

Run modes:

- Default without `--resume`: current behavior, except with improved profiling
  and render-path optimization.
- `--overwrite`: delete the suite output and start from a new plan.
- `--resume`: load or create the stable plan, skip tasks with valid done
  markers, process missing/failed tasks, then rebuild final meta from all done
  markers.
- `--force-task`: ignore done markers for the named tasks and reprocess them
  using the existing plan. This supports targeted repair after code or mapping
  fixes.
- `--fail-fast`: abort on the first failed task. Without it, continue running
  other tasks and write `preprocess_failed_tasks.json`.

Worker execution should avoid `Pool.map()` for resumable mode because `map()`
re-raises the first worker exception and discards later task outcomes. Use a
result collection pattern that can capture success and failure per job, such as
`imap_unordered()` with wrapped task results.

## 8. Final Meta Rebuild

At the end of every run, the preprocessor rebuilds dataset-level metadata from
the stable plan and done markers:

```text
meta/tasks.jsonl
meta/episodes.jsonl
meta/info.json
meta/modality.json
meta/uamvla_aux_coverage.json
camera_params.json
```

If all tasks in the plan are done, the dataset is complete. If some tasks are
missing or failed, metadata still reflects completed episodes, but the run logs
and failed report clearly state that the suite is incomplete.

The rebuild must not renumber completed episodes. For missing tasks, their
planned episode indices remain reserved until the task is processed. This keeps
resumed outputs compatible with the original plan.

## 9. Hot-Path Optimization

The first render-path optimization is to remove the duplicate static RGB render:

```python
rgb_static = self._render_rgb(STATIC_CAM)
rgb_static_aligned = rgb_static
```

This is safe because `_render_rgb_aligned()` currently delegates directly to
`_render_rgb()` and applies no additional transform. The resulting arrays should
be treated as immutable or copied only when a downstream operation can mutate
them.

Additional safe caching:

- cache body name to body id;
- cache body name to geom ids;
- cache body name to instance id;
- cache camera id and intrinsics;
- cache writer directory paths for each episode.

These caches must not cache frame-dependent values such as body pose, camera
extrinsics, depth, masks, or point clouds.

## 10. Writer Optimization

The writer keeps the exact final layout. It may still reduce overhead by:

- precomputing per-episode directories in `_write_sidecars()`;
- creating each sidecar directory once per episode instead of once per frame;
- grouping depth, grounding, and affordance directory creation before loops;
- preserving existing `.npy`, `.json`, `.png`, `.mp4`, and `.parquet` paths.

This is intentionally less aggressive than changing to packed sidecar files.
The goal is compatibility first, speed second.

## 11. Profiling

With `--profile`, each task result includes:

```text
elapsed_sec
frames
frames_per_sec
retry_count
```

If implementation remains simple, optionally include coarse timings:

```text
replay_render_sec
write_sec
```

Profiling should be task-level or episode-level. Per-frame logging is too noisy
for full-suite runs and should not be enabled by default.

Example log shape:

```text
Task libero_10/... started worker=3 gpu=0 demos=50
Task libero_10/... done frames=12345 elapsed=1820.4s fps=6.78 retries=0
Suite libero_10 incomplete failed=1 done=9/10 report=meta/preprocess_failed_tasks.json
```

## 12. Recommended Remote Command

For a single RTX 4090, start conservatively:

```bash
CUDA_VISIBLE_DEVICES=0 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
python -u runners/preprocess_libero.py \
  --input-root /path/to/raw \
  --output-root /path/to/libero2uam \
  --suite all \
  --render-gpus 0 \
  --num-workers 4 \
  --debug-rgb-check-frames 0 \
  --resume \
  --profile \
  --max-retries 1 \
  2>&1 | tee logs/preprocess_libero_all_resume_w4.log
```

If the GPU stays underutilized and CPU/I/O pressure is acceptable, try
`--num-workers 5` or `--num-workers 6`. If EGL errors increase, return to 4.

## 13. Testing

Required tests:

- `--overwrite` and `--resume` are mutually exclusive.
- A saved plan is reused by `--resume`.
- A plan mismatch fails with a clear error.
- Existing done markers cause jobs to be skipped.
- `--force-task` reprocesses a done task.
- Non-fail-fast mode records a failed task and continues processing other jobs.
- Fail-fast mode propagates the first failure.
- Dataset meta rebuild uses the stable plan and does not renumber episodes.
- Duplicate static RGB render is removed; a fake worker verifies `_render_rgb`
  is called once for `STATIC_CAM` per frame payload.
- Writer directory creation optimization preserves all output paths.

Existing LIBERO tests should continue to pass:

```bash
python -m pytest \
  tests/test_libero_target_mapping.py \
  tests/test_libero_preprocessor_planning.py \
  tests/test_libero_preprocess_utils.py \
  tests/test_libero_lerobot_writer.py -q
```

## 14. Risks And Mitigations

Risk: a partial dataset could be mistaken for complete.

Mitigation: write explicit failed reports and only mark all tasks complete when
every task in the plan has a valid done marker.

Risk: resume could use stale indices after input data changes.

Mitigation: strict plan validation against task filenames, demo counts, frame
counts, and shape-affecting options.

Risk: reprocessing a forced task leaves stale files from an older failed run.

Mitigation: before a forced task starts, delete only that task's planned episode
outputs and sidecars, not the whole suite. The deletion list is derived from the
stable plan.

Risk: increasing workers overloads EGL or I/O.

Mitigation: keep worker count explicit, log task-level throughput, and recommend
4 workers as the first production setting on a single RTX 4090.

Risk: RGB reuse accidentally shares mutable arrays.

Mitigation: downstream code should treat rendered frames as immutable. If a
consumer mutates the aligned frame, it must copy locally before mutation.

## 15. Acceptance Criteria

The implementation is complete when:

- full output layout remains unchanged;
- resume can skip completed tasks and finish a suite after a previous task
  failure;
- failed tasks are reported without discarding completed outputs;
- `--force-task` can repair one task using stable episode indices;
- static RGB is rendered only once per frame payload;
- all relevant unit tests pass;
- a smoke preprocessing run produces the same sidecar paths as the current
  writer.

## 16. Implementation Notes

Implemented locally on branch `starVLA_dev`.

Primary verification:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_libero_resume.py \
  tests/test_libero_target_mapping.py \
  tests/test_libero_preprocessor_planning.py \
  tests/test_libero_preprocess_utils.py \
  tests/test_libero_lerobot_writer.py -q
```

CLI smoke verification:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python - <<'PY'
from runners.preprocess_libero import parse_args
args = parse_args([
    "--input-root", "datasets/libero",
    "--output-root", "datasets/libero2uam",
    "--suite", "libero_10",
    "--resume",
    "--profile",
    "--max-retries", "1",
])
assert args.resume is True
assert args.profile is True
assert args.max_retries == 1
print("preprocess_libero cli smoke ok")
PY
```
