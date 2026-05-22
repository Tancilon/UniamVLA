# CALVIN Multiscene Preprocessing Throughput Optimization

Date: 2026-05-22
Status: implemented locally
Implementation: `runners/preprocess_calvin_multiscene.py` and `tools/preprocess/calvin_multiscene_resume.py`
Related design: `docs/superpowers/specs/2026-05-22-calvin-aux-sidecars-abcd-design.md`

## 1. Purpose

The CALVIN preprocessing path now produces the full UamVLA-GR00T auxiliary
sidecar contract. The remaining pain point for official `task_ABC_D` and
`task_ABCD_D` preprocessing is throughput and recoverability: the current
multiscene runner preprocesses scenes serially and has no scene-level resume
state.

This design accelerates multiscene CALVIN preprocessing while preserving the
exact final LeRobot + aux sidecar output layout. Training code, dataloaders,
single-scene preprocessor semantics, and merger output semantics must not
change.

## 2. Scope

In scope:

- `runners/preprocess_calvin_multiscene.py`;
- official multi-scene CALVIN splits such as `task_ABC_D/training` and
  `task_ABCD_D/training`;
- scene-level subprocess parallelism;
- scene-level done markers, failed reports, resume, forced scene reruns, and
  retry/profile logging;
- final merge only after all requested scenes are complete.

Out of scope:

- changing `tools/preprocess/calvin_preprocessor_lerobot.py` internals;
- language-window-level multiprocessing inside a single scene;
- changing final LeRobot parquet/video/sidecar paths;
- changing `tools/preprocess/calvin_lerobot_merger.py` output semantics;
- changing model or dataloader sidecar lookup behavior.

## 3. Existing Flow

The current multiscene runner does:

```text
split_calvin_by_scene(input_dir, work_dir/split)
  -> for scene in scenes:
       subprocess.run(runners/preprocess_calvin.py scene)
  -> merge_lerobot_scene_outputs(scene_outputs, output_dir)
```

This is safe because each scene subprocess owns a fresh PyBullet/EGL state and a
scene-specific Hydra config. It is also slow because scenes A/B/C/D run
serially even though they are independent after splitting.

The single-scene preprocessor currently ignores `num_workers > 1` and runs
single-process by design. This design keeps that behavior.

## 4. Proposed Flow

The new multiscene flow is:

```text
split or reuse per-scene virtual splits
  -> schedule scene preprocess subprocesses with --scene-workers
  -> write per-scene done markers and failed report under work_dir/meta
  -> merge only when every requested scene has a done marker
```

Each scene still invokes:

```text
runners/preprocess_calvin.py
  --input_dir work_dir/split/<scene>
  --output_dir work_dir/preprocessed/<scene>
  --default_scene <scene>
```

The key change is that `A`, `B`, `C`, and `D` can run concurrently as separate
subprocesses. No in-process PyBullet sharing is introduced.

## 5. CLI

Add these arguments to `runners/preprocess_calvin_multiscene.py`:

```text
--scene-workers N
--resume
--force-scene A
--max-retries K
--profile
--fail-fast
```

Rules:

- `--scene-workers` controls the number of concurrent scene subprocesses.
- `--num_workers` is still forwarded to each single-scene subprocess. It should
  remain `1` for the current CALVIN preprocessor.
- `--overwrite` and `--resume` are mutually exclusive.
- `--overwrite` starts from scratch by cleaning split work, preprocessed work,
  scene markers, failed reports, and final output.
- `--resume` reuses `work_dir/split` and `work_dir/preprocessed`, skips scenes
  with valid done markers, and reruns incomplete scenes.
- `--force-scene` is repeatable and reruns a scene even if a done marker exists.
- `--fail-fast` stops after the first failed scene. Without it, remaining scenes
  continue and the runner writes a failed report.
- `--max-retries` retries a failed scene subprocess before marking it failed.
- `--profile` records elapsed seconds, return code, attempt count, and simple
  output statistics for each scene.

## 6. Markers And Reports

Scene done markers live under:

```text
<work_dir>/meta/preprocess_scenes/<scene>.done.json
```

Each marker records:

- scene letter;
- scene input dir;
- scene output dir;
- command argv;
- return code;
- elapsed seconds;
- retry count;
- number of emitted parquet episode files;
- number of emitted primary videos;
- number of sidecar directories for image targets, point clouds, depth,
  grounding, and affordance.

Failed scene reports live at:

```text
<work_dir>/meta/preprocess_failed_scenes.json
```

Each failure record contains:

- scene letter;
- command argv;
- return code if a subprocess was launched;
- exception type and message for launch-time errors;
- elapsed seconds;
- retry count.

The failed report describes the latest run state and is overwritten on each
multiscene invocation.

## 7. Resume Semantics

Resume is scene-level:

```text
if marker exists and scene not in --force-scene:
    skip scene preprocess subprocess
else:
    run scene preprocess subprocess
```

Resume does not rewrite or inspect individual language windows. The
single-scene preprocessor remains responsible for producing complete per-scene
LeRobot outputs. If a scene output is suspected stale or corrupt, use
`--force-scene <scene>`.

If all requested scenes have valid done markers after the run, the final merge
executes. If any scene is failed or missing, merge is skipped and the failure
report points to the scenes that need repair.

## 8. Overwrite Semantics

`--overwrite` means start from zero:

- remove `work_dir/split`;
- remove `work_dir/preprocessed`;
- remove `work_dir/meta/preprocess_scenes`;
- remove `work_dir/meta/preprocess_failed_scenes.json`;
- allow final `output_dir` replacement during merge.

Because the existing runner already deletes stale split dirs and per-scene
preprocessed dirs, the implementation should keep that behavior in overwrite
mode and make resume mode explicitly preserve work dirs.

## 9. Merge Semantics

The final merge remains:

```python
merge_lerobot_scene_outputs(
    scene_output_dirs,
    output_dir,
    overwrite=args.overwrite or args.resume,
    skip_stats=args.skip_stats,
    robot_type=args.robot_type,
    action_mode=args.action_mode,
)
```

In `--resume`, final `output_dir` may already exist from a prior partial or full
run. Once all scenes are done, the runner may safely re-merge with
`overwrite=True` because final output is derived from completed per-scene
outputs.

If not all scenes are done, the merge must not run. This avoids presenting a
partial final dataset as complete.

## 10. Recommended Remote Commands

For `task_ABC_D` on one GPU:

```bash
UAMVLA_CALVIN_NATIVE_SAFE_EXIT=1 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
python -u runners/preprocess_calvin_multiscene.py \
  --input_dir /path/to/task_ABC_D/training \
  --work_dir /path/to/calvin2uam/work_calvin_abc \
  --output_dir /path/to/calvin2uam/lerobot_calvin_abc \
  --scenes A,B,C \
  --scene-workers 3 \
  --num_workers 1 \
  --on_missing_target skip \
  --resume \
  --profile \
  --max-retries 1 \
  2>&1 | tee logs/preprocess_calvin_abc_resume_scene3.log
```

For `task_ABCD_D`, use:

```text
--scenes A,B,C,D
--scene-workers 4
```

If EGL contention appears, reduce `--scene-workers` to `2`.

## 11. Testing

Required tests:

- CLI parses `--scene-workers`, `--resume`, `--force-scene`, `--max-retries`,
  `--profile`, and `--fail-fast`.
- `--overwrite` and `--resume` are mutually exclusive.
- done scenes are skipped under `--resume`.
- `--force-scene` reruns a done scene.
- scene subprocess failures are recorded.
- without `--fail-fast`, one failed scene does not prevent other scheduled
  scenes from running.
- merge is skipped when any requested scene is failed or missing.
- merge runs when all requested scenes have done markers.
- existing output path guard tests continue to pass.

Suggested verification:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_preprocess_calvin_multiscene.py \
  tests/test_calvin_lerobot_merger.py \
  tests/test_calvin_scene_splitter.py -q
```

## 12. Risks And Mitigations

Risk: multiple scene subprocesses contend for the same GPU/EGL context.

Mitigation: keep concurrency explicit through `--scene-workers`; recommend 3
for `task_ABC_D`, 4 for `task_ABCD_D`, and 2 if EGL instability appears.

Risk: final output exists but scenes are incomplete.

Mitigation: merge only when all requested scenes have done markers; otherwise
write a failed report and leave final output untouched.

Risk: resume skips a stale scene output.

Mitigation: provide repeatable `--force-scene`; done markers record command argv
and output statistics for manual inspection.

Risk: subprocess logs interleave.

Mitigation: main process logs scene start/end/failure; optional future work can
add per-scene log files, but this design does not require changing log layout.

## 13. Acceptance Criteria

The implementation is complete when:

- multiscene preprocessing can run scenes concurrently;
- final output layout is unchanged;
- resume skips completed scenes and reruns incomplete scenes;
- failed scene reports are written;
- final merge only runs when all requested scenes complete;
- all listed tests pass.
