# LIBERO Preprocessing Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable, profiled, single-GPU throughput mode for LIBERO preprocessing while keeping the final LeRobot + aux sidecar layout unchanged.

**Architecture:** Keep the existing `runners/preprocess_libero.py -> LiberoPreprocessor -> TaskJob -> _TaskReplayWorker -> LiberoLerobotWriter` pipeline. Add a small resume/marker helper module, integrate task-level resume scheduling in `LiberoPreprocessor.process()`, then make low-risk render and writer hot-path optimizations.

**Tech Stack:** Python 3.10, argparse, multiprocessing spawn pools, h5py, NumPy, pyarrow, PIL/imageio/OpenCV, pytest.

---

## File Structure

- Create `tools/preprocess/libero_resume.py`
  - Owns serializable preprocessing plans, done markers, failed reports, option fingerprints, and per-task cleanup helpers.
- Modify `runners/preprocess_libero.py`
  - Adds CLI flags: `--resume`, `--force-task`, `--fail-fast`, `--profile`, and `--max-retries`.
  - Enforces `--overwrite` and `--resume` mutual exclusion.
- Modify `tools/preprocess/libero_preprocessor.py`
  - Accepts new runtime options.
  - Creates or validates a stable plan.
  - Skips completed tasks under `--resume`.
  - Wraps task jobs so one failed task can be reported without losing other results.
  - Writes done markers and failed reports.
  - Removes duplicate static RGB rendering.
  - Adds safe body/geom/instance caches.
- Modify `tools/preprocess/libero_lerobot_writer.py`
  - Reduces repeated sidecar directory creation while preserving every output path.
- Create `tests/test_libero_resume.py`
  - Unit tests for plan serialization, validation, markers, failed reports, and force-task cleanup path calculation.
- Modify `tests/test_libero_preprocessor_planning.py`
  - CLI parsing and resume scheduling tests.
- Modify `tests/test_libero_preprocess_utils.py`
  - Keep existing utility coverage; no required changes unless shared helpers move.
- Modify `tests/test_libero_lerobot_writer.py`
  - Verify optimized writer still writes the same sidecar paths.

---

### Task 1: CLI Contract

**Files:**
- Modify: `runners/preprocess_libero.py`
- Test: `tests/test_libero_preprocessor_planning.py`

- [ ] **Step 1: Write the failing CLI parse test**

Add this test to `tests/test_libero_preprocessor_planning.py`:

```python
def test_parse_args_accepts_resume_throughput_options():
    args = parse_args(
        [
            "--input-root",
            "datasets/libero",
            "--output-root",
            "datasets/libero2uam",
            "--suite",
            "libero_10",
            "--num-workers",
            "4",
            "--render-gpus",
            "0",
            "--resume",
            "--force-task",
            "task_a",
            "--force-task",
            "task_b",
            "--fail-fast",
            "--profile",
            "--max-retries",
            "1",
        ]
    )
    assert args.resume is True
    assert args.force_task == ["task_a", "task_b"]
    assert args.fail_fast is True
    assert args.profile is True
    assert args.max_retries == 1
```

Add this second test:

```python
def test_parse_args_rejects_overwrite_with_resume():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input-root",
                "datasets/libero",
                "--output-root",
                "datasets/libero2uam",
                "--suite",
                "libero_goal",
                "--overwrite",
                "--resume",
            ]
        )
```

Also add `import pytest` at the top of the file.

- [ ] **Step 2: Run the CLI tests and verify failure**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_libero_preprocessor_planning.py::test_parse_args_accepts_resume_throughput_options \
  tests/test_libero_preprocessor_planning.py::test_parse_args_rejects_overwrite_with_resume -q
```

Expected: both tests fail because the new arguments do not exist yet.

- [ ] **Step 3: Implement argparse changes**

In `runners/preprocess_libero.py`, add arguments after `--overwrite`:

```python
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing suite output by skipping completed task markers.",
    )
    parser.add_argument(
        "--force-task",
        action="append",
        default=[],
        help="Task stem to reprocess even when a done marker exists. Repeatable.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the suite on the first task failure instead of recording it.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Log and store task-level throughput metrics.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Number of task-level retries before reporting failure.",
    )
```

At the end of `parse_args()`, before `return`, add:

```python
    args = parser.parse_args(argv)
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    return args
```

Replace the existing `return parser.parse_args(argv)` with that block.

- [ ] **Step 4: Pass options into `LiberoPreprocessor`**

In `runners/preprocess_libero.py`, extend the constructor call:

```python
        preprocessor = LiberoPreprocessor(
            suite=job.suite,
            num_workers=args.num_workers,
            render_gpus=args.render_gpus,
            min_segment_len=args.min_segment_len,
            active_target_score_window=args.active_target_score_window,
            debug_rgb_check_frames=args.debug_rgb_check_frames,
            max_tasks=args.max_tasks,
            max_demos_per_task=args.max_demos_per_task,
            max_frames_per_demo=args.max_frames_per_demo,
            resume=args.resume,
            force_tasks=tuple(args.force_task),
            fail_fast=args.fail_fast,
            profile=args.profile,
            max_retries=args.max_retries,
        )
```

The constructor will be updated in Task 3.

- [ ] **Step 5: Run the CLI tests and verify pass**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_libero_preprocessor_planning.py::test_parse_args_accepts_resume_throughput_options \
  tests/test_libero_preprocessor_planning.py::test_parse_args_rejects_overwrite_with_resume -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add runners/preprocess_libero.py tests/test_libero_preprocessor_planning.py
git commit -m "feat(libero): add resume throughput cli options"
```

---

### Task 2: Plan And Marker Helpers

**Files:**
- Create: `tools/preprocess/libero_resume.py`
- Create: `tests/test_libero_resume.py`

- [ ] **Step 1: Write failing tests for plan creation and validation**

Create `tests/test_libero_resume.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.preprocess.libero_preprocess_utils import compute_episode_plan
from tools.preprocess.libero_resume import (
    DoneMarker,
    PlanMismatchError,
    PreprocessOptions,
    TaskFailureRecord,
    build_preprocess_plan,
    done_marker_path,
    failed_tasks_path,
    load_done_markers,
    load_or_create_plan,
    planned_episode_paths,
    write_done_marker,
    write_failed_tasks,
)


def test_load_or_create_plan_writes_stable_episode_indices(tmp_path: Path):
    output_dir = tmp_path / "lerobot_libero_goal"
    frame_counts = {
        "task_a.hdf5": [3, 2],
        "task_b.hdf5": [4],
    }
    options = PreprocessOptions(
        suite="libero_goal",
        min_segment_len=3,
        active_target_score_window=8,
        max_tasks=None,
        max_demos_per_task=None,
        max_frames_per_demo=None,
    )

    plan = load_or_create_plan(output_dir, frame_counts, options, resume=False)

    assert plan.suite == "libero_goal"
    assert plan.frame_counts == frame_counts
    assert plan.episode_plan["task_a.hdf5"]["0"]["episode_index"] == 0
    assert plan.episode_plan["task_a.hdf5"]["1"]["row_start"] == 3
    assert (output_dir / "meta/preprocess_plan.json").exists()


def test_load_or_create_plan_rejects_mismatch_on_resume(tmp_path: Path):
    output_dir = tmp_path / "lerobot_libero_goal"
    options = PreprocessOptions(
        suite="libero_goal",
        min_segment_len=3,
        active_target_score_window=8,
        max_tasks=None,
        max_demos_per_task=None,
        max_frames_per_demo=None,
    )
    load_or_create_plan(output_dir, {"task_a.hdf5": [3]}, options, resume=False)

    with pytest.raises(PlanMismatchError, match="frame_counts"):
        load_or_create_plan(output_dir, {"task_a.hdf5": [4]}, options, resume=True)


def test_done_marker_round_trip(tmp_path: Path):
    marker = DoneMarker(
        task_stem="task_a",
        task_filename="task_a.hdf5",
        task_name="task a",
        task_index=0,
        demo_indices=[0, 1],
        episode_indices=[0, 1],
        episode_lengths={"0": 3, "1": 2},
        episode_to_task={"0": 0, "1": 0},
        frame_count=5,
        coverage={"depth": {"valid": 5, "total": 5}},
        camera_params={"width": 256},
        elapsed_sec=12.5,
        frames_per_sec=0.4,
        retry_count=1,
        options_hash="abc",
    )

    write_done_marker(tmp_path, marker)

    assert done_marker_path(tmp_path, "task_a").exists()
    loaded = load_done_markers(tmp_path)
    assert loaded["task_a"].frame_count == 5
    assert loaded["task_a"].retry_count == 1


def test_write_failed_tasks_replaces_report(tmp_path: Path):
    records = [
        TaskFailureRecord(
            task_stem="task_a",
            task_filename="task_a.hdf5",
            task_name="task a",
            exception_type="RuntimeError",
            message="boom",
            traceback="RuntimeError: boom",
            retry_count=1,
            gpu_id="0",
            worker_pid=123,
        )
    ]

    write_failed_tasks(tmp_path, records)

    payload = json.loads(failed_tasks_path(tmp_path).read_text())
    assert payload["failed_count"] == 1
    assert payload["failed_tasks"][0]["message"] == "boom"


def test_planned_episode_paths_for_force_task_cleanup(tmp_path: Path):
    frame_counts = {"task_a.hdf5": [3, 2]}
    options = PreprocessOptions(
        suite="libero_goal",
        min_segment_len=3,
        active_target_score_window=8,
        max_tasks=None,
        max_demos_per_task=None,
        max_frames_per_demo=None,
    )
    plan = build_preprocess_plan(frame_counts, options)

    paths = planned_episode_paths(tmp_path, plan, "task_a.hdf5")

    assert tmp_path / "data/chunk-000/episode_000000.parquet" in paths
    assert tmp_path / "videos/chunk-000/video.primary_image/episode_000001.mp4" in paths
    assert tmp_path / "point_clouds/1" in paths
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/test_libero_resume.py -q
```

Expected: import failure because `tools.preprocess.libero_resume` does not exist.

- [ ] **Step 3: Implement `tools/preprocess/libero_resume.py`**

Create `tools/preprocess/libero_resume.py` with:

```python
from __future__ import annotations

import hashlib
import json
import traceback as traceback_module
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.preprocess.libero_preprocess_utils import compute_episode_plan


PLAN_VERSION = 1


class PlanMismatchError(RuntimeError):
    """Raised when an existing resume plan does not match current inputs."""


@dataclass(frozen=True)
class PreprocessOptions:
    suite: str
    min_segment_len: int
    active_target_score_window: int
    max_tasks: int | None
    max_demos_per_task: int | None
    max_frames_per_demo: int | None

    def stable_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stable_hash(self) -> str:
        payload = json.dumps(self.stable_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class PreprocessPlan:
    version: int
    suite: str
    created_at: str
    options: dict[str, Any]
    options_hash: str
    frame_counts: dict[str, list[int]]
    episode_plan: dict[str, dict[str, dict[str, int]]]


@dataclass
class DoneMarker:
    task_stem: str
    task_filename: str
    task_name: str
    task_index: int
    demo_indices: list[int]
    episode_indices: list[int]
    episode_lengths: dict[str, int]
    episode_to_task: dict[str, int]
    frame_count: int
    coverage: dict[str, Any]
    camera_params: dict[str, Any] | None
    elapsed_sec: float
    frames_per_sec: float
    retry_count: int
    options_hash: str


@dataclass
class TaskFailureRecord:
    task_stem: str
    task_filename: str
    task_name: str
    exception_type: str
    message: str
    traceback: str
    retry_count: int
    gpu_id: str
    worker_pid: int | None


def plan_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "meta" / "preprocess_plan.json"


def failed_tasks_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "meta" / "preprocess_failed_tasks.json"


def done_marker_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "meta" / "preprocess_tasks"


def done_marker_path(output_dir: str | Path, task_stem: str) -> Path:
    return done_marker_dir(output_dir) / f"{task_stem}.done.json"


def build_preprocess_plan(
    frame_counts: dict[str, list[int]],
    options: PreprocessOptions,
) -> PreprocessPlan:
    raw_plan = compute_episode_plan(frame_counts)
    serial_plan: dict[str, dict[str, dict[str, int]]] = {}
    for (filename, demo_idx), item in raw_plan.items():
        serial_plan.setdefault(filename, {})[str(demo_idx)] = {
            "episode_index": int(item.episode_index),
            "row_start": int(item.row_start),
            "length": int(item.length),
        }
    return PreprocessPlan(
        version=PLAN_VERSION,
        suite=options.suite,
        created_at=datetime.now(timezone.utc).isoformat(),
        options=options.stable_dict(),
        options_hash=options.stable_hash(),
        frame_counts={k: [int(v) for v in vals] for k, vals in frame_counts.items()},
        episode_plan=serial_plan,
    )


def load_or_create_plan(
    output_dir: str | Path,
    frame_counts: dict[str, list[int]],
    options: PreprocessOptions,
    resume: bool,
) -> PreprocessPlan:
    path = plan_path(output_dir)
    if path.exists():
        plan = _plan_from_dict(json.loads(path.read_text()))
        _validate_plan(plan, frame_counts, options)
        return plan
    plan = build_preprocess_plan(frame_counts, options)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(plan), indent=2, sort_keys=True) + "\n")
    return plan


def _plan_from_dict(payload: dict[str, Any]) -> PreprocessPlan:
    return PreprocessPlan(
        version=int(payload["version"]),
        suite=str(payload["suite"]),
        created_at=str(payload["created_at"]),
        options=dict(payload["options"]),
        options_hash=str(payload["options_hash"]),
        frame_counts={
            str(k): [int(v) for v in vals]
            for k, vals in dict(payload["frame_counts"]).items()
        },
        episode_plan={
            str(filename): {
                str(demo_idx): {
                    "episode_index": int(item["episode_index"]),
                    "row_start": int(item["row_start"]),
                    "length": int(item["length"]),
                }
                for demo_idx, item in demos.items()
            }
            for filename, demos in dict(payload["episode_plan"]).items()
        },
    )


def _validate_plan(
    plan: PreprocessPlan,
    frame_counts: dict[str, list[int]],
    options: PreprocessOptions,
) -> None:
    if plan.version != PLAN_VERSION:
        raise PlanMismatchError(f"version mismatch: {plan.version} != {PLAN_VERSION}")
    if plan.suite != options.suite:
        raise PlanMismatchError(f"suite mismatch: {plan.suite} != {options.suite}")
    expected_counts = {k: [int(v) for v in vals] for k, vals in frame_counts.items()}
    if plan.frame_counts != expected_counts:
        raise PlanMismatchError("frame_counts mismatch")
    if plan.options != options.stable_dict():
        raise PlanMismatchError("options mismatch")


def write_done_marker(output_dir: str | Path, marker: DoneMarker) -> None:
    path = done_marker_path(output_dir, marker.task_stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(marker), indent=2, sort_keys=True) + "\n")


def load_done_markers(output_dir: str | Path) -> dict[str, DoneMarker]:
    root = done_marker_dir(output_dir)
    if not root.exists():
        return {}
    markers: dict[str, DoneMarker] = {}
    for path in sorted(root.glob("*.done.json")):
        payload = json.loads(path.read_text())
        marker = DoneMarker(**payload)
        markers[marker.task_stem] = marker
    return markers


def write_failed_tasks(
    output_dir: str | Path,
    records: list[TaskFailureRecord],
) -> None:
    path = failed_tasks_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "failed_count": len(records),
        "failed_tasks": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def traceback_text(exc: BaseException) -> str:
    return "".join(
        traceback_module.format_exception(type(exc), exc, exc.__traceback__)
    )


def planned_episode_paths(
    output_dir: str | Path,
    plan: PreprocessPlan,
    task_filename: str,
) -> set[Path]:
    output = Path(output_dir)
    paths: set[Path] = set()
    for item in plan.episode_plan.get(task_filename, {}).values():
        episode_index = int(item["episode_index"])
        chunk = episode_index // 1000
        ep_name = f"episode_{episode_index:06d}"
        paths.add(output / "data" / f"chunk-{chunk:03d}" / f"{ep_name}.parquet")
        paths.add(
            output
            / "videos"
            / f"chunk-{chunk:03d}"
            / "video.primary_image"
            / f"{ep_name}.mp4"
        )
        paths.add(
            output
            / "videos"
            / f"chunk-{chunk:03d}"
            / "video.wrist_image"
            / f"{ep_name}.mp4"
        )
        for root in (
            "image_targets",
            "point_clouds",
            "depths/static",
            "grounding_masks/static",
            "affordance_heatmaps/static",
        ):
            paths.add(output / root / str(episode_index))
    return paths
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/test_libero_resume.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add tools/preprocess/libero_resume.py tests/test_libero_resume.py
git commit -m "feat(libero): add resume plan markers"
```

---

### Task 3: Integrate Resume Planning In `LiberoPreprocessor`

**Files:**
- Modify: `tools/preprocess/libero_preprocessor.py`
- Test: `tests/test_libero_preprocessor_planning.py`

- [ ] **Step 1: Write failing constructor option test**

Add this test:

```python
def test_libero_preprocessor_stores_resume_options():
    from tools.preprocess.libero_preprocessor import LiberoPreprocessor

    pre = LiberoPreprocessor(
        suite="libero_10",
        resume=True,
        force_tasks=("task_a",),
        fail_fast=True,
        profile=True,
        max_retries=2,
    )

    assert pre.resume is True
    assert pre.force_tasks == {"task_a"}
    assert pre.fail_fast is True
    assert pre.profile is True
    assert pre.max_retries == 2
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_libero_preprocessor_planning.py::test_libero_preprocessor_stores_resume_options -q
```

Expected: fail because constructor does not accept the new keyword arguments.

- [ ] **Step 3: Extend `LiberoPreprocessor.__init__`**

In `tools/preprocess/libero_preprocessor.py`, extend the constructor signature:

```python
        resume: bool = False,
        force_tasks: tuple[str, ...] = (),
        fail_fast: bool = False,
        profile: bool = False,
        max_retries: int = 0,
```

Add these assignments:

```python
        self.resume = bool(resume)
        self.force_tasks = {str(task) for task in force_tasks}
        self.fail_fast = bool(fail_fast)
        self.profile = bool(profile)
        self.max_retries = max(0, int(max_retries))
```

- [ ] **Step 4: Run constructor test and verify pass**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_libero_preprocessor_planning.py::test_libero_preprocessor_stores_resume_options -q
```

Expected: `1 passed`.

- [ ] **Step 5: Write failing job-filter test**

Add this test:

```python
def test_resume_filters_done_tasks(tmp_path: Path):
    from tools.preprocess.libero_preprocessor import LiberoPreprocessor
    from tools.preprocess.libero_resume import DoneMarker, write_done_marker

    input_dir = tmp_path / "raw" / "libero_goal"
    output_dir = tmp_path / "out" / "lerobot_libero_goal"
    input_dir.mkdir(parents=True)
    for name in ["task_a.hdf5", "task_b.hdf5"]:
        (input_dir / name).write_bytes(b"hdf5 test bytes")

    marker = DoneMarker(
        task_stem="task_a",
        task_filename="task_a.hdf5",
        task_name="task a",
        task_index=0,
        demo_indices=[0],
        episode_indices=[0],
        episode_lengths={"0": 3},
        episode_to_task={"0": 0},
        frame_count=3,
        coverage={},
        camera_params=None,
        elapsed_sec=1.0,
        frames_per_sec=3.0,
        retry_count=0,
        options_hash="abc",
    )
    write_done_marker(output_dir, marker)

    pre = LiberoPreprocessor(suite="libero_goal", resume=True)
    jobs = [
        SimpleNamespace(hdf5_path=input_dir / "task_a.hdf5"),
        SimpleNamespace(hdf5_path=input_dir / "task_b.hdf5"),
    ]

    remaining = pre._filter_resume_jobs(output_dir, jobs, {"task_a": marker})

    assert [job.hdf5_path.name for job in remaining] == ["task_b.hdf5"]
```

Ensure `SimpleNamespace` is already imported from `types`; it is already present
in the file.

- [ ] **Step 6: Run job-filter test and verify failure**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_libero_preprocessor_planning.py::test_resume_filters_done_tasks -q
```

Expected: fail because `_filter_resume_jobs` does not exist.

- [ ] **Step 7: Implement `_filter_resume_jobs`**

Add this method to `LiberoPreprocessor`:

```python
    def _filter_resume_jobs(
        self,
        output_path: Path,
        jobs: list[TaskJob],
        done_markers: dict[str, Any],
    ) -> list[TaskJob]:
        if not self.resume:
            return jobs
        remaining = []
        for job in jobs:
            task_stem = job.hdf5_path.stem
            if task_stem.endswith("_demo"):
                task_stem = task_stem[: -len("_demo")]
            if task_stem in self.force_tasks:
                remaining.append(job)
                continue
            if task_stem in done_markers:
                logger.info("Skipping completed LIBERO task under --resume: %s", task_stem)
                continue
            remaining.append(job)
        return remaining
```

- [ ] **Step 8: Run planning tests and verify pass**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/test_libero_preprocessor_planning.py -q
```

Expected: all tests in this file pass.

- [ ] **Step 9: Commit**

```bash
git add tools/preprocess/libero_preprocessor.py tests/test_libero_preprocessor_planning.py
git commit -m "feat(libero): wire resume options into preprocessor"
```

---

### Task 4: Resumable Scheduler, Failure Reports, And Markers

**Files:**
- Modify: `tools/preprocess/libero_preprocessor.py`
- Test: `tests/test_libero_preprocessor_planning.py`

- [ ] **Step 1: Write failing safe-result test**

Add this test:

```python
def test_run_task_job_with_retries_reports_failure(monkeypatch):
    from tools.preprocess.libero_preprocessor import _run_task_job_with_retries

    job = SimpleNamespace(
        hdf5_path=Path("bad_task_demo.hdf5"),
        gpu_id="0",
    )

    def boom(_job):
        raise RuntimeError("bad target")

    monkeypatch.setattr("tools.preprocess.libero_preprocessor._run_task_job", boom)

    result = _run_task_job_with_retries(job, max_retries=1)

    assert result["ok"] is False
    assert result["retry_count"] == 1
    assert result["task_filename"] == "bad_task_demo.hdf5"
    assert result["exception_type"] == "RuntimeError"
    assert "bad target" in result["message"]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_libero_preprocessor_planning.py::test_run_task_job_with_retries_reports_failure -q
```

Expected: fail because `_run_task_job_with_retries` does not exist.

- [ ] **Step 3: Implement retry wrapper**

In `tools/preprocess/libero_preprocessor.py`, import:

```python
import time
```

and from `tools.preprocess.libero_resume` import:

```python
from tools.preprocess.libero_resume import traceback_text
```

Add a module-level helper after `_run_task_job`:

```python
def _run_task_job_with_retries(job: TaskJob, max_retries: int = 0) -> dict[str, Any]:
    started = time.perf_counter()
    attempts = max(0, int(max_retries)) + 1
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            result = _run_task_job(job)
            elapsed = time.perf_counter() - started
            result = dict(result)
            result["ok"] = True
            result["elapsed_sec"] = float(elapsed)
            result["frames_per_sec"] = (
                float(result.get("total_frames", 0)) / elapsed
                if elapsed > 0
                else 0.0
            )
            result["retry_count"] = attempt
            result["task_filename"] = job.hdf5_path.name
            result["task_stem"] = _task_stem(job.hdf5_path)
            result["gpu_id"] = job.gpu_id
            result["worker_pid"] = os.getpid()
            return result
        except BaseException as exc:
            last_exc = exc
            logger.exception(
                "LIBERO task failed attempt %d/%d: %s",
                attempt + 1,
                attempts,
                job.hdf5_path.name,
            )
    assert last_exc is not None
    elapsed = time.perf_counter() - started
    return {
        "ok": False,
        "task_name": task_name_from_hdf5(job.hdf5_path),
        "task_filename": job.hdf5_path.name,
        "task_stem": _task_stem(job.hdf5_path),
        "exception_type": type(last_exc).__name__,
        "message": str(last_exc),
        "traceback": traceback_text(last_exc),
        "retry_count": attempts - 1,
        "gpu_id": job.gpu_id,
        "worker_pid": os.getpid(),
        "elapsed_sec": float(elapsed),
    }
```

Add helper:

```python
def _task_stem(path: Path) -> str:
    stem = Path(path).stem
    return stem[: -len("_demo")] if stem.endswith("_demo") else stem
```

- [ ] **Step 4: Run safe-result test and verify pass**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_libero_preprocessor_planning.py::test_run_task_job_with_retries_reports_failure -q
```

Expected: `1 passed`.

- [ ] **Step 5: Integrate plan, markers, and failure reports in `process()`**

In `LiberoPreprocessor.process()`, after `frame_counts` is computed, build
`PreprocessOptions` and load or create the plan:

```python
        options = PreprocessOptions(
            suite=self.suite,
            min_segment_len=self.min_segment_len,
            active_target_score_window=self.active_target_score_window,
            max_tasks=self.max_tasks,
            max_demos_per_task=self.max_demos_per_task,
            max_frames_per_demo=self.max_frames_per_demo,
        )
        plan = load_or_create_plan(
            output_path,
            frame_counts,
            options,
            resume=self.resume,
        )
        done_markers = load_done_markers(output_path)
```

Import these names:

```python
from tools.preprocess.libero_resume import (
    DoneMarker,
    PreprocessOptions,
    TaskFailureRecord,
    load_done_markers,
    load_or_create_plan,
    write_done_marker,
    write_failed_tasks,
)
```

After jobs are built, filter them:

```python
        jobs_to_run = self._filter_resume_jobs(output_path, jobs, done_markers)
```

Replace worker execution with `_run_task_job_with_retries` and collect
successful and failed results. For multiprocessing, use:

```python
            with ctx.Pool(processes=worker_count) as pool:
                async_results = [
                    pool.apply_async(
                        _run_task_job_with_retries,
                        (job, self.max_retries),
                    )
                    for job in jobs_to_run
                ]
                results = [item.get() for item in async_results]
```

For single worker:

```python
            results = [
                _run_task_job_with_retries(job, self.max_retries)
                for job in jobs_to_run
            ]
```

Split results:

```python
        failures = [r for r in results if not r.get("ok")]
        successes = [r for r in results if r.get("ok")]
        if failures and self.fail_fast:
            first = failures[0]
            raise RuntimeError(
                f"LIBERO task failed: {first['task_filename']}: {first['message']}"
            )
```

For each successful result, write a done marker using its episodes and coverage.
For failures, write `TaskFailureRecord` entries with `write_failed_tasks()`.

- [ ] **Step 6: Rebuild metadata from done markers and successes**

Keep the existing final meta writing path for now, but ensure it includes
results from both newly successful tasks and already loaded done markers. The
minimum acceptable implementation is:

```python
        all_results = successes
        for marker in done_markers.values():
            if marker.task_stem in {r["task_stem"] for r in successes}:
                continue
            all_results.append(_result_from_done_marker(marker))
```

Add `_result_from_done_marker(marker: DoneMarker) -> dict[str, Any]` near the
retry helper:

```python
def _result_from_done_marker(marker: DoneMarker) -> dict[str, Any]:
    return {
        "ok": True,
        "task_name": marker.task_name,
        "total_frames": marker.frame_count,
        "episode_lengths": marker.episode_lengths,
        "episode_to_task": marker.episode_to_task,
        "coverage": marker.coverage,
        "camera_params": marker.camera_params,
        "task_filename": marker.task_filename,
        "task_stem": marker.task_stem,
        "elapsed_sec": marker.elapsed_sec,
        "frames_per_sec": marker.frames_per_sec,
        "retry_count": marker.retry_count,
    }
```

- [ ] **Step 7: Run planning and resume tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_libero_resume.py \
  tests/test_libero_preprocessor_planning.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add tools/preprocess/libero_preprocessor.py tools/preprocess/libero_resume.py tests/test_libero_resume.py tests/test_libero_preprocessor_planning.py
git commit -m "feat(libero): resume failed preprocessing tasks"
```

---

### Task 5: Remove Duplicate Static RGB Render And Add Caches

**Files:**
- Modify: `tools/preprocess/libero_preprocessor.py`
- Test: `tests/test_libero_preprocessor_planning.py`

- [ ] **Step 1: Write failing render reuse test**

Add this test:

```python
def test_extract_frame_payload_reuses_static_rgb_render(monkeypatch):
    worker = object.__new__(_TaskReplayWorker)
    worker._candidate_bodies = []
    worker._fallback_body = "target"
    worker._intrinsics = {"agentview": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}}

    class SimData:
        cam_xpos = [np.zeros(3)]
        cam_xmat = [np.eye(3).reshape(-1)]

    class SimModel:
        def camera_name2id(self, _name):
            return 0

    class Sim:
        model = SimModel()
        data = SimData()

        def set_state_from_flattened(self, _state):
            return None

        def forward(self):
            return None

    worker.env = SimpleNamespace(sim=Sim())
    calls = []

    def fake_render_rgb(camera_name):
        calls.append(camera_name)
        return np.zeros((4, 4, 3), dtype=np.uint8)

    worker._render_rgb = fake_render_rgb
    worker._render_depth = lambda _camera: np.ones((4, 4), dtype=np.float32)
    worker._render_segmentation_instance = lambda _camera: np.zeros((4, 4), dtype=np.int32)
    worker._render_segmentation_geom = lambda _camera: np.zeros((4, 4), dtype=np.int32)

    payload = worker._extract_frame_payload(
        state=np.zeros(4, dtype=np.float32),
        future_tcp_positions=np.zeros((1, 3), dtype=np.float32),
    )

    assert calls.count("agentview") == 1
    assert calls.count("robot0_eye_in_hand") == 1
    assert payload["rgb_static"] is payload["rgb_static_aligned"]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_libero_preprocessor_planning.py::test_extract_frame_payload_reuses_static_rgb_render -q
```

Expected: fail because `agentview` is rendered twice.

- [ ] **Step 3: Reuse static RGB in `_extract_frame_payload()`**

Change:

```python
        rgb_static = self._render_rgb(STATIC_CAM)
        rgb_static_aligned = self._render_rgb_aligned(STATIC_CAM)
```

to:

```python
        rgb_static = self._render_rgb(STATIC_CAM)
        rgb_static_aligned = rgb_static
```

Leave `_render_rgb_aligned()` in place for compatibility until no callers use
it.

- [ ] **Step 4: Add body/geom/instance caches**

In `_TaskReplayWorker.__init__`, add:

```python
        self._body_id_cache: dict[str, int] = {}
        self._geom_ids_cache: dict[str, list[int]] = {}
        self._instance_id_cache: dict[str, int | None] = {}
```

Modify `_body_payload()`:

```python
        body_id = self._body_id(body_name)
```

Add:

```python
    def _body_id(self, body_name: str) -> int:
        if body_name not in self._body_id_cache:
            self._body_id_cache[body_name] = int(self.env.sim.model.body_name2id(body_name))
        return self._body_id_cache[body_name]
```

Modify `_geom_ids_for_body()`:

```python
    def _geom_ids_for_body(self, body_name: str) -> list[int]:
        if body_name in self._geom_ids_cache:
            return self._geom_ids_cache[body_name]
        body_id = self._body_id(body_name)
        geom_bodyids = self.env.sim.model.geom_bodyid
        ids = [i for i in range(len(geom_bodyids)) if int(geom_bodyids[i]) == body_id]
        self._geom_ids_cache[body_name] = ids
        return ids
```

Modify `_instance_id_for_body()`:

```python
    def _instance_id_for_body(self, body_name: str) -> int | None:
        if body_name in self._instance_id_cache:
            return self._instance_id_cache[body_name]
        instance_name = self._instance_name_for_body(body_name)
        if instance_name is None:
            self._instance_id_cache[body_name] = None
            return None
        instance_names = list(self.env.env.model.instances_to_ids.keys())
        value = instance_names.index(instance_name) + 1
        self._instance_id_cache[body_name] = value
        return value
```

- [ ] **Step 5: Run render/caching tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_libero_preprocessor_planning.py::test_extract_frame_payload_reuses_static_rgb_render \
  tests/test_libero_preprocessor_planning.py::test_replay_worker_maps_main_body_to_instance_id -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add tools/preprocess/libero_preprocessor.py tests/test_libero_preprocessor_planning.py
git commit -m "perf(libero): avoid duplicate static rgb render"
```

---

### Task 6: Writer Directory Optimization

**Files:**
- Modify: `tools/preprocess/libero_lerobot_writer.py`
- Test: `tests/test_libero_lerobot_writer.py`

- [ ] **Step 1: Inspect existing writer tests**

Run:

```bash
sed -n '1,240p' tests/test_libero_lerobot_writer.py
```

Use the existing buffer fixtures and assertions instead of inventing a second
writer fixture.

- [ ] **Step 2: Add sidecar path preservation test**

Add a test that creates a tiny `LiberoEpisodeBuffers`, writes it, and asserts
the existing sidecar paths:

```python
def test_writer_preserves_aux_sidecar_paths_after_directory_optimization(tmp_path: Path):
    writer = LiberoLerobotWriter(tmp_path, fps=10)
    buffers = LiberoEpisodeBuffers(
        episode_index=7,
        task_index=0,
        task_name="task",
        rows=[
            {
                "episode_index": 7,
                "frame_index": 0,
                "timestamp": 0.0,
                "index": 0,
                "task_index": 0,
            }
        ],
        primary_frames=[np.zeros((8, 8, 3), dtype=np.uint8)],
        wrist_frames=[np.zeros((8, 8, 3), dtype=np.uint8)],
        image_targets=[np.zeros((4, 4, 3), dtype=np.uint8)],
        point_clouds=[np.zeros((1024, 3), dtype=np.float32)],
        depth_targets=[np.ones((8, 8), dtype=np.float32)],
        grounding_masks=[np.ones((1, 20, 20), dtype=np.float32)],
        grounding_levels=["object"],
        affordance_heatmaps=[np.ones((1, 20, 20), dtype=np.float32)],
    )

    writer.write_episode(buffers)

    assert (tmp_path / "image_targets/7/0.png").exists()
    assert (tmp_path / "point_clouds/7/0.npy").exists()
    assert (tmp_path / "depths/static/7/0.npy").exists()
    assert (tmp_path / "grounding_masks/static/7/0.npy").exists()
    assert (tmp_path / "grounding_masks/static/7/0.json").exists()
    assert (tmp_path / "affordance_heatmaps/static/7/0.npy").exists()
```

- [ ] **Step 3: Run writer tests before optimization**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/test_libero_lerobot_writer.py -q
```

Expected: tests pass before optimization, proving path behavior is captured.

- [ ] **Step 4: Optimize `_write_sidecars()` directory creation**

In `tools/preprocess/libero_lerobot_writer.py`, rewrite the start of
`_write_sidecars()`:

```python
    def _write_sidecars(self, buffers: LiberoEpisodeBuffers) -> None:
        self._validate_optional_lengths(buffers)
        episode = str(buffers.episode_index)
        image_dir = self.output_dir / "image_targets" / episode
        pc_dir = self.output_dir / "point_clouds" / episode
        depth_dir = self.output_dir / "depths" / "static" / episode
        grounding_dir = self.output_dir / "grounding_masks" / "static" / episode
        affordance_dir = self.output_dir / "affordance_heatmaps" / "static" / episode

        if any(target is not None for target in buffers.image_targets):
            image_dir.mkdir(parents=True, exist_ok=True)
        if any(pc is not None for pc in buffers.point_clouds):
            pc_dir.mkdir(parents=True, exist_ok=True)
        if any(depth is not None for depth in buffers.depth_targets):
            depth_dir.mkdir(parents=True, exist_ok=True)
        if any(mask is not None for mask in buffers.grounding_masks):
            grounding_dir.mkdir(parents=True, exist_ok=True)
        if any(heatmap is not None for heatmap in buffers.affordance_heatmaps):
            affordance_dir.mkdir(parents=True, exist_ok=True)
```

Then update the loops to write directly to those directories instead of calling
`write_aux_denoising_sidecar()` for depth, grounding, and affordance. Preserve
the same dtype casts and JSON payload:

```python
            np.save(depth_dir / f"{frame_idx}.npy", np.asarray(depth, dtype=np.float32))
```

```python
            np.save(grounding_dir / f"{frame_idx}.npy", np.asarray(mask, dtype=np.float32))
            with open(grounding_dir / f"{frame_idx}.json", "w") as f:
                json.dump({"grounding_level": buffers.grounding_levels[frame_idx] or "object"}, f)
```

```python
            np.save(affordance_dir / f"{frame_idx}.npy", np.asarray(heatmap, dtype=np.float32))
```

Keep `write_aux_denoising_sidecar()` unchanged because CALVIN and tests may call
it directly.

- [ ] **Step 5: Run writer tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/test_libero_lerobot_writer.py -q
```

Expected: all writer tests pass.

- [ ] **Step 6: Commit**

```bash
git add tools/preprocess/libero_lerobot_writer.py tests/test_libero_lerobot_writer.py
git commit -m "perf(libero): reduce sidecar directory churn"
```

---

### Task 7: Final Verification And Remote Command Update

**Files:**
- Modify: `docs/superpowers/specs/2026-05-22-libero-preprocess-throughput-design.md`
- Test: existing LIBERO-related tests

- [ ] **Step 1: Run the full LIBERO preprocessing test subset**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_libero_resume.py \
  tests/test_libero_target_mapping.py \
  tests/test_libero_preprocessor_planning.py \
  tests/test_libero_preprocess_utils.py \
  tests/test_libero_lerobot_writer.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run import smoke for runner**

Run:

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

Expected output includes `preprocess_libero cli smoke ok`.

- [ ] **Step 3: Update the spec implementation status**

In `docs/superpowers/specs/2026-05-22-libero-preprocess-throughput-design.md`,
change:

```text
Status: approved design, awaiting implementation plan
```

to:

```text
Status: implemented locally
```

Append a section:

```markdown
## 16. Implementation Notes

Implemented in local branch `starVLA_dev`.

Primary verification:

```bash
python -m pytest \
  tests/test_libero_resume.py \
  tests/test_libero_target_mapping.py \
  tests/test_libero_preprocessor_planning.py \
  tests/test_libero_preprocess_utils.py \
  tests/test_libero_lerobot_writer.py -q
```
```

- [ ] **Step 4: Commit final docs**

```bash
git add docs/superpowers/specs/2026-05-22-libero-preprocess-throughput-design.md
git commit -m "docs(libero): record throughput implementation notes"
```

- [ ] **Step 5: Provide final remote command**

Report this command to the user:

```bash
CUDA_VISIBLE_DEVICES=0 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
python -u runners/preprocess_libero.py \
  --input-root /inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UamVLA/datasets/libero2uam/raw \
  --output-root /inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UamVLA/datasets/libero2uam \
  --suite all \
  --render-gpus 0 \
  --num-workers 4 \
  --debug-rgb-check-frames 0 \
  --resume \
  --profile \
  --max-retries 1 \
  2>&1 | tee logs/preprocess_libero_all_resume_w4.log
```

---

## Self-Review Checklist

- Spec coverage:
  - Stable plan: Tasks 2, 4.
  - Done markers and failed reports: Tasks 2, 4.
  - CLI behavior: Task 1.
  - Resume scheduling and `force-task`: Tasks 3, 4.
  - Duplicate static RGB render: Task 5.
  - Writer directory optimization: Task 6.
  - Profiling and verification: Tasks 4, 7.
- Placeholder scan:
  - This plan contains no incomplete work markers and no unspecified test steps.
- Type consistency:
  - `PreprocessOptions`, `PreprocessPlan`, `DoneMarker`, and
    `TaskFailureRecord` are introduced in Task 2 and reused by subsequent tasks.
  - `resume`, `force_tasks`, `fail_fast`, `profile`, and `max_retries` are added
    to the CLI in Task 1 and to `LiberoPreprocessor` in Task 3.
