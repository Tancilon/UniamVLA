# CALVIN Multiscene Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add scene-level parallelism, resume markers, failure reports, retries, and profile logging to official CALVIN multi-scene preprocessing while preserving the exact final LeRobot + UamVLA auxiliary sidecar layout.

**Architecture:** Keep `tools/preprocess/calvin_preprocessor_lerobot.py` and merged dataset semantics unchanged. Extend `runners/preprocess_calvin_multiscene.py` so each scene still runs in its own `runners/preprocess_calvin.py` subprocess, but subprocesses are scheduled concurrently and tracked with scene-level JSON markers under `work_dir/meta`. Add a small helper module for marker/report IO and output summaries, mirroring the LIBERO resume style without sharing MuJoCo/PyBullet state across scenes.

**Tech Stack:** Python stdlib `argparse`, `concurrent.futures.ThreadPoolExecutor`, `dataclasses`, `json`, `pathlib`, `subprocess`, `time`; existing CALVIN splitter, single-scene preprocessor runner, LeRobot merger, and pytest tests.

---

## File Structure

- Modify `runners/preprocess_calvin_multiscene.py`
  - Add throughput CLI flags.
  - Add command builder, retry wrapper, scene scheduling, resume/force-scene handling, failure report writing, and merge gating.
  - Keep final call to `merge_lerobot_scene_outputs(...)` semantically identical except `overwrite=args.overwrite or args.resume`.
- Create `tools/preprocess/calvin_multiscene_resume.py`
  - Own scene marker paths, marker dataclasses, failed report dataclasses, JSON read/write, output summary counts, and marker removal.
- Modify `tests/test_preprocess_calvin_multiscene.py`
  - Extend runner tests for parser flags, overwrite/resume exclusivity, resume skip, forced rerun, failure report, merge skip, and parallel scheduling.
- Create `tests/test_calvin_multiscene_resume.py`
  - Unit-test marker/report serialization and scene output summary counts.
- Leave unchanged:
  - `tools/preprocess/calvin_preprocessor_lerobot.py`
  - `tools/preprocess/calvin_lerobot_merger.py`
  - `tools/preprocess/calvin_scene_splitter.py`
  - Dataset layout and aux sidecar layout.

## CLI Contract

The completed runner accepts these new flags:

```text
--scene-workers N
--resume
--force-scene A
--max-retries K
--profile
--fail-fast
```

Runtime rules:

- `--scene-workers` is parent-runner scene subprocess concurrency.
- `--num_workers` continues to be forwarded to each single-scene preprocessor subprocess.
- `--overwrite` and `--resume` are mutually exclusive.
- `--overwrite` clears split work, preprocessed work, scene markers, failed report, and allows final output replacement.
- `--resume` preserves split work and preprocessed work, skips scenes with done markers, and allows the final output directory to exist.
- `--force-scene` is repeatable; each listed scene reruns even when its marker exists.
- `--fail-fast` stops submitting new scene jobs after the first failed scene result. Already-running scene subprocesses are allowed to finish.
- `--max-retries K` means one initial attempt plus `K` retries.
- `--profile` logs per-scene elapsed seconds, attempt count, return code, and summary counts.

---

### Task 1: Add CLI Tests And Parser Flags

**Files:**
- Modify: `tests/test_preprocess_calvin_multiscene.py`
- Modify: `runners/preprocess_calvin_multiscene.py`

- [ ] **Step 1: Write failing parser tests**

Append these tests to `tests/test_preprocess_calvin_multiscene.py`:

```python
def test_parse_args_accepts_scene_throughput_flags(tmp_path):
    args = runner.parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--work_dir",
            str(tmp_path / "work"),
            "--output_dir",
            str(tmp_path / "output"),
            "--scenes",
            "A,B,C,D",
            "--scene-workers",
            "3",
            "--resume",
            "--force-scene",
            "B",
            "--force-scene",
            "D",
            "--max-retries",
            "2",
            "--profile",
            "--fail-fast",
        ]
    )

    assert args.scene_workers == 3
    assert args.resume is True
    assert args.force_scene == ["B", "D"]
    assert args.max_retries == 2
    assert args.profile is True
    assert args.fail_fast is True


def test_parse_args_rejects_overwrite_with_resume(tmp_path):
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--work_dir",
                str(tmp_path / "work"),
                "--output_dir",
                str(tmp_path / "output"),
                "--overwrite",
                "--resume",
            ]
        )
```

- [ ] **Step 2: Run parser tests and confirm they fail**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_preprocess_calvin_multiscene.py::test_parse_args_accepts_scene_throughput_flags \
  tests/test_preprocess_calvin_multiscene.py::test_parse_args_rejects_overwrite_with_resume -q
```

Expected: first test fails because `args.scene_workers` does not exist, and the second test fails because `--overwrite --resume` is accepted.

- [ ] **Step 3: Implement parser flags**

In `runners/preprocess_calvin_multiscene.py`, replace the standalone `--overwrite` argument with a mutually exclusive group and add the new flags after `--num_workers`:

```python
    p.add_argument("--num_workers", "--num-workers", type=int, default=1,
                   dest="num_workers")
    p.add_argument("--scene_workers", "--scene-workers", type=int, default=1,
                   dest="scene_workers",
                   help="Number of scene subprocesses to run concurrently.")
    p.add_argument("--force_scene", "--force-scene", action="append", default=[],
                   dest="force_scene",
                   help="Scene letter to rerun even when a resume marker exists. Repeatable.")
    p.add_argument("--max_retries", "--max-retries", type=int, default=0,
                   dest="max_retries",
                   help="Number of retries after the first failed scene subprocess attempt.")
    p.add_argument("--profile", action="store_true",
                   help="Log per-scene timing and output summary counts.")
    p.add_argument("--fail_fast", "--fail-fast", action="store_true",
                   dest="fail_fast",
                   help="Stop submitting new scene jobs after the first failed scene.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true",
                      help="Replace the final output directory if it already exists.")
    mode.add_argument("--resume", action="store_true",
                      help="Reuse split/preprocessed work dirs and skip scenes with done markers.")
```

Remove the old `p.add_argument("--overwrite", ...)` block so the flag is defined once.

- [ ] **Step 4: Clamp numeric flags after parsing**

At the end of `parse_args`, replace `return p.parse_args(argv)` with:

```python
    args = p.parse_args(argv)
    if args.scene_workers < 1:
        p.error("--scene-workers must be >= 1")
    if args.max_retries < 0:
        p.error("--max-retries must be >= 0")
    return args
```

- [ ] **Step 5: Run parser tests and confirm they pass**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_preprocess_calvin_multiscene.py::test_parse_args_accepts_scene_throughput_flags \
  tests/test_preprocess_calvin_multiscene.py::test_parse_args_rejects_overwrite_with_resume -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit parser contract**

Run:

```bash
git add runners/preprocess_calvin_multiscene.py tests/test_preprocess_calvin_multiscene.py
git commit -m "feat(calvin): add multiscene throughput cli flags"
```

---

### Task 2: Add Scene Marker And Report Helpers

**Files:**
- Create: `tools/preprocess/calvin_multiscene_resume.py`
- Create: `tests/test_calvin_multiscene_resume.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_calvin_multiscene_resume.py` with:

```python
import json

from tools.preprocess.calvin_multiscene_resume import (
    SceneDoneMarker,
    SceneFailureRecord,
    failed_scenes_path,
    load_scene_done_markers,
    remove_scene_done_marker,
    scene_done_marker_path,
    summarize_scene_output,
    write_failed_scenes,
    write_scene_done_marker,
)


def test_scene_done_marker_roundtrip(tmp_path):
    marker = SceneDoneMarker(
        scene="A",
        scene_input_dir=str(tmp_path / "split" / "A"),
        scene_output_dir=str(tmp_path / "preprocessed" / "A"),
        command=["python", "runners/preprocess_calvin.py"],
        return_code=0,
        elapsed_sec=1.25,
        attempt_count=2,
        retry_count=1,
        output_summary={"parquet_files": 3, "primary_videos": 3},
    )

    write_scene_done_marker(tmp_path, marker)

    assert scene_done_marker_path(tmp_path, "A").exists()
    loaded = load_scene_done_markers(tmp_path)
    assert loaded == {"A": marker}

    remove_scene_done_marker(tmp_path, "A")
    assert load_scene_done_markers(tmp_path) == {}


def test_failed_scene_report_roundtrip(tmp_path):
    record = SceneFailureRecord(
        scene="B",
        command=["python", "runners/preprocess_calvin.py"],
        return_code=7,
        exception_type="CalledProcessError",
        message="exit status 7",
        elapsed_sec=2.5,
        attempt_count=3,
        retry_count=2,
    )

    write_failed_scenes(tmp_path, [record])

    payload = json.loads(failed_scenes_path(tmp_path).read_text())
    assert payload["failed_count"] == 1
    assert payload["failed_scenes"][0]["scene"] == "B"
    assert payload["failed_scenes"][0]["retry_count"] == 2


def test_summarize_scene_output_counts_expected_files(tmp_path):
    scene_root = tmp_path / "preprocessed" / "A"
    (scene_root / "data" / "chunk-000").mkdir(parents=True)
    (scene_root / "data" / "chunk-000" / "episode_000000.parquet").write_text("x")
    (scene_root / "videos" / "chunk-000" / "video.primary_image").mkdir(parents=True)
    (scene_root / "videos" / "chunk-000" / "video.primary_image" / "episode_000000.mp4").write_text("x")
    (scene_root / "videos" / "chunk-000" / "video.wrist_image").mkdir(parents=True)
    (scene_root / "videos" / "chunk-000" / "video.wrist_image" / "episode_000000.mp4").write_text("x")
    for root in (
        "image_targets",
        "point_clouds",
        "depths/static",
        "grounding_masks/static",
        "affordance_heatmaps/static",
    ):
        (scene_root / root / "0").mkdir(parents=True)

    assert summarize_scene_output(scene_root) == {
        "parquet_files": 1,
        "primary_videos": 1,
        "wrist_videos": 1,
        "image_target_episodes": 1,
        "point_cloud_episodes": 1,
        "depth_static_episodes": 1,
        "grounding_static_episodes": 1,
        "affordance_static_episodes": 1,
    }
```

- [ ] **Step 2: Run helper tests and confirm they fail**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/test_calvin_multiscene_resume.py -q
```

Expected: collection fails because `tools.preprocess.calvin_multiscene_resume` does not exist.

- [ ] **Step 3: Implement helper module**

Create `tools/preprocess/calvin_multiscene_resume.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(eq=True)
class SceneDoneMarker:
    scene: str
    scene_input_dir: str
    scene_output_dir: str
    command: list[str]
    return_code: int
    elapsed_sec: float
    attempt_count: int
    retry_count: int
    output_summary: dict[str, int]


@dataclass(eq=True)
class SceneFailureRecord:
    scene: str
    command: list[str]
    return_code: int | None
    exception_type: str
    message: str
    elapsed_sec: float
    attempt_count: int
    retry_count: int


@dataclass(eq=True)
class SceneRunResult:
    scene: str
    scene_input_dir: str
    scene_output_dir: str
    command: list[str]
    return_code: int | None
    elapsed_sec: float
    attempt_count: int
    retry_count: int
    status: str
    output_summary: dict[str, int]
    exception_type: str | None = None
    message: str | None = None

    def to_done_marker(self) -> SceneDoneMarker:
        if self.status != "done" or self.return_code != 0:
            raise ValueError(f"cannot create done marker from status={self.status!r}")
        return SceneDoneMarker(
            scene=self.scene,
            scene_input_dir=self.scene_input_dir,
            scene_output_dir=self.scene_output_dir,
            command=self.command,
            return_code=0,
            elapsed_sec=self.elapsed_sec,
            attempt_count=self.attempt_count,
            retry_count=self.retry_count,
            output_summary=self.output_summary,
        )

    def to_failure_record(self) -> SceneFailureRecord:
        return SceneFailureRecord(
            scene=self.scene,
            command=self.command,
            return_code=self.return_code,
            exception_type=self.exception_type or "RuntimeError",
            message=self.message or f"scene {self.scene} failed",
            elapsed_sec=self.elapsed_sec,
            attempt_count=self.attempt_count,
            retry_count=self.retry_count,
        )


def scene_marker_dir(work_dir: str | Path) -> Path:
    return Path(work_dir) / "meta" / "preprocess_scenes"


def scene_done_marker_path(work_dir: str | Path, scene: str) -> Path:
    return scene_marker_dir(work_dir) / f"{scene.upper()}.done.json"


def failed_scenes_path(work_dir: str | Path) -> Path:
    return Path(work_dir) / "meta" / "preprocess_failed_scenes.json"


def write_scene_done_marker(work_dir: str | Path, marker: SceneDoneMarker) -> None:
    path = scene_done_marker_path(work_dir, marker.scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(marker), indent=2, sort_keys=True) + "\n")


def load_scene_done_markers(work_dir: str | Path) -> dict[str, SceneDoneMarker]:
    root = scene_marker_dir(work_dir)
    if not root.exists():
        return {}
    markers: dict[str, SceneDoneMarker] = {}
    for path in sorted(root.glob("*.done.json")):
        payload = json.loads(path.read_text())
        marker = SceneDoneMarker(**payload)
        markers[marker.scene.upper()] = marker
    return markers


def remove_scene_done_marker(work_dir: str | Path, scene: str) -> None:
    scene_done_marker_path(work_dir, scene).unlink(missing_ok=True)


def write_failed_scenes(
    work_dir: str | Path,
    records: list[SceneFailureRecord],
) -> None:
    path = failed_scenes_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "failed_count": len(records),
        "failed_scenes": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.glob(pattern) if path.is_file())


def _count_episode_dirs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.iterdir() if path.is_dir())


def summarize_scene_output(scene_output_dir: str | Path) -> dict[str, int]:
    root = Path(scene_output_dir)
    return {
        "parquet_files": _count_files(root / "data", "chunk-*/episode_*.parquet"),
        "primary_videos": _count_files(
            root / "videos",
            "chunk-*/video.primary_image/episode_*.mp4",
        ),
        "wrist_videos": _count_files(
            root / "videos",
            "chunk-*/video.wrist_image/episode_*.mp4",
        ),
        "image_target_episodes": _count_episode_dirs(root / "image_targets"),
        "point_cloud_episodes": _count_episode_dirs(root / "point_clouds"),
        "depth_static_episodes": _count_episode_dirs(root / "depths" / "static"),
        "grounding_static_episodes": _count_episode_dirs(root / "grounding_masks" / "static"),
        "affordance_static_episodes": _count_episode_dirs(root / "affordance_heatmaps" / "static"),
    }
```

- [ ] **Step 4: Run helper tests and confirm they pass**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/test_calvin_multiscene_resume.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit helper module**

Run:

```bash
git add tools/preprocess/calvin_multiscene_resume.py tests/test_calvin_multiscene_resume.py
git commit -m "feat(calvin): add multiscene resume markers"
```

---

### Task 3: Split Command Building From Subprocess Execution

**Files:**
- Modify: `runners/preprocess_calvin_multiscene.py`
- Modify: `tests/test_preprocess_calvin_multiscene.py`

- [ ] **Step 1: Write failing command builder test**

Append this test to `tests/test_preprocess_calvin_multiscene.py`:

```python
def test_build_preprocessor_cmd_contains_scene_specific_arguments(tmp_path):
    args = argparse.Namespace(
        num_workers=4,
        on_resolve_failure="skip",
        on_missing_target="abort",
    )

    cmd = runner._build_preprocessor_cmd(
        "C",
        tmp_path / "split" / "C",
        tmp_path / "preprocessed" / "C",
        args,
    )

    assert cmd[cmd.index("--dataset_source") + 1] == "calvin_scene_C"
    assert cmd[cmd.index("--input_dir") + 1] == str(tmp_path / "split" / "C")
    assert cmd[cmd.index("--output_dir") + 1] == str(tmp_path / "preprocessed" / "C")
    assert cmd[cmd.index("--num_workers") + 1] == "4"
    assert cmd[cmd.index("--default_scene") + 1] == "C"
    assert cmd[cmd.index("--on_resolve_failure") + 1] == "skip"
    assert cmd[cmd.index("--on_missing_target") + 1] == "abort"
```

- [ ] **Step 2: Run command builder test and confirm it fails**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_preprocess_calvin_multiscene.py::test_build_preprocessor_cmd_contains_scene_specific_arguments -q
```

Expected: failure because `_build_preprocessor_cmd` does not exist.

- [ ] **Step 3: Add command builder and reuse it**

In `runners/preprocess_calvin_multiscene.py`, add `_build_preprocessor_cmd` above `_run_preprocessor` and replace `_run_preprocessor` body:

```python
def _build_preprocessor_cmd(
    scene: str,
    scene_input_dir: Path,
    scene_output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve().parent / "preprocess_calvin.py"),
        "--input_dir", str(scene_input_dir),
        "--output_dir", str(scene_output_dir),
        "--dataset_source", f"calvin_scene_{scene}",
        "--num_workers", str(args.num_workers),
        "--default_scene", scene,
        "--on_resolve_failure", args.on_resolve_failure,
        "--on_missing_target", args.on_missing_target,
    ]
```

Then update `_run_preprocessor`:

```python
    cmd = _build_preprocessor_cmd(scene, scene_input_dir, scene_output_dir, args)
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return cmd
```

Change its return annotation from `-> None` to `-> list[str]`.

- [ ] **Step 4: Update existing subprocess test expectation**

In `test_run_preprocessor_passes_scene_specific_dataset_source`, keep the subprocess assertions and add:

```python
    returned_cmd = runner._run_preprocessor(
        "A",
        tmp_path / "input_scene",
        tmp_path / "output_scene",
        args,
    )

    assert returned_cmd == cmd
```

Remove the old direct call in that test so `_run_preprocessor(...)` is invoked once and stored in `returned_cmd`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_preprocess_calvin_multiscene.py::test_build_preprocessor_cmd_contains_scene_specific_arguments \
  tests/test_preprocess_calvin_multiscene.py::test_run_preprocessor_passes_scene_specific_dataset_source -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit command builder**

Run:

```bash
git add runners/preprocess_calvin_multiscene.py tests/test_preprocess_calvin_multiscene.py
git commit -m "refactor(calvin): expose multiscene subprocess command builder"
```

---

### Task 4: Add Retry Wrapper With Profile Results

**Files:**
- Modify: `runners/preprocess_calvin_multiscene.py`
- Modify: `tests/test_preprocess_calvin_multiscene.py`

- [ ] **Step 1: Write failing retry wrapper tests**

Append these tests to `tests/test_preprocess_calvin_multiscene.py`:

```python
def test_run_scene_preprocessor_with_retries_records_success(tmp_path, monkeypatch):
    calls = []

    def fake_run_preprocessor(scene, scene_input_dir, scene_output_dir, args):
        calls.append(scene)
        scene_output_dir.mkdir(parents=True)
        return ["python", "runner.py", scene]

    monkeypatch.setattr(runner, "_run_preprocessor", fake_run_preprocessor)
    args = argparse.Namespace(max_retries=1, profile=True)

    result = runner._run_scene_preprocessor_with_retries(
        "A",
        tmp_path / "split" / "A",
        tmp_path / "preprocessed" / "A",
        args,
    )

    assert calls == ["A"]
    assert result.status == "done"
    assert result.return_code == 0
    assert result.attempt_count == 1
    assert result.retry_count == 0
    assert result.command == ["python", "runner.py", "A"]


def test_run_scene_preprocessor_with_retries_retries_called_process_error(tmp_path, monkeypatch):
    calls = []

    def flaky_run_preprocessor(scene, scene_input_dir, scene_output_dir, args):
        calls.append(scene)
        if len(calls) == 1:
            raise runner.subprocess.CalledProcessError(
                returncode=9,
                cmd=["python", "runner.py", scene],
            )
        scene_output_dir.mkdir(parents=True)
        return ["python", "runner.py", scene]

    monkeypatch.setattr(runner, "_run_preprocessor", flaky_run_preprocessor)
    args = argparse.Namespace(max_retries=1, profile=True)

    result = runner._run_scene_preprocessor_with_retries(
        "B",
        tmp_path / "split" / "B",
        tmp_path / "preprocessed" / "B",
        args,
    )

    assert calls == ["B", "B"]
    assert result.status == "done"
    assert result.attempt_count == 2
    assert result.retry_count == 1


def test_run_scene_preprocessor_with_retries_returns_failure_after_exhaustion(tmp_path, monkeypatch):
    def failing_run_preprocessor(scene, scene_input_dir, scene_output_dir, args):
        raise runner.subprocess.CalledProcessError(
            returncode=11,
            cmd=["python", "runner.py", scene],
        )

    monkeypatch.setattr(runner, "_run_preprocessor", failing_run_preprocessor)
    args = argparse.Namespace(max_retries=1, profile=True)

    result = runner._run_scene_preprocessor_with_retries(
        "C",
        tmp_path / "split" / "C",
        tmp_path / "preprocessed" / "C",
        args,
    )

    assert result.status == "failed"
    assert result.return_code == 11
    assert result.attempt_count == 2
    assert result.retry_count == 1
    assert result.exception_type == "CalledProcessError"
```

- [ ] **Step 2: Run retry wrapper tests and confirm they fail**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_preprocess_calvin_multiscene.py::test_run_scene_preprocessor_with_retries_records_success \
  tests/test_preprocess_calvin_multiscene.py::test_run_scene_preprocessor_with_retries_retries_called_process_error \
  tests/test_preprocess_calvin_multiscene.py::test_run_scene_preprocessor_with_retries_returns_failure_after_exhaustion -q
```

Expected: failure because `_run_scene_preprocessor_with_retries` does not exist.

- [ ] **Step 3: Add imports**

In `runners/preprocess_calvin_multiscene.py`, add:

```python
import time
```

Add helper imports below existing `tools.preprocess` imports:

```python
from tools.preprocess.calvin_multiscene_resume import (
    SceneRunResult,
    summarize_scene_output,
)
```

- [ ] **Step 4: Implement retry wrapper**

Add this function below `_run_preprocessor`:

```python
def _run_scene_preprocessor_with_retries(
    scene: str,
    scene_input_dir: Path,
    scene_output_dir: Path,
    args: argparse.Namespace,
) -> SceneRunResult:
    command = _build_preprocessor_cmd(scene, scene_input_dir, scene_output_dir, args)
    start = time.perf_counter()
    max_attempts = int(args.max_retries) + 1
    last_exc: BaseException | None = None
    last_return_code: int | None = None

    for attempt_idx in range(max_attempts):
        attempt_count = attempt_idx + 1
        try:
            command = _run_preprocessor(scene, scene_input_dir, scene_output_dir, args)
            elapsed = time.perf_counter() - start
            summary = summarize_scene_output(scene_output_dir)
            if args.profile:
                logger.info(
                    "Scene %s complete in %.2fs after %d attempt(s): %s",
                    scene,
                    elapsed,
                    attempt_count,
                    summary,
                )
            return SceneRunResult(
                scene=scene,
                scene_input_dir=str(scene_input_dir),
                scene_output_dir=str(scene_output_dir),
                command=command,
                return_code=0,
                elapsed_sec=elapsed,
                attempt_count=attempt_count,
                retry_count=attempt_idx,
                status="done",
                output_summary=summary,
            )
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            last_return_code = int(exc.returncode)
            command = list(exc.cmd) if isinstance(exc.cmd, list) else command
            logger.warning(
                "Scene %s attempt %d/%d failed with return code %s",
                scene,
                attempt_count,
                max_attempts,
                last_return_code,
            )
        except Exception as exc:
            last_exc = exc
            last_return_code = None
            logger.warning(
                "Scene %s attempt %d/%d failed before subprocess completion: %s",
                scene,
                attempt_count,
                max_attempts,
                exc,
            )

    elapsed = time.perf_counter() - start
    message = str(last_exc) if last_exc is not None else f"scene {scene} failed"
    exception_type = type(last_exc).__name__ if last_exc is not None else "RuntimeError"
    return SceneRunResult(
        scene=scene,
        scene_input_dir=str(scene_input_dir),
        scene_output_dir=str(scene_output_dir),
        command=command,
        return_code=last_return_code,
        elapsed_sec=elapsed,
        attempt_count=max_attempts,
        retry_count=max_attempts - 1,
        status="failed",
        output_summary=summarize_scene_output(scene_output_dir),
        exception_type=exception_type,
        message=message,
    )
```

- [ ] **Step 5: Run retry wrapper tests and confirm they pass**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_preprocess_calvin_multiscene.py::test_run_scene_preprocessor_with_retries_records_success \
  tests/test_preprocess_calvin_multiscene.py::test_run_scene_preprocessor_with_retries_retries_called_process_error \
  tests/test_preprocess_calvin_multiscene.py::test_run_scene_preprocessor_with_retries_returns_failure_after_exhaustion -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit retry wrapper**

Run:

```bash
git add runners/preprocess_calvin_multiscene.py tests/test_preprocess_calvin_multiscene.py
git commit -m "feat(calvin): retry scene preprocessing subprocesses"
```

---

### Task 5: Implement Resume, Force-Scene, And Work Dir Cleanup Semantics

**Files:**
- Modify: `runners/preprocess_calvin_multiscene.py`
- Modify: `tests/test_preprocess_calvin_multiscene.py`

- [ ] **Step 1: Write failing resume and force-scene tests**

Append these tests to `tests/test_preprocess_calvin_multiscene.py`:

```python
def test_runner_resume_skips_done_scene_and_merges_all_requested_scenes(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    output_dir.mkdir()
    (work_dir / "split" / "A").mkdir(parents=True)
    (work_dir / "split" / "B").mkdir(parents=True)
    (work_dir / "preprocessed" / "A").mkdir(parents=True)

    from tools.preprocess.calvin_multiscene_resume import SceneDoneMarker, write_scene_done_marker

    write_scene_done_marker(
        work_dir,
        SceneDoneMarker(
            scene="A",
            scene_input_dir=str(work_dir / "split" / "A"),
            scene_output_dir=str(work_dir / "preprocessed" / "A"),
            command=["python", "runner.py", "A"],
            return_code=0,
            elapsed_sec=1.0,
            attempt_count=1,
            retry_count=0,
            output_summary={},
        ),
    )

    preprocess_calls = []
    merge_calls = []

    def fake_split(*, input_dir, output_dir, scenes, scene_config_dir=None):
        return {scene: output_dir / scene for scene in scenes}

    def fake_run_with_retries(scene, scene_input_dir, scene_output_dir, args):
        preprocess_calls.append(scene)
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        return runner.SceneRunResult(
            scene=scene,
            scene_input_dir=str(scene_input_dir),
            scene_output_dir=str(scene_output_dir),
            command=["python", "runner.py", scene],
            return_code=0,
            elapsed_sec=1.0,
            attempt_count=1,
            retry_count=0,
            status="done",
            output_summary={},
        )

    def fake_merge(scene_dirs, output_dir, *, overwrite, skip_stats, robot_type, action_mode):
        merge_calls.append((list(scene_dirs), output_dir, overwrite))

    monkeypatch.setattr(runner, "split_calvin_by_scene", fake_split)
    monkeypatch.setattr(runner, "_run_scene_preprocessor_with_retries", fake_run_with_retries)
    monkeypatch.setattr(runner, "merge_lerobot_scene_outputs", fake_merge, raising=False)

    runner.main(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--work_dir",
            str(work_dir),
            "--scenes",
            "A,B",
            "--resume",
            "--skip_stats",
        ]
    )

    assert preprocess_calls == ["B"]
    assert merge_calls == [(
        [work_dir / "preprocessed" / "A", work_dir / "preprocessed" / "B"],
        output_dir,
        True,
    )]


def test_runner_force_scene_reruns_done_scene(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    output_dir.mkdir()
    (work_dir / "split" / "A").mkdir(parents=True)
    stale_scene_output = work_dir / "preprocessed" / "A"
    stale_scene_output.mkdir(parents=True)
    (stale_scene_output / "stale.txt").write_text("stale")

    from tools.preprocess.calvin_multiscene_resume import SceneDoneMarker, write_scene_done_marker

    write_scene_done_marker(
        work_dir,
        SceneDoneMarker(
            scene="A",
            scene_input_dir=str(work_dir / "split" / "A"),
            scene_output_dir=str(stale_scene_output),
            command=["python", "runner.py", "A"],
            return_code=0,
            elapsed_sec=1.0,
            attempt_count=1,
            retry_count=0,
            output_summary={},
        ),
    )

    preprocess_calls = []

    def fake_split(*, input_dir, output_dir, scenes, scene_config_dir=None):
        return {"A": output_dir / "A"}

    def fake_run_with_retries(scene, scene_input_dir, scene_output_dir, args):
        assert not (scene_output_dir / "stale.txt").exists()
        preprocess_calls.append(scene)
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        return runner.SceneRunResult(
            scene=scene,
            scene_input_dir=str(scene_input_dir),
            scene_output_dir=str(scene_output_dir),
            command=["python", "runner.py", scene],
            return_code=0,
            elapsed_sec=1.0,
            attempt_count=1,
            retry_count=0,
            status="done",
            output_summary={},
        )

    monkeypatch.setattr(runner, "split_calvin_by_scene", fake_split)
    monkeypatch.setattr(runner, "_run_scene_preprocessor_with_retries", fake_run_with_retries)
    monkeypatch.setattr(runner, "merge_lerobot_scene_outputs", lambda *args, **kwargs: None, raising=False)

    runner.main(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--work_dir",
            str(work_dir),
            "--scenes",
            "A",
            "--resume",
            "--force-scene",
            "A",
            "--skip_stats",
        ]
    )

    assert preprocess_calls == ["A"]
```

- [ ] **Step 2: Run resume tests and confirm they fail**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_preprocess_calvin_multiscene.py::test_runner_resume_skips_done_scene_and_merges_all_requested_scenes \
  tests/test_preprocess_calvin_multiscene.py::test_runner_force_scene_reruns_done_scene -q
```

Expected: failure because resume markers are not read and `_run_scene_preprocessor_with_retries` is not used in `main`.

- [ ] **Step 3: Add resume helper imports**

In `runners/preprocess_calvin_multiscene.py`, extend helper imports to:

```python
from tools.preprocess.calvin_multiscene_resume import (
    SceneFailureRecord,
    SceneRunResult,
    failed_scenes_path,
    load_scene_done_markers,
    remove_scene_done_marker,
    scene_marker_dir,
    summarize_scene_output,
    write_failed_scenes,
    write_scene_done_marker,
)
```

- [ ] **Step 4: Add scene normalization helpers**

Add these functions above `main`:

```python
def _parse_scene_list(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _normalise_force_scenes(values: list[str]) -> set[str]:
    scenes: set[str] = set()
    for value in values:
        for scene in _parse_scene_list(value):
            scenes.add(scene)
    return scenes
```

- [ ] **Step 5: Replace setup section in `main`**

In `main`, replace:

```python
    scenes = [s.strip().upper() for s in args.scenes.split(",") if s.strip()]
    split_root = args.work_dir / "split"
    preprocessed_root = args.work_dir / "preprocessed"
```

with:

```python
    scenes = _parse_scene_list(args.scenes)
    force_scenes = _normalise_force_scenes(args.force_scene)
    split_root = args.work_dir / "split"
    preprocessed_root = args.work_dir / "preprocessed"
    marker_root = scene_marker_dir(args.work_dir)
    failed_report_path = failed_scenes_path(args.work_dir)
```

Replace the output-dir guard:

```python
    if args.output_dir.exists() and not args.overwrite:
```

with:

```python
    if args.output_dir.exists() and not (args.overwrite or args.resume):
```

Replace split/preprocessed cleanup before splitting with:

```python
    if args.overwrite:
        for stale_path in (split_root, preprocessed_root, marker_root):
            if stale_path.exists():
                logger.info("Cleaning stale work dir: %s", stale_path)
                shutil.rmtree(stale_path)
        failed_report_path.unlink(missing_ok=True)
    elif not args.resume and split_root.exists():
        logger.info("Cleaning stale split work dir: %s", split_root)
        shutil.rmtree(split_root)

    split_root.mkdir(parents=True, exist_ok=True)
    preprocessed_root.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 6: Replace serial preprocess loop with resume-aware loop**

Replace the existing `scene_output_dirs = []` loop with this serial version. Parallel scheduling is added in Task 6.

```python
    done_markers = load_scene_done_markers(args.work_dir) if args.resume else {}
    scene_output_dirs = []
    results: list[SceneRunResult] = []
    failures: list[SceneFailureRecord] = []

    for scene in scenes:
        if scene not in scene_inputs:
            failures.append(
                SceneFailureRecord(
                    scene=scene,
                    command=[],
                    return_code=None,
                    exception_type="MissingSceneInput",
                    message=f"splitter did not return scene {scene}",
                    elapsed_sec=0.0,
                    attempt_count=0,
                    retry_count=0,
                )
            )
            continue

        scene_output_dir = preprocessed_root / scene
        scene_output_dirs.append(scene_output_dir)
        if args.resume and scene in done_markers and scene not in force_scenes:
            logger.info("Skipping scene %s because done marker exists", scene)
            continue

        if scene_output_dir.exists():
            logger.info("Cleaning per-scene preprocessed dir: %s", scene_output_dir)
            shutil.rmtree(scene_output_dir)
        remove_scene_done_marker(args.work_dir, scene)

        result = _run_scene_preprocessor_with_retries(
            scene,
            scene_inputs[scene],
            scene_output_dir,
            args,
        )
        results.append(result)
        if result.status == "done":
            write_scene_done_marker(args.work_dir, result.to_done_marker())
        else:
            failures.append(result.to_failure_record())
            if args.fail_fast:
                break

    write_failed_scenes(args.work_dir, failures)
    if failures:
        raise SystemExit(
            f"CALVIN multiscene preprocessing incomplete: "
            f"{len(failures)} scene(s) failed. See {failed_report_path}"
        )
```

- [ ] **Step 7: Update merge overwrite flag**

Change the merge call from:

```python
        overwrite=args.overwrite,
```

to:

```python
        overwrite=args.overwrite or args.resume,
```

- [ ] **Step 8: Run resume tests and existing runner tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/test_preprocess_calvin_multiscene.py -q
```

Expected: all tests in `tests/test_preprocess_calvin_multiscene.py` pass.

- [ ] **Step 9: Commit resume semantics**

Run:

```bash
git add runners/preprocess_calvin_multiscene.py tests/test_preprocess_calvin_multiscene.py
git commit -m "feat(calvin): resume multiscene preprocessing by scene"
```

---

### Task 6: Add Scene-Level Parallel Scheduling And Fail-Fast Semantics

**Files:**
- Modify: `runners/preprocess_calvin_multiscene.py`
- Modify: `tests/test_preprocess_calvin_multiscene.py`

- [ ] **Step 1: Write failing scheduling tests**

Append these tests to `tests/test_preprocess_calvin_multiscene.py`:

```python
def test_runner_uses_thread_pool_when_scene_workers_exceeds_one(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    work_dir.mkdir()
    submitted = []
    merge_calls = []

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, scene, scene_input_dir, scene_output_dir, args):
            submitted.append((self.max_workers, scene))
            return FakeFuture(fn(scene, scene_input_dir, scene_output_dir, args))

    def fake_as_completed(futures):
        return list(futures)

    def fake_split(*, input_dir, output_dir, scenes, scene_config_dir=None):
        return {scene: output_dir / scene for scene in scenes}

    def fake_run_with_retries(scene, scene_input_dir, scene_output_dir, args):
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        return runner.SceneRunResult(
            scene=scene,
            scene_input_dir=str(scene_input_dir),
            scene_output_dir=str(scene_output_dir),
            command=["python", "runner.py", scene],
            return_code=0,
            elapsed_sec=1.0,
            attempt_count=1,
            retry_count=0,
            status="done",
            output_summary={},
        )

    monkeypatch.setattr(runner, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(runner, "as_completed", fake_as_completed)
    monkeypatch.setattr(runner, "split_calvin_by_scene", fake_split)
    monkeypatch.setattr(runner, "_run_scene_preprocessor_with_retries", fake_run_with_retries)
    monkeypatch.setattr(runner, "merge_lerobot_scene_outputs", lambda *args, **kwargs: merge_calls.append(args), raising=False)

    runner.main(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--work_dir",
            str(work_dir),
            "--scenes",
            "A,B,C",
            "--overwrite",
            "--scene-workers",
            "2",
            "--skip_stats",
        ]
    )

    assert submitted == [(2, "A"), (2, "B"), (2, "C")]
    assert len(merge_calls) == 1


def test_runner_records_failed_scene_and_skips_merge(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    work_dir.mkdir()
    merge_calls = []

    def fake_split(*, input_dir, output_dir, scenes, scene_config_dir=None):
        return {scene: output_dir / scene for scene in scenes}

    def fake_run_with_retries(scene, scene_input_dir, scene_output_dir, args):
        if scene == "B":
            return runner.SceneRunResult(
                scene=scene,
                scene_input_dir=str(scene_input_dir),
                scene_output_dir=str(scene_output_dir),
                command=["python", "runner.py", scene],
                return_code=5,
                elapsed_sec=1.0,
                attempt_count=1,
                retry_count=0,
                status="failed",
                output_summary={},
                exception_type="CalledProcessError",
                message="exit status 5",
            )
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        return runner.SceneRunResult(
            scene=scene,
            scene_input_dir=str(scene_input_dir),
            scene_output_dir=str(scene_output_dir),
            command=["python", "runner.py", scene],
            return_code=0,
            elapsed_sec=1.0,
            attempt_count=1,
            retry_count=0,
            status="done",
            output_summary={},
        )

    monkeypatch.setattr(runner, "split_calvin_by_scene", fake_split)
    monkeypatch.setattr(runner, "_run_scene_preprocessor_with_retries", fake_run_with_retries)
    monkeypatch.setattr(runner, "merge_lerobot_scene_outputs", lambda *args, **kwargs: merge_calls.append(args), raising=False)

    with pytest.raises(SystemExit, match="incomplete"):
        runner.main(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                "--work_dir",
                str(work_dir),
                "--scenes",
                "A,B,C",
                "--overwrite",
                "--scene-workers",
                "2",
                "--skip_stats",
            ]
        )

    assert merge_calls == []
```

- [ ] **Step 2: Run scheduling tests and confirm they fail**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_preprocess_calvin_multiscene.py::test_runner_uses_thread_pool_when_scene_workers_exceeds_one \
  tests/test_preprocess_calvin_multiscene.py::test_runner_records_failed_scene_and_skips_merge -q
```

Expected: first test fails because `ThreadPoolExecutor` is not imported and used. Second test may pass under the serial loop from Task 5; keep it to protect merge gating during the parallel rewrite.

- [ ] **Step 3: Add executor imports**

In `runners/preprocess_calvin_multiscene.py`, add:

```python
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
```

- [ ] **Step 4: Add helper to submit and collect scene jobs**

Add this function above `main`:

```python
def _run_scene_jobs(
    jobs: list[tuple[str, Path, Path]],
    args: argparse.Namespace,
) -> list[SceneRunResult]:
    if not jobs:
        return []
    if args.scene_workers == 1:
        return [
            _run_scene_preprocessor_with_retries(scene, scene_input_dir, scene_output_dir, args)
            for scene, scene_input_dir, scene_output_dir in jobs
        ]

    max_workers = min(args.scene_workers, len(jobs))
    results: list[SceneRunResult] = []
    if not args.fail_fast:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_scene_preprocessor_with_retries,
                    scene,
                    scene_input_dir,
                    scene_output_dir,
                    args,
                )
                for scene, scene_input_dir, scene_output_dir in jobs
            ]
            for future in as_completed(futures):
                results.append(future.result())
        return results

    pending_jobs = list(jobs)
    running: dict[Future[SceneRunResult], str] = {}
    stop_submitting = False
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending_jobs or running:
            while pending_jobs and not stop_submitting and len(running) < max_workers:
                scene, scene_input_dir, scene_output_dir = pending_jobs.pop(0)
                future = executor.submit(
                    _run_scene_preprocessor_with_retries,
                    scene,
                    scene_input_dir,
                    scene_output_dir,
                    args,
                )
                running[future] = scene
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                running.pop(future)
                result = future.result()
                results.append(result)
                if result.status != "done":
                    stop_submitting = True
            if stop_submitting and not running:
                break
    return results
```

- [ ] **Step 5: Replace serial run loop with job preparation and result handling**

In `main`, keep the split and marker loading logic from Task 5, then replace the loop body from Task 5 with:

```python
    scene_output_dirs = []
    jobs: list[tuple[str, Path, Path]] = []
    failures: list[SceneFailureRecord] = []

    for scene in scenes:
        if scene not in scene_inputs:
            failures.append(
                SceneFailureRecord(
                    scene=scene,
                    command=[],
                    return_code=None,
                    exception_type="MissingSceneInput",
                    message=f"splitter did not return scene {scene}",
                    elapsed_sec=0.0,
                    attempt_count=0,
                    retry_count=0,
                )
            )
            continue

        scene_output_dir = preprocessed_root / scene
        scene_output_dirs.append(scene_output_dir)
        if args.resume and scene in done_markers and scene not in force_scenes:
            logger.info("Skipping scene %s because done marker exists", scene)
            continue

        if scene_output_dir.exists():
            logger.info("Cleaning per-scene preprocessed dir: %s", scene_output_dir)
            shutil.rmtree(scene_output_dir)
        remove_scene_done_marker(args.work_dir, scene)
        jobs.append((scene, scene_inputs[scene], scene_output_dir))

    results = _run_scene_jobs(jobs, args)
    for result in sorted(results, key=lambda item: scenes.index(item.scene)):
        if result.status == "done":
            write_scene_done_marker(args.work_dir, result.to_done_marker())
        else:
            failures.append(result.to_failure_record())

    write_failed_scenes(args.work_dir, failures)
    if failures:
        raise SystemExit(
            f"CALVIN multiscene preprocessing incomplete: "
            f"{len(failures)} scene(s) failed. See {failed_report_path}"
        )
```

- [ ] **Step 6: Run scheduling tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_preprocess_calvin_multiscene.py::test_runner_uses_thread_pool_when_scene_workers_exceeds_one \
  tests/test_preprocess_calvin_multiscene.py::test_runner_records_failed_scene_and_skips_merge -q
```

Expected: `2 passed`.

- [ ] **Step 7: Run full multiscene runner tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/test_preprocess_calvin_multiscene.py -q
```

Expected: all tests in `tests/test_preprocess_calvin_multiscene.py` pass.

- [ ] **Step 8: Commit parallel scheduler**

Run:

```bash
git add runners/preprocess_calvin_multiscene.py tests/test_preprocess_calvin_multiscene.py
git commit -m "feat(calvin): parallelize multiscene preprocessing"
```

---

### Task 7: Verify Merge Gating, Marker Output, And Existing CALVIN Tests

**Files:**
- Modify: `tests/test_preprocess_calvin_multiscene.py`
- Modify: `docs/superpowers/specs/2026-05-22-calvin-multiscene-throughput-design.md`

- [ ] **Step 1: Add final merge gating test for missing scene inputs**

Append this test to `tests/test_preprocess_calvin_multiscene.py`:

```python
def test_runner_skips_merge_when_requested_scene_is_missing_from_split(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    work_dir.mkdir()
    merge_calls = []

    def fake_split(*, input_dir, output_dir, scenes, scene_config_dir=None):
        return {"A": output_dir / "A"}

    monkeypatch.setattr(runner, "split_calvin_by_scene", fake_split)
    monkeypatch.setattr(runner, "merge_lerobot_scene_outputs", lambda *args, **kwargs: merge_calls.append(args), raising=False)

    with pytest.raises(SystemExit, match="incomplete"):
        runner.main(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                "--work_dir",
                str(work_dir),
                "--scenes",
                "A,B",
                "--overwrite",
                "--skip_stats",
            ]
        )

    assert merge_calls == []
```

- [ ] **Step 2: Run missing-scene test**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_preprocess_calvin_multiscene.py::test_runner_skips_merge_when_requested_scene_is_missing_from_split -q
```

Expected: `1 passed`.

- [ ] **Step 3: Run all targeted tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_calvin_multiscene_resume.py \
  tests/test_preprocess_calvin_multiscene.py \
  tests/test_calvin_lerobot_merger.py \
  tests/test_calvin_scene_splitter.py -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Update design status after implementation**

In `docs/superpowers/specs/2026-05-22-calvin-multiscene-throughput-design.md`, change:

```text
Status: approved design, awaiting implementation plan
```

to:

```text
Status: implemented locally
```

Add this line under the status line:

```text
Implementation: `runners/preprocess_calvin_multiscene.py` and `tools/preprocess/calvin_multiscene_resume.py`
```

- [ ] **Step 5: Run formatting and red-flag text checks**

Run:

```bash
git diff --check
```

Expected: no output.

Run:

```bash
python -c 'import pathlib,sys; terms=["TB"+"D","TO"+"DO","FIX"+"ME","defer"+" this","fill"+" in"]; files=["runners/preprocess_calvin_multiscene.py","tools/preprocess/calvin_multiscene_resume.py","tests/test_preprocess_calvin_multiscene.py","tests/test_calvin_multiscene_resume.py","docs/superpowers/specs/2026-05-22-calvin-multiscene-throughput-design.md"]; hits=[]; [hits.append((f,i,t)) for f in files for i,line in enumerate(pathlib.Path(f).read_text().splitlines(),1) for t in terms if t in line]; print("\\n".join(f"{f}:{i}: {t}" for f,i,t in hits)); sys.exit(1 if hits else 0)'
```

Expected: no output.

- [ ] **Step 6: Commit final verification and spec status**

Run:

```bash
git add \
  runners/preprocess_calvin_multiscene.py \
  tools/preprocess/calvin_multiscene_resume.py \
  tests/test_preprocess_calvin_multiscene.py \
  tests/test_calvin_multiscene_resume.py \
  docs/superpowers/specs/2026-05-22-calvin-multiscene-throughput-design.md
git commit -m "test(calvin): verify multiscene throughput resume flow"
```

---

## Final Verification

After all tasks are complete, run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/test_calvin_multiscene_resume.py \
  tests/test_preprocess_calvin_multiscene.py \
  tests/test_calvin_lerobot_merger.py \
  tests/test_calvin_scene_splitter.py -q
```

Expected: all selected tests pass.

Then run:

```bash
git status --short
```

Expected: no uncommitted changes.

## Remote Command After Implementation

For `task_ABC_D` on the remote server:

```bash
cd /inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UniamVLA
mkdir -p logs
conda activate calvin_env
UAMVLA_CALVIN_NATIVE_SAFE_EXIT=1 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
python -u runners/preprocess_calvin_multiscene.py \
  --input_dir /inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UniamVLA/datasets/uam_dataset/calvin/task_ABC_D/training \
  --work_dir /inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UniamVLA/datasets/calvin2uam/work_calvin_abc \
  --output_dir /inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UniamVLA/datasets/calvin2uam/lerobot_calvin_abc \
  --scenes A,B,C \
  --scene-workers 3 \
  --num_workers 1 \
  --on_missing_target skip \
  --resume \
  --profile \
  --max-retries 1 \
  --skip_stats \
  2>&1 | tee logs/preprocess_calvin_abc_resume_scene3.log
```

For `task_ABCD_D`, use `--scenes A,B,C,D`, `--scene-workers 4`, and a distinct `--work_dir` plus `--output_dir`.
