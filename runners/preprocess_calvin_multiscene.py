"""End-to-end driver: split a multi-scene CALVIN dir into LeRobot output.

This wraps the three pieces required to build one unified LeRobot dataset
from a CALVIN multi-scene split:

    1. tools.preprocess.calvin_scene_splitter
       Build per-scene virtual single-scene splits (each with its own .hydra
       so calvin_env loads the right URDF).

    2. runners.preprocess_calvin
       Run the LeRobot preprocessor once per virtual split.

    3. tools.preprocess.calvin_lerobot_merger
       Fuse the per-scene LeRobot outputs into one unified dataset.

Usage
-----
    python runners/preprocess_calvin_multiscene.py \\
        --input_dir datasets/calvin/task_ABCD_D/training \\
        --work_dir datasets/calvin2uam/work_multiscene_abcd \\
        --output_dir datasets/calvin2uam/lerobot_calvin_abcd \\
        --scenes A,B,C,D \\
        --overwrite \\
        --num_workers 8 \\
        --on_missing_target skip

The intermediate per-scene work dirs (`work_dir/split` and
`work_dir/preprocessed`) are kept on disk by default; pass `--clean_work` to
wipe them after a successful merge.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

# Make project root importable for direct module use.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.preprocess.calvin_scene_splitter import split_calvin_by_scene
from tools.preprocess.calvin_lerobot_merger import merge_lerobot_scene_outputs
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

logger = logging.getLogger(__name__)


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


def _run_preprocessor(
    scene: str,
    scene_input_dir: Path,
    scene_output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Invoke runners/preprocess_calvin.py as a subprocess.

    Subprocess (rather than in-process) so each scene gets a fresh
    PyBullet/EGL state — avoids any cross-scene env handle leaks.
    """
    cmd = _build_preprocessor_cmd(scene, scene_input_dir, scene_output_dir, args)
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return cmd


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-scene CALVIN -> LeRobot preprocess driver.",
    )
    p.add_argument("--input_dir", "--input-dir", required=True, type=Path,
                   dest="input_dir",
                   help="A multi-scene CALVIN split (with scene_info.npy).")
    p.add_argument("--work_dir", "--work-dir", required=True, type=Path,
                   dest="work_dir",
                   help="Where to keep splits + per-scene preprocessed outputs.")
    p.add_argument("--output_dir", "--output-dir", required=True, type=Path,
                   dest="output_dir",
                   help="Final unified LeRobot dataset dir.")
    p.add_argument("--scenes", default="A,B,C,D")
    p.add_argument("--scene_config_dir", "--scene-config-dir", default=None,
                   type=Path, dest="scene_config_dir",
                   help="Forwarded to the splitter when calvin_env auto-discovery fails.")
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
    p.add_argument("--on_resolve_failure", "--on-resolve-failure",
                   choices=("skip", "abort"), default="abort",
                   dest="on_resolve_failure")
    p.add_argument("--on_missing_target", "--on-missing-target",
                   choices=("skip", "abort"), default="skip",
                   dest="on_missing_target")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true",
                      help="Replace the final output directory if it already exists.")
    mode.add_argument("--resume", action="store_true",
                      help="Reuse split/preprocessed work dirs and skip scenes with done markers.")
    p.add_argument("--skip_stats", "--skip-stats", action="store_true",
                   dest="skip_stats",
                   help="Skip GR00T/LeRobot stats generation after merging.")
    p.add_argument("--robot_type", "--robot-type", default="uamvla_calvin_franka",
                   dest="robot_type",
                   help="Robot type key used for merged metadata and stats.")
    p.add_argument("--action_mode", "--action-mode", default="abs",
                   dest="action_mode",
                   help="Action stats mode forwarded to the LeRobot merger.")
    p.add_argument("--clean_work", "--clean-work", action="store_true",
                   dest="clean_work",
                   help="Remove --work_dir at the end (after a successful merge).")
    args = p.parse_args(argv)
    if args.scene_workers < 1:
        p.error("--scene-workers must be >= 1")
    if args.max_retries < 0:
        p.error("--max-retries must be >= 0")
    return args


def _path_is_or_contains(parent: Path, candidate: Path) -> bool:
    parent_resolved = parent.resolve(strict=False)
    candidate_resolved = candidate.resolve(strict=False)
    if candidate_resolved == parent_resolved:
        return True
    try:
        candidate_resolved.relative_to(parent_resolved)
    except ValueError:
        return False
    return True


def _guard_output_path(
    *,
    output_dir: Path,
    work_dir: Path,
    split_root: Path,
    preprocessed_root: Path,
    clean_work: bool,
) -> None:
    if _path_is_or_contains(split_root, output_dir):
        raise SystemExit(
            f"output_dir must not be inside the split work area: "
            f"output_dir={output_dir}, split_dir={split_root}"
        )
    if _path_is_or_contains(preprocessed_root, output_dir):
        raise SystemExit(
            f"output_dir must not be inside the preprocessed work area: "
            f"output_dir={output_dir}, preprocessed_dir={preprocessed_root}"
        )
    if clean_work and _path_is_or_contains(work_dir, output_dir):
        raise SystemExit(
            f"output_dir must not be inside work_dir when --clean_work is set: "
            f"output_dir={output_dir}, work_dir={work_dir}"
        )


def _parse_scene_list(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _normalise_force_scenes(values: list[str]) -> set[str]:
    scenes: set[str] = set()
    for value in values:
        for scene in _parse_scene_list(value):
            scenes.add(scene)
    return scenes


def _run_scene_jobs(
    jobs: list[tuple[str, Path, Path]],
    args: argparse.Namespace,
) -> list[SceneRunResult]:
    if not jobs:
        return []
    if args.scene_workers == 1:
        return [
            _run_scene_preprocessor_with_retries(
                scene,
                scene_input_dir,
                scene_output_dir,
                args,
            )
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


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)
    scenes = _parse_scene_list(args.scenes)
    force_scenes = _normalise_force_scenes(args.force_scene)
    split_root = args.work_dir / "split"
    preprocessed_root = args.work_dir / "preprocessed"
    marker_root = scene_marker_dir(args.work_dir)
    failed_report_path = failed_scenes_path(args.work_dir)

    _guard_output_path(
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        split_root=split_root,
        preprocessed_root=preprocessed_root,
        clean_work=args.clean_work,
    )

    if args.output_dir.exists() and not (args.overwrite or args.resume):
        raise SystemExit(
            f"Output directory already exists: {args.output_dir}. "
            "Pass --overwrite to replace it."
        )

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

    # 1) split
    scene_inputs = split_calvin_by_scene(
        input_dir=args.input_dir,
        output_dir=split_root,
        scenes=scenes,
        scene_config_dir=args.scene_config_dir,
    )

    # 2) preprocess each
    done_markers = load_scene_done_markers(args.work_dir) if args.resume else {}
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

    # 3) merge
    merge_lerobot_scene_outputs(
        scene_output_dirs,
        args.output_dir,
        overwrite=args.overwrite or args.resume,
        skip_stats=args.skip_stats,
        robot_type=args.robot_type,
        action_mode=args.action_mode,
    )
    logger.info("Multi-scene LeRobot preprocess complete: %s", args.output_dir)

    if args.clean_work:
        logger.info("Removing work dir: %s", args.work_dir)
        shutil.rmtree(args.work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
