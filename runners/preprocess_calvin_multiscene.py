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
from pathlib import Path

# Make project root importable for direct module use.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.preprocess.calvin_scene_splitter import split_calvin_by_scene
from tools.preprocess.calvin_lerobot_merger import merge_lerobot_scene_outputs

logger = logging.getLogger(__name__)


def _run_preprocessor(
    scene: str,
    scene_input_dir: Path,
    scene_output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Invoke runners/preprocess_calvin.py as a subprocess.

    Subprocess (rather than in-process) so each scene gets a fresh
    PyBullet/EGL state — avoids any cross-scene env handle leaks.
    """
    cmd = [
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
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


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
    p.add_argument("--on_resolve_failure", "--on-resolve-failure",
                   choices=("skip", "abort"), default="abort",
                   dest="on_resolve_failure")
    p.add_argument("--on_missing_target", "--on-missing-target",
                   choices=("skip", "abort"), default="skip",
                   dest="on_missing_target")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace the final output directory if it already exists.")
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
    return p.parse_args(argv)


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


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)
    scenes = [s.strip().upper() for s in args.scenes.split(",") if s.strip()]
    split_root = args.work_dir / "split"
    preprocessed_root = args.work_dir / "preprocessed"

    _guard_output_path(
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        split_root=split_root,
        preprocessed_root=preprocessed_root,
        clean_work=args.clean_work,
    )

    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(
            f"Output directory already exists: {args.output_dir}. "
            "Pass --overwrite to replace it."
        )

    if split_root.exists():
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
    scene_output_dirs = []
    for scene in scenes:
        if scene not in scene_inputs:
            continue
        scene_output_dir = preprocessed_root / scene
        if scene_output_dir.exists():
            logger.info("Cleaning stale per-scene preprocessed dir: %s", scene_output_dir)
            shutil.rmtree(scene_output_dir)
        _run_preprocessor(
            scene,
            scene_inputs[scene],
            scene_output_dir,
            args,
        )
        scene_output_dirs.append(scene_output_dir)

    # 3) merge
    merge_lerobot_scene_outputs(
        scene_output_dirs,
        args.output_dir,
        overwrite=args.overwrite,
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
