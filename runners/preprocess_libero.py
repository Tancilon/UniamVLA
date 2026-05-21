"""CLI for official LIBERO HDF5 → UamVLA LeRobot preprocessing."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.preprocess.libero_preprocessor import LiberoPreprocessor
from tools.preprocess.libero_target_mapping import LIBERO_STANDARD_SUITES


@dataclass(frozen=True)
class SuiteJob:
    suite: str
    input_dir: Path
    output_dir: Path


def default_output_name(suite: str) -> str:
    return f"lerobot_{suite}"


def build_suite_jobs(
    input_root: Path,
    output_root: Path,
    suite: str,
) -> list[SuiteJob]:
    suites = LIBERO_STANDARD_SUITES if suite == "all" else (suite,)
    return [
        SuiteJob(
            suite=s,
            input_dir=input_root / s,
            output_dir=output_root / default_output_name(s),
        )
        for s in suites
    ]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Preprocess official LIBERO HDF5 into UamVLA LeRobot datasets."
    )
    parser.add_argument(
        "--input-root",
        default="datasets/libero",
        help="Root containing libero_spatial/object/goal/10 HDF5 directories.",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/libero2uam",
        help="Root where lerobot_libero_* datasets are written.",
    )
    parser.add_argument(
        "--suite",
        choices=("all",) + LIBERO_STANDARD_SUITES,
        default="all",
        help="Suite to process.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker count. Defaults to visible render GPU count.",
    )
    parser.add_argument(
        "--render-gpus",
        default=None,
        help="Comma-separated render GPU ids. Defaults to CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing suite output before preprocessing.",
    )
    parser.add_argument(
        "--min-segment-len",
        type=int,
        default=3,
        help="Minimum active-target segment length for smoothing.",
    )
    parser.add_argument(
        "--debug-rgb-check-frames",
        type=int,
        default=3,
        help="Number of replay-vs-HDF5 debug frames per task.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Bound the number of HDF5 task files for smoke runs.",
    )
    parser.add_argument(
        "--max-demos-per-task",
        type=int,
        default=None,
        help="Bound demos per task for smoke runs.",
    )
    parser.add_argument(
        "--max-frames-per-demo",
        type=int,
        default=None,
        help="Bound frames per demo for smoke runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    jobs = build_suite_jobs(input_root, output_root, args.suite)
    for job in jobs:
        if job.output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output directory exists: {job.output_dir}. "
                    "Pass --overwrite to replace it."
                )
            shutil.rmtree(job.output_dir)
        preprocessor = LiberoPreprocessor(
            suite=job.suite,
            num_workers=args.num_workers,
            render_gpus=args.render_gpus,
            min_segment_len=args.min_segment_len,
            debug_rgb_check_frames=args.debug_rgb_check_frames,
            max_tasks=args.max_tasks,
            max_demos_per_task=args.max_demos_per_task,
            max_frames_per_demo=args.max_frames_per_demo,
        )
        preprocessor.process(str(job.input_dir), str(job.output_dir))


if __name__ == "__main__":
    main()
