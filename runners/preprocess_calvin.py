"""CLI entry point for CALVIN → UamVLA dataset preprocessing.

Usage:
    # Process the debug dataset (the path on the spec author's local box):
    python runners/preprocess_calvin.py \\
        --input_dir /Users/tancilon/develop/localgit/UamVLA/datasets/calvin_debug_dataset/training \\
        --output_dir datasets/uamvla_calvin/task_D_D \\
        --dataset_source calvin_debug

    # Process a real task_D_D split:
    python runners/preprocess_calvin.py \\
        --input_dir /path/to/calvin/task_D_D/training \\
        --output_dir datasets/uamvla_calvin/task_D_D \\
        --dataset_source task_D_D

This driver requires `calvin_env` (PyBullet) at runtime — the worker calls
env.reset(robot_obs, scene_obs) + env.render_cameras for each frame. On
machines without calvin_env, the CLI still loads (no module-level import),
but `--input_dir` actually invoking the worker will fail at make_calvin_env_adapter.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so tools/starVLA packages are importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.preprocess.calvin_preprocessor import CalvinPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a CALVIN split to UamVLA unified format.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="A CALVIN split dir containing episode_*.npz, ep_start_end_ids.npy, "
             "lang_annotations/auto_lang_ann.npy, scene_info.npy.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Target UAM dataset dir; created if missing. "
             "Recommended layout: datasets/uamvla_calvin/<split>/.",
    )
    parser.add_argument(
        "--dataset_source",
        type=str,
        default="calvin",
        help="String prefix for episode_id and JSONL dataset_source field. "
             "Override with the split name (task_D_D, calvin_debug) for clarity. "
             "Default: 'calvin'.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="multiprocessing.Pool size. Use >1 only on machines with reliable "
             "PyBullet/EGL. Default: 1.",
    )
    parser.add_argument(
        "--default_scene",
        type=str,
        default=None,
        help="Forwarded to SceneResolver. Set when the split's scene_info.npy "
             "does not cover all windows. Default: None.",
    )
    parser.add_argument(
        "--on_resolve_failure",
        choices=("skip", "abort"),
        default="abort",
        help="What to do if a window's task_label is unknown to calvin_task_map. "
             "Default: abort.",
    )
    parser.add_argument(
        "--on_missing_target",
        choices=("skip", "abort"),
        default="abort",
        help="What to do if a frame's target object is not visible in the "
             "static-camera seg mask. Default: abort.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    preprocessor = CalvinPreprocessor(
        dataset_source=args.dataset_source,
        default_scene=args.default_scene,
        num_workers=args.num_workers,
        on_resolve_failure=args.on_resolve_failure,
        on_missing_target=args.on_missing_target,
    )

    preprocessor.process(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
