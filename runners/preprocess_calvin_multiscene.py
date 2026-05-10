"""End-to-end driver: split a multi-scene CALVIN dir → preprocess each scene → merge.

This wraps the three pieces required by Plan A from
docs/superpowers/specs/.../calvin-preprocess-multiscene-design.md:

    1. tools.preprocess.calvin_scene_splitter
       Build per-scene virtual single-scene splits (each with its own .hydra
       so calvin_env loads the right URDF).

    2. runners.preprocess_calvin
       Run the existing UAM preprocessor once per virtual split. Each run
       gets a unique --dataset_source so sample ids are globally unique.

    3. tools.preprocess.calvin_split_merger
       Fuse the per-scene UAM outputs into one unified UAM dataset.

Usage
-----
    python runners/preprocess_calvin_multiscene.py \\
        --input_dir  /data/calvin/task_ABCD_D/training \\
        --work_dir   /tmp/abcd_split_work \\
        --output_dir datasets/uam_dataset/uamvla_calvin/task_ABCD_D/training \\
        --dataset_source task_ABCD_D \\
        --num_workers 8 \\
        --on_missing_target skip

The intermediate per-scene work dirs (`work_dir/virtual/scene_X` and
`work_dir/preproc/scene_X`) are kept on disk by default so reruns can
skip work; pass `--clean_work` to wipe them at the end.
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
from tools.preprocess.calvin_split_merger import merge_scene_outputs

logger = logging.getLogger(__name__)


def _run_preprocessor(
    virtual_split: Path,
    output_dir: Path,
    dataset_source: str,
    scene_letter: str,
    num_workers: int,
    on_resolve_failure: str,
    on_missing_target: str,
) -> None:
    """Invoke runners/preprocess_calvin.py as a subprocess.

    Subprocess (rather than in-process) so each scene gets a fresh
    PyBullet/EGL state — avoids any cross-scene env handle leaks.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "preprocess_calvin.py"),
        "--input_dir",  str(virtual_split),
        "--output_dir", str(output_dir),
        "--dataset_source", dataset_source,
        "--num_workers", str(num_workers),
        "--default_scene", scene_letter,
        "--on_resolve_failure", on_resolve_failure,
        "--on_missing_target",  on_missing_target,
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-scene CALVIN -> UAM preprocess driver.",
    )
    p.add_argument("--input_dir", required=True, type=Path,
                   help="A multi-scene CALVIN split (with scene_info.npy).")
    p.add_argument("--work_dir", required=True, type=Path,
                   help="Where to keep virtual splits + per-scene preproc outputs.")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Final unified UAM dataset dir.")
    p.add_argument("--dataset_source", required=True,
                   help="Unified dataset_source written into every merged row.")
    p.add_argument("--scenes", default="A,B,C,D")
    p.add_argument("--scene_config_dir", default=None, type=Path,
                   help="Forwarded to the splitter when calvin_env auto-discovery fails.")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--on_resolve_failure", choices=("skip", "abort"), default="abort")
    p.add_argument("--on_missing_target", choices=("skip", "abort"), default="skip")
    p.add_argument("--clean_work", action="store_true",
                   help="Remove --work_dir at the end (after a successful merge).")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    scenes = [s.strip().upper() for s in args.scenes.split(",") if s.strip()]

    virt_root = args.work_dir / "virtual"
    pp_root = args.work_dir / "preproc"
    virt_root.mkdir(parents=True, exist_ok=True)
    pp_root.mkdir(parents=True, exist_ok=True)

    # 1) split
    virtual_splits = split_calvin_by_scene(
        input_dir=args.input_dir,
        output_dir=virt_root,
        scenes=scenes,
        scene_config_dir=args.scene_config_dir,
    )

    # 2) preprocess each
    pp_outs = []
    for letter, vs in virtual_splits.items():
        pp_out = pp_root / f"scene_{letter}"
        # Wipe any prior per-scene output before rerunning. The
        # preprocessor's shard merger blindly concatenates every file
        # under shards/, so a stale shard from a failed prior run would
        # silently leak old windows into the merged data.jsonl.
        if pp_out.exists():
            logger.info("Cleaning stale per-scene preproc dir: %s", pp_out)
            shutil.rmtree(pp_out)
        scene_source = f"{args.dataset_source}_scene{letter}"
        _run_preprocessor(
            virtual_split=vs,
            output_dir=pp_out,
            dataset_source=scene_source,
            scene_letter=letter,
            num_workers=args.num_workers,
            on_resolve_failure=args.on_resolve_failure,
            on_missing_target=args.on_missing_target,
        )
        pp_outs.append(pp_out)

    # 3) merge
    merge_scene_outputs(
        inputs=pp_outs,
        output_dir=args.output_dir,
        dataset_source=args.dataset_source,
    )
    logger.info("Multi-scene preprocess complete: %s", args.output_dir)

    if args.clean_work:
        logger.info("Removing work dir: %s", args.work_dir)
        shutil.rmtree(args.work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
