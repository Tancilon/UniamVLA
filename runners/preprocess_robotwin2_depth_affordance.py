#!/usr/bin/env python3
"""
Add depth and affordance sidecar files to RoboTwin2 LeRobot datasets.

Mirrors the Calvin sidecar pipeline in
runners/convert_calvin_dir_to_lerobot_starvla_with_uam_state.py but
adapted for RoboTwin2's task-based LeRobot layout and video-only inputs
(no simulator — depth via VDA, affordance via GroundingDINO + VRB).

Output layout (per task directory, per camera):
  depth          {task_dir}/depth/chunk-{C:03d}/{cam}/episode_{E:06d}.npz
                   key="depths"  (T, H, W) float16   relative inverse depth
  affordance_vrb {task_dir}/affordance/chunk-{C:03d}/{cam}/episode_{E:06d}.npz
                   key="heatmaps"  (T, H, W) float16  raw VRB contact heatmap
  affordance_px  {task_dir}/affordance_px/chunk-{C:03d}/{cam}/episode_{E:06d}.npy
                   (T, 112, 112) float16  area-downsampled heatmap, mmap-able

Supported aux heads once sidecars are written:
  depth_latent   — run preprocess_depth_latent.py afterward (VAE encode depth)
  affordance_px  — AffordanceTokenHead reads affordance_px/ directly

Usage:
  # Smoke test (zero-filled, no GPU):
  python runners/preprocess_robotwin2_depth_affordance.py \\
      --task_dir datasets/robotwin2/Clean/adjust_bottle --stages lite

  # Single task, depth only:
  CUDA_VISIBLE_DEVICES=0 python runners/preprocess_robotwin2_depth_affordance.py \\
      --task_dir datasets/robotwin2/Clean/adjust_bottle --stages depth

  # Single task, full pipeline, all three cameras:
  CUDA_VISIBLE_DEVICES=0 python runners/preprocess_robotwin2_depth_affordance.py \\
      --task_dir datasets/robotwin2/Clean/adjust_bottle \\
      --stages depth,affordance_vrb,affordance_px \\
      --objects "adjust bottle,robot gripper"

  # All Clean tasks, all cameras:
  CUDA_VISIBLE_DEVICES=0 python runners/preprocess_robotwin2_depth_affordance.py \\
      --dataset_root datasets/robotwin2/Clean \\
      --stages depth,affordance_vrb,affordance_px \\
      --objects "adjust bottle,robot gripper"
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Args:
    # Input: single task dir OR root dir containing multiple task dirs.
    task_dir: str | None = None
    dataset_root: str | None = None

    # Comma-separated camera keys. RoboTwin default: all three cameras.
    cameras: str = (
        "observation.images.cam_high,"
        "observation.images.cam_left_wrist,"
        "observation.images.cam_right_wrist"
    )

    # Comma-separated stage list, or "lite" for zero-filled smoke test.
    # Valid stages: depth, affordance_vrb, affordance_px
    #   affordance_px requires affordance_vrb to have run first.
    stages: str = "depth,affordance_vrb,affordance_px"

    # VDA (depth stage)
    vda_encoder: Literal["vits", "vitb", "vitl"] = "vitl"
    vda_input_size: int = 518
    vda_checkpoint: str | None = None  # defaults to VDA checkpoints dir

    # GroundingDINO + VRB (affordance_vrb stage)
    # Text prompt sent to GroundingDINO — must describe the objects to grasp.
    # Example: "robot gripper,bottle,block,container"
    objects: str = ""
    gdino_path: str | None = None     # defaults to ckpt/grounding-dino-base
    vrb_ckpt: str | None = None       # defaults to third_party/Splat-MOVER/...
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    k_ratio: float = 6.0
    frame_stride: int = 1             # affordance inference stride (1 = every frame)
    gdino_batch: int = 16
    max_boxes_per_frame: int = 8

    # Affordance_px stage
    qwen_image_size: int = 224        # must match training YAML; grid = size // 32

    device: str = "cuda"
    max_episodes: int | None = None
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_task_meta(task_dir: Path) -> tuple[int, int, float]:
    """Return (total_episodes, chunks_size, fps) from meta/info.json."""
    info = json.loads((task_dir / "meta" / "info.json").read_text())
    return info["total_episodes"], info.get("chunks_size", 1000), info["fps"]


def _depth_path(task_dir: Path, camera: str, ep_idx: int, chunks_size: int) -> Path:
    chunk = ep_idx // chunks_size
    return task_dir / "depth" / f"chunk-{chunk:03d}" / camera / f"episode_{ep_idx:06d}.npz"


def _affordance_path(task_dir: Path, camera: str, ep_idx: int, chunks_size: int) -> Path:
    chunk = ep_idx // chunks_size
    return task_dir / "affordance" / f"chunk-{chunk:03d}" / camera / f"episode_{ep_idx:06d}.npz"


def _affordance_px_path(task_dir: Path, camera: str, ep_idx: int, chunks_size: int) -> Path:
    chunk = ep_idx // chunks_size
    return task_dir / "affordance_px" / f"chunk-{chunk:03d}" / camera / f"episode_{ep_idx:06d}.npy"


def _video_path(task_dir: Path, camera: str, ep_idx: int, chunks_size: int) -> Path:
    chunk = ep_idx // chunks_size
    return task_dir / "videos" / f"chunk-{chunk:03d}" / camera / f"episode_{ep_idx:06d}.mp4"


def _collect_task_dirs(args: Args) -> list[Path]:
    if args.task_dir:
        dirs = [Path(args.task_dir)]
    elif args.dataset_root:
        root = Path(args.dataset_root)
        dirs = sorted(
            d for d in root.iterdir()
            if d.is_dir() and (d / "meta" / "info.json").exists()
        )
    else:
        raise ValueError("Provide --task_dir (single task) or --dataset_root (all tasks).")
    return dirs


# ---------------------------------------------------------------------------
# Lite mode — zero-filled smoke test without any GPU/model
# ---------------------------------------------------------------------------

def _write_lite_sidecars(task_dir: Path, camera: str, stages: list[str], args: Args) -> None:
    """Write zero-filled sidecar files for smoke-testing the training pipeline."""
    import av

    total_eps, chunks_size, _ = _load_task_meta(task_dir)
    if args.max_episodes:
        total_eps = min(total_eps, args.max_episodes)

    latent_hw = args.qwen_image_size // 8
    target_resize = (args.qwen_image_size // 32) * 16  # = 112 for qwen_size=224

    for ep_idx in tqdm(range(total_eps), desc=f"  {task_dir.name}/{camera} lite", leave=False):
        vpath = _video_path(task_dir, camera, ep_idx, chunks_size)
        if not vpath.exists():
            continue
        # Decode only frame count to build zero arrays of the right T dimension.
        with av.open(str(vpath)) as c:
            T = c.streams.video[0].frames or sum(1 for _ in c.decode(c.streams.video[0]))
        H, W = 480, 640  # RoboTwin default resolution

        if "depth" in stages:
            p = _depth_path(task_dir, camera, ep_idx, chunks_size)
            if not p.exists() or args.overwrite:
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(p, depths=np.zeros((T, H, W), dtype=np.float16))

        if "affordance_vrb" in stages:
            p = _affordance_path(task_dir, camera, ep_idx, chunks_size)
            if not p.exists() or args.overwrite:
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(p, heatmaps=np.zeros((T, H, W), dtype=np.float16))

        if "affordance_px" in stages:
            p = _affordance_px_path(task_dir, camera, ep_idx, chunks_size)
            if not p.exists() or args.overwrite:
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, np.zeros((T, target_resize, target_resize), dtype=np.float16))


# ---------------------------------------------------------------------------
# Stage dispatch — delegate to existing per-stage preprocessors
# ---------------------------------------------------------------------------

def _argv_overwrite(args: Args) -> list[str]:
    return ["--overwrite"] if args.overwrite else []


def _argv_limit(args: Args) -> list[str]:
    return ["--limit", str(args.max_episodes)] if args.max_episodes else []


def _run_depth(task_dir: Path, camera: str, args: Args) -> None:
    """Depth stage: Video-Depth-Anything → depth/chunk-*/cam/episode_*.npz."""
    from runners.preprocess_depth_vda import main as vda_main

    argv = [
        "--dataset_root", str(task_dir),
        "--camera", camera,
        "--encoder", args.vda_encoder,
        "--input_size", str(args.vda_input_size),
    ]
    if args.vda_checkpoint:
        argv += ["--checkpoint", args.vda_checkpoint]
    argv += _argv_overwrite(args) + _argv_limit(args)
    vda_main(argv)


def _run_affordance_vrb(task_dir: Path, camera: str, args: Args) -> None:
    """Affordance VRB stage: GroundingDINO + VRB → affordance/chunk-*/cam/episode_*.npz.

    Requires --objects (text prompt for GroundingDINO, e.g. "bottle,block,gripper").
    """
    if not args.objects:
        raise ValueError(
            "affordance_vrb stage requires --objects (GroundingDINO text prompt). "
            "Example: --objects 'bottle,block,robot gripper'"
        )
    from runners.preprocess_affordance_vrb import main as vrb_main

    argv = [
        "--dataset_root", str(task_dir),
        "--camera", camera,
        "--objects", args.objects,
        "--box_threshold", str(args.box_threshold),
        "--text_threshold", str(args.text_threshold),
        "--k_ratio", str(args.k_ratio),
        "--frame_stride", str(args.frame_stride),
        "--gdino_batch", str(args.gdino_batch),
        "--max_boxes_per_frame", str(args.max_boxes_per_frame),
        "--device", args.device,
    ]
    if args.gdino_path:
        argv += ["--gdino_path", args.gdino_path]
    if args.vrb_ckpt:
        argv += ["--vrb_ckpt", args.vrb_ckpt]
    argv += _argv_overwrite(args) + _argv_limit(args)
    vrb_main(argv)


def _run_affordance_px(task_dir: Path, camera: str, args: Args) -> None:
    """Affordance px stage: area-downsample VRB heatmaps → affordance_px/chunk-*/cam/episode_*.npy."""
    from runners.preprocess_affordance_px import main as px_main

    argv = [
        "--dataset_root", str(task_dir),
        "--camera", camera,
        "--qwen_image_size", str(args.qwen_image_size),
    ]
    argv += _argv_overwrite(args) + _argv_limit(args)
    px_main(argv)


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------

def _process_task(task_dir: Path, cameras: list[str], stages: list[str],
                  is_lite: bool, args: Args) -> None:
    print(f"\n==> {task_dir}")

    for camera in cameras:
        # Check that videos exist for this camera before committing to any stage.
        vid_glob = list((task_dir / "videos").glob(f"chunk-*/{camera}/episode_*.mp4"))
        if not vid_glob:
            print(f"  [SKIP] no videos found for camera {camera}")
            continue

        print(f"  camera: {camera}  ({len(vid_glob)} episodes)")

        if is_lite:
            _write_lite_sidecars(task_dir, camera, stages, args)
            continue

        for stage in stages:
            print(f"  stage: {stage}")
            if stage == "depth":
                _run_depth(task_dir, camera, args)
            elif stage == "affordance_vrb":
                _run_affordance_vrb(task_dir, camera, args)
            elif stage == "affordance_px":
                _run_affordance_px(task_dir, camera, args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: Args) -> None:
    stages_raw = [s.strip() for s in args.stages.split(",")]
    is_lite = "lite" in stages_raw
    stages = ["depth", "affordance_vrb", "affordance_px"] if is_lite else stages_raw

    unknown = set(stages) - {"depth", "affordance_vrb", "affordance_px"}
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Valid: depth, affordance_vrb, affordance_px")

    cameras = [c.strip() for c in args.cameras.split(",")]
    task_dirs = _collect_task_dirs(args)

    print(f"Tasks   : {len(task_dirs)}")
    print(f"Cameras : {cameras}")
    print(f"Stages  : {stages}")
    print(f"Mode    : {'lite (zero-filled)' if is_lite else 'model-based'}")

    if not is_lite and "affordance_vrb" in stages and not args.objects:
        raise ValueError(
            "affordance_vrb stage requires --objects (GroundingDINO text prompt). "
            "Example: --objects 'bottle,block,robot gripper'"
        )

    for task_dir in task_dirs:
        _process_task(task_dir, cameras, stages, is_lite, args)

    print("\n✓ Done.")


if __name__ == "__main__":
    main(tyro.cli(Args))
