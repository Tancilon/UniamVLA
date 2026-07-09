"""Batch Video-Depth-Anything inference over a LeRobot dataset's camera videos.

Writes per-episode compressed npz depth files mirroring the videos/ layout:

    {dataset_root}/depth/chunk-XXX/{camera}/episode_XXXXXX.npz   # key "depths"

Output semantics: relative inverse depth (disparity), raw model output,
unnormalized, float16. Producer metadata is written to depth/meta.json.

Spec: docs/superpowers/specs/2026-07-08-calvin-vda-depth-preprocess-design.md

Usage:
    CUDA_VISIBLE_DEVICES=0 python runners/preprocess_depth_vda.py \\
        --dataset_root datasets/task_ABC_D_scene_D_lerobot
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VDA_ROOT = PROJECT_ROOT / "third_party" / "Video-Depth-Anything"


def list_camera_videos(dataset_root, camera):
    """Return sorted per-episode mp4 paths for one camera across all chunks."""
    videos_dir = Path(dataset_root) / "videos"
    return sorted(videos_dir.glob(f"chunk-*/{camera}/episode_*.mp4"))


def depth_output_path(video_path, dataset_root):
    """Map videos/chunk-X/{cam}/episode_Y.mp4 -> depth/chunk-X/{cam}/episode_Y.npz."""
    rel = Path(video_path).relative_to(Path(dataset_root) / "videos")
    return (Path(dataset_root) / "depth" / rel).with_suffix(".npz")


def save_depth_npz(out_path, depths, num_frames):
    """Truncate to num_frames (defends against VDA's internal last-frame
    padding), cast to float16, and write compressed npz under key "depths"."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    depths = np.asarray(depths)[:num_frames].astype(np.float16)
    np.savez_compressed(out_path, depths=depths)
    return depths.shape
