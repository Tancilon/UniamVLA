#!/usr/bin/env python3
"""
Add UamVLA sidecar files to existing RoboTwin2 LeRobot datasets.

Output format mirrors runners/preprocess_rgb_latent.py and preprocess_depth_vda.py:
  {task_dir}/rgb_latent/chunk-{chunk:03d}/{camera}/episode_{ep:06d}.npy   (T, C, H, W) fp16
  {task_dir}/depth_latent/chunk-{chunk:03d}/{camera}/episode_{ep:06d}.npy (T, C, H, W) fp16
  {task_dir}/depth/chunk-{chunk:03d}/{camera}/episode_{ep:06d}.npz        key="depths" (T,H,W) fp16

Each file covers one full episode and is loaded via np.load(mmap_mode='r') by the model.
The future_latent head reuses rgb_latent at index t+future_offset — no extra sidecar needed.

Supported aux heads once sidecars are written:
  recon          -> rgb_latent
  depth          -> depth/
  depth_latent   -> depth_latent/ (requires depth stage first)
  future_latent  -> rgb_latent    (free: reads at t+N, no extra files)

Unsupported (no simulator / no grounding labels):
  pose, grounding, affordance, action_conditioned_future

Usage:
  # Smoke test (zero-filled, no GPU)
  python runners/convert_robotwin2_lerobot_add_uam_sidecars.py \\
      --task_dir datasets/robotwin2/Clean/adjust_bottle --stages lite

  # Full pipeline, one task
  CUDA_VISIBLE_DEVICES=0 python runners/convert_robotwin2_lerobot_add_uam_sidecars.py \\
      --task_dir datasets/robotwin2/Clean/adjust_bottle \\
      --stages depth,depth_latent,rgb_latent \\
      --vae_path ckpt/pretrained_vae

  # All tasks under Clean/
  CUDA_VISIBLE_DEVICES=0 python runners/convert_robotwin2_lerobot_add_uam_sidecars.py \\
      --dataset_root datasets/robotwin2/Clean \\
      --stages depth,depth_latent,rgb_latent \\
      --vae_path ckpt/pretrained_vae
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
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
    # Input: single task dir OR root dir containing multiple task dirs
    task_dir: str | None = None
    dataset_root: str | None = None

    camera: str = "observation.images.cam_high"

    # Comma-separated subset of: depth, depth_latent, rgb_latent
    # Use "lite" for zero-filled smoke test (no GPU, all three stages)
    stages: str = "depth,depth_latent,rgb_latent"

    # VDA settings
    vda_encoder: Literal["vits", "vitb", "vitl"] = "vitl"
    vda_input_size: int = 518
    vda_checkpoint: str | None = None

    # VAE settings
    vae_path: str = "ckpt/pretrained_vae"
    qwen_image_size: int = 224  # must match training YAML
    vae_batch_size: int = 32

    device: str = "cuda"
    max_episodes: int | None = None
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Sidecar path helpers  (mirror _depth_sidecar_path in UamVLAOFT)
# ---------------------------------------------------------------------------

def _sidecar_path(task_dir: Path, kind: str, camera: str, ep_idx: int, chunks_size: int) -> Path:
    """
    kind: "rgb_latent" | "depth_latent" | "depth"
    Returns: {task_dir}/{kind}/chunk-{chunk:03d}/{camera}/episode_{ep:06d}.npy
    """
    chunk = ep_idx // chunks_size
    suffix = ".npz" if kind == "depth" else ".npy"
    return task_dir / kind / f"chunk-{chunk:03d}" / camera / f"episode_{ep_idx:06d}{suffix}"


# ---------------------------------------------------------------------------
# Video decoding
# ---------------------------------------------------------------------------

def _decode_video(video_path: Path) -> np.ndarray:
    """Return (T, H, W, 3) uint8 RGB via PyAV (matches LeRobot training decode)."""
    import av

    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(container.streams.video[0]):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_vda(encoder: str, input_size: int, checkpoint: str | None, device: str):
    vda_root = PROJECT_ROOT / "third_party" / "Video-Depth-Anything"
    if str(vda_root) not in sys.path:
        sys.path.insert(0, str(vda_root))
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    cfgs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**cfgs[encoder])
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def _load_vae(vae_path: str, device: str):
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(vae_path).to(device).eval()


# ---------------------------------------------------------------------------
# Sidecar computation
# ---------------------------------------------------------------------------

def _compute_depth(frames_rgb: np.ndarray, vda_model, input_size: int, device: str) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (T, H, W) float16 relative inverse depth."""
    import torch
    import torch.nn.functional as F

    T, H, W, _ = frames_rgb.shape
    out = []
    for frame in frames_rgb:
        t = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)
        t = F.interpolate(t.unsqueeze(0), size=(input_size, input_size), mode="bilinear", align_corners=False).to(device)
        with torch.no_grad():
            d = vda_model(t)  # (1, H_out, W_out)
        d = F.interpolate(d.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
        out.append(d.squeeze().cpu().numpy().astype(np.float16))
    return np.stack(out, axis=0)  # (T, H, W)


def _compute_latents(
    frames: np.ndarray, vae, qwen_size: int, batch_size: int, device: str, is_depth: bool
) -> np.ndarray:
    """
    frames: (T, H, W, 3) uint8 for RGB  |  (T, H, W) float16 for depth
    Returns: (T, C, latent_h, latent_w) float16  [channel-first, matches mmap loader]
    """
    import torch
    import torch.nn.functional as F

    latents = []
    T = frames.shape[0]
    for i in range(0, T, batch_size):
        batch = frames[i : i + batch_size]
        if is_depth:
            # (B, H, W) -> (B, 3, qwen_size, qwen_size) via channel replication
            t = torch.from_numpy(batch).float().unsqueeze(1)
            t = F.interpolate(t, size=(qwen_size, qwen_size), mode="bilinear", align_corners=False)
            t = t.repeat(1, 3, 1, 1)
        else:
            # (B, H, W, 3) uint8 -> (B, 3, qwen_size, qwen_size) normalized
            t = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div(255.0)
            t = F.interpolate(t, size=(qwen_size, qwen_size), mode="bilinear", align_corners=False)
        t = (t * 2.0 - 1.0).to(device)
        with torch.no_grad():
            z = vae.encode(t).latent_dist.sample()  # (B, C, latent_h, latent_w)
        latents.append(z.cpu().numpy().astype(np.float16))
    return np.concatenate(latents, axis=0)  # (T, C, latent_h, latent_w)


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------

def _process_task(task_dir: Path, camera: str, stages: list[str], args: Args,
                  vda_model, vae, is_lite: bool) -> None:
    info = json.loads((task_dir / "meta" / "info.json").read_text())
    total_episodes = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)
    if args.max_episodes:
        total_episodes = min(total_episodes, args.max_episodes)

    print(f"\n==> {task_dir.name}  ({total_episodes} episodes)")

    for ep_idx in tqdm(range(total_episodes), desc=f"  {task_dir.name}", leave=False):
        chunk = ep_idx // chunks_size
        video_path = (
            task_dir / "videos" / f"chunk-{chunk:03d}" / camera / f"episode_{ep_idx:06d}.mp4"
        )
        if not video_path.exists():
            tqdm.write(f"    [SKIP] ep {ep_idx}: {video_path.name} not found")
            continue

        # --- depth ---
        if "depth" in stages:
            out_path = _sidecar_path(task_dir, "depth", camera, ep_idx, chunks_size)
            if out_path.exists() and not args.overwrite:
                pass
            else:
                frames_rgb = _decode_video(video_path)
                T, H, W, _ = frames_rgb.shape
                if is_lite:
                    depths = np.zeros((T, H, W), dtype=np.float16)
                else:
                    depths = _compute_depth(frames_rgb, vda_model, args.vda_input_size, args.device)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_path, depths=depths)

        # --- depth_latent ---
        if "depth_latent" in stages:
            out_path = _sidecar_path(task_dir, "depth_latent", camera, ep_idx, chunks_size)
            if out_path.exists() and not args.overwrite:
                pass
            else:
                depth_npz_path = _sidecar_path(task_dir, "depth", camera, ep_idx, chunks_size)
                if not depth_npz_path.exists():
                    tqdm.write(f"    [SKIP] ep {ep_idx}: depth_latent needs depth stage first")
                    continue
                depths = np.load(depth_npz_path)["depths"]  # (T, H, W) float16
                if is_lite:
                    latent_hw = args.qwen_image_size // 8
                    latents = np.zeros((depths.shape[0], 4, latent_hw, latent_hw), dtype=np.float16)
                else:
                    latents = _compute_latents(depths, vae, args.qwen_image_size, args.vae_batch_size,
                                               args.device, is_depth=True)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, latents)

        # --- rgb_latent ---
        if "rgb_latent" in stages:
            out_path = _sidecar_path(task_dir, "rgb_latent", camera, ep_idx, chunks_size)
            if out_path.exists() and not args.overwrite:
                pass
            else:
                frames_rgb = _decode_video(video_path)
                if is_lite:
                    latent_hw = args.qwen_image_size // 8
                    latents = np.zeros((frames_rgb.shape[0], 4, latent_hw, latent_hw), dtype=np.float16)
                else:
                    latents = _compute_latents(frames_rgb, vae, args.qwen_image_size, args.vae_batch_size,
                                               args.device, is_depth=False)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, latents)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: Args) -> None:
    if args.task_dir is None and args.dataset_root is None:
        raise ValueError("Provide --task_dir (single task) or --dataset_root (all tasks).")

    stages_raw = [s.strip() for s in args.stages.split(",")]
    is_lite = "lite" in stages_raw
    stages = ["depth", "depth_latent", "rgb_latent"] if is_lite else stages_raw

    if args.task_dir:
        task_dirs = [Path(args.task_dir)]
    else:
        root = Path(args.dataset_root)
        task_dirs = sorted(d for d in root.iterdir()
                           if d.is_dir() and (d / "meta" / "info.json").exists())

    print(f"Tasks  : {len(task_dirs)}")
    print(f"Camera : {args.camera}")
    print(f"Stages : {', '.join(stages)}")
    print(f"Mode   : {'lite (zero-filled)' if is_lite else 'model-based'}")

    vda_model = vae = None
    if not is_lite:
        if "depth" in stages:
            print("Loading VDA...")
            vda_model = _load_vda(args.vda_encoder, args.vda_input_size, args.vda_checkpoint, args.device)
        if "depth_latent" in stages or "rgb_latent" in stages:
            print("Loading VAE...")
            vae = _load_vae(args.vae_path, args.device)

    for task_dir in task_dirs:
        _process_task(task_dir, args.camera, stages, args, vda_model, vae, is_lite)

    print("\n✓ Done.")


if __name__ == "__main__":
    main(tyro.cli(Args))
