"""Encode RGB video frames into frozen-VAE latents for UamGR00T_LT future supervision.

Reads the per-episode mp4 under ``videos/`` and writes one mmap-able
per-episode npy mirroring the videos/ chunk layout:

    {root}/rgb_latent/chunk-XXX/{cam}/episode_XXXXXX.npy   (T, 64, 7, 7)  fp16

``rgb_latent[k]`` is the VAE latent of frame k; the training loader indexes it
at ``base_index + future_offset`` (clamped to the episode tail), so it serves
any foresight horizon N without re-running this script.  It is the regression
target of FutureLatentHead (Seer-style foresight, LT convention).

Per-frame pipeline (mirrors preprocess_depth_latent.py, minus the disparity
normalisation -- RGB is already bounded):

    frame (H, W, 3) uint8
      -> [0, 1] float
      -> bilinear resize to target_resize (= grid * 16 = 112)
      -> scale to [-1, 1]
      -> frozen VAE encode, latent_dist.mode()                 <-- NOT .sample()
      -> (z - shift) * scaling
      -> 2x2 patch grouping                                    == rgb_latent

Usage:
    CUDA_VISIBLE_DEVICES=0 python runners/preprocess_rgb_latent.py \\
        --dataset_root datasets/task_ABC_D_scene_D_lerobot --limit 10 --report_psnr
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

# 8 (VAE downsample) * 2 (2x2 latent grouping); mirrors _UAMVLA_VAE_TARGET_SCALE.
VAE_TARGET_SCALE = 16


def list_camera_videos(dataset_root, camera):
    """Return sorted per-episode mp4 paths for one camera across all chunks."""
    videos_dir = Path(dataset_root) / "videos"
    return sorted(videos_dir.glob(f"chunk-*/{camera}/episode_*.mp4"))


def output_path(video_path, dataset_root):
    """Map videos/chunk-X/{cam}/episode_Y.mp4 -> rgb_latent/chunk-X/{cam}/episode_Y.npy"""
    rel = Path(video_path).relative_to(Path(dataset_root) / "videos")
    return (Path(dataset_root) / "rgb_latent" / rel).with_suffix(".npy")


def decode_video_frames(video_path):
    """Decode ALL frames as (T, H, W, 3) uint8 RGB via pyav (matches the
    LeRobot training-time decode chain; decord's AV1 support is unreliable)."""
    import av

    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def load_vae(vae_path, device):
    from diffusers import AutoencoderKL

    # TF32 convolutions make the encoder's output depend on the batch size:
    # harmless for generation, but this is a precomputed regression target --
    # it must be exact and reproducible (see preprocess_depth_latent.py).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    vae = AutoencoderKL.from_pretrained(vae_path).float().eval().to(device)
    vae.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_latents(rgb: torch.Tensor, vae, grid: int) -> torch.Tensor:
    """(N, 3, R, R) in [0,1] -> (N, latent_channels*4, grid, grid).

    ``.mode()`` not ``.sample()``: this GT is a regression target, so a
    stochastic sample would hand the head a different label for the same
    frame on every run.
    """
    x = rgb * 2.0 - 1.0
    z = vae.encode(x).latent_dist.mode()
    z = (z - vae.config.shift_factor) * vae.config.scaling_factor
    z = z.unfold(2, 2, 2).unfold(3, 2, 2)
    z = rearrange(z, "b c h w p1 p2 -> b (c p1 p2) h w").contiguous()
    if z.shape[-1] != grid:
        raise RuntimeError(f"latent grid {z.shape[-1]} != token grid {grid}")
    return z


@torch.no_grad()
def decode_latents(z: torch.Tensor, vae) -> torch.Tensor:
    """Inverse of encode_latents, for the --report_psnr sanity gate."""
    z = rearrange(z, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2)
    z = z / vae.config.scaling_factor + vae.config.shift_factor
    px = vae.decode(z).sample
    return (px / 2 + 0.5).clamp(0, 1)


def roundtrip_psnr(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """PSNR between two (N, 3, R, R) RGB batches in [0, 1]."""
    mse = F.mse_loss(pred, gt).item()
    return 10.0 * float(np.log10(1.0 / max(mse, 1e-12)))


def process_episode(video_path, dataset_root, vae, *, grid, target_resize, device,
                    batch_size, overwrite, report_psnr):
    """Encode one episode. Returns (status, psnr_or_None)."""
    latent_path = output_path(video_path, dataset_root)
    if latent_path.exists() and not overwrite:
        return "skip", None

    frames = decode_video_frames(video_path)                       # (T, H, W, 3) uint8
    rgb = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    rgb = F.interpolate(
        rgb, size=(target_resize, target_resize),
        mode="bilinear", align_corners=False,
    )                                                              # (T, 3, R, R)

    latents, psnr = [], None
    for start in range(0, rgb.shape[0], batch_size):
        chunk = rgb[start:start + batch_size].to(device)
        z = encode_latents(chunk, vae, grid)
        latents.append(z.cpu())
        if report_psnr and psnr is None:
            psnr = roundtrip_psnr(chunk.cpu(), decode_latents(z, vae).cpu())
    latent = torch.cat(latents).numpy().astype(np.float16)         # (T, 64, g, g)

    latent_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(latent_path, latent)                                   # uncompressed -> mmap
    return "ok", psnr


def write_meta(dataset_root, *, camera, vae_path, grid, target_resize, vae):
    meta = {
        "producer": "runners/preprocess_rgb_latent.py",
        "source": "videos/ per-episode mp4 (RGB frames)",
        "vae_path": str(vae_path),
        "grid": grid,
        "target_resize": target_resize,
        "latent_channels": int(vae.config.latent_channels) * 4,   # after 2x2 grouping
        "group": 2,
        "normalization": "rgb_unit_to_pm1",
        "posterior": "mode",
        "scaling_factor": float(vae.config.scaling_factor),
        "shift_factor": float(vae.config.shift_factor),
        "camera": camera,
        "dtype": "float16",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = Path(dataset_root) / "rgb_latent" / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def verify_dataset(dataset_root, camera, grid, latent_channels=64):
    """Check every episode's artifact against meta/episodes.jsonl.

    Reports missing and corrupt separately: a partially processed dataset is a
    normal resumable state, a wrong shape is not.  Returns (missing, corrupt).
    """
    root = Path(dataset_root)
    with open(root / "meta" / "episodes.jsonl") as f:
        lengths = {json.loads(l)["episode_index"]: json.loads(l)["length"] for l in f}
    chunks_size = json.loads((root / "meta" / "info.json").read_text())["chunks_size"]

    missing, corrupt = [], []
    for ep, length in sorted(lengths.items()):
        chunk = f"chunk-{ep // chunks_size:03d}"
        path = root / "rgb_latent" / chunk / camera / f"episode_{ep:06d}.npy"
        if not path.exists():
            missing.append(str(path))
            continue
        arr = np.load(path, mmap_mode="r")
        want = (length, latent_channels, grid, grid)
        if arr.shape != want:
            corrupt.append(f"{path}: shape {arr.shape} expected {want}")
    return missing, corrupt


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Encode RGB video frames into frozen-VAE latents.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--camera", type=str, default="image")
    p.add_argument("--vae_path", type=str, default="ckpt/pretrained_vae")
    p.add_argument("--qwen_image_size", type=int, default=224,
                   help="must match framework.qwen_image_size; grid = size // 32")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=-1, help="process only the first N episodes")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report_psnr", action="store_true",
                   help="report encode->decode PSNR per episode")
    p.add_argument("--verify_only", action="store_true",
                   help="only check artifact presence, frame counts and shapes")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args(argv)


def main(argv=None):
    from tqdm import tqdm

    args = parse_args(argv)
    if args.qwen_image_size % 32 != 0:
        raise ValueError(f"qwen_image_size must be a multiple of 32, got {args.qwen_image_size}")
    grid = args.qwen_image_size // 32
    target_resize = grid * VAE_TARGET_SCALE

    if args.verify_only:
        missing, corrupt = verify_dataset(args.dataset_root, args.camera, grid)
        for path in missing[:20]:
            print(f"missing: {path}")
        if len(missing) > 20:
            print(f"... and {len(missing) - 20} more missing")
        for problem in corrupt:
            print(f"corrupt: {problem}")
        print(f"verify: {len(missing)} missing, {len(corrupt)} corrupt")
        return 1 if corrupt else 0  # missing is a resumable state, corrupt is not

    episodes = list_camera_videos(args.dataset_root, args.camera)
    if not episodes:
        print(f"no videos under {args.dataset_root}/videos/*/{args.camera}")
        return 1
    if args.limit > 0:
        episodes = episodes[:args.limit]

    vae = load_vae(args.vae_path, args.device)
    meta = write_meta(args.dataset_root, camera=args.camera, vae_path=args.vae_path,
                      grid=grid, target_resize=target_resize, vae=vae)
    print(f"{len(episodes)} episodes | grid={grid} target_resize={target_resize} "
          f"| latent_channels={meta['latent_channels']} | vae={args.vae_path}")

    n_ok = n_skip = 0
    psnrs = []
    for video_path in tqdm(episodes, desc="rgb latent"):
        status, psnr = process_episode(
            video_path, args.dataset_root, vae, grid=grid, target_resize=target_resize,
            device=args.device, batch_size=args.batch_size,
            overwrite=args.overwrite, report_psnr=args.report_psnr,
        )
        n_ok += status == "ok"
        n_skip += status == "skip"
        if psnr is not None:
            psnrs.append(psnr)

    if psnrs:
        print(f"round-trip @{target_resize}px: PSNR {float(np.mean(psnrs)):.2f} dB -- "
              "below 30 dB means the frozen VAE fails on this domain.")
    print(f"done: ok={n_ok} skipped={n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
