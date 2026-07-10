"""Encode VDA disparity into frozen-VAE latents for UamGR00T_LT depth supervision.

Reads the per-episode npz written by ``runners/preprocess_depth_vda.py`` and
writes two mmap-able per-episode npy files mirroring the videos/ chunk layout:

    {root}/depth_latent/chunk-XXX/{cam}/episode_XXXXXX.npy   (T, 64, 7, 7)  fp16
    {root}/depth_px/chunk-XXX/{cam}/episode_XXXXXX.npy       (T, 112, 112)  fp16

``depth_latent`` is the regression target of LatentDepthHead's training head.
``depth_px`` is the target of its detached linear probe -- it must stay in pixel
space, otherwise the probe inherits the VAE encoder's R^2 ~= 0.69 linearity
ceiling and stops measuring the backbone.

Per-frame pipeline (no inversion -- VDA already emits inverse depth):

    disparity (H, W) fp16
      -> per-frame 1%/99% robust normalisation to [0, 1]      == depth_px source
      -> bilinear resize to target_resize (= grid * 16 = 112)
      -> repeat to 3 channels, scale to [-1, 1]
      -> frozen VAE encode, latent_dist.mode()                 <-- NOT .sample()
      -> (z - shift) * scaling
      -> 2x2 patch grouping                                    == depth_latent

Usage:
    CUDA_VISIBLE_DEVICES=0 python runners/preprocess_depth_latent.py \\
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
NORMALIZATION = "per_frame_p1p99"


def list_depth_npz(dataset_root, camera):
    """Return sorted per-episode VDA npz paths for one camera across all chunks."""
    root = Path(dataset_root) / "depth"
    return sorted(root.glob(f"chunk-*/{camera}/episode_*.npz"))


def output_paths(npz_path, dataset_root):
    """Map depth/chunk-X/{cam}/ep.npz -> depth_latent/... and depth_px/... .npy"""
    rel = Path(npz_path).relative_to(Path(dataset_root) / "depth")
    root = Path(dataset_root)
    return (
        (root / "depth_latent" / rel).with_suffix(".npy"),
        (root / "depth_px" / rel).with_suffix(".npy"),
    )


def normalize_disparity(frame: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-frame robust 1%/99% percentile normalisation to [0, 1].

    No reciprocal: VDA emits relative *inverse* depth (disparity) already, so
    near surfaces are large. Inverting again -- as DepthDenoisingHead does for
    metric sim depth -- would flip the semantics.

    Per-frame rather than per-episode: the VLM only ever sees one frame, so it
    cannot infer an episode-level scale. Normalising per episode would inject
    unlearnable variance into the target.
    """
    frame = frame.astype(np.float32)
    lo, hi = np.percentile(frame, 1), np.percentile(frame, 99)
    return np.clip((frame - lo) / max(float(hi - lo), eps), 0.0, 1.0)


def load_vae(vae_path, device):
    from diffusers import AutoencoderKL

    # TF32 convolutions make the encoder's output depend on the batch size:
    # measured max|encode(x, bs=1) - encode(x, bs=32)| = 0.42 on a latent whose
    # std is 0.85.  Harmless for generation, but this is a precomputed
    # regression target -- it must be exact and reproducible.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    vae = AutoencoderKL.from_pretrained(vae_path).float().eval().to(device)
    vae.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_latents(depth_px: torch.Tensor, vae, grid: int) -> torch.Tensor:
    """(N, R, R) in [0,1] -> (N, latent_channels*4, grid, grid).

    Mirrors ReconHead._encode_to_latent, except it uses ``.mode()``: this GT is
    a regression target, so a stochastic ``.sample()`` would hand the head a
    different label for the same frame on every run.
    """
    x = depth_px[:, None].repeat(1, 3, 1, 1) * 2.0 - 1.0
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
    return (px / 2 + 0.5).clamp(0, 1).mean(1)


def roundtrip_metrics(gt: torch.Tensor, pred: torch.Tensor) -> tuple[float, float, float]:
    """PSNR / AbsRel / delta1 between two (N, R, R) maps in [0, 1]."""
    mse = F.mse_loss(pred, gt).item()
    psnr = 10.0 * float(np.log10(1.0 / max(mse, 1e-12)))
    g, p = gt.clamp(1e-2, 1.0), pred.clamp(1e-2, 1.0)
    absrel = ((p - g).abs() / g).mean().item()
    delta1 = (torch.max(p / g, g / p) < 1.25).float().mean().item()
    return psnr, absrel, delta1


def process_episode(npz_path, dataset_root, vae, *, grid, target_resize, device,
                    batch_size, overwrite, report_psnr):
    """Encode one episode. Returns (status, metrics_or_None)."""
    latent_path, px_path = output_paths(npz_path, dataset_root)
    if latent_path.exists() and px_path.exists() and not overwrite:
        return "skip", None

    disparity = np.load(npz_path)["depths"]                       # (T, H, W) fp16
    px = np.stack([normalize_disparity(f) for f in disparity])    # (T, H, W) float32
    px_t = F.interpolate(
        torch.from_numpy(px)[:, None], size=(target_resize, target_resize),
        mode="bilinear", align_corners=False,
    )[:, 0]                                                        # (T, R, R)

    latents, metrics = [], None
    for start in range(0, px_t.shape[0], batch_size):
        chunk = px_t[start:start + batch_size].to(device)
        z = encode_latents(chunk, vae, grid)
        latents.append(z.cpu())
        if report_psnr and metrics is None:
            metrics = roundtrip_metrics(chunk.cpu(), decode_latents(z, vae).cpu())
    latent = torch.cat(latents).numpy().astype(np.float16)         # (T, 64, g, g)

    for path, arr in ((latent_path, latent), (px_path, px_t.numpy().astype(np.float16))):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)                                         # uncompressed -> mmap
    return "ok", metrics


def write_meta(dataset_root, *, camera, vae_path, grid, target_resize, vae):
    meta = {
        "producer": "runners/preprocess_depth_latent.py",
        "source": "runners/preprocess_depth_vda.py (relative inverse depth / disparity)",
        "vae_path": str(vae_path),
        "grid": grid,
        "target_resize": target_resize,
        "latent_channels": int(vae.config.latent_channels) * 4,   # after 2x2 grouping
        "group": 2,
        "normalization": NORMALIZATION,
        "posterior": "mode",
        "scaling_factor": float(vae.config.scaling_factor),
        "shift_factor": float(vae.config.shift_factor),
        "camera": camera,
        "dtype": "float16",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = Path(dataset_root) / "depth_latent" / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def verify_dataset(dataset_root, camera, grid, target_resize, latent_channels=64):
    """Check every episode's two artifacts against meta/episodes.jsonl.

    Reports missing and corrupt separately: a partially processed dataset is a
    normal resumable state, a wrong shape is not.  Returns (missing, corrupt).
    """
    root = Path(dataset_root)
    with open(root / "meta" / "episodes.jsonl") as f:
        lengths = {json.loads(l)["episode_index"]: json.loads(l)["length"] for l in f}
    chunks_size = json.loads((root / "meta" / "info.json").read_text())["chunks_size"]

    expected_shape = {
        "depth_latent": (latent_channels, grid, grid),
        "depth_px": (target_resize, target_resize),
    }
    missing, corrupt = [], []
    for ep, length in sorted(lengths.items()):
        chunk = f"chunk-{ep // chunks_size:03d}"
        for kind, want in expected_shape.items():
            path = root / kind / chunk / camera / f"episode_{ep:06d}.npy"
            if not path.exists():
                missing.append(str(path))
                continue
            arr = np.load(path, mmap_mode="r")
            if arr.shape != (length, *want):
                corrupt.append(f"{path}: shape {arr.shape} expected {(length, *want)}")
    return missing, corrupt


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Encode VDA disparity into frozen-VAE latents.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--camera", type=str, default="image")
    p.add_argument("--vae_path", type=str, default="ckpt/pretrained_vae")
    p.add_argument("--qwen_image_size", type=int, default=224,
                   help="must match framework.qwen_image_size; grid = size // 32")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=-1, help="process only the first N episodes")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report_psnr", action="store_true",
                   help="report encode->decode PSNR/AbsRel/delta1 per episode")
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
        missing, corrupt = verify_dataset(args.dataset_root, args.camera, grid, target_resize)
        for path in missing[:20]:
            print(f"missing: {path}")
        if len(missing) > 20:
            print(f"... and {len(missing) - 20} more missing")
        for problem in corrupt:
            print(f"corrupt: {problem}")
        print(f"verify: {len(missing)} missing, {len(corrupt)} corrupt")
        return 1 if corrupt else 0  # missing is a resumable state, corrupt is not

    episodes = list_depth_npz(args.dataset_root, args.camera)
    if not episodes:
        print(f"no depth npz under {args.dataset_root}/depth/*/{args.camera} "
              f"-- run runners/preprocess_depth_vda.py first")
        return 1
    if args.limit > 0:
        episodes = episodes[:args.limit]

    vae = load_vae(args.vae_path, args.device)
    meta = write_meta(args.dataset_root, camera=args.camera, vae_path=args.vae_path,
                      grid=grid, target_resize=target_resize, vae=vae)
    print(f"{len(episodes)} episodes | grid={grid} target_resize={target_resize} "
          f"| latent_channels={meta['latent_channels']} | vae={args.vae_path}")

    n_ok = n_skip = 0
    reports = []
    for npz_path in tqdm(episodes, desc="depth latent"):
        status, metrics = process_episode(
            npz_path, args.dataset_root, vae, grid=grid, target_resize=target_resize,
            device=args.device, batch_size=args.batch_size,
            overwrite=args.overwrite, report_psnr=args.report_psnr,
        )
        n_ok += status == "ok"
        n_skip += status == "skip"
        if metrics:
            reports.append(metrics)

    if reports:
        psnr, absrel, delta1 = (float(np.mean([r[i] for r in reports])) for i in range(3))
        print(f"round-trip @{target_resize}px: PSNR {psnr:.2f} dB | "
              f"AbsRel {absrel:.4f} | delta1 {delta1:.4f}")
        print("CALVIN reference: 39.00 dB / 0.0345 / 0.9712 -- "
              "PSNR below 30 dB means the frozen VAE fails on this domain.")
    print(f"done: ok={n_ok} skipped={n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
