"""Downsample VRB affordance heatmaps into mmap-able per-episode npy files.

Reads the compressed per-episode npz written by
``runners/preprocess_affordance_vrb.py`` and writes one uncompressed npy per
episode mirroring the videos/ chunk layout:

    {root}/affordance_px/chunk-XXX/{cam}/episode_XXXXXX.npy   (T, 112, 112)  fp16

``affordance_px`` is the regression target of AffordanceTokenHead (UamGR00T_LT).
Uncompressed npy so a single training frame is an O(1) mmap read; the source
npz would need a full-episode decompression per random access.

Per-frame pipeline:

    heatmap (H, W) fp16 in [0, 1]
      -> area interpolation to target_resize (= grid * 16 = 112)   # anti-aliased
      -> clip to [0, 1]

No per-frame re-normalisation: rescaling to max=1 would amplify near-zero
noise frames and destroy the "all-zero = no detection" semantics.

Usage:
    python runners/preprocess_affordance_px.py \\
        --dataset_root datasets/task_ABC_D_scene_D_lerobot
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# grid * 16 target, mirroring depth_px (VAE_TARGET_SCALE in preprocess_depth_latent.py).
TARGET_SCALE = 16


def list_affordance_npz(dataset_root, camera):
    """Return sorted per-episode VRB npz paths for one camera across all chunks."""
    root = Path(dataset_root) / "affordance"
    return sorted(root.glob(f"chunk-*/{camera}/episode_*.npz"))


def output_path(npz_path, dataset_root):
    """Map affordance/chunk-X/{cam}/ep.npz -> affordance_px/... .npy"""
    rel = Path(npz_path).relative_to(Path(dataset_root) / "affordance")
    return (Path(dataset_root) / "affordance_px" / rel).with_suffix(".npy")


def process_episode(npz_path, dataset_root, *, target_resize, overwrite):
    """Downsample one episode. Returns (status, zero_frame_count, total_frames)."""
    px_path = output_path(npz_path, dataset_root)
    if px_path.exists() and not overwrite:
        return "skip", 0, 0

    heatmaps = np.load(npz_path)["heatmaps"].astype(np.float32)     # (T, H, W)
    px = F.interpolate(
        torch.from_numpy(heatmaps)[:, None], size=(target_resize, target_resize),
        mode="area",
    )[:, 0].clamp_(0, 1)                                            # (T, R, R)

    px_path.parent.mkdir(parents=True, exist_ok=True)
    arr = px.numpy().astype(np.float16)
    np.save(px_path, arr)                                           # uncompressed -> mmap
    n_zero = int((arr.reshape(arr.shape[0], -1).max(axis=1) <= 0).sum())
    return "ok", n_zero, arr.shape[0]


def write_meta(dataset_root, *, camera, grid, target_resize, stats=None):
    source_meta_path = Path(dataset_root) / "affordance" / "meta.json"
    source_meta = (
        json.loads(source_meta_path.read_text()) if source_meta_path.exists() else None
    )
    meta = {
        "producer": "runners/preprocess_affordance_px.py",
        "source": "runners/preprocess_affordance_vrb.py (VRB contact heatmaps)",
        "grid": grid,
        "target_resize": target_resize,
        "resize": "area",
        "renormalize": False,
        "semantics": (
            "VRB contact heatmap area-downsampled to target_resize, values in "
            "[0, 1]; all-zero frame = no detection in the source."
        ),
        "npz_key": "heatmaps",
        "dtype": "float16",
        "camera": camera,
        "source_meta": source_meta,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if stats:
        meta["stats"] = stats
    path = Path(dataset_root) / "affordance_px" / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return path


def verify_dataset(dataset_root, camera, target_resize):
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
        path = (
            root / "affordance_px" / f"chunk-{ep // chunks_size:03d}"
            / camera / f"episode_{ep:06d}.npy"
        )
        if not path.exists():
            missing.append(str(path))
            continue
        arr = np.load(path, mmap_mode="r")
        want = (length, target_resize, target_resize)
        if arr.shape != want:
            corrupt.append(f"{path}: shape {arr.shape} expected {want}")
    return missing, corrupt


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Downsample VRB affordance heatmaps into mmap-able npy."
    )
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--camera", type=str, default="image")
    p.add_argument("--qwen_image_size", type=int, default=224,
                   help="must match framework.qwen_image_size; grid = size // 32")
    p.add_argument("--limit", type=int, default=-1, help="process only the first N episodes")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify_only", action="store_true",
                   help="only check artifact presence, frame counts and shapes")
    return p.parse_args(argv)


def main(argv=None):
    from tqdm import tqdm

    args = parse_args(argv)
    if args.qwen_image_size % 32 != 0:
        raise ValueError(f"qwen_image_size must be a multiple of 32, got {args.qwen_image_size}")
    grid = args.qwen_image_size // 32
    target_resize = grid * TARGET_SCALE

    if args.verify_only:
        missing, corrupt = verify_dataset(args.dataset_root, args.camera, target_resize)
        for path in missing[:20]:
            print(f"missing: {path}")
        if len(missing) > 20:
            print(f"... and {len(missing) - 20} more missing")
        for problem in corrupt:
            print(f"corrupt: {problem}")
        print(f"verify: {len(missing)} missing, {len(corrupt)} corrupt")
        return 1 if corrupt else 0  # missing is a resumable state, corrupt is not

    episodes = list_affordance_npz(args.dataset_root, args.camera)
    if not episodes:
        print(f"no affordance npz under {args.dataset_root}/affordance/*/{args.camera} "
              f"-- run runners/preprocess_affordance_vrb.py first")
        return 1
    if args.limit > 0:
        episodes = episodes[:args.limit]

    print(f"{len(episodes)} episodes | grid={grid} target_resize={target_resize}")
    n_ok = n_skip = n_zero = n_frames = 0
    for npz_path in tqdm(episodes, desc="affordance px"):
        status, zero, total = process_episode(
            npz_path, args.dataset_root,
            target_resize=target_resize, overwrite=args.overwrite,
        )
        n_ok += status == "ok"
        n_skip += status == "skip"
        n_zero += zero
        n_frames += total

    if n_ok > 0:
        stats = {
            "episodes_ok": n_ok,
            "episodes_skipped": n_skip,
            "frames_written": n_frames,
            "all_zero_frames": n_zero,
            "all_zero_ratio": round(n_zero / n_frames, 4) if n_frames else None,
        }
    else:
        # Nothing processed (all skipped): keep the previous run's stats
        # instead of overwriting them with zeros.
        prev = Path(args.dataset_root) / "affordance_px" / "meta.json"
        stats = json.loads(prev.read_text()).get("stats") if prev.exists() else None
    write_meta(args.dataset_root, camera=args.camera, grid=grid,
               target_resize=target_resize, stats=stats)
    print(f"done: ok={n_ok} skipped={n_skip} "
          f"all_zero_frames={n_zero}/{n_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
