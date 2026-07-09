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

import argparse
import json
import sys
from datetime import datetime
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


def save_depth_npz(out_path, depths, num_frames):
    """Truncate to num_frames (defends against VDA's internal last-frame
    padding), cast to float16, and write compressed npz under key "depths"."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    depths = np.asarray(depths)[:num_frames].astype(np.float16)
    np.savez_compressed(out_path, depths=depths)
    return depths.shape


def load_episode_meta(dataset_root):
    """Read per-episode lengths, chunks_size, and fps from LeRobot meta files."""
    root = Path(dataset_root)
    info = json.loads((root / "meta" / "info.json").read_text())
    lengths = {}
    with open(root / "meta" / "episodes.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            lengths[rec["episode_index"]] = rec["length"]
    return lengths, info["chunks_size"], info["fps"]


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _ensure_vda_on_path():
    if str(VDA_ROOT) not in sys.path:
        sys.path.insert(0, str(VDA_ROOT))


def load_model(encoder, checkpoint, device):
    """Load the relative-depth VDA model; return an infer_fn closure."""
    import torch

    _ensure_vda_on_path()
    from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(**MODEL_CONFIGS[encoder], metric=False)
    model.load_state_dict(
        torch.load(str(checkpoint), map_location="cpu"), strict=True
    )
    model = model.to(device).eval()

    def infer_fn(frames, fps, input_size):
        depths, _ = model.infer_video_depth(
            frames, fps, input_size=input_size, device=device, fp32=False
        )
        return np.asarray(depths)

    return infer_fn


def save_depth_vis(depths, out_path, fps):
    """Colormapped mp4 for eyeballing (smoke test only, not a training input)."""
    _ensure_vda_on_path()
    from utils.dc_utils import save_video

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(np.asarray(depths, dtype=np.float32), str(out_path), fps=fps, is_depths=True)


def write_meta_json(dataset_root, *, camera, encoder, checkpoint, input_size):
    depth_root = Path(dataset_root) / "depth"
    depth_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "producer": "runners/preprocess_depth_vda.py",
        "model": "Video-Depth-Anything (relative)",
        "encoder": encoder,
        "checkpoint": str(checkpoint),
        "input_size": input_size,
        "precision": "fp16 (torch.autocast)",
        "semantics": (
            "relative inverse depth (disparity), raw model output, unnormalized. "
            "NOTE: already inverse — downstream target prep must NOT apply 1/d again."
        ),
        "dtype": "float16",
        "npz_key": "depths",
        "camera": camera,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = depth_root / "meta.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return path


def process_videos(videos, infer_fn, dataset_root, *, fps, input_size=518,
                   overwrite=False, limit=-1, vis_first_n=0):
    """Run infer_fn over each video; write truncated float16 npz per episode.

    Returns (n_ok, n_skip, failures). Failures are appended to
    depth/failures.txt and never abort the loop. Alignment contract: the
    depth stack must cover every RGB frame (>= T after decode, saved as [:T]).
    """
    from tqdm import tqdm

    depth_root = Path(dataset_root) / "depth"
    failures_path = depth_root / "failures.txt"
    if limit > 0:
        videos = videos[:limit]
    n_ok = n_skip = 0
    failures = []
    for i, video_path in enumerate(tqdm(videos, desc="depth inference")):
        out_path = depth_output_path(video_path, dataset_root)
        if out_path.exists() and not overwrite:
            n_skip += 1
            continue
        try:
            frames = decode_video_frames(video_path)
            depths = np.asarray(infer_fn(frames, fps, input_size))
            if depths.shape[0] < frames.shape[0]:
                raise ValueError(
                    f"depth frames {depths.shape[0]} < rgb frames {frames.shape[0]}"
                )
            save_depth_npz(out_path, depths, frames.shape[0])
            if n_ok < vis_first_n:
                save_depth_vis(
                    depths[: frames.shape[0]],
                    depth_root / "vis" / f"{Path(video_path).stem}_vis.mp4",
                    fps,
                )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - per-video isolation by design
            msg = f"{video_path}: {exc!r}"
            failures.append(msg)
            depth_root.mkdir(parents=True, exist_ok=True)
            with open(failures_path, "a") as f:
                f.write(msg + "\n")
    return n_ok, n_skip, failures


def verify_dataset(dataset_root, camera):
    """Compare every episode length in meta against its depth npz frame count."""
    lengths, chunks_size, _ = load_episode_meta(dataset_root)
    problems = []
    for ep_idx in sorted(lengths):
        length = lengths[ep_idx]
        npz_path = (
            Path(dataset_root) / "depth" / f"chunk-{ep_idx // chunks_size:03d}"
            / camera / f"episode_{ep_idx:06d}.npz"
        )
        if not npz_path.exists():
            problems.append(f"missing: {npz_path}")
            continue
        n = np.load(npz_path)["depths"].shape[0]
        if n != length:
            problems.append(f"frame mismatch: {npz_path} has {n} expected {length}")
    return problems


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Batch VDA relative-depth inference over LeRobot camera videos."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--camera", type=str, default="image")
    parser.add_argument("--encoder", type=str, default="vitl",
                        choices=["vits", "vitb", "vitl"])
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="defaults to VDA checkpoints/video_depth_anything_{encoder}.pth")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run episodes whose npz already exists")
    parser.add_argument("--verify_only", action="store_true",
                        help="only check npz frame counts against meta/episodes.jsonl")
    parser.add_argument("--limit", type=int, default=-1,
                        help="process only the first N videos (smoke test aid)")
    parser.add_argument("--vis_first_n", type=int, default=0,
                        help="save colormapped mp4 for the first N processed videos")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.verify_only:
        problems = verify_dataset(args.dataset_root, args.camera)
        for p in problems:
            print(p)
        print(f"verify: {len(problems)} problem(s)")
        return 1 if problems else 0

    videos = list_camera_videos(args.dataset_root, args.camera)
    if not videos:
        print(f"no videos found under {args.dataset_root}/videos/*/{args.camera}")
        return 1
    _, _, fps = load_episode_meta(args.dataset_root)
    checkpoint = args.checkpoint or (
        VDA_ROOT / "checkpoints" / f"video_depth_anything_{args.encoder}.pth"
    )
    print(f"{len(videos)} videos | encoder={args.encoder} | fps={fps} | ckpt={checkpoint}")
    infer_fn = load_model(args.encoder, checkpoint, device="cuda")
    write_meta_json(
        args.dataset_root, camera=args.camera, encoder=args.encoder,
        checkpoint=checkpoint, input_size=args.input_size,
    )
    n_ok, n_skip, failures = process_videos(
        videos, infer_fn, args.dataset_root, fps=fps,
        input_size=args.input_size, overwrite=args.overwrite,
        limit=args.limit, vis_first_n=args.vis_first_n,
    )
    print(f"done: ok={n_ok} skipped={n_skip} failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
