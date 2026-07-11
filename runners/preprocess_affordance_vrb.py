"""Batch VRB affordance-heatmap inference over a LeRobot dataset's camera videos.

Per-frame pipeline (ported from third_party/Splat-MOVER
sagesplat/data/utils/affordance_dataloader.py, deterministic rewrite):

    GroundingDINO (fixed open-vocab text prompt -> xyxy boxes)
      -> square crop per box -> VRB contact-point prediction
      -> 50 samples around each contact point
      -> splat + GaussianBlur -> max-normalized heatmap

Writes per-episode compressed npz files mirroring the videos/ layout:

    {dataset_root}/affordance/chunk-XXX/{camera}/episode_XXXXXX.npz  # key "heatmaps"

Output semantics: (T, H, W) float16 in [0, 1] at the RGB resolution;
an all-zero frame means no detection. Producer metadata goes to
affordance/meta.json.

Weights (fully offline, no runtime downloads):
  - GroundingDINO: download `IDEA-Research/grounding-dino-base` 
  - VRB: third_party/Splat-MOVER/sagesplat/vrb/models/model_checkpoint_1249.pth.tar

Usage:
    CUDA_VISIBLE_DEVICES=0 python runners/preprocess_affordance_vrb.py \\
        --dataset_root datasets/task_ABC_D_scene_D_lerobot --camera image
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLATMOVER_ROOT = PROJECT_ROOT / "third_party" / "Splat-MOVER"

DEFAULT_VRB_CKPT = (
    SPLATMOVER_ROOT / "sagesplat" / "vrb" / "models" / "model_checkpoint_1249.pth.tar"
)
DEFAULT_GDINO_PATH = PROJECT_ROOT / "ckpt" / "grounding-dino-base"
# CALVIN scene A-D tabletop objects.
DEFAULT_OBJECTS = (
    "a red block,a blue block,a pink block,a drawer,"
    "a sliding door,a switch,a button"
)
SAMPLES_PER_BOX = 50


def list_camera_videos(dataset_root, camera):
    """Return sorted per-episode mp4 paths for one camera across all chunks."""
    videos_dir = Path(dataset_root) / "videos"
    return sorted(videos_dir.glob(f"chunk-*/{camera}/episode_*.mp4"))


def affordance_output_path(video_path, dataset_root):
    """Map videos/chunk-X/{cam}/episode_Y.mp4 -> affordance/chunk-X/{cam}/episode_Y.npz."""
    rel = Path(video_path).relative_to(Path(dataset_root) / "videos")
    return (Path(dataset_root) / "affordance" / rel).with_suffix(".npz")


def episode_index_from_path(video_path):
    match = re.search(r"episode_(\d+)", Path(video_path).stem)
    if match is None:
        raise ValueError(f"cannot parse episode index from {video_path}")
    return int(match.group(1))


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


def _ensure_splatmover_on_path():
    if str(SPLATMOVER_ROOT) not in sys.path:
        sys.path.insert(0, str(SPLATMOVER_ROOT))


def build_prompt(objects):
    """Join comma-separated object names into GroundingDINO's expected
    lowercase period-separated phrase format."""
    parts = [obj.strip().lower().rstrip(".") for obj in objects.split(",") if obj.strip()]
    if not parts:
        raise ValueError("--objects produced an empty prompt")
    return ". ".join(parts) + "."


def load_gdino(gdino_path, device):
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    gdino_path = str(gdino_path)
    processor = AutoProcessor.from_pretrained(gdino_path, local_files_only=True)
    model = GroundingDinoForObjectDetection.from_pretrained(
        gdino_path, local_files_only=True
    )
    return processor, model.to(device).eval()


def load_vrb(checkpoint, device):
    """Build VRBModel with the exact hyperparameters used by Splat-MOVER
    (affordance_dataloader.py create()) and load the vendored checkpoint."""
    import torch

    _ensure_splatmover_on_path()
    from sagesplat.vrb.networks.model import VRBModel
    from sagesplat.vrb.networks.traj import TrajAffCVAE

    hand_head = TrajAffCVAE(
        in_dim=2 * 5,
        hidden_dim=192,
        latent_dim=4,
        condition_dim=256,
        coord_dim=64,
        traj_len=5,
    )
    model = VRBModel(
        src_in_features=512,
        num_patches=1,
        hidden_dim=192,
        hand_head=hand_head,
        encoder_time_embed_type="sin",
        num_frames_input=10,
        resnet_type="resnet18",
        embed_dim=256,
        coord_dim=64,
        num_heads=8,
        enc_depth=6,
        attn_kp=1,
        attn_kp_fc=1,
        n_maps=5,
    )
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
    return model.to(device).eval()


def build_vrb_transform():
    """Deterministic inference preprocessing (the Splat-MOVER original also
    applied ColorJitter/RandomGrayscale, which are training-time augments)."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def detect_boxes(processor, model, frames, prompt, *, box_threshold, text_threshold,
                 batch_size, max_boxes, device):
    """Run GroundingDINO over frames in batches. Returns one (K, 4) xyxy
    float array per frame (K <= max_boxes, sorted by descending score)."""
    import torch

    height, width = frames[0].shape[:2]
    per_frame_boxes = []
    for start in range(0, len(frames), batch_size):
        chunk = list(frames[start : start + batch_size])
        inputs = processor(
            images=chunk, text=[prompt] * len(chunk), return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(height, width)] * len(chunk),
        )
        for res in results:
            boxes = res["boxes"].cpu().numpy().astype(np.float32)
            scores = res["scores"].cpu().numpy()
            order = np.argsort(-scores)[:max_boxes]
            per_frame_boxes.append(boxes[order])
    return per_frame_boxes


def squarify_box(box, img_w, img_h):
    """Shrink the longer side symmetrically so the box becomes square
    (deterministic version of Splat-MOVER's jittered shrink), then clip.
    Returns int (x1, y1, x2, y2) or None if the crop degenerates."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w > h:
        shrink = (w - h) / 2.0
        x1, x2 = x1 + shrink, x2 - shrink
    else:
        shrink = (h - w) / 2.0
        y1, y2 = y1 + shrink, y2 - shrink
    x1, y1 = max(int(round(x1)), 0), max(int(round(y1)), 0)
    x2, y2 = min(int(round(x2)), img_w), min(int(round(y2)), img_h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def vrb_contact_points(vrb, transform, frame, boxes, rng, device):
    """Predict n_maps=5 contact points per box with VRB, then draw
    SAMPLES_PER_BOX Gaussian samples spread evenly around them (sigma scaled
    to crop size; replaces the original fit-GMM-then-sample(50), same spirit
    but seedable and sklearn-free). Returns (N, 2) float array of (x, y)
    points in original-image coords."""
    import torch
    from PIL import Image

    img_h, img_w = frame.shape[:2]
    crops, origins, sizes = [], [], []
    for box in boxes:
        sq = squarify_box(box, img_w, img_h)
        if sq is None:
            continue
        x1, y1, x2, y2 = sq
        crops.append(transform(Image.fromarray(frame[y1:y2, x1:x2])))
        origins.append((x1, y1))
        sizes.append((x2 - x1, y2 - y1))
    if not crops:
        return np.zeros((0, 2), dtype=np.float32)

    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        _, pred_contact = vrb.inference(batch, None, None)
    # pred_contact is (B, 2, n_maps, 2); [:, 0] is mu: n_maps normalized
    # (x, y) contact points per crop.
    mu = pred_contact[:, 0].cpu().numpy()

    points = []
    for (ox, oy), (cw, ch), centers in zip(origins, sizes, mu):
        centers_px = np.array([ox, oy]) + centers * np.array([cw, ch])
        sigma = 0.05 * min(cw, ch)
        per_center = max(SAMPLES_PER_BOX // len(centers_px), 1)
        noise = rng.normal(0.0, sigma, size=(len(centers_px), per_center, 2))
        points.append((centers_px[:, None, :] + noise).reshape(-1, 2))
    return np.concatenate(points, axis=0).astype(np.float32)


def compute_heatmap(points, img_h, img_w, k_ratio):
    """Splat points, Gaussian-blur, max-normalize to [0, 1] (ported from
    affordance_dataloader.compute_heatmap with standard (x, y) indexing)."""
    import cv2

    heatmap = np.zeros((img_h, img_w), dtype=np.float32)
    if len(points) == 0:
        return heatmap
    xs = np.clip(points[:, 0].astype(int), 0, img_w - 1)
    ys = np.clip(points[:, 1].astype(int), 0, img_h - 1)
    np.add.at(heatmap, (ys, xs), 1.0)
    k_size = int(np.sqrt(img_h * img_w) / k_ratio)
    if k_size % 2 == 0:
        k_size += 1
    heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def process_episode(frames, models, *, prompt, box_threshold, text_threshold,
                    gdino_batch, max_boxes, k_ratio, frame_stride, rng, device):
    """Compute (T, H, W) float32 heatmaps for one episode. With stride > 1
    only every stride-th frame is inferred and the result is repeated forward
    (suitable for static cameras only). Also returns per-inferred-frame boxes
    (for visualization) and the number of inferred frames with detections."""
    processor, gdino, vrb, transform = models
    n_frames, img_h, img_w = frames.shape[:3]

    sub_indices = list(range(0, n_frames, frame_stride))
    sub_frames = [frames[i] for i in sub_indices]
    per_frame_boxes = detect_boxes(
        processor, gdino, sub_frames, prompt,
        box_threshold=box_threshold, text_threshold=text_threshold,
        batch_size=gdino_batch, max_boxes=max_boxes, device=device,
    )

    sub_heatmaps = np.zeros((len(sub_frames), img_h, img_w), dtype=np.float32)
    n_detected = 0
    for i, (frame, boxes) in enumerate(zip(sub_frames, per_frame_boxes)):
        if boxes.shape[0] == 0:
            continue
        points = vrb_contact_points(vrb, transform, frame, boxes, rng, device)
        sub_heatmaps[i] = compute_heatmap(points, img_h, img_w, k_ratio)
        if points.shape[0] > 0:
            n_detected += 1

    if frame_stride == 1:
        heatmaps = sub_heatmaps
    else:
        fill = np.minimum(np.arange(n_frames) // frame_stride, len(sub_frames) - 1)
        heatmaps = sub_heatmaps[fill]
    return heatmaps, per_frame_boxes, n_detected, len(sub_frames)


def save_heatmap_npz(out_path, heatmaps, num_frames):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heatmaps = np.asarray(heatmaps)[:num_frames].astype(np.float16)
    np.savez_compressed(out_path, heatmaps=heatmaps)
    return heatmaps.shape


def save_vis_overlay(frame, heatmap, boxes, out_path):
    """RGB + JET heatmap overlay with GroundingDINO boxes, for eyeballing
    (lets you tell detection failures from VRB prediction failures)."""
    import cv2

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    jet = cv2.applyColorMap(
        (np.clip(heatmap.astype(np.float32), 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    overlay = cv2.addWeighted(bgr, 0.5, jet, 0.5, 0)
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def write_meta_json(dataset_root, *, camera, gdino_path, vrb_ckpt, prompt, args,
                    stats=None):
    aff_root = Path(dataset_root) / "affordance"
    aff_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "producer": "runners/preprocess_affordance_vrb.py",
        "detector": f"GroundingDINO ({gdino_path})",
        "affordance_model": f"VRB ({vrb_ckpt})",
        "prompt": prompt,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "k_ratio": args.k_ratio,
        "frame_stride": args.frame_stride,
        "max_boxes_per_frame": args.max_boxes_per_frame,
        "samples_per_box": SAMPLES_PER_BOX,
        "seed": args.seed,
        "semantics": (
            "per-frame VRB contact heatmap at RGB resolution, max-normalized "
            "to [0, 1]; all-zero frame = no detection. contact_directions "
            "are not exported (heatmap only)."
        ),
        "dtype": "float16",
        "npz_key": "heatmaps",
        "camera": camera,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if stats:
        meta["stats"] = stats
    path = aff_root / "meta.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return path


def process_videos(videos, models, dataset_root, *, prompt, args):
    """Run the pipeline over each video; write float16 npz per episode.

    Returns (n_ok, n_skip, failures, stats). Failures are appended to
    affordance/failures.txt and never abort the loop. Each episode reseeds
    numpy/torch from (seed + episode_index) so resumed runs reproduce."""
    import torch
    from tqdm import tqdm

    aff_root = Path(dataset_root) / "affordance"
    failures_path = aff_root / "failures.txt"
    videos = videos[: args.limit] if args.limit > 0 else videos
    n_ok = n_skip = 0
    n_inferred_frames = n_detected_frames = 0
    failures = []
    for video_path in tqdm(videos, desc="affordance inference"):
        out_path = affordance_output_path(video_path, dataset_root)
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue
        try:
            ep_idx = episode_index_from_path(video_path)
            rng = np.random.default_rng(args.seed + ep_idx)
            torch.manual_seed(args.seed + ep_idx)
            frames = decode_video_frames(video_path)
            heatmaps, per_frame_boxes, n_det, n_sub = process_episode(
                frames, models, prompt=prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                gdino_batch=args.gdino_batch,
                max_boxes=args.max_boxes_per_frame,
                k_ratio=args.k_ratio, frame_stride=args.frame_stride,
                rng=rng, device=args.device,
            )
            save_heatmap_npz(out_path, heatmaps, frames.shape[0])
            n_inferred_frames += n_sub
            n_detected_frames += n_det
            if n_ok < 4 and args.save_vis > 0:
                vis_ids = rng.choice(
                    len(per_frame_boxes),
                    size=min(args.save_vis, len(per_frame_boxes)),
                    replace=False,
                )
                for fi in sorted(int(v) for v in vis_ids):
                    save_vis_overlay(
                        frames[fi * args.frame_stride],
                        heatmaps[fi * args.frame_stride],
                        per_frame_boxes[fi],
                        aff_root / "vis" / f"episode_{ep_idx:06d}_f{fi:04d}.png",
                    )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - per-video isolation by design
            msg = f"{video_path}: {exc!r}"
            failures.append(msg)
            aff_root.mkdir(parents=True, exist_ok=True)
            with open(failures_path, "a") as f:
                f.write(msg + "\n")
    stats = {
        "episodes_ok": n_ok,
        "episodes_skipped": n_skip,
        "episodes_failed": len(failures),
        "inferred_frames": n_inferred_frames,
        "frames_with_detection": n_detected_frames,
        "detection_rate": (
            round(n_detected_frames / n_inferred_frames, 4)
            if n_inferred_frames else None
        ),
    }
    return n_ok, n_skip, failures, stats


def verify_dataset(dataset_root, camera):
    """Compare every episode length in meta against its heatmap npz frame count."""
    lengths, chunks_size, _ = load_episode_meta(dataset_root)
    problems = []
    for ep_idx in sorted(lengths):
        length = lengths[ep_idx]
        npz_path = (
            Path(dataset_root) / "affordance" / f"chunk-{ep_idx // chunks_size:03d}"
            / camera / f"episode_{ep_idx:06d}.npz"
        )
        if not npz_path.exists():
            problems.append(f"missing: {npz_path}")
            continue
        n = np.load(npz_path)["heatmaps"].shape[0]
        if n != length:
            problems.append(f"frame mismatch: {npz_path} has {n} expected {length}")
    return problems


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Batch VRB affordance-heatmap inference over LeRobot camera videos."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--camera", type=str, default="image")
    parser.add_argument("--gdino_path", type=str, default=str(DEFAULT_GDINO_PATH),
                        help="local GroundingDINO HF directory (offline)")
    parser.add_argument("--vrb_ckpt", type=str, default=str(DEFAULT_VRB_CKPT))
    parser.add_argument("--objects", type=str, default=DEFAULT_OBJECTS,
                        help="comma-separated object phrases for the detection prompt")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--k_ratio", type=float, default=6.0,
                        help="Gaussian kernel = sqrt(H*W)/k_ratio (odd-ified)")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="infer every N-th frame and repeat forward; only "
                             "safe for static cameras (do NOT use >1 for wrist)")
    parser.add_argument("--gdino_batch", type=int, default=16)
    parser.add_argument("--max_boxes_per_frame", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run episodes whose npz already exists")
    parser.add_argument("--verify_only", action="store_true",
                        help="only check npz frame counts against meta/episodes.jsonl")
    parser.add_argument("--limit", type=int, default=-1,
                        help="process only the first N videos (smoke test)")
    parser.add_argument("--save_vis", type=int, default=0,
                        help="save N overlay pngs for each of the first 4 processed episodes")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
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
    prompt = build_prompt(args.objects)
    print(f"{len(videos)} videos | prompt={prompt!r} | gdino={args.gdino_path}")
    processor, gdino = load_gdino(args.gdino_path, args.device)
    vrb = load_vrb(args.vrb_ckpt, args.device)
    models = (processor, gdino, vrb, build_vrb_transform())
    write_meta_json(
        args.dataset_root, camera=args.camera, gdino_path=args.gdino_path,
        vrb_ckpt=args.vrb_ckpt, prompt=prompt, args=args,
    )
    n_ok, n_skip, failures, stats = process_videos(
        videos, models, args.dataset_root, prompt=prompt, args=args
    )
    write_meta_json(
        args.dataset_root, camera=args.camera, gdino_path=args.gdino_path,
        vrb_ckpt=args.vrb_ckpt, prompt=prompt, args=args, stats=stats,
    )
    print(
        f"done: ok={n_ok} skipped={n_skip} failed={len(failures)} "
        f"detection_rate={stats['detection_rate']}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
