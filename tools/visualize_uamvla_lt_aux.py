from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


DEFAULT_CKPT = Path(
    "playground/Checkpoints/uamvla_gr00t_lt_calvin_abc_depth_afford_future"
    "/final_model/pytorch_model.pt"
)
LT_HEADS = ("depth_latent", "affordance_px", "future_latent")


def _as_pil_image(value: Any) -> Image.Image:
    """Return the PIL image inside a visualization object."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    image = getattr(value, "image", None)
    if isinstance(image, Image.Image):
        return image
    image = getattr(value, "_image", None)
    if isinstance(image, Image.Image):
        return image
    raise TypeError(f"Unsupported visualization object: {type(value)!r}")


def save_visualizations(viz: dict[str, Any], output_dir: Path, start_index: int = 0) -> int:
    """Save LT aux visualizations and return the number of PNG files written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for key, value in sorted(viz.items()):
        if not any(f"/{head}_" in key for head in LT_HEADS):
            continue
        image = _as_pil_image(value).convert("RGB")
        safe_key = key.replace("/", "_").replace(":", "_")
        image.save(output_dir / f"{start_index + written:04d}_{safe_key}.png")
        written += 1
    return written


def _attach_vae_for_lt_visualization(model: Any, device: Any) -> None:
    """Attach the frozen VAE required by latent depth/future visualization."""
    from starVLA.model.modules.uamvla.components.pixel_decoder.vae import VAEPixelDecoder

    aux_heads = getattr(model, "aux_heads", {})
    needs_vae = any(name in aux_heads for name in ("depth_latent", "future_latent"))
    if not needs_vae:
        return

    vae_path = str(model.config.framework.vae.path)
    vae = VAEPixelDecoder(vae_path).to(device).eval()
    model.vae = vae
    for name in ("depth_latent", "future_latent"):
        if name in aux_heads:
            aux_heads[name].vae = vae


def _build_eval_loader(model: Any, batch_size: int, num_workers: int) -> Any:
    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
    from torch.utils.data import DataLoader

    cfg = model.config
    cfg.datasets.vla_data.per_device_batch_size = int(batch_size)
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        collate_fn=collate_fn,
        num_workers=int(num_workers),
        shuffle=False,
    )


def _state_dim_for_visualization(model: Any) -> int:
    vla_data = model.config.datasets.vla_data
    state_dim = int(
        vla_data.get(
            "full_state_dim",
            vla_data.get("state_dim", model.config.framework.action_model.get("state_dim", 7)),
        )
    )
    aux_state_slice = getattr(model, "aux_state_slice", {}) or {}
    for value in aux_state_slice.values():
        if len(value) == 2:
            state_dim = max(state_dim, int(value[1]))
    for idx in vla_data.get("gr00t_state_indices", []) or []:
        state_dim = max(state_dim, int(idx) + 1)
    return state_dim


def _ensure_visualization_state(batch: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    """Add a zero state when the dataset omits it but UAM sidecar unpacking needs it."""
    import numpy as np

    state_dim = _state_dim_for_visualization(model)
    out = []
    for sample in batch:
        if "state" in sample:
            out.append(sample)
        else:
            out.append({**sample, "state": np.zeros((1, state_dim), dtype=np.float32)})
    return out


def run(args: argparse.Namespace) -> int:
    import torch

    from starVLA.model.framework.base_framework import baseframework

    ckpt = Path(args.ckpt)
    output_dir = Path(args.output_dir) if args.output_dir else ckpt.parents[1] / "aux_viz"
    device = torch.device(args.device)

    model = baseframework.from_pretrained(str(ckpt))
    model = model.to(device)
    if args.bf16:
        model = model.to(torch.bfloat16)
    model.eval()
    _attach_vae_for_lt_visualization(model, device)

    loader = _build_eval_loader(model, args.batch_size, args.num_workers)

    saved = 0
    seen_samples = 0
    with torch.inference_mode():
        for batch in loader:
            remaining = int(args.num_samples) - seen_samples
            if remaining <= 0:
                break
            n = min(len(batch), remaining)
            selected = _ensure_visualization_state(batch[:n], model)
            viz = model.visualize_batch(selected, n_samples=n)
            saved += save_visualizations(viz, output_dir, start_index=saved)
            seen_samples += n

    print(f"Saved {saved} aux visualization PNGs to {output_dir}")
    if saved == 0:
        print(
            "No LT aux images were saved. Check that sidecars exist and the batch has "
            "valid depth_latent/affordance_px/future_latent masks."
        )
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save GT-vs-pred PNGs for UamGR00T_LT aux heads.",
    )
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    return parser


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(run(build_argparser().parse_args()))


if __name__ == "__main__":
    main()
