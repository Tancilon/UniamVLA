from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CKPT = Path(
    "playground/Checkpoints/uamvla_gr00t_lt_calvin_abc_depth_afford_future"
    "/final_model/pytorch_model.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "playground/Checkpoints/uamvla_gr00t_lt_calvin_abc_depth_afford_future"
    "/eval_aux_viz"
)
DEFAULT_INPUT_DIR = Path(
    "playground/Checkpoints/uamvla_gr00t_lt_calvin_abc_depth_afford_future"
    "/eval_inputs"
)


def _to_rgb_pil(rgb: Any):
    import numpy as np
    import torch
    from PIL import Image

    if torch.is_tensor(rgb):
        if rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError(f"Expected tensor [3,H,W], got {tuple(rgb.shape)}")
        arr = ((rgb.float() * 0.5 + 0.5).clamp(0, 1) * 255).round()
        arr = arr.to(torch.uint8).cpu().permute(1, 2, 0).numpy()
        return Image.fromarray(arr, mode="RGB")

    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB array [H,W,3], got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _map_to_pil(map_01: Any, size: int = 224):
    import numpy as np
    import torch
    from PIL import Image

    if torch.is_tensor(map_01):
        if map_01.ndim == 3:
            map_01 = map_01.squeeze(0)
        image = (
            map_01.detach()
            .float()
            .clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
    else:
        image = (np.asarray(map_01, dtype=np.float32).clip(0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(image, mode="L").convert("RGB").resize((size, size), Image.NEAREST)


def _concat_images_h(images: list[Any]):
    from PIL import Image

    if not images:
        raise ValueError("Expected at least one image")
    pil_images = [image if isinstance(image, Image.Image) else _to_rgb_pil(image) for image in images]
    max_h = max(image.height for image in pil_images)
    resized = []
    for image in pil_images:
        if image.height != max_h:
            new_w = int(image.width * max_h / image.height)
            image = image.resize((new_w, max_h), Image.LANCZOS)
        resized.append(image)
    out = Image.new("RGB", (sum(image.width for image in resized), max_h))
    x = 0
    for image in resized:
        out.paste(image, (x, 0))
        x += image.width
    return out


def _save_png(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(image, "save"):
        image = _to_rgb_pil(image)
    image.convert("RGB").save(path)


def _load_exported_examples(input_dir: Path, num_samples: int) -> list[dict[str, Any]]:
    from PIL import Image

    with open(input_dir / "manifest.json") as f:
        manifest = json.load(f)

    examples = []
    for sample in manifest.get("samples", [])[: max(0, int(num_samples))]:
        static_image = Image.open(input_dir / sample["static_image"]).convert("RGB")
        wrist_image = Image.open(input_dir / sample["wrist_image"]).convert("RGB")
        example = {
            "image": [static_image, wrist_image],
            "lang": sample["lang"],
            "state": sample["state"],
            "_eval_viz_name": (
                f"{int(sample['index']):04d}_seq{int(sample['sequence_index']):03d}"
                f"_subtask{int(sample['subtask_index'])}"
            ),
        }
        examples.append(example)
    return examples


def _predict_aux_images(model: Any, example: dict) -> dict[str, Any]:
    import torch

    selected = model._prepare_examples([example]) if hasattr(model, "_prepare_examples") else [example]
    examples, qwen_inputs, hidden = model._encode_qwen_hidden(selected)
    image_batch = model._make_visualization_image_batch(examples, device=qwen_inputs["input_ids"].device)
    rgb = image_batch[0, 0].detach().cpu()
    rgb_pil = _to_rgb_pil(rgb)

    outputs = {}
    aux_heads = getattr(model, "aux_heads", {})
    input_ids = qwen_inputs["input_ids"]

    if "affordance_px" in aux_heads:
        head = aux_heads["affordance_px"]
        with torch.no_grad():
            pred = head.predict(hidden, {"input_ids": input_ids}).predictions["affordance_maps"][0].detach().cpu()
        outputs["affordance_px"] = _concat_images_h([rgb_pil, head._overlay_map_on_rgb(rgb, pred)])

    if "depth_latent" in aux_heads:
        head = aux_heads["depth_latent"]
        with torch.no_grad():
            pred = head.predict(hidden, {"input_ids": input_ids}).predictions["depth_maps"][0].detach().cpu()
        outputs["depth_latent"] = _concat_images_h([rgb_pil, _map_to_pil(pred, size=rgb_pil.height)])

    if "future_latent" in aux_heads:
        head = aux_heads["future_latent"]
        with torch.no_grad():
            pred = head.predict(hidden, {"input_ids": input_ids}).predictions["future_frames"][0].detach().cpu()
        outputs["future_latent"] = _concat_images_h([rgb_pil, _to_rgb_pil(pred)])

    return outputs


def _load_model(ckpt: Path, device: str, bf16: bool):
    import torch

    from starVLA.model.framework.base_framework import baseframework
    from tools.visualize_uamvla_lt_aux import _attach_vae_for_lt_visualization

    model = baseframework.from_pretrained(str(ckpt)).to(device).eval()
    if bf16:
        model = model.to(torch.bfloat16)
    _attach_vae_for_lt_visualization(model, torch.device(device))
    return model


def run(args: argparse.Namespace) -> int:
    import torch

    model = _load_model(Path(args.ckpt), args.device, args.bf16)
    examples = _load_exported_examples(Path(args.input_dir), args.num_samples)

    output_dir = Path(args.output_dir)
    saved = 0
    with torch.inference_mode():
        for example in examples:
            pred_images = _predict_aux_images(model, example)
            for branch, image in pred_images.items():
                path = output_dir / f"{example['_eval_viz_name']}_{branch}.png"
                _save_png(image, path)
                saved += 1

    print(f"Saved {saved} eval aux visualization PNGs to {output_dir}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize UamGR00T_LT aux predictions on CALVIN eval inputs.")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    return parser


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(run(build_argparser().parse_args()))


if __name__ == "__main__":
    main()
