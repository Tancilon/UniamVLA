from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch

from tools.visualize_uamvla_lt_aux import (
    DEFAULT_CKPT,
    _as_pil_image,
    _ensure_visualization_state,
    build_argparser,
    save_visualizations,
)
from tools.visualize_uamvla_lt_eval_aux import (
    DEFAULT_OUTPUT_DIR as DEFAULT_EVAL_OUTPUT_DIR,
    _concat_images_h,
    _map_to_pil,
    build_argparser as build_eval_argparser,
)


def _load_affordance_token_head(monkeypatch):
    package = types.ModuleType("starVLA.model.modules.uamvla.aux_heads")
    package.__path__ = []
    base = types.ModuleType("starVLA.model.modules.uamvla.aux_heads.base")

    class AuxHead(torch.nn.Module):
        pass

    @dataclass
    class HeadOutput:
        loss: object = None
        metrics: dict | None = None
        predictions: dict | None = None

    base.AuxHead = AuxHead
    base.HeadOutput = HeadOutput
    monkeypatch.setitem(sys.modules, "starVLA.model.modules.uamvla.aux_heads", package)
    monkeypatch.setitem(sys.modules, "starVLA.model.modules.uamvla.aux_heads.base", base)

    path = Path("starVLA/model/modules/uamvla/aux_heads/affordance_token_head.py")
    spec = importlib.util.spec_from_file_location("affordance_token_head_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.AffordanceTokenHead


def _load_future_latent_head(monkeypatch):
    package = types.ModuleType("starVLA.model.modules.uamvla.aux_heads")
    package.__path__ = []
    base = types.ModuleType("starVLA.model.modules.uamvla.aux_heads.base")
    spatial_reader = types.ModuleType("starVLA.model.modules.uamvla.components.spatial_reader")

    class AuxHead(torch.nn.Module):
        pass

    @dataclass
    class HeadOutput:
        loss: object = None
        metrics: dict | None = None
        predictions: dict | None = None

    def slice_image_tokens(hidden, input_ids, image_token_id, patches_per_view, view_idx):
        return hidden[:, :patches_per_view]

    base.AuxHead = AuxHead
    base.HeadOutput = HeadOutput
    spatial_reader.slice_image_tokens = slice_image_tokens
    monkeypatch.setitem(sys.modules, "starVLA.model.modules.uamvla.aux_heads", package)
    monkeypatch.setitem(sys.modules, "starVLA.model.modules.uamvla.aux_heads.base", base)
    monkeypatch.setitem(
        sys.modules,
        "starVLA.model.modules.uamvla.components.spatial_reader",
        spatial_reader,
    )

    path = Path("starVLA/model/modules/uamvla/aux_heads/future_latent_head.py")
    spec = importlib.util.spec_from_file_location("future_latent_head_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.FutureLatentHead, HeadOutput


class _WandbLikeImage:
    def __init__(self, image):
        self.image = image


def test_default_checkpoint_points_to_requested_model():
    assert DEFAULT_CKPT == Path(
        "playground/Checkpoints/uamvla_gr00t_lt_calvin_abc_depth_afford_future"
        "/final_model/pytorch_model.pt"
    )


def test_save_visualizations_filters_lt_aux_heads(tmp_path):
    image = Image.new("RGB", (4, 4), color=(10, 20, 30))
    saved = save_visualizations(
        {
            "viz/uamvla_gr00t/action_0": image,
            "viz/uamvla_gr00t/depth_latent_0": image,
            "viz/uamvla_gr00t/affordance_px_0": _WandbLikeImage(image),
            "viz/uamvla_gr00t/future_latent_0": image,
        },
        tmp_path,
    )

    assert saved == 3
    assert sorted(path.name for path in tmp_path.glob("*.png")) == [
        "0000_viz_uamvla_gr00t_affordance_px_0.png",
        "0001_viz_uamvla_gr00t_depth_latent_0.png",
        "0002_viz_uamvla_gr00t_future_latent_0.png",
    ]


def test_as_pil_image_rejects_unknown_object():
    try:
        _as_pil_image(object())
    except TypeError as exc:
        assert "Unsupported visualization object" in str(exc)
    else:
        raise AssertionError("expected TypeError")


def test_parser_accepts_minimal_args():
    args = build_argparser().parse_args(["--num-samples", "1", "--batch-size", "1"])
    assert args.num_samples == 1
    assert args.batch_size == 1


def test_ensure_visualization_state_adds_zero_state():
    class _CfgDict(dict):
        def __getattr__(self, name):
            return self[name]

    class _Model:
        config = _CfgDict(
            datasets=_CfgDict(vla_data=_CfgDict(full_state_dim=26)),
            framework=_CfgDict(action_model=_CfgDict(state_dim=7)),
        )

    batch = _ensure_visualization_state([{"lang": "open drawer"}], _Model())

    assert batch[0]["lang"] == "open drawer"
    assert batch[0]["state"].shape == (1, 26)
    assert batch[0]["state"].sum() == 0


def test_affordance_overlay_uses_rgb_background(monkeypatch):
    affordance_head = _load_affordance_token_head(monkeypatch)
    rgb = torch.zeros(3, 8, 8)
    rgb[1] = 1.0
    heatmap = torch.zeros(2, 2)
    heatmap[0, 0] = 1.0

    image = affordance_head._overlay_map_on_rgb(rgb, heatmap, alpha=0.75)

    assert image.size == (8, 8)
    red_pixel = image.getpixel((1, 1))
    dark_pixel = image.getpixel((7, 7))
    assert red_pixel[0] > red_pixel[1]
    assert red_pixel[0] > dark_pixel[0]
    assert max(dark_pixel) < 180
    assert abs(dark_pixel[0] - dark_pixel[2]) < 10
    assert dark_pixel[1] - dark_pixel[0] < 25


def test_eval_aux_defaults_are_separate_from_training_viz():
    args = build_eval_argparser().parse_args(["--num-samples", "3"])

    assert args.num_samples == 3
    assert args.output_dir == DEFAULT_EVAL_OUTPUT_DIR
    assert "eval_aux_viz" in str(args.output_dir)


def test_eval_aux_concat_and_map_helpers_make_png(tmp_path):
    rgb = Image.new("RGB", (8, 8), color=(10, 20, 30))
    heat = torch.ones(1, 2, 2)
    heat_image = _map_to_pil(heat, size=8)
    combined = _concat_images_h([rgb, heat_image])
    out = tmp_path / "eval_pred_only.png"
    combined.save(out)

    assert combined.size == (16, 8)
    assert out.exists()


def test_future_latent_current_rgb_conversion(monkeypatch):
    future_head, _head_output = _load_future_latent_head(monkeypatch)
    rgb = torch.tensor(
        [
            [[-1.0, 0.0], [1.0, 2.0]],
            [[-1.0, 0.0], [1.0, 2.0]],
            [[-1.0, 0.0], [1.0, 2.0]],
        ]
    )

    image = future_head._normalized_rgb_to_pil(rgb)

    assert image.size == (2, 2)
    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert image.getpixel((1, 0)) == (128, 128, 128)
    assert image.getpixel((0, 1)) == (255, 255, 255)
    assert image.getpixel((1, 1)) == (255, 255, 255)


def test_future_latent_visualize_includes_current_primary_rgb(monkeypatch):
    future_head, head_output = _load_future_latent_head(monkeypatch)

    vis_draw = types.ModuleType("starVLA.utils.vis_draw")
    spatial_map = types.ModuleType("starVLA.model.modules.uamvla.aux_heads.spatial_map_denoising_head")

    def concat_images_h(images):
        width = sum(image.width for image in images)
        height = max(image.height for image in images)
        out = Image.new("RGB", (width, height))
        x = 0
        for image in images:
            out.paste(image, (x, 0))
            x += image.width
        return out

    class SpatialMapDenoisingHead:
        @staticmethod
        def _maybe_wandb_image(image, caption=None):
            image.caption = caption
            return image

    vis_draw.concat_images_h = concat_images_h
    spatial_map.SpatialMapDenoisingHead = SpatialMapDenoisingHead
    monkeypatch.setitem(sys.modules, "starVLA.utils.vis_draw", vis_draw)
    monkeypatch.setitem(
        sys.modules,
        "starVLA.model.modules.uamvla.aux_heads.spatial_map_denoising_head",
        spatial_map,
    )

    head = future_head(
        hidden_size=4,
        image_token_id=1,
        patches_per_view=1,
        view_idx=0,
        latent_channels=4,
        vae=object(),
    )
    head.predict = lambda hidden, batch: head_output(
        predictions={"future_frames": torch.ones(1, 3, 4, 4)}
    )
    head._latent_to_rgb = lambda z: torch.zeros(1, 3, 4, 4)

    outputs = head.visualize(
        hidden_states=torch.zeros(1, 1, 4),
        batch={
            "input_ids": torch.ones(1, 1, dtype=torch.long),
            "future_latent": torch.zeros(1, 4, 1, 1),
            "image": torch.zeros(1, 1, 3, 4, 4),
            "instruction": ["open drawer"],
        },
        mask=torch.tensor([True]),
        num_samples=1,
    )

    assert len(outputs) == 1
    assert outputs[0].size == (12, 4)
    assert outputs[0].caption == "future_latent: Current RGB | GT future | Pred | open drawer"
