#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", required=True)
    parser.add_argument("--input-mode", choices=["official_compose", "auxvla_compose"], default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_image(height: int, width: int, channel: int) -> np.ndarray:
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., channel] = ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8)
    image[..., (channel + 1) % 3] = np.broadcast_to(x, (height, width))
    return image


def main():
    args = parse_args()
    config_yaml = Path(args.config_yaml)
    cfg = OmegaConf.load(config_yaml)
    OmegaConf.set_struct(cfg, False)
    apply_config_compat(cfg)
    if args.input_mode is not None:
        cfg.framework.reconvla.ar_input_mode = args.input_mode
    cfg.framework.reconvla.inference_mode = "reconvla_ar_normalized"

    model = build_framework(cfg).to(args.device).eval()
    example = {
        "image": [make_image(200, 200, 0), make_image(84, 84, 2)],
        "lang": "move the slider left",
        "robot_obs": np.zeros(15, dtype=np.float32),
    }
    with torch.inference_mode():
        out = model.predict_action([example])

    actions = np.asarray(out["normalized_actions"], dtype=np.float32)
    print(f"config_yaml={config_yaml}")
    print(f"input_mode={cfg.framework.reconvla.ar_input_mode}")
    print(f"normalized_actions_shape={actions.shape}")
    print(f"normalized_actions_dtype={actions.dtype}")
    print(f"normalized_actions_min={float(actions.min()):.6f}")
    print(f"normalized_actions_max={float(actions.max()):.6f}")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
