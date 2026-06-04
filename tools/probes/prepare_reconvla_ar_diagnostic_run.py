#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-run-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower-path", required=True)
    parser.add_argument("--action-stat-path", required=True)
    parser.add_argument(
        "--input-mode",
        choices=["official_compose", "auxvla_compose"],
        default="official_compose",
    )
    parser.add_argument("--single-view-mode", default="concat_vertical")
    parser.add_argument("--action-horizon", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    source_run_dir = Path(args.source_run_dir)
    output_run_dir = Path(args.output_run_dir)
    checkpoint_dir = output_run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    source_stats = source_run_dir / "dataset_statistics.json"
    if not source_stats.exists():
        raise FileNotFoundError(f"Missing source dataset_statistics.json: {source_stats}")

    cfg = OmegaConf.load(args.base_config)
    OmegaConf.set_struct(cfg, False)
    cfg.framework.name = "AuxVLAGR00T"
    cfg.framework.reconvla.model_path = args.model_path
    cfg.framework.reconvla.vision_tower_path = args.vision_tower_path
    cfg.framework.reconvla.inference_mode = "reconvla_ar_normalized"
    cfg.framework.reconvla.ar_input_mode = args.input_mode
    cfg.framework.reconvla.single_view_mode = args.single_view_mode
    cfg.framework.reconvla.action_stat_path = args.action_stat_path
    cfg.framework.reconvla.double_instruction = True
    cfg.framework.reconvla.max_new_tokens = 128
    cfg.framework.reconvla.temperature = 0.0
    cfg.framework.reconvla.top_p = None
    cfg.framework.reconvla.num_beams = 1
    cfg.framework.action_model.action_horizon = int(args.action_horizon)
    cfg.framework.action_model.future_action_window_size = int(args.action_horizon) - 1

    config_path = output_run_dir / "config.yaml"
    stats_path = output_run_dir / "dataset_statistics.json"
    sentinel_ckpt = checkpoint_dir / "reconvla_ar_diagnostic.pt"
    OmegaConf.save(cfg, config_path)
    shutil.copy2(source_stats, stats_path)
    torch.save({}, sentinel_ckpt)

    print(f"config_yaml={config_path}")
    print(f"ckpt_path={sentinel_ckpt}")
    print(f"dataset_statistics={stats_path}")


if __name__ == "__main__":
    main()
