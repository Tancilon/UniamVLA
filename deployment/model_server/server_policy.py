# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import json
import logging
import os
import socket
from pathlib import Path

import torch
from omegaconf import OmegaConf

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework, build_framework
from starVLA.model.framework.share_tools import apply_config_compat


def load_policy_from_config(config_yaml: str | os.PathLike):
    config_yaml = Path(config_yaml)
    cfg = OmegaConf.load(config_yaml)
    apply_config_compat(cfg)
    vla = build_framework(cfg)

    stats_path = config_yaml.parent / "dataset_statistics.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            vla.norm_stats = json.load(f)
    else:
        logging.warning("No dataset_statistics.json found beside config_yaml: %s", stats_path)
    return vla


def load_policy(args):
    if getattr(args, "config_yaml", None):
        return load_policy_from_config(args.config_yaml)
    return baseframework.from_pretrained(args.ckpt_path)


def main(args) -> None:
    vla = load_policy(args)
    if args.use_bf16:  # False
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()
    framework_cfg = getattr(getattr(vla, "config", None), "framework", None)
    framework_name = getattr(framework_cfg, "name", None)
    embodiment_cfg = getattr(framework_cfg, "embodiment", None)
    action_dim = embodiment_cfg.get("action_dim", None) if hasattr(embodiment_cfg, "get") else None
    logging.info(
        "Loaded policy class=%s framework=%s action_horizon=%s action_dim=%s "
        "action_start_id=%s act0_id=%s",
        type(vla).__name__,
        framework_name,
        getattr(vla, "action_horizon", None),
        action_dim,
        getattr(vla, "action_start_id", None),
        getattr(vla, "_act0_id", None),
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help="Eval-only config path. When set, build the policy from config and skip checkpoint state loading.",
    )
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
