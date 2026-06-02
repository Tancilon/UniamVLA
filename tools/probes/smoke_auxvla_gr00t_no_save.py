"""No-save smoke test for AuxVLAGR00T.

This script loads the ReconVLA backbone, freezes it by default, trains the
GR00T action head for one optimizer step, and prints the key shape/loss/gradient
checks. With --enable_lora, it trains GR00T plus ReconVLA language LoRA adapters
while keeping the non-LoRA backbone, vision tower, and projectors frozen. It
intentionally does not call the main trainer and does not save a model checkpoint.

Example:
    CUDA_VISIBLE_DEVICES=0 python tools/probes/smoke_auxvla_gr00t_no_save.py \
        --config_yaml starVLA/config/training/auxvla_gr00t_libero.yaml \
        --data_root_dir datasets/libero2uam \
        --data_mix uamvla_libero_goal_h8
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat


def ensure_single_process_group(port: int) -> bool:
    """Initialize a CPU process group for dataloader code that calls dist.get_rank()."""
    if not dist.is_available() or dist.is_initialized():
        return False
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    return True


def _is_lora_param(name: str) -> bool:
    return "lora_" in name.lower()


def freeze_backbone(model) -> None:
    lora_enabled = bool(getattr(model.qwen_vl_interface, "lora_enabled", False))
    for name, param in model.qwen_vl_interface.named_parameters():
        param.requires_grad_(lora_enabled and _is_lora_param(name))
    model.qwen_vl_interface.train(lora_enabled)
    model.action_model.train()


def grad_norm(module: torch.nn.Module, device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for param in module.parameters():
        if param.grad is not None:
            total = total + param.grad.detach().float().norm().pow(2)
    return total.sqrt()


def count_named_parameters(module: torch.nn.Module, predicate) -> int:
    return sum(
        param.numel()
        for name, param in module.named_parameters()
        if predicate(name, param)
    )


def grad_norm_named(module: torch.nn.Module, device: torch.device, predicate) -> torch.Tensor:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for name, param in module.named_parameters():
        if predicate(name, param) and param.grad is not None:
            total = total + param.grad.detach().float().norm().pow(2)
    return total.sqrt()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", default="starVLA/config/training/auxvla_gr00t_libero.yaml")
    parser.add_argument("--data_root_dir", default="datasets/libero2uam")
    parser.add_argument("--data_mix", default="uamvla_libero_goal_h8")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--vision_tower_path", default=None)
    parser.add_argument("--output_dir", default="/tmp/auxvla_gr00t_no_save_smoke")
    parser.add_argument("--master_port", type=int, default=29631)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--repeated_diffusion_steps", type=int, default=None)
    parser.add_argument("--enable_lora", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    initialized_pg = ensure_single_process_group(args.master_port)
    try:
        cfg = OmegaConf.load(args.config_yaml)
        cfg.datasets.vla_data.data_root_dir = args.data_root_dir
        cfg.datasets.vla_data.data_mix = args.data_mix
        cfg.datasets.vla_data.per_device_batch_size = 1
        cfg.output_dir = args.output_dir
        cfg.trainer.freeze_modules = None if args.enable_lora else "qwen_vl_interface"
        cfg.trainer.visualization.enabled = False
        cfg.framework.aux_loss_control.enabled = False
        if "lora" not in cfg.framework.reconvla:
            cfg.framework.reconvla.lora = OmegaConf.create(
                {
                    "enabled": False,
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "target_modules": [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                }
            )
        cfg.framework.reconvla.lora.enabled = bool(args.enable_lora)
        if args.model_path:
            cfg.framework.reconvla.model_path = args.model_path
        if args.vision_tower_path:
            cfg.framework.reconvla.vision_tower_path = args.vision_tower_path
        if args.repeated_diffusion_steps is not None:
            cfg.framework.action_model.repeated_diffusion_steps = args.repeated_diffusion_steps
        cfg = apply_config_compat(cfg)

        if not torch.cuda.is_available():
            raise RuntimeError("This smoke test requires a CUDA device.")
        device = torch.device("cuda")

        print(f"config_yaml={args.config_yaml}")
        print(f"data_root_dir={cfg.datasets.vla_data.data_root_dir}")
        print(f"data_mix={cfg.datasets.vla_data.data_mix}")
        print(f"model_path={cfg.framework.reconvla.model_path}")
        print(f"vision_tower_path={cfg.framework.reconvla.vision_tower_path}")
        print(f"lora_enabled={cfg.framework.reconvla.lora.enabled}")
        print(f"output_dir={cfg.output_dir}")

        t0 = time.perf_counter()
        model = build_framework(cfg).to(device).train()
        freeze_backbone(model)
        print(f"model_load_seconds={time.perf_counter() - t0:.2f}")

        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        backbone_trainable = sum(
            param.numel() for param in model.qwen_vl_interface.parameters() if param.requires_grad
        )
        lora_trainable = count_named_parameters(
            model.qwen_vl_interface,
            lambda name, param: param.requires_grad and _is_lora_param(name),
        )
        backbone_base_trainable = count_named_parameters(
            model.qwen_vl_interface,
            lambda name, param: param.requires_grad and not _is_lora_param(name),
        )
        projector_trainable = count_named_parameters(
            model.qwen_vl_interface,
            lambda name, param: param.requires_grad and "mm_projector" in name,
        )
        action_trainable = sum(
            param.numel() for param in model.action_model.parameters() if param.requires_grad
        )
        print(f"trainable_params={trainable / 1e6:.2f}M")
        print(f"action_trainable_params={action_trainable / 1e6:.2f}M")
        print(f"backbone_trainable_params={backbone_trainable}")
        print(f"lora_trainable_params={lora_trainable / 1e6:.2f}M")
        print(f"backbone_base_trainable_params={backbone_base_trainable}")
        print(f"projector_trainable_params={projector_trainable}")
        if args.enable_lora:
            assert lora_trainable > 0, "LoRA is enabled but no adapter parameters are trainable"
            assert backbone_base_trainable == 0, "non-LoRA backbone parameters are trainable"
            assert projector_trainable == 0, "projector parameters are trainable"
        else:
            assert backbone_trainable == 0, "backbone is not fully frozen"
        assert action_trainable > 0, "action head has no trainable parameters"

        dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
        batch = next(iter(dataloader))
        examples = model._prepare_examples(batch)

        if args.enable_lora:
            recon_inputs, hidden = model._encode_reconvla_hidden(examples)
        else:
            with torch.no_grad():
                recon_inputs, hidden = model._encode_reconvla_hidden(examples)
            hidden = hidden.detach()

        hidden_for_action, actions_target, state = model._prepare_gr00t_action_inputs(
            examples,
            hidden,
        )

        repeat = int(cfg.framework.action_model.get("repeated_diffusion_steps", 4))
        hidden_repeated = hidden_for_action.repeat(repeat, 1, 1)
        actions_repeated = actions_target.repeat(repeat, 1, 1)
        state_repeated = state.repeat(repeat, 1, 1) if state is not None else None

        optim_params = [param for param in model.action_model.parameters() if param.requires_grad]
        if args.enable_lora:
            optim_params.extend(
                param
                for name, param in model.qwen_vl_interface.named_parameters()
                if param.requires_grad and _is_lora_param(name)
            )
        optimizer = torch.optim.AdamW(optim_params, lr=args.lr)
        optimizer.zero_grad(set_to_none=True)
        loss = model.action_model(hidden_repeated, actions_repeated, state_repeated)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss}")
        loss.backward()
        action_grad_norm = grad_norm(model.action_model, hidden.device)
        lora_grad_norm = grad_norm_named(
            model.qwen_vl_interface,
            hidden.device,
            lambda name, param: _is_lora_param(name),
        )
        backbone_grad_count = sum(
            param.grad is not None
            for name, param in model.qwen_vl_interface.named_parameters()
            if not _is_lora_param(name)
        )
        optimizer.step()

        synthetic_id = int(cfg.framework.reconvla.synthetic_image_token_id)
        synthetic_counts = (recon_inputs["input_ids"] == synthetic_id).sum(dim=1).tolist()

        print(f"hidden_shape={tuple(hidden.shape)}")
        print(f"hidden_dtype={hidden.dtype}")
        print(f"action_input_dtype={actions_target.dtype}")
        print(f"aux_input_ids_shape={tuple(recon_inputs['input_ids'].shape)}")
        print(f"synthetic_image_tokens={synthetic_counts}")
        print(f"actions_target_shape={tuple(actions_target.shape)}")
        print(f"state_shape={None if state is None else tuple(state.shape)}")
        print(f"loss={loss.detach().float().item():.6f}")
        print(f"action_grad_norm={action_grad_norm.item():.6f}")
        print(f"lora_grad_norm={lora_grad_norm.item():.6f}")
        print(f"backbone_grad_count={backbone_grad_count}")

        assert hidden.shape[-1] == 3584, f"unexpected hidden dim: {hidden.shape}"
        assert synthetic_counts == [729], f"unexpected synthetic image token count: {synthetic_counts}"
        assert tuple(actions_target.shape) == (1, model.action_horizon, 7)
        assert state is None or tuple(state.shape) == (1, 1, 7)
        assert action_grad_norm.item() > 0, "action head did not receive gradients"
        if args.enable_lora:
            assert lora_grad_norm.item() > 0, "LoRA adapters did not receive gradients"
        assert backbone_grad_count == 0, "frozen backbone unexpectedly has gradients"
        print("SMOKE_OK")
    finally:
        if initialized_pg and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
