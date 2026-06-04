"""No-save smoke test for AuxVLAGR00T ReconVLA AR + internal recon mode.

This probe loads one CALVIN batch, forces the framework into the official-style
ReconVLA AR training path, runs one forward/backward pass, and verifies that:

* action labels contain exactly 5 * 7 ReconVLA action tokens,
* raw 15-D robot_obs and raw unnormalized actions are available,
* internal ReconVLA vm_loss is returned,
* only LoRA adapters receive backbone gradients,
* GR00T action head stays frozen.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat


def ensure_single_process_group(port: int) -> bool:
    if not dist.is_available() or dist.is_initialized():
        return False
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", default="starVLA/config/training/auxvla_gr00t_lora.yaml")
    parser.add_argument("--data_root_dir", default="datasets/calvin2uam")
    parser.add_argument("--data_mix", default="uamvla_calvin_abc")
    parser.add_argument("--action_type", default="calvin_rel_action")
    parser.add_argument("--model_path", default="ckpt/pretrain-checkpoint-10388")
    parser.add_argument("--vision_tower_path", default="ckpt/siglip-so400m-patch14-384")
    parser.add_argument(
        "--action_stat_path",
        default="third_party/ReconVLA/reconvla/statistics.yaml",
    )
    parser.add_argument("--output_dir", default="/tmp/auxvla_reconvla_ar_recon_no_save_smoke")
    parser.add_argument("--master_port", type=int, default=29641)
    return parser.parse_args()


def _is_lora_param(name: str) -> bool:
    return "lora_" in name.lower()


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


def grad_count_named(module: torch.nn.Module, predicate) -> int:
    return sum(
        param.grad is not None
        for name, param in module.named_parameters()
        if predicate(name, param)
    )


def named_grad_examples(module: torch.nn.Module, predicate, limit: int = 8) -> list[str]:
    examples: list[str] = []
    for name, param in module.named_parameters():
        if predicate(name, param) and param.grad is not None:
            examples.append(f"{name}:{param.grad.detach().float().norm().item():.3e}")
            if len(examples) >= limit:
                break
    return examples


def force_ar_recon_smoke_cfg(cfg, args):
    OmegaConf.set_struct(cfg, False)
    cfg.datasets.vla_data.data_root_dir = args.data_root_dir
    cfg.datasets.vla_data.data_mix = args.data_mix
    cfg.datasets.vla_data.action_type = args.action_type
    cfg.datasets.vla_data.per_device_batch_size = 1
    cfg.datasets.vla_data.image_resize = 384
    cfg.datasets.vla_data.obs_image_size = 384
    cfg.output_dir = args.output_dir

    cfg.framework.name = "AuxVLAGR00T"
    cfg.framework.reconvla.model_path = args.model_path
    cfg.framework.reconvla.vision_tower_path = args.vision_tower_path
    cfg.framework.reconvla.action_stat_path = args.action_stat_path
    cfg.framework.reconvla.training_mode = "reconvla_ar_recon"
    cfg.framework.reconvla.inference_mode = "reconvla_ar_normalized"
    cfg.framework.reconvla.ar_input_mode = "official_compose"
    cfg.framework.reconvla.action_token_source = "raw_with_reconvla_statistics"
    cfg.framework.reconvla.disable_internal_recon_loss = False
    cfg.framework.reconvla.reconstruct_image = False

    cfg.framework.action_model.action_horizon = 5
    cfg.framework.action_model.future_action_window_size = 4
    cfg.framework.action_model.repeated_diffusion_steps = 1
    if "aux_heads" in cfg.framework and "recon" in cfg.framework.aux_heads:
        cfg.framework.aux_heads.recon.enabled = False
    if "aux_loss_control" in cfg.framework:
        cfg.framework.aux_loss_control.enabled = False

    if "lora" not in cfg.framework.reconvla:
        cfg.framework.reconvla.lora = OmegaConf.create({})
    cfg.framework.reconvla.lora.enabled = True
    cfg.framework.reconvla.lora.r = int(cfg.framework.reconvla.lora.get("r", 32))
    cfg.framework.reconvla.lora.lora_alpha = int(
        cfg.framework.reconvla.lora.get("lora_alpha", 16)
    )
    cfg.framework.reconvla.lora.lora_dropout = float(
        cfg.framework.reconvla.lora.get("lora_dropout", 0.0)
    )
    cfg.framework.reconvla.lora.init_lora_weights = cfg.framework.reconvla.lora.get(
        "init_lora_weights",
        "gaussian",
    )
    cfg.framework.reconvla.lora.bias = "none"
    cfg.framework.reconvla.lora.task_type = "CAUSAL_LM"
    cfg.framework.reconvla.lora.train_mm_projector = True
    cfg.framework.reconvla.lora.train_mm_inv_projector = True
    cfg.framework.reconvla.lora.exclude_modules = [
        "vision_tower",
        "pixel_decoder",
    ]
    cfg.framework.reconvla.lora.target_modules = [
        "all-linear-except-frozen",
    ]
    cfg.trainer.freeze_modules = None
    cfg.trainer.visualization.enabled = False
    return apply_config_compat(cfg)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    initialized_pg = ensure_single_process_group(args.master_port)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("This smoke test requires a CUDA device.")
        cfg = force_ar_recon_smoke_cfg(OmegaConf.load(args.config_yaml), args)
        device = torch.device("cuda")

        print(f"config_yaml={args.config_yaml}")
        print(f"data_root_dir={cfg.datasets.vla_data.data_root_dir}")
        print(f"data_mix={cfg.datasets.vla_data.data_mix}")
        print(f"action_type={cfg.datasets.vla_data.action_type}")
        print(f"model_path={cfg.framework.reconvla.model_path}")
        print(f"vision_tower_path={cfg.framework.reconvla.vision_tower_path}")
        print(f"training_mode={cfg.framework.reconvla.training_mode}")
        print(f"disable_internal_recon_loss={cfg.framework.reconvla.disable_internal_recon_loss}")
        print(f"lora_enabled={cfg.framework.reconvla.lora.enabled}")
        print(f"train_mm_projector={cfg.framework.reconvla.lora.train_mm_projector}")
        print(f"train_mm_inv_projector={cfg.framework.reconvla.lora.train_mm_inv_projector}")

        t0 = time.perf_counter()
        model = build_framework(cfg).to(device).train()
        print(f"model_load_seconds={time.perf_counter() - t0:.2f}")

        action_trainable = sum(
            param.numel() for param in model.action_model.parameters() if param.requires_grad
        )
        lora_trainable = count_named_parameters(
            model.qwen_vl_interface,
            lambda name, param: param.requires_grad and _is_lora_param(name),
        )
        backbone_base_trainable = count_named_parameters(
            model.qwen_vl_interface,
            lambda name, param: param.requires_grad and not _is_lora_param(name),
        )
        mm_projector_lora_trainable = count_named_parameters(
            model.qwen_vl_interface,
            lambda name, param: (
                param.requires_grad
                and _is_lora_param(name)
                and "mm_projector" in name
                and "mm_inv_projector" not in name
            ),
        )
        mm_inv_projector_lora_trainable = count_named_parameters(
            model.qwen_vl_interface,
            lambda name, param: (
                param.requires_grad and _is_lora_param(name) and "mm_inv_projector" in name
            ),
        )
        lm_head_lora_trainable = count_named_parameters(
            model.qwen_vl_interface,
            lambda name, param: (
                param.requires_grad and _is_lora_param(name) and "lm_head" in name
            ),
        )
        vision_tower_lora_trainable = count_named_parameters(
            model.qwen_vl_interface,
            lambda name, param: (
                param.requires_grad and _is_lora_param(name) and "vision_tower" in name
            ),
        )
        pixel_decoder_lora_trainable = count_named_parameters(
            model.qwen_vl_interface,
            lambda name, param: (
                param.requires_grad and _is_lora_param(name) and "pixel_decoder" in name
            ),
        )
        print(f"action_trainable_params={action_trainable / 1e6:.2f}M")
        print(f"lora_trainable_params={lora_trainable / 1e6:.2f}M")
        print(f"backbone_base_trainable_params={backbone_base_trainable}")
        print(f"mm_projector_lora_trainable_params={mm_projector_lora_trainable / 1e6:.2f}M")
        print(
            "mm_inv_projector_lora_trainable_params="
            f"{mm_inv_projector_lora_trainable / 1e6:.2f}M"
        )
        print(f"lm_head_lora_trainable_params={lm_head_lora_trainable / 1e6:.2f}M")
        print(f"vision_tower_lora_trainable_params={vision_tower_lora_trainable}")
        print(f"pixel_decoder_lora_trainable_params={pixel_decoder_lora_trainable}")
        assert action_trainable == 0, "GR00T action head should be frozen in AR training mode"
        assert lora_trainable > 0, "LoRA is enabled but no adapter parameters are trainable"
        assert backbone_base_trainable == 0, "non-LoRA backbone parameters are trainable"
        assert mm_projector_lora_trainable > 0, "mm_projector LoRA parameters are not trainable"
        assert mm_inv_projector_lora_trainable > 0, "mm_inv_projector LoRA parameters are not trainable"
        assert lm_head_lora_trainable > 0, "lm_head LoRA parameters are not trainable"
        assert vision_tower_lora_trainable == 0, "vision_tower should not have trainable LoRA"
        assert pixel_decoder_lora_trainable == 0, "pixel_decoder should not have trainable LoRA"

        dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
        batch = next(iter(dataloader))
        prepared = model._prepare_examples(batch, require_reconvla_target=True)
        raw_robot_obs = model._ar_training_robot_obs(prepared[0])
        raw_action = model._ar_training_action_array(prepared[0])
        ar_inputs = model.qwen_vl_interface.build_reconvla_ar_training_inputs(
            images=model._reconvla_ar_images(prepared, cfg.framework.reconvla.ar_input_mode),
            target_images=[example["image_target"] for example in prepared],
            instructions=[example["lang"] for example in prepared],
            robot_obs=np.stack([raw_robot_obs], axis=0),
            actions=np.stack([raw_action], axis=0),
            input_mode=cfg.framework.reconvla.ar_input_mode,
        )
        action_token_count = int(ar_inputs["labels"].ne(-100).sum().item())
        print(f"raw_robot_obs_shape={raw_robot_obs.shape}")
        print(f"raw_action_shape={raw_action.shape}")
        print(f"ar_input_ids_shape={tuple(ar_inputs['input_ids'].shape)}")
        print(f"ar_labels_shape={tuple(ar_inputs['labels'].shape)}")
        print(f"ar_action_token_count={action_token_count}")
        print(f"ar_images_shape={tuple(ar_inputs['images'].shape)}")
        print(f"ar_target_images_shape={tuple(ar_inputs['target_images'].shape)}")
        assert raw_robot_obs.shape == (15,)
        assert raw_action.shape == (35,)
        assert action_token_count == 35
        assert tuple(ar_inputs["target_images"].shape) == (1, 3, 384, 384)

        model.zero_grad(set_to_none=True)
        out = model(batch, global_step=0)
        loss = out["action_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss}")
        assert "reconvla_vm_loss" in out, "internal ReconVLA vm_loss was not returned"
        loss.backward()

        lora_grad_norm = grad_norm_named(
            model.qwen_vl_interface,
            device,
            lambda name, param: _is_lora_param(name),
        )
        mm_projector_lora_grad_norm = grad_norm_named(
            model.qwen_vl_interface,
            device,
            lambda name, param: (
                _is_lora_param(name)
                and "mm_projector" in name
                and "mm_inv_projector" not in name
            ),
        )
        mm_inv_projector_lora_grad_norm = grad_norm_named(
            model.qwen_vl_interface,
            device,
            lambda name, param: _is_lora_param(name) and "mm_inv_projector" in name,
        )
        lm_head_lora_grad_norm = grad_norm_named(
            model.qwen_vl_interface,
            device,
            lambda name, param: _is_lora_param(name) and "lm_head" in name,
        )
        backbone_grad_count = grad_count_named(
            model.qwen_vl_interface,
            lambda name, param: not _is_lora_param(name),
        )
        action_grad_count = grad_count_named(model.action_model, lambda name, param: True)

        print(f"loss={loss.detach().float().item():.6f}")
        print(f"action_loss_ar={out['action_loss_ar'].detach().float().item():.6f}")
        print(f"reconvla_vm_loss={out['reconvla_vm_loss'].detach().float().item():.6f}")
        print(f"reconvla_action_token_count={int(out['reconvla_action_token_count'].item())}")
        print(f"lora_grad_norm={lora_grad_norm.item():.6f}")
        print(f"mm_projector_lora_grad_norm={mm_projector_lora_grad_norm.item():.6f}")
        print(f"mm_inv_projector_lora_grad_norm={mm_inv_projector_lora_grad_norm.item():.6f}")
        print(f"lm_head_lora_grad_norm={lm_head_lora_grad_norm.item():.6f}")
        print(
            "mm_inv_projector_lora_grad_examples="
            f"{named_grad_examples(model.qwen_vl_interface, lambda name, param: _is_lora_param(name) and 'mm_inv_projector' in name)}"
        )
        print(f"backbone_grad_count={backbone_grad_count}")
        print(f"action_grad_count={action_grad_count}")

        assert int(out["reconvla_action_token_count"].item()) == 35
        assert lora_grad_norm.item() > 0, "LoRA adapters did not receive gradients"
        assert mm_projector_lora_grad_norm.item() > 0, "mm_projector LoRA gradients are zero"
        assert mm_inv_projector_lora_grad_norm.item() > 0, "mm_inv_projector LoRA gradients are zero"
        assert lm_head_lora_grad_norm.item() > 0, "lm_head LoRA gradients are zero"
        assert backbone_grad_count == 0, "frozen non-LoRA backbone parameters received gradients"
        assert action_grad_count == 0, "frozen GR00T action head received gradients"
        print("SMOKE_OK")
    finally:
        if initialized_pg and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
