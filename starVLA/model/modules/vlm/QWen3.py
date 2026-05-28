# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

from typing import Optional

import torch
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    153716  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
)


import torch.nn as nn


def _cfg_get(container, key: str, default=None):
    if container is None:
        return default
    if hasattr(container, "get"):
        try:
            return container.get(key, default)
        except TypeError:
            pass
    return getattr(container, key, default)


def _resolve_square_obs_image_size(config) -> int | None:
    framework = _cfg_get(config, "framework", None)
    value = _cfg_get(framework, "qwen_image_size", None)
    if value is None:
        value = _cfg_get(framework, "obs_image_size", None)
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or (
        not isinstance(value, (str, bytes))
        and hasattr(value, "__len__")
        and hasattr(value, "__getitem__")
    ):
        if len(value) != 2:
            raise ValueError(f"obs_image_size must be [S, S], got {value}")
        height, width = int(value[0]), int(value[1])
        if height != width:
            raise ValueError(f"Qwen3-VL expects square obs_image_size here, got {value}")
        return height
    return int(value)


def _fix_processor_pixel_budget(processor, image_size: int | None) -> int | None:
    if image_size is None:
        return None
    pixels = int(image_size) * int(image_size)
    targets = [processor]
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        targets.append(image_processor)
    for target in targets:
        for attr in ("min_pixels", "max_pixels"):
            if hasattr(target, attr):
                setattr(target, attr, pixels)
        size = getattr(target, "size", None)
        if isinstance(size, dict):
            size["shortest_edge"] = pixels
            size["longest_edge"] = pixels
    return pixels


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        trainer_config = getattr(config, "trainer", None)
        enable_grad_ckpt = bool(
            qwenvl_config.get(
                "enable_gradient_checkpointing",
                getattr(trainer_config, "enable_gradient_checkpointing", False),
            )
        )

        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, # resize image no longer needed? @TODO check bug
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"
        self.fixed_image_pixels = _fix_processor_pixel_budget(
            processor,
            _resolve_square_obs_image_size(config),
        )

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size
        # Training/inference forward paths consume full hidden states, not
        # autoregressive KV cache. Keep cache off unless generate() opts in.
        self.model.config.use_cache = False

        if enable_grad_ckpt:
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()
            except Exception as exc:
                print(f"[Qwen3] failed to enable gradient_checkpointing: {exc}", flush=True)

        print(
            f"[Qwen3] use_cache={getattr(self.model.config, 'use_cache', None)}, "
            f"gradient_checkpointing={bool(getattr(self.model, 'is_gradient_checkpointing', False))}",
            flush=True,
        )

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3-VL backbone.
        """

        kwargs.setdefault("use_cache", False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        kwargs.setdefault("use_cache", True)
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = []
            for img in imgs:
                image_content = {"type": "image", "image": img}
                if self.fixed_image_pixels is not None:
                    image_content["min_pixels"] = self.fixed_image_pixels
                    image_content["max_pixels"] = self.fixed_image_pixels
                content.append(image_content)

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        "action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./starVLA/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
