from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from starVLA.model.modules.uamvla.data.embodiment_registry import get_embodiment_config


# Qwen3-VL vocab-internal special tokens (fixed across the 2B/4B/8B family).
# We hardcode them here rather than tokenizer-lookup because the chat template
# runs before tokenization and these IDs are part of Qwen3-VL's published spec.
QWEN3_VL_VISION_START = "<|vision_start|>"
QWEN3_VL_VISION_END = "<|vision_end|>"
QWEN3_VL_IMAGE_PAD = "<|image_pad|>"

# Structural marker for the action chunk in the assistant turn.
# Re-exported from state_encoder.special_tokens (the registration site is
# the single source of truth for the literal string). Used by inference-time
# LogitsProcessors to trigger action-bin masking in generate().
from starVLA.model.modules.uamvla.state_encoder.special_tokens import (  # noqa: E402
    ACTION_START_TOKEN,
)


def expand_image_placeholders(text: str, patches_per_view: int) -> str:
    """Replace each '<image>' occurrence with patches_per_view copies.

    Used by the SigLIP path: each <image> is a registered special token and
    the model expects ppv copies in a row.

    Shared by RobotDataCollator and prepare_inference_inputs so training and
    inference produce identical token layouts.
    """
    if patches_per_view <= 0:
        raise ValueError(f"patches_per_view must be positive, got {patches_per_view}")
    return text.replace("<image>", "<image>" * patches_per_view)


def expand_image_placeholders_qwen3_vl(text: str, patches_per_view: int) -> str:
    """Expand each '<image>' to Qwen3-VL's image_pad block.

    Qwen3-VL expects each image to appear in the input as
    ``<|image_pad|>`` repeated ``patches_per_view`` times. The surrounding
    ``<|vision_start|>`` / ``<|vision_end|>`` markers are emitted by
    Qwen3VLChatTemplate.format itself so this helper deals only with the
    inner image_pad expansion (mirroring the SigLIP-path symmetry).
    """
    if patches_per_view <= 0:
        raise ValueError(f"patches_per_view must be positive, got {patches_per_view}")
    return text.replace("<image>", QWEN3_VL_IMAGE_PAD * patches_per_view)


class ChatTemplate(ABC):
    """Abstract base class for all chat templates."""

    SYSTEM_PROMPT: ClassVar[str]

    @staticmethod
    def _format_state_tokens(embodiment: str) -> str:
        """Look up special tokens for the embodiment and join into a single string.

        Used by concrete subclasses to inject state placeholders at the
        canonical P1 position in their format() output.
        """
        return "".join(get_embodiment_config(embodiment)["special_tokens"])

    @abstractmethod
    def format(
        self,
        instruction: str,
        embodiment: str,
        *,
        view_names: list[str],
    ) -> str:
        ...


_VLA_SYSTEM_PROMPT = (
    "You are a vision-language-action policy. Given an instruction, one or more "
    "camera views, the embodiment type, and the current robot state, predict a "
    "chunk of future actions over the next H timesteps as a sequence of discrete "
    "action tokens. Do not output natural language."
)


class Qwen3ChatTemplate(ChatTemplate):
    SYSTEM_PROMPT = _VLA_SYSTEM_PROMPT

    def format(
        self,
        instruction: str,
        embodiment: str,
        *,
        view_names: list[str],
    ) -> str:
        if not view_names:
            raise ValueError("view_names must be a non-empty list of camera names")
        view_lines = "\n".join(f"View {name}: <image>" for name in view_names)
        # Raises KeyError if embodiment is not registered.
        state_tokens = self._format_state_tokens(embodiment)
        return (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}\n"
            f"Embodiment: {embodiment}\n"
            f"Robot state: {state_tokens}\n"
            f"{view_lines}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n{ACTION_START_TOKEN}"
        )


class Qwen3VLChatTemplate(ChatTemplate):
    """Chat template for Qwen3-VL backbones.

    Wraps each ``<image>`` placeholder with Qwen3-VL's vision delimiters
    ``<|vision_start|>...<|vision_end|>``. The ``<image>`` itself is later
    expanded via :func:`expand_image_placeholders_qwen3_vl` to
    ``<|image_pad|>`` * patches_per_view.

    Note on the assistant prelude: we deliberately do NOT emit an empty
    ``<think>\\n\\n</think>\\n\\n`` block before ``<|action_start|>``. That
    pattern is a Qwen3 *text*-version mechanism for hard-disabling the
    thinking mode (the ``enable_thinking=False`` branch of its tokenizer
    chat_template). Qwen3-VL-Instruct's official chat_template has no such
    branch — its assistant turn starts directly with content after
    ``<|im_start|>assistant\\n``. Adding the empty think block here would
    push training away from the model's pretraining distribution and
    burn ~6 tokens per sample for no gain. Do not add it back.
    """

    SYSTEM_PROMPT = _VLA_SYSTEM_PROMPT

    def format(
        self,
        instruction: str,
        embodiment: str,
        *,
        view_names: list[str],
    ) -> str:
        if not view_names:
            raise ValueError("view_names must be a non-empty list of camera names")
        view_lines = "\n".join(
            f"View {name}: {QWEN3_VL_VISION_START}<image>{QWEN3_VL_VISION_END}"
            for name in view_names
        )
        # Raises KeyError if embodiment is not registered.
        state_tokens = self._format_state_tokens(embodiment)
        return (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}\n"
            f"Embodiment: {embodiment}\n"
            f"Robot state: {state_tokens}\n"
            f"{view_lines}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n{ACTION_START_TOKEN}"
        )
