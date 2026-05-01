"""UamVLA backbone wrapper — folds Qwen3-VL + ModularStateEncoder + state-token
replacement into a single nn.Module with a clean public interface.

Phase 1 supports the Qwen3-VL backbone only (the SigLIP path from the original
UamVLA codebase is not ported). The class exposes:

* ``build_inputs(images, instructions, canonical_state, view_names)`` — produce
  Qwen3-VL forward kwargs (input_ids, pixel_values, image_grid_thw, ...).
* ``forward(...)`` — embed text, encode canonical_state via ``ModularStateEncoder``,
  splice the state embeddings into ``inputs_embeds`` at the special token
  positions, then run the underlying HF model and return its output dict.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from starVLA.model.modules.uamvla.data.chat_template import (
    Qwen3VLChatTemplate,
    expand_image_placeholders_qwen3_vl,
)
from starVLA.model.modules.uamvla.data.embodiment_registry import (
    get_embodiment_config,
)
from starVLA.model.modules.uamvla.state_encoder.modular_state_encoder import (
    ModularStateEncoder,
)
from starVLA.model.modules.uamvla.state_encoder.special_tokens import (
    register_state_tokens,
    register_structural_tokens,
)

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _assert_transformers_version() -> None:
    """Qwen3-VL requires transformers >= 4.57.0."""
    import transformers

    parts = transformers.__version__.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return  # dev/non-semver build; trust import success
    if (major, minor) < (4, 57):
        raise RuntimeError(
            f"Qwen3-VL backbone requires transformers >= 4.57.0, got "
            f"{transformers.__version__}. Install: "
            "pip install 'transformers>=4.57.0'"
        )


def _replace_state_tokens(
    inputs_embeds: torch.Tensor,    # (B, S, h)
    input_ids: torch.Tensor,        # (B, S)
    state_embeds: torch.Tensor,     # (B, n_state_tokens, h)
    embodiment: str,
    tokenizer,
) -> torch.Tensor:
    """Replace special state token positions in inputs_embeds with state encoder output.

    Order: special_tokens[i] -> state_embeds[:, i, :], for i in 0..n-1.
    Each special token must appear exactly once per sample (asserted).

    Returns a new tensor (clone of inputs_embeds with positions replaced) to
    preserve gradient flow through state_embeds.
    """
    special_tokens = get_embodiment_config(embodiment)["special_tokens"]

    if state_embeds.shape[1] != len(special_tokens):
        raise ValueError(
            f"state_embeds has {state_embeds.shape[1]} tokens but registry expects "
            f"{len(special_tokens)} for embodiment '{embodiment}'"
        )

    # Clone to avoid in-place modification of an upstream tensor and to
    # preserve gradient flow through state_embeds into the computation graph.
    new_embeds = inputs_embeds.clone()

    for i, special_token in enumerate(special_tokens):
        special_id = tokenizer.convert_tokens_to_ids(special_token)
        if special_id is None or special_id < 0:
            raise RuntimeError(
                f"Token '{special_token}' not registered with tokenizer "
                f"(convert_tokens_to_ids returned {special_id}). "
                f"Did you call register_state_tokens()?"
            )

        mask = (input_ids == special_id)            # (B, S)
        n_per_sample = mask.sum(dim=1)               # (B,)

        if not (n_per_sample == 1).all():
            raise RuntimeError(
                f"State token '{special_token}' (id={special_id}) appears "
                f"{n_per_sample.tolist()} times per sample, expected exactly 1. "
                f"Check chat_template construction."
            )

        # Inplace assignment on the cloned tensor — preserves gradient flow.
        # Vectorized over batch: nonzero() yields (B,) row and col indices
        # since each special token appears exactly once per sample (asserted).
        row_idx, col_idx = mask.nonzero(as_tuple=True)
        new_embeds[row_idx, col_idx, :] = state_embeds[row_idx, i, :]

    return new_embeds


class _ResizeShim(nn.Module):
    """Tiny adapter exposing the ``register_state_tokens`` contract.

    ``register_state_tokens`` expects an object with ``resize_token_embeddings``
    and ``get_embed_tokens``/``get_input_embeddings``. The HF Qwen3-VL model
    already supports ``resize_token_embeddings`` and ``get_input_embeddings``,
    but we wrap it in a thin shim so the call site stays decoupled from the
    HF API surface (same convention as the source ``BackboneLLM`` accessor).
    """

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self._hf_model = hf_model

    def resize_token_embeddings(self, new_size: int) -> None:
        self._hf_model.resize_token_embeddings(new_size)

    def get_embed_tokens(self) -> nn.Embedding:
        # Qwen3-VL: language_model lives at ``model.model.language_model``.
        # Fall back to top-level get_input_embeddings if the deeper path
        # isn't present (e.g. when the HF layout changes in a future release).
        try:
            return self._hf_model.model.language_model.get_input_embeddings()
        except AttributeError:
            return self._hf_model.get_input_embeddings()


class UamVLABackboneInterface(nn.Module):
    """Single-class assembly of Qwen3-VL + ModularStateEncoder + state injection.

    Public contract (used by the ``UamVLA`` framework class):

    * ``self.config``: HF config (for ``hidden_size`` and other introspection).
    * ``self.tokenizer``: HF tokenizer, post-``register_state_tokens``.
    * ``self.processor``: HF AutoProcessor (used by ``build_inputs``).
    * ``self.image_token_id``: vocab id of ``<|image_pad|>``.
    * ``self.embodiment``: embodiment name (e.g. ``"franka_libero"``).
    * ``build_inputs(images, instructions, canonical_state, view_names)`` →
      dict of Qwen3-VL forward kwargs.
    * ``forward(...)``: runs HF model with state-token injection if
      ``canonical_state`` is provided; returns the HF output object.

    The wrapper only supports the Qwen3-VL backbone path. SigLIP is not ported.
    """

    consumes_pixel_values: bool = True

    def __init__(
        self,
        qwen_model: nn.Module,
        tokenizer,
        processor,
        state_encoder: ModularStateEncoder,
        embodiment: str,
        image_token_id: int,
        chat_template: Optional[Qwen3VLChatTemplate] = None,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.model = qwen_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.state_encoder = state_encoder
        self.embodiment = embodiment
        self.image_token_id = int(image_token_id)
        self.chat_template = chat_template or Qwen3VLChatTemplate()
        self._patch_size = patch_size

    # ------------------------------------------------------------------ Config
    @property
    def config(self):
        """Underlying HF model config (exposes ``hidden_size`` etc.)."""
        return self.model.config

    @property
    def hidden_size(self) -> int:
        text_cfg = getattr(self.model.config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            return int(text_cfg.hidden_size)
        return int(self.model.config.hidden_size)

    # ----------------------------------------------------- Backbone accessors
    def get_embed_tokens(self) -> nn.Embedding:
        try:
            return self.model.model.language_model.get_input_embeddings()
        except AttributeError:
            return self.model.get_input_embeddings()

    def get_lm_head(self) -> nn.Linear:
        return self.model.lm_head

    def get_tokenizer(self):
        return self.tokenizer

    def resize_token_embeddings(self, new_size: int) -> None:
        self.model.resize_token_embeddings(new_size)

    # ------------------------------------------------------------ Input prep
    def build_inputs(
        self,
        images: Sequence[Sequence],
        instructions: Sequence[str],
        canonical_state: Optional[dict] = None,
        view_names: Optional[Sequence[str]] = None,
    ) -> dict:
        """Build a dict of Qwen3-VL forward kwargs from raw inputs.

        Args:
            images: per-sample list of PIL.Image-like view images. Length B,
                each entry is itself a list of length ``num_views``.
            instructions: list of per-sample instruction strings (length B).
            canonical_state: optional batched canonical_state dict mapping
                limb_id to tensors of shape ``(B, ...)``.  Forwarded back
                through the returned kwargs so the caller can pass it to
                ``forward``.  Callers are responsible for batching per-sample
                dicts into this form before calling ``build_inputs``.
            view_names: ordered camera view names (e.g. ``["primary"]``).
                Defaults to numeric placeholders ``["view_0", "view_1", ...]``.

        Returns:
            dict with at least ``input_ids``, ``attention_mask``, ``pixel_values``
            and ``image_grid_thw`` (the kwargs Qwen3-VL ``forward`` accepts).
            If ``canonical_state`` was provided it is also included so a single
            dict can drive ``forward``.
        """
        if len(images) != len(instructions):
            raise ValueError(
                f"images (B={len(images)}) and instructions (B={len(instructions)}) "
                "must have the same length"
            )

        # Default view_names from the first sample's image count.
        if view_names is None:
            n_views = len(images[0]) if len(images) > 0 else 0
            view_names = [f"view_{i}" for i in range(n_views)]
        view_names = list(view_names)
        n_views = len(view_names)
        mismatched = [i for i, s in enumerate(images) if len(s) != n_views]
        if mismatched:
            raise ValueError(
                f"All samples must have len(images[i]) == len(view_names) ({n_views}); "
                f"got per-sample lengths {[len(s) for s in images]} (mismatched indices: {mismatched})"
            )

        # Build chat-template text per sample. The processor will expand
        # <|image_pad|> placeholders to the correct count using image_grid_thw.
        # Note: we emit raw <image> placeholders here and rely on the processor
        # to expand them (Qwen3VLProcessor handles this when given PIL images).
        texts = [
            self._format_chat_text(view_names, instr)
            for instr in instructions
        ]

        # Flatten per-sample image lists into a flat list (Qwen3VLProcessor
        # accepts a flat list of images and an aligned text batch with one
        # <|image_pad|> placeholder per image).
        flat_images: List = []
        for sample_imgs in images:
            flat_images.extend(sample_imgs)

        proc_out = self.processor(
            text=texts,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        )

        # === TEMP ppv-probe (Task 9 step 2) — REMOVE BEFORE COMMIT ===
        try:
            _input_ids = proc_out.get("input_ids", None)
            _image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if _input_ids is not None and _image_token_id is not None:
                _per_view_count = int((_input_ids[0] == _image_token_id).sum())
                _n_views = max(1, len(images[0]) if images else 1)
                logger.info(
                    "[ppv-probe] vision <|image_pad|> tokens in row 0: %d total, "
                    "%d per view (n_views=%d). Expected 400 to match UamVLA.py:181 ppv.",
                    _per_view_count, _per_view_count // _n_views, _n_views,
                )
        except Exception as _e:  # pragma: no cover
            logger.warning("[ppv-probe] failed: %s", _e)
        # === END TEMP ppv-probe ===

        # Normalize to a plain dict so callers can mutate freely.
        kwargs = dict(proc_out)
        if canonical_state is not None:
            kwargs["canonical_state"] = canonical_state
        return kwargs

    def _format_chat_text(
        self,
        view_names: Sequence[str],
        instruction: str,
    ) -> str:
        """Emit a chat-template string with ``<image>`` placeholders intact.

        The HF Qwen3VLProcessor will replace each ``<image>`` with the
        appropriate ``<|image_pad|>`` * patches_per_view block at processing
        time. Our chat template wraps each placeholder in
        ``<|vision_start|>...<|vision_end|>`` so the processor's image
        substitution lands inside the vision block.
        """
        return self.chat_template.format(
            instruction=instruction,
            embodiment=self.embodiment,
            view_names=list(view_names),
        )

    # ------------------------------------------------------------ Forward
    def _derive_mm_token_type_ids(
        self,
        input_ids: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor],
        provided: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Synthesize mm_token_type_ids when the caller bypassed the processor.

        Mirrors ``Qwen3VLProcessor.create_mm_token_type_ids``: zeros (text)
        with 1 (image) at positions where ``input_ids == image_token_id``.
        Skipped when the caller already supplied it, when there's no image
        modality data, or when input_ids is missing.
        """
        if provided is not None:
            return provided
        if input_ids is None or pixel_values is None:
            return None
        mm = torch.zeros_like(input_ids, dtype=torch.int32)
        mm[input_ids == self.image_token_id] = 1
        return mm

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        canonical_state: Optional[dict] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs,
    ):
        """Run the underlying Qwen3-VL model with optional state-token injection.

        Behavior:
        * When ``canonical_state`` is provided, ``input_ids`` are first embedded,
          state encoder output is spliced in at the special-token positions
          via ``_replace_state_tokens``, and ``mm_token_type_ids`` is derived
          from the *original* ``input_ids`` before they're zeroed out.
        * Otherwise, ``input_ids`` (or pre-built ``inputs_embeds``) are passed
          straight through.

        The HF model is invoked via ``self.model.model`` (the inner
        ``Qwen3VLModel``) so that callers receive the last hidden state without
        the LM head projection — aux heads run their own (potentially masked)
        ``lm_head`` pass on top.
        """
        # State injection branch: splice state embeddings into inputs_embeds.
        if canonical_state is not None:
            if input_ids is None:
                raise ValueError(
                    "canonical_state was supplied but input_ids is None — "
                    "state-token replacement requires the original token ids "
                    "to locate special token positions."
                )
            embed_tokens = self.get_embed_tokens()
            base_embeds = embed_tokens(input_ids)
            state_embeds = self.state_encoder(canonical_state)
            inputs_embeds = _replace_state_tokens(
                inputs_embeds=base_embeds,
                input_ids=input_ids,
                state_embeds=state_embeds,
                embodiment=self.embodiment,
                tokenizer=self.tokenizer,
            )
            # Pre-derive mm_token_type_ids while input_ids is still available;
            # then zero out input_ids per HF contract (cannot pass both).
            mm_token_type_ids = self._derive_mm_token_type_ids(
                input_ids, pixel_values, provided=mm_token_type_ids,
            )
            input_ids = None
        else:
            mm_token_type_ids = self._derive_mm_token_type_ids(
                input_ids, pixel_values, provided=mm_token_type_ids,
            )

        # Use the inner Qwen3VLModel (no lm_head). Aux heads run their own
        # head projections on the returned hidden states.
        return self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


def _read_cfg(cfg, *keys, default=None):
    """Tolerant nested-attr lookup that also accepts dict-style configs.

    Supports both OmegaConf/DictConfig (attribute access) and plain dicts
    (item access). Returns ``default`` if any segment is missing.
    """
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        if hasattr(cur, k):
            cur = getattr(cur, k)
        elif isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def build_uamvla_backbone(cfg) -> UamVLABackboneInterface:
    """Factory: load Qwen3-VL + ModularStateEncoder + register state tokens.

    Reads from ``cfg.framework.qwenvl.*`` and ``cfg.framework.embodiment.*``.
    Falls back to ``cfg.framework.qwen3_vl`` / ``cfg.qwenvl`` for compatibility
    with different config layouts.

    Returns a fully assembled ``UamVLABackboneInterface`` ready for a forward
    pass. Embedding resize is performed in-place on the HF model.
    """
    _assert_transformers_version()

    # Lazy-import HF classes so the module imports cleanly on machines without
    # transformers installed (smoke-test friendly).
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        Qwen3VLForConditionalGeneration,
    )

    # Resolve paths/options from cfg with a few aliases.
    base_vlm = (
        _read_cfg(cfg, "framework", "qwenvl", "base_vlm")
        or _read_cfg(cfg, "framework", "qwen3_vl", "base_vlm")
        or _read_cfg(cfg, "framework", "qwenvl", "model_path")
        or _read_cfg(cfg, "framework", "qwen3_vl", "model_path")
        or _read_cfg(cfg, "qwenvl", "base_vlm")
        or _read_cfg(cfg, "qwen3_vl", "model_path")
    )
    if base_vlm is None:
        raise KeyError(
            "build_uamvla_backbone: could not find base_vlm/model_path in cfg "
            "(looked under framework.qwenvl, framework.qwen3_vl)."
        )

    attn_implementation = (
        _read_cfg(cfg, "framework", "qwenvl", "attn_implementation")
        or _read_cfg(cfg, "framework", "qwen3_vl", "attn_implementation")
    )
    dtype_str = (
        _read_cfg(cfg, "framework", "qwenvl", "dtype")
        or _read_cfg(cfg, "framework", "qwen3_vl", "dtype")
        or "bfloat16"
    )
    low_cpu_mem_usage = bool(
        _read_cfg(cfg, "framework", "qwenvl", "low_cpu_mem_usage", default=True)
    )
    embodiment_name = (
        _read_cfg(cfg, "framework", "embodiment", "name")
        or _read_cfg(cfg, "embodiment", "name")
        or _read_cfg(cfg, "embodiment")
    )
    if embodiment_name is None:
        raise KeyError(
            "build_uamvla_backbone: cfg.framework.embodiment.name (or "
            "cfg.embodiment.name) is required."
        )

    torch_dtype = _DTYPE_MAP.get(dtype_str, torch.bfloat16)
    extra_kwargs: dict = {"low_cpu_mem_usage": low_cpu_mem_usage}
    if attn_implementation:
        extra_kwargs["attn_implementation"] = attn_implementation

    logger.info(
        "build_uamvla_backbone: loading Qwen3VL from %s (dtype=%s, attn=%s)",
        base_vlm, dtype_str, attn_implementation or "default",
    )
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_vlm,
        torch_dtype=torch_dtype,
        **extra_kwargs,
    )

    # Tokenizer + processor.
    try:
        processor = AutoProcessor.from_pretrained(base_vlm)
        tokenizer = processor.tokenizer
    except Exception:  # pragma: no cover — processor optional fallback
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(base_vlm)

    # Register state + structural special tokens via shim (resizes embeddings in-place).
    shim = _ResizeShim(qwen_model)
    register_state_tokens(tokenizer, shim)
    register_structural_tokens(tokenizer, shim)

    # State encoder dim = LM hidden size.
    text_cfg = getattr(qwen_model.config, "text_config", None)
    hidden_size = int(
        text_cfg.hidden_size if text_cfg is not None and hasattr(text_cfg, "hidden_size")
        else qwen_model.config.hidden_size
    )
    state_encoder = ModularStateEncoder(
        embodiment=embodiment_name,
        hidden_dim=hidden_size,
    )

    # Image token id — Qwen3-VL exposes it on config. Tokenizer lookup is a
    # fallback that should agree.
    image_token_id = getattr(qwen_model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None or image_token_id < 0:
        raise RuntimeError(
            "Could not resolve Qwen3-VL image_token_id from config or tokenizer."
        )

    # Patch size — used by callers that want to derive image_grid_thw shape.
    vision_cfg = getattr(qwen_model.config, "vision_config", None)
    patch_size = int(getattr(vision_cfg, "patch_size", 16)) if vision_cfg else 16

    return UamVLABackboneInterface(
        qwen_model=qwen_model,
        tokenizer=tokenizer,
        processor=processor,
        state_encoder=state_encoder,
        embodiment=embodiment_name,
        image_token_id=int(image_token_id),
        patch_size=patch_size,
    )


__all__ = [
    "UamVLABackboneInterface",
    "build_uamvla_backbone",
    "_replace_state_tokens",
]
