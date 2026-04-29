"""Special token registration for state encoder.

Strategy: pre-register ALL possible state tokens (Phase 1 _main + Phase 2 _L/_R)
at model construction time, even if only Phase 1 tokens are actively used.
This avoids vocabulary resize during training when adding new embodiments.

Token naming convention:
  <|state_<limb_part>_<side>|>  where side ∈ {main, L, R}
  - main: single-arm robots (Franka in LIBERO/CALVIN)
  - L/R:  bimanual robots (Phase 2 — RoboTwin 2.0)
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# All tokens registered upfront (Phase 1 + Phase 2 forward compat).
# Order is LOAD-BEARING: must match EmbodimentRegistry's special_tokens entries
# and ArmEncoder output order (ee, joint, gripper per side). Do not reorder.
ALL_STATE_SPECIAL_TOKENS: list[str] = [
    # Phase 1 — single-arm
    "<|state_ee_main|>",
    "<|state_joint_main|>",
    "<|state_gripper_main|>",
    # Phase 2 — bimanual left arm
    "<|state_ee_L|>",
    "<|state_joint_L|>",
    "<|state_gripper_L|>",
    # Phase 2 — bimanual right arm
    "<|state_ee_R|>",
    "<|state_joint_R|>",
    "<|state_gripper_R|>",
]


def register_state_tokens(tokenizer, backbone) -> dict[str, int]:
    """Register all state special tokens with the tokenizer and resize backbone embeddings.

    backbone: a BackboneLLM instance (or any object exposing
              resize_token_embeddings(int) and get_embed_tokens()).

    Idempotent: calling multiple times has no effect after the first.

    Returns:
        dict mapping token string → token id (after registration).
    """
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": ALL_STATE_SPECIAL_TOKENS}
    )
    if n_added > 0:
        new_size = len(tokenizer)
        backbone.resize_token_embeddings(new_size)
        logger.info(
            f"[State Encoder] Registered {n_added} new state tokens. "
            f"Resized embeddings to {new_size}."
        )

    # Sanity check via backbone's public API — works on BackboneLLM wrappers
    # (get_embed_tokens) and raw HF models (get_input_embeddings).
    embeds = (backbone.get_embed_tokens() if hasattr(backbone, "get_embed_tokens")
              else backbone.get_input_embeddings())
    embed_size = embeds.weight.shape[0]
    if embed_size != len(tokenizer):
        raise RuntimeError(
            f"Vocab/embedding size mismatch after token registration: "
            f"tokenizer={len(tokenizer)}, embeddings={embed_size}"
        )

    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok)
                 for tok in ALL_STATE_SPECIAL_TOKENS}

    logger.debug(
        "[State Encoder] Token IDs: " +
        ", ".join(f"{tok}={tid}" for tok, tid in token_ids.items())
    )
    return token_ids
