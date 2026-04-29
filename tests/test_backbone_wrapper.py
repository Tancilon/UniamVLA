"""Smoke + unit tests for backbone_wrapper.

The full ``build_uamvla_backbone`` factory needs the Qwen3-VL-8B-Instruct
checkpoint plus a CUDA device, so it is gated behind importorskip. Mac-side
coverage is the module-import smoke test plus a pure-tensor unit test for
``_replace_state_tokens`` which exercises the only piece of logic added by
the wrapper that doesn't touch HF.
"""
from __future__ import annotations

import pytest


def test_backbone_wrapper_module_imports():
    """Module + public factory must import cleanly without any HF model load."""
    from starVLA.model.modules.uamvla.backbone_wrapper import (
        UamVLABackboneInterface,
        build_uamvla_backbone,
        _replace_state_tokens,
    )

    assert callable(build_uamvla_backbone)
    assert callable(_replace_state_tokens)
    assert UamVLABackboneInterface is not None


def test__replace_state_tokens_unit():
    """Unit test for the state-token splice helper.

    Constructs a fake tokenizer that maps three special tokens to known ids,
    builds an ``input_ids`` batch with each token at distinct positions, and
    verifies the helper splices ``state_embeds`` into the right rows/columns
    while leaving the rest of ``inputs_embeds`` untouched.
    """
    torch = pytest.importorskip("torch")

    from starVLA.model.modules.uamvla.backbone_wrapper import _replace_state_tokens

    # Mirror the franka_libero registry order.
    special_tokens = [
        "<|state_ee_main|>",
        "<|state_joint_main|>",
        "<|state_gripper_main|>",
    ]
    token_ids = {tok: 1000 + i for i, tok in enumerate(special_tokens)}

    class _FakeTokenizer:
        def convert_tokens_to_ids(self, tok):
            return token_ids[tok]

    tokenizer = _FakeTokenizer()

    B, S, H = 2, 8, 4
    n_state = 3

    # Place state tokens at columns 1, 3, 5 in every sample. The remaining
    # positions hold a random "filler" id (=42).
    input_ids = torch.full((B, S), 42, dtype=torch.long)
    state_positions = [1, 3, 5]
    for i, tok in enumerate(special_tokens):
        input_ids[:, state_positions[i]] = token_ids[tok]

    # Distinguishable embeddings so we can verify positional swaps.
    inputs_embeds = torch.arange(B * S * H, dtype=torch.float32).reshape(B, S, H)
    state_embeds = torch.full((B, n_state, H), -1.0, dtype=torch.float32)
    # Mark each state-embed slot with a unique signature.
    for b in range(B):
        for j in range(n_state):
            state_embeds[b, j, :] = (b + 1) * 100 + (j + 1)

    out = _replace_state_tokens(
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        state_embeds=state_embeds,
        embodiment="franka_libero",
        tokenizer=tokenizer,
    )

    # 1. Output is a fresh tensor (clone), not the same object.
    assert out.data_ptr() != inputs_embeds.data_ptr()
    # 2. Non-state positions preserved exactly.
    for b in range(B):
        for s in range(S):
            if s in state_positions:
                continue
            assert torch.equal(out[b, s], inputs_embeds[b, s])
    # 3. State positions match the corresponding state_embeds slot.
    for b in range(B):
        for j, pos in enumerate(state_positions):
            assert torch.equal(out[b, pos], state_embeds[b, j])


def test__replace_state_tokens_rejects_missing_token():
    """If a special token is absent from input_ids the helper must raise."""
    torch = pytest.importorskip("torch")

    from starVLA.model.modules.uamvla.backbone_wrapper import _replace_state_tokens

    special_tokens = [
        "<|state_ee_main|>",
        "<|state_joint_main|>",
        "<|state_gripper_main|>",
    ]
    token_ids = {tok: 1000 + i for i, tok in enumerate(special_tokens)}

    class _FakeTokenizer:
        def convert_tokens_to_ids(self, tok):
            return token_ids[tok]

    B, S, H = 1, 4, 3
    # Drop the gripper token entirely — should fail the "exactly 1" check.
    input_ids = torch.tensor([[token_ids[special_tokens[0]],
                               token_ids[special_tokens[1]],
                               42, 42]], dtype=torch.long)
    inputs_embeds = torch.zeros(B, S, H)
    state_embeds = torch.zeros(B, 3, H)

    with pytest.raises(RuntimeError, match="appears"):
        _replace_state_tokens(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            state_embeds=state_embeds,
            embodiment="franka_libero",
            tokenizer=_FakeTokenizer(),
        )


def test_build_uamvla_backbone_requires_cfg_keys():
    """Factory must raise KeyError when cfg lacks base_vlm or embodiment."""
    pytest.importorskip("transformers", minversion="4.57.0")
    from starVLA.model.modules.uamvla.backbone_wrapper import build_uamvla_backbone

    # Empty cfg → missing base_vlm.
    with pytest.raises(KeyError, match="base_vlm|model_path"):
        build_uamvla_backbone({})

    # base_vlm present, embodiment missing.
    with pytest.raises(KeyError, match="embodiment"):
        build_uamvla_backbone(
            {"framework": {"qwenvl": {"base_vlm": "fake/path"}}}
        )


def test_build_inputs_returns_qwen_kwargs():
    """End-to-end smoke gated on the actual Qwen3-VL model (skipped on Mac)."""
    pytest.importorskip("transformers", minversion="4.57.0")
    pytest.importorskip("PIL")
    import os

    base_vlm = os.environ.get("UAMVLA_QWEN3VL_PATH")
    if not base_vlm:
        pytest.skip(
            "Set UAMVLA_QWEN3VL_PATH to a local Qwen3-VL checkpoint to run "
            "the full backbone build smoke test."
        )

    from starVLA.model.modules.uamvla.backbone_wrapper import build_uamvla_backbone

    cfg = {
        "framework": {
            "qwenvl": {"base_vlm": base_vlm, "low_cpu_mem_usage": True},
            "embodiment": {"name": "franka_libero"},
        },
    }
    backbone = build_uamvla_backbone(cfg)
    assert backbone.config is not None
    assert backbone.tokenizer is not None
    assert backbone.image_token_id >= 0
