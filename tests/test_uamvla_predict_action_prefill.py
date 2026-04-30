"""L0 unit tests for _build_prefill_generate_kwargs.

Uses a hand-built stub for `self` because the real UamVLA constructor pulls in
the full backbone + action head graph; we want to exercise the prefill helper
in isolation."""
import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch

from starVLA.model.framework.VLM4A.UamVLA import UamVLA


def _make_stub(B=2, S=8, hidden_size=4, vocab_size=200):
    """Stub with the qwen_vl_interface attributes _build_prefill_generate_kwargs reads."""
    stub = MagicMock(spec=UamVLA)
    iface = MagicMock()
    iface.embodiment = "franka_libero"
    iface.tokenizer = MagicMock()
    # Real-tensor embedding so torch.cat / clone in _replace_state_tokens works
    iface.get_embed_tokens.return_value = nn.Embedding(vocab_size, hidden_size)
    # state_encoder returns a (B, n_state_tokens, hidden) tensor
    iface.state_encoder = MagicMock(
        return_value=torch.zeros(B, 3, hidden_size)
    )
    # _derive_mm_token_type_ids passthrough that returns a fixed tensor
    iface._derive_mm_token_type_ids = MagicMock(
        return_value=torch.ones(B, S, dtype=torch.int32)
    )
    stub.qwen_vl_interface = iface
    stub._build_prefill_generate_kwargs = (
        lambda qwen_inputs:
        UamVLA._build_prefill_generate_kwargs(stub, qwen_inputs)
    )
    return stub, B, S, hidden_size


def _make_qwen_inputs(B, S, with_canonical_state=True, with_mm_token_type_ids=False):
    qi = {
        "input_ids": torch.zeros(B, S, dtype=torch.long),
        "attention_mask": torch.ones(B, S, dtype=torch.long),
        "pixel_values": torch.zeros(B, 3, 32, 32),
        "image_grid_thw": torch.tensor([[1, 4, 4]] * B),
    }
    if with_canonical_state:
        qi["canonical_state"] = {
            "arm_0": {"ee_pose": torch.zeros(B, 9), "joint_pos": torch.zeros(B, 7)},
            "gripper_0": torch.zeros(B, 1),
        }
    if with_mm_token_type_ids:
        qi["mm_token_type_ids"] = torch.zeros(B, S, dtype=torch.int32)
    return qi


def test_prefill_calls_state_encoder_when_canonical_state_given():
    """When canonical_state present, state_encoder is invoked and _replace_state_tokens called."""
    stub, B, S, hidden_size = _make_stub()
    qi = _make_qwen_inputs(B, S, with_canonical_state=True)
    with patch(
        "starVLA.model.modules.uamvla.backbone_wrapper._replace_state_tokens"
    ) as mock_replace:
        mock_replace.return_value = torch.zeros(B, S, hidden_size)
        gen_kwargs = stub._build_prefill_generate_kwargs(qi)
    assert stub.qwen_vl_interface.state_encoder.called
    assert mock_replace.called
    # _replace_state_tokens called with the right embodiment + tokenizer
    _, kwargs = mock_replace.call_args
    assert kwargs["embodiment"] == "franka_libero"
    assert kwargs["tokenizer"] is stub.qwen_vl_interface.tokenizer


def test_prefill_skips_state_encoder_when_canonical_state_absent():
    """No canonical_state → returns plain embed_tokens(input_ids)."""
    stub, B, S, hidden_size = _make_stub()
    qi = _make_qwen_inputs(B, S, with_canonical_state=False)
    gen_kwargs = stub._build_prefill_generate_kwargs(qi)
    assert not stub.qwen_vl_interface.state_encoder.called


def test_prefill_output_kwargs_shape_and_keys():
    """Returned dict has input_ids, inputs_embeds, attention_mask, pixel_values, image_grid_thw."""
    stub, B, S, hidden_size = _make_stub()
    qi = _make_qwen_inputs(B, S, with_canonical_state=False)
    gen_kwargs = stub._build_prefill_generate_kwargs(qi)
    assert "input_ids" in gen_kwargs
    assert "inputs_embeds" in gen_kwargs
    assert "attention_mask" in gen_kwargs
    assert "pixel_values" in gen_kwargs
    assert "image_grid_thw" in gen_kwargs
    assert gen_kwargs["inputs_embeds"].shape == (B, S, hidden_size)
    # canonical_state itself should NOT be in gen_kwargs (it's not a generate() kwarg)
    assert "canonical_state" not in gen_kwargs


def test_prefill_preserves_mm_token_type_ids_when_processor_returns_them():
    """If qwen_inputs has mm_token_type_ids, prefill passes it through (no derive call)."""
    stub, B, S, hidden_size = _make_stub()
    qi = _make_qwen_inputs(B, S, with_canonical_state=False, with_mm_token_type_ids=True)
    expected_mm = qi["mm_token_type_ids"]
    gen_kwargs = stub._build_prefill_generate_kwargs(qi)
    # Derive helper must NOT have been called since processor already provided it
    assert not stub.qwen_vl_interface._derive_mm_token_type_ids.called
    assert torch.equal(gen_kwargs["mm_token_type_ids"], expected_mm)


def test_prefill_derives_mm_token_type_ids_when_absent():
    """If qwen_inputs lacks mm_token_type_ids, prefill derives via wrapper helper."""
    stub, B, S, hidden_size = _make_stub()
    qi = _make_qwen_inputs(B, S, with_canonical_state=False, with_mm_token_type_ids=False)
    gen_kwargs = stub._build_prefill_generate_kwargs(qi)
    assert stub.qwen_vl_interface._derive_mm_token_type_ids.called
    # The derived tensor (ones, per stub) should be in gen_kwargs
    assert "mm_token_type_ids" in gen_kwargs
    assert torch.equal(
        gen_kwargs["mm_token_type_ids"], torch.ones(B, S, dtype=torch.int32)
    )
