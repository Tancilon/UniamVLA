"""Tests for labels wiring (deferred sub-task before Phase 1F).

Validates that _build_labels_and_extend appends action tokens to prompt
input_ids and produces correctly masked labels — no HF checkpoint needed.
"""
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock


@pytest.fixture
def mock_tokenizer_and_action_tokenizer():
    """Build a minimal real ActionTokenizer backed by a mock HF tokenizer."""
    from starVLA.model.modules.uamvla.data.action_tokenizer import ActionTokenizer
    # Real tokenizer mock: add_tokens is a no-op; convert_tokens_to_ids returns sequential ints
    mock_tok = MagicMock()
    mock_tok.pad_token_id = 0
    call_count = [0]
    def _cvt(tok_str):
        # Return a unique int per unique string, starting from 100
        if not hasattr(_cvt, '_cache'):
            _cvt._cache = {}
        if tok_str not in _cvt._cache:
            _cvt._cache[tok_str] = 100 + len(_cvt._cache)
        return _cvt._cache[tok_str]
    mock_tok.convert_tokens_to_ids.side_effect = _cvt
    mock_tok.__len__ = lambda self: 356  # 100 base + 256 act tokens
    at = ActionTokenizer(mock_tok, n_bins=256)
    return mock_tok, at


def _make_uamvla_with_mock(monkeypatch, action_tokenizer, mock_tokenizer):
    """Build a UamVLA instance with mocked backbone and real ActionTokenizer."""
    import starVLA.model.framework.VLM4A.UamVLA as uamvla_mod
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA
    from omegaconf import OmegaConf

    fake_backbone = MagicMock()
    fake_backbone.config.hidden_size = 64
    fake_backbone.image_token_id = 151655
    fake_backbone.tokenizer = mock_tokenizer
    fake_backbone.get_tokenizer.return_value = mock_tokenizer

    monkeypatch.setattr(uamvla_mod, "build_uamvla_backbone", lambda cfg: fake_backbone)

    cfg = OmegaConf.create({"framework": {"aux_heads": {
        "action": {"enabled": True, "lr": 1e-4},
        "pose":   {"enabled": False},
        "future": {"enabled": False},
        "recon":  {"enabled": False},
    }}})
    model = UamVLA(config=cfg)
    # Replace action_tokenizer with the controlled fixture one
    model.action_tokenizer = action_tokenizer
    return model


def test_labels_shape(monkeypatch, mock_tokenizer_and_action_tokenizer):
    mock_tok, at = mock_tokenizer_and_action_tokenizer
    model = _make_uamvla_with_mock(monkeypatch, at, mock_tok)

    B, L_p, H = 2, 10, 2  # batch=2, prompt_len=10, action_horizon=2
    model.action_horizon = H

    qwen_inputs = {
        "input_ids":      torch.randint(1, 50, (B, L_p)),
        "attention_mask": torch.ones(B, L_p, dtype=torch.long),
    }
    examples = [{
        "action":      torch.zeros(H, 7),
        "action_mask": torch.ones(H, 7, dtype=torch.long),
    } for _ in range(B)]

    ext, labels = model._build_labels_and_extend(examples, qwen_inputs)

    L_total = L_p + H * 7
    assert ext["input_ids"].shape == (B, L_total)
    assert ext["attention_mask"].shape == (B, L_total)
    assert labels.shape == (B, L_total)


def test_labels_prompt_positions_are_minus100(monkeypatch, mock_tokenizer_and_action_tokenizer):
    mock_tok, at = mock_tokenizer_and_action_tokenizer
    model = _make_uamvla_with_mock(monkeypatch, at, mock_tok)

    B, L_p, H = 1, 8, 1
    model.action_horizon = H
    qwen_inputs = {
        "input_ids":      torch.ones(B, L_p, dtype=torch.long),
        "attention_mask": torch.ones(B, L_p, dtype=torch.long),
    }
    examples = [{
        "action":      torch.zeros(H, 7),
        "action_mask": torch.ones(H, 7, dtype=torch.long),
    }]

    _, labels = model._build_labels_and_extend(examples, qwen_inputs)
    # First L_p positions must be -100 (prompt tokens are not supervision targets)
    assert (labels[0, :L_p] == -100).all()


def test_labels_action_positions_nonzero_for_valid_mask(monkeypatch, mock_tokenizer_and_action_tokenizer):
    mock_tok, at = mock_tokenizer_and_action_tokenizer
    model = _make_uamvla_with_mock(monkeypatch, at, mock_tok)

    B, L_p, H = 1, 6, 1
    model.action_horizon = H
    qwen_inputs = {
        "input_ids":      torch.ones(B, L_p, dtype=torch.long),
        "attention_mask": torch.ones(B, L_p, dtype=torch.long),
    }
    examples = [{
        "action":      torch.zeros(H, 7),  # action values (will be encoded)
        "action_mask": torch.ones(H, 7, dtype=torch.long),  # all valid
    }]

    _, labels = model._build_labels_and_extend(examples, qwen_inputs)
    # Action positions (L_p .. L_p+7) should NOT be -100
    action_labels = labels[0, L_p:L_p + 7]
    assert (action_labels != -100).all(), f"Expected action labels != -100, got {action_labels}"


def test_labels_padded_timestep_is_minus100(monkeypatch, mock_tokenizer_and_action_tokenizer):
    mock_tok, at = mock_tokenizer_and_action_tokenizer
    model = _make_uamvla_with_mock(monkeypatch, at, mock_tok)

    B, L_p, H = 1, 6, 2  # H=2 steps; second step is padded
    model.action_horizon = H
    qwen_inputs = {
        "input_ids":      torch.ones(B, L_p, dtype=torch.long),
        "attention_mask": torch.ones(B, L_p, dtype=torch.long),
    }
    # First step valid, second step all-zero mask (padded)
    action_mask = torch.zeros(H, 7, dtype=torch.long)
    action_mask[0] = 1  # only first step is real
    examples = [{"action": torch.zeros(H, 7), "action_mask": action_mask}]

    _, labels = model._build_labels_and_extend(examples, qwen_inputs)
    # Second step (positions L_p+7 .. L_p+14) should be -100
    step2_labels = labels[0, L_p + 7: L_p + 14]
    assert (step2_labels == -100).all(), f"Expected -100 for padded step, got {step2_labels}"
