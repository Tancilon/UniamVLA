"""L0 unit tests for predict_action decode helpers (_extract_generated_tail
and _decode_generated_actions). Uses a stub object for `self` so we don't
have to spin up a full UamVLA instance for these pure-tensor routines."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from unittest.mock import MagicMock

from starVLA.model.framework.VLM4A.UamVLA import UamVLA


def _make_stub(act0_id=100, n_bins=4):
    """Minimal stub with the attributes _decode_generated_actions and
    _extract_generated_tail need. Other UamVLA features are absent."""
    stub = MagicMock(spec=UamVLA)
    stub._act0_id = act0_id
    stub.config = OmegaConf.create(
        {"framework": {"action_model": {"num_bins": n_bins}}}
    )
    # Real ActionTokenizer.decode requires a tokenizer; for these tests we
    # synthesise a deterministic decode that maps id → (id - act0_id) / n_bins
    # so we can verify slicing/order independently of bin centers.
    def fake_decode(token_ids):
        return np.array(
            [(t - act0_id) / n_bins for t in token_ids], dtype=np.float32
        )
    stub.action_tokenizer = MagicMock()
    stub.action_tokenizer.decode = fake_decode
    # Bind the real method so test exercises real logic, not a mock.
    stub._extract_generated_tail = (
        lambda generated_ids, prompt_len, chunk_len:
        UamVLA._extract_generated_tail(stub, generated_ids, prompt_len, chunk_len)
    )
    return stub


def test_extract_generated_tail_accepts_prompt_plus_new_shape():
    """When generate returns (B, S_prompt + N_new), helper slices off the prompt."""
    stub = _make_stub()
    generated_ids = torch.arange(20).unsqueeze(0)  # (1, 20)
    out = UamVLA._extract_generated_tail(stub, generated_ids, prompt_len=15, chunk_len=5)
    assert out.shape == (1, 5)
    assert torch.equal(out, torch.tensor([[15, 16, 17, 18, 19]]))


def test_extract_generated_tail_accepts_new_only_shape():
    """When generate returns (B, N_new) only, helper passes the whole sequence through."""
    stub = _make_stub()
    generated_ids = torch.arange(5).unsqueeze(0)  # (1, 5)
    out = UamVLA._extract_generated_tail(stub, generated_ids, prompt_len=15, chunk_len=5)
    assert out.shape == (1, 5)
    assert torch.equal(out, generated_ids)


def test_decode_extracts_action_tokens_in_range():
    """Tokens in [_act0_id, _act0_id + n_bins) are extracted in order; chunk shape (B, H, action_dim)."""
    stub = _make_stub(act0_id=100, n_bins=4)
    H, action_dim = 1, 4
    # 4 ACT tokens (ids 100, 101, 102, 103) in the new portion
    generated_ids = torch.tensor([[999, 998, 100, 101, 102, 103]])  # prompt_len=2
    out = UamVLA._decode_generated_actions(
        stub, generated_ids, prompt_len=2, H=H, action_dim=action_dim,
    )
    assert out.shape == (1, H, action_dim)
    # fake_decode maps id → (id - 100) / 4 → [0.0, 0.25, 0.5, 0.75]
    expected = np.array([[[0.0, 0.25, 0.5, 0.75]]], dtype=np.float32)
    assert np.allclose(out, expected)


def test_decode_pads_with_mid_bin_when_short():
    """Fewer ACT tokens than chunk_len → pad with mid_bin id (= _act0_id + n_bins // 2)."""
    stub = _make_stub(act0_id=100, n_bins=4)
    H, action_dim = 1, 4
    # Only 2 ACT tokens; chunk_len=4 → pad 2 mid_bin (id=102, decoded to 0.5)
    generated_ids = torch.tensor([[100, 101]])  # new-only shape
    out = UamVLA._decode_generated_actions(
        stub, generated_ids, prompt_len=0, H=H, action_dim=action_dim,
    )
    expected = np.array([[[0.0, 0.25, 0.5, 0.5]]], dtype=np.float32)
    assert np.allclose(out, expected)


def test_decode_truncates_when_too_long():
    """More ACT tokens than chunk_len → truncate to chunk_len."""
    stub = _make_stub(act0_id=100, n_bins=4)
    H, action_dim = 1, 2
    # 4 ACT tokens; chunk_len=2 → keep first 2
    generated_ids = torch.tensor([[100, 101, 102, 103]])
    out = UamVLA._decode_generated_actions(
        stub, generated_ids, prompt_len=0, H=H, action_dim=action_dim,
    )
    expected = np.array([[[0.0, 0.25]]], dtype=np.float32)
    assert np.allclose(out, expected)


def test_decode_filters_non_action_tokens():
    """Non-ACT ids (outside [_act0_id, _act0_id + n_bins)) are skipped."""
    stub = _make_stub(act0_id=100, n_bins=4)
    H, action_dim = 1, 2
    # 999 and 50 are non-ACT; should be skipped
    generated_ids = torch.tensor([[999, 100, 50, 101]])
    out = UamVLA._decode_generated_actions(
        stub, generated_ids, prompt_len=0, H=H, action_dim=action_dim,
    )
    expected = np.array([[[0.0, 0.25]]], dtype=np.float32)
    assert np.allclose(out, expected)


def test_decode_per_row_independent_for_batched_input():
    """B=2 with different ACT token counts → each row decoded independently."""
    stub = _make_stub(act0_id=100, n_bins=4)
    H, action_dim = 1, 2
    # row 0: 2 ACT tokens (full); row 1: 1 ACT + 1 noise → pad mid_bin
    generated_ids = torch.tensor(
        [[100, 101],
         [102, 999]]
    )
    out = UamVLA._decode_generated_actions(
        stub, generated_ids, prompt_len=0, H=H, action_dim=action_dim,
    )
    expected = np.array(
        [[[0.0, 0.25]],   # row 0
         [[0.5, 0.5]]],   # row 1: id=102 → 0.5, then mid_bin id=102 → 0.5
        dtype=np.float32,
    )
    assert np.allclose(out, expected)
