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
