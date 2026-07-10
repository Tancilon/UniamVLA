"""Pin the mask format the GR00T DiT's cross-attention actually honours.

``FlowmatchingActionHead`` cross-attends over the whole Qwen sequence, and the
Qwen tokenizer right-pads.  Passing no ``encoder_attention_mask`` lets the DiT
attend to padding hidden states; passing the raw int64 0/1 mask raises.  Only a
bool (or additive -inf) mask does the right thing.
"""
from __future__ import annotations

import pytest
import torch
from diffusers.models.attention import Attention

B, Q, KV, D = 2, 4, 6, 32
N_VALID = 4


def _attn_outputs(mask):
    torch.manual_seed(0)
    attn = Attention(query_dim=D, heads=4, dim_head=8, cross_attention_dim=D).eval()
    q = torch.randn(B, Q, D)
    kv = torch.randn(B, KV, D)
    kv_polluted = kv.clone()
    kv_polluted[:, N_VALID:] = 999.0  # only the padded key positions
    with torch.no_grad():
        return (
            attn(q, encoder_hidden_states=kv, attention_mask=mask),
            attn(q, encoder_hidden_states=kv_polluted, attention_mask=mask),
        )


def _valid_mask():
    m = torch.zeros(B, KV, dtype=torch.long)
    m[:, :N_VALID] = 1
    return m


def test_without_mask_the_dit_attends_to_padding():
    """Documents the pre-existing bug the use_encoder_mask flag fixes."""
    clean, polluted = _attn_outputs(None)
    assert (clean - polluted).abs().max() > 1.0


def test_raw_int_mask_is_rejected():
    """qwen_inputs['attention_mask'] is int64 -- passing it straight through raises."""
    with pytest.raises(RuntimeError, match="attn_mask dtype"):
        _attn_outputs(_valid_mask())


def test_bool_mask_ignores_padding():
    clean, polluted = _attn_outputs(_valid_mask().bool())
    assert (clean - polluted).abs().max() < 1e-5


def test_additive_mask_ignores_padding():
    additive = torch.where(_valid_mask().bool(), 0.0, float("-inf"))
    clean, polluted = _attn_outputs(additive)
    assert (clean - polluted).abs().max() < 1e-5
