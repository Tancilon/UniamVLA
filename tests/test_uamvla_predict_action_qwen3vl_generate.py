"""L1 smoke tests against a real Qwen3-VL model. Skipped unless a local
fixture is available (typically a small offline checkpoint configured via env var).

These tests verify behavior that text-only mocks cannot:
1. Real Qwen3-VL accepts model.generate(input_ids=..., inputs_embeds=...) together.
2. We can record whether the return shape is (B, S_prompt + N_new) or (B, N_new),
   which feeds back into _extract_generated_tail's design.
"""
import os
import pytest
import torch


_QWEN3_VL_PATH = os.environ.get("QWEN3VL_TEST_CKPT")


@pytest.fixture(scope="module")
def qwen3vl_model():
    if not _QWEN3_VL_PATH or not os.path.isdir(_QWEN3_VL_PATH):
        pytest.skip(
            "QWEN3VL_TEST_CKPT env var not set or path missing; "
            "skipping real-model smoke (track as L5 follow-up if needed)."
        )
    pytest.importorskip("transformers")
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        _QWEN3_VL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    return model


def test_qwen3vl_generate_accepts_input_ids_plus_inputs_embeds(qwen3vl_model):
    """Real Qwen3-VL must accept both input_ids and inputs_embeds in generate()."""
    B, S, H = 1, 5, qwen3vl_model.config.text_config.hidden_size
    input_ids = torch.zeros(B, S, dtype=torch.long)
    inputs_embeds = torch.zeros(B, S, H, dtype=torch.bfloat16)
    attention_mask = torch.ones(B, S, dtype=torch.long)
    out = qwen3vl_model.generate(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=2,
        min_new_tokens=2,
        do_sample=False,
    )
    assert isinstance(out, torch.Tensor)
    assert out.dim() == 2
    assert out.shape[0] == B


def test_qwen3vl_generate_return_shape_is_decodable(qwen3vl_model):
    """Record the return shape and verify _extract_generated_tail handles it.

    With input_ids supplied, current HF behavior is to return prompt+new IDs
    (B, S_prompt + N_new). New-only return would still be acceptable downstream
    via _extract_generated_tail's normalization."""
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA

    B, S = 1, 5
    Hdim = qwen3vl_model.config.text_config.hidden_size
    input_ids = torch.zeros(B, S, dtype=torch.long)
    inputs_embeds = torch.zeros(B, S, Hdim, dtype=torch.bfloat16)
    chunk_len = 3
    out = qwen3vl_model.generate(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones(B, S, dtype=torch.long),
        max_new_tokens=chunk_len,
        min_new_tokens=chunk_len,
        do_sample=False,
    )
    # Shape must be either (B, S + chunk_len) or (B, chunk_len)
    assert out.shape[1] in (S + chunk_len, chunk_len), (
        f"Unexpected generate output shape {tuple(out.shape)}; "
        "_extract_generated_tail handles both prompt+new and new-only forms"
    )

    # Verify _extract_generated_tail returns a (B, chunk_len) tensor either way.
    class _Stub:
        pass
    stub = _Stub()
    tail = UamVLA._extract_generated_tail(stub, out, prompt_len=S, chunk_len=chunk_len)
    assert tail.shape == (B, chunk_len)
