"""L1 smoke tests for UamVLA.predict_action.

These tests build a minimal UamVLA via the same monkeypatch pattern as
test_uamvla_l1_init.py, mock model.generate to return predetermined ACT token
sequences, and verify the full inference path: input building → prefill →
generate → decode → normalized actions."""
import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from omegaconf import OmegaConf


def _build_minimal_uamvla(monkeypatch, n_bins=256, hidden_size=4, vocab_size=400):
    """Build a UamVLA with an action-only aux head and a hand-rolled fake backbone.

    Returns (model, fake_backbone, n_bins, hidden_size).
    """
    pytest.importorskip("torch")
    import starVLA.model.framework.VLM4A.UamVLA as uamvla_mod
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA

    # Tokenizer that registers <ACT_i> tokens at sequential ids and supports add_tokens.
    class _FakeTokenizer:
        def __init__(self):
            self.unk_token_id = 1
            self.padding_side = "right"
            self._vocab = {}
            self._next = 100
        def add_tokens(self, tokens):
            added = 0
            for t in tokens:
                if t not in self._vocab:
                    self._vocab[t] = self._next
                    self._next += 1
                    added += 1
            return added
        def convert_tokens_to_ids(self, tok):
            return self._vocab.get(tok, self.unk_token_id)
        def __len__(self):
            return self._next

    tok = _FakeTokenizer()
    # Pre-register <|action_start|> at id 99 (must differ from unk_token_id=1)
    tok._vocab["<|action_start|>"] = 99

    fake_backbone = MagicMock()
    fake_backbone.config.hidden_size = hidden_size
    fake_backbone.image_token_id = 0
    fake_backbone.tokenizer = tok
    fake_backbone.embodiment = "franka_libero"

    backbone_params = [nn.Parameter(torch.randn(2))]
    state_encoder_params = [nn.Parameter(torch.randn(2))]
    fake_backbone.parameters.side_effect = lambda: iter(
        backbone_params + state_encoder_params
    )
    fake_backbone.state_encoder = MagicMock()
    fake_backbone.state_encoder.parameters.side_effect = lambda: iter(state_encoder_params)
    fake_backbone.get_tokenizer = lambda: tok
    fake_backbone.resize_token_embeddings = lambda n: None

    monkeypatch.setattr(uamvla_mod, "build_uamvla_backbone", lambda cfg: fake_backbone)

    cfg = OmegaConf.create({
        "framework": {
            "action_model": {"num_bins": n_bins, "action_dim": 7,
                              "future_action_window_size": 7},
            "embodiment": {"name": "franka_libero", "action_dim": 7},
            "aux_heads": {
                "action": {"enabled": True, "lr": 1e-4, "loss_weight": 1.0},
                "pose":   {"enabled": False},
                "future": {"enabled": False},
                "recon":  {"enabled": False},
            },
        },
    })
    model = UamVLA(config=cfg)
    return model, fake_backbone, n_bins, hidden_size


def _patch_predict_dependencies(model, fake_backbone, generated_ids,
                                 prompt_len=10, hidden_size=4):
    """Wire up the minimal mocks for predict_action: build_inputs returns synthetic
    qwen_inputs, model.generate returns the test's fixed generated_ids."""
    # Skip the real build_inputs (would require a true HF processor); inject synthetic.
    B = generated_ids.shape[0]
    qi = {
        "input_ids": torch.zeros(B, prompt_len, dtype=torch.long),
        "attention_mask": torch.ones(B, prompt_len, dtype=torch.long),
        "pixel_values": torch.zeros(B, 3, 32, 32),
        "image_grid_thw": torch.tensor([[1, 4, 4]] * B),
    }
    model.qwen_vl_interface.build_inputs = MagicMock(return_value=qi)

    # Embedding lookup needs to be a real callable to keep state-splice path simple
    model.qwen_vl_interface.get_embed_tokens = lambda: nn.Embedding(400, hidden_size)
    # Skip state splice (no canonical_state) — production tests cover state branch
    # via the prefill L0 tests.
    model.qwen_vl_interface.state_encoder = MagicMock(
        return_value=torch.zeros(B, 3, hidden_size)
    )
    model.qwen_vl_interface._derive_mm_token_type_ids = MagicMock(return_value=None)

    fake_backbone.model = MagicMock()
    fake_backbone.model.generate = MagicMock(return_value=generated_ids)


def _make_examples(B=1):
    from PIL import Image
    img = Image.new("RGB", (32, 32))
    return [
        {"image": [img], "lang": f"task {i}"}
        for i in range(B)
    ]


# ============================================================================
# Tests
# ============================================================================

def test_predict_action_returns_correct_shape_b1(monkeypatch):
    """B=1 produces (1, H, action_dim) normalized_actions."""
    model, fake_backbone, n_bins, hidden_size = _build_minimal_uamvla(monkeypatch)
    H = model.action_horizon
    action_dim = 7
    chunk_len = H * action_dim
    # Fixed ACT sequence: prompt (10) + chunk_len ACT tokens
    prompt_len = 10
    act_ids = torch.full((1, chunk_len), model._act0_id, dtype=torch.long)
    generated_ids = torch.cat(
        [torch.zeros(1, prompt_len, dtype=torch.long), act_ids], dim=1
    )
    _patch_predict_dependencies(
        model, fake_backbone, generated_ids,
        prompt_len=prompt_len, hidden_size=hidden_size,
    )
    out = model.predict_action(_make_examples(B=1))
    assert "normalized_actions" in out
    assert out["normalized_actions"].shape == (1, H, action_dim)
    assert out["normalized_actions"].dtype == np.float32


def test_predict_action_returns_correct_shape_b_geq_1(monkeypatch):
    """B=2 produces (2, H, action_dim); rows independent."""
    model, fake_backbone, n_bins, hidden_size = _build_minimal_uamvla(monkeypatch)
    H = model.action_horizon
    action_dim = 7
    chunk_len = H * action_dim
    prompt_len = 10
    # Row 0: all ACT_0; Row 1: all ACT_5 (different bin → different decoded value)
    row0 = torch.full((1, chunk_len), model._act0_id, dtype=torch.long)
    row1 = torch.full((1, chunk_len), model._act0_id + 5, dtype=torch.long)
    act_ids = torch.cat([row0, row1], dim=0)
    generated_ids = torch.cat(
        [torch.zeros(2, prompt_len, dtype=torch.long), act_ids], dim=1
    )
    _patch_predict_dependencies(
        model, fake_backbone, generated_ids,
        prompt_len=prompt_len, hidden_size=hidden_size,
    )
    out = model.predict_action(_make_examples(B=2))
    assert out["normalized_actions"].shape == (2, H, action_dim)
    # Verify rows differ (different bins decode to different values)
    assert not np.allclose(out["normalized_actions"][0], out["normalized_actions"][1])


def test_predict_action_handles_missing_canonical_state(monkeypatch):
    """Examples without canonical_state still complete (no state splice)."""
    model, fake_backbone, n_bins, hidden_size = _build_minimal_uamvla(monkeypatch)
    H = model.action_horizon
    chunk_len = H * 7
    prompt_len = 10
    act_ids = torch.full((1, chunk_len), model._act0_id, dtype=torch.long)
    generated_ids = torch.cat(
        [torch.zeros(1, prompt_len, dtype=torch.long), act_ids], dim=1
    )
    _patch_predict_dependencies(
        model, fake_backbone, generated_ids,
        prompt_len=prompt_len, hidden_size=hidden_size,
    )
    examples = _make_examples(B=1)
    # Explicitly no canonical_state in examples
    assert "canonical_state" not in examples[0]
    out = model.predict_action(examples)
    assert out["normalized_actions"].shape == (1, H, 7)
    # state_encoder was not invoked
    assert not model.qwen_vl_interface.state_encoder.called


def test_predict_action_rejects_mixed_canonical_state_presence(monkeypatch):
    """Some examples with canonical_state and some without raises ValueError."""
    model, fake_backbone, _, hidden_size = _build_minimal_uamvla(monkeypatch)
    examples = _make_examples(B=2)
    examples[0]["canonical_state"] = {"arm_0": {"ee_pose": torch.zeros(9),
                                                 "joint_pos": torch.zeros(7)},
                                       "gripper_0": torch.zeros(1)}
    # examples[1] has no canonical_state
    with pytest.raises(ValueError, match="all examples or none"):
        model.predict_action(examples)


def test_predict_action_uses_left_padding_temporarily_and_restores_tokenizer(monkeypatch):
    """Variable-length prompts use left padding inside predict_action; tokenizer
    padding_side is restored on exit."""
    model, fake_backbone, _, hidden_size = _build_minimal_uamvla(monkeypatch)
    H = model.action_horizon
    chunk_len = H * 7
    prompt_len = 10

    tokenizer = model.qwen_vl_interface.tokenizer
    tokenizer.padding_side = "right"  # baseline (training default)
    observed_padding_side = []

    def fake_build_inputs(**kwargs):
        observed_padding_side.append(tokenizer.padding_side)
        B = len(kwargs["images"])
        return {
            "input_ids": torch.zeros(B, prompt_len, dtype=torch.long),
            "attention_mask": torch.ones(B, prompt_len, dtype=torch.long),
            "pixel_values": torch.zeros(B, 3, 32, 32),
            "image_grid_thw": torch.tensor([[1, 4, 4]] * B),
        }
    act_ids = torch.full((1, chunk_len), model._act0_id, dtype=torch.long)
    generated_ids = torch.cat(
        [torch.zeros(1, prompt_len, dtype=torch.long), act_ids], dim=1
    )
    _patch_predict_dependencies(
        model, fake_backbone, generated_ids,
        prompt_len=prompt_len, hidden_size=hidden_size,
    )
    # Override the MagicMock build_inputs installed by _patch_predict_dependencies
    # with our observer that records tokenizer.padding_side at call time.
    model.qwen_vl_interface.build_inputs = fake_build_inputs

    model.predict_action(_make_examples(B=1))
    # Inside predict_action, padding_side was "left"
    assert observed_padding_side == ["left"]
    # After predict_action returns, padding_side restored to "right"
    assert tokenizer.padding_side == "right"


def test_predict_action_enables_action_logits_processor_by_default(monkeypatch):
    """Default kwargs → model.generate receives a LogitsProcessorList containing ActionLogitsProcessor."""
    from transformers import LogitsProcessorList
    from starVLA.model.modules.uamvla.inference import ActionLogitsProcessor

    model, fake_backbone, _, hidden_size = _build_minimal_uamvla(monkeypatch)
    H = model.action_horizon
    chunk_len = H * 7
    prompt_len = 10
    act_ids = torch.full((1, chunk_len), model._act0_id, dtype=torch.long)
    generated_ids = torch.cat(
        [torch.zeros(1, prompt_len, dtype=torch.long), act_ids], dim=1
    )
    _patch_predict_dependencies(
        model, fake_backbone, generated_ids,
        prompt_len=prompt_len, hidden_size=hidden_size,
    )
    model.predict_action(_make_examples(B=1))

    _, kwargs = fake_backbone.model.generate.call_args
    lp = kwargs.get("logits_processor")
    assert isinstance(lp, LogitsProcessorList)
    assert any(isinstance(p, ActionLogitsProcessor) for p in lp)


def test_predict_action_can_disable_action_logits_processor(monkeypatch):
    """constrain_action_logits=False → no ActionLogitsProcessor passed."""
    model, fake_backbone, _, hidden_size = _build_minimal_uamvla(monkeypatch)
    H = model.action_horizon
    chunk_len = H * 7
    prompt_len = 10
    act_ids = torch.full((1, chunk_len), model._act0_id, dtype=torch.long)
    generated_ids = torch.cat(
        [torch.zeros(1, prompt_len, dtype=torch.long), act_ids], dim=1
    )
    _patch_predict_dependencies(
        model, fake_backbone, generated_ids,
        prompt_len=prompt_len, hidden_size=hidden_size,
    )
    model.predict_action(_make_examples(B=1), constrain_action_logits=False)

    _, kwargs = fake_backbone.model.generate.call_args
    assert kwargs.get("logits_processor") is None


def test_predict_action_generate_kwargs_keep_input_ids_and_inputs_embeds(monkeypatch):
    """model.generate is called with BOTH input_ids and inputs_embeds (required for
    ActionLogitsProcessor activation)."""
    model, fake_backbone, _, hidden_size = _build_minimal_uamvla(monkeypatch)
    H = model.action_horizon
    chunk_len = H * 7
    prompt_len = 10
    act_ids = torch.full((1, chunk_len), model._act0_id, dtype=torch.long)
    generated_ids = torch.cat(
        [torch.zeros(1, prompt_len, dtype=torch.long), act_ids], dim=1
    )
    _patch_predict_dependencies(
        model, fake_backbone, generated_ids,
        prompt_len=prompt_len, hidden_size=hidden_size,
    )
    model.predict_action(_make_examples(B=1))

    _, kwargs = fake_backbone.model.generate.call_args
    assert "input_ids" in kwargs
    assert "inputs_embeds" in kwargs
    assert kwargs["input_ids"] is not None
    assert kwargs["inputs_embeds"] is not None
