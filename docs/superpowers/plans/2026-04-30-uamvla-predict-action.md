# UamVLA `predict_action` Autoregressive Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder `predict_action` with a real autoregressive inference path that prefill-injects `<|state_*|>` embeddings, calls HuggingFace `model.generate(input_ids=..., inputs_embeds=...)` constrained by `ActionLogitsProcessor`, and decodes the generated tail back to normalized actions.

**Architecture:** Prefill stage builds `inputs_embeds` via the wrapper's existing `_replace_state_tokens` to preserve training-equivalent state splice; both `input_ids` and `inputs_embeds` are passed to `model.generate` so `ActionLogitsProcessor` can detect `<|action_start|>`. A return-shape-aware tail extractor handles both `(B, S_prompt+N_new)` and `(B, N_new)` returns. Inference uses left padding temporarily (restored via try/finally). All callers see zero interface change.

**Tech Stack:** Python 3, PyTorch, HuggingFace `transformers` (`>=4.57.0`), pytest, OmegaConf.

**Reference spec:** [`docs/superpowers/specs/2026-04-30-uamvla-predict-action-design.md`](../specs/2026-04-30-uamvla-predict-action-design.md)

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `starVLA/model/framework/VLM4A/UamVLA.py` | **Modify** | Promote `_act0_id` to instance attr; add ID contiguity assertion; add `_build_prefill_generate_kwargs` / `_extract_generated_tail` / `_decode_generated_actions`; rewrite `predict_action` body; remove `_extract_action_tokens_for_sample` and old `_decode_action_tokens` |
| `tests/test_uamvla_predict_action_decode.py` | **Create** | 7 L0 unit tests covering `_extract_generated_tail` + `_decode_generated_actions` |
| `tests/test_uamvla_predict_action_prefill.py` | **Create** | 5 L0 unit tests covering `_build_prefill_generate_kwargs` |
| `tests/test_uamvla_predict_action.py` | **Modify** | Replace skip-placeholder with 8 L1 smoke tests covering full `predict_action` |
| `tests/test_uamvla_predict_action_qwen3vl_generate.py` | **Create** | 2 L1 smoke tests against real Qwen3-VL (skip if no local fixture) |
| `tests/test_uamvla_predict_action_alignment.py` | **Create** | 1 critical training-vs-inference numerical equality test |

Net total: ~180 lines main code + ~480 lines test code.

---

## Task 1: Init changes — `_act0_id` instance attribute + ID contiguity assertion

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLA.py:108-133` (the `_act0_id` resolution block)

- [ ] **Step 1: Read current init**

Run: `sed -n '108,135p' starVLA/model/framework/VLM4A/UamVLA.py`

Expected output: lines around the existing `_act0_id` try/except (resolves `<ACT_0>` token id) and the `action_start_id` block.

- [ ] **Step 2: Add instance attribute and contiguity assertion**

Edit `starVLA/model/framework/VLM4A/UamVLA.py`. Find this block (the `_act0_id` resolution, around line 108-112):

```python
        # Resolve action_token_begin_id from the newly added <ACT_0> token.
        try:
            _act0_id = int(_tokenizer.convert_tokens_to_ids("<ACT_0>"))
        except (TypeError, ValueError):
            _act0_id = 0
```

Replace with:

```python
        # Resolve action_token_begin_id from the newly added <ACT_0> token.
        try:
            _act0_id = int(_tokenizer.convert_tokens_to_ids("<ACT_0>"))
        except (TypeError, ValueError):
            _act0_id = 0
        self._act0_id = _act0_id

        # Verify <ACT_i> token IDs are contiguous (required for ActionLogitsProcessor
        # and _decode_generated_actions). Skip on mock tokenizers that return the
        # same id for every <ACT_*> query.
        try:
            _act_last_id = int(_tokenizer.convert_tokens_to_ids(f"<ACT_{_n_bins - 1}>"))
            if _act_last_id != _act0_id and _act_last_id != _act0_id + _n_bins - 1:
                raise RuntimeError(
                    f"Action token IDs are not contiguous: <ACT_0>={_act0_id}, "
                    f"<ACT_{_n_bins - 1}>={_act_last_id}, "
                    f"expected {_act0_id + _n_bins - 1}. "
                    "ActionLogitsProcessor and _decode_generated_actions require "
                    "contiguous IDs."
                )
        except (TypeError, ValueError):
            # Mock tokenizer in unit tests; cannot verify.
            pass
```

The condition `_act_last_id != _act0_id` skips the check when both queries return the same id (typical of `MagicMock`); `_act_last_id != _act0_id + _n_bins - 1` then enforces contiguity when the tokenizer actually distinguishes the two endpoints.

- [ ] **Step 3: Verify existing init tests still pass**

Run: `pytest tests/test_uamvla_l1_init.py tests/test_uamvla_framework_init.py -v`

Expected: all PASS, 1 SKIP (the existing intentional skip).

If any test fails because the contiguity check fired incorrectly under a mock fixture, narrow the condition rather than weaken the production guard — typically the fix is to make the test fixture's `convert_tokens_to_ids` return the same id for both endpoints.

- [ ] **Step 4: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py
git commit -m "[uamvla] Promote _act0_id to instance attr; assert ACT token IDs contiguous"
```

---

## Task 2: `_extract_generated_tail` — return-shape normalization helper

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLA.py` (add new method)
- Create: `tests/test_uamvla_predict_action_decode.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uamvla_predict_action_decode.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_uamvla_predict_action_decode.py::test_extract_generated_tail_accepts_prompt_plus_new_shape tests/test_uamvla_predict_action_decode.py::test_extract_generated_tail_accepts_new_only_shape -v`

Expected: both FAIL with `AttributeError: type object 'UamVLA' has no attribute '_extract_generated_tail'`.

- [ ] **Step 3: Implement `_extract_generated_tail`**

Edit `starVLA/model/framework/VLM4A/UamVLA.py`. Find the `predict_action` method (around line 340). Add the following method **immediately after** `predict_action` ends (and before any other method):

```python
    def _extract_generated_tail(
        self,
        generated_ids: torch.Tensor,
        prompt_len: int,
        chunk_len: int,
    ) -> torch.Tensor:
        """Return generated new-token IDs from either prompt+new or new-only sequences.

        HuggingFace `generate(inputs_embeds=...)` may return (B, S_prompt + N_new)
        or (B, N_new) depending on model and version. Normalize to "new tokens only"
        before downstream decoding.
        """
        seq_len = int(generated_ids.shape[1])

        # New-only return: usually exactly chunk_len, or shorter if generation stopped early.
        if seq_len <= chunk_len:
            return generated_ids

        # Prompt+new return: slice after the prompt. This is the expected path when
        # input_ids are supplied to generate().
        if seq_len >= prompt_len:
            return generated_ids[:, prompt_len:]

        # Defensive fallback for unusual wrappers: keep the final action window.
        return generated_ids[:, -chunk_len:]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_uamvla_predict_action_decode.py::test_extract_generated_tail_accepts_prompt_plus_new_shape tests/test_uamvla_predict_action_decode.py::test_extract_generated_tail_accepts_new_only_shape -v`

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py tests/test_uamvla_predict_action_decode.py
git commit -m "[uamvla] Add _extract_generated_tail for HF generate() shape normalization"
```

---

## Task 3: `_decode_generated_actions` — ID-range scan with mid_bin padding

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLA.py` (add method, replace old `_decode_action_tokens`)
- Modify: `tests/test_uamvla_predict_action_decode.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_uamvla_predict_action_decode.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_uamvla_predict_action_decode.py -v -k "test_decode"`

Expected: 5 tests FAIL with `AttributeError: ... has no attribute '_decode_generated_actions'`.

- [ ] **Step 3: Implement `_decode_generated_actions`**

Edit `starVLA/model/framework/VLM4A/UamVLA.py`. Add this method **immediately after** `_extract_generated_tail` (added in Task 2):

```python
    def _decode_generated_actions(
        self,
        generated_ids: torch.Tensor,
        prompt_len: int,
        H: int,
        action_dim: int,
    ) -> np.ndarray:
        """Extract action tokens from the generated tail and decode to normalized actions.

        Strategy: normalize generate() return shape via _extract_generated_tail, then
        scan for action token ids in [_act0_id, _act0_id + n_bins). Tokens outside
        the range are skipped defensively. Length is normalized to H * action_dim
        via mid_bin padding (action ≈ 0) on shortfall, or truncation on overflow.

        Args:
            generated_ids: (B, S_prompt + N_new) or (B, N_new) long tensor from model.generate
            prompt_len: number of prompt tokens (= inputs_embeds.shape[1])
            H: action horizon
            action_dim: action dimensionality (typically 7 for Franka)

        Returns:
            np.ndarray of shape (B, H, action_dim), float32, in normalized action space
        """
        B = generated_ids.shape[0]
        chunk_len = H * action_dim
        n_bins = int(self.config.framework.action_model.get("num_bins", 256))
        act_min = self._act0_id
        act_max = self._act0_id + n_bins  # exclusive
        mid_id = self._act0_id + n_bins // 2

        new_ids = self._extract_generated_tail(generated_ids, prompt_len, chunk_len)

        decoded = np.zeros((B, H, action_dim), dtype=np.float32)
        for b in range(B):
            row = new_ids[b]
            mask = (row >= act_min) & (row < act_max)
            action_ids = row[mask].tolist()
            if len(action_ids) < chunk_len:
                action_ids += [mid_id] * (chunk_len - len(action_ids))
            elif len(action_ids) > chunk_len:
                action_ids = action_ids[:chunk_len]
            chunk = self.action_tokenizer.decode(action_ids).reshape(H, action_dim)
            decoded[b] = chunk
        return decoded
```

Also add `import numpy as np` near the top of the file if not already present. Verify with:
```bash
grep -n "^import numpy\|^from numpy" starVLA/model/framework/VLM4A/UamVLA.py
```
If absent, add `import numpy as np` to the top imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_uamvla_predict_action_decode.py -v`

Expected: 7/7 PASS (2 from Task 2 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py tests/test_uamvla_predict_action_decode.py
git commit -m "[uamvla] Add _decode_generated_actions: ID-range scan + mid_bin padding"
```

---

## Task 4: `_build_prefill_generate_kwargs` — state splice + multimodal kwargs

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLA.py` (add method)
- Create: `tests/test_uamvla_predict_action_prefill.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_uamvla_predict_action_prefill.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_uamvla_predict_action_prefill.py -v`

Expected: 5 tests FAIL with `AttributeError: ... has no attribute '_build_prefill_generate_kwargs'`.

- [ ] **Step 3: Implement `_build_prefill_generate_kwargs`**

Edit `starVLA/model/framework/VLM4A/UamVLA.py`. Add this method **immediately before** `_extract_generated_tail` (added in Task 2):

```python
    def _build_prefill_generate_kwargs(self, qwen_inputs: dict) -> dict:
        """Prefill: embed input_ids, splice in state-token embeddings, and return
        kwargs suitable for model.generate().

        The state splice is identical to the training forward path. We keep the
        original input_ids in the generation kwargs so generation-time helpers
        and logits processors can see the textual prompt.
        """
        from starVLA.model.modules.uamvla.backbone_wrapper import _replace_state_tokens

        iface = self.qwen_vl_interface
        input_ids = qwen_inputs["input_ids"]
        canonical_state = qwen_inputs.get("canonical_state")

        embed_tokens = iface.get_embed_tokens()
        base_embeds = embed_tokens(input_ids)

        if canonical_state is not None:
            state_embeds = iface.state_encoder(canonical_state)
            inputs_embeds = _replace_state_tokens(
                inputs_embeds=base_embeds,
                input_ids=input_ids,
                state_embeds=state_embeds,
                embodiment=iface.embodiment,
                tokenizer=iface.tokenizer,
            )
        else:
            inputs_embeds = base_embeds

        # Preserve or derive mm_token_type_ids for Qwen3-VL multimodal generation.
        mm_token_type_ids = qwen_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None and hasattr(iface, "_derive_mm_token_type_ids"):
            mm_token_type_ids = iface._derive_mm_token_type_ids(
                input_ids=input_ids,
                pixel_values=qwen_inputs.get("pixel_values"),
                provided=None,
            )

        gen_kwargs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": qwen_inputs.get("attention_mask"),
            "pixel_values": qwen_inputs.get("pixel_values"),
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
        }
        if mm_token_type_ids is not None:
            gen_kwargs["mm_token_type_ids"] = mm_token_type_ids
        return {k: v for k, v in gen_kwargs.items() if v is not None}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_uamvla_predict_action_prefill.py -v`

Expected: 5/5 PASS.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py tests/test_uamvla_predict_action_prefill.py
git commit -m "[uamvla] Add _build_prefill_generate_kwargs: state splice + mm_token_type_ids"
```

---

## Task 5: Rewrite `predict_action` main loop

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLA.py:340-381` (replace existing `predict_action` body)

- [ ] **Step 1: Read current `predict_action`**

Run: `sed -n '340,385p' starVLA/model/framework/VLM4A/UamVLA.py`

Expected: the current placeholder implementation that calls `ActionHead.predict` and `_decode_action_tokens`.

- [ ] **Step 2: Replace the body**

Edit `starVLA/model/framework/VLM4A/UamVLA.py`. Find the current `predict_action` method (around line 340-381). Replace the entire method (from `@torch.inference_mode()` decorator down to the end of `return {"normalized_actions": normalized_actions}`) with:

```python
    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        """Eval-time inference: autoregressive action chunk generation.

        Builds prompt + state-spliced inputs_embeds, calls model.generate with
        ActionLogitsProcessor active by default, and decodes the generated
        action tokens via ID-range scan.

        Args:
            examples: a single example dict or list of example dicts. Each must have:
              - "image": List[PIL.Image] or np.ndarray (multi-view per spec §4.6)
              - "lang": str instruction
              - "canonical_state": canonical state tensor / dict (or omit for all examples
                if state-less inference is desired; mixing within a batch raises ValueError)
            **kwargs:
              - constrain_action_logits (bool, default True): if True, apply
                ActionLogitsProcessor to constrain post-<|action_start|> logits to
                action-bin tokens. Set False for ablations / debugging.
              - other kwargs: ignored (compat with diffusion-style frameworks).

        Returns:
            {"normalized_actions": np.ndarray of shape (B, H, action_dim)}
        """
        if not isinstance(examples, list):
            examples = [examples]

        from deployment.model_server.tools.image_tools import to_pil_preserve
        from transformers import LogitsProcessorList
        from starVLA.model.modules.uamvla.inference import ActionLogitsProcessor

        # Reject mixed canonical_state presence (stack_canonical can't handle partial dicts).
        states = [e.get("canonical_state") for e in examples]
        has_state = [s is not None for s in states]
        if any(has_state) and not all(has_state):
            raise ValueError(
                "predict_action requires canonical_state for all examples or none; "
                "mixed presence is not supported."
            )

        # Inference uses left padding so each row's last non-pad prompt token
        # aligns to the same column (required for batched generate to step
        # uniformly). NOTE: this mutates a process-global tokenizer attribute;
        # do not call predict_action from multiple threads in the same process.
        tokenizer = self.qwen_vl_interface.tokenizer
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            qwen_inputs = self.qwen_vl_interface.build_inputs(
                images=[[to_pil_preserve(img) for img in e["image"]] for e in examples],
                instructions=[e["lang"] for e in examples],
                canonical_state=stack_canonical(states) if all(has_state) else None,
            )
        finally:
            tokenizer.padding_side = old_padding_side

        gen_kwargs = self._build_prefill_generate_kwargs(qwen_inputs)

        H = self.action_horizon
        action_dim = int(self.config.framework.embodiment.get("action_dim", 7))
        chunk_len = H * action_dim
        n_bins = int(self.config.framework.action_model.get("num_bins", 256))

        constrain_action_logits = kwargs.get("constrain_action_logits", True)
        logits_processor = None
        if constrain_action_logits:
            logits_processor = LogitsProcessorList([
                ActionLogitsProcessor(
                    action_start_id=self.action_start_id,
                    action_begin_id=self._act0_id,
                    n_bins=n_bins,
                    action_chunk_len=chunk_len,
                )
            ])

        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.qwen_vl_interface.model.generate(
                **gen_kwargs,
                max_new_tokens=chunk_len,
                min_new_tokens=chunk_len,
                logits_processor=logits_processor,
                do_sample=False,
                use_cache=True,
            )

        normalized_actions = self._decode_generated_actions(
            generated_ids,
            prompt_len=gen_kwargs["inputs_embeds"].shape[1],
            H=H,
            action_dim=action_dim,
        )
        return {"normalized_actions": normalized_actions}
```

- [ ] **Step 3: Verify import compiles**

Run: `python -c "from starVLA.model.framework.VLM4A.UamVLA import UamVLA; print('OK')"`

Expected: `OK`. If `ImportError`, fix the imports (most likely missing `LogitsProcessorList` or `ActionLogitsProcessor` re-export issues).

- [ ] **Step 4: Verify existing init tests still pass (no regression on non-inference paths)**

Run: `pytest tests/test_uamvla_l1_init.py tests/test_uamvla_framework_init.py -v`

Expected: same baseline as Task 1 (3 PASS + 1 SKIP for `test_uamvla_l1_init.py`, 2 PASS for `test_uamvla_framework_init.py`).

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py
git commit -m "[uamvla] Rewrite predict_action: prefill + HF generate + ID-range decode"
```

---

## Task 6: Cleanup dead code

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLA.py` (remove `_extract_action_tokens_for_sample` and old `_decode_action_tokens`)

- [ ] **Step 1: Find the dead methods**

Run: `grep -n "def _extract_action_tokens_for_sample\|def _decode_action_tokens" starVLA/model/framework/VLM4A/UamVLA.py`

Expected: two matches, around lines 383 and 407 (line numbers may have shifted after Tasks 1-5).

- [ ] **Step 2: Remove `_extract_action_tokens_for_sample` and old `_decode_action_tokens`**

Edit `starVLA/model/framework/VLM4A/UamVLA.py`. Delete the entire method bodies of:

1. `_decode_action_tokens` (the OLD one that calls `_extract_action_tokens_for_sample`; do NOT delete the new `_decode_generated_actions` from Task 3)
2. `_extract_action_tokens_for_sample` (currently a `NotImplementedError` shim)

These are no longer reachable: Task 5's new `predict_action` calls `_decode_generated_actions` directly.

After the edit, verify with:
```bash
grep -c "def _extract_action_tokens_for_sample\|def _decode_action_tokens" starVLA/model/framework/VLM4A/UamVLA.py
```
Expected: `0`

And confirm `_decode_generated_actions` still exists:
```bash
grep -c "def _decode_generated_actions" starVLA/model/framework/VLM4A/UamVLA.py
```
Expected: `1`

- [ ] **Step 3: Verify no callers of the deleted methods remain**

Run: `grep -rn "_extract_action_tokens_for_sample\|_decode_action_tokens" starVLA/ tests/ --include="*.py"`

Expected: zero matches. (If any remain, those are stale references that need to be removed.)

- [ ] **Step 4: Quick smoke that file still imports**

Run: `python -c "from starVLA.model.framework.VLM4A.UamVLA import UamVLA; print('OK')"`

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py
git commit -m "[uamvla] Remove dead _extract_action_tokens_for_sample and old _decode_action_tokens"
```

---

## Task 7: L1 smoke tests for full `predict_action`

**Files:**
- Modify: `tests/test_uamvla_predict_action.py` (replace existing skip-placeholder)

- [ ] **Step 1: Replace the existing skip-only file**

Replace the entire content of `tests/test_uamvla_predict_action.py` with:

```python
"""L1 smoke tests for UamVLA.predict_action.

These tests build a minimal UamVLA via the same monkeypatch pattern as
test_uamvla_l1_init.py, mock model.generate to return predetermined ACT token
sequences, and verify the full inference path: input building → prefill →
generate → decode → normalized actions."""
import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
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
        unk_token_id = 1
        padding_side = "right"
        def __init__(self):
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
    model.qwen_vl_interface.build_inputs = fake_build_inputs
    act_ids = torch.full((1, chunk_len), model._act0_id, dtype=torch.long)
    generated_ids = torch.cat(
        [torch.zeros(1, prompt_len, dtype=torch.long), act_ids], dim=1
    )
    _patch_predict_dependencies(
        model, fake_backbone, generated_ids,
        prompt_len=prompt_len, hidden_size=hidden_size,
    )
    # The patch above re-set build_inputs; re-set our observer build_inputs again
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
```

- [ ] **Step 2: Run the new tests**

Run: `pytest tests/test_uamvla_predict_action.py -v`

Expected: 8/8 PASS.

If any test fails, inspect its assertion against the implementation in Task 5 — most likely the test's mock path needs adjustment, not the production code.

- [ ] **Step 3: Commit**

```bash
git add tests/test_uamvla_predict_action.py
git commit -m "[test] L1 smoke tests for predict_action: shape, padding, logits processor"
```

---

## Task 8: Qwen3-VL real-model generation-shape smoke (skip-friendly)

**Files:**
- Create: `tests/test_uamvla_predict_action_qwen3vl_generate.py`

This task adds a smoke test that runs against a real Qwen3-VL model when one is locally available, otherwise skips. It exists to verify two things that text-only mocks can't: (a) the real backbone accepts `input_ids` + `inputs_embeds` together, and (b) we can record whether the return shape is prompt+new or new-only.

- [ ] **Step 1: Create the skip-friendly test**

Create `tests/test_uamvla_predict_action_qwen3vl_generate.py`:

```python
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
```

- [ ] **Step 2: Run the test (skipped if no checkpoint)**

Run: `pytest tests/test_uamvla_predict_action_qwen3vl_generate.py -v`

Expected (without `QWEN3VL_TEST_CKPT` env var): 2 SKIP. With `QWEN3VL_TEST_CKPT` pointing at a local Qwen3-VL checkpoint: 2 PASS.

If you have a checkpoint locally, set the env var and run once to record the actual return shape:
```bash
QWEN3VL_TEST_CKPT=/path/to/local/qwen3-vl pytest tests/test_uamvla_predict_action_qwen3vl_generate.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_uamvla_predict_action_qwen3vl_generate.py
git commit -m "[test] Qwen3-VL real-model generate-shape smoke (skip if no fixture)"
```

---

## Task 9: Training-vs-inference numerical alignment test

**Files:**
- Create: `tests/test_uamvla_predict_action_alignment.py`

This is the spec's critical safety net: verify that the prefix hidden state at the `<|action_start|>` position is identical between the training forward path and the inference prefill path. Catches any divergence in state splice, chat template, or tokenization.

- [ ] **Step 1: Create the alignment test**

Create `tests/test_uamvla_predict_action_alignment.py`:

```python
"""Critical safety net: training forward and inference prefill must produce
identical hidden states at the <|action_start|> position.

Catches any divergence in:
- State token splice
- chat_template rendering
- Tokenization / padding
- Multimodal kwargs (mm_token_type_ids, image_grid_thw)
- inputs_embeds construction order

Skipped if a real Qwen3-VL backbone is not available (the test requires a real
forward pass to compare hidden states meaningfully)."""
import os
import pytest
import torch


_QWEN3_VL_PATH = os.environ.get("QWEN3VL_TEST_CKPT")


@pytest.mark.skipif(
    not _QWEN3_VL_PATH or not os.path.isdir(_QWEN3_VL_PATH),
    reason="QWEN3VL_TEST_CKPT not set; alignment requires a real Qwen3-VL backbone",
)
def test_prefix_hidden_states_match_between_training_and_inference():
    """Run training forward and inference prefill on the same example; the
    hidden state at <|action_start|>'s position must match within atol=1e-4.

    This is the strongest invariant-check available in the test suite. If this
    test fails, the inference path has silently diverged from training and
    LIBERO eval results will not reflect what the model actually learned."""
    pytest.importorskip("transformers")
    from PIL import Image

    from starVLA.model.framework.VLM4A.UamVLA import UamVLA, UamVLADefaultConfig
    from starVLA.model.modules.uamvla.backbone_wrapper import _replace_state_tokens
    from omegaconf import OmegaConf

    # Build a minimal real config pointing at the local fixture.
    cfg = OmegaConf.create({
        "framework": {
            "qwenvl": {
                "base_vlm": _QWEN3_VL_PATH,
                "attn_implementation": "eager",
                "dtype": "float32",  # fp32 for numerical comparison
            },
            "embodiment": {"name": "franka_libero", "action_dim": 7},
            "action_model": {"num_bins": 256, "future_action_window_size": 7,
                              "action_dim": 7},
            "aux_heads": {
                "action": {"enabled": True, "lr": 1e-4, "loss_weight": 1.0},
                "pose":   {"enabled": False},
                "future": {"enabled": False},
                "recon":  {"enabled": False},
            },
        },
    })
    model = UamVLA(config=cfg)
    model.eval()

    img = Image.new("RGB", (640, 640))
    example = {
        "image": [img],
        "lang": "pick up the red block",
        "canonical_state": {
            "arm_0": {
                "ee_pose": torch.zeros(9),
                "joint_pos": torch.zeros(7),
            },
            "gripper_0": torch.zeros(1),
        },
    }

    from starVLA.model.modules.uamvla.collator_helpers import stack_canonical
    from deployment.model_server.tools.image_tools import to_pil_preserve

    # Path A: training-style forward with canonical_state
    qwen_inputs = model.qwen_vl_interface.build_inputs(
        images=[[to_pil_preserve(img) for img in example["image"]]],
        instructions=[example["lang"]],
        canonical_state=stack_canonical([example["canonical_state"]]),
    )
    with torch.no_grad():
        out_train = model.qwen_vl_interface(
            **qwen_inputs, output_hidden_states=True, return_dict=True,
        )
        hidden_train = out_train.hidden_states[-1]  # (1, S, H)

    # Path B: inference prefill via _build_prefill_generate_kwargs, then forward
    gen_kwargs = model._build_prefill_generate_kwargs(qwen_inputs)
    with torch.no_grad():
        out_infer = model.qwen_vl_interface.model.model(
            input_ids=None,
            inputs_embeds=gen_kwargs["inputs_embeds"],
            attention_mask=gen_kwargs["attention_mask"],
            pixel_values=gen_kwargs.get("pixel_values"),
            image_grid_thw=gen_kwargs.get("image_grid_thw"),
            mm_token_type_ids=gen_kwargs.get("mm_token_type_ids"),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_infer = out_infer.hidden_states[-1]

    # Locate the <|action_start|> position in input_ids
    action_start_id = model.action_start_id
    input_ids = qwen_inputs["input_ids"]
    pos = (input_ids[0] == action_start_id).nonzero(as_tuple=True)[0]
    assert pos.numel() == 1, (
        f"Expected exactly one <|action_start|> in prompt, found {pos.numel()}"
    )
    p = int(pos.item())

    h_train = hidden_train[0, p, :]
    h_infer = hidden_infer[0, p, :]

    assert torch.allclose(h_train, h_infer, atol=1e-4), (
        f"Hidden state at <|action_start|> diverges between training and inference: "
        f"max diff = {(h_train - h_infer).abs().max().item():.6f}"
    )
```

- [ ] **Step 2: Run the alignment test**

Run: `pytest tests/test_uamvla_predict_action_alignment.py -v`

Expected (without `QWEN3VL_TEST_CKPT`): 1 SKIP. With it: 1 PASS.

If the test fails with `QWEN3VL_TEST_CKPT` set, **do not raise the `atol` to make it pass**. Instead investigate:
1. Is autocast accidentally enabled? (Use fp32, `model.eval()`)
2. Does the state splice happen at the same position in both paths?
3. Are pixel_values / image_grid_thw / mm_token_type_ids identical?
4. Does the chat template emit `<|action_start|>` at the same position?

A genuine divergence here means the inference path has silently broken training-equivalent behavior.

- [ ] **Step 3: Commit**

```bash
git add tests/test_uamvla_predict_action_alignment.py
git commit -m "[test] Train-vs-infer prefix hidden state alignment (skip if no Qwen3-VL ckpt)"
```

---

## Task 10: Final verification

**Files:** none modified — verification-only.

- [ ] **Step 1: Run all new test files**

Run:
```bash
pytest tests/test_uamvla_predict_action_decode.py \
       tests/test_uamvla_predict_action_prefill.py \
       tests/test_uamvla_predict_action.py \
       tests/test_uamvla_predict_action_qwen3vl_generate.py \
       tests/test_uamvla_predict_action_alignment.py \
       -v
```

Expected (without local Qwen3-VL ckpt):
- decode: 7 PASS
- prefill: 5 PASS
- predict_action L1: 8 PASS
- qwen3vl: 2 SKIP
- alignment: 1 SKIP

Total: 20 PASS, 3 SKIP.

With `QWEN3VL_TEST_CKPT` set: 23 PASS.

- [ ] **Step 2: Run full repo regression**

Run: `pytest tests/ -v`

Expected: no NEW failures vs baseline. Baseline (after the action-start-token spec) was 74 passed, 8 skipped. After this plan: 94 passed, 11 skipped (added 20 PASS + 3 SKIP).

If a previously-passing test now fails, investigate. Likely culprits:
- A test that depended on `_extract_action_tokens_for_sample` raising NotImplementedError (it's removed now)
- A test that asserted specific `predict_action` output format (the contract is preserved, but inspect anyway)

- [ ] **Step 3: Verify dead code is fully removed**

Run: `grep -rn "_extract_action_tokens_for_sample\|def _decode_action_tokens" starVLA/ tests/ --include="*.py"`

Expected: zero matches. (The new `_decode_generated_actions` should not match this grep.)

- [ ] **Step 4: Final sanity check on the new module surface**

Run:
```bash
python -c "
from starVLA.model.framework.VLM4A.UamVLA import UamVLA
assert hasattr(UamVLA, '_build_prefill_generate_kwargs'), 'missing _build_prefill_generate_kwargs'
assert hasattr(UamVLA, '_extract_generated_tail'), 'missing _extract_generated_tail'
assert hasattr(UamVLA, '_decode_generated_actions'), 'missing _decode_generated_actions'
assert not hasattr(UamVLA, '_extract_action_tokens_for_sample'), 'dead method still present'
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 5: Commit any final fixes (only if regressions were found)**

If Step 2 or 4 surfaced issues:
```bash
git add <fixed_files>
git commit -m "[<scope>] Fix regression introduced by predict_action rewrite"
```

If everything passed cleanly: no commit needed for Task 10.

---

## Self-Review

**1. Spec coverage:**
- §2 Goal "Build prefill `inputs_embeds` with state splice" → Task 4
- §2 Goal "Call `model.generate(input_ids=..., inputs_embeds=..., max_new_tokens=...)` with multimodal kwargs" → Task 5
- §2 Goal "return-shape aware extractor" → Task 2
- §2 Goal "Decode via `ActionTokenizer`" → Task 3
- §2 Goal "Preserve interface; B≥1 from day one" → Task 5 + Task 7 (B≥1 test)
- §2 Goal "Training-vs-inference alignment test" → Task 9
- §4.1 Plan A architecture → Tasks 4 + 5
- §4.1 Choice 2 default `ActionLogitsProcessor` → Task 5 (with `constrain_action_logits` kwarg) + Task 7 default-and-disable tests
- §4.1 Choice 5 left padding → Task 5 try/finally + Task 7 padding-restore test
- §4.5 ID contiguity assertion → Task 1
- §4.5 `_extract_generated_tail` shape policy → Task 2
- §4.7 `constrain_action_logits` kwarg → Task 5 (default True)
- §4.8 Strong invariants 1-6 → covered across tests
- §5 file changes → Tasks 1-9 cover all 5 deliverable files
- §6.1 / §6.2 / §6.3 / §6.4 / §6.5 testing plan → Tasks 2/3 / 4 / 7 / 8 / 9 respectively

**2. Placeholder scan:** No "TBD" / "TODO" / "implement later" in the plan. Every step contains complete code or exact commands.

**3. Type consistency:**
- `_build_prefill_generate_kwargs(self, qwen_inputs: dict) -> dict` — defined Task 4, called Task 5. Consistent.
- `_extract_generated_tail(self, generated_ids, prompt_len, chunk_len) -> torch.Tensor` — defined Task 2, called by `_decode_generated_actions` in Task 3 and tested directly in Task 8. Consistent.
- `_decode_generated_actions(self, generated_ids, prompt_len, H, action_dim) -> np.ndarray` — defined Task 3, called Task 5. Consistent.
- `self._act0_id: int` — defined Task 1, used Tasks 3, 5, and tests. Consistent.
- `self.action_start_id: int` — pre-existing from prior spec, used Task 5. Consistent with prior spec.
- `ActionLogitsProcessor(action_start_id, action_begin_id, n_bins, action_chunk_len)` signature — used Task 5; matches the existing class definition in `inference/action_logits_processor.py`.

No type / naming inconsistencies found.
