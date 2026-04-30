# `<|action_start|>` Special Token Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the chat-template literal `<action_start>` to a single registered Qwen-style special token `<|action_start|>`, expose `action_start_id` on the `UamVLA` framework, and provide a reusable `ActionLogitsProcessor` for generate-time logits masking — so future generate-based inference has a stable, O(1) sentinel for the action segment.

**Architecture:** Co-locate structural-token registration with state-token registration in `state_encoder/special_tokens.py` (both are bootstrap-time, embodiment-independent). Update both Qwen3 chat-template variants to emit the new token. Resolve `action_start_id` in `UamVLA.__init__` with hard-error semantics (catch silent `unk_token_id` collisions). Implement `ActionLogitsProcessor` as a stateful `transformers.LogitsProcessor` with per-row independence under a new `inference/` package.

**Tech Stack:** Python 3, PyTorch, HuggingFace `transformers` (tokenizer API + `LogitsProcessor`), pytest.

**Reference spec:** [`docs/superpowers/specs/2026-04-30-action-start-special-token-design.md`](../specs/2026-04-30-action-start-special-token-design.md)

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `starVLA/model/modules/uamvla/state_encoder/special_tokens.py` | **Modify** | Add `STRUCTURAL_SPECIAL_TOKENS` constant + `register_structural_tokens()` sibling function |
| `starVLA/model/modules/uamvla/backbone_wrapper.py` | **Modify** (+1 line) | Call `register_structural_tokens()` after `register_state_tokens()` in `build_uamvla_backbone()` |
| `starVLA/model/modules/uamvla/data/chat_template.py` | **Modify** (2 sites + constant) | Replace `<action_start>` → `<\|action_start\|>` in both chat templates; add `ACTION_START_TOKEN` constant |
| `starVLA/model/framework/VLM4A/UamVLA.py` | **Modify** (~10 lines) | Resolve `self.action_start_id` after `_act0_id`; hard-error on silent unk collision |
| `starVLA/model/modules/uamvla/inference/__init__.py` | **Create** | Empty package marker |
| `starVLA/model/modules/uamvla/inference/action_logits_processor.py` | **Create** | `ActionLogitsProcessor` class — per-row state machine constraining logits to action bins after `<\|action_start\|>` |
| `tests/test_structural_tokens.py` | **Create** | Tests for `register_structural_tokens` (registration, idempotency, id sanity) |
| `tests/test_chat_template.py` | **Create** | Tests verifying both chat templates emit `<\|action_start\|>` exactly once |
| `tests/test_action_logits_processor.py` | **Create** | Tests for `ActionLogitsProcessor` (off-mode, activation, duration, per-row independence, re-entry) |
| `tests/test_uamvla_framework_init.py` | **Modify** | Extend existing init test to assert `action_start_id` is exposed and != `unk_token_id` |

---

## Task 1: `STRUCTURAL_SPECIAL_TOKENS` + `register_structural_tokens()`

**Files:**
- Modify: `starVLA/model/modules/uamvla/state_encoder/special_tokens.py`
- Create: `tests/test_structural_tokens.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_structural_tokens.py`:

```python
"""Tests for register_structural_tokens — adds <|action_start|> as a single special token."""
import pytest


class _FakeTokenizer:
    """Minimal HF-tokenizer-like stub: tracks added special tokens and assigns ids."""
    unk_token_id = 1

    def __init__(self):
        self._vocab = {}
        self._special_tokens = []
        self._next_id = 100

    def add_special_tokens(self, payload):
        added = 0
        for tok in payload.get("additional_special_tokens", []):
            if tok not in self._vocab:
                self._vocab[tok] = self._next_id
                self._next_id += 1
                self._special_tokens.append(tok)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)

    def get_input_embeddings(self):
        return self  # stand-in; only .weight.shape[0] is checked

    @property
    def weight(self):
        class _W:
            shape = (self._next_id,)
        w = _W()
        w.shape = (self._next_id,)
        return w

    def __len__(self):
        return self._next_id


class _FakeBackbone:
    """Stub exposing resize_token_embeddings + get_embed_tokens."""
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.resized_to = None

    def resize_token_embeddings(self, new_size):
        self.resized_to = new_size

    def get_embed_tokens(self):
        # Return an object with .weight.shape[0] == tokenizer length
        class _E:
            class weight:
                shape = (len(self._tokenizer),)
        e = _E()
        e.weight.shape = (len(self._tokenizer),)
        return e


def test_register_structural_tokens_adds_action_start():
    from starVLA.model.modules.uamvla.state_encoder.special_tokens import (
        register_structural_tokens, STRUCTURAL_SPECIAL_TOKENS,
    )
    assert "<|action_start|>" in STRUCTURAL_SPECIAL_TOKENS

    tok = _FakeTokenizer()
    bb = _FakeBackbone(tok)
    ids = register_structural_tokens(tok, bb)

    assert "<|action_start|>" in ids
    assert ids["<|action_start|>"] == tok.convert_tokens_to_ids("<|action_start|>")
    assert ids["<|action_start|>"] != tok.unk_token_id
    assert bb.resized_to == len(tok)


def test_register_structural_tokens_is_idempotent():
    from starVLA.model.modules.uamvla.state_encoder.special_tokens import (
        register_structural_tokens,
    )
    tok = _FakeTokenizer()
    bb = _FakeBackbone(tok)
    ids_first = register_structural_tokens(tok, bb)
    bb.resized_to = None  # reset to detect a second resize
    ids_second = register_structural_tokens(tok, bb)

    assert ids_first == ids_second
    # Second call must NOT trigger another resize (no new tokens added).
    assert bb.resized_to is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_structural_tokens.py -v`
Expected: FAIL with `ImportError: cannot import name 'register_structural_tokens'`.

- [ ] **Step 3: Implement `STRUCTURAL_SPECIAL_TOKENS` + `register_structural_tokens`**

Edit `starVLA/model/modules/uamvla/state_encoder/special_tokens.py` — append after the existing `register_state_tokens` function (do not modify existing code):

```python
# ---------------------------------------------------------------------------
# Structural tokens — segment markers used by inference-time logits processors
# (e.g. <|action_start|> as a generate-time sentinel for the action chunk).
# Co-located with state tokens because both are bootstrap-time, embodiment-
# independent, and registered via the same HF add_special_tokens path.
# ---------------------------------------------------------------------------

STRUCTURAL_SPECIAL_TOKENS: list[str] = [
    "<|action_start|>",
]


def register_structural_tokens(tokenizer, backbone) -> dict[str, int]:
    """Register structural special tokens (e.g. <|action_start|>) with the tokenizer.

    Same contract as register_state_tokens: idempotent, resizes embeddings on
    first call only, returns dict[token_str -> token_id].
    """
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": STRUCTURAL_SPECIAL_TOKENS}
    )
    if n_added > 0:
        new_size = len(tokenizer)
        backbone.resize_token_embeddings(new_size)
        logger.info(
            f"[Structural Tokens] Registered {n_added} new structural tokens. "
            f"Resized embeddings to {new_size}."
        )

    embeds = (backbone.get_embed_tokens() if hasattr(backbone, "get_embed_tokens")
              else backbone.get_input_embeddings())
    embed_size = embeds.weight.shape[0]
    if embed_size != len(tokenizer):
        raise RuntimeError(
            f"Vocab/embedding size mismatch after structural token registration: "
            f"tokenizer={len(tokenizer)}, embeddings={embed_size}"
        )

    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok)
                 for tok in STRUCTURAL_SPECIAL_TOKENS}
    logger.debug(
        "[Structural Tokens] Token IDs: " +
        ", ".join(f"{tok}={tid}" for tok, tid in token_ids.items())
    )
    return token_ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_structural_tokens.py -v`
Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/modules/uamvla/state_encoder/special_tokens.py tests/test_structural_tokens.py
git commit -m "[state_encoder] Add register_structural_tokens for <|action_start|>"
```

---

## Task 2: Wire `register_structural_tokens` into backbone factory

**Files:**
- Modify: `starVLA/model/modules/uamvla/backbone_wrapper.py:504-505`

- [ ] **Step 1: Read current backbone factory wiring**

Run: `sed -n '500,510p' starVLA/model/modules/uamvla/backbone_wrapper.py`
Expected output:
```
    # Register state special tokens via shim (resizes embeddings in-place).
    shim = _ResizeShim(qwen_model)
    register_state_tokens(tokenizer, shim)
```

- [ ] **Step 2: Update the import and add the call**

Edit `starVLA/model/modules/uamvla/backbone_wrapper.py`:

Find this block (around line 31-33):
```python
from starVLA.model.modules.uamvla.state_encoder.special_tokens import (
    register_state_tokens,
)
```

Replace with:
```python
from starVLA.model.modules.uamvla.state_encoder.special_tokens import (
    register_state_tokens,
    register_structural_tokens,
)
```

Then find this block (around line 503-505):
```python
    # Register state special tokens via shim (resizes embeddings in-place).
    shim = _ResizeShim(qwen_model)
    register_state_tokens(tokenizer, shim)
```

Replace with:
```python
    # Register state + structural special tokens via shim (resizes embeddings in-place).
    shim = _ResizeShim(qwen_model)
    register_state_tokens(tokenizer, shim)
    register_structural_tokens(tokenizer, shim)
```

- [ ] **Step 3: Verify the import compiles**

Run: `python -c "from starVLA.model.modules.uamvla.backbone_wrapper import build_uamvla_backbone, _ResizeShim; print('OK')"`
Expected: `OK` (no import error).

- [ ] **Step 4: Commit**

```bash
git add starVLA/model/modules/uamvla/backbone_wrapper.py
git commit -m "[backbone] Register <|action_start|> structural token at backbone-build time"
```

---

## Task 3: Update chat templates

**Files:**
- Modify: `starVLA/model/modules/uamvla/data/chat_template.py:100,137`
- Create: `tests/test_chat_template.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chat_template.py`:

```python
"""Tests for chat templates — verify <|action_start|> is emitted as a single literal."""


def test_qwen3_vl_chat_template_emits_action_start_token():
    from starVLA.model.modules.uamvla.data.chat_template import (
        Qwen3VLChatTemplate, ACTION_START_TOKEN,
    )
    assert ACTION_START_TOKEN == "<|action_start|>"

    tpl = Qwen3VLChatTemplate()
    out = tpl.format(
        instruction="pick up the red block",
        embodiment="franka_libero",
        view_names=["primary"],
    )

    # Exactly one occurrence of the new token, zero of the old literal.
    assert out.count("<|action_start|>") == 1
    assert "<action_start>" not in out  # old literal must be gone


def test_qwen3_chat_template_emits_action_start_token():
    from starVLA.model.modules.uamvla.data.chat_template import Qwen3ChatTemplate

    tpl = Qwen3ChatTemplate()
    out = tpl.format(
        instruction="pick up the red block",
        embodiment="franka_libero",
        view_names=["primary"],
    )

    assert out.count("<|action_start|>") == 1
    assert "<action_start>" not in out


def test_action_start_token_constant_exported():
    from starVLA.model.modules.uamvla.data import chat_template
    assert hasattr(chat_template, "ACTION_START_TOKEN")
    assert chat_template.ACTION_START_TOKEN == "<|action_start|>"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat_template.py -v`
Expected: FAIL on the first assertion or `ImportError: cannot import name 'ACTION_START_TOKEN'`.

- [ ] **Step 3: Add `ACTION_START_TOKEN` constant**

Edit `starVLA/model/modules/uamvla/data/chat_template.py`. Find this block near the top (around line 12-14):

```python
QWEN3_VL_VISION_START = "<|vision_start|>"
QWEN3_VL_VISION_END = "<|vision_end|>"
QWEN3_VL_IMAGE_PAD = "<|image_pad|>"
```

Append directly below:

```python

# Structural marker for the action chunk in the assistant turn.
# Registered as a single special token by register_structural_tokens (see
# state_encoder/special_tokens.py). Used by inference-time LogitsProcessors
# to trigger action-bin masking in generate().
ACTION_START_TOKEN = "<|action_start|>"
```

- [ ] **Step 4: Update both chat templates**

Edit `starVLA/model/modules/uamvla/data/chat_template.py`. Find line 100 inside `Qwen3ChatTemplate.format`:

```python
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n<action_start>"
```

Replace with:

```python
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{ACTION_START_TOKEN}"
```

Find line 137 inside `Qwen3VLChatTemplate.format`:

```python
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n<action_start>"
```

Replace with:

```python
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{ACTION_START_TOKEN}"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_chat_template.py -v`
Expected: PASS for all three tests.

- [ ] **Step 6: Sweep for any other references to the old literal**

Run: `grep -rn '<action_start>' starVLA/ tests/ --include='*.py'`
Expected: no matches (the only occurrences were the two we just edited).

If any matches appear, replace them with `<|action_start|>` (or import `ACTION_START_TOKEN` and use it) and re-run the test.

- [ ] **Step 7: Commit**

```bash
git add starVLA/model/modules/uamvla/data/chat_template.py tests/test_chat_template.py
git commit -m "[chat_template] Replace literal <action_start> with single-token <|action_start|>"
```

---

## Task 4: Wire `action_start_id` into `UamVLA` framework

**Files:**
- Modify: `starVLA/model/framework/VLM4A/UamVLA.py:108-112`
- Modify: `tests/test_uamvla_framework_init.py`

- [ ] **Step 1: Read the existing UamVLA init test**

Run: `cat tests/test_uamvla_framework_init.py`
Note the existing test patterns — we will add to them rather than replace.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_uamvla_framework_init.py` (do not modify existing tests):

```python


def test_uamvla_framework_exposes_action_start_id():
    """UamVLA must expose a positive action_start_id distinct from unk_token_id."""
    import torch
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA

    # Use the same minimal config that test_uamvla_framework_init already builds.
    # If your existing tests use a fixture, reuse it via importing/calling here.
    # Otherwise, construct UamVLA the same way the existing init test does.
    # (Adapt this block to match the existing test setup pattern.)
    framework = _build_minimal_uamvla()  # helper from existing test file

    assert hasattr(framework, "action_start_id"), (
        "UamVLA must expose self.action_start_id"
    )
    assert isinstance(framework.action_start_id, int)
    assert framework.action_start_id > 0
    tok = framework._tokenizer
    if hasattr(tok, "unk_token_id") and tok.unk_token_id is not None:
        assert framework.action_start_id != tok.unk_token_id
```

**Note for implementer:** the existing `test_uamvla_framework_init.py` already constructs a minimal UamVLA. Reuse whatever construction helper / fixture it uses (it may be inline in another test). If the existing tests construct UamVLA directly inline, copy that exact construction code into a new `_build_minimal_uamvla()` helper at the top of the file and have both the new and existing tests call it. The point is: do not invent a new construction path; mirror the existing one.

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_uamvla_framework_init.py::test_uamvla_framework_exposes_action_start_id -v`
Expected: FAIL with `AttributeError: 'UamVLA' object has no attribute 'action_start_id'`.

- [ ] **Step 4: Implement `action_start_id` resolution in `UamVLA.__init__`**

Edit `starVLA/model/framework/VLM4A/UamVLA.py`. Find lines 108-112:

```python
        # Resolve action_token_begin_id from the newly added <ACT_0> token.
        try:
            _act0_id = int(_tokenizer.convert_tokens_to_ids("<ACT_0>"))
        except (TypeError, ValueError):
            _act0_id = 0
```

Append directly below (before the blank line that precedes `# Aux heads`):

```python

        # Resolve action_start_id from the structural <|action_start|> token registered
        # at backbone-build time. Hard-error on silent unk_token_id collisions so that
        # a missing register_structural_tokens() call surfaces here, not at training time.
        try:
            _action_start_id = int(_tokenizer.convert_tokens_to_ids("<|action_start|>"))
            _unk_id = getattr(_tokenizer, "unk_token_id", None)
            if _action_start_id is None or (
                _unk_id is not None and _action_start_id == _unk_id
            ):
                raise RuntimeError(
                    "<|action_start|> was not registered as a special token. "
                    "Did you call register_structural_tokens() in build_uamvla_backbone?"
                )
        except (TypeError, ValueError):
            # Mock tokenizer in unit tests — convert_tokens_to_ids returned a non-int.
            _action_start_id = -1
        self.action_start_id = _action_start_id
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_uamvla_framework_init.py -v`
Expected: PASS for all tests, including the new `test_uamvla_framework_exposes_action_start_id`.

- [ ] **Step 6: Commit**

```bash
git add starVLA/model/framework/VLM4A/UamVLA.py tests/test_uamvla_framework_init.py
git commit -m "[uamvla] Expose action_start_id with hard-error on unk collision"
```

---

## Task 5: `ActionLogitsProcessor` — package + class + tests

**Files:**
- Create: `starVLA/model/modules/uamvla/inference/__init__.py`
- Create: `starVLA/model/modules/uamvla/inference/action_logits_processor.py`
- Create: `tests/test_action_logits_processor.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_action_logits_processor.py`:

```python
"""Tests for ActionLogitsProcessor — generate-time logits constraint for action chunks."""
import pytest
import torch


def _make_processor(action_start_id=42, action_begin_id=100, n_bins=4, action_chunk_len=3):
    """Helper: small, debuggable parameter set."""
    from starVLA.model.modules.uamvla.inference.action_logits_processor import (
        ActionLogitsProcessor,
    )
    return ActionLogitsProcessor(
        action_start_id=action_start_id,
        action_begin_id=action_begin_id,
        n_bins=n_bins,
        action_chunk_len=action_chunk_len,
    )


def _scores(B, vocab=200):
    """Helper: deterministic scores for inspection."""
    return torch.arange(B * vocab, dtype=torch.float32).view(B, vocab)


def test_processor_inactive_before_action_start():
    """Without seeing action_start_id, scores must pass through unchanged."""
    p = _make_processor()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    scores = _scores(B=1)
    out = p(input_ids, scores.clone())
    assert torch.equal(out, scores), "scores were modified before action_start was seen"


def test_processor_activates_immediately_after_action_start():
    """The step right after action_start: only [action_begin_id, action_begin_id+n_bins) survives."""
    p = _make_processor(action_begin_id=100, n_bins=4, action_chunk_len=3)
    # Last generated token IS action_start_id (42) → mask the next prediction.
    input_ids = torch.tensor([[1, 2, 42]])
    scores = _scores(B=1)
    out = p(input_ids, scores.clone())

    # Action positions [100..104) untouched
    assert torch.equal(out[0, 100:104], scores[0, 100:104])
    # Everything else is -inf
    mask = torch.ones(scores.shape[1], dtype=torch.bool)
    mask[100:104] = False
    assert torch.isinf(out[0, mask]).all()
    assert (out[0, mask] < 0).all()  # specifically -inf, not +inf


def test_processor_masks_for_exactly_action_chunk_len_steps():
    """Mask is active for action_chunk_len total steps, then automatically deactivates."""
    p = _make_processor(action_begin_id=100, n_bins=4, action_chunk_len=3)
    # Step 0: action_start in input_ids → mask THIS step
    input_ids = torch.tensor([[42]])
    s0 = _scores(B=1)
    out0 = p(input_ids, s0.clone())
    assert torch.isinf(out0[0, 0]), "step 0 should be masked"

    # Step 1: a regular action token in input_ids (not action_start) → should still mask
    input_ids = torch.tensor([[42, 101]])
    s1 = _scores(B=1)
    out1 = p(input_ids, s1.clone())
    assert torch.isinf(out1[0, 0]), "step 1 should still be masked"

    # Step 2: another action token → still mask (3rd and final masked step)
    input_ids = torch.tensor([[42, 101, 102]])
    s2 = _scores(B=1)
    out2 = p(input_ids, s2.clone())
    assert torch.isinf(out2[0, 0]), "step 2 should still be masked (final masked step)"

    # Step 3: action_chunk_len exhausted → mask must be off
    input_ids = torch.tensor([[42, 101, 102, 103]])
    s3 = _scores(B=1)
    out3 = p(input_ids, s3.clone())
    assert torch.equal(out3, s3), "step 3 should be unmasked (action_chunk_len exhausted)"


def test_processor_per_row_independence():
    """Two batch rows enter action mode at different times — masks must be independent."""
    p = _make_processor(action_begin_id=100, n_bins=4, action_chunk_len=2)

    # Row 0 just emitted action_start; row 1 emitted a normal token.
    input_ids = torch.tensor([[1, 2, 42], [1, 2, 9]])
    scores = _scores(B=2)
    out = p(input_ids, scores.clone())

    # Row 0 masked: only [100..104) survives
    assert torch.isinf(out[0, 0])
    assert torch.equal(out[0, 100:104], scores[0, 100:104])
    # Row 1 unchanged
    assert torch.equal(out[1], scores[1])


def test_processor_resets_counter_on_re_entry():
    """A second action_start within the same generate() call resets the counter."""
    p = _make_processor(action_begin_id=100, n_bins=4, action_chunk_len=2)

    # Enter action mode
    p(torch.tensor([[42]]), _scores(1).clone())  # step 0: mask, _remaining→1
    p(torch.tensor([[42, 101]]), _scores(1).clone())  # step 1: mask, _remaining→0

    # Re-enter — last token is action_start again
    out = p(torch.tensor([[42, 101, 42]]), _scores(1).clone())
    # Should be masked again (counter reset to action_chunk_len=2)
    assert torch.isinf(out[0, 0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_action_logits_processor.py -v`
Expected: All tests FAIL with `ModuleNotFoundError: No module named 'starVLA.model.modules.uamvla.inference'`.

- [ ] **Step 3: Create the package marker**

Create `starVLA/model/modules/uamvla/inference/__init__.py`:

```python
```

(Empty file — pure package marker.)

- [ ] **Step 4: Implement `ActionLogitsProcessor`**

Create `starVLA/model/modules/uamvla/inference/action_logits_processor.py`:

```python
"""ActionLogitsProcessor — generate-time logits constraint for action chunks.

Activates on detecting <|action_start|> as the most recent generated token, then
masks all logits outside [action_begin_id, action_begin_id + n_bins) for exactly
action_chunk_len subsequent steps. State is per-batch-row.

Caller must construct a fresh instance per generate() call.
"""
from __future__ import annotations

import torch

try:
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover — transformers is a hard dep at runtime
    LogitsProcessor = object  # type: ignore[assignment]


class ActionLogitsProcessor(LogitsProcessor):
    """Constrains generation to action-bin tokens after <|action_start|>.

    Args:
        action_start_id: Token id of <|action_start|> (sentinel marking action segment start).
        action_begin_id: First action-bin token id (= <ACT_0> id).
        n_bins: Number of action bins (typically 256). Allowed range is
                [action_begin_id, action_begin_id + n_bins).
        action_chunk_len: Number of generation steps to keep masking active.
                          Typically action_horizon * action_dim.
    """

    def __init__(
        self,
        action_start_id: int,
        action_begin_id: int,
        n_bins: int,
        action_chunk_len: int,
    ):
        self.action_start_id = int(action_start_id)
        self.action_begin_id = int(action_begin_id)
        self.action_end_id = int(action_begin_id) + int(n_bins)  # exclusive
        self.action_chunk_len = int(action_chunk_len)
        # Lazily allocated on first __call__ once we know batch size + device.
        self._remaining: torch.Tensor | None = None

    def __call__(
        self,
        input_ids: torch.LongTensor,    # (B, S_so_far)
        scores: torch.FloatTensor,      # (B, vocab)
    ) -> torch.FloatTensor:
        B = scores.shape[0]
        device = scores.device

        # Lazy-init / re-init if batch size changed across calls.
        if self._remaining is None or self._remaining.shape[0] != B:
            self._remaining = torch.zeros(B, dtype=torch.long, device=device)
        elif self._remaining.device != device:
            self._remaining = self._remaining.to(device)

        # Detect entry: did the last-generated token equal action_start_id?
        last_tokens = input_ids[:, -1]
        just_entered = (last_tokens == self.action_start_id)
        # Set/reset remaining for rows that just entered (overwrites prior state
        # on re-entry, matching the documented "counter reset" semantics).
        self._remaining = torch.where(
            just_entered,
            torch.full_like(self._remaining, self.action_chunk_len),
            self._remaining,
        )

        active = self._remaining > 0
        if active.any():
            # Build a (vocab,) mask: 0 inside action range, -inf elsewhere.
            mask = torch.full(
                (scores.shape[1],), float("-inf"), dtype=scores.dtype, device=device
            )
            mask[self.action_begin_id : self.action_end_id] = 0.0
            # Apply per-row: scores + mask for active rows, scores unchanged for inactive.
            scores = torch.where(
                active.unsqueeze(1),
                scores + mask.unsqueeze(0),
                scores,
            )
            # Decrement only the active rows.
            self._remaining = torch.where(
                active,
                self._remaining - 1,
                self._remaining,
            )

        return scores
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_action_logits_processor.py -v`
Expected: PASS for all five tests.

- [ ] **Step 6: Commit**

```bash
git add starVLA/model/modules/uamvla/inference/__init__.py \
        starVLA/model/modules/uamvla/inference/action_logits_processor.py \
        tests/test_action_logits_processor.py
git commit -m "[inference] Add ActionLogitsProcessor — per-row mask after <|action_start|>"
```

---

## Task 6: Integration smoke test

**Files:**
- Create: `tests/test_action_start_integration.py`

This task verifies that the parts integrate correctly — registration at backbone build → token visible in tokenizer → chat template emits the new token → it tokenizes to a single id.

- [ ] **Step 1: Check whether the existing UamVLA forward smoke test runs locally**

Run: `pytest tests/test_uamvla_forward.py -v --collect-only`
Expected: tests collect without error.

If collection requires a Qwen3-VL checkpoint that isn't present, this integration test will be skipped automatically by the same fixture.

- [ ] **Step 2: Write the integration test**

Create `tests/test_action_start_integration.py`:

```python
"""Integration smoke: register_structural_tokens → chat template → tokenizer round-trip."""
import pytest


def test_action_start_token_round_trips_through_real_tokenizer():
    """After register_structural_tokens, the chat template's emitted <|action_start|>
    must tokenize to a single token id (not a multi-subtoken BPE split)."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    from starVLA.model.modules.uamvla.state_encoder.special_tokens import (
        register_structural_tokens,
    )
    from starVLA.model.modules.uamvla.data.chat_template import (
        Qwen3VLChatTemplate, ACTION_START_TOKEN,
    )

    # Use a small public Qwen tokenizer for the smoke test. If unavailable
    # offline, skip — this test is best-effort.
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except Exception as e:
        pytest.skip(f"Qwen tokenizer not available offline: {e}")

    class _StubBackbone:
        def __init__(self, tk):
            self._tk = tk
        def resize_token_embeddings(self, n):
            pass  # not exercising the model path here
        def get_input_embeddings(self):
            class _E:
                class weight:
                    pass
            e = _E()
            e.weight.shape = (len(self._tk),)
            return e

    register_structural_tokens(tok, _StubBackbone(tok))

    # Chat template renders the prompt; tokenize it and verify the action-start
    # token appears as exactly one id in the output.
    prompt = Qwen3VLChatTemplate().format(
        instruction="pick up the red block",
        embodiment="franka_libero",
        view_names=["primary"],
    )
    assert ACTION_START_TOKEN in prompt

    encoded = tok.encode(prompt, add_special_tokens=False)
    action_start_id = tok.convert_tokens_to_ids(ACTION_START_TOKEN)

    assert action_start_id != tok.unk_token_id
    assert encoded.count(action_start_id) == 1, (
        f"<|action_start|> was not tokenized as a single id "
        f"(found {encoded.count(action_start_id)} occurrences of id {action_start_id})"
    )


def test_no_old_action_start_literal_remains_in_codebase():
    """Belt-and-suspenders: assert no Python file under starVLA/ or tests/
    still references the old literal '<action_start>'."""
    import subprocess
    result = subprocess.run(
        ["grep", "-rn", "<action_start>", "starVLA/", "tests/", "--include=*.py"],
        capture_output=True, text=True,
    )
    # grep returns 1 when no matches found — that's what we want.
    assert result.returncode == 1, (
        f"Old literal '<action_start>' still present:\n{result.stdout}"
    )
```

- [ ] **Step 3: Run the integration test**

Run: `pytest tests/test_action_start_integration.py -v`
Expected:
- `test_action_start_token_round_trips_through_real_tokenizer`: PASS, or SKIP if no Qwen tokenizer is available offline.
- `test_no_old_action_start_literal_remains_in_codebase`: PASS.

If the round-trip test FAILS with "tokenized as N occurrences", investigate: the BPE may have residual rules that swallow `<|action_start|>`. Workaround: register via `add_tokens(..., special_tokens=True)` instead of `add_special_tokens`. (Should not be necessary for Qwen, but worth knowing.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_action_start_integration.py
git commit -m "[test] Integration smoke for <|action_start|> single-token round-trip"
```

---

## Task 7: Final verification

**Files:** none modified — verification-only task.

- [ ] **Step 1: Run the full new-test suite**

Run:
```bash
pytest tests/test_structural_tokens.py \
       tests/test_chat_template.py \
       tests/test_action_logits_processor.py \
       tests/test_uamvla_framework_init.py \
       tests/test_action_start_integration.py \
       -v
```
Expected: all PASS (integration test may SKIP if Qwen tokenizer not available offline).

- [ ] **Step 2: Run the full repo test suite for regressions**

Run: `pytest tests/ -v --timeout=120`
Expected: no NEW failures vs baseline. Pre-existing failures (if any) should be unchanged.

If a previously-passing test now fails, investigate immediately — most likely culprits:
- A test that hardcoded the old `<action_start>` literal expectation
- A test that asserted a specific vocab size that's now off by 1 due to the new structural token

- [ ] **Step 3: Verify no stray TODO / FIXME introduced**

Run: `git diff main..HEAD -- '*.py' | grep -E '^\+' | grep -E 'TODO|FIXME|XXX'`
Expected: no output.

- [ ] **Step 4: Final review checklist**

Confirm by inspection:
- [ ] `<action_start>` literal appears nowhere in the codebase (`grep -rn '<action_start>' starVLA/ tests/ --include='*.py'` returns nothing).
- [ ] `<|action_end|>` is NOT registered anywhere (`grep -rn 'action_end' starVLA/ --include='*.py'` returns nothing related to a token registration).
- [ ] `UamVLA.action_start_id` is an integer property accessible after framework init.
- [ ] `ActionLogitsProcessor` is importable from `starVLA.model.modules.uamvla.inference`.

- [ ] **Step 5: Commit any final fixes (if needed)**

If any regression fixes were required, commit them with a clear scope:

```bash
git add <files>
git commit -m "[<scope>] Fix regression caused by <|action_start|> registration"
```

---

## Self-Review

**1. Spec coverage:**
- §2 Goal: register `<|action_start|>` as single special token → Task 1 + 2.
- §2 Goal: replace literal in chat templates → Task 3.
- §2 Goal: expose `action_start_id` on framework → Task 4.
- §2 Goal: do NOT introduce `<|action_end|>` → enforced by Task 7 Step 4 checklist.
- §2 Goal: `ActionLogitsProcessor` → Task 5.
- §2 Goal: training tokenization self-consistent → covered by Task 6 round-trip integration test.
- §6 Testing Plan §6.1: unit tests for special_tokens → Task 1; for chat_template → Task 3; for ActionLogitsProcessor → Task 5.
- §6 Testing Plan §6.2: integration smoke → Task 6.
- §6 Testing Plan §6.3: backward compat (no `<action_start>` literal, no `<|action_end|>`) → Task 6 + Task 7 Step 4.
- §7 Migration: no checkpoint compat work — captured (no migration tasks needed).

**2. Placeholder scan:** No "TBD" / "TODO" / "implement later" in the plan. The one place that delegates judgment to the engineer ("adapt to existing test setup pattern" in Task 4 Step 2) is a justified hand-off (the existing test file's structure is the source of truth) and includes explicit guidance ("copy that exact construction code into a new helper").

**3. Type consistency:**
- `register_structural_tokens(tokenizer, backbone) -> dict[str, int]` — defined in Task 1, signature unchanged across Tasks 2 and 6.
- `STRUCTURAL_SPECIAL_TOKENS: list[str]` — defined in Task 1, referenced in Task 1 test.
- `ACTION_START_TOKEN = "<|action_start|>"` — defined in Task 3, referenced in Tasks 3, 6.
- `UamVLA.action_start_id: int` — defined in Task 4, referenced in Task 4 test.
- `ActionLogitsProcessor(action_start_id, action_begin_id, n_bins, action_chunk_len)` — signature consistent across Task 5 implementation, tests, and docstring.
- `action_chunk_len = action_horizon * action_dim` — used as the masking duration; derivation is the caller's responsibility (this plan does not wire it up to a `generate()` call, that's a follow-up tied to `predict_action`).

No type/naming inconsistencies found.
