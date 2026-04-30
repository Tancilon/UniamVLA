# `<|action_start|>` Special Token Registration — Design Spec

**Date:** 2026-04-30
**Author:** UniamVLA migration team
**Status:** Draft, awaiting user review
**References:**
- OpenVLA action tokenizer (`prismatic/vla/action_tokenizer.py`) — vocab-tail overwrite, no sentinel
- OpenVLA-OFT (`prismatic/extern/hf/modeling_prismatic.py`, `prismatic/training/train_utils.py`) — single `<ACT>` special token, label mask via `token_ids > ACTION_TOKEN_BEGIN_IDX`
- π0-FAST (Physical Intelligence) — vocab-tail + EOS for variable-length action chunks
- HF transformers issue #2155, vLLM issue #17468 — known BPE-splits-special-looking-literals failures

## 1. Problem

UniamVLA's chat template ([`chat_template.py:100,137`](starVLA/model/modules/uamvla/data/chat_template.py#L100)) hardcodes a literal string `<action_start>` as the marker between the assistant prelude (`<think>\n\n</think>\n\n`) and the action token chunk:

```
<|im_start|>assistant
<think>

</think>

<action_start>
```

`<action_start>` is **not registered** with the tokenizer. The Qwen3-VL BPE will split it into ~3–5 sub-tokens (e.g. `<`, `action`, `_start`, `>`), depending on the merges in the base vocab. This is a working-but-fragile design that becomes untenable once we move to `model.generate()`-based autoregressive inference (planned for the L5 evaluation path; currently `_extract_action_tokens_for_sample` is `NotImplementedError`).

### Concrete failure modes the current design enables

1. **Logits masking has no clean trigger.** A generate-time `LogitsProcessor` that constrains action-segment logits to the 256 `<ACT_*>` ids needs to detect "the model just emitted the segment-start marker" in O(1). With multi-token BPE split, it has to do sliding-window matching on the last ~5 generated ids — fragile and prone to drift if any character upstream changes.
2. **KV-cache prefix instability.** Adding/removing a single character around `<action_start>` (e.g. `\n\n<action_start>` vs `\n<action_start>`) can change its BPE segmentation, which shifts position ids and invalidates KV-cache reuse across calls.
3. **Streaming inference is awkward.** Real-time control loops can't reliably commit to "we just entered the action segment" until all sub-tokens have been generated, lagging the masking decision.
4. **Beam search / batched sampling drift.** Any path that detokenizes-then-retokenizes (some HF post-processing flows) can permute the BPE split across beams or batch positions.
5. **Decoding hygiene.** `tokenizer.decode(..., skip_special_tokens=True)` cannot strip `<action_start>` because it's not a special token. Logs will contain the raw string.

### Why now

- We're committing to generate-based inference (KV cache + logits masking + possibly streaming).
- We have **no checkpoint baggage** — the project is pre-large-scale-training, so the ~5 sub-token shift in prompt tokenization caused by the migration is acceptable.
- Industry survey (see References) shows **no public VLA project uses an unregistered BPE literal as an action sentinel**. OpenVLA / RT-2 / π0-FAST use no sentinel at all (relying on fixed length); OpenVLA-OFT uses a single `<ACT>` special token. We are the outlier with the most fragile design.

## 2. Goals

- Register `<|action_start|>` as a single special token in the Qwen3-VL tokenizer; resize embedding table to match.
- Replace the literal `<action_start>` in `Qwen3VLChatTemplate` and `Qwen3ChatTemplate` with `<|action_start|>` (Qwen-style `<|...|>` convention, matching `<|im_start|>` / `<|vision_start|>`).
- Expose `action_start_id` on the `UamVLA` framework alongside the existing `_act0_id` (`action_token_begin_id`), so downstream inference code has a clean handle.
- Do **not** introduce a paired `<|action_end|>`. Action chunk length is fixed (`action_horizon * action_dim`); end markers add cost without value (matches OpenVLA / OpenVLA-OFT / RT-2).
- Provide a reusable `ActionLogitsProcessor` (HF `LogitsProcessor` subclass) that, on detecting `<|action_start|>` as the most recent generated token, masks all logits outside `[action_token_begin_id, action_token_begin_id + n_bins)` for exactly `action_horizon * action_dim` subsequent steps.
- Land the changes such that training prompt tokenization is fully self-consistent — every training sample contains `<|action_start|>` as a single token, and label masks remain correct.

## 3. Non-Goals

- **`<|action_end|>` registration.** Fixed-length chunks; out of scope. Revisit only if we adopt FAST-style variable-length action compression.
- **Implementing `predict_action` end-to-end.** This spec wires up the *primitive* (special token + logits processor); full L5 eval integration depends on the dataloader plugin (Task 24) and is a separate effort.
- **Old checkpoint compatibility.** None exist; retraining is required (and is acceptable per user confirmation).
- **Renaming or restructuring `register_state_tokens`.** We extend the same module with a sibling registration function, not a refactor.
- **Changing how training labels are computed.** `labels != -100` mask continues to be the source of truth for action positions during training. `<|action_start|>` is **not** used as a position marker in loss code — it's purely an inference-side sentinel.
- **Switching action token registration from `add_tokens` to vocab-tail overwrite.** Out of scope; orthogonal decision.

## 4. Design

### 4.1 Token name choice — `<|action_start|>`

Picked over alternatives:

| Candidate | Verdict |
|---|---|
| `<action_start>` | Current literal. Reject — that's what we're fixing. |
| `<ACT_START>` | Visually clashes with `<ACT_0>` ~ `<ACT_255>` (looks like a 257th bin). Reject. |
| `<\|action_start\|>` | **Pick.** Matches Qwen convention (`<\|im_start\|>`, `<\|vision_start\|>`, `<\|image_pad\|>`). Self-documents as a structural marker. Some Qwen tooling treats this pattern as special by default. |
| `<\|act_start\|>` | Acceptable but inconsistent with our registry naming (`<\|state_*\|>`, full words). Reject for naming hygiene. |

### 4.2 Registration mechanism

Extend [`starVLA/model/modules/uamvla/state_encoder/special_tokens.py`](starVLA/model/modules/uamvla/state_encoder/special_tokens.py) — same module that already registers state tokens at backbone-build time. State tokens and `<|action_start|>` are both **structural / bootstrap-time** tokens (always present, embodiment-independent, registered before any training step), so co-locating their registration is a natural fit.

```python
# In special_tokens.py — new constant
STRUCTURAL_SPECIAL_TOKENS: list[str] = [
    "<|action_start|>",
]

# Extend register_state_tokens to also register structural tokens, OR add
# a sibling function register_structural_tokens(tokenizer, backbone) that
# follows the exact same pattern. Implementation chooses whichever yields
# cleaner call sites in build_uamvla_backbone.
```

The choice between "extend existing function" vs "add sibling function" is left to the implementation plan; both are equivalent in mechanism. The recommended choice is **add a sibling function** `register_structural_tokens(tokenizer, backbone)` so the `state_encoder/` module's name doesn't become misleading.

`register_structural_tokens` calls:

```python
n_added = tokenizer.add_special_tokens(
    {"additional_special_tokens": STRUCTURAL_SPECIAL_TOKENS}
)
if n_added > 0:
    backbone.resize_token_embeddings(len(tokenizer))
```

Idempotent (matching `register_state_tokens` semantics). Returns `dict[str, int]` mapping each registered token to its assigned id.

### 4.3 Backbone factory wiring

In [`backbone_wrapper.py:504-505`](starVLA/model/modules/uamvla/backbone_wrapper.py#L504), after `register_state_tokens(...)`:

```python
register_state_tokens(tokenizer, shim)
register_structural_tokens(tokenizer, shim)   # new
```

Registration order (state tokens → structural tokens, both inside `build_uamvla_backbone`; then later `ActionTokenizer.add_tokens([...])` inside `UamVLA.__init__`) is the natural fit: structural and state tokens are both bootstrap-time and pin to the lower id range; per-bin action tokens get the highest ids. Correctness does **not** depend on this exact order — `_act0_id` is resolved by `convert_tokens_to_ids("<ACT_0>")` after all registration is done, so id assignments self-consistently work regardless of order. What does matter: **all** registrations must happen before either training or inference loads its checkpoint, and the same registration sequence must run in both contexts (training and inference share the same `build_uamvla_backbone` factory, so this is automatically satisfied).

### 4.4 Chat template update

[`chat_template.py:100`](starVLA/model/modules/uamvla/data/chat_template.py#L100) and [`chat_template.py:137`](starVLA/model/modules/uamvla/data/chat_template.py#L137) — replace the literal:

```diff
-    f"<|im_start|>assistant\n<think>\n\n</think>\n\n<action_start>"
+    f"<|im_start|>assistant\n<think>\n\n</think>\n\n<|action_start|>"
```

Both `Qwen3ChatTemplate` (SigLIP path, currently unused but kept for compat) and `Qwen3VLChatTemplate` (active path) get the same edit. No structural change to the template otherwise.

Add an optional class-level constant `ACTION_START_TOKEN = "<|action_start|>"` on the base `ChatTemplate` so downstream code can reference it symbolically rather than hardcoding the string.

**Implementation note (post-implementation amendment):** the `_VLA_SYSTEM_PROMPT` originally ended with the descriptive clause `"...action tokens that begin with <action_start>. Do not output natural language."` This clause was removed during implementation. Reason: keeping the literal in the system prompt — once `<|action_start|>` becomes a single special token — would either create dual occurrences of the token in the prompt (encouraging the model to imitate system instructions in its own output) or require relaxing the chat-template test's `count == 1` contract. The descriptive mention was informational, not load-bearing — the model learns marker position from training labels and from the assistant turn's `<|action_start|>`. Cleaner final clause: `"...action tokens. Do not output natural language."`

### 4.5 Framework-level wiring (`UamVLA.py`)

In `UamVLA.__init__` ([`UamVLA.py:108-112`](starVLA/model/framework/VLM4A/UamVLA.py#L108-L112)), alongside `_act0_id`:

```python
try:
    _action_start_id = int(_tokenizer.convert_tokens_to_ids("<|action_start|>"))
    # Sanity: HF tokenizers return either unk_token_id (fast) or None (some slow
    # variants) when a token isn't registered. Both must be caught.
    _unk_id = getattr(_tokenizer, "unk_token_id", None)
    if _action_start_id is None or (_unk_id is not None and _action_start_id == _unk_id):
        raise RuntimeError(
            "<|action_start|> was not registered as a special token. "
            "Did you call register_structural_tokens() in build_uamvla_backbone?"
        )
except (TypeError, ValueError):
    _action_start_id = -1   # mock/test fallback (MagicMock tokenizer in unit tests)
self.action_start_id = _action_start_id
```

The `RuntimeError` is intentionally raised inside the `try` block; only `TypeError`/`ValueError` (from `int()` on a non-numeric mock return value) are caught. A real "token not registered" misconfiguration in production code therefore surfaces loudly at framework construction, not silently via id-collision-with-`<unk>` at training time.

### 4.6 `ActionLogitsProcessor` — new module

New file: [`starVLA/model/modules/uamvla/inference/action_logits_processor.py`](starVLA/model/modules/uamvla/inference/action_logits_processor.py)

```python
import torch
from transformers import LogitsProcessor

class ActionLogitsProcessor(LogitsProcessor):
    """Constrains generation to action tokens after <|action_start|>.

    Behavior:
      - Watches for action_start_id in the most recently generated token.
      - On detection, enters "action mode" for action_chunk_len steps:
        masks all logits outside [action_begin_id, action_begin_id + n_bins)
        to -inf.
      - After action_chunk_len steps in action mode, automatically exits.
      - State is maintained per batch row independently (different rows may
        enter action mode at different positions).
    """

    def __init__(
        self,
        action_start_id: int,
        action_begin_id: int,
        n_bins: int,
        action_chunk_len: int,   # = action_horizon * action_dim
    ):
        self.action_start_id = action_start_id
        self.action_begin_id = action_begin_id
        self.action_end_id = action_begin_id + n_bins   # exclusive
        self.action_chunk_len = action_chunk_len
        # Per-row remaining count; lazily allocated on first __call__
        self._remaining: torch.Tensor | None = None

    def __call__(
        self,
        input_ids: torch.LongTensor,    # (B, S_so_far)
        scores: torch.FloatTensor,      # (B, vocab)
    ) -> torch.FloatTensor:
        B, vocab = scores.shape
        if self._remaining is None or self._remaining.shape[0] != B:
            self._remaining = torch.zeros(B, dtype=torch.long, device=scores.device)

        # Detect entry: did the last-generated token equal action_start_id?
        last_tokens = input_ids[:, -1]
        just_entered = (last_tokens == self.action_start_id)
        # Set remaining = action_chunk_len for rows that just entered.
        # (Overwrites whatever was there — re-entry resets the counter.)
        self._remaining[just_entered] = self.action_chunk_len

        # Apply mask to active rows
        active = self._remaining > 0
        if active.any():
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.action_begin_id : self.action_end_id] = 0.0
            scores = torch.where(active.unsqueeze(1), scores + mask, scores)
            self._remaining[active] -= 1

        return scores
```

Design decisions:
- **Per-row state** — different batch rows may have arrived at action segment at different positions (e.g. variable instruction length). Mask per-row, not per-batch.
- **Counter resets on re-entry** — if `<|action_start|>` appears twice (shouldn't happen with correct training but possible with badly-trained ckpts), the second occurrence resets the counter rather than nesting. Simpler and safer than tracking nested entries.
- **Stateless across `generate()` calls** — `_remaining` is per-instance, so a new `ActionLogitsProcessor` should be constructed per `generate()` call (cheap, ~3 ints + a small tensor). Documented in the docstring.
- **No EOS injection** — the processor only constrains *which* token can be emitted, not when generation stops. Caller continues to use `max_new_tokens = action_chunk_len + epsilon` to bound generation.

### 4.7 What does NOT change

- `ActionTokenizer` — unchanged. `<ACT_*>` registration stays as-is.
- `ActionHead.compute_loss` — unchanged. Continues to use `labels != -100` mask.
- `_replace_state_tokens` — unchanged. Operates on `<|state_*|>` only; `<|action_start|>` is downstream of state token splice.
- `RobotDataCollator` — unchanged in interface. Implicitly affected: prompt is ~4 tokens shorter per sample (BPE subtokens collapse to single token), but the collator builds labels based on action-position offsets, not absolute positions.
- `_extract_action_tokens_for_sample` — still `NotImplementedError`. This spec doesn't unblock it; it just provides the primitive `action_start_id` it would later use.

## 5. Concrete File Changes

| File | Change Type | Lines |
|---|---|---|
| `starVLA/model/modules/uamvla/state_encoder/special_tokens.py` | Add `STRUCTURAL_SPECIAL_TOKENS` constant + `register_structural_tokens()` function | New code, ~30 lines |
| `starVLA/model/modules/uamvla/backbone_wrapper.py` | Call `register_structural_tokens(tokenizer, shim)` after `register_state_tokens` | +1 line at L505 |
| `starVLA/model/modules/uamvla/data/chat_template.py` | Replace `<action_start>` → `<\|action_start\|>` in two format() methods; add `ACTION_START_TOKEN` constant | 2 line edits + 1 constant |
| `starVLA/model/framework/VLM4A/UamVLA.py` | Resolve `action_start_id` in `__init__`; expose as `self.action_start_id`; sanity-check vs unk | +6 lines after L112 |
| `starVLA/model/modules/uamvla/inference/__init__.py` | New package | New file |
| `starVLA/model/modules/uamvla/inference/action_logits_processor.py` | New module per §4.6 | New file, ~50 lines |

Rough total: ~90 lines added, 2 lines edited.

## 6. Testing Plan

### 6.1 Unit tests

`tests/uamvla/state_encoder/test_special_tokens.py`:
- Test that `register_structural_tokens` adds `<|action_start|>` and resizes embeddings.
- Test idempotency (calling twice is a no-op).
- Test that the assigned id is positive and != `unk_token_id`.

`tests/uamvla/data/test_chat_template.py`:
- Test that both `Qwen3VLChatTemplate` and `Qwen3ChatTemplate` emit `<|action_start|>` exactly once per sample.
- Round-trip test: tokenizer.encode(format_output) → decode → assert `<|action_start|>` reappears as a single special token (using `convert_ids_to_tokens` to inspect).

`tests/uamvla/inference/test_action_logits_processor.py`:
- Test that masking is OFF before any `<|action_start|>` is generated (scores unchanged).
- Test that masking activates immediately on the step *after* `<|action_start|>` is the last token, and lasts exactly `action_chunk_len` steps.
- Test that masking sets non-action logits to `-inf` and leaves action logits unchanged.
- Test per-row independence: B=2, row 0 enters action mode at step 5, row 1 at step 7; verify each masks for the correct window.
- Test re-entry resets the counter.

### 6.2 Integration smoke test

End-to-end mini-batch through `UamVLA.forward`:
- Build `UamVLA` from default config (which now includes `<|action_start|>` registration).
- Confirm `self.action_start_id > 0` and `!= tokenizer.unk_token_id`.
- Confirm tokenizer encodes the chat-template output such that `<|action_start|>` corresponds to a single token (use `convert_ids_to_tokens(input_ids[0])` and assert presence).
- Confirm a forward pass completes without shape mismatches (state token splice still works).

### 6.3 Backward compatibility check

- Confirm no test or code path references the old literal `<action_start>` (grep + assertion).
- Confirm `<|action_end|>` is *not* introduced (negative test: `assert "<|action_end|>" not in tokenizer.get_vocab()`).

## 7. Migration

No checkpoints exist; no migration logic required beyond:
- Bumping any cached preprocessed dataset that stored tokenized prompts (none currently — `LiberoPreprocessor` stores raw fields, tokenization happens in collator).
- A note in CHANGELOG / commit message that any pre-existing in-flight training run must be restarted.

## 8. Open Questions

None requiring user input. Implementation-level decisions deferred to the plan:
- Whether `register_structural_tokens` is a sibling function or an extension of `register_state_tokens` (recommendation: sibling, see §4.2).
- Whether `ActionLogitsProcessor` lives in `inference/` or a peer dir (recommendation: `inference/`, since it's solely for `generate()` time).

## 9. Effort Estimate

- Token registration + chat template + framework wiring: ~30 min
- `ActionLogitsProcessor` + tests: ~45 min
- Integration smoke test + spec compliance review: ~20 min
- Total: **~1.5 hours** of focused implementation, plus retraining (separate).
