# UamVLA `predict_action` Autoregressive Inference — Design Spec

**Date:** 2026-04-30
**Author:** UniamVLA migration team
**Status:** Draft, awaiting user review
**References:**
- starVLA framework contract: [`base_framework.py:119-131`](starVLA/model/framework/base_framework.py#L119-L131)
- Reference implementation: [`QwenFast.py:181-225`](starVLA/model/framework/VLM4A/QwenFast.py#L181-L225) (autoregressive token-generation VLA, in-tree)
- Prior spec (dependency): [`2026-04-30-action-start-special-token-design.md`](2026-04-30-action-start-special-token-design.md)
- OpenVLA / OpenVLA-OFT / RT-2 / π0-FAST / VLA-0 — full-vocab CE + ID-range scan decoding (industry standard for AR-VLAs)

## 1. Problem

The current `UamVLA.predict_action` ([`UamVLA.py:340-381`](starVLA/model/framework/VLM4A/UamVLA.py#L340-L381)) is a **placeholder implementation**, not a real autoregressive inference path:

- It runs the backbone forward **exactly once** with a prompt that ends at `<|action_start|>` — no action tokens are present in `input_ids`.
- It calls `ActionHead.predict(hidden_states)` which applies `lm_head` over the prompt sequence and argmax-decodes "the next token at each prompt position." This is a training-time auxiliary signal, not action generation.
- The decoder `_extract_action_tokens_for_sample` raises `NotImplementedError` ([UamVLA.py:407-413](starVLA/model/framework/VLM4A/UamVLA.py#L407-L413)) — it has nothing meaningful to extract because no autoregressive generation occurred.

What's missing for a real AR-VLA inference loop:
1. Autoregressive generation of `H × action_dim` action tokens (one at a time, KV cache reused).
2. Each step appends the just-sampled token back to the input for the next step.
3. State-token splice (`canonical_state` → `<|state_*|>` embedding injection) integrated into the inference path.
4. Decoding the generated action tokens into normalized continuous actions.

The previous Action-Start Special Token spec laid all the infrastructure: `<|action_start|>` is a single registered special token, `self.action_start_id` is exposed on the framework, `ActionLogitsProcessor` is implemented and tested. **What remains is wiring these into a working `predict_action`**.

### Why now

- The starVLA framework's `predict_action` contract is well-established. Three callers depend on it: training-time `eval_action_model` ([`train_starvla.py:397`](starVLA/training/train_starvla.py#L397)), LIBERO benchmark ([`eval_libero_model.py:227`](examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py#L227)), and the websocket deployment server ([`websocket_policy_server.py:117`](deployment/model_server/tools/websocket_policy_server.py#L117)).
- The in-tree `QwenFast.py` provides a fully working reference implementation of `model.generate()`-based autoregressive action token generation. It demonstrates the framework-level wiring; UamVLA needs the same wiring **plus** state-token splice handling.
- Industry survey (OpenVLA / RT-2 / π0-FAST / VLA-0 / OpenVLA-OFT) confirms the standard pattern: full-vocab `model.generate()` + ID-range scan over generated tokens for action extraction. No public AR-VLA uses logits masking during inference.

## 2. Goals

- Replace the placeholder `predict_action` with a real autoregressive inference path:
  - Build prefill `inputs_embeds` that include `state_encoder` output spliced into `<|state_*|>` positions (training-equivalent).
  - Call HuggingFace `model.generate(inputs_embeds=..., max_new_tokens=H × action_dim, do_sample=False)`.
  - Extract action tokens from the generated tail via ID-range scan, decode via `ActionTokenizer`.
- Preserve the `predict_action` interface contract: `(examples, **kwargs) -> dict[{"normalized_actions": np.ndarray (B, H, action_dim)}]`. **All existing callers see zero change.**
- Support batched inference (B ≥ 1) from day one — required for parallel-env LIBERO evaluation.
- Land a training-vs-inference numerical alignment test that asserts the prefix hidden state at `<|action_start|>` is identical between the training forward path and the inference prefill path. This is the safety net for the entire approach.
- Mirror the QwenFast implementation pattern where applicable; deviate only where state splice or other UamVLA-specific concerns require it.

## 3. Non-Goals

- **Real-checkpoint LIBERO end-to-end runs.** That's an L5 remote-GPU task tracked separately; this spec only goes as far as mock-backbone smoke tests + the training-vs-inference alignment test.
- **Performance tuning.** No KV-cache micro-optimization, no batched padding optimization, no quantization. Use HuggingFace `generate()` defaults.
- **`ActionLogitsProcessor` integration.** It stays available as an opt-in tool but is **not** wired into the default inference path. Industry standard for AR-VLAs is unconstrained `generate()`.
- **`train_starvla.py` modifications.** The training-time `eval_action_model` will benefit automatically without code change there.
- **Old checkpoint compatibility.** No checkpoints exist that would conflict; the framework forward path is unchanged so existing training behavior is preserved.
- **Sampling strategies (top-k / top-p / temperature).** Default to greedy (`do_sample=False`); add later if needed.
- **Streaming inference.** Out of scope; `generate()` returns the full chunk in one call.
- **Variable-length action chunks.** Fixed `H × action_dim` per call; matches OpenVLA / OpenVLA-OFT.

## 4. Design

### 4.1 Architecture overview

```
predict_action(examples, **kwargs)
├── Step 1: build_inputs (UNCHANGED, reuses training pipeline)
│       images + instructions + canonical_state
│           ↓
│       qwen_inputs = {input_ids, attention_mask, pixel_values,
│                       image_grid_thw, canonical_state}
│
├── Step 2: Prefill — build inputs_embeds with state splice
│       inputs_embeds = embed_tokens(input_ids)
│       state_embeds = state_encoder(canonical_state)
│       inputs_embeds = _replace_state_tokens(inputs_embeds, ...)
│
├── Step 3: HF generate (batched, autoregressive)
│       generated_ids = model.generate(
│           inputs_embeds=inputs_embeds,
│           pixel_values=pixel_values, image_grid_thw=image_grid_thw,
│           attention_mask=attention_mask,
│           max_new_tokens=H * action_dim,
│           do_sample=False, use_cache=True,
│       )
│
├── Step 4: Extract action tokens (ID-range scan on new tokens)
│       new_ids = generated_ids[:, prompt_len:]
│       mask = (new_ids >= _act0_id) & (new_ids < _act0_id + n_bins)
│       per-row: select masked tokens, pad with mid_bin to chunk_len
│
└── Step 5: Decode → normalized actions
        action_tokenizer.decode(action_ids).reshape(H, action_dim)
        → return {"normalized_actions": np.ndarray (B, H, action_dim)}
```

**Key design choices** (all confirmed during brainstorming):
1. **Prefill+generate split (Plan A)** — first compute `inputs_embeds` via the wrapper's state-splice path, then hand off to HuggingFace `generate(inputs_embeds=...)`. Guarantees training-equivalent prefix; HF handles KV cache and incremental decoding internally.
2. **No `ActionLogitsProcessor`** — unconstrained `generate()` matches OpenVLA / RT-2 / π0-FAST / VLA-0. ID-range scan on the generated tail handles any noise tokens defensively.
3. **ID-range scan on `generated_ids[:, prompt_len:]`** — same technique as QwenFast and OpenVLA-OFT (`token_ids ∈ [_act0_id, _act0_id + n_bins)`). No `<|action_start|>` ID scan needed because prompt termination is structurally guaranteed by chat_template.
4. **Greedy decoding** (`do_sample=False`) — matches OpenVLA / QwenFast; action prediction is deterministic.
5. **Batched B ≥ 1 from day one** — `build_inputs` is already batched; `model.generate` natively supports batched inputs.

### 4.2 What changes vs current code

| Component | Current | New |
|---|---|---|
| `predict_action` body | 1× backbone forward + `ActionHead.predict` + `_decode_action_tokens` (broken) | `_build_prefill_embeds` + `model.generate` + `_decode_generated_actions` |
| `_extract_action_tokens_for_sample` | `raise NotImplementedError` | **Removed** |
| `_decode_action_tokens` | Calls broken sample-extractor | **Replaced** by `_decode_generated_actions` |
| `_build_prefill_embeds` | (does not exist) | **New** method, reuses `_replace_state_tokens` |
| `self._act0_id` | Local variable in `__init__` | **Promoted** to instance attribute |
| `ActionHead.predict` | Used by `predict_action` | **No longer called from inference** (kept; future visualizers may use it) |

### 4.3 `_build_prefill_embeds` — state splice extracted to its own method

```python
def _build_prefill_embeds(self, qwen_inputs: dict) -> torch.Tensor:
    """Prefill: embed input_ids and splice in state-token embeddings.

    Returns inputs_embeds suitable to be passed to model.generate(inputs_embeds=...).
    The state splice is identical to the training forward path (reuses
    _replace_state_tokens from backbone_wrapper), guaranteeing the prefix
    hidden state at <|action_start|> matches training exactly.
    """
    iface = self.qwen_vl_interface
    input_ids = qwen_inputs["input_ids"]
    canonical_state = qwen_inputs.get("canonical_state")

    embed_tokens = iface.get_embed_tokens()
    base_embeds = embed_tokens(input_ids)

    if canonical_state is not None:
        from starVLA.model.modules.uamvla.backbone_wrapper import _replace_state_tokens
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

    return inputs_embeds
```

This method does **not** run any transformer layers. It only prepares embeddings; the transformer runs inside `model.generate(inputs_embeds=...)`.

### 4.4 `predict_action` — main loop

```python
@torch.inference_mode()
def predict_action(self, examples, **kwargs) -> dict:
    if not isinstance(examples, list):
        examples = [examples]

    from deployment.model_server.tools.image_tools import to_pil_preserve

    qwen_inputs = self.qwen_vl_interface.build_inputs(
        images=[[to_pil_preserve(img) for img in e["image"]] for e in examples],
        instructions=[e["lang"] for e in examples],
        canonical_state=stack_canonical([e["canonical_state"] for e in examples])
            if "canonical_state" in examples[0] else None,
    )

    inputs_embeds = self._build_prefill_embeds(qwen_inputs)

    H = self.action_horizon
    action_dim = int(self.config.framework.embodiment.get("action_dim", 7))
    chunk_len = H * action_dim

    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated_ids = self.qwen_vl_interface.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=qwen_inputs["attention_mask"],
            pixel_values=qwen_inputs["pixel_values"],
            image_grid_thw=qwen_inputs["image_grid_thw"],
            max_new_tokens=chunk_len,
            do_sample=False,
            use_cache=True,
        )

    normalized_actions = self._decode_generated_actions(
        generated_ids,
        prompt_len=inputs_embeds.shape[1],
        H=H,
        action_dim=action_dim,
    )
    return {"normalized_actions": normalized_actions}
```

### 4.5 `_decode_generated_actions` — ID-range scan with mid_bin padding

```python
def _decode_generated_actions(
    self,
    generated_ids: torch.Tensor,
    prompt_len: int,
    H: int,
    action_dim: int,
) -> np.ndarray:
    """Extract action tokens from the generated tail and decode to normalized actions.

    Strategy: ID-range scan on tokens generated AFTER the prompt. Tokens outside
    [_act0_id, _act0_id + n_bins) are skipped (defensive against undertrained
    models that may emit non-action tokens). Length is normalized to H * action_dim
    via mid_bin padding (action ≈ 0) on truncation.

    Args:
        generated_ids: (B, S_prompt + N_new) long tensor from model.generate
        prompt_len: number of prompt tokens (= inputs_embeds.shape[1])
        H: action horizon
        action_dim: action dimensionality (typically 7 for Franka)

    Returns:
        np.ndarray of shape (B, H, action_dim), float32, in normalized action space
    """
    import numpy as np

    B = generated_ids.shape[0]
    chunk_len = H * action_dim
    n_bins = int(self.config.framework.action_model.get("num_bins", 256))
    act_min = self._act0_id
    act_max = self._act0_id + n_bins  # exclusive
    mid_id = self._act0_id + n_bins // 2

    new_ids = generated_ids[:, prompt_len:]

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

### 4.6 Instance attribute promotion

```python
# UamVLA.__init__, after _act0_id resolution:
self._act0_id = _act0_id   # promoted from local var (was previously only passed to aux_heads)
```

This is the only change to existing init logic. `_act0_id` is still passed to `_build_aux_heads(action_token_begin_id=_act0_id)` unchanged.

### 4.7 Framework contract — preserved

```python
def predict_action(self, examples: Union[dict, List[dict]], **kwargs) -> dict:
    """Returns {"normalized_actions": np.ndarray of shape (B, H, action_dim)}."""
```

- Signature: `(examples, **kwargs) -> dict` — matches `base_framework.predict_action`.
- Return: dict with `normalized_actions` key, shape `(B, H, action_dim)`, dtype `float32`.
- All `**kwargs` are accepted and ignored (e.g., `train_starvla.py:397` passes `use_ddim=True, num_ddim_steps=20` for diffusion frameworks; UamVLA silently drops these).
- All callers (`eval_libero_model.py`, `train_starvla.py`, `websocket_policy_server.py`) need **zero modification**.

### 4.8 Invariants and failure modes

#### Strong invariants
1. **`generated_ids[:, :prompt_len]` is a valid slice** — HF `generate(inputs_embeds=..., use_cache=True)` returns `(B, S_prompt + N_new)` with prompt-equivalent placeholders for the first `S_prompt` positions; we never index beyond this.
2. **Training `forward` is unchanged** — all training tests stay green; this spec touches only the inference path.
3. **`<|action_start|>` remains the chat_template terminator** — chat_template.py is not modified by this spec.

#### Failure modes and fallbacks

| Failure | Behavior |
|---|---|
| Undertrained model emits non-ACT tokens (e.g. `<\|im_end\|>`) | ID-range scan filters them out; remaining ACT tokens are kept in order |
| Model emits fewer than `H × action_dim` ACT tokens (early stop) | Pad with `mid_id` (action ≈ 0) up to `chunk_len`. Surface as a logged warning in future iterations. |
| Model emits more than `H × action_dim` ACT tokens (rare; `max_new_tokens` should prevent this) | Truncate to `chunk_len` |
| Example dict lacks `canonical_state` | `_build_prefill_embeds` skips state splice and returns plain `embed_tokens(input_ids)`. Matches the no-state forward path. |
| Mock tokenizer in unit tests | `_decode_generated_actions` works for any `_act0_id ≥ 0`; tests construct mock `generate` outputs with IDs in the expected range. |
| Batched inputs with different prompt lengths | HuggingFace processor right-pads; `inputs_embeds.shape[1]` is the max length; HF `generate` respects `attention_mask`. |

### 4.9 Performance and resource budget

- **Peak memory**: comparable to training forward (one prefill pass) minus backward; should be lower than training step.
- **Latency** (estimate, Qwen3-VL-8B + B=8 + H=8 + action_dim=7 = 56 new tokens): prefill ~2-4s + decode ~0.5-1s ≈ 3-5s/batch. To be validated at L5.
- **KV cache**: HF `generate(use_cache=True)` manages it automatically; each new token reuses the prefix KV.

### 4.10 Compatibility

- **transformers**: requires `>= 4.57.0` (already enforced by [`backbone_wrapper.py:45`](starVLA/model/modules/uamvla/backbone_wrapper.py#L45)). `generate(inputs_embeds=...)` has been stable since 4.40+.
- **DeepSpeed / Accelerate**: callers already do `accelerator.unwrap_model(...)` before invoking `predict_action`; no change needed.

## 5. Concrete File Changes

| File | Change Type | Lines |
|---|---|---|
| `starVLA/model/framework/VLM4A/UamVLA.py` | **Modify** | ~80 lines net (rewrite `predict_action` body, add `_build_prefill_embeds` + `_decode_generated_actions`, remove `_extract_action_tokens_for_sample` and old `_decode_action_tokens`, promote `_act0_id` to instance attr) |
| `starVLA/model/modules/uamvla/backbone_wrapper.py` | **No change** | `_replace_state_tokens` is already exported (already in `__all__`) |
| `tests/test_uamvla_predict_action_decode.py` | **Create** | ~120 lines, 5 L0 unit tests |
| `tests/test_uamvla_predict_action_prefill.py` | **Create** | ~80 lines, 3 L0 unit tests |
| `tests/test_uamvla_predict_action.py` | **Modify** | Replace existing skip-placeholder with 3 L1 smoke tests (~80 lines net add) |
| `tests/test_uamvla_predict_action_alignment.py` | **Create** | ~100 lines, 1 critical training-vs-inference numerical equality test |

Net total: ~150 lines main + ~380 lines test code added/modified.

## 6. Testing Plan

### 6.1 L0 unit tests (decode path)

`tests/test_uamvla_predict_action_decode.py`:
- `test_decode_extracts_action_tokens_in_range` — ID-range filter keeps only `[_act0_id, _act0_id + n_bins)`
- `test_decode_pads_with_mid_bin_when_short` — fewer ACT tokens than `chunk_len` → pad with mid_bin id
- `test_decode_truncates_when_too_long` — more ACT tokens than `chunk_len` → truncate
- `test_decode_filters_non_action_tokens` — non-ACT tokens (e.g. `<|im_end|>`) skipped
- `test_decode_per_row_independent_for_batched_input` — B=2 with different ACT-token counts decode independently

### 6.2 L0 unit tests (prefill path)

`tests/test_uamvla_predict_action_prefill.py`:
- `test_prefill_calls_state_encoder_when_canonical_state_given` — verify `state_encoder` invoked and `_replace_state_tokens` called (use `MagicMock` to assert call sequence)
- `test_prefill_skips_state_encoder_when_canonical_state_absent` — returns plain `embed_tokens(input_ids)`
- `test_prefill_output_shape_matches_input_ids` — output `(B, S_prompt, hidden_size)`

### 6.3 L1 smoke tests (full predict_action with mock generate)

`tests/test_uamvla_predict_action.py` (replace existing skip-placeholder):
- `test_predict_action_returns_correct_shape_b1` — B=1, mock `generate` returns fixed ACT sequence, verify `(1, H, action_dim)` output
- `test_predict_action_returns_correct_shape_b_geq_1` — B=2, verify per-row independence
- `test_predict_action_handles_missing_canonical_state` — example without `canonical_state` still completes

### 6.4 Training-vs-inference alignment test (critical safety net)

`tests/test_uamvla_predict_action_alignment.py`:
- `test_prefix_hidden_states_match_between_training_and_inference` — construct identical example, run training `forward` and inference prefill independently, assert `<|action_start|>`-position hidden states match within `atol=1e-4` (fp32, no autocast). Catches any divergence in state splice / chat template / tokenization.

### 6.5 L5 real-checkpoint validation (out of scope, listed for completeness)

```bash
python examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py \
    --model_ckpt <trained_uamvla_ckpt> --suite spatial --num_episodes 1
```
Tracked separately under L5 remote-GPU tasks.

## 7. Migration

- No checkpoint compatibility work required (no checkpoints exist to migrate).
- The `forward` path and all training tests are unaffected.
- `_extract_action_tokens_for_sample` (currently `NotImplementedError`) is removed; any code paths that try to call it would have already been broken — the removal makes the contract explicit.

## 8. Open Questions

None requiring user input. Implementation-level details (exact import sites, type hint style, docstring formatting) deferred to the plan.

## 9. Effort Estimate

- Instance attribute promotion: 5 min
- `_build_prefill_embeds` + 3 L0 tests: 30 min
- `_decode_generated_actions` + 5 L0 tests: 40 min
- `predict_action` rewrite: 20 min
- Cleanup (`_extract_action_tokens_for_sample` removal + old `_decode_action_tokens` cleanup): 5 min
- L1 smoke tests: 30 min
- Training-vs-inference alignment test: 45 min
- Final regression + verification: 15 min

**Total**: ~3 hours implementation + tests, plus L5 real-checkpoint validation tracked separately.
