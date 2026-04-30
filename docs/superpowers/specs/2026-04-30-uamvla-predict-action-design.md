# UamVLA `predict_action` Autoregressive Inference — Design Spec

**Date:** 2026-04-30
**Author:** UniamVLA migration team
**Status:** Draft, awaiting user review
**References:**
- starVLA framework contract: [`base_framework.py:119-131`](starVLA/model/framework/base_framework.py#L119-L131)
- Reference implementation: [`QwenFast.py:181-225`](starVLA/model/framework/VLM4A/QwenFast.py#L181-L225) (autoregressive token-generation VLA, in-tree)
- Prior spec (dependency): [`2026-04-30-action-start-special-token-design.md`](2026-04-30-action-start-special-token-design.md)
- HuggingFace generation API: [`GenerationConfig`](https://huggingface.co/docs/transformers/main_classes/text_generation) documents `max_new_tokens`, `min_new_tokens`, and custom `logits_processor` generation controls.
- HuggingFace decoder-only generation guidance: [LLM tutorial](https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side) warns that batched decoder-only generation should use left padding; right padding can make the model continue from a pad token.
- Qwen3-VL HuggingFace docs: [model docs](https://huggingface.co/docs/transformers/model_doc/qwen3_vl) expose multimodal forward/generation kwargs including image/video grids and `mm_token_type_ids`.
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
- Industry survey (OpenVLA / RT-2 / π0-FAST / VLA-0 / OpenVLA-OFT) confirms the standard pattern: full-vocab training + `model.generate()` + ID-range scan over generated tokens for action extraction. UamVLA additionally has an existing `<|action_start|>`-aware `ActionLogitsProcessor`; because inference should emit a fixed action chunk, this spec wires that processor by default and keeps ID-range scan as the defensive decoder.

## 2. Goals

- Replace the placeholder `predict_action` with a real autoregressive inference path:
  - Build prefill `inputs_embeds` that include `state_encoder` output spliced into `<|state_*|>` positions (training-equivalent).
  - Call HuggingFace `model.generate(input_ids=..., inputs_embeds=..., max_new_tokens=H × action_dim, do_sample=False)` so generation still has textual prompt IDs for processors, position handling, and output-shape consistency.
  - Preserve multimodal generation kwargs such as `pixel_values`, `image_grid_thw`, and `mm_token_type_ids` when present or derivable.
  - Extract action tokens from the generated sequence with a helper that supports both known HF return shapes: `(B, S_prompt + N_new)` and `(B, N_new)`.
  - Decode via `ActionTokenizer`.
- Preserve the `predict_action` interface contract: `(examples, **kwargs) -> dict[{"normalized_actions": np.ndarray (B, H, action_dim)}]`. **All existing callers see zero change.**
- Support batched inference (B ≥ 1) from day one — required for parallel-env LIBERO evaluation.
- Land a training-vs-inference numerical alignment test that asserts the prefix hidden state at `<|action_start|>` is identical between the training forward path and the inference prefill path. This is the safety net for the entire approach.
- Mirror the QwenFast implementation pattern where applicable; deviate only where state splice or other UamVLA-specific concerns require it.

## 3. Non-Goals

- **Real-checkpoint LIBERO end-to-end runs.** That's an L5 remote-GPU task tracked separately; this spec only goes as far as mock-backbone smoke tests + the training-vs-inference alignment test.
- **Performance tuning.** No KV-cache micro-optimization, no padding-bucket optimization, no quantization. Use HuggingFace `generate()` defaults except for the correctness-critical controls explicitly listed below.
- **New logits-processor designs.** Reuse the existing `ActionLogitsProcessor`; do not introduce a second masking mechanism unless implementation proves the existing one cannot support the prompt/`inputs_embeds` path.
- **`train_starvla.py` modifications.** The training-time `eval_action_model` will benefit automatically without code change there.
- **Old checkpoint compatibility.** No checkpoints exist that would conflict; the framework forward path is unchanged so existing training behavior is preserved.
- **Sampling strategies (top-k / top-p / temperature).** Default to greedy (`do_sample=False`); add later if needed.
- **Streaming inference.** Out of scope; `generate()` returns the full chunk in one call.
- **Variable-length action chunks.** Fixed `H × action_dim` per call; matches OpenVLA / OpenVLA-OFT.

## 4. Design

### 4.1 Architecture overview

```
predict_action(examples, **kwargs)
├── Step 1: build_inputs (reuses training pipeline, but inference uses left padding)
│       images + instructions + canonical_state
│           ↓
│       qwen_inputs = {input_ids, attention_mask, pixel_values,
│                       image_grid_thw, mm_token_type_ids?, canonical_state}
│
├── Step 2: Prefill kwargs — build inputs_embeds with state splice
│       inputs_embeds = embed_tokens(input_ids)
│       state_embeds = state_encoder(canonical_state)
│       inputs_embeds = _replace_state_tokens(inputs_embeds, ...)
│       generation_kwargs keeps original input_ids, attention_mask,
│       multimodal tensors, optional mm_token_type_ids, and inputs_embeds
│
├── Step 3: HF generate (batched, autoregressive)
│       generated_ids = model.generate(
│           input_ids=input_ids,
│           inputs_embeds=inputs_embeds,
│           pixel_values=pixel_values, image_grid_thw=image_grid_thw,
│           mm_token_type_ids=mm_token_type_ids,
│           attention_mask=attention_mask,
│           max_new_tokens=H * action_dim,
│           min_new_tokens=H * action_dim,
│           logits_processor=ActionLogitsProcessor(...),
│           do_sample=False, use_cache=True,
│       )
│
├── Step 4: Extract action tokens (return-shape aware tail extraction + ID scan)
│       new_ids = _extract_generated_tail(generated_ids, prompt_len, chunk_len)
│       mask = (new_ids >= _act0_id) & (new_ids < _act0_id + n_bins)
│       per-row: select masked tokens, pad with mid_bin to chunk_len
│
└── Step 5: Decode → normalized actions
        action_tokenizer.decode(action_ids).reshape(H, action_dim)
        → return {"normalized_actions": np.ndarray (B, H, action_dim)}
```

**Key design choices** (all confirmed during brainstorming):
1. **Prefill+generate split (Plan B)** — compute `inputs_embeds` via the wrapper's state-splice path, but still pass the original `input_ids` into `generate()`. This keeps prompt token IDs visible to logits processors and Qwen3-VL generation helpers while preserving training-equivalent state-token embeddings.
2. **Default `ActionLogitsProcessor`** — after `<|action_start|>`, generation is constrained to `<ACT_i>` tokens by default. ID-range scan remains in the decoder as a final guard and as the opt-out fallback when `constrain_action_logits=False`.
3. **Return-shape aware token extraction** — do not assume `generated_ids[:, prompt_len:]`. HF model/version combinations can return prompt+new IDs or new-only IDs when `inputs_embeds` are involved. The decoder must normalize either shape before ID-range scanning.
4. **Greedy decoding** (`do_sample=False`) — matches OpenVLA / QwenFast; action prediction is deterministic.
5. **Inference-only left padding** — temporarily set `tokenizer.padding_side = "left"` while building inference inputs, then restore the previous value. Training keeps the existing right-padded label construction.
6. **Batched B ≥ 1 from day one** — `build_inputs` is already batched; `model.generate` natively supports batched inputs.

### 4.2 What changes vs current code

| Component | Current | New |
|---|---|---|
| `predict_action` body | 1× backbone forward + `ActionHead.predict` + `_decode_action_tokens` (broken) | inference-left-padded `build_inputs` + `_build_prefill_generate_kwargs` + constrained `model.generate` + `_decode_generated_actions` |
| `_extract_action_tokens_for_sample` | `raise NotImplementedError` | **Removed** |
| `_decode_action_tokens` | Calls broken sample-extractor | **Replaced** by `_decode_generated_actions` |
| `_build_prefill_generate_kwargs` | (does not exist) | **New** method, reuses `_replace_state_tokens`, preserves `input_ids`, `attention_mask`, multimodal kwargs, and optional `mm_token_type_ids` |
| `_extract_generated_tail` | (does not exist) | **New** helper, normalizes prompt+new vs new-only `generate()` outputs |
| `self._act0_id` | Local variable in `__init__` | **Promoted** to instance attribute |
| `ActionHead.predict` | Used by `predict_action` | **No longer called from inference** (kept; future visualizers may use it) |

### 4.3 `_build_prefill_generate_kwargs` — state splice extracted to its own method

```python
def _build_prefill_generate_kwargs(self, qwen_inputs: dict) -> dict:
    """Prefill: embed input_ids, splice in state-token embeddings, and
    return kwargs suitable for model.generate().

    The state splice is identical to the training forward path. We still keep
    the original input_ids in the generation kwargs so generation-time helpers
    and logits processors can see the textual prompt.
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

This method does **not** run any transformer layers. It only prepares embeddings and generation kwargs; the transformer runs inside `model.generate(...)`.

`mm_token_type_ids` policy:
- Preserve `qwen_inputs["mm_token_type_ids"]` when the processor returns it.
- If absent, derive it through the existing wrapper helper (`image_pad` positions become image type, everything else text type), matching the forward path.
- Pass it to `generate()` only when non-`None`. This keeps the code compatible with both current local Qwen3-VL wrappers and newer official API surfaces that expose the argument explicitly.

### 4.4 `predict_action` — main loop

```python
@torch.inference_mode()
def predict_action(self, examples, **kwargs) -> dict:
    if not isinstance(examples, list):
        examples = [examples]

    from deployment.model_server.tools.image_tools import to_pil_preserve
    from transformers import LogitsProcessorList
    from starVLA.model.modules.uamvla.inference import ActionLogitsProcessor

    states = [e.get("canonical_state") for e in examples]
    if any(s is None for s in states) and any(s is not None for s in states):
        raise ValueError("predict_action requires canonical_state for all examples or none.")

    tokenizer = self.qwen_vl_interface.tokenizer
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        qwen_inputs = self.qwen_vl_interface.build_inputs(
            images=[[to_pil_preserve(img) for img in e["image"]] for e in examples],
            instructions=[e["lang"] for e in examples],
            canonical_state=stack_canonical(states) if states[0] is not None else None,
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

Padding policy:
- `predict_action` uses **left padding only while building inference inputs**. The original tokenizer padding side is restored immediately after `build_inputs`, so the training/data-loader path keeps its current right-padded assumptions.
- Tests must cover `B > 1` with different prompt lengths and verify that each row's last non-pad prompt token is `<|action_start|>`.

Logits-mask policy:
- `constrain_action_logits=True` by default.
- The caller can pass `predict_action(..., constrain_action_logits=False)` for ablations or debugging; decoding still uses ID-range scan and mid-bin padding.
- Passing `input_ids` alongside `inputs_embeds` is required for the default processor because it activates when the last prompt token is `self.action_start_id`.
- Construct a fresh `ActionLogitsProcessor` per `generate()` call because the processor tracks per-row remaining action steps.

### 4.5 `_extract_generated_tail` and `_decode_generated_actions`

`generate()` output shape is treated as a compatibility boundary, not a fixed invariant:
- With `input_ids` + `inputs_embeds`, current HF behavior is expected to return prompt+new IDs: `(B, S_prompt + N_new)`.
- With `inputs_embeds`-only, some model/version combinations return new-only IDs: `(B, N_new)`.
- The implementation must support both because this path depends on a multimodal model wrapper and HF generation internals.

```python
def _extract_generated_tail(
    self,
    generated_ids: torch.Tensor,
    prompt_len: int,
    chunk_len: int,
) -> torch.Tensor:
    """Return generated new-token IDs from either prompt+new or new-only sequences."""
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

Then decode the normalized tail via ID-range scan:

```python
def _decode_generated_actions(
    self,
    generated_ids: torch.Tensor,
    prompt_len: int,
    H: int,
    action_dim: int,
) -> np.ndarray:
    """Extract action tokens from the generated tail and decode to normalized actions.

    Strategy: normalize generate() return shape, then scan for action token ids.
    Tokens outside [_act0_id, _act0_id + n_bins) are skipped defensively.
    Length is normalized to H * action_dim via mid_bin padding (action ≈ 0)
    or truncation.

    Args:
        generated_ids: (B, S_prompt + N_new) or (B, N_new) long tensor from model.generate
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

Action-token range policy:
- The primary mapping is the `ActionTokenizer`'s explicit token-to-bin mapping.
- At initialization, assert the registered `<ACT_i>` token IDs are contiguous before using `[self._act0_id, self._act0_id + n_bins)` for masks and fast range checks.
- If a future tokenizer breaks contiguity, fail fast and switch this decoder to use `action_tokenizer.token_id_to_bin` membership instead of range checks.

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
- Existing caller kwargs remain harmless (e.g., `train_starvla.py:397` passes `use_ddim=True, num_ddim_steps=20` for diffusion frameworks; UamVLA ignores unknown diffusion-only kwargs). `constrain_action_logits` is the only new recognized inference kwarg in this spec.
- All callers (`eval_libero_model.py`, `train_starvla.py`, `websocket_policy_server.py`) need **zero modification**.

### 4.8 Invariants and failure modes

#### Strong invariants
1. **Generated-output shape is normalized before decoding** — support both `(B, S_prompt + N_new)` and `(B, N_new)`; never rely unconditionally on `generated_ids[:, prompt_len:]`.
2. **Inference padding is left-sided and local** — `predict_action` temporarily changes tokenizer padding side only while building inference inputs, then restores the previous value.
3. **Action-token IDs are contiguous or fail fast** — default logits masking and range scan require `<ACT_0>...<ACT_{n-1}>` to occupy `[self._act0_id, self._act0_id + n_bins)`.
4. **Default generation is action-constrained** — after `<|action_start|>`, `ActionLogitsProcessor` masks non-action logits for exactly `H × action_dim` steps unless explicitly disabled by caller kwargs.
5. **Training `forward` is unchanged** — all training tests stay green; this spec touches only the inference path.
6. **`<|action_start|>` remains the chat_template terminator** — chat_template.py is not modified by this spec.

#### Failure modes and fallbacks

| Failure | Behavior |
|---|---|
| Undertrained model emits non-ACT tokens (e.g. `<\|im_end\|>`) | With default mask, this should not happen after `<\|action_start\|>`; if mask is disabled or activation fails, ID-range scan filters them out and keeps remaining ACT tokens in order. |
| `ActionLogitsProcessor` does not activate on first step | Treat as a test failure. The intended generate call passes `input_ids` so the processor sees the prompt-ending `<\|action_start\|>` token. If a future HF path hides prompt IDs, add an "active immediately" mode to the processor instead of silently falling back. |
| Model emits fewer than `H × action_dim` ACT tokens (early stop) | `min_new_tokens=chunk_len` plus the logits mask should prevent early EOS. Decoder still pads with `mid_id` (action ≈ 0) if an unusual wrapper returns short output. |
| Model emits more than `H × action_dim` ACT tokens (rare; `max_new_tokens` should prevent this) | Truncate to `chunk_len` |
| Some examples lack `canonical_state` while others include it | Raise `ValueError`; batched state splice should be all-or-none. |
| All examples lack `canonical_state` | `_build_prefill_generate_kwargs` skips state splice and returns plain `embed_tokens(input_ids)`. Matches the no-state forward path. |
| Processor omits `mm_token_type_ids` | Derive via `UamVLABackboneInterface._derive_mm_token_type_ids` when possible; otherwise omit the kwarg. |
| Mock tokenizer in unit tests | `_decode_generated_actions` works for any `_act0_id ≥ 0`; tests construct mock `generate` outputs with IDs in the expected range. |
| Batched inputs with different prompt lengths | Inference build uses left padding; `attention_mask` marks real tokens; each row's last non-pad token must be `<\|action_start\|>`. |
| Real Qwen3-VL rejects `input_ids` + `inputs_embeds` in `generate()` | Treat as an implementation blocker to resolve with a real-checkpoint smoke test. Fallback design is manual prefill/step decoding with explicit processor state; do not silently drop `input_ids` because that would break mask activation and may affect multimodal position handling. |

### 4.9 Performance and resource budget

- **Peak memory**: comparable to training forward (one prefill pass) minus backward; should be lower than training step.
- **Latency** (estimate, Qwen3-VL-8B + B=8 + H=8 + action_dim=7 = 56 new tokens): prefill ~2-4s + decode ~0.5-1s ≈ 3-5s/batch. To be validated at L5.
- **KV cache**: HF `generate(use_cache=True)` manages it automatically; each new token reuses the prefix KV.

### 4.10 Compatibility

- **transformers**: requires `>= 4.57.0` (already enforced by [`backbone_wrapper.py:45`](starVLA/model/modules/uamvla/backbone_wrapper.py#L45)). `generate(inputs_embeds=...)` has been stable since 4.40+.
- **Qwen3-VL multimodal kwargs**: preserve `pixel_values`, `image_grid_thw`, and `mm_token_type_ids` when available. Because `inputs_embeds` generation is model-specific around multimodal RoPE/position handling, implementation must include a Qwen3-VL smoke test, not only text-only dummy-model tests.
- **DeepSpeed / Accelerate**: callers already do `accelerator.unwrap_model(...)` before invoking `predict_action`; no change needed.

## 5. Concrete File Changes

| File | Change Type | Lines |
|---|---|---|
| `starVLA/model/framework/VLM4A/UamVLA.py` | **Modify** | ~110 lines net (rewrite `predict_action` body, add `_build_prefill_generate_kwargs` + `_extract_generated_tail` + `_decode_generated_actions`, remove `_extract_action_tokens_for_sample` and old `_decode_action_tokens`, promote `_act0_id` to instance attr, add action-token contiguity assertion) |
| `starVLA/model/modules/uamvla/backbone_wrapper.py` | **No change** | `_replace_state_tokens` is already exported (already in `__all__`) |
| `tests/test_uamvla_predict_action_decode.py` | **Create** | ~150 lines, 7 L0 unit tests |
| `tests/test_uamvla_predict_action_prefill.py` | **Create** | ~110 lines, 5 L0 unit tests |
| `tests/test_uamvla_predict_action.py` | **Modify** | Replace existing skip-placeholder with 8 L1 smoke tests (~150 lines net add) |
| `tests/test_uamvla_predict_action_alignment.py` | **Create** | ~100 lines, 1 critical training-vs-inference numerical equality test |

Net total: ~180 lines main + ~480 lines test code added/modified.

## 6. Testing Plan

### 6.1 L0 unit tests (decode path)

`tests/test_uamvla_predict_action_decode.py`:
- `test_decode_extracts_action_tokens_in_range` — ID-range filter keeps only `[_act0_id, _act0_id + n_bins)`
- `test_decode_pads_with_mid_bin_when_short` — fewer ACT tokens than `chunk_len` → pad with mid_bin id
- `test_decode_truncates_when_too_long` — more ACT tokens than `chunk_len` → truncate
- `test_decode_filters_non_action_tokens` — non-ACT tokens (e.g. `<|im_end|>`) skipped
- `test_decode_per_row_independent_for_batched_input` — B=2 with different ACT-token counts decode independently
- `test_extract_generated_tail_accepts_prompt_plus_new_shape` — `(B, S_prompt + chunk_len)` returns only new IDs
- `test_extract_generated_tail_accepts_new_only_shape` — `(B, chunk_len)` returns the whole sequence

### 6.2 L0 unit tests (prefill path)

`tests/test_uamvla_predict_action_prefill.py`:
- `test_prefill_calls_state_encoder_when_canonical_state_given` — verify `state_encoder` invoked and `_replace_state_tokens` called (use `MagicMock` to assert call sequence)
- `test_prefill_skips_state_encoder_when_canonical_state_absent` — returns plain `embed_tokens(input_ids)`
- `test_prefill_output_shape_matches_input_ids` — output `(B, S_prompt, hidden_size)`
- `test_prefill_preserves_mm_token_type_ids_when_processor_returns_them` — generated kwargs include the original tensor
- `test_prefill_derives_mm_token_type_ids_when_absent` — generated kwargs match wrapper helper output for image-token positions

### 6.3 L1 smoke tests (full predict_action with mock generate)

`tests/test_uamvla_predict_action.py` (replace existing skip-placeholder):
- `test_predict_action_returns_correct_shape_b1` — B=1, mock `generate` returns fixed ACT sequence, verify `(1, H, action_dim)` output
- `test_predict_action_returns_correct_shape_b_geq_1` — B=2, verify per-row independence
- `test_predict_action_handles_missing_canonical_state` — example without `canonical_state` still completes
- `test_predict_action_rejects_mixed_canonical_state_presence` — mixed state/no-state batch raises `ValueError`
- `test_predict_action_uses_left_padding_temporarily_and_restores_tokenizer` — variable-length prompts use left padding during inference, then restore prior tokenizer setting
- `test_predict_action_enables_action_logits_processor_by_default` — mock `generate` receives a `LogitsProcessorList` containing `ActionLogitsProcessor`
- `test_predict_action_can_disable_action_logits_processor` — `constrain_action_logits=False` passes no action constraint
- `test_predict_action_generate_kwargs_keep_input_ids_and_inputs_embeds` — mock `generate` receives both tensors so the processor can see `<|action_start|>`

### 6.4 L1 Qwen3-VL generation-shape smoke test

`tests/test_uamvla_predict_action_qwen3vl_generate.py` (skip unless a tiny/local Qwen3-VL fixture is available):
- `test_qwen3vl_generate_accepts_input_ids_plus_inputs_embeds` — verifies the actual backbone accepts the planned kwargs (`input_ids`, `inputs_embeds`, multimodal tensors, optional `mm_token_type_ids`)
- `test_qwen3vl_generate_return_shape_is_decodable` — records whether the return is prompt+new or new-only and confirms `_extract_generated_tail` handles it

### 6.5 Training-vs-inference alignment test (critical safety net)

`tests/test_uamvla_predict_action_alignment.py`:
- `test_prefix_hidden_states_match_between_training_and_inference` — construct identical example, run training `forward` and inference prefill independently, assert `<|action_start|>`-position hidden states match within `atol=1e-4` (fp32, no autocast). Catches any divergence in state splice / chat template / tokenization.

### 6.6 L5 real-checkpoint validation (out of scope, listed for completeness)

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

None requiring user input.

Implementation verification item: run a real Qwen3-VL smoke test before calling the implementation complete. Text-only dummy models are insufficient for the `input_ids` + `inputs_embeds` + multimodal-kwargs combination because Qwen3-VL position/RoPE handling is model-specific.

## 9. Effort Estimate

- Instance attribute promotion: 5 min
- `_build_prefill_generate_kwargs` + 5 L0 tests: 45 min
- `_extract_generated_tail` / `_decode_generated_actions` + 7 L0 tests: 55 min
- `predict_action` rewrite with left padding + default logits mask: 35 min
- Cleanup (`_extract_action_tokens_for_sample` removal + old `_decode_action_tokens` cleanup): 5 min
- L1 smoke tests: 45 min
- Qwen3-VL generate-shape smoke test: 30 min if a local fixture/checkpoint is available; otherwise document as skipped L5 follow-up
- Training-vs-inference alignment test: 45 min
- Final regression + verification: 15 min

**Total**: ~4 hours implementation + tests, plus L5 real-checkpoint validation tracked separately.
