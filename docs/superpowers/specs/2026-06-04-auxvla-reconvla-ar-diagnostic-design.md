# AuxVLAGR00T ReconVLA AR Diagnostic Mode Design

Date: 2026-06-04

## Goal

Add an eval-only diagnostic inference mode to AuxVLAGR00T that uses a ReconVLA official fine-tuned checkpoint for autoregressive action-token generation while keeping the StarVLA websocket, ModelClient, action chunk, and `dataset_statistics.json` unnormalization path.

This mode is not intended to reproduce the official ReconVLA raw-action eval path. That raw path has already been validated separately. The purpose here is to test whether the StarVLA/AuxVLAGR00T framework organization, server-client protocol, action chunk buffering, and unnormalization path are healthy when the action source is the official ReconVLA AR policy.

## Diagnostic Question

The diagnostic should isolate the failure location:

- If official ReconVLA raw eval works, the official checkpoint and CALVIN environment are viable.
- If ReconVLA AR normalized actions through StarVLA unnormalization also work, the StarVLA server-client/action pipeline is probably healthy.
- If official-compose AR works but AuxVLAGR00T-compose AR drops, the issue is likely image/prompt/state input organization.
- If both AR diagnostic modes work but GR00T-head AuxVLAGR00T is weak, the issue is likely GR00T action head training, LoRA adaptation, or action-head conditioning rather than the eval framework.

## Non-Goals

- Do not change AuxVLAGR00T training forward behavior.
- Do not replace the GR00T action head.
- Do not add a new CALVIN eval environment.
- Do not implement the official raw-action Flask server path.
- Do not support LIBERO or other benchmarks in this first step.
- Do not support `.pt` ReconVLA checkpoint conversion. The diagnostic assumes a Hugging Face-style checkpoint directory.

## Proposed Config

AuxVLAGR00T keeps the existing default GR00T inference path. The diagnostic is opt-in:

```yaml
framework:
  reconvla:
    inference_mode: gr00t              # gr00t | reconvla_ar_normalized
    ar_input_mode: official_compose    # official_compose | auxvla_compose
    action_stat_path: third_party/ReconVLA/reconvla/statistics.yaml
    double_instruction: true
    max_new_tokens: 128
    temperature: 0.0
    top_p: null
    num_beams: 1
```

The config used for the diagnostic should set the action horizon to match ReconVLA's CALVIN action-token training:

```yaml
framework:
  action_model:
    action_horizon: 5
    future_action_window_size: 4
```

The `action_model` is not used by AR generation, but `ModelClient.get_action_chunk_size()` reads `future_action_window_size + 1` from `config.yaml`. This must match the returned chunk length.

## Data Flow

The diagnostic path should use the existing StarVLA websocket server:

```text
CALVIN eval_calvin.py
  -> CalvinPolicyWrapper
  -> ModelClient.step
  -> WebsocketPolicyServer
  -> AuxVLAGR00T.predict_action
  -> ReconVLA AR generate
  -> ActionTokenizer.decode_token_ids_to_actions
  -> {"normalized_actions": np.ndarray [B, T, 7]}
  -> ModelClient.unnormalize_actions(dataset_statistics.json)
  -> CALVIN env.step
```

This intentionally returns `normalized_actions` rather than raw actions. The result is then processed by the same StarVLA client-side unnormalization and chunk buffering used by GR00T checkpoints.

## Input Modes

### official_compose

This mode mirrors ReconVLA's official inference input as closely as possible while returning StarVLA-compatible normalized actions:

- static and gripper images are resized and vertically concatenated into a `334 x 334` RGB image.
- `robot_obs` must be the raw 15-D CALVIN observation.
- `robot_obs` is encoded with ReconVLA `encode_robot_obs(...)` and `action_stat_path`.
- prompt uses the official system message and double-instruction pattern when `double_instruction=true`.
- image tensor is produced by the ReconVLA vision tower image processor.

Use this first. It keeps the AR model input close to what official ReconVLA expects and mainly tests StarVLA's outer action pipeline.

### auxvla_compose

This mode uses AuxVLAGR00T's current image composition for the visual input:

- use the existing `single_view_mode` path, including `concat_vertical`.
- keep the ReconVLA AR prompt and robot_obs token path so the action-token policy still receives state in the official tokenized form.
- image tensor is produced by the ReconVLA vision tower image processor after AuxVLAGR00T composition.

Use this second. It tests whether AuxVLAGR00T's visual input organization is a source of the performance gap.

## Required Eval Input

The CALVIN eval client must send raw 15-D `robot_obs` through the websocket example for AR diagnostic mode. The current GR00T state path sends only normalized 7-D state for AuxVLAGR00T. That is insufficient for official `encode_robot_obs`, which requires 15 values.

The preferred server-side example field is:

```python
example["robot_obs"] = np.asarray(obs["robot_obs"], dtype=np.float32)
```

For compatibility, the implementation may also accept:

```python
example["uamvla_raw_state"]["robot_obs"]
```

AuxVLAGR00T should raise a clear `RuntimeError` if `reconvla_ar_normalized` mode is active and no 15-D `robot_obs` is available.

## Action Decoding

AR generation returns token ids. The diagnostic should:

1. remove prompt tokens and terminal tokens as needed.
2. decode action token ids with `ActionTokenizer.decode_token_ids_to_actions`.
3. flatten/pad/trim to `action_horizon * 7` values.
4. reshape to `(action_horizon, 7)`.
5. return:

```python
{"normalized_actions": actions[None].astype(np.float32)}
```

The returned values remain in ReconVLA action-token normalized space. StarVLA ModelClient then applies `dataset_statistics.json` unnormalization.

## Statistics Check

Because this mode intentionally mixes ReconVLA action-token normalization with StarVLA dataset-statistics unnormalization, the implementation should log a one-time statistics comparison:

- ReconVLA `statistics.yaml` `act_min_bound`
- ReconVLA `statistics.yaml` `act_max_bound`
- StarVLA `dataset_statistics.json` action min/max as seen by the eval client, if available at the framework side

At minimum, the framework should log:

- the `action_stat_path`
- action horizon
- generated decoded action shape
- decoded action min/max before StarVLA unnormalization

The result of this diagnostic should not be interpreted as official ReconVLA reproduction unless these statistics are known to match.

## Error Handling

- Missing checkpoint directory: let the existing ReconVLA load path fail with the original error.
- Missing `action_stat_path`: raise `FileNotFoundError`.
- Missing 15-D `robot_obs`: raise `RuntimeError` with the expected field names.
- Unsupported `ar_input_mode`: raise `ValueError`.
- Generated action sequence shorter than expected: pad with zeros after logging a warning.
- Generated action sequence longer than expected: trim after logging a warning.

## Implementation Boundaries

Expected code changes:

- `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`
  - add ReconVLA AR input builders in `ReconVLAInterface`
  - add `generate_normalized_actions(...)`
  - branch in `AuxVLAGR00T.predict_action`
- `examples/calvin/eval_files/eval_calvin.py`
  - always provide raw 15-D `robot_obs` to the websocket example when framework config requests ReconVLA AR diagnostic mode
- focused tests under `tests/framework/test_auxvla_gr00t_static.py`
  - no GPU generation required; mock the interface and verify routing/shape/error behavior

No training configs should be changed globally. Create a separate diagnostic config or use CLI overrides.

## Acceptance Criteria

- Default AuxVLAGR00T inference remains GR00T-based and returns `normalized_actions`.
- `inference_mode=reconvla_ar_normalized` routes through ReconVLA AR generation.
- `official_compose` and `auxvla_compose` are both supported.
- CALVIN eval sends raw 15-D `robot_obs` when AR diagnostic mode is active.
- Server response remains `{"normalized_actions": ...}` so StarVLA ModelClient unnormalization is exercised.
- Static tests cover routing, missing robot_obs failure, chunk length, and input-mode selection.
- Remote smoke can load an official ReconVLA checkpoint directory and print decoded normalized action shape/min/max before long evaluation.
