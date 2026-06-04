# AuxVLAGR00T ReconVLA AR + Recon Training Design

Date: 2026-06-04

## Goal

Add a training mode to `AuxVLAGR00T` that reproduces the official ReconVLA CALVIN fine-tuning recipe inside the StarVLA training framework:

- load the ReconVLA pretrain checkpoint from `ckpt/pretrain-checkpoint-10388`
- train the ReconVLA/Qwen backbone with LoRA
- use official autoregressive action-token prediction instead of the GR00T flow-matching action head
- inject the official 15-D CALVIN `robot_obs` token sequence into the VLM prompt
- enable the official ReconVLA image reconstruction loss (`vm_loss`)
- train on `uamvla_calvin_abc` with 5-step action chunks

The experiment should answer whether the StarVLA/AuxVLAGR00T data, training, checkpointing, websocket, and CALVIN eval organization can reproduce a ReconVLA-style policy when the action objective and reconstruction objective are aligned with the official method.

## Non-Goals

- Do not remove or alter the existing GR00T action-head path.
- Do not use the StarVLA `aux_heads.recon` head for this official-recipe experiment.
- Do not train the GR00T action head in this mode.
- Do not change the default `AuxVLAGR00T` behavior; the new path must be opt-in.
- Do not change existing `uamvla_calvin_abc_h8` or GR00T H8 experiments.
- Do not claim official ReconVLA reproduction until action/statistics/image/state details are verified by smoke tests and eval.

## Recommended Config Shape

The mode should be explicit:

```yaml
framework:
  name: AuxVLAGR00T
  reconvla:
    model_path: ckpt/pretrain-checkpoint-10388
    vision_tower_path: ckpt/siglip-so400m-patch14-384
    training_mode: reconvla_ar_recon
    inference_mode: reconvla_ar_normalized
    ar_input_mode: official_compose
    action_stat_path: third_party/ReconVLA/reconvla/statistics.yaml
    disable_internal_recon_loss: false
    lora:
      enabled: true
      r: 32
      lora_alpha: 16
      lora_dropout: 0.0
      init_lora_weights: gaussian
      train_mm_projector: true
      train_mm_inv_projector: true
      target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
        - gate_proj
        - up_proj
        - down_proj
  action_model:
    action_horizon: 5
    future_action_window_size: 4
```

The dataset should use the existing 5-step CALVIN mixture:

```bash
DATA_MIX=uamvla_calvin_abc
ACTION_TYPE=calvin_rel_action
```

## Architecture

`AuxVLAGR00T.forward()` should branch on `framework.reconvla.training_mode`.

Default path:

```text
image + instruction
  -> ReconVLA hidden states
  -> GR00T action head
  -> optional StarVLA aux heads
```

Official ReconVLA recipe path:

```text
official-composed image + instruction + tokenized 15-D robot_obs
  -> ReconQwen2ForCausalLM
  -> LM CE loss on action tokens
  -> official internal vm_loss from target_images
  -> total loss = lm_loss + vm_loss
```

This keeps both approaches in one framework class but makes the objective and data flow unambiguous.

## Data Flow

For each training example:

1. Read `video.primary_image` and `video.wrist_image`.
2. Compose the input image with the official ReconVLA CALVIN layout.
3. Build the official target image from the already available gaze crop/image target plus wrist image.
4. Extract raw 15-D `robot_obs`; do not use the 7-D GR00T state slice for AR prompt injection.
5. Encode `robot_obs` using ReconVLA `encode_robot_obs(..., statistics=action_stat_path)`.
6. Encode the 5-step, 7-D action chunk into 35 action tokens.
7. Build `input_ids`, `attention_mask`, and `labels` so only action tokens contribute to LM CE.
8. Call the ReconVLA model with `labels` and `target_images`.

The batch output should log:

```text
loss/action_loss_ar
loss/reconvla_lm_loss
loss/reconvla_vm_loss
loss/total
```

The training script can keep reading `action_loss` as the total loss for compatibility.

## Official Recon Loss

Use the internal ReconVLA loss implemented in `ReconQwen2ForCausalLM.inner_forward()`:

```text
if self.training and config.recon_enable:
    vm_loss = compute_vm_loss(target_images, hidden_states, boi_ids, eoi_ids, ...)
    loss = lm_loss + vm_loss
```

This is the right loss for this experiment because it uses the official `mm_inv_projector` and pixel decoder path. The StarVLA `aux_heads.recon` head should remain disabled for this run.

`disable_internal_recon_loss` should be false in this mode. If the checkpoint/config does not expose a usable pixel decoder or `mm_inv_projector`, setup should fail early with a clear error rather than silently training action CE only.

## Action Tokenization

Avoid double normalization.

The official `encode_actions(..., statistics=...)` expects raw continuous actions and maps them through `statistics.yaml` into `[-1, 1]` before tokenization. StarVLA's current CALVIN dataloader applies `StateActionTransform(..., normalization_modes=min_max)` to `example["action"]`.

The implementation must choose one of these explicit contracts:

- preferred: preserve raw action values for AR tokenization and pass them to official `encode_actions(..., statistics=action_stat_path)`;
- fallback: use StarVLA-normalized actions and call `ActionTokenizer` directly without applying `statistics.yaml` again.

Smoke tests must print a small action-tokenization diagnostic:

```text
raw_action_available=true|false
action_token_source=raw_with_reconvla_statistics|starvla_normalized_direct
action_labels_count=35
decoded_action_min=...
decoded_action_max=...
```

## Robot State

The AR training path requires raw 15-D CALVIN `robot_obs`. It should accept:

```python
example["robot_obs"]
```

or:

```python
example["uamvla_raw_state"]["robot_obs"]
```

If only the normalized packed StarVLA state is available, this mode should raise a `RuntimeError`. Training with normalized 7-D GR00T state would not match official ReconVLA.

## LoRA Scope

The baseline LoRA scope should be:

- language backbone attention and MLP projections
- `mm_projector`
- `mm_inv_projector`

Vision tower and pixel decoder should remain frozen unless a later ablation explicitly changes that. This keeps the experiment parameter-efficient while allowing both the visual-to-language bridge and language-to-reconstruction bridge to adapt.

## Checkpointing And Eval

Saved checkpoints should contain the full StarVLA model state, including LoRA parameters and any trainable projector adapters. Eval should use the already validated diagnostic inference path:

```yaml
framework:
  reconvla:
    inference_mode: reconvla_ar_normalized
    ar_input_mode: official_compose
```

The websocket client should continue returning `normalized_actions`, then apply StarVLA `dataset_statistics.json` unnormalization.

## Error Handling

- Missing `action_stat_path`: raise `FileNotFoundError`.
- Missing raw 15-D `robot_obs`: raise `RuntimeError` naming the expected fields.
- Missing target image while `reconvla_ar_recon` is active: raise `RuntimeError`.
- Missing internal ReconVLA recon modules when recon is required: raise `RuntimeError`.
- Action labels not equal to `action_horizon * action_dim`: raise `RuntimeError`.
- NaN/Inf `lm_loss` or `vm_loss`: fail the smoke test.

## Testing

Add a no-save smoke test for `reconvla_ar_recon` that runs one batch and verifies:

- dataset mixture is `uamvla_calvin_abc`
- `action_horizon == 5`
- LoRA is enabled
- GR00T action head has zero trainable parameters or receives no gradients
- base backbone parameters are frozen except LoRA
- `mm_projector` LoRA and `mm_inv_projector` LoRA receive gradients
- raw 15-D robot state is tokenized into the prompt
- labels contain exactly 35 non-ignored action tokens
- `lm_loss`, `vm_loss`, and total loss are finite

Keep static unit tests GPU-light where possible by mocking tokenization/model calls. Use the no-save probe for the real CUDA path.

## Acceptance Criteria

- Existing GR00T training and eval behavior is unchanged by default.
- `training_mode=reconvla_ar_recon` trains with official AR action-token CE plus official `vm_loss`.
- `uamvla_calvin_abc` uses 5-step chunks and produces 35 action labels.
- Raw 15-D `robot_obs` is required and injected into the prompt.
- StarVLA `aux_heads.recon` remains disabled for this official-recipe run.
- A no-save smoke test proves loss, shape, label, and gradient behavior before long training.
