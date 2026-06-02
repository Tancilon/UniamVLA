# AuxVLAGR00T LoRA Design

## Purpose

Add a conservative LoRA baseline for `AuxVLAGR00T`: train the GR00T action
head plus LoRA adapters on the ReconVLA/Ross-Qwen2 language backbone, while
keeping the SigLIP vision tower and ReconVLA multimodal projectors frozen.

This baseline follows the already-validated AuxVLAGR00T smoke path and keeps
the first LoRA experiment easy to attribute:

- `B0`: frozen ReconVLA backbone + GR00T action head.
- `B1`: ReconVLA language-backbone LoRA + GR00T action head.

The first version intentionally avoids aux heads, vision tower LoRA,
projector finetuning, adapter-only checkpointing, and QLoRA.

## Current Context

`AuxVLAGR00T` is an isolated framework under
`starVLA/model/framework/VLM4A/AuxVLAGR00T.py`. It wraps the ReconVLA
checkpoint through `ReconVLAInterface`, feeds the final hidden states into the
existing GR00T action head, and delegates UamVLA sidecar utilities for optional
auxiliary heads.

The current training path has already passed:

- no-save smoke on LIBERO/CALVIN data;
- single-GPU short trainer validation;
- 8-GPU DeepSpeed short trainer validation.

The repository does not currently contain PEFT/LoRA integration code. The
Python environment has `peft` available, so the integration can use PEFT
directly.

## Selected Approach

Implement LoRA inside `AuxVLAGR00T`, not in the generic trainer and not as a
new framework class.

Rejected alternatives:

- Trainer-level LoRA injection would require the trainer to understand each
  framework's backbone topology, which blurs boundaries.
- A separate `AuxVLAGR00TLoRA` class would duplicate the existing
  ReconVLA/action-head/aux-head logic and make future fixes harder to keep in
  sync.

The selected design keeps LoRA local to the ReconVLA-backed framework and
preserves the existing frozen-backbone behavior when LoRA is disabled.

## Configuration

Add a LoRA block under `framework.reconvla`:

```yaml
framework:
  reconvla:
    lora:
      enabled: false
      r: 16
      lora_alpha: 32
      lora_dropout: 0.05
      bias: none
      task_type: CAUSAL_LM
      target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
        - gate_proj
        - up_proj
        - down_proj
```

For the LoRA baseline config, set `enabled: true`. The existing
`auxvla_gr00t_libero.yaml` can remain the frozen/action-head baseline config;
a separate LoRA config is preferred to keep experiments distinct.

## Architecture

### ReconVLAInterface

`ReconVLAInterface` remains responsible for loading:

- tokenizer;
- `ReconQwen2ForCausalLM`;
- SigLIP vision tower;
- ReconVLA image processor and visual span metadata.

It gains a LoRA application method, conceptually:

```python
apply_language_lora(lora_cfg) -> None
```

This method applies PEFT LoRA to the loaded ReconVLA model and verifies that
LoRA trainable parameters exist. It does not modify the GR00T action head or
aux heads.

### AuxVLAGR00T

Initialization order:

1. Load ReconVLA through `ReconVLAInterface`.
2. If `framework.reconvla.lora.enabled` is true, apply LoRA.
3. Explicitly freeze non-target multimodal components:
   - vision tower;
   - `mm_projector`;
   - `mm_inv_projector`.
4. Build the GR00T action head.
5. Build optional aux heads.

`trainer.freeze_modules` must not include `qwen_vl_interface` for LoRA
training, because that would freeze PEFT adapters after they are created.

## Trainable Parameter Policy

When LoRA is disabled:

- current behavior is preserved;
- frozen-backbone training still uses `--trainer.freeze_modules qwen_vl_interface`
  when desired.

When LoRA is enabled:

- base ReconVLA/Qwen2 weights are frozen by PEFT;
- LoRA adapter weights are trainable;
- GR00T action head is trainable;
- vision tower is frozen;
- `mm_projector` is frozen;
- `mm_inv_projector` is frozen;
- aux heads are trainable only if enabled in the config.

`AuxVLAGR00T` should define `get_lr_groups(lr_cfg)` so the optimizer receives
only parameters with `requires_grad=True`. This avoids passing the frozen
ReconVLA base weights into DeepSpeed optimizer groups.

Recommended LR grouping:

- `action_model`: `trainer.learning_rate.action_model`;
- LoRA parameters under `qwen_vl_interface`: `trainer.learning_rate.qwen_vl_interface`;
- enabled aux heads or any other trainable fallback parameters:
  `trainer.learning_rate.base`.

## Error Handling

LoRA initialization should fail early for invalid setups:

- If `peft` cannot be imported, raise a clear error telling the user to install
  `peft` or disable LoRA.
- If no target modules are matched by PEFT, raise a clear error instead of
  silently training only the action head.
- If LoRA is enabled but no trainable LoRA parameters exist, raise a clear
  error.
- If LoRA is enabled and `get_lr_groups()` sees no trainable LoRA parameters,
  raise an error suggesting the user remove
  `--trainer.freeze_modules qwen_vl_interface`.

The goal is to avoid runs that appear successful while adapter parameters are
not actually training.

## Validation Plan

### Unit And Static Tests

Add tests covering:

- LoRA config defaults are present.
- Enabling LoRA calls the PEFT wrapper.
- After LoRA application, only LoRA parameters in `qwen_vl_interface` remain
  trainable.
- Vision tower, `mm_projector`, and `mm_inv_projector` parameters are frozen.
- `get_lr_groups()` includes action-head parameters and LoRA parameters, and
  excludes frozen base-backbone parameters.
- LoRA disabled preserves the existing AuxVLAGR00T import and forward tests.

### No-Save Smoke

Extend the existing no-save probe or add a LoRA mode so it can print and assert:

- `lora_trainable_params > 0`;
- `backbone_base_trainable_params == 0`;
- `vision_projector_trainable_params == 0`;
- `action_grad_norm > 0`;
- `lora_grad_norm > 0`;
- frozen base/vision/projector gradients remain absent.

### Trainer Smoke

Run a 2-step trainer smoke on CALVIN with:

- `framework.reconvla.lora.enabled=true`;
- no `--trainer.freeze_modules qwen_vl_interface`;
- aux loss control disabled;
- aux heads disabled;
- 8-GPU DeepSpeed ZeRO-3.

The validation target is correct trainable parameter accounting, forward,
backward, optimizer step, scheduler step, and final save.

### Baseline Training

After smoke tests pass:

- 500-step CALVIN ABC LoRA baseline;
- 5k-step CALVIN ABC LoRA baseline;
- compare against the frozen-backbone action-head baseline.

Suggested run IDs:

- `auxvla_gr00t_lora_calvin_abc_8gpu_s500`;
- `auxvla_gr00t_lora_calvin_abc_8gpu_s5k`.

## Non-Goals

This design does not include:

- LoRA on SigLIP vision tower;
- unfreezing `mm_projector` or `mm_inv_projector`;
- adapter-only checkpoint saving;
- LoRA merging for inference;
- QLoRA or bitsandbytes quantization;
- enabling recon/future/depth/grounding/affordance aux heads.

Those can be added after the B1 baseline is validated.
