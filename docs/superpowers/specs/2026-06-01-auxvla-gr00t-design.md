# AuxVLAGR00T Design

Date: 2026-06-01
Status: approved design, awaiting implementation plan
Scope: add an isolated `AuxVLAGR00T` framework that uses the ReconVLA
Ross-Qwen2 checkpoint as the main VLA backbone, then connects it to the
existing GR00T action head and UamVLA auxiliary heads.

## 1. Purpose

`ckpt/pretrain-checkpoint-10388` is a ReconVLA pretrained checkpoint from
`third_party/ReconVLA`. Its backbone has been trained with reconstructive
supervision, so it is a plausible starting point for a VLA model whose
auxiliary branches also include reconstruction-like objectives.

The goal is to test a stronger route than partial initialization:

```text
ReconVLA / Ross-Qwen2 pretrained backbone
        |
GR00T flow-matching action head
        |
optional UamVLA auxiliary denoising heads
```

This should be implemented as a separate framework named `AuxVLAGR00T`, not as
a modification of the existing `UamVLAGR00T` Qwen3-VL path.

## 2. Design Constraints

The implementation must preserve the current training environment:

- Do not modify `UamVLAGR00T` behavior.
- Do not change existing Qwen3-VL configs as part of this work.
- Do not start training or run `torchrun` during implementation review.
- Do not run GPU model-load or forward smoke tests unless the user explicitly
  assigns a GPU.
- Prefer CPU/static checks for import, config, and registry validation.
- If a GPU test is later approved, first inspect GPU usage and restrict the
  test with `CUDA_VISIBLE_DEVICES=<user-approved-id>`.

## 3. Current Code Anchors

The current repository already has most of the pieces needed:

- `starVLA/model/framework/base_framework.py`
  auto-imports framework modules under `starVLA/model/framework/*`, so a new
  file under `VLM4A/` can register itself with `FRAMEWORK_REGISTRY`.
- `starVLA/model/framework/VLM4A/UamVLAGR00T.py`
  shows how to combine a VLM hidden-state encoder with the GR00T action head
  and UamVLA aux-head helpers without calling the full `UamVLAOFT.__init__`.
- `starVLA/model/framework/VLM4A/UamVLAOFT.py`
  provides sidecar loading, aux-head construction, aux collation, masks, and
  visualization helpers.
- `starVLA/model/modules/action_model/GR00T_ActionHeader.py`
  supports arbitrary VLM hidden size through
  `framework.action_model.diffusion_model_cfg.cross_attention_dim`.
- `third_party/ReconVLA/reconvla/recon/model/language_model/recon_qwen.py`
  defines `ReconQwen2ForCausalLM`, the Ross-Qwen2 model class used by the
  checkpoint.
- `third_party/ReconVLA/reconvla/recon/model/recon_arch.py`
  implements the `<image>` placeholder replacement and returns `boi_ids` /
  `eoi_ids` for the visual embedding span.

## 4. Architecture

Add:

```text
starVLA/model/framework/VLM4A/AuxVLAGR00T.py
```

Register:

```python
@FRAMEWORK_REGISTRY.register("AuxVLAGR00T")
class AuxVLAGR00T(...):
    ...
```

The framework owns:

```text
self.qwen_vl_interface     # ReconVLA/Ross-Qwen2 wrapper
self.action_model          # existing GR00T flow-matching action head
self.aux_heads             # existing UamVLA aux-head ModuleDict
self.aux_suite             # existing AuxDenoisingSuite when configured
```

The `qwen_vl_interface` should be a thin local wrapper around
`ReconQwen2ForCausalLM` and the ReconVLA tokenizer/image processor. It should
match the small interface that `UamVLAGR00T` expects:

```python
build_qwenvl_inputs(images, instructions, solutions=None) -> dict
forward(**inputs, output_hidden_states=True, return_dict=True) -> output
```

The wrapper should keep ReconVLA internal reconstruction loss disabled in the
first implementation:

```python
model.config.recon_enable = False
model.config.reconstruct_image = False
```

This preserves the pretrained representational bias while keeping the training
objective attributable to the GR00T action loss and the external UamVLA aux
heads.

## 5. Single-View Input Policy

The first version should prioritize matching the released ReconVLA checkpoint.
Use one image stream by default:

```yaml
framework:
  reconvla:
    single_view_mode: primary
```

Supported design values:

- `primary`: use `example["image"][0]`.
- `wrist`: use `example["image"][1]` if present.
- `concat_horizontal`: reserved for a later ablation; not required in the
  first implementation.

The default should be `primary`. This avoids changing ReconVLA's expected
single-image embedding layout and gives the cleanest baseline for the
checkpoint.

## 6. ReconVLA Input Construction

Do not use Qwen3 `AutoProcessor.apply_chat_template`. Use ReconVLA/LLaVA-style
inputs:

```text
prompt = "<image>\n{instruction}"
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
images = ReconVLA image_processor(primary_image)
```

The wrapper returns tensors compatible with `ReconQwen2ForCausalLM.forward`:

```text
input_ids
attention_mask
images
image_sizes
```

`ReconQwen2ForCausalLM.forward` replaces the `<image>` placeholder with the
SigLIP visual embedding span. It produces:

```text
hidden_states[-1]: [B, L', 3584]
boi_ids/eoi_ids:   visual token span per sample
```

The checkpoint config declares:

```text
hidden_size = 3584
image_embed_len = 729
```

So spatial aux heads should treat the visual tokens as a `27 x 27` grid.

## 7. Auxiliary Token Adaptation

Existing UamVLA aux heads locate visual tokens through:

```python
input_ids == image_token_id
```

ReconVLA does not keep repeated image-token ids after multimodal embedding
replacement. It exposes the visual span with `boi_ids` and `eoi_ids`.

To reuse existing aux heads without broad edits, `AuxVLAGR00T` should create a
synthetic aux input-id tensor:

```text
aux_input_ids: [B, L']
visual span:   synthetic_image_token_id
other tokens:  0
```

Then the current `slice_image_tokens(...)` utility can keep working:

```python
slice_image_tokens(
    hidden_states,
    aux_input_ids,
    synthetic_image_token_id,
    patches_per_view=729,
    view_idx=0,
)
```

This adapter localizes ReconVLA-specific token handling inside
`AuxVLAGR00T`.

## 8. Training Objective

First implementation objective:

```text
L_total = L_action_gr00t + L_aux_external
```

Where:

- `L_action_gr00t` is produced by the existing GR00T flow-matching action head.
- `L_aux_external` is produced by the existing UamVLA `AuxDenoisingSuite`.
- ReconVLA internal `vm_loss` is disabled.

Aux heads remain optional and gated by config:

```yaml
framework:
  aux_heads:
    recon:
      enabled: false
```

The first smoke configuration should keep aux heads off. Once action training
runs without shape or loading errors, enable `recon` first, then other aux
heads in controlled ablations.

## 9. Checkpoint Loading

`AuxVLAGR00T` should not use the current trainer-level
`trainer.pretrained_checkpoint` path for the ReconVLA backbone. That path is
intended for starVLA single-file checkpoints or safetensors files.

Use a framework-level path instead:

```yaml
framework:
  reconvla:
    model_path: ckpt/pretrain-checkpoint-10388
    attn_implementation: flash_attention_2
    synthetic_image_token_id: -200
    disable_internal_recon_loss: true
```

The wrapper should load:

```python
ReconQwen2ForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_implementation,
    ignore_mismatched_sizes=False,
)
AutoTokenizer.from_pretrained(model_path, use_fast=False)
```

The checkpoint directory already contains sharded safetensors and the required
tokenizer/config files.

## 10. Freeze and LoRA Strategy

Use staged training:

1. Smoke test:
   - freeze ReconVLA backbone;
   - train only GR00T action head;
   - aux heads off.
2. Action adaptation:
   - enable LoRA on Ross-Qwen2 language backbone linear layers;
   - train GR00T action head;
   - aux heads off.
3. Auxiliary adaptation:
   - keep LoRA enabled;
   - train GR00T action head and selected aux heads;
   - enable `recon` first, then add other heads by ablation.

Recommended first freeze policy:

```text
Trainable:
  - action_model
  - aux_heads when enabled
  - LoRA parameters when enabled

Frozen:
  - vision_tower
  - pixel_decoder / VAE
  - base Ross-Qwen2 weights except LoRA
```

The initial implementation may only define config fields and parameter-group
hooks for LoRA. Full LoRA wiring can be implemented in a follow-up step if it
requires broader trainer changes.

## 11. Configuration File

Add a new config rather than modifying the existing UamVLA config:

```text
starVLA/config/training/auxvla_gr00t_libero.yaml
```

It should start from the current LIBERO GR00T settings but change:

```yaml
run_id: auxvla_gr00t_libero_primary_h8

framework:
  name: AuxVLAGR00T
  reconvla:
    model_path: ckpt/pretrain-checkpoint-10388
    attn_implementation: flash_attention_2
    single_view_mode: primary
    synthetic_image_token_id: -200
    disable_internal_recon_loss: true
```

`datasets.vla_data.obs` may remain two-view for compatibility with existing
data loading, but `AuxVLAGR00T` should consume only the configured single view
in the first version.

## 12. Verification Plan

No GPU work should run by default.

Allowed without further approval:

- static imports;
- registry lookup;
- config parse checks;
- CPU-only path checks;
- checkpoint file existence checks.

Requires explicit user approval:

- loading `ReconQwen2ForCausalLM` on GPU;
- any forward pass;
- any `torchrun` launch;
- any training or evaluation script.

If a GPU smoke test is approved later:

1. run `nvidia-smi` first;
2. use only a user-selected GPU;
3. set `CUDA_VISIBLE_DEVICES` to that single GPU;
4. use one sample and no optimizer step unless separately approved;
5. terminate the process after the shape check.

## 13. Success Criteria

The first implementation is complete when:

- `framework.name: AuxVLAGR00T` builds through the existing registry.
- The new config parses without touching the existing `UamVLAGR00T` config.
- The ReconVLA checkpoint path is configured at framework level.
- The forward path has a clear adapter from ReconVLA `boi_ids/eoi_ids` to aux
  synthetic image-token ids.
- The GR00T action head receives `[B, L', 3584]` hidden states.
- Existing UamVLA aux heads can be enabled without modifying their public
  interface.
- No training process is launched during implementation verification.

