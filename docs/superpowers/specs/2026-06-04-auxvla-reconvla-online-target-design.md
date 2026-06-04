# AuxVLAGR00T ReconVLA-Style Online Target Design

Date: 2026-06-04

## Goal

Align AuxVLAGR00T more closely with the ReconVLA CALVIN reconstruction recipe while keeping the GR00T continuous action head unchanged. The change should avoid reprocessing the CALVIN dataset and should construct the ReconVLA-style target image online during training.

The intended behavior is:

- Keep AuxVLAGR00T as the framework.
- Keep the GR00T action head and current action/state path.
- Keep the existing `image_targets/<trajectory>/<base>.png` sidecar crop.
- At training time, replace the crop-only `image_target` with a composed target image:
  - top region: existing target crop
  - bottom region: current wrist/gripper image

This makes the aux reconstruction target closer to ReconVLA's CALVIN `target_image` format without changing dataset files.

## Non-Goals

- Do not replace the GR00T action head with ReconVLA's action-token policy.
- Do not regenerate or rewrite CALVIN LeRobot datasets.
- Do not change UamVLAGR00T, CosmosGR00T, or shared preprocessing behavior.
- Do not enable ReconVLA's internal `compute_vm_loss` path in this step.
- Do not add a config switch for this behavior. AuxVLAGR00T should use the ReconVLA-style target by default.

## Current State

AuxVLAGR00T currently:

- Unpacks LeRobot samples through `UamVLAOFT._unpack_lerobot_sample`.
- Loads `image_target` from the existing sidecar crop.
- Uses `single_view_mode=concat_vertical` for the ReconVLA/Qwen backbone input image.
- Trains the GR00T action head from ReconVLA hidden states.
- Optionally trains an external `ReconHead` using `batch["image_target"]`.

The current external recon target is crop-only. ReconVLA's CALVIN processing instead builds a target image by resizing and pasting the object crop above the gripper image.

## Design

### Data Flow

After AuxVLAGR00T unpacks each sample, `_prepare_examples` will enforce and construct the online target:

1. Require `example["image_target"]`.
2. Require `example["image"]` to contain at least two views.
3. Use `example["image"][1]` as the wrist/gripper image.
4. Compose a new square tensor and store it back in `example["image_target"]`.

The existing `ReconHead` will continue to read `batch["image_target"]`; it does not need a new interface.

### Target Layout

Use a 384 x 384 target image to match the current AuxVLAGR00T/SigLIP training setup.

Use ReconVLA CALVIN's vertical ratio:

- `target_size = 384`
- `crop_height = target_size * 14 // 27`
- `wrist_height = target_size - crop_height`

The top crop is resized to `(384, crop_height)`.
The wrist image is resized to `(384, wrist_height)`.
The result is a `torch.Tensor` with shape `(3, 384, 384)` and values in `[0, 1]`.

### Error Handling

This behavior is mandatory for AuxVLAGR00T. Missing inputs should fail early:

- Missing `image_target`: raise `RuntimeError`.
- Missing wrist image: raise `RuntimeError`.
- Invalid `image_target` tensor shape: raise `RuntimeError`.
- Unconvertible wrist image: raise `RuntimeError`.

There should be no silent fallback to crop-only reconstruction.

### Scope

Only AuxVLAGR00T should change. The implementation should be local to the framework and should not alter:

- preprocessing scripts
- dataset registries
- UamVLAGR00T
- ReconHead public API
- training configs

This keeps the change narrow and reversible while testing the effect of official-style reconstruction targets.

## Testing

Add or update focused tests for AuxVLAGR00T:

- A prepared sample with `image_target` and wrist image produces `image_target.shape == (3, 384, 384)`.
- The top and bottom regions are populated from different sources.
- Missing `image_target` raises `RuntimeError`.
- Missing wrist image raises `RuntimeError`.

Run the existing static AuxVLAGR00T tests and a no-save smoke on the remote server before launching training.

## Expected Experiment

Use the same baseline checkpoint and training setup as the current AuxVLAGR00T LoRA + GR00T runs. Compare:

- crop-only external recon target
- online ReconVLA-style crop-plus-wrist external recon target

The key metric is CALVIN ABC chain success. The diagnostic metrics are recon loss, action loss, visualization quality, and eval stability.

## Risks

- The target distribution changes abruptly from crop-only to crop-plus-wrist, so previous recon loss scale may shift.
- External `ReconHead` is still not identical to ReconVLA's internal `compute_vm_loss`.
- The input image uses current `concat_vertical` behavior, which is 1/2 static and 1/2 wrist, while the target uses the ReconVLA 14/27 and 13/27 ratio.
- If existing remote datasets lack wrist images for some samples, training will fail early by design.

## Acceptance Criteria

- AuxVLAGR00T composes `image_target` online by default.
- Existing sidecar crops and wrist images are sufficient; no dataset rewrite is required.
- Recon training and visualization consume the composed target image.
- Tests cover shape and failure behavior.
- Remote no-save smoke confirms forward, backward, action gradients, LoRA gradients, and recon loss remain healthy.
