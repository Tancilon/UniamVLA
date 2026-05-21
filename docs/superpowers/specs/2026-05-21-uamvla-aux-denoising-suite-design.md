# UamVLA Auxiliary Denoising Suite Design

Date: 2026-05-21
Status: approved design, awaiting implementation plan
Supersedes: `docs/superpowers/specs/2026-05-20-uamvla-aux-branches-design.md`

## 1. Purpose

This document turns the 2026-05-20 auxiliary branch discussion draft into a
complete design for a paper-ready UamVLA auxiliary-denoising framework. The
goal is not to add temporary heads for one experiment. The goal is to define a
unified, pluggable auxiliary learning suite whose instances can be used as core
experimental branches and described cleanly in the paper.

UamVLA is framed as a unified auxiliary denoising framework:

```text
RGB observations + language + optional proprio state
        |
Qwen-VL hidden states
        |
main GR00T action objective
        |
AuxDenoisingSuite
        |
task-relevant auxiliary variables are denoised or score-matched
```

All auxiliary labels are training-time supervision only. Evaluation and
`predict_action()` must continue to consume the same inputs as the baseline
policy: RGB views, language, and any existing proprio/state branch used by
`UamVLAGR00T`. Depth, masks, affordance maps, future RGB targets, pose labels,
and reconstruction sidecars must not become inference-time sensors.

## 2. Design Decisions

The approved design makes these decisions:

1. Use a unified framework plus pluggable task variables. Existing `recon`,
   terminal `future`, and `pose` heads plus the new `depth`,
   `action_conditioned_future`, `grounding`, and `affordance` heads are all
   representative framework instances.
2. In experiments, implement all listed branches as complete core branches and
   use leave-one-branch-out ablations.
3. Split the objective family by variable type:
   - visual latent and token-grid spatial variables use DDPM-style
     epsilon-prediction denoising;
   - 6D pose keeps the existing pose-space score matching formulation.
4. Use static view only for the new four heads. The default `view_idx` is `0`.
5. Represent `depth`, `grounding`, and `affordance` as 20x20 token-grid
   single-channel targets aligned to Qwen image tokens.
6. Store dense sidecars in raw or `[0, 1]` form. Heads map `[0, 1]` targets to
   `[-1, 1]` internally before diffusion training.
7. Use per-frame robust relative inverse depth for the depth target.
8. Define grounding hierarchically: prefer manipulable part masks; fall back to
   target object masks; record `grounding_level`.
9. Define affordance as the target point-cloud point closest to the future TCP
   trajectory, projected into the static camera and rendered as a 20x20 Gaussian
   heatmap.
10. Keep terminal `future` and `action_conditioned_future` as separate heads.
    Terminal `future` predicts goal/episode-end appearance. The new
    `action_conditioned_future` predicts the local RGB consequence of executing
    the current action chunk.
11. Condition `action_conditioned_future` on the normalized training action
    chunk, not on unnormalized physical actions.
12. Fuse action and spatial conditions through FiLM modulation.
13. Encode the action chunk with a 1-2 layer TransformerEncoder over action
    steps and enable action dropout.
14. Use a global auxiliary loss budget with warmup and an EMA action-loss ratio
    cap so auxiliary losses serve the action objective rather than dominate it.
15. Mask missing labels per sample. Missing sidecars do not fail training; they
    produce `<head>_mask=False` and are visible through valid-ratio metrics.
16. Use GT-vs-pred visualization as the required visualization standard.

## 3. Current Code Anchors

The design builds on the existing UamVLA/GR00T integration:

- `starVLA/model/framework/VLM4A/UamVLAGR00T.py`
  - `_encode_qwen_hidden()` returns prepared examples, Qwen inputs, and
    `hidden_states[-1]`.
  - `forward()` computes the GR00T flow-matching action loss, then runs the
    auxiliary heads.
  - `predict_action()` is the inference path and must remain independent from
    auxiliary sidecars.

- `starVLA/model/framework/VLM4A/UamVLAOFT.py`
  - `_maybe_build_aux_heads()` is the current aux-head construction point.
  - `_unpack_lerobot_sample()` is the natural place to load sidecar labels.
  - `_collate_aux()` stacks optional sidecar fields and creates masks.
  - `_resolve_head_mask()` already implements per-head mask resolution.

- `starVLA/model/modules/uamvla/aux_heads/`
  - `ReconHead` and `FutureHead` already implement VAE-latent denoising
    conditioned on spatial Qwen image-token hidden states.
  - `PoseHead` already implements the pose-space score matching branch.
  - All heads implement the `AuxHead` interface and return dummy zero loss when
    masks are empty.

- `starVLA/model/modules/uamvla/components/denoiser/`
  - `ReconDenoiser` and `DiT` provide the DDPM epsilon-prediction path used by
    visual latent denoising and reusable for token-grid spatial maps.

## 4. High-Level Architecture

Introduce an `AuxDenoisingSuite` training component around the existing aux
head loop. It is responsible for cross-head loss control and logging, while
individual heads remain responsible for variable-specific modeling.

```text
RGB + language + optional state
        |
Qwen-VL hidden states
        |
GR00T flow-matching action loss: L_action
        |
AuxDenoisingSuite
        |
recon / future / pose / depth / action_conditioned_future / grounding / affordance
        |
L_aux_pre_budget
        |
EMA action-loss budget + warmup + ratio cap
        |
L_total = L_action + L_aux_post_budget
```

Responsibility split:

- The framework computes the main action loss and builds `batch_dict`.
- Each head computes:
  - `raw_loss`;
  - `weighted_loss_pre_budget = head.loss_weight * raw_loss`;
  - `valid_ratio`.
- `AuxDenoisingSuite` computes:
  - `aux_total_pre_budget`;
  - `action_loss_ema`;
  - warmup scale;
  - global budget scale;
  - post-budget contribution per head;
  - `aux_total_post_budget`.
- The inference path does not call `AuxDenoisingSuite`.

This keeps the paper story and code structure aligned: UamVLA instantiates a
family of plug-in denoising objectives over task-relevant latent, spatial, and
geometric variables, while action prediction remains the only inference-time
objective.

## 5. Variable Families

### 5.1 Visual Latent Denoising

Branches:

- `recon`: reconstruct target object/region appearance.
- `future`: predict terminal or goal frame appearance.
- `action_conditioned_future`: predict the local future frame after executing
  the current action chunk.

These heads operate on VAE latents and use DDPM epsilon-prediction denoising.

### 5.2 Token-Grid Spatial Denoising

Branches:

- `depth`
- `grounding`
- `affordance`

These heads operate on static-view 20x20 single-channel maps aligned to Qwen
image tokens. They share a common `SpatialMapDenoisingHead` implementation.

### 5.3 Geometric Score Matching

Branch:

- `pose`

This branch keeps the existing 6D pose score-matching formulation and is
treated as the geometric member of the same auxiliary-denoising family.

## 6. Shared Spatial Conditioning

All spatial and visual-latent heads read Qwen image-token hidden states:

```text
hidden_states[-1]: [B, L, H]
input_ids:         [B, L]
        |
slice_image_tokens(image_token_id, patches_per_view=400, view_idx=0)
        |
image_tokens:      [B, 400, H]
        |
LayerNorm over H
        |
spatial_cond:      [B, H, 20, 20]
```

The shared rules are:

- use static view only in the approved default design;
- use `batch["input_ids"]` only to locate image-token positions;
- use `slice_image_tokens(...)` for consistency with existing heads;
- reshape with `h = w = sqrt(patches_per_view) = 20`;
- return dummy zero loss when no valid samples are present.

## 7. SpatialMapDenoisingHead

`depth`, `grounding`, and `affordance` share a base head:

```text
SpatialMapDenoisingHead
  - target_key
  - mask_key
  - metric_prefix
  - view_idx = 0
  - map_size = 20
  - x_channel = 1
  - target_range = [0, 1]
  - diffusion_range = [-1, 1]
  - loss_weight
  - denoiser
```

Training:

```text
target_01: [B, 1, 20, 20] in [0, 1]
target_x0 = target_01 * 2 - 1
noise, timestep sampled by DDPM
x_t = q_sample(target_x0, timestep, noise)
denoiser(spatial_cond, x_t, timestep) predicts noise
loss = epsilon MSE
```

Sidecars stay raw or `[0, 1]`; the head does the `[0, 1] -> [-1, 1]`
conversion internally. Predictions are mapped back by `(pred + 1) / 2` for
visualization.

## 8. DepthDenoisingHead

### Purpose

Depth supervision teaches the hidden state to preserve dense geometric layout:
near/far structure, object boundaries, tabletop geometry, and approximate 3D
relations. It does not turn the policy into an RGB-D policy because depth is
training-time supervision only.

### Target

The sidecar stores raw metric depth when available. The head converts it to
per-frame robust relative inverse depth:

```text
valid = finite(depth) and depth > 0
inverse_depth = 1 / clamp(depth, eps, max_depth)
lo = percentile(inverse_depth[valid], 1)
hi = percentile(inverse_depth[valid], 99)
relative_inverse_depth = clip((inverse_depth - lo) / (hi - lo), 0, 1)
```

The resulting convention is:

```text
1 = near
0 = far
```

After resizing/downsampling to 20x20, `SpatialMapDenoisingHead` maps it to
`[-1, 1]` and trains an epsilon-prediction DDPM objective.

### Config

```yaml
depth:
  enabled: false
  loss_weight: 0.1
  view_idx: 0
  target_kind: relative_inverse_depth_per_frame
  target_size: 20
  denoiser_depth: 3
  denoiser_embed_dim: 512
  gen_timesteps: "1000"
```

## 9. GroundingMaskDenoisingHead

### Purpose

Grounding teaches the hidden state where the instruction-relevant entity is in
the full static observation. It is distinct from reconstruction:

```text
recon:     what the target looks like
grounding: where the target or manipulated part is
```

### Target

The grounding target is a single-channel 20x20 soft mask in `[0, 1]`.

The semantic level is hierarchical:

1. Use manipulable part mask when reliable part labels are available.
2. Fall back to target object mask when part labels are unavailable or
   unreliable.
3. Record `grounding_level` as `part` or `object`.

Examples:

```text
move red block    -> red block mask
open drawer       -> drawer handle/front if available, otherwise drawer
press button      -> button surface
turn on lightbulb -> switch/button
```

The head does not use `grounding_level` for the denoising loss by default; it is
metadata for logging, analysis, and paper reporting.

### Config

```yaml
grounding:
  enabled: false
  loss_weight: 0.1
  view_idx: 0
  target_size: 20
  denoiser_depth: 3
  denoiser_embed_dim: 512
```

## 10. AffordanceHeatmapDenoisingHead

### Purpose

Affordance teaches where the robot should interact, not just where the target
is. It complements grounding:

```text
grounding:  target or part location
affordance: actionable interaction point on the target
```

### Target

The affordance target is a 20x20 Gaussian heatmap in `[0, 1]`.

The approved source of the heatmap center is the target point-cloud point
closest to the future TCP trajectory:

```text
1. recover future TCP positions from robot state/action rollout
2. recover target object point cloud
3. find target point closest to the future TCP trajectory
4. project that 3D point into the static camera
5. render a Gaussian heatmap centered at the projected point
6. downsample or rasterize to 20x20
```

This is intentionally stronger than directly projecting the TCP trajectory. It
ties affordance to an actionable point on the target object.

### Config

```yaml
affordance:
  enabled: false
  loss_weight: 0.1
  view_idx: 0
  target_size: 20
  heatmap_sigma: 2.0
  denoiser_depth: 3
  denoiser_embed_dim: 512
```

## 11. ActionConditionedFutureHead

### Purpose

The existing `FutureHead` predicts terminal or goal appearance. The new
`ActionConditionedFutureHead` predicts the local visual consequence of the
demonstrated action chunk:

```text
terminal future:           what the task goal/episode end should look like
action-conditioned future: what the scene should look like after this action chunk
```

Both heads coexist because they supervise different semantics.

### Target

The target frame is the static RGB frame at:

```text
target_frame_index = base_index + future_action_window_size
```

For the current GR00T setting:

```text
action_horizon = 8
future_action_window_size = 7
```

This aligns the target with the end of the supervised action chunk.

The target is loaded from LeRobot `video.primary_image`; no new sidecar is
required.

### Action Condition

The action condition is the normalized action chunk already present in the
training batch. It uses the same normalized action tensor as the GR00T action
loss.

The `ActionChunkEncoder` is:

```text
action_chunk: [B, action_horizon, action_dim]
        |
Linear(action_dim -> action_embed_dim)
        |
step positional embedding
        |
1-2 layer TransformerEncoder over action steps
        |
mean or CLS pooling
        |
Linear(action_embed_dim -> hidden_size)
        |
action_emb: [B, hidden_size]
```

### FiLM Fusion

The action embedding modulates the static-view spatial condition:

```text
spatial_cond: [B, H, 20, 20]
action_emb:   [B, H]

scale, shift = Linear(action_emb).chunk(2)
fused_cond = spatial_cond * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
```

The fused condition is passed to the same VAE-latent denoising pattern as
`FutureHead`.

### Action Dropout

Training uses action dropout, for example `p=0.1`, by zeroing the action
embedding before FiLM for a random subset of samples. This reduces the shortcut
risk that the denoiser becomes action-only and ignores visual hidden states.

### Config

```yaml
action_conditioned_future:
  enabled: false
  loss_weight: 1.0
  view_idx: 0
  target_resize: 320
  denoiser_depth: 3
  denoiser_embed_dim: 1024
  repeat_factor: 4
  action_encoder_layers: 2
  action_embed_dim: 512
  action_dropout: 0.1
  target_frame_offset: future_action_window_size
```

## 12. Sidecar Layout And Batch Contract

### 12.1 Unified Sidecar Layout

The new dense sidecars use one strict physical layout across CALVIN and LIBERO:

```text
depths/static/<trajectory_id>/<base_index>.npy
grounding_masks/static/<trajectory_id>/<base_index>.npy
affordance_heatmaps/static/<trajectory_id>/<base_index>.npy
```

Storage conventions:

- `depths` stores raw metric depth when available.
- `grounding_masks` stores binary or soft `[0, 1]` masks.
- `affordance_heatmaps` stores `[0, 1]` Gaussian heatmaps.
- no wrist sidecars are part of the approved default design.
- no action-conditioned future sidecar is stored; it is read from video.

### 12.2 Framework Sample Fields

`_unpack_lerobot_sample()` should expose:

```python
out["depth_target"] = Tensor[1, H, W]
out["grounding_mask"] = Tensor[1, 20, 20]
out["grounding_level"] = "part" | "object"
out["affordance_heatmap"] = Tensor[1, 20, 20]
out["image_action_future"] = Tensor[3, H, W]
out["action"] = Tensor[action_horizon, action_dim]
```

The depth head may accept full-resolution raw depth and perform resizing and
relative-depth conversion internally. Grounding and affordance are expected to
arrive as 20x20 maps unless a preprocessing path stores higher resolution maps
and explicitly configures a resize adapter.

### 12.3 Batch Fields

`_collate_aux()` should expose:

```python
batch["input_ids"]
batch["action"]

batch["depth_target"]
batch["depth_mask"]

batch["grounding_mask"]
batch["grounding_mask_mask"]
batch["grounding_level"]

batch["affordance_heatmap"]
batch["affordance_mask"]

batch["image_action_future"]
batch["action_conditioned_future_mask"]
```

Existing fields remain:

```python
batch["image_target"]
batch["recon_mask"]
batch["image_future"]
batch["future_mask"]
batch["pose_gt"]
batch["pose_mask"]
batch["point_cloud"]
batch["static_cam_extrinsic"]
```

### 12.4 Missing Label Policy

Missing labels are masked per sample. If a sample lacks a sidecar or future
frame target, its head mask is `False`. If an entire batch lacks valid samples
for a head, that head returns `get_dummy_loss()` so distributed training stays
well-defined.

The valid-ratio metrics make missing labels visible in logs.

## 13. Aux Loss Control

The total training objective is:

```text
L_action = GR00T flow-matching action loss
L_aux_pre_budget = sum_i w_i * L_i_raw
L_total = L_action + warmup(t) * aux_budget * aux_scale * L_aux_pre_budget
```

`aux_scale` is controlled by an EMA action-loss budget:

```text
action_loss_ema <- beta * action_loss_ema + (1 - beta) * detach(L_action)
aux_cap = aux_ratio_cap * action_loss_ema
aux_scale = min(1, aux_cap / (detach(L_aux_pre_budget) + eps))
```

This ensures auxiliary losses remain subordinate to action learning.

Default config:

```yaml
aux_loss_control:
  enabled: true
  aux_budget: 1.0
  warmup_steps: 2000
  aux_ratio_cap: 0.5
  action_loss_ema_beta: 0.99
  eps: 1.0e-8
```

Default variable-family weights:

```yaml
aux_heads:
  recon:
    loss_weight: 1.0
  future:
    loss_weight: 1.0
  action_conditioned_future:
    loss_weight: 1.0
  pose:
    loss_weight: 0.5
  depth:
    loss_weight: 0.1
  grounding:
    loss_weight: 0.1
  affordance:
    loss_weight: 0.1
```

## 14. Logging

Each head must log:

```text
<head>_loss_raw
<head>_loss_weighted_pre_budget
<head>_loss_contribution_post_budget
<head>_valid_ratio
```

The suite must log:

```text
aux_total_pre_budget
aux_total_post_budget
aux_scale_budget
aux_budget_warmup
action_loss_ema
aux_cap
```

These metrics separate four questions:

1. Is the head learning its variable?
2. How large is the head inside the auxiliary sum?
3. How much does it actually influence the total loss after global budget
   control?
4. How much valid supervision did this batch contain?

## 15. Visualization

The required visualization standard is GT vs predicted target:

```text
depth:                    GT relative depth vs predicted relative depth
grounding:                GT mask vs predicted mask
affordance:               GT heatmap vs predicted heatmap
action_conditioned_future: GT local future RGB vs predicted RGB
```

For 20x20 maps, prediction and target may be upsampled for readability. Error
maps and overlays are useful extensions but are not required by this design.

Visualization is a training-diagnostics path only and must not affect
`predict_action()`.

## 16. Experiment Plan

The main experiment can enable the complete auxiliary suite:

```text
action + recon + terminal future + pose + depth
       + action_conditioned_future + grounding + affordance
```

Core leave-one-branch-out ablations:

```text
Full
- recon
- terminal future
- pose
- depth
- action_conditioned_future
- grounding
- affordance
```

Additional `action_conditioned_future` ablations:

```text
hidden-only local future
action-only local future
hidden + action local future
terminal future vs action-aligned local future
with action dropout vs without action dropout
```

Additional spatial-supervision reporting:

```text
grounding part/object label coverage
affordance valid ratio
depth valid ratio
aux budget scale statistics
```

The paper should present the method as a unified framework with representative
task-variable instantiations, while experiments treat all implemented branches
as complete core branches and analyze their individual contributions.

## 17. Implementation Units

New files:

```text
starVLA/model/modules/uamvla/aux_heads/spatial_map_denoising_head.py
starVLA/model/modules/uamvla/aux_heads/depth_head.py
starVLA/model/modules/uamvla/aux_heads/grounding_head.py
starVLA/model/modules/uamvla/aux_heads/affordance_head.py
starVLA/model/modules/uamvla/aux_heads/action_conditioned_future_head.py
starVLA/model/modules/uamvla/components/action_chunk_encoder.py
starVLA/model/modules/uamvla/aux_loss_control.py
```

Framework edits:

```text
starVLA/model/framework/VLM4A/UamVLAOFT.py
starVLA/model/framework/VLM4A/UamVLAGR00T.py
starVLA/model/modules/uamvla/collator_helpers.py
starVLA/config/training/uamvla_gr00t_calvin_d.yaml
```

Preprocessing edits:

```text
tools/preprocess/calvin_preprocessor_lerobot.py
tools/preprocess/libero_preprocessor.py
```

The implementation keeps the current `AuxHead.compute_loss(...) -> HeadOutput`
interface. `HeadOutput.loss` is defined as the head's
`weighted_loss_pre_budget`. Each head reports its raw loss and valid ratio in
`HeadOutput.metrics` using these names:

```text
loss_raw
valid_ratio
```

`AuxDenoisingSuite` is the only component that applies global warmup and budget
scaling. It converts each head's `HeadOutput.loss` into
`<head>_loss_contribution_post_budget` after computing the shared
`aux_scale_budget`.

## 18. Test Plan

Required test coverage:

1. Config constructs each new head and the `AuxDenoisingSuite`.
2. `_unpack_lerobot_sample()` loads each new sidecar when present.
3. `_collate_aux()` creates the expected batch keys and masks.
4. `SpatialMapDenoisingHead` handles valid masks and empty masks.
5. `DepthDenoisingHead` converts raw depth to per-frame robust relative
   inverse depth.
6. `GroundingMaskDenoisingHead` accepts part/object masks and preserves
   `grounding_level` metadata.
7. `AffordanceHeatmapDenoisingHead` consumes 20x20 heatmaps and logs valid
   ratio.
8. `ActionConditionedFutureHead` produces correctly shaped losses and applies
   action dropout only in training.
9. `ActionChunkEncoder` handles `[B, H, action_dim]` normalized action chunks.
10. `AuxDenoisingSuite` applies warmup, EMA action-loss budget, ratio cap, and
    post-budget contribution logging.
11. `predict_action()` remains unchanged when aux sidecars are absent.
12. Visualization returns GT-vs-pred artifacts for targets that exist.

## 19. Non-Goals

This design does not introduce inference-time depth, segmentation, affordance,
future-frame, or pose inputs.

This design does not require wrist-view auxiliary supervision for the four new
heads.

This design does not require automatic loss balancing such as GradNorm or
learned uncertainty weighting. The approved loss-control mechanism is global
auxiliary budget plus warmup plus EMA action-loss ratio cap.

This design does not require high-resolution map denoising. The approved
spatial-map resolution is the 20x20 visual-token grid.
