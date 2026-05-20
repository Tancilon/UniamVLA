# UamVLAGR00T Auxiliary Branch Expansion Design

Date: 2026-05-20  
Status: discussion draft  
Scope: add four optional training-time auxiliary denoising branches to `UamVLAGR00T`:

1. relative depth denoising
2. action-conditioned dynamics / action-conditioned future denoising
3. grounding mask denoising
4. single-channel affordance heatmap denoising

This document is intended as a durable discussion base for later dataset preprocessing and code implementation work. It does not prescribe final hyperparameters; it fixes the conceptual interfaces, data contracts, and implementation direction.

## 1. Method Positioning

The intended UAMVLA framing is a general auxiliary-denoising framework for VLA training:

```text
RGB observations + language
        |
Qwen / VLM hidden states
        |
main action objective
        |
auxiliary denoising heads read hidden states as conditions
        |
denoise task-relevant perceptual / geometric / dynamic variables
```

The auxiliary heads are training-time only. At evaluation time on CALVIN, LIBERO, and related RGB policy benchmarks, the policy should still consume the same inputs as the baseline policy: RGB views, language, and any existing proprio/state branch used by `UamVLAGR00T`. The new labels are supervision, not inference-time sensors.

This distinction is important for fair comparison: if depth, segmentation, affordance, or future-frame sidecars are used, they must be used only in the training loss and visualization path. `predict_action()` must not require them.

## 2. Current Code Anchors

The current `UamVLAGR00T` implementation already provides the right hook points:

- `starVLA/model/framework/VLM4A/UamVLAGR00T.py`
  - `_encode_qwen_hidden()` returns `examples`, `qwen_inputs`, and `hidden_states[-1]`.
  - `forward()` computes the GR00T flow-matching action loss, then loops over `self.aux_heads`.
  - each aux head receives `hidden`, `batch_dict`, and a per-head mask.

- `starVLA/model/framework/VLM4A/UamVLAOFT.py`
  - `_maybe_build_aux_heads()` builds enabled aux heads from `config.framework.aux_heads`.
  - `_unpack_lerobot_sample()` is the natural place to load sidecar labels using `trajectory_id` and `base_index`.
  - `_collate_aux()` stacks optional sidecar fields and creates masks.

- Existing aux-head convention:
  - `ReconHead` and `FutureHead` use Qwen image-token hidden states as spatial conditioning for a DiT denoiser over VAE latents.
  - `PoseHead` uses Qwen hidden states as semantic conditioning for SE(3) pose score matching.
  - every head must return a dummy zero loss when its mask is empty so distributed training stays well-defined.

The four new branches should follow this pattern rather than changing the main action path.

## 3. Shared Design Rules

All four branches should be optional and controlled through `framework.aux_heads.<name>.enabled`.

Each head should implement the existing `AuxHead` interface:

```python
compute_loss(hidden_states: torch.Tensor, batch: dict, mask: torch.Tensor) -> HeadOutput
predict(hidden_states: torch.Tensor, batch: dict) -> HeadOutput
```

Each head should:

- accept full `hidden_states[-1]` of shape `[B, L, H]`
- use `batch["input_ids"]` only to locate image-token positions
- use `slice_image_tokens(...)` to extract one view's image tokens
- reshape image tokens to `[B, H, 20, 20]` when `patches_per_view=400`
- return `get_dummy_loss()` when `mask.any()` is false
- log both weighted loss and raw loss
- support visualization so failures are visible early

The shared spatial conditioning pattern is:

```text
hidden_states[-1]: [B, L, hidden_size]
input_ids:         [B, L]
        |
slice image token block by image_token_id and view_idx
        |
image_tokens:      [B, 400, hidden_size]
        |
reshape
        |
spatial_cond:      [B, hidden_size, 20, 20]
```

For spatial-map branches, a reusable base class is desirable:

```text
SpatialMapDenoisingHead
  - x_channel
  - target_key
  - mask_key
  - metric_prefix
  - view_idx
  - map_size
  - loss_weight
  - denoiser
```

`depth`, `grounding_mask`, and `affordance_heatmap` can all be instantiations of this base with different targets and minor preprocessing differences.

## 4. Branch 1: Relative Depth Denoising

### Purpose

The depth branch should teach the VLA hidden state to preserve dense geometric layout: near/far structure, object boundaries, tabletop geometry, and approximate 3D spatial relations. It should not turn the runtime policy into an RGB-D policy.

Because the UAMVLA method identity is denoising-based auxiliary learning, the preferred branch is not a plain depth regression head. It should be a relative-depth denoising head:

```text
relative inverse depth map D0
        |
add diffusion noise
        |
D_t
        |
DiT denoiser conditioned on Qwen spatial hidden states
        |
predict noise / denoised depth
```

### Target Definition

For CALVIN and LIBERO, raw simulator depth should be stored in meters when available. The training target should be converted to per-frame robust relative inverse depth:

```text
valid = finite(depth) and depth > 0
inverse_depth = 1 / clamp(depth, eps, max_depth)
lo = percentile(inverse_depth[valid], 1)
hi = percentile(inverse_depth[valid], 99)
relative_inverse_depth = clip((inverse_depth - lo) / (hi - lo), 0, 1)
```

This produces:

```text
1 = near
0 = far
```

The diffusion target can then be mapped to `[-1, 1]`:

```text
target = relative_inverse_depth * 2 - 1
```

The raw depth should be stored, not pre-normalized, so later experiments can compare relative inverse depth, normalized log depth, and metric depth without regenerating the dataset.

### Data Contract

Recommended sidecar layout:

```text
depths/
  static/<trajectory_id>/<base_index>.npy
  wrist/<trajectory_id>/<base_index>.npy
depth_masks/
  static/<trajectory_id>/<base_index>.npy
  wrist/<trajectory_id>/<base_index>.npy
meta/depth_stats.json
```

First implementation can use only `static`.

`_unpack_lerobot_sample()` should expose:

```python
out["depth_target"] = Tensor[1, H, W]      # raw metric depth in meters or normalized target, decided by config
out["depth_valid_mask"] = Tensor[1, H, W]  # optional
```

`_collate_aux()` should create:

```python
batch["depth_target"]
batch["depth_valid_mask"]      # optional
batch["depth_mask"]
```

### Model

Use a spatial-map DiT denoiser:

```text
target D0:     [B, 1, 20, 20]
condition:     [B, hidden_size, 20, 20]
denoiser in:   noisy D_t + timestep + condition
denoiser out:  noise prediction or clean map prediction
```

Initial settings:

```yaml
depth:
  enabled: false
  loss_weight: 0.1
  view_idx: 0
  target_kind: relative_inverse_depth
  target_size: 20
  denoiser_depth: 3
  denoiser_embed_dim: 512
  gen_timesteps: "1000"
```

Depth is useful but should be framed as a geometry ablation branch rather than the strongest core contribution. Its main value is testing whether UAMVLA can absorb dense geometric supervision through the same denoising abstraction.

## 5. Branch 2: Action-Conditioned Dynamics / Future Denoising

### Purpose

The existing `FutureHead` predicts a future or terminal image from current hidden states. This can teach goal appearance, but it does not explicitly teach controllable dynamics: what will happen if the demonstrated action chunk is executed?

The action-conditioned dynamics branch should denoise a future visual latent conditioned on both:

1. Qwen spatial hidden states
2. the demonstrated action chunk

This branch is the most central proposed addition because it directly links action supervision with future visual consequences.

### Target Definition

Use the RGB frame after the current action chunk:

```text
target frame index = base_index + future_action_window_size
```

For `action_horizon=8` and `future_action_window_size=7`, this is the frame after executing the 8-step action chunk.

This differs from the existing terminal-frame future target. It is local, action-aligned, and better suited to dynamics learning.

### Data Contract

Recommended sidecar or loader field:

```python
out["image_action_future"] = Tensor[3, H, W]
```

The target can be loaded directly from the LeRobot mp4:

```text
video.primary_image frame at base_index + future_action_window_size
```

Near-terminal samples that lack this frame should set `action_future_mask=False`.

`_collate_aux()` should expose:

```python
batch["image_action_future"]
batch["action"] or batch["action_chunk"]
batch["action_future_mask"]
```

The current examples already carry `example["action"]`, so the first version can reuse that instead of duplicating a new sidecar.

### Model

Reuse the VAE latent target and DiT denoiser pattern from `FutureHead`, but change the condition.

```text
Qwen hidden states
        |
spatial reader
        |
spatial_cond: [B, hidden_size, 20, 20]

GT action chunk: [B, action_horizon, action_dim]
        |
ActionChunkEncoder
        |
action_cond: [B, hidden_size]
        |
FiLM or broadcast fusion

fused_cond: [B, hidden_size, 20, 20]
        |
denoise VAE(image_action_future)
```

Recommended first fusion:

```text
scale, shift = Linear(action_cond) -> [B, 2 * hidden_size]
fused_cond = spatial_cond * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
```

This is cleaner than simple addition because the action modulates the visual condition.

### ActionChunkEncoder

Add a small encoder, not a second action policy:

```text
action_chunk [B, H, 7]
        |
Linear(7 -> hidden_size)
        |
add step positional embedding
        |
mean pool or 2-layer TransformerEncoder
        |
Linear(hidden_size -> hidden_size)
```

First version should be intentionally light:

```yaml
action_conditioned_dynamics:
  enabled: false
  loss_weight: 0.2
  view_idx: 0
  target_resize: 320
  denoiser_depth: 3
  denoiser_embed_dim: 1024
  action_encoder_layers: 1
  action_dropout: 0.1
```

`action_dropout` is important: randomly zeroing the action condition prevents the denoiser from ignoring hidden states and relying only on action.

### Required Ablations

This branch needs explicit ablations:

```text
hidden-only local future
action-only future
hidden + action future
terminal future vs action-aligned future
```

The desired result is that `hidden + action` is better than both `hidden-only` and `action-only`, showing that the branch learns controllable scene dynamics rather than a shortcut.

## 6. Branch 3: Grounding Mask Denoising

### Purpose

The grounding mask branch teaches the hidden state where the instruction target is in the original image. It is distinct from reconstruction:

```text
recon:
  What does the target look like?

grounding_mask:
  Where is the target or manipulable target part in the full observation?
```

This distinction should be explicit in the paper to avoid the criticism that mask supervision duplicates reconstruction.

### Target Definition

Use a single-channel target or target-part mask:

```text
grounding_target: [B, 1, 20, 20]
```

Values are soft or binary in `[0, 1]`, then mapped to `[-1, 1]` for diffusion.

For object manipulation tasks:

```text
mask = target object region
```

For articulated or part-based tasks:

```text
mask = manipulated part if available
```

Examples:

```text
move red block        -> red block mask
open drawer           -> drawer handle / drawer front mask
press button          -> button mask
turn on lightbulb     -> switch or button mask
```

If part masks are not reliably available, first version should use target object masks and note the limitation.

### Data Contract

Recommended layout:

```text
grounding_masks/
  static/<trajectory_id>/<base_index>.npy
  wrist/<trajectory_id>/<base_index>.npy
```

CALVIN already has rendered segmentation in preprocessing:

```python
rendered["seg_static"]
target_seg_id = worker.env.get_target_seg_id(target_object_id)
mask = rendered["seg_static"] == target_seg_id
```

LIBERO preprocessing also renders segmentation and can generate analogous masks.

`_unpack_lerobot_sample()` should expose:

```python
out["grounding_mask"] = Tensor[1, H, W]
```

`_collate_aux()` should expose:

```python
batch["grounding_mask"]
batch["grounding_mask_mask"]
```

The head name can be `grounding`, but the target key should remain explicit, for example `grounding_mask`.

### Model

Use the same spatial-map denoising pattern:

```text
target mask M0: [B, 1, 20, 20]
condition:      [B, hidden_size, 20, 20]
loss:           diffusion denoising loss
```

Suggested config:

```yaml
grounding:
  enabled: false
  loss_weight: 0.1
  view_idx: 0
  target_size: 20
  denoiser_depth: 3
  denoiser_embed_dim: 512
```

This branch should be presented as language-conditioned target localization, not as another reconstruction objective.

## 7. Branch 4: Single-Channel Affordance Heatmap Denoising

### Purpose

The affordance branch teaches the hidden state where the robot should interact, not merely where the target object is.

```text
grounding_mask:
  target / part location

affordance_heatmap:
  actionable interaction location
```

For a drawer task, grounding may cover the drawer handle or front, while affordance should peak near the contact/grasp point. For a block task, grounding covers the block, while affordance should peak near a likely grasp/contact point.

### Target Definition

First version should be single-channel:

```text
affordance_heatmap: [B, 1, 20, 20]
```

Value:

```text
1 = most actionable location
0 = unrelated region
```

Use a Gaussian heatmap centered at an estimated interaction point.

Recommended automatic label generation:

```text
1. recover future TCP / gripper positions from robot state or action rollout
2. recover target object point cloud or target mask
3. find target point closest to the future TCP trajectory
4. project that 3D point into the static camera
5. draw a Gaussian heatmap around the projected point
6. downsample to 20x20
```

If 3D point-cloud matching is unreliable, use a weaker 2D fallback:

```text
project future TCP positions into camera
take the first point near the target mask
draw Gaussian
```

The 3D version is preferred because it ties affordance to the target object rather than just the gripper trajectory.

### Data Contract

Recommended layout:

```text
affordance_heatmaps/
  static/<trajectory_id>/<base_index>.npy
  wrist/<trajectory_id>/<base_index>.npy
```

First version can use only `static`.

`_unpack_lerobot_sample()` should expose:

```python
out["affordance_heatmap"] = Tensor[1, H, W]  # or already [1, 20, 20]
```

`_collate_aux()` should expose:

```python
batch["affordance_heatmap"]
batch["affordance_mask"]
```

The target should be a soft Gaussian map in `[0, 1]`, mapped to `[-1, 1]` for diffusion.

### Model

Use the same spatial-map denoising pattern:

```text
target heatmap A0: [B, 1, 20, 20]
condition:         [B, hidden_size, 20, 20]
loss:              diffusion denoising loss
```

Suggested config:

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

This branch is a core candidate because it is more action-relevant than target reconstruction or target masking.

## 8. Preprocessing Work Items

### CALVIN

Current CALVIN LeRobot preprocessing already renders:

```text
rgb_static
rgb_wrist
depth_static
depth_wrist
seg_static
seg_wrist
static camera intrinsics / extrinsics
target point cloud
target object pose
```

Proposed additions:

```text
save raw depth maps:
  depths/static/<episode>/<frame>.npy
  depths/wrist/<episode>/<frame>.npy

save grounding masks:
  grounding_masks/static/<episode>/<frame>.npy
  grounding_masks/wrist/<episode>/<frame>.npy

save affordance heatmaps:
  affordance_heatmaps/static/<episode>/<frame>.npy

support action-aligned future image lookup:
  no new sidecar required if reading from mp4 by frame index works
```

The existing point-cloud and image-target sidecars should remain unchanged.

### LIBERO

The LIBERO preprocessing path already saves:

```text
depth/static/*.npy
depth/wrist/*.npy
point_clouds/*.npy
image_targets/*.png
```

Proposed additions:

```text
standardize depth layout with CALVIN or add a framework resolver for both schemas
save grounding masks from rendered segmentation
save affordance heatmaps using TCP trajectory + target point cloud/mask
support action-aligned future image loading
```

The first implementation should avoid forcing CALVIN and LIBERO into one physical file layout if that causes churn. A small resolver layer can map dataset-specific sidecar paths to common batch keys.

## 9. Framework Integration Plan Sketch

New batch keys:

```text
depth_target
depth_valid_mask
image_action_future
grounding_mask
affordance_heatmap
```

New masks:

```text
depth_mask
action_dynamics_mask
grounding_mask_mask
affordance_mask
```

Potential new files:

```text
starVLA/model/modules/uamvla/aux_heads/spatial_map_denoising_head.py
starVLA/model/modules/uamvla/aux_heads/depth_head.py
starVLA/model/modules/uamvla/aux_heads/action_conditioned_future_head.py
starVLA/model/modules/uamvla/aux_heads/grounding_head.py
starVLA/model/modules/uamvla/aux_heads/affordance_head.py
starVLA/model/modules/uamvla/components/action_chunk_encoder.py
```

`depth`, `grounding`, and `affordance` can share most of the implementation through `SpatialMapDenoisingHead`.

`action_conditioned_future` should probably stay separate because it operates on VAE image latents and needs `ActionChunkEncoder`.

## 10. Suggested Experiment Order

Do not enable all four branches at once. The recommended order is:

```text
0. current baseline: recon / future / pose as configured
1. add action-conditioned dynamics only
2. add affordance heatmap only
3. add grounding mask only
4. add depth only
5. action-conditioned dynamics + affordance
6. grounding + affordance
7. all selected branches with tuned weights
```

Depth should likely remain a geometry ablation. Action-conditioned dynamics and affordance are stronger candidates for core method claims.

For each branch, evaluate:

```text
train action loss
aux raw loss
validation success rate
visualization quality
whether aux loss learns non-trivial labels
whether action success improves without inference-time sidecar input
```

## 11. Risks and Mitigations

### Risk: too many heads look like task stacking

Mitigation: present UAMVLA as a unified conditional denoising framework, not as arbitrary head accumulation. Group heads by target variable type:

```text
appearance: recon
visual dynamics: future, action-conditioned future
geometry: pose, depth
spatial task grounding: grounding, affordance
```

### Risk: grounding overlaps with reconstruction

Mitigation: explicitly distinguish:

```text
recon = target appearance
grounding = target location in full observation
affordance = actionable interaction point
```

Include ablations that show grounding and recon have different effects.

### Risk: action-conditioned future ignores hidden states

Mitigation:

```text
small ActionChunkEncoder
action dropout
action-only ablation
hidden-only ablation
```

### Risk: depth does not fit the denoising story

Mitigation: implement depth as relative-depth denoising, not plain regression, and treat it as optional geometry supervision rather than the main contribution.

### Risk: preprocessing labels differ across CALVIN and LIBERO

Mitigation: store raw sidecars when possible, normalize at training time, and use dataset-specific path resolvers that emit common framework keys.

## 12. Open Decisions for Next Discussion

1. Should the spatial-map denoising branches predict noise, `x0`, or velocity? The current `ReconDenoiser` path likely predicts the same objective as the existing diffusion utility; reuse is preferred unless experiments suggest otherwise.
2. Should map targets be stored at full resolution or pre-downsampled to `20x20`? Full resolution preserves flexibility; `20x20` reduces storage and preprocessing complexity.
3. Should action-conditioned dynamics replace the existing terminal `FutureHead`, or coexist with it under a new key?
4. Should `grounding_mask` mark the whole target object or the manipulated part? First version may need object masks; part masks are better but harder.
5. Should affordance labels use target point cloud nearest to TCP trajectory, projected TCP trajectory, or task-specific heuristics?
6. Should depth use static view only in the first version, or static + wrist from the start?

## 13. Recommended Naming

Config keys:

```yaml
aux_heads:
  depth:
    enabled: false
  action_conditioned_future:
    enabled: false
  grounding:
    enabled: false
  affordance:
    enabled: false
```

Class names:

```text
DepthDenoisingHead
ActionConditionedFutureHead
GroundingMaskDenoisingHead
AffordanceHeatmapDenoisingHead
```

Metric names:

```text
depth_loss_raw
action_conditioned_future_loss_raw
grounding_loss_raw
affordance_loss_raw
```

Visualization keys:

```text
viz/uamvla_gr00t/depth_*
viz/uamvla_gr00t/action_conditioned_future_*
viz/uamvla_gr00t/grounding_*
viz/uamvla_gr00t/affordance_*
```

