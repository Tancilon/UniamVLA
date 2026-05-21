# UamVLA LIBERO Auxiliary Preprocessing Design

Date: 2026-05-21
Status: approved design, awaiting implementation plan
Related design: `docs/superpowers/specs/2026-05-21-uamvla-aux-denoising-suite-design.md`

## 1. Purpose

This document defines the final LIBERO preprocessing path for UamVLA-GR00T
auxiliary-head training. The goal is to convert official LIBERO HDF5 demos into
four independent LeRobot v2 datasets with replay-aligned RGB videos and all
sidecars required by the UamVLA auxiliary denoising suite.

The current `tools/preprocess/libero_preprocessor.py` still follows the old
`data.jsonl` / `statistics.yaml` path, and its statistics writer is already
deprecated. This design replaces that path entirely. The implementation should
delete the old JSONL/statistics logic rather than maintaining two LIBERO
formats.

## 2. Inputs And Outputs

Input data is the official LIBERO HDF5 dataset already downloaded locally:

```text
datasets/libero/
  libero_spatial/*.hdf5
  libero_object/*.hdf5
  libero_goal/*.hdf5
  libero_10/*.hdf5
```

Output data is four independent LeRobot v2 datasets:

```text
datasets/libero2uam/
  lerobot_libero_spatial/
  lerobot_libero_object/
  lerobot_libero_goal/
  lerobot_libero_10/
```

Each output dataset contains normal LeRobot v2 files:

```text
data/chunk-000/episode_000000.parquet
videos/chunk-000/video.primary_image/episode_000000.mp4
videos/chunk-000/video.wrist_image/episode_000000.mp4
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/modality.json
```

It also contains UamVLA sidecars:

```text
image_targets/<episode>/<frame>.png
point_clouds/<episode>/<frame>.npy
depths/static/<episode>/<frame>.npy
grounding_masks/static/<episode>/<frame>.npy
grounding_masks/static/<episode>/<frame>.json
affordance_heatmaps/static/<episode>/<frame>.npy
camera_params.json
debug/replay_rgb_check/
meta/uamvla_aux_coverage.json
```

Each `demo_i` in an official HDF5 task file becomes one LeRobot episode. Episode
indices are contiguous within a suite. The four suites remain physically
separate and are combined at training time through `data_mix`.

## 3. Replay Source Of Truth

RGB training videos must be rendered from the LIBERO replay environment, not
copied from HDF5 `obs/*_rgb`. The HDF5 RGB frames are kept only for lightweight
debug comparison.

Replay-generated images are the source of truth:

```text
agentview replay RGB            -> video.primary_image
robot0_eye_in_hand replay RGB   -> video.wrist_image
```

Depth, segmentation, target crops, point clouds, affordance heatmaps, target
poses, and camera extrinsics are generated from the same replayed MuJoCo state.
This keeps all visual and geometric supervision pixel-aligned.

The preprocessor runs in a dedicated `libero_env` conda environment. Training
continues to run in the existing `uamvla` environment and consumes only the
generated LeRobot datasets and sidecars.

Required replay environment variables:

```text
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

The preprocessor must fail fast if it cannot import LIBERO / robosuite, create
an `OffScreenRenderEnv`, reset it, and render a frame.

## 4. Per-Frame Replay Pipeline

For each HDF5 task file, the preprocessor resolves the correct BDDL through the
LIBERO benchmark API. For each demo and frame:

1. Read the flattened MuJoCo state from `states[t]`.
2. Replay it with `env.sim.set_state_from_flattened(states[t])` and
   `env.sim.forward()`.
3. Render static and wrist RGB.
4. Render static and wrist metric depth.
5. Render static instance segmentation.
6. Read end-effector pose, joint state, gripper state, static camera extrinsic,
   and active target pose from MuJoCo.
7. Write the LeRobot parquet row and sidecars.

The HDF5 schema must contain:

```text
actions
states
obs/ee_pos
obs/ee_ori
obs/joint_states
obs/gripper_states
```

Missing required fields are schema errors and stop the suite.

## 5. State And Action Layout

Each row stores a 33-D packed state for UamVLA auxiliary supervision:

```text
state.robot_obs            = ee_pos(3) + ee_ori(3) + joint_states(7) + gripper_states(2)
state.target_pose_rot6d    = active target rotation in 6D
state.target_pose_trans    = active target translation
state.static_cam_rot6d     = static camera rotation in 6D
state.static_cam_trans     = static camera translation
```

The concatenated state layout is:

```text
robot_obs(15)
+ target_pose_rot6d(6)
+ target_pose_trans(3)
+ static_cam_rot6d(6)
+ static_cam_trans(3)
= 33
```

The aux state slices match the existing CALVIN UamVLA layout:

```yaml
target_pose_rot6d: [15, 21]
target_pose_trans: [21, 24]
static_cam_rot6d:  [24, 30]
static_cam_trans:  [30, 33]
```

GR00T's main action model uses only the first 7 state dimensions, exactly as in
the CALVIN GR00T setup. The full 33-D state is still present so
`UamVLAOFT._unpack_lerobot_sample()` can recover target pose and camera
extrinsic for auxiliary heads.

Actions are the raw 7-D LIBERO HDF5 `actions[t]` values:

```text
action.x
action.y
action.z
action.roll
action.pitch
action.yaw
action.gripper
```

The LeRobot dataloader applies action normalization. The
`action_conditioned_future` head consumes the normalized training action chunk;
the preprocessor does not create a separately normalized action sidecar.

## 6. Curated Target Mapping

The preprocessor uses a deterministic curated mapping for the 40 standard
LIBERO tasks. The mapping is part of the repository and must be auditable in
code review.

Each task entry defines a target policy, not only a single body:

```yaml
task_name:
  candidate_targets:
    - body: ...
      kind: object
    - body: ...
      kind: object
  active_target_rule: closest_pointcloud_to_future_tcp
  fallback_target: ...
  part_grounding:
    enabled: true|false
    body_or_geom_patterns: [...]
```

Single-target tasks normally have one candidate. Multi-target tasks, such as
`put both A and B ...`, list all legal candidate objects. Each frame still
produces supervision for one active target so existing UamVLA heads and
collators do not need multi-target changes.

Mapping coverage is strict:

- all 40 official task HDF5 files must match a mapping entry;
- unmapped tasks fail preprocessing;
- invalid body or geom patterns fail preprocessing.

The LLM target resolver path is not part of this final pipeline.

## 7. Active Target Assignment

For each frame in a multi-target task:

1. Generate an object mask for each candidate target.
2. Generate a candidate point cloud from static depth and the candidate mask.
3. Compute the minimum distance between the candidate point cloud and the
   future TCP trajectory.
4. Choose the target with the smallest distance as the raw active target.
5. Smooth the raw target sequence within the episode.

The smoothing step removes isolated short target segments while preserving true
longer target switches:

```text
raw:      A A A A B A A A A B B B B
smoothed: A A A A A A A A A B B B B
```

The smoothing is parameterized by `min_segment_len`. It only merges short
segments bounded by the same neighboring target and does not use contact-phase
heuristics.

If all candidate point clouds are empty for a frame, the frame and episode are
retained. The affected sidecars are omitted and the training mask for those
heads becomes false.

## 8. Part Grounding Policy

Part-level grounding is conservative. It is enabled only for high-confidence
manipulable parts listed in the curated mapping, such as drawer handles, stove
buttons or knobs, and microwave handle or door parts.

Grounding behavior:

- if a high-confidence part mask is available and visible, write the part mask
  and `grounding_level=part`;
- otherwise write the active object mask and `grounding_level=object`;
- if neither mask is usable, omit the grounding sidecar for that frame.

Other branches remain object-centric:

- pose uses active target object pose;
- point cloud uses active target object point cloud;
- recon uses active target object crop;
- affordance uses the active target object point cloud and chooses the point
  closest to the future TCP trajectory.

This keeps the semantics of each branch clear while still preserving the
part/object grounding hierarchy.

## 9. Auxiliary Sidecar Semantics

The generated sidecars support all UamVLA-GR00T auxiliary heads:

```text
recon:
  image_targets/<episode>/<frame>.png
  active object crop from replay RGB and segmentation

future:
  terminal frame from video.primary_image

action_conditioned_future:
  frame = base_index + future_action_window_size from video.primary_image
  missing when the frame index is out of bounds

pose:
  state.target_pose_rot6d + state.target_pose_trans
  active target object pose

depth:
  depths/static/<episode>/<frame>.npy
  raw metric static depth; the head converts to robust relative inverse depth

grounding:
  grounding_masks/static/<episode>/<frame>.npy
  grounding_masks/static/<episode>/<frame>.json
  1x20x20 object or part token-grid mask in [0, 1]

affordance:
  affordance_heatmaps/static/<episode>/<frame>.npy
  1x20x20 Gaussian heatmap around the projected object point nearest the
  future TCP trajectory

pose point cloud conditioning:
  point_clouds/<episode>/<frame>.npy
  active object point cloud, 1024x3 float32, world frame
```

The training/inference boundary remains unchanged: all sidecars are
training-only supervision and are never inference-time sensors.

Shape and value constraints:

- `point_clouds`: `1024x3 float32`; omit on empty target;
- `grounding_masks`: `1x20x20 float32` in `[0, 1]`;
- `affordance_heatmaps`: `1x20x20 float32` in `[0, 1]`;
- `depths/static`: raw metric `float32` depth, not pre-normalized;
- `image_targets`: RGB PNG crops;
- missing sidecars are valid and are represented by existing batch masks.

## 10. Parallel Preprocessing

Default preprocessing uses visible-GPU-aware multiprocessing.

If the visible GPU set is `0,1,2,3`, the default worker assignment is:

```text
worker 0 -> CUDA_VISIBLE_DEVICES=0, MUJOCO_EGL_DEVICE_ID=0
worker 1 -> CUDA_VISIBLE_DEVICES=1, MUJOCO_EGL_DEVICE_ID=0
worker 2 -> CUDA_VISIBLE_DEVICES=2, MUJOCO_EGL_DEVICE_ID=0
worker 3 -> CUDA_VISIBLE_DEVICES=3, MUJOCO_EGL_DEVICE_ID=0
```

Each worker sees one GPU, so `MUJOCO_EGL_DEVICE_ID=0` is local to that worker.
Workers are launched with spawn semantics. LIBERO, robosuite, MuJoCo, and
OpenGL imports happen inside the worker after the EGL environment is set.

The CLI supports explicit fallback and assignment:

```bash
--num-workers 1
--render-gpus 0,1,2,3
```

To avoid episode and row-index collisions, the main process first pre-scans all
HDF5 files in the suite and computes:

- task index per HDF5 task file;
- demo count per task;
- episode index per demo;
- frame count per episode;
- global row index start per episode.

Workers then write unique parquet, video, and sidecar paths directly. The main
process writes suite-level metadata after all workers finish successfully.

## 11. Debug And Quality Reports

The preprocessor writes debug and coverage artifacts by default.

### Replay RGB Check

For each task, sample a small number of frames and compare replay RGB against
HDF5 RGB. The check writes:

```text
debug/replay_rgb_check/*.jpg
debug/replay_rgb_check/metrics.jsonl
```

The metrics include frame identity, camera name, mean absolute difference, and
a PSNR-like score after resizing to a common resolution. These metrics are not
training inputs and do not fail the run by default. They are used to catch
camera convention, flip, BDDL, or state replay mistakes early.

### Coverage Report

Each suite writes:

```text
meta/uamvla_aux_coverage.json
```

It records valid-frame ratios for:

```text
recon
pose
point_cloud
depth
grounding
affordance
action_conditioned_future
```

It also records task-level counts for missing target masks, empty point clouds,
part/object grounding counts, and active-target switches.

## 12. Failure Policy

Fail fast:

- missing replay dependencies;
- EGL render failure;
- missing required HDF5 fields;
- unmapped task;
- invalid mapping body/geom pattern;
- worker exception for a task;
- existing output directory unless `--overwrite` is set.

Retain data with masks:

- target temporarily invisible;
- empty target point cloud;
- action-conditioned future frame out of bounds;
- part mask unavailable but object mask available.

The preprocessor should not silently produce a partially trustworthy suite. If
the suite run fails, the output directory is considered incomplete.

## 13. Registry And Training Config

The implementation must also make the generated data directly trainable with
UamVLA-GR00T.

Add a LIBERO UamVLA h8 data config:

```text
robot type: uamvla_libero_franka_h8
```

It uses:

```text
video.primary_image
video.wrist_image

state.robot_obs
state.target_pose_rot6d
state.target_pose_trans
state.static_cam_rot6d
state.static_cam_trans

action.x
action.y
action.z
action.roll
action.pitch
action.yaw
action.gripper

annotation.human.action.task_description
```

Action horizon is 8:

```text
action_indices = [0,1,2,3,4,5,6,7]
```

State uses the current frame only:

```text
state_indices = [0]
```

Transforms follow the CALVIN UamVLA pattern:

- actions are converted to tensors and min-max normalized for the first six
  action dimensions;
- `state.robot_obs` is converted to tensor and may be mean/std normalized;
- `state.target_pose_rot6d`, `state.target_pose_trans`,
  `state.static_cam_rot6d`, and `state.static_cam_trans` remain raw passthrough
  values for auxiliary supervision.

Register mixtures:

```text
uamvla_libero_all_h8:
  - lerobot_libero_object
  - lerobot_libero_goal
  - lerobot_libero_spatial
  - lerobot_libero_10

uamvla_libero_spatial_h8
uamvla_libero_object_h8
uamvla_libero_goal_h8
uamvla_libero_10_h8
```

Add a training config:

```text
starVLA/config/training/uamvla_gr00t_libero.yaml
```

It is based on the CALVIN GR00T config, with LIBERO data paths and mixture:

```yaml
datasets:
  vla_data:
    data_root_dir: datasets/libero2uam
    data_mix: uamvla_libero_all_h8
    obs: ["video.primary_image", "video.wrist_image"]
    include_state: true
    image_resize: 640
    aux_state_slice:
      target_pose_rot6d: [15, 21]
      target_pose_trans: [21, 24]
      static_cam_rot6d:  [24, 30]
      static_cam_trans:  [30, 33]

framework:
  action_model:
    action_dim: 7
    state_dim: 7
    action_horizon: 8
    future_action_window_size: 7
```

The config is full-capable for all auxiliary heads. It may keep conservative
defaults for `enabled` flags and weights, but every branch must have a valid
configuration so paper runs can enable all heads and leave-one-branch-out
ablations without changing code.

## 14. Verification

Required verification before declaring implementation complete:

1. Unit tests for curated mapping coverage and body/geom pattern resolution.
2. Unit tests for active target smoothing.
3. Unit tests for sidecar writer shapes and missing-sidecar behavior.
4. Import tests that skip gracefully outside `libero_env`.
5. A replay smoke that processes a tiny subset in `libero_env`.
6. A dataloader smoke that loads one processed suite from the `uamvla` env.
7. A framework smoke that confirms `UamVLAGR00T` can unpack one batch with
   state, sidecars, and masks.
8. A config load smoke for `uamvla_gr00t_libero.yaml`.

The implementation plan should treat `libero_env` replay tests and `uamvla`
training/dataloader tests as separate verification lanes.
