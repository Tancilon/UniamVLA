# CALVIN Aux Sidecar Completion for UamVLA-GR00T

Date: 2026-05-22
Status: approved design, awaiting implementation plan

## 1. Purpose

The LIBERO preprocessing path now produces the sidecars needed by the full
UamVLA-GR00T auxiliary-denoising suite. CALVIN should provide the same training
contract so the GR00T model can train all auxiliary heads on both single-scene
CALVIN splits and merged ABCD datasets.

This design uses a minimal patch strategy. It does not rewrite the CALVIN
preprocessing pipeline or introduce a shared writer abstraction. It completes the
current CALVIN LeRobot preprocessor and multi-scene merger so their outputs keep
the same auxiliary sidecar layout expected by the training dataloader.

## 2. Scope

In scope:

- single-scene CALVIN LeRobot outputs from `runners/preprocess_calvin.py`;
- merged multi-scene CALVIN outputs from `runners/preprocess_calvin_multiscene.py`;
- sidecar completeness for `image_target`, `point_cloud`, `depth`,
  `grounding`, and `affordance`;
- `grounding_level` metadata for CALVIN masks;
- dataset-level `meta/uamvla_aux_coverage.json`;
- strict merge-time validation that required sidecars exist.

Out of scope:

- changing model or dataloader sidecar lookup semantics;
- changing CALVIN action/state definitions;
- adding new CALVIN target object mappings beyond the existing task map;
- changing the existing CALVIN language window segmentation;
- adding active-target segment switching to CALVIN affordance.

## 3. Existing Code Anchors

The current CALVIN preprocessing code already contains most of the per-frame
signals:

- `runners/preprocess_calvin.py` is the single-scene CLI.
- `tools/preprocess/calvin_preprocessor_lerobot.py` writes LeRobot parquet,
  videos, image targets, point clouds, depth sidecars, grounding masks, and
  affordance heatmaps.
- `tools/preprocess/calvin_task_map.py` maps CALVIN task labels to PyBullet
  target ids and infers stack/unstack block targets.
- `tools/preprocess/calvin_lerobot_merger.py` merges scene-specific LeRobot
  datasets into an ABCD dataset, but currently only copies image targets and
  point clouds.

The full aux-head dataset contract is already represented by the LIBERO writer:

```text
image_targets/<episode>/<frame>.png
point_clouds/<episode>/<frame>.npy
depths/static/<episode>/<frame>.npy
grounding_masks/static/<episode>/<frame>.npy
grounding_masks/static/<episode>/<frame>.json
affordance_heatmaps/static/<episode>/<frame>.npy
meta/uamvla_aux_coverage.json
```

CALVIN should match this layout.

## 4. Data Contract

For every retained CALVIN frame, the preprocessor must write:

```text
video.primary_image
video.wrist_image
state.robot_obs
state.target_pose_rot6d
state.target_pose_trans
state.static_cam_rot6d
state.static_cam_trans
action.x/y/z/roll/pitch/yaw/gripper
trajectory_id
base_index
```

and sidecars:

```text
image_targets/<trajectory_id>/<base_index>.png
point_clouds/<trajectory_id>/<base_index>.npy
depths/static/<trajectory_id>/<base_index>.npy
grounding_masks/static/<trajectory_id>/<base_index>.npy
grounding_masks/static/<trajectory_id>/<base_index>.json
affordance_heatmaps/static/<trajectory_id>/<base_index>.npy
```

The sidecar meanings are:

- `image_targets`: static-view crop of the target object or part.
- `point_clouds`: cleaned target point cloud with shape `(1024, 3)`.
- `depths/static`: raw static-camera metric depth, stored as `float32`.
- `grounding_masks/static`: static-view 20x20 token-grid mask with shape
  `(1, 20, 20)` and values in `[0, 1]`.
- grounding json: `{"grounding_level": "part"}` or
  `{"grounding_level": "object"}`.
- `affordance_heatmaps/static`: static-view 20x20 Gaussian heatmap with shape
  `(1, 20, 20)` and values in `[0, 1]`.

The heads remain responsible for training-time normalization. In particular,
the depth head converts raw depth to per-frame robust relative inverse depth,
and spatial-map diffusion heads map `[0, 1]` targets to `[-1, 1]` internally.

## 5. CALVIN Grounding Semantics

CALVIN target ids already distinguish many manipulable fixture parts from
movable objects. The `grounding_level` should therefore be derived naturally
from the resolved target id:

```text
target_id starts with "table__" -> grounding_level = "part"
otherwise                       -> grounding_level = "object"
```

This maps fixture/link tasks to `part`:

```text
table__drawer_link
table__slide_link
table__switch_link
table__button_link
```

and block tasks to `object`:

```text
block_red
block_blue
block_pink
```

The grounding mask itself remains the static-view segmentation mask for the
resolved target id, downsampled to the 20x20 token grid.

## 6. CALVIN Affordance Semantics

CALVIN language windows are treated as atomic task windows. The affordance
target remains:

```text
target point cloud at frame i
        versus
TCP positions from frame i through the end of the language window
```

The nearest point on the current target point cloud to the future TCP trajectory
is projected to the static camera and rendered as a 20x20 Gaussian heatmap.

No active-target segment switching is introduced for CALVIN. Unlike some LIBERO
tasks, CALVIN language windows do not normally switch active targets within a
single annotation window, so the window-end TCP definition is simpler and more
faithful to the dataset structure.

## 7. Single-Scene Preprocessing Changes

`tools/preprocess/calvin_preprocessor_lerobot.py` should keep its current
single-process structure. The required minimal changes are:

1. compute `grounding_level` per retained frame from the resolved target id;
2. pass per-frame grounding levels into sidecar writing;
3. write the grounding json next to each grounding mask;
4. accumulate per-task and global coverage counts;
5. write `meta/uamvla_aux_coverage.json` after all episodes are emitted.

Coverage should count only retained episodes and retained frames. Since the
current strict frame policy drops or aborts windows when a required target is
missing, coverage for retained frames should normally be full. The coverage file
is still useful because it confirms that sidecars are present and supports
future missing-label policies.

The coverage schema should mirror LIBERO:

```json
{
  "tasks": {
    "<instruction>": {
      "image_target": {"valid": 0, "total": 0},
      "point_cloud": {"valid": 0, "total": 0},
      "depth": {"valid": 0, "total": 0},
      "grounding": {"valid": 0, "total": 0},
      "affordance": {"valid": 0, "total": 0}
    }
  },
  "totals": {
    "image_target": {"valid": 0, "total": 0},
    "point_cloud": {"valid": 0, "total": 0},
    "depth": {"valid": 0, "total": 0},
    "grounding": {"valid": 0, "total": 0},
    "affordance": {"valid": 0, "total": 0}
  }
}
```

## 8. ABCD Merge Changes

`tools/preprocess/calvin_lerobot_merger.py` should preserve all aux sidecars
when merging scene-specific outputs.

For each scene-local episode:

```text
local_episode -> global_episode = scene.episode_index_to_global[local_episode]
```

the merger must copy:

```text
scene/depths/static/<local_episode>/<frame>.npy
scene/grounding_masks/static/<local_episode>/<frame>.npy
scene/grounding_masks/static/<local_episode>/<frame>.json
scene/affordance_heatmaps/static/<local_episode>/<frame>.npy
```

to:

```text
output/depths/static/<global_episode>/<frame>.npy
output/grounding_masks/static/<global_episode>/<frame>.npy
output/grounding_masks/static/<global_episode>/<frame>.json
output/affordance_heatmaps/static/<global_episode>/<frame>.npy
```

Frame indices are not remapped. Only the episode directory changes.

The merger should also merge `meta/uamvla_aux_coverage.json`:

- `totals` are summed across scenes;
- `tasks` are merged by task/instruction text, summing matching metric counts;
- the merged file is written to the final dataset meta directory.

The existing merge contract for parquet, videos, image targets, point clouds,
`tasks.jsonl`, `episodes.jsonl`, and `info.json` remains unchanged.

## 9. Strict Validation

This design chooses strict mode for aux sidecars. A CALVIN dataset meant to train
all UamVLA-GR00T aux heads should fail fast if required sidecars are missing.

The merger validation should check every retained frame for:

```text
image_targets/<episode>/<frame>.png
point_clouds/<episode>/<frame>.npy
depths/static/<episode>/<frame>.npy
grounding_masks/static/<episode>/<frame>.npy
grounding_masks/static/<episode>/<frame>.json
affordance_heatmaps/static/<episode>/<frame>.npy
```

If any required file is missing, the merger raises `CalvinLeRobotMergeError`
with the local scene root, episode id, frame id, and missing path. Silent
partial output is not acceptable for this path.

## 10. Testing and Verification

Implementation should be verified in three layers:

1. Unit-style checks for helper behavior:
   - target id to grounding level;
   - coverage accumulation;
   - sidecar path remapping for merged episodes.
2. Smoke preprocessing on a small CALVIN split:
   - one or two windows;
   - confirm every retained frame has all sidecars;
   - inspect `meta/uamvla_aux_coverage.json`.
3. Multi-scene merge smoke test:
   - merge small per-scene outputs;
   - confirm global episode directories exist for all sidecar families;
   - confirm merged coverage totals equal the sum of scene coverage totals.

Optional visual debug can reuse the existing modality visualization approach:
show static RGB, wrist RGB, depth, grounding overlay, affordance overlay, and
image target for selected frames. This is useful for audits but not required for
the minimal patch.

## 11. Risks

- The existing CALVIN merger currently validates only image targets and point
  clouds. Extending it to strict aux validation may surface older preprocessed
  scene outputs as incompatible. That is intentional for full aux-head training.
- `grounding_level` derived from `target_id.startswith("table__")` assumes the
  current CALVIN target map convention. If future CALVIN variants introduce
  non-table articulated parts, this rule should be revisited.
- CALVIN affordance remains window-end based rather than active-segment based.
  This is appropriate for CALVIN's language-window structure, but it should be
  documented as a dataset-specific target definition.

## 12. Final Design Choice

Use a minimal patch:

- complete single-scene CALVIN sidecar metadata and coverage;
- keep CALVIN affordance as future TCP through the language window end;
- derive `grounding_level` naturally from the resolved target id;
- extend ABCD merge to validate, copy, remap, and merge all aux sidecars;
- do not refactor CALVIN around a shared LIBERO writer in this iteration.
