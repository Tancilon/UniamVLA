# UamVLA LIBERO Auxiliary Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy LIBERO JSONL preprocessor with a replay-based official-HDF5 to LeRobot-v2 pipeline that produces all UamVLA-GR00T auxiliary sidecars and trainable LIBERO mixtures.

**Architecture:** Split the implementation into focused modules: curated target mapping, pure preprocessing utilities, LeRobot writer, replay worker/orchestrator, CLI, and registry/config. Pure helpers and writer behavior are tested without LIBERO; replay integration is tested behind `pytest.importorskip` and a small `libero_env` smoke.

**Tech Stack:** Python 3.10, h5py, numpy, PIL, imageio.v3, pyarrow/parquet, MuJoCo/LIBERO/robosuite in `libero_env`, LeRobot v2 dataset layout, PyTorch dataloader registry.

---

## File Structure

- Create `tools/preprocess/libero_target_mapping.py`
  - Owns `TargetSpec`, `PartGroundingSpec`, `TaskTargetPolicy`, curated 40-task policy table, task-name normalization, and mapping coverage checks.
- Create `tools/preprocess/libero_preprocess_utils.py`
  - Owns pure helpers for episode planning, GPU assignment, mask resizing, active-target smoothing, target-distance scoring, heatmaps, robot-state packing, and debug RGB metrics.
- Create `tools/preprocess/libero_lerobot_writer.py`
  - Owns LeRobot parquet/video/meta writing and strict sidecar writing.
- Replace `tools/preprocess/libero_preprocessor.py`
  - Owns replay env creation, HDF5 schema validation, per-task worker execution, active target extraction, and suite orchestration.
- Modify `runners/preprocess_libero.py`
  - Main CLI for full-suite and single-suite preprocessing into `datasets/libero2uam`.
- Modify `tools/preprocess/run_libero_preprocess.py`
  - Keep as a compatibility wrapper that delegates to `runners/preprocess_libero.py` semantics.
- Modify `examples/LIBERO/train_files/data_registry/data_config.py`
  - Add `UamVLALiberoH8DataConfig`, `uamvla_libero_franka_h8`, and four suite/all mixtures.
- Create `starVLA/config/training/uamvla_gr00t_libero.yaml`
  - Full-capable UamVLA-GR00T LIBERO training config.
- Tests:
  - `tests/test_libero_target_mapping.py`
  - `tests/test_libero_preprocess_utils.py`
  - `tests/test_libero_lerobot_writer.py`
  - `tests/test_libero_preprocessor_planning.py`
  - `tests/dataloader/test_uamvla_libero_registry.py`
  - `tests/framework/test_uamvla_gr00t_libero_state.py`
  - `tests/test_libero_replay_env_smoke.py`

---

### Task 1: Curated Target Mapping

**Files:**
- Create: `tools/preprocess/libero_target_mapping.py`
- Test: `tests/test_libero_target_mapping.py`

- [ ] **Step 1: Write the failing mapping coverage tests**

Create `tests/test_libero_target_mapping.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from tools.preprocess.libero_target_mapping import (
    LIBERO_STANDARD_SUITES,
    TASK_TARGET_POLICIES,
    get_task_policy,
    normalize_task_name,
    validate_policy_table,
)


EXPECTED_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
}


def test_standard_suite_list_is_four_suite_benchmark():
    assert LIBERO_STANDARD_SUITES == (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    )


def test_policy_table_has_40_entries_with_expected_suite_counts():
    counts = {suite: 0 for suite in EXPECTED_COUNTS}
    for key in TASK_TARGET_POLICIES:
        suite, task = key.split("/", 1)
        assert task
        counts[suite] += 1
    assert counts == EXPECTED_COUNTS
    assert len(TASK_TARGET_POLICIES) == 40


def test_normalize_task_name_strips_demo_suffix_and_extension():
    assert (
        normalize_task_name("libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5")
        == "libero_goal/open_the_middle_drawer_of_the_cabinet"
    )


def test_representative_single_and_multi_target_policies():
    bowl = get_task_policy(
        "libero_spatial",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5",
    )
    assert [t.body_pattern for t in bowl.candidate_targets] == ["black_bowl"]
    assert bowl.fallback_target_pattern == "black_bowl"
    assert bowl.part_grounding.enabled is False

    both = get_task_policy(
        "libero_10",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5",
    )
    assert [t.body_pattern for t in both.candidate_targets] == [
        "alphabet_soup",
        "cream_cheese",
    ]
    assert both.active_target_rule == "closest_pointcloud_to_future_tcp"

    stove = get_task_policy(
        "libero_goal",
        "turn_on_the_stove_demo.hdf5",
    )
    assert stove.part_grounding.enabled is True
    assert stove.part_grounding.grounding_level == "part"
    assert "button" in stove.part_grounding.geom_patterns


def test_policy_table_validates_against_local_hdf5_names_when_available():
    root = Path("datasets/libero")
    if not root.exists():
        pytest.skip("official LIBERO HDF5 root is not present")
    missing = validate_policy_table(root)
    assert missing == []
```

- [ ] **Step 2: Run the mapping tests to verify they fail**

Run:

```bash
pytest tests/test_libero_target_mapping.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.preprocess.libero_target_mapping'`.

- [ ] **Step 3: Implement the mapping module**

Create `tools/preprocess/libero_target_mapping.py` with these public objects and the full 40-task table:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


LIBERO_STANDARD_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


@dataclass(frozen=True)
class TargetSpec:
    body_pattern: str
    kind: str = "object"


@dataclass(frozen=True)
class PartGroundingSpec:
    enabled: bool = False
    body_patterns: tuple[str, ...] = ()
    geom_patterns: tuple[str, ...] = ()
    grounding_level: str = "object"


@dataclass(frozen=True)
class TaskTargetPolicy:
    suite: str
    task_name: str
    candidate_targets: tuple[TargetSpec, ...]
    active_target_rule: str = "closest_pointcloud_to_future_tcp"
    fallback_target_pattern: str = ""
    part_grounding: PartGroundingSpec = field(default_factory=PartGroundingSpec)

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.task_name}"


def _policy(suite: str, task_name: str, candidates: tuple[str, ...], fallback: str, part: PartGroundingSpec | None = None) -> TaskTargetPolicy:
    return TaskTargetPolicy(
        suite=suite,
        task_name=task_name,
        candidate_targets=tuple(TargetSpec(c) for c in candidates),
        fallback_target_pattern=fallback,
        part_grounding=part or PartGroundingSpec(),
    )


DRAWER_PART = PartGroundingSpec(
    enabled=True,
    body_patterns=("wooden_cabinet",),
    geom_patterns=("handle", "drawer", "cabinet"),
    grounding_level="part",
)
STOVE_PART = PartGroundingSpec(
    enabled=True,
    body_patterns=("stove",),
    geom_patterns=("button", "knob"),
    grounding_level="part",
)
MICROWAVE_PART = PartGroundingSpec(
    enabled=True,
    body_patterns=("microwave",),
    geom_patterns=("handle", "door"),
    grounding_level="part",
)


_POLICIES = [
    _policy("libero_spatial", "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate", ("black_bowl", "wooden_cabinet"), "black_bowl", DRAWER_PART),
    _policy("libero_spatial", "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_object", "pick_up_the_alphabet_soup_and_place_it_in_the_basket", ("alphabet_soup",), "alphabet_soup"),
    _policy("libero_object", "pick_up_the_bbq_sauce_and_place_it_in_the_basket", ("bbq_sauce",), "bbq_sauce"),
    _policy("libero_object", "pick_up_the_butter_and_place_it_in_the_basket", ("butter",), "butter"),
    _policy("libero_object", "pick_up_the_chocolate_pudding_and_place_it_in_the_basket", ("chocolate_pudding",), "chocolate_pudding"),
    _policy("libero_object", "pick_up_the_cream_cheese_and_place_it_in_the_basket", ("cream_cheese",), "cream_cheese"),
    _policy("libero_object", "pick_up_the_ketchup_and_place_it_in_the_basket", ("ketchup",), "ketchup"),
    _policy("libero_object", "pick_up_the_milk_and_place_it_in_the_basket", ("milk",), "milk"),
    _policy("libero_object", "pick_up_the_orange_juice_and_place_it_in_the_basket", ("orange_juice",), "orange_juice"),
    _policy("libero_object", "pick_up_the_salad_dressing_and_place_it_in_the_basket", ("salad_dressing",), "salad_dressing"),
    _policy("libero_object", "pick_up_the_tomato_sauce_and_place_it_in_the_basket", ("tomato_sauce",), "tomato_sauce"),
    _policy("libero_goal", "open_the_middle_drawer_of_the_cabinet", ("wooden_cabinet",), "wooden_cabinet", DRAWER_PART),
    _policy("libero_goal", "open_the_top_drawer_and_put_the_bowl_inside", ("wooden_cabinet", "black_bowl"), "black_bowl", DRAWER_PART),
    _policy("libero_goal", "push_the_plate_to_the_front_of_the_stove", ("plate",), "plate"),
    _policy("libero_goal", "put_the_bowl_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_goal", "put_the_bowl_on_the_stove", ("black_bowl",), "black_bowl"),
    _policy("libero_goal", "put_the_bowl_on_top_of_the_cabinet", ("black_bowl",), "black_bowl"),
    _policy("libero_goal", "put_the_cream_cheese_in_the_bowl", ("cream_cheese",), "cream_cheese"),
    _policy("libero_goal", "put_the_wine_bottle_on_the_rack", ("wine_bottle",), "wine_bottle"),
    _policy("libero_goal", "put_the_wine_bottle_on_top_of_the_cabinet", ("wine_bottle",), "wine_bottle"),
    _policy("libero_goal", "turn_on_the_stove", ("stove",), "stove", STOVE_PART),
    _policy("libero_10", "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", ("stove", "moka_pot"), "moka_pot", STOVE_PART),
    _policy("libero_10", "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", ("black_bowl", "wooden_cabinet"), "black_bowl", DRAWER_PART),
    _policy("libero_10", "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it", ("yellow_and_white_mug", "microwave"), "yellow_and_white_mug", MICROWAVE_PART),
    _policy("libero_10", "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove", ("moka_pot",), "moka_pot"),
    _policy("libero_10", "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", ("alphabet_soup", "cream_cheese"), "alphabet_soup"),
    _policy("libero_10", "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket", ("alphabet_soup", "tomato_sauce"), "alphabet_soup"),
    _policy("libero_10", "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket", ("cream_cheese", "butter"), "cream_cheese"),
    _policy("libero_10", "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate", ("white_mug", "yellow_and_white_mug"), "white_mug"),
    _policy("libero_10", "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate", ("white_mug", "chocolate_pudding"), "white_mug"),
    _policy("libero_10", "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy", ("book",), "book"),
]

TASK_TARGET_POLICIES = {p.key: p for p in _POLICIES}


def normalize_task_name(path_or_name: str | Path) -> str:
    path = Path(path_or_name)
    suite = path.parent.name if path.parent.name in LIBERO_STANDARD_SUITES else ""
    stem = path.name
    if stem.endswith(".hdf5"):
        stem = stem[:-5]
    if stem.endswith("_demo"):
        stem = stem[:-5]
    return f"{suite}/{stem}" if suite else stem


def get_task_policy(suite: str, hdf5_name_or_task_name: str | Path) -> TaskTargetPolicy:
    task = normalize_task_name(Path(suite) / Path(hdf5_name_or_task_name).name)
    if task not in TASK_TARGET_POLICIES:
        raise KeyError(f"No curated LIBERO target policy for {task}")
    return TASK_TARGET_POLICIES[task]


def validate_policy_table(libero_root: str | Path) -> list[str]:
    root = Path(libero_root)
    missing: list[str] = []
    for suite in LIBERO_STANDARD_SUITES:
        for h5_path in sorted((root / suite).glob("*.hdf5")):
            key = normalize_task_name(h5_path)
            if key not in TASK_TARGET_POLICIES:
                missing.append(key)
    return missing
```

- [ ] **Step 4: Run mapping tests**

Run:

```bash
pytest tests/test_libero_target_mapping.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit mapping work**

Run:

```bash
git add tools/preprocess/libero_target_mapping.py tests/test_libero_target_mapping.py
git commit -m "feat(libero): add curated target mapping"
```

---

### Task 2: Pure Preprocessing Utilities

**Files:**
- Create: `tools/preprocess/libero_preprocess_utils.py`
- Test: `tests/test_libero_preprocess_utils.py`

- [ ] **Step 1: Write failing utility tests**

Create `tests/test_libero_preprocess_utils.py`:

```python
from __future__ import annotations

import numpy as np

from tools.preprocess.libero_preprocess_utils import (
    EpisodePlan,
    compute_episode_plan,
    gaussian_heatmap_from_pixel,
    mask_to_token_grid,
    pack_robot_obs,
    pointcloud_to_tcp_distance,
    select_render_gpus,
    smooth_active_targets,
)


def test_smooth_active_targets_removes_short_island():
    raw = ["A", "A", "A", "B", "A", "A", "C", "C", "C"]
    assert smooth_active_targets(raw, min_segment_len=2) == [
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "C",
        "C",
        "C",
    ]


def test_smooth_active_targets_keeps_real_switch():
    raw = ["A", "A", "A", "B", "B", "B"]
    assert smooth_active_targets(raw, min_segment_len=2) == raw


def test_mask_to_token_grid_shape_and_range():
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 1
    grid = mask_to_token_grid(mask, target_size=20)
    assert grid.shape == (1, 20, 20)
    assert grid.dtype == np.float32
    assert 0.0 <= float(grid.min()) <= float(grid.max()) <= 1.0
    assert float(grid[:, 8:12, 8:12].mean()) > 0.8


def test_gaussian_heatmap_from_pixel_peaks_near_expected_cell():
    heat = gaussian_heatmap_from_pixel(np.array([128.0, 128.0]), 256, 256, 20, sigma=1.5)
    assert heat.shape == (1, 20, 20)
    y, x = np.unravel_index(int(heat[0].argmax()), heat[0].shape)
    assert abs(x - 10) <= 1
    assert abs(y - 10) <= 1
    assert np.isclose(float(heat.max()), 1.0)


def test_pointcloud_to_tcp_distance():
    points = np.array([[0, 0, 0], [2, 0, 0], [5, 0, 0]], dtype=np.float32)
    tcp = np.array([[3, 0, 0]], dtype=np.float32)
    assert pointcloud_to_tcp_distance(points, tcp) == 1.0
    assert np.isinf(pointcloud_to_tcp_distance(np.zeros((0, 3), dtype=np.float32), tcp))


def test_pack_robot_obs_is_15d_float32():
    obs = pack_robot_obs(
        ee_pos=np.array([1, 2, 3]),
        ee_ori=np.array([4, 5, 6]),
        joint_states=np.arange(7),
        gripper_states=np.array([0.1, 0.2]),
    )
    assert obs.shape == (15,)
    assert obs.dtype == np.float32
    assert np.allclose(obs[:7], np.array([1, 2, 3, 4, 5, 6, 0], dtype=np.float32))


def test_compute_episode_plan_assigns_contiguous_indices():
    frame_counts = {
        "task_a.hdf5": [3, 2],
        "task_b.hdf5": [4],
    }
    plans = compute_episode_plan(frame_counts)
    assert plans == {
        ("task_a.hdf5", 0): EpisodePlan(episode_index=0, row_start=0, length=3),
        ("task_a.hdf5", 1): EpisodePlan(episode_index=1, row_start=3, length=2),
        ("task_b.hdf5", 0): EpisodePlan(episode_index=2, row_start=5, length=4),
    }


def test_select_render_gpus_defaults_to_visible_list():
    assert select_render_gpus("0,1,3", None) == ["0", "1", "3"]
    assert select_render_gpus("", None) == ["0"]
    assert select_render_gpus("0,1,2", "2,0") == ["2", "0"]
```

- [ ] **Step 2: Run utility tests to verify they fail**

Run:

```bash
pytest tests/test_libero_preprocess_utils.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.preprocess.libero_preprocess_utils'`.

- [ ] **Step 3: Implement pure helpers**

Create `tools/preprocess/libero_preprocess_utils.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EpisodePlan:
    episode_index: int
    row_start: int
    length: int


def smooth_active_targets(raw_targets: Sequence[str | None], min_segment_len: int = 3) -> list[str | None]:
    values = list(raw_targets)
    if min_segment_len <= 1 or len(values) < 3:
        return values

    segments: list[tuple[int, int, str | None]] = []
    start = 0
    for idx in range(1, len(values) + 1):
        if idx == len(values) or values[idx] != values[start]:
            segments.append((start, idx, values[start]))
            start = idx

    out = values[:]
    for seg_idx, (start, end, value) in enumerate(segments):
        length = end - start
        if length >= min_segment_len or value is None:
            continue
        if seg_idx == 0 or seg_idx == len(segments) - 1:
            continue
        prev_value = segments[seg_idx - 1][2]
        next_value = segments[seg_idx + 1][2]
        if prev_value is not None and prev_value == next_value:
            for i in range(start, end):
                out[i] = prev_value
    return out


def mask_to_token_grid(mask: np.ndarray, target_size: int = 20) -> np.ndarray:
    arr = (np.asarray(mask) > 0).astype(np.float32)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img = img.resize((target_size, target_size), Image.BILINEAR)
    grid = np.asarray(img, dtype=np.float32) / 255.0
    return grid[None, ...].astype(np.float32)


def gaussian_heatmap_from_pixel(pixel_xy: np.ndarray, image_width: int, image_height: int, target_size: int = 20, sigma: float = 1.5) -> np.ndarray:
    x, y = float(pixel_xy[0]), float(pixel_xy[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return np.zeros((1, target_size, target_size), dtype=np.float32)
    cx = (x + 0.5) * target_size / max(1, image_width) - 0.5
    cy = (y + 0.5) * target_size / max(1, image_height) - 0.5
    yy, xx = np.mgrid[0:target_size, 0:target_size].astype(np.float32)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    peak = float(heatmap.max())
    if peak > 0:
        heatmap /= peak
    return heatmap[None, ...].astype(np.float32)


def pointcloud_to_tcp_distance(point_cloud: np.ndarray, tcp_positions: np.ndarray) -> float:
    points = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
    tcp = np.asarray(tcp_positions, dtype=np.float32).reshape(-1, 3)
    if points.size == 0 or tcp.size == 0:
        return float("inf")
    distances = np.linalg.norm(points[:, None, :] - tcp[None, :, :], axis=-1)
    return float(distances.min())


def pack_robot_obs(ee_pos: np.ndarray, ee_ori: np.ndarray, joint_states: np.ndarray, gripper_states: np.ndarray) -> np.ndarray:
    obs = np.concatenate([
        np.asarray(ee_pos, dtype=np.float32).reshape(3),
        np.asarray(ee_ori, dtype=np.float32).reshape(3),
        np.asarray(joint_states, dtype=np.float32).reshape(7),
        np.asarray(gripper_states, dtype=np.float32).reshape(2),
    ])
    return obs.astype(np.float32)


def compute_episode_plan(frame_counts: Mapping[str, Sequence[int]]) -> dict[tuple[str, int], EpisodePlan]:
    plans: dict[tuple[str, int], EpisodePlan] = {}
    episode_index = 0
    row_start = 0
    for task_file in sorted(frame_counts):
        for demo_idx, length in enumerate(frame_counts[task_file]):
            plans[(task_file, demo_idx)] = EpisodePlan(
                episode_index=episode_index,
                row_start=row_start,
                length=int(length),
            )
            episode_index += 1
            row_start += int(length)
    return plans


def select_render_gpus(cuda_visible_devices: str | None, render_gpus: str | None) -> list[str]:
    if render_gpus:
        return [v.strip() for v in render_gpus.split(",") if v.strip()]
    if cuda_visible_devices:
        values = [v.strip() for v in cuda_visible_devices.split(",") if v.strip()]
        return values or ["0"]
    return ["0"]


def task_name_from_hdf5(path: str | Path) -> str:
    stem = Path(path).name
    if stem.endswith(".hdf5"):
        stem = stem[:-5]
    if stem.endswith("_demo"):
        stem = stem[:-5]
    return stem
```

- [ ] **Step 4: Run utility tests**

Run:

```bash
pytest tests/test_libero_preprocess_utils.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit utility work**

Run:

```bash
git add tools/preprocess/libero_preprocess_utils.py tests/test_libero_preprocess_utils.py
git commit -m "feat(libero): add preprocessing utility helpers"
```

---

### Task 3: LeRobot Writer And Sidecar Writer

**Files:**
- Create: `tools/preprocess/libero_lerobot_writer.py`
- Test: `tests/test_libero_lerobot_writer.py`

- [ ] **Step 1: Write failing writer tests**

Create `tests/test_libero_lerobot_writer.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tools.preprocess.libero_lerobot_writer import (
    LiberoEpisodeBuffers,
    LiberoLerobotWriter,
)


def _sample_row(ep: int, frame: int, index: int) -> dict:
    return {
        "episode_index": ep,
        "frame_index": frame,
        "timestamp": frame / 10.0,
        "index": index,
        "task_index": 0,
        "state.robot_obs": [0.0] * 15,
        "state.target_pose_rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "state.target_pose_trans": [0.0, 0.0, 0.0],
        "state.static_cam_rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "state.static_cam_trans": [0.0, 0.0, 0.0],
        "action.x": [0.0],
        "action.y": [0.0],
        "action.z": [0.0],
        "action.roll": [0.0],
        "action.pitch": [0.0],
        "action.yaw": [0.0],
        "action.gripper": [0.0],
        "annotation.human.action.task_description": "dummy task",
        "trajectory_id": ep,
        "base_index": frame,
    }


def test_writer_emits_lerobot_episode_and_sidecars(tmp_path: Path):
    writer = LiberoLerobotWriter(tmp_path, fps=10)
    rgb = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    buffers = LiberoEpisodeBuffers(
        episode_index=0,
        task_index=0,
        task_name="dummy task",
        rows=[_sample_row(0, 0, 0), _sample_row(0, 1, 1)],
        primary_frames=[rgb[0], rgb[1]],
        wrist_frames=[rgb[0], rgb[1]],
        image_targets=[rgb[0], None],
        point_clouds=[np.zeros((1024, 3), dtype=np.float32), None],
        depth_targets=[np.ones((32, 32), dtype=np.float32), np.ones((32, 32), dtype=np.float32)],
        grounding_masks=[np.ones((1, 20, 20), dtype=np.float32), None],
        grounding_levels=["object", None],
        affordance_heatmaps=[np.ones((1, 20, 20), dtype=np.float32), None],
    )
    writer.write_episode(buffers)
    writer.write_meta(
        tasks=["dummy task"],
        episode_lengths={0: 2},
        episode_to_task={0: 0},
        total_frames=2,
        coverage={"depth": {"valid": 2, "total": 2}},
    )
    assert (tmp_path / "data/chunk-000/episode_000000.parquet").exists()
    assert (tmp_path / "videos/chunk-000/video.primary_image/episode_000000.mp4").exists()
    assert (tmp_path / "videos/chunk-000/video.wrist_image/episode_000000.mp4").exists()
    assert (tmp_path / "image_targets/0/0.png").exists()
    assert not (tmp_path / "image_targets/0/1.png").exists()
    assert np.load(tmp_path / "point_clouds/0/0.npy").shape == (1024, 3)
    assert np.load(tmp_path / "depths/static/0/0.npy").shape == (32, 32)
    assert np.load(tmp_path / "grounding_masks/static/0/0.npy").shape == (1, 20, 20)
    with open(tmp_path / "grounding_masks/static/0/0.json") as f:
        assert json.load(f)["grounding_level"] == "object"
    table = pq.read_table(tmp_path / "data/chunk-000/episode_000000.parquet")
    assert table.num_rows == 2
    with open(tmp_path / "meta/info.json") as f:
        info = json.load(f)
    assert info["total_episodes"] == 1
    assert info["video_path"] == "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
```

- [ ] **Step 2: Run writer tests to verify they fail**

Run:

```bash
pytest tests/test_libero_lerobot_writer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.preprocess.libero_lerobot_writer'`.

- [ ] **Step 3: Implement writer module**

Create `tools/preprocess/libero_lerobot_writer.py` with:

```python
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


LEROBOT_CHUNK_SIZE = 1000
FPS = 10


@dataclass
class LiberoEpisodeBuffers:
    episode_index: int
    task_index: int
    task_name: str
    rows: list[dict[str, Any]]
    primary_frames: list[np.ndarray]
    wrist_frames: list[np.ndarray]
    image_targets: list[np.ndarray | None]
    point_clouds: list[np.ndarray | None]
    depth_targets: list[np.ndarray | None]
    grounding_masks: list[np.ndarray | None]
    grounding_levels: list[str | None]
    affordance_heatmaps: list[np.ndarray | None]


class LiberoLerobotWriter:
    def __init__(self, output_dir: str | Path, fps: int = FPS):
        self.output_dir = Path(output_dir)
        self.fps = int(fps)

    def write_episode(self, buffers: LiberoEpisodeBuffers) -> None:
        self._write_parquet(buffers.rows, buffers.episode_index)
        self._write_videos(buffers.primary_frames, buffers.wrist_frames, buffers.episode_index)
        self._write_sidecars(buffers)

    def _write_parquet(self, rows: list[dict[str, Any]], episode_index: int) -> None:
        chunk = episode_index // LEROBOT_CHUNK_SIZE
        path = self.output_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)

    def _write_videos(self, primary_frames: list[np.ndarray], wrist_frames: list[np.ndarray], episode_index: int) -> None:
        chunk = episode_index // LEROBOT_CHUNK_SIZE
        base = self.output_dir / "videos" / f"chunk-{chunk:03d}"
        primary_path = base / "video.primary_image" / f"episode_{episode_index:06d}.mp4"
        wrist_path = base / "video.wrist_image" / f"episode_{episode_index:06d}.mp4"
        primary_path.parent.mkdir(parents=True, exist_ok=True)
        wrist_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(primary_path, np.stack(primary_frames), fps=self.fps, codec="libx264")
        iio.imwrite(wrist_path, np.stack(wrist_frames), fps=self.fps, codec="libx264")

    def _write_sidecars(self, buffers: LiberoEpisodeBuffers) -> None:
        for frame_idx, target in enumerate(buffers.image_targets):
            if target is not None:
                path = self.output_dir / "image_targets" / str(buffers.episode_index) / f"{frame_idx}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.asarray(target, dtype=np.uint8)).save(path)
        for frame_idx, pc in enumerate(buffers.point_clouds):
            if pc is not None:
                arr = np.asarray(pc, dtype=np.float32)
                if arr.shape != (1024, 3):
                    raise RuntimeError(f"point cloud shape {arr.shape} != (1024, 3)")
                path = self.output_dir / "point_clouds" / str(buffers.episode_index) / f"{frame_idx}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, arr)
        for frame_idx, depth in enumerate(buffers.depth_targets):
            if depth is not None:
                path = self.output_dir / "depths" / "static" / str(buffers.episode_index) / f"{frame_idx}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.asarray(depth, dtype=np.float32))
        for frame_idx, mask in enumerate(buffers.grounding_masks):
            if mask is not None:
                path = self.output_dir / "grounding_masks" / "static" / str(buffers.episode_index) / f"{frame_idx}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.asarray(mask, dtype=np.float32))
                level = buffers.grounding_levels[frame_idx] or "object"
                with open(path.with_suffix(".json"), "w") as f:
                    json.dump({"grounding_level": level}, f)
        for frame_idx, heatmap in enumerate(buffers.affordance_heatmaps):
            if heatmap is not None:
                path = self.output_dir / "affordance_heatmaps" / "static" / str(buffers.episode_index) / f"{frame_idx}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.asarray(heatmap, dtype=np.float32))

    def write_camera_params(self, camera_params: dict[str, float | int | str]) -> None:
        with open(self.output_dir / "camera_params.json", "w") as f:
            json.dump(camera_params, f, indent=2)

    def write_meta(self, tasks: list[str], episode_lengths: dict[int, int], episode_to_task: dict[int, int], total_frames: int, coverage: dict[str, Any]) -> None:
        meta_dir = self.output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        self._write_modality(meta_dir)
        with open(meta_dir / "tasks.jsonl", "w") as f:
            for idx, task in enumerate(tasks):
                f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for ep_idx in sorted(episode_lengths):
                f.write(json.dumps({
                    "episode_index": ep_idx,
                    "tasks": [tasks[episode_to_task[ep_idx]]],
                    "length": int(episode_lengths[ep_idx]),
                }) + "\n")
        info = {
            "codebase_version": "v2.0",
            "robot_type": "uamvla_libero_franka_h8",
            "total_episodes": len(episode_lengths),
            "total_frames": int(total_frames),
            "total_tasks": len(tasks),
            "fps": self.fps,
            "splits": {"train": f"0:{len(episode_lengths)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "chunks_size": LEROBOT_CHUNK_SIZE,
            "features": {
                "video.primary_image": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"], "info": {"video.fps": self.fps, "video.channels": 3}},
                "video.wrist_image": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"], "info": {"video.fps": self.fps, "video.channels": 3}},
            },
        }
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        with open(meta_dir / "uamvla_aux_coverage.json", "w") as f:
            json.dump(coverage, f, indent=2)

    @staticmethod
    def _write_modality(meta_dir: Path) -> None:
        modality = {
            "state": {
                "robot_obs": {"start": 0, "end": 15, "original_key": "state.robot_obs"},
                "target_pose_rot6d": {"start": 0, "end": 6, "original_key": "state.target_pose_rot6d"},
                "target_pose_trans": {"start": 0, "end": 3, "original_key": "state.target_pose_trans"},
                "static_cam_rot6d": {"start": 0, "end": 6, "original_key": "state.static_cam_rot6d"},
                "static_cam_trans": {"start": 0, "end": 3, "original_key": "state.static_cam_trans"},
            },
            "action": {
                "x": {"start": 0, "end": 1, "original_key": "action.x"},
                "y": {"start": 0, "end": 1, "original_key": "action.y"},
                "z": {"start": 0, "end": 1, "original_key": "action.z"},
                "roll": {"start": 0, "end": 1, "original_key": "action.roll"},
                "pitch": {"start": 0, "end": 1, "original_key": "action.pitch"},
                "yaw": {"start": 0, "end": 1, "original_key": "action.yaw"},
                "gripper": {"start": 0, "end": 1, "original_key": "action.gripper"},
            },
            "video": {
                "primary_image": {"original_key": "video.primary_image"},
                "wrist_image": {"original_key": "video.wrist_image"},
            },
            "annotation": {
                "human.action.task_description": {"original_key": "task_index"},
            },
        }
        with open(meta_dir / "modality.json", "w") as f:
            json.dump(modality, f, indent=2)
```

- [ ] **Step 4: Run writer tests**

Run:

```bash
pytest tests/test_libero_lerobot_writer.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit writer work**

Run:

```bash
git add tools/preprocess/libero_lerobot_writer.py tests/test_libero_lerobot_writer.py
git commit -m "feat(libero): add LeRobot writer"
```

---

### Task 4: Preprocessor Planning And CLI Contract

**Files:**
- Modify: `runners/preprocess_libero.py`
- Modify: `tools/preprocess/run_libero_preprocess.py`
- Test: `tests/test_libero_preprocessor_planning.py`

- [ ] **Step 1: Write failing CLI/planning tests**

Create `tests/test_libero_preprocessor_planning.py`:

```python
from __future__ import annotations

from pathlib import Path

from runners.preprocess_libero import build_suite_jobs, default_output_name, parse_args


def test_default_output_name():
    assert default_output_name("libero_spatial") == "lerobot_libero_spatial"
    assert default_output_name("libero_10") == "lerobot_libero_10"


def test_build_suite_jobs_for_all_suites(tmp_path: Path):
    input_root = tmp_path / "libero"
    output_root = tmp_path / "libero2uam"
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        (input_root / suite).mkdir(parents=True)
    jobs = build_suite_jobs(input_root, output_root, "all")
    assert [(j.suite, j.input_dir.name, j.output_dir.name) for j in jobs] == [
        ("libero_spatial", "libero_spatial", "lerobot_libero_spatial"),
        ("libero_object", "libero_object", "lerobot_libero_object"),
        ("libero_goal", "libero_goal", "lerobot_libero_goal"),
        ("libero_10", "libero_10", "lerobot_libero_10"),
    ]


def test_parse_args_accepts_parallel_options():
    args = parse_args([
        "--input-root",
        "datasets/libero",
        "--output-root",
        "datasets/libero2uam",
        "--suite",
        "libero_goal",
        "--num-workers",
        "2",
        "--render-gpus",
        "0,1",
        "--max-tasks",
        "1",
        "--max-demos-per-task",
        "2",
        "--max-frames-per-demo",
        "3",
        "--overwrite",
    ])
    assert args.input_root == "datasets/libero"
    assert args.output_root == "datasets/libero2uam"
    assert args.suite == "libero_goal"
    assert args.num_workers == 2
    assert args.render_gpus == "0,1"
    assert args.max_tasks == 1
    assert args.max_demos_per_task == 2
    assert args.max_frames_per_demo == 3
    assert args.overwrite is True
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```bash
pytest tests/test_libero_preprocessor_planning.py -v
```

Expected: FAIL because `build_suite_jobs` and `default_output_name` do not exist.

- [ ] **Step 3: Update `runners/preprocess_libero.py`**

Replace the old target-resolver CLI with a LeRobot-oriented CLI containing these public helpers:

```python
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.preprocess.libero_preprocessor import LiberoPreprocessor
from tools.preprocess.libero_target_mapping import LIBERO_STANDARD_SUITES


@dataclass(frozen=True)
class SuiteJob:
    suite: str
    input_dir: Path
    output_dir: Path


def default_output_name(suite: str) -> str:
    return f"lerobot_{suite}"


def build_suite_jobs(input_root: Path, output_root: Path, suite: str) -> list[SuiteJob]:
    suites = LIBERO_STANDARD_SUITES if suite == "all" else (suite,)
    return [
        SuiteJob(
            suite=s,
            input_dir=input_root / s,
            output_dir=output_root / default_output_name(s),
        )
        for s in suites
    ]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Preprocess official LIBERO HDF5 into UamVLA LeRobot datasets.")
    parser.add_argument("--input-root", default="datasets/libero", help="Root containing libero_spatial/object/goal/10 HDF5 directories.")
    parser.add_argument("--output-root", default="datasets/libero2uam", help="Root where lerobot_libero_* datasets are written.")
    parser.add_argument("--suite", choices=("all",) + LIBERO_STANDARD_SUITES, default="all", help="Suite to process.")
    parser.add_argument("--num-workers", type=int, default=None, help="Worker count. Defaults to visible render GPU count.")
    parser.add_argument("--render-gpus", default=None, help="Comma-separated render GPU ids. Defaults to CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing suite output before preprocessing.")
    parser.add_argument("--min-segment-len", type=int, default=3, help="Minimum active-target segment length for smoothing.")
    parser.add_argument("--debug-rgb-check-frames", type=int, default=3, help="Number of replay-vs-HDF5 debug frames per task.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Bound the number of HDF5 task files for smoke runs.")
    parser.add_argument("--max-demos-per-task", type=int, default=None, help="Bound demos per task for smoke runs.")
    parser.add_argument("--max-frames-per-demo", type=int, default=None, help="Bound frames per demo for smoke runs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args(argv)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    jobs = build_suite_jobs(input_root, output_root, args.suite)
    for job in jobs:
        if job.output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output directory exists: {job.output_dir}. Pass --overwrite to replace it.")
            shutil.rmtree(job.output_dir)
        preprocessor = LiberoPreprocessor(
            suite=job.suite,
            num_workers=args.num_workers,
            render_gpus=args.render_gpus,
            min_segment_len=args.min_segment_len,
            debug_rgb_check_frames=args.debug_rgb_check_frames,
            max_tasks=args.max_tasks,
            max_demos_per_task=args.max_demos_per_task,
            max_frames_per_demo=args.max_frames_per_demo,
        )
        preprocessor.process(str(job.input_dir), str(job.output_dir))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update `tools/preprocess/run_libero_preprocess.py`**

Replace its body with a compatibility wrapper:

```python
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.preprocess_libero import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run CLI/planning tests**

Run:

```bash
pytest tests/test_libero_preprocessor_planning.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit CLI planning work**

Run:

```bash
git add runners/preprocess_libero.py tools/preprocess/run_libero_preprocess.py tests/test_libero_preprocessor_planning.py
git commit -m "feat(libero): update preprocessing CLI contract"
```

---

### Task 5: Replay Preprocessor Core

**Files:**
- Replace: `tools/preprocess/libero_preprocessor.py`
- Test: `tests/test_libero_replay_env_smoke.py`

- [ ] **Step 1: Write replay smoke test that skips outside `libero_env`**

Create `tests/test_libero_replay_env_smoke.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

libero = pytest.importorskip("libero.libero")
pytest.importorskip("robosuite")
h5py = pytest.importorskip("h5py")

from tools.preprocess.libero_preprocessor import LiberoPreprocessor


@pytest.mark.libero_env
def test_libero_preprocessor_env_probe():
    pre = LiberoPreprocessor(suite="libero_spatial", num_workers=1, render_gpus="0")
    result = pre.probe_replay_environment()
    assert result["libero_import"] is True
    assert result["render_ok"] is True


@pytest.mark.libero_env
def test_libero_preprocessor_processes_tiny_subset(tmp_path: Path):
    input_dir = Path("datasets/libero/libero_spatial")
    if not input_dir.exists() or not list(input_dir.glob("*.hdf5")):
        pytest.skip("official LIBERO HDF5 input is not available")
    output_dir = tmp_path / "lerobot_libero_spatial_smoke"
    pre = LiberoPreprocessor(
        suite="libero_spatial",
        num_workers=1,
        render_gpus="0",
        max_tasks=1,
        max_demos_per_task=1,
        max_frames_per_demo=4,
        debug_rgb_check_frames=1,
    )
    pre.process(str(input_dir), str(output_dir))
    assert (output_dir / "meta/info.json").exists()
    assert (output_dir / "data/chunk-000/episode_000000.parquet").exists()
    assert (output_dir / "videos/chunk-000/video.primary_image/episode_000000.mp4").exists()
    assert (output_dir / "meta/uamvla_aux_coverage.json").exists()
```

- [ ] **Step 2: Run replay smoke outside `libero_env`**

Run:

```bash
pytest tests/test_libero_replay_env_smoke.py -v
```

Expected outside `libero_env`: SKIP due to missing `libero.libero` or `robosuite`.

- [ ] **Step 3: Replace preprocessor imports and constructor**

Replace `tools/preprocess/libero_preprocessor.py` with a new module. The top-level imports must avoid importing LIBERO/robosuite at module import time:

```python
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from tools.preprocess.base_preprocessor import BasePreprocessor
from tools.preprocess.libero_lerobot_writer import (
    FPS,
    LiberoEpisodeBuffers,
    LiberoLerobotWriter,
)
from tools.preprocess.libero_preprocess_utils import (
    compute_episode_plan,
    gaussian_heatmap_from_pixel,
    mask_to_token_grid,
    pack_robot_obs,
    pointcloud_to_tcp_distance,
    select_render_gpus,
    smooth_active_targets,
    task_name_from_hdf5,
)
from tools.preprocess.libero_target_mapping import (
    TaskTargetPolicy,
    get_task_policy,
    validate_policy_table,
)
from starVLA.utils.geometry import (
    crop_target_from_seg,
    depth_to_world_points,
    get_camera_intrinsic_from_fovy,
    linearize_depth,
    mat_to_6d,
)


logger = logging.getLogger(__name__)

STATIC_CAM = "agentview"
WRIST_CAM = "robot0_eye_in_hand"
RENDER_W = 256
RENDER_H = 256
NUM_POINTS = 1024


@dataclass(frozen=True)
class TaskJob:
    suite: str
    hdf5_path: Path
    output_dir: Path
    task_index: int
    demo_episode_indices: dict[int, int]
    demo_row_starts: dict[int, int]
    gpu_id: str
    min_segment_len: int
    debug_rgb_check_frames: int
    max_demos_per_task: int | None = None
    max_frames_per_demo: int | None = None


class LiberoPreprocessor(BasePreprocessor):
    def __init__(
        self,
        suite: str = "libero_spatial",
        num_workers: int | None = None,
        render_gpus: str | None = None,
        min_segment_len: int = 3,
        debug_rgb_check_frames: int = 3,
        max_tasks: int | None = None,
        max_demos_per_task: int | None = None,
        max_frames_per_demo: int | None = None,
    ):
        self.suite = suite
        self.num_workers = num_workers
        self.render_gpus = render_gpus
        self.min_segment_len = int(min_segment_len)
        self.debug_rgb_check_frames = int(debug_rgb_check_frames)
        self.max_tasks = max_tasks
        self.max_demos_per_task = max_demos_per_task
        self.max_frames_per_demo = max_frames_per_demo
```

- [ ] **Step 4: Implement process orchestration**

Add `process`, `_scan_frame_counts`, and `_build_jobs` methods:

```python
    def process(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"LIBERO input directory not found: {input_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        missing = validate_policy_table(input_path.parent)
        suite_missing = [m for m in missing if m.startswith(f"{self.suite}/")]
        if suite_missing:
            raise RuntimeError(f"Missing curated target policies for {suite_missing}")

        hdf5_files = sorted(input_path.glob("*.hdf5"))
        if self.max_tasks is not None:
            hdf5_files = hdf5_files[: self.max_tasks]
        if not hdf5_files:
            raise FileNotFoundError(f"No .hdf5 files found in {input_path}")

        frame_counts = self._scan_frame_counts(hdf5_files)
        episode_plan = compute_episode_plan(frame_counts)
        render_gpus = select_render_gpus(os.environ.get("CUDA_VISIBLE_DEVICES"), self.render_gpus)
        worker_count = self.num_workers if self.num_workers is not None else len(render_gpus)
        worker_count = max(1, int(worker_count))
        jobs = self._build_jobs(hdf5_files, output_path, episode_plan, render_gpus)

        if worker_count == 1:
            results = [_run_task_job(job) for job in jobs]
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=worker_count) as pool:
                results = pool.map(_run_task_job, jobs)

        writer = LiberoLerobotWriter(output_path, fps=FPS)
        tasks = [task_name_from_hdf5(p) for p in hdf5_files]
        episode_lengths: dict[int, int] = {}
        episode_to_task: dict[int, int] = {}
        total_frames = 0
        coverage = {"tasks": {}, "totals": {}}
        camera_params = None
        for result in results:
            total_frames += int(result["total_frames"])
            episode_lengths.update({int(k): int(v) for k, v in result["episode_lengths"].items()})
            episode_to_task.update({int(k): int(v) for k, v in result["episode_to_task"].items()})
            coverage["tasks"][result["task_name"]] = result["coverage"]
            camera_params = camera_params or result.get("camera_params")
        coverage["totals"] = _sum_coverage(coverage["tasks"])
        writer.write_meta(tasks, episode_lengths, episode_to_task, total_frames, coverage)
        if camera_params is not None:
            writer.write_camera_params(camera_params)

    def _scan_frame_counts(self, hdf5_files: list[Path]) -> dict[str, list[int]]:
        counts: dict[str, list[int]] = {}
        for h5_path in hdf5_files:
            with h5py.File(h5_path, "r") as f:
                demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
                if self.max_demos_per_task is not None:
                    demo_keys = demo_keys[: self.max_demos_per_task]
                lengths = []
                for demo_key in demo_keys:
                    length = int(f[f"data/{demo_key}/actions"].shape[0])
                    if self.max_frames_per_demo is not None:
                        length = min(length, int(self.max_frames_per_demo))
                    lengths.append(length)
                counts[h5_path.name] = lengths
        return counts

    def _build_jobs(self, hdf5_files: list[Path], output_path: Path, episode_plan: dict[tuple[str, int], Any], render_gpus: list[str]) -> list[TaskJob]:
        jobs = []
        for task_idx, h5_path in enumerate(hdf5_files):
            demo_episode_indices = {}
            demo_row_starts = {}
            for (filename, demo_idx), plan in episode_plan.items():
                if filename == h5_path.name:
                    demo_episode_indices[demo_idx] = plan.episode_index
                    demo_row_starts[demo_idx] = plan.row_start
            gpu_id = render_gpus[task_idx % len(render_gpus)]
            jobs.append(TaskJob(
                suite=self.suite,
                hdf5_path=h5_path,
                output_dir=output_path,
                task_index=task_idx,
                demo_episode_indices=demo_episode_indices,
                demo_row_starts=demo_row_starts,
                gpu_id=gpu_id,
                min_segment_len=self.min_segment_len,
                debug_rgb_check_frames=self.debug_rgb_check_frames,
                max_demos_per_task=self.max_demos_per_task,
                max_frames_per_demo=self.max_frames_per_demo,
            ))
        return jobs
```

- [ ] **Step 5: Implement worker setup and replay functions**

Add module-level worker functions to keep multiprocessing pickle-friendly:

```python
def _configure_worker_env(gpu_id: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _run_task_job(job: TaskJob) -> dict[str, Any]:
    _configure_worker_env(job.gpu_id)
    from libero.libero.envs import OffScreenRenderEnv

    policy = get_task_policy(job.suite, job.hdf5_path.name)
    bddl_path = _resolve_bddl_path(job.suite, job.hdf5_path)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        robots=["Panda"],
        controller="OSC_POSE",
        camera_names=[STATIC_CAM, WRIST_CAM],
        camera_heights=RENDER_H,
        camera_widths=RENDER_W,
        camera_depths=True,
        camera_segmentations="instance",
    )
    env.reset()
    try:
        worker = _TaskReplayWorker(job, env, policy)
        return worker.run()
    finally:
        env.close()
```

Add `_resolve_bddl_path` equivalent to the old preprocessor but module-level:

```python
def _resolve_bddl_path(suite_name: str, hdf5_file: Path) -> str:
    from libero.libero import benchmark, get_libero_path

    task_name = task_name_from_hdf5(hdf5_file)
    bench_dict = benchmark.get_benchmark_dict()
    suite = bench_dict[suite_name]()
    bddl_root = get_libero_path("bddl_files")
    for idx in range(suite.n_tasks):
        task = suite.get_task(idx)
        if task.name == task_name:
            return str(Path(bddl_root) / task.problem_folder / task.bddl_file)
    raise ValueError(f"Task {task_name!r} not found in benchmark suite {suite_name!r}")
```

- [ ] **Step 6: Implement `_TaskReplayWorker`**

Add a private worker class with methods for schema validation, rendering, candidate matching, object masks, point clouds, row assembly, and debug RGB checks. Keep the following public-to-module method names so tests and future reviews can target focused pieces:

```python
class _TaskReplayWorker:
    def __init__(self, job: TaskJob, env, policy: TaskTargetPolicy):
        self.job = job
        self.env = env
        self.policy = policy
        self.writer = LiberoLerobotWriter(job.output_dir, fps=FPS)
        self.task_name = task_name_from_hdf5(job.hdf5_path)

    def run(self) -> dict[str, Any]:
        results = {
            "task_name": self.task_name,
            "total_frames": 0,
            "episode_lengths": {},
            "episode_to_task": {},
            "coverage": self._empty_coverage(),
            "camera_params": self._camera_params(),
        }
        with h5py.File(self.job.hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
            if self.job.max_demos_per_task is not None:
                demo_keys = demo_keys[: self.job.max_demos_per_task]
            for demo_key in demo_keys:
                demo_idx = int(demo_key.split("_")[1])
                buffers = self._process_demo(f[f"data/{demo_key}"], demo_idx)
                self.writer.write_episode(buffers)
                results["total_frames"] += len(buffers.rows)
                results["episode_lengths"][buffers.episode_index] = len(buffers.rows)
                results["episode_to_task"][buffers.episode_index] = self.job.task_index
                self._merge_coverage(results["coverage"], buffers)
        return results
```

The plan for `_process_demo` is:

1. Read `actions`, `states`, `obs/ee_pos`, `obs/ee_ori`, `obs/joint_states`, `obs/gripper_states`.
2. Limit to `max_frames_per_demo` if set.
3. Replay each frame once and cache candidate masks/point clouds/scores.
4. Smooth active target ids.
5. Build final per-frame buffers with RGB, sidecars, parquet rows.

Use the helper names already defined in Tasks 1-3. Reuse `depth_to_world_points`, `crop_target_from_seg`, `linearize_depth`, and `mat_to_6d` from `starVLA.utils.geometry`.

- [ ] **Step 7: Add `probe_replay_environment`**

Implement:

```python
    def probe_replay_environment(self) -> dict[str, bool]:
        gpu = select_render_gpus(os.environ.get("CUDA_VISIBLE_DEVICES"), self.render_gpus)[0]
        _configure_worker_env(gpu)
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        import robosuite

        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        bddl = suite.get_task_bddl_file_path(0)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            robots=["Panda"],
            controller="OSC_POSE",
            camera_names=[STATIC_CAM],
            camera_heights=64,
            camera_widths=64,
            camera_depths=True,
        )
        try:
            env.reset()
            rgb = env.sim.render(camera_name=STATIC_CAM, width=64, height=64)
            return {"libero_import": True, "robosuite_import": robosuite is not None, "render_ok": rgb is not None}
        finally:
            env.close()
```

- [ ] **Step 8: Run non-LIBERO tests and replay skip**

Run:

```bash
pytest tests/test_libero_target_mapping.py tests/test_libero_preprocess_utils.py tests/test_libero_lerobot_writer.py tests/test_libero_preprocessor_planning.py tests/test_libero_replay_env_smoke.py -v
```

Expected outside `libero_env`: helper tests PASS, replay env smoke SKIP.

- [ ] **Step 9: Commit preprocessor core**

Run:

```bash
git add tools/preprocess/libero_preprocessor.py tests/test_libero_replay_env_smoke.py
git commit -m "feat(libero): replay HDF5 into LeRobot episodes"
```

---

### Task 6: LIBERO UamVLA Registry And Config

**Files:**
- Modify: `examples/LIBERO/train_files/data_registry/data_config.py`
- Create: `starVLA/config/training/uamvla_gr00t_libero.yaml`
- Test: `tests/dataloader/test_uamvla_libero_registry.py`

- [ ] **Step 1: Write failing registry/config tests**

Create `tests/dataloader/test_uamvla_libero_registry.py`:

```python
from __future__ import annotations

from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
)


def test_uamvla_libero_h8_registry_entries_exist():
    assert "uamvla_libero_franka_h8" in ROBOT_TYPE_CONFIG_MAP
    assert "uamvla_libero_all_h8" in DATASET_NAMED_MIXTURES
    assert DATASET_NAMED_MIXTURES["uamvla_libero_all_h8"] == [
        ("lerobot_libero_object", 1.0, "uamvla_libero_franka_h8"),
        ("lerobot_libero_goal", 1.0, "uamvla_libero_franka_h8"),
        ("lerobot_libero_spatial", 1.0, "uamvla_libero_franka_h8"),
        ("lerobot_libero_10", 1.0, "uamvla_libero_franka_h8"),
    ]


def test_uamvla_libero_data_config_shapes():
    cfg = ROBOT_TYPE_CONFIG_MAP["uamvla_libero_franka_h8"]
    modality = cfg.modality_config()
    assert modality["video"].delta_indices == [0]
    assert modality["action"].delta_indices == list(range(8))
    assert modality["state"].delta_indices == [0]
    assert cfg.state_keys == [
        "state.robot_obs",
        "state.target_pose_rot6d",
        "state.target_pose_trans",
        "state.static_cam_rot6d",
        "state.static_cam_trans",
    ]


def test_uamvla_gr00t_libero_yaml_loads():
    cfg = OmegaConf.load("starVLA/config/training/uamvla_gr00t_libero.yaml")
    assert cfg.framework.name == "UamVLAGR00T"
    assert cfg.datasets.vla_data.data_root_dir == "datasets/libero2uam"
    assert cfg.datasets.vla_data.data_mix == "uamvla_libero_all_h8"
    assert cfg.framework.action_model.state_dim == 7
    assert cfg.framework.action_model.action_horizon == 8
    assert cfg.datasets.vla_data.aux_state_slice.target_pose_rot6d == [15, 21]
```

- [ ] **Step 2: Run registry/config tests to verify they fail**

Run:

```bash
pytest tests/dataloader/test_uamvla_libero_registry.py -v
```

Expected: FAIL because `uamvla_libero_franka_h8` and `uamvla_gr00t_libero.yaml` do not exist.

- [ ] **Step 3: Modify `examples/LIBERO/train_files/data_registry/data_config.py`**

Add a UamVLA-specific config next to the existing LIBERO config:

```python
class UamVLALiberoH8DataConfig:
    video_keys = ["video.primary_image", "video.wrist_image"]
    state_keys = [
        "state.robot_obs",
        "state.target_pose_rot6d",
        "state.target_pose_trans",
        "state.static_cam_rot6d",
        "state.static_cam_trans",
    ]
    action_keys = [
        "action.x", "action.y", "action.z",
        "action.roll", "action.pitch", "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    action_horizon = 8
    action_indices = list(range(action_horizon))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=[0], modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=[0], modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=[0], modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={k: "min_max" for k in self.action_keys[:-1]},
            ),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=["state.robot_obs"],
                normalization_modes={"state.robot_obs": "mean_std"},
            ),
        ])
```

Extend registries:

```python
ROBOT_TYPE_CONFIG_MAP.update({
    "uamvla_libero_franka_h8": UamVLALiberoH8DataConfig(),
})

ROBOT_TYPE_TO_EMBODIMENT_TAG.update({
    "uamvla_libero_franka_h8": EmbodimentTag.FRANKA,
})

DATASET_NAMED_MIXTURES.update({
    "uamvla_libero_all_h8": [
        ("lerobot_libero_object", 1.0, "uamvla_libero_franka_h8"),
        ("lerobot_libero_goal", 1.0, "uamvla_libero_franka_h8"),
        ("lerobot_libero_spatial", 1.0, "uamvla_libero_franka_h8"),
        ("lerobot_libero_10", 1.0, "uamvla_libero_franka_h8"),
    ],
    "uamvla_libero_spatial_h8": [
        ("lerobot_libero_spatial", 1.0, "uamvla_libero_franka_h8"),
    ],
    "uamvla_libero_object_h8": [
        ("lerobot_libero_object", 1.0, "uamvla_libero_franka_h8"),
    ],
    "uamvla_libero_goal_h8": [
        ("lerobot_libero_goal", 1.0, "uamvla_libero_franka_h8"),
    ],
    "uamvla_libero_10_h8": [
        ("lerobot_libero_10", 1.0, "uamvla_libero_franka_h8"),
    ],
})
```

- [ ] **Step 4: Create `starVLA/config/training/uamvla_gr00t_libero.yaml`**

Copy `starVLA/config/training/uamvla_gr00t_calvin_d.yaml` and change these fields:

```yaml
run_id: uamvla_gr00t_libero_all_4b_h8

framework:
  name: UamVLAGR00T
  action_model:
    action_dim: 7
    state_dim: 7
    action_horizon: 8
    future_action_window_size: 7

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: datasets/libero2uam
    data_mix: uamvla_libero_all_h8
    action_type: delta_eef
    per_device_batch_size: 1
    obs: ["video.primary_image", "video.wrist_image"]
    include_state: true
    image_resize: 640
    aux_state_slice:
      target_pose_rot6d: [15, 21]
      target_pose_trans: [21, 24]
      static_cam_rot6d:  [24, 30]
      static_cam_trans:  [30, 33]
```

Keep the aux-head blocks from the CALVIN config so the config is full-capable.

- [ ] **Step 5: Run registry/config tests**

Run:

```bash
pytest tests/dataloader/test_uamvla_libero_registry.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit registry/config work**

Run:

```bash
git add examples/LIBERO/train_files/data_registry/data_config.py starVLA/config/training/uamvla_gr00t_libero.yaml tests/dataloader/test_uamvla_libero_registry.py
git commit -m "feat(libero): register UamVLA GR00T data config"
```

---

### Task 7: GR00T State Compatibility Test

**Files:**
- Test: `tests/framework/test_uamvla_gr00t_libero_state.py`

- [ ] **Step 1: Write GR00T 33-D packed state test**

Create `tests/framework/test_uamvla_gr00t_libero_state.py`:

```python
from __future__ import annotations

import numpy as np

from starVLA.model.framework.VLM4A.UamVLAGR00T import UamVLAGR00T


def test_gr00t_extracts_first_7_dims_from_libero_33d_state():
    packed = np.arange(33, dtype=np.float32).reshape(1, 33)
    state = UamVLAGR00T._extract_gr00t_state_from_packed_calvin_state(packed)
    assert tuple(state.shape) == (1, 7)
    assert np.allclose(state.numpy(), np.arange(7, dtype=np.float32).reshape(1, 7))
```

- [ ] **Step 2: Run GR00T state test**

Run:

```bash
pytest tests/framework/test_uamvla_gr00t_libero_state.py -v
```

Expected: PASS, because the existing helper already extracts `state[..., :7]`.

- [ ] **Step 3: Commit test**

Run:

```bash
git add tests/framework/test_uamvla_gr00t_libero_state.py
git commit -m "test(libero): cover GR00T packed state extraction"
```

---

### Task 8: End-To-End Verification Commands

**Files:**
- Modify only if a previous task uncovered a small import issue.

- [ ] **Step 1: Run fast non-replay tests**

Run:

```bash
pytest \
  tests/test_libero_target_mapping.py \
  tests/test_libero_preprocess_utils.py \
  tests/test_libero_lerobot_writer.py \
  tests/test_libero_preprocessor_planning.py \
  tests/dataloader/test_uamvla_libero_registry.py \
  tests/framework/test_uamvla_gr00t_libero_state.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run replay smoke inside `libero_env`**

Run:

```bash
conda run -n libero_env pytest tests/test_libero_replay_env_smoke.py -v
```

Expected in a correctly installed `libero_env`: PASS. If `libero_env` is not installed, the command should report import-related skips or failures that identify the missing package.

- [ ] **Step 3: Run tiny real preprocessing smoke inside `libero_env`**

Run:

```bash
conda run -n libero_env python runners/preprocess_libero.py \
  --input-root datasets/libero \
  --output-root /tmp/libero2uam_smoke \
  --suite libero_spatial \
  --num-workers 1 \
  --render-gpus 0 \
  --max-tasks 1 \
  --max-demos-per-task 1 \
  --max-frames-per-demo 4 \
  --overwrite
```

Expected: `/tmp/libero2uam_smoke/lerobot_libero_spatial/meta/info.json` exists and the first task's first demo writes four frames of videos/parquet plus available sidecars.

- [ ] **Step 4: Run config load smoke**

Run:

```bash
python - <<'PY'
from omegaconf import OmegaConf
cfg = OmegaConf.load("starVLA/config/training/uamvla_gr00t_libero.yaml")
print(cfg.framework.name, cfg.datasets.vla_data.data_mix, cfg.framework.action_model.state_dim)
PY
```

Expected output:

```text
UamVLAGR00T uamvla_libero_all_h8 7
```

- [ ] **Step 5: Final status check**

Run:

```bash
git status --short
```

Expected: clean working tree after all task commits.
