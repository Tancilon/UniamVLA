"""Unit tests for tools.preprocess.calvin_task_map and its wiring into
CalvinWorker.process_window.

Two surfaces:
  - `resolve_target_object(task_label)` — pure dict lookup for the 32
    deterministic CALVIN tasks; MUST refuse stack/unstack tasks because
    their target depends on the trajectory.
  - `infer_stack_block(task_label, scene_letter, scene_obs_start,
    scene_obs_end)` — runtime inference for `stack_block` /
    `unstack_block`: the block being manipulated is the one whose xyz
    position changes most across the language window. Scene letter is
    required because CALVIN scenes A/B/C/D each list `movable_objects` in
    a different order in their scene yaml, and that order drives the
    block→slot mapping inside scene_obs.

Plus integration tests that drive `CalvinWorker.process_window` with a
duck-typed fake env and a real `SceneResolver`, confirming the
preprocessor wires inference in (and that the scene-letter context
propagates) instead of the old hardcoded `block_red` mapping.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# Scene-obs structure (CALVIN, euler_obs=True, 24-dim):
#   [0]     slider
#   [1]     drawer
#   [2]     button
#   [3]     switch
#   [4]     lightbulb
#   [5]     led
#   [6:12]  movable block #1 (xyz, euler) per scene yaml order
#   [12:18] movable block #2
#   [18:24] movable block #3
#
# Block-order per scene (read from third_party/calvin/calvin_env/conf/scene/
# calvin_scene_<X>.yaml's movable_objects dict insertion order):
SCENE_ORDER = {
    "A": ("block_pink", "block_blue", "block_red"),
    "B": ("block_red",  "block_blue", "block_pink"),
    "C": ("block_blue", "block_red",  "block_pink"),
    "D": ("block_red",  "block_blue", "block_pink"),
}


def _make_scene_obs(scene_letter: str, **block_xyz) -> np.ndarray:
    """Build a 24-dim scene_obs respecting `scene_letter`'s block order.

    Pass any subset of pink_xyz / blue_xyz / red_xyz as 3-tuples; blocks
    not provided get (0, 0, 0). Euler slots stay zero (inference uses
    xyz only).
    """
    order = SCENE_ORDER[scene_letter.upper()]
    arr = np.zeros(24, dtype=np.float32)
    for i, name in enumerate(order):
        color = name.replace("block_", "")
        xyz = block_xyz.get(f"{color}_xyz", (0.0, 0.0, 0.0))
        lo = 6 + i * 6
        arr[lo:lo + 3] = xyz
    return arr


# ---------- resolve_target_object: stack tasks rejected ------------------

def test_resolve_target_object_rejects_unstack_block():
    """`unstack_block` was previously hardcoded to `block_red`, mislabelling
    ~2/3 of unstack windows (pink-on-blue, blue-on-pink, etc.). The static
    resolver must now refuse the task so callers route through the runtime
    inference path.
    """
    from tools.preprocess.calvin_task_map import resolve_target_object
    with pytest.raises(ValueError, match="infer_stack_block"):
        resolve_target_object("unstack_block")


def test_resolve_target_object_rejects_stack_block():
    """Same as unstack: stack_block's target depends on the trajectory."""
    from tools.preprocess.calvin_task_map import resolve_target_object
    with pytest.raises(ValueError, match="infer_stack_block"):
        resolve_target_object("stack_block")


def test_resolve_target_object_returns_expected_for_every_known_task():
    """Sanity: every entry in CALVIN_TASK_TO_OBJECT round-trips through
    `resolve_target_object` to the same value. Guards against a future
    edit that removes a fixture-task entry by accident."""
    from tools.preprocess.calvin_task_map import (
        CALVIN_TASK_TO_OBJECT,
        resolve_target_object,
    )
    assert len(CALVIN_TASK_TO_OBJECT) == 32, (
        "CALVIN has 34 task labels; 32 are color-/fixture-specific and "
        "must live in CALVIN_TASK_TO_OBJECT. stack_block and unstack_block "
        "are color-agnostic and inferred at runtime instead."
    )
    for task, expected in CALVIN_TASK_TO_OBJECT.items():
        assert resolve_target_object(task) == expected, task


# ---------- infer_stack_block: per-scene block-slot order ----------------

@pytest.mark.parametrize("scene_letter", ["A", "B", "C", "D"])
def test_infer_stack_block_picks_moved_red_across_all_scenes(scene_letter):
    """Red block lifted off a stack; pink and blue untouched. Must work
    regardless of which slot red occupies in this scene's scene_obs."""
    from tools.preprocess.calvin_task_map import infer_stack_block

    start = _make_scene_obs(
        scene_letter,
        pink_xyz=(0.10, -0.05, 0.46),
        blue_xyz=(0.20, -0.05, 0.46),
        red_xyz =(0.20, -0.05, 0.50),
    )
    end = _make_scene_obs(
        scene_letter,
        pink_xyz=(0.10, -0.05, 0.46),
        blue_xyz=(0.20, -0.05, 0.46),
        red_xyz =(-0.05, 0.10, 0.65),
    )
    assert infer_stack_block(
        "unstack_block", scene_letter, start, end,
    ) == "block_red"


@pytest.mark.parametrize("scene_letter", ["A", "B", "C", "D"])
def test_infer_stack_block_picks_moved_blue_across_all_scenes(scene_letter):
    """The motivating bug: language 'remove the top block' on a window
    where blue is the actual top block. Old hardcoded mapping returned
    block_red; correct answer is block_blue."""
    from tools.preprocess.calvin_task_map import infer_stack_block

    start = _make_scene_obs(
        scene_letter,
        pink_xyz=(0.10, -0.05, 0.46),
        blue_xyz=(0.10, -0.05, 0.50),
        red_xyz =(0.05,  0.18, 0.36),
    )
    end = _make_scene_obs(
        scene_letter,
        pink_xyz=(0.10, -0.05, 0.46),
        blue_xyz=(-0.10, 0.10, 0.62),
        red_xyz =(0.05,  0.18, 0.36),
    )
    assert infer_stack_block(
        "unstack_block", scene_letter, start, end,
    ) == "block_blue"


@pytest.mark.parametrize("scene_letter", ["A", "B", "C", "D"])
def test_infer_stack_block_picks_moved_pink_across_all_scenes(scene_letter):
    """Pink stacked on red, then lifted away."""
    from tools.preprocess.calvin_task_map import infer_stack_block

    start = _make_scene_obs(
        scene_letter,
        pink_xyz=(0.20, -0.05, 0.50),
        blue_xyz=(0.05,  0.18, 0.36),
        red_xyz =(0.20, -0.05, 0.46),
    )
    end = _make_scene_obs(
        scene_letter,
        pink_xyz=(-0.15, 0.08, 0.65),
        blue_xyz=(0.05,  0.18, 0.36),
        red_xyz =(0.20, -0.05, 0.46),
    )
    assert infer_stack_block(
        "unstack_block", scene_letter, start, end,
    ) == "block_pink"


@pytest.mark.parametrize("scene_letter", ["A", "B", "C", "D"])
@pytest.mark.parametrize("moved_color", ["red", "blue", "pink"])
def test_infer_stack_block_stack_task_per_color_per_scene(
    scene_letter, moved_color,
):
    """`stack_block`: the block being PLACED on another is the manipulated
    one. Same xyz-displacement heuristic, exercised across every scene ×
    every block color."""
    from tools.preprocess.calvin_task_map import infer_stack_block

    starts = {
        "red":  (0.30, -0.05, 0.46),
        "blue": (0.10, -0.05, 0.46),
        "pink": (-0.20, 0.10, 0.46),
    }
    moved_end = (0.10, -0.05, 0.50)  # placed on top of where blue starts
    start_kwargs = {f"{c}_xyz": starts[c] for c in ("red", "blue", "pink")}
    end_kwargs = dict(start_kwargs)
    end_kwargs[f"{moved_color}_xyz"] = moved_end

    start = _make_scene_obs(scene_letter, **start_kwargs)
    end = _make_scene_obs(scene_letter, **end_kwargs)
    assert infer_stack_block(
        "stack_block", scene_letter, start, end,
    ) == f"block_{moved_color}"


def test_infer_stack_block_ignores_sub_mm_jitter_on_uninvolved_blocks():
    """Physics simulation can introduce sub-mm jitter on the two blocks
    the robot didn't touch. The manipulated block's ~tens-of-cm delta
    must still win cleanly."""
    from tools.preprocess.calvin_task_map import infer_stack_block

    start = _make_scene_obs(
        "A",
        pink_xyz=(0.10, -0.05, 0.46),
        blue_xyz=(0.20, -0.05, 0.46),
        red_xyz =(0.20, -0.05, 0.50),
    )
    end = _make_scene_obs(
        "A",
        pink_xyz=(0.1003, -0.0498, 0.4602),
        blue_xyz=(0.2001, -0.0501, 0.4599),
        red_xyz =(-0.05,   0.12,   0.65),
    )
    assert infer_stack_block(
        "unstack_block", "A", start, end,
    ) == "block_red"


def test_infer_stack_block_accepts_lowercase_scene_letter():
    """CLI surfaces sometimes pass scene letters as lowercase; the
    inference function normalizes."""
    from tools.preprocess.calvin_task_map import infer_stack_block

    start = _make_scene_obs("D")
    end = _make_scene_obs("D", red_xyz=(0.5, 0.5, 0.5))
    assert infer_stack_block(
        "unstack_block", "d", start, end,
    ) == "block_red"


# ---------- infer_stack_block: invalid inputs ----------------------------

def test_infer_stack_block_rejects_non_stack_task():
    from tools.preprocess.calvin_task_map import infer_stack_block

    start = _make_scene_obs("A")
    end = _make_scene_obs("A")
    with pytest.raises(ValueError, match="stack"):
        infer_stack_block("lift_red_block_table", "A", start, end)


def test_infer_stack_block_rejects_unknown_scene_letter():
    from tools.preprocess.calvin_task_map import infer_stack_block

    start = _make_scene_obs("A")
    end = _make_scene_obs("A")
    with pytest.raises(ValueError, match="scene"):
        infer_stack_block("unstack_block", "Z", start, end)


def test_infer_stack_block_rejects_wrong_scene_obs_shape():
    """Defensive: the CALVIN ABCD_D dumps this repo handles are 24-dim
    (euler). A 27-dim (quaternion) dump or a flattened multi-frame array
    must fail loudly rather than silently slicing wrong block slots."""
    from tools.preprocess.calvin_task_map import infer_stack_block

    bad_start = np.zeros(27, dtype=np.float32)
    end = _make_scene_obs("A")
    with pytest.raises(ValueError, match="shape"):
        infer_stack_block("unstack_block", "A", bad_start, end)

    start = _make_scene_obs("A")
    bad_end = np.zeros((2, 24), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        infer_stack_block("unstack_block", "A", start, bad_end)


def test_infer_stack_block_rejects_non_finite_values():
    """NaN/Inf in scene_obs would silently flow into np.linalg.norm and
    yield a meaningless 'displacement'. Fail loudly instead."""
    from tools.preprocess.calvin_task_map import infer_stack_block

    start = _make_scene_obs("A")
    end = _make_scene_obs("A")
    end[6] = np.nan
    with pytest.raises(ValueError, match="finite"):
        infer_stack_block("unstack_block", "A", start, end)

    end2 = _make_scene_obs("A")
    end2[12] = np.inf
    with pytest.raises(ValueError, match="finite"):
        infer_stack_block("unstack_block", "A", start, end2)


# ---------- CalvinWorker integration: stack-task path --------------------

class _FakeBlocksEnv:
    """Minimal duck-typed env exposing the 5-method surface CalvinWorker
    touches. Records the most recent target_object_id seen by
    `get_target_seg_id` so the test can assert which block was resolved.
    """
    def __init__(self):
        self.last_target_object_id: str | None = None
        self._seg_id = {
            "block_pink": 7,
            "block_blue": 8,
            "block_red":  9,
        }

    def reset(self, robot_obs, scene_obs):
        pass

    def render_cameras(self, width, height):
        seg_val = self._seg_id.get(self.last_target_object_id, 0)
        seg = np.full((height, width), seg_val, dtype=np.int32)
        rgb = np.full((height, width, 3), 128, dtype=np.uint8)
        depth = np.full((height, width), 1.0, dtype=np.float32)
        intr = np.eye(3, dtype=np.float32)
        R = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)
        return {
            "rgb_static":   rgb,
            "rgb_wrist":    rgb,
            "depth_static": depth,
            "depth_wrist":  depth,
            "seg_static":   seg,
            "seg_wrist":    seg,
            "static_intrinsic": intr,
            "wrist_intrinsic":  intr,
            "static_cam_R": R,
            "static_cam_t": t,
            "wrist_cam_R":  R,
            "wrist_cam_t":  t,
        }

    def get_object_pose(self, object_id: str):
        return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)

    def get_target_seg_id(self, object_id: str) -> int:
        self.last_target_object_id = object_id
        return self._seg_id[object_id]

    def close(self):
        pass


def _write_episode_npz(dir_path: Path, frame: int, scene_obs: np.ndarray) -> None:
    np.savez(
        dir_path / f"episode_{frame:07d}.npz",
        rel_actions=np.zeros(7, dtype=np.float32),
        robot_obs=np.zeros(15, dtype=np.float32),
        scene_obs=scene_obs.astype(np.float32),
    )


def _stub_geometry(monkeypatch) -> None:
    """Stub crop_target_from_seg and depth_to_world_points so the test
    does not depend on the real geometry pipeline. Behavior assertion
    lives at the env level via _FakeBlocksEnv.last_target_object_id."""
    from PIL import Image
    import tools.preprocess.calvin_preprocessor as pp_module

    def _fake_crop(rgb, seg, target_id):
        return Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8))

    def _fake_pts(**kwargs):
        return np.zeros((1024, 3), dtype=np.float32)

    monkeypatch.setattr(pp_module, "crop_target_from_seg", _fake_crop)
    monkeypatch.setattr(pp_module, "depth_to_world_points", _fake_pts)


def _make_worker(env, scene_letter: str, *, on_resolve_failure="abort"):
    """Build a CalvinWorker with a real SceneResolver pinned to one scene
    letter. SceneResolver falls back to default_scene when scene_info.npy
    is absent, so we point it at a non-existent dir."""
    from tools.preprocess.calvin_preprocessor import (
        CalvinWorker,
        SceneResolver,
    )
    resolver = SceneResolver(
        split_dir=Path("/dev/null"),
        default_scene=scene_letter,
    )
    return CalvinWorker(
        env=env,
        dataset_source="calvin_test",
        rank=0,
        scene_resolver=resolver,
        on_resolve_failure=on_resolve_failure,
        on_missing_target="abort",
    )


def _setup_output_dirs(output_dir: Path) -> None:
    for sub in (
        "images/obs/static", "images/obs/wrist",
        "images/target", "images/future",
        "depth/static", "depth/wrist", "point_clouds",
    ):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize("scene_letter", ["A", "B", "C", "D"])
def test_process_window_uses_inferred_block_for_unstack_per_scene(
    tmp_path, monkeypatch, scene_letter,
):
    """End-to-end: a 2-frame unstack window where BLUE is the moved block.
    Across every scene letter the worker must resolve target=block_blue
    via inference. Get the scene-specific slot order wrong (as the
    initial fix did) and non-A scenes silently pick the wrong block."""
    from tools.preprocess.calvin_preprocessor import LangWindow

    _stub_geometry(monkeypatch)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    start_so = _make_scene_obs(
        scene_letter,
        pink_xyz=(0.10, -0.05, 0.46),
        blue_xyz=(0.10, -0.05, 0.50),
        red_xyz =(0.05,  0.18, 0.36),
    )
    end_so = _make_scene_obs(
        scene_letter,
        pink_xyz=(0.10, -0.05, 0.46),
        blue_xyz=(-0.05, 0.10, 0.62),
        red_xyz =(0.05,  0.18, 0.36),
    )
    _write_episode_npz(input_dir, 100, start_so)
    _write_episode_npz(input_dir, 101, end_so)

    output_dir = tmp_path / "output"
    _setup_output_dirs(output_dir)

    env = _FakeBlocksEnv()
    worker = _make_worker(env, scene_letter)
    window = LangWindow(
        window_idx=0, ep_start=100, ep_end=101,
        instruction="remove the top block",
        task_label="unstack_block",
    )
    shard = worker.process_window(
        window=window, input_dir=input_dir, output_dir=output_dir,
    )
    assert shard is not None
    assert env.last_target_object_id == "block_blue", (
        f"scene {scene_letter}: expected block_blue (inferred), got "
        f"{env.last_target_object_id!r}"
    )


def test_process_window_uses_inferred_block_for_stack(tmp_path, monkeypatch):
    """Symmetric to unstack: stack_block window with pink as the placed
    block. Exercised under scene C so the non-A slot order is hit."""
    from tools.preprocess.calvin_preprocessor import LangWindow

    _stub_geometry(monkeypatch)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    start_so = _make_scene_obs(
        "C",
        pink_xyz=(-0.20, 0.10, 0.46),
        blue_xyz=(0.10, -0.05, 0.46),
        red_xyz =(0.30, -0.05, 0.46),
    )
    end_so = _make_scene_obs(
        "C",
        pink_xyz=(0.10, -0.05, 0.50),
        blue_xyz=(0.10, -0.05, 0.46),
        red_xyz =(0.30, -0.05, 0.46),
    )
    _write_episode_npz(input_dir, 200, start_so)
    _write_episode_npz(input_dir, 201, end_so)

    output_dir = tmp_path / "output"
    _setup_output_dirs(output_dir)

    env = _FakeBlocksEnv()
    worker = _make_worker(env, "C")
    window = LangWindow(
        window_idx=0, ep_start=200, ep_end=201,
        instruction="stack the blocks",
        task_label="stack_block",
    )
    worker.process_window(window=window, input_dir=input_dir, output_dir=output_dir)
    assert env.last_target_object_id == "block_pink"


def test_process_window_non_stack_task_still_uses_static_map(
    tmp_path, monkeypatch,
):
    """Regression: the new branch must not regress fixture-task resolution.
    `open_drawer` still routes through `resolve_target_object` to
    `table__drawer_link` — no scene_obs read, no inference."""
    from tools.preprocess.calvin_preprocessor import LangWindow

    _stub_geometry(monkeypatch)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_episode_npz(input_dir, 300, _make_scene_obs("A"))

    output_dir = tmp_path / "output"
    _setup_output_dirs(output_dir)

    env = _FakeBlocksEnv()
    env._seg_id["table__drawer_link"] = 42
    worker = _make_worker(env, "A")
    window = LangWindow(
        window_idx=0, ep_start=300, ep_end=300,
        instruction="open the drawer",
        task_label="open_drawer",
    )
    worker.process_window(window=window, input_dir=input_dir, output_dir=output_dir)
    assert env.last_target_object_id == "table__drawer_link"


def test_process_window_unknown_task_skips_when_on_resolve_failure_skip(
    tmp_path, monkeypatch,
):
    """Existing skip behavior preserved for unknown labels (the only
    failure mode the skip path is meant for)."""
    from tools.preprocess.calvin_preprocessor import LangWindow

    _stub_geometry(monkeypatch)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_episode_npz(input_dir, 400, _make_scene_obs("A"))

    output_dir = tmp_path / "output"
    _setup_output_dirs(output_dir)

    env = _FakeBlocksEnv()
    worker = _make_worker(env, "A", on_resolve_failure="skip")
    window = LangWindow(
        window_idx=0, ep_start=400, ep_end=400,
        instruction="frobnicate the widget",
        task_label="frobnicate_widget",  # not in CALVIN_TASK_TO_OBJECT
    )
    shard = worker.process_window(
        window=window, input_dir=input_dir, output_dir=output_dir,
    )
    assert shard is None, "unknown task with skip mode should drop the window"
    assert env.last_target_object_id is None


def test_process_window_stack_with_missing_scene_obs_aborts(
    tmp_path, monkeypatch,
):
    """If a stack window's start npz is missing, the failure must surface
    (the skip path is for unknown-task vocabulary gaps, not dataset
    corruption). Even with on_resolve_failure='skip', missing scene_obs
    aborts."""
    from tools.preprocess.calvin_preprocessor import LangWindow

    _stub_geometry(monkeypatch)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # ep_start=500's npz is intentionally NOT written.
    _write_episode_npz(input_dir, 501, _make_scene_obs("A"))

    output_dir = tmp_path / "output"
    _setup_output_dirs(output_dir)

    env = _FakeBlocksEnv()
    worker = _make_worker(env, "A", on_resolve_failure="skip")
    window = LangWindow(
        window_idx=0, ep_start=500, ep_end=501,
        instruction="remove the top block",
        task_label="unstack_block",
    )
    with pytest.raises(FileNotFoundError):
        worker.process_window(
            window=window, input_dir=input_dir, output_dir=output_dir,
        )


def test_process_window_stack_with_missing_scene_obs_key_aborts(
    tmp_path, monkeypatch,
):
    """If the npz exists but lacks a `scene_obs` array, the load helper
    raises KeyError. Skip mode does NOT swallow it — only unknown task
    labels qualify for skipping."""
    from tools.preprocess.calvin_preprocessor import LangWindow

    _stub_geometry(monkeypatch)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # Start npz: present but missing scene_obs.
    np.savez(
        input_dir / "episode_0000600.npz",
        rel_actions=np.zeros(7, dtype=np.float32),
        robot_obs=np.zeros(15, dtype=np.float32),
    )
    _write_episode_npz(input_dir, 601, _make_scene_obs("A"))

    output_dir = tmp_path / "output"
    _setup_output_dirs(output_dir)

    env = _FakeBlocksEnv()
    worker = _make_worker(env, "A", on_resolve_failure="skip")
    window = LangWindow(
        window_idx=0, ep_start=600, ep_end=601,
        instruction="remove the top block",
        task_label="unstack_block",
    )
    with pytest.raises(KeyError, match="scene_obs"):
        worker.process_window(
            window=window, input_dir=input_dir, output_dir=output_dir,
        )


def test_process_window_stack_with_wrong_shape_scene_obs_aborts(
    tmp_path, monkeypatch,
):
    """A 27-dim quaternion-mode scene_obs slipping through must abort
    via the shape check inside infer_stack_block."""
    from tools.preprocess.calvin_preprocessor import LangWindow

    _stub_geometry(monkeypatch)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    bad_scene_obs = np.zeros(27, dtype=np.float32)
    _write_episode_npz(input_dir, 700, bad_scene_obs)
    _write_episode_npz(input_dir, 701, _make_scene_obs("A"))

    output_dir = tmp_path / "output"
    _setup_output_dirs(output_dir)

    env = _FakeBlocksEnv()
    worker = _make_worker(env, "A", on_resolve_failure="skip")
    window = LangWindow(
        window_idx=0, ep_start=700, ep_end=701,
        instruction="remove the top block",
        task_label="unstack_block",
    )
    with pytest.raises(ValueError, match="shape"):
        worker.process_window(
            window=window, input_dir=input_dir, output_dir=output_dir,
        )


def test_process_window_stack_with_nonfinite_scene_obs_aborts(
    tmp_path, monkeypatch,
):
    """NaN/Inf in scene_obs at the window edges must abort cleanly."""
    from tools.preprocess.calvin_preprocessor import LangWindow

    _stub_geometry(monkeypatch)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    start_so = _make_scene_obs("A")
    end_so = _make_scene_obs("A")
    end_so[18] = np.nan  # red block x in scene A (block_red is slot 3)
    _write_episode_npz(input_dir, 800, start_so)
    _write_episode_npz(input_dir, 801, end_so)

    output_dir = tmp_path / "output"
    _setup_output_dirs(output_dir)

    env = _FakeBlocksEnv()
    worker = _make_worker(env, "A", on_resolve_failure="skip")
    window = LangWindow(
        window_idx=0, ep_start=800, ep_end=801,
        instruction="remove the top block",
        task_label="unstack_block",
    )
    with pytest.raises(ValueError, match="finite"):
        worker.process_window(
            window=window, input_dir=input_dir, output_dir=output_dir,
        )


def test_scene_resolver_parses_calvin_scene_prefix_keys(tmp_path):
    """Real CALVIN ABCD_D dumps ship scene_info.npy keyed as
    `calvin_scene_A`/etc. The SceneResolver must normalise those to
    plain letters A/B/C/D so infer_stack_block accepts them — the
    previous code naively stripped `scene_` and emitted `CALVIN_A`.
    """
    from tools.preprocess.calvin_preprocessor import SceneResolver, LangWindow

    info = {
        "calvin_scene_A": np.array([0, 99], dtype=np.int64),
        "calvin_scene_C": np.array([100, 199], dtype=np.int64),
    }
    np.save(tmp_path / "scene_info.npy",
            np.array(info, dtype=object), allow_pickle=True)

    resolver = SceneResolver(split_dir=tmp_path, default_scene=None)
    win_a = LangWindow(
        window_idx=0, ep_start=10, ep_end=50,
        instruction="x", task_label="unstack_block",
    )
    win_c = LangWindow(
        window_idx=1, ep_start=110, ep_end=150,
        instruction="y", task_label="unstack_block",
    )
    assert resolver.resolve_for_window(win_a) == "A"
    assert resolver.resolve_for_window(win_c) == "C"
