"""Smoke test on the 3-episode CALVIN_ABCD smoke dataset (PR 2).

Asserts the layout that :mod:`tools.preprocess.calvin_preprocessor_lerobot`
produces matches spec §4.3 (LeRobot v2 parquet + sidecar):

    <dataset>/
    ├── data/chunk-XXX/episode_NNNNNN.parquet
    ├── videos/chunk-XXX/video.{primary,wrist}_image/episode_NNNNNN.mp4
    ├── image_targets/<traj>/<base>.png
    ├── point_clouds/<traj>/<base>.npy
    ├── camera_params.json
    └── meta/{modality.json, episodes.jsonl, tasks.jsonl, info.json}

Skipped automatically when the smoke dataset hasn't been produced — run
``runners/preprocess_calvin.py --max_episodes 3 --output_dir
playground/Datasets/UAMVLA_LEROBOT_CALVIN_ABCD_SMOKE`` to populate it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

CALVIN_PATH = Path(
    "playground/Datasets/UAMVLA_LEROBOT_CALVIN_ABCD_SMOKE",
).resolve()


pytestmark = pytest.mark.skipif(
    not CALVIN_PATH.exists(),
    reason=(
        f"Smoke dataset not found at {CALVIN_PATH}. "
        "Run `python runners/preprocess_calvin.py --max_episodes 3 "
        f"--output_dir {CALVIN_PATH}` to produce it."
    ),
)


def test_layout_top_level_dirs():
    """All seven top-level paths from spec §4.3 must exist."""
    expected = [
        "data/chunk-000",
        "videos/chunk-000/video.primary_image",
        "videos/chunk-000/video.wrist_image",
        "image_targets",
        "point_clouds",
        "camera_params.json",
        "meta",
    ]
    missing = [
        p for p in expected if not (CALVIN_PATH / p).exists()
    ]
    assert not missing, f"Missing entries: {missing}"


def test_parquet_columns_and_state_dim():
    """Every per-episode parquet must have the LeRobot v2 columns from
    modality.json — split state (15+6+3+6+3 = 33) and 7-component action.
    """
    import pyarrow.parquet as pq

    parquet_files = sorted(
        (CALVIN_PATH / "data" / "chunk-000").glob("episode_*.parquet"),
    )
    assert parquet_files, "No per-episode parquet files emitted"
    assert len(parquet_files) >= 1

    expected_state_cols = {
        "state.robot_obs", "state.target_pose_rot6d",
        "state.target_pose_trans", "state.static_cam_rot6d",
        "state.static_cam_trans",
    }
    expected_action_cols = {
        "action.x", "action.y", "action.z",
        "action.roll", "action.pitch", "action.yaw", "action.gripper",
    }
    expected_provenance = {
        "episode_index", "frame_index", "timestamp", "index", "task_index",
        "annotation.human.action.task_description",
        "trajectory_id", "base_index",
    }

    for pqf in parquet_files:
        tbl = pq.read_table(pqf)
        cols = set(tbl.column_names)
        assert expected_state_cols <= cols, (
            f"{pqf.name} missing state cols: "
            f"{expected_state_cols - cols}"
        )
        assert expected_action_cols <= cols, (
            f"{pqf.name} missing action cols: "
            f"{expected_action_cols - cols}"
        )
        assert expected_provenance <= cols, (
            f"{pqf.name} missing provenance cols: "
            f"{expected_provenance - cols}"
        )

        # First row state shapes — make sure modality.json slice ranges
        # actually match the on-disk vectors.
        row0 = tbl.slice(0, 1).to_pylist()[0]
        assert len(row0["state.robot_obs"]) == 15
        assert len(row0["state.target_pose_rot6d"]) == 6
        assert len(row0["state.target_pose_trans"]) == 3
        assert len(row0["state.static_cam_rot6d"]) == 6
        assert len(row0["state.static_cam_trans"]) == 3


def test_sidecars_and_camera_params_loadable():
    """Image targets, point clouds, and camera_params.json must be
    readable and have the shapes the framework expects.
    """
    # camera_params.json
    cam_path = CALVIN_PATH / "camera_params.json"
    cam = json.loads(cam_path.read_text())
    for key in ("fx", "fy", "cx", "cy", "width", "height", "camera_name"):
        assert key in cam, f"camera_params.json missing {key}"
    assert cam["width"] == 256 and cam["height"] == 256

    episodes = [
        json.loads(line)
        for line in (CALVIN_PATH / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    has_frame_targets = bool(list((CALVIN_PATH / "image_targets").glob("*/*.png")))
    has_legacy_targets = bool(list((CALVIN_PATH / "image_targets").glob("*.png")))
    if not has_frame_targets and has_legacy_targets:
        pytest.skip(
            "Smoke dataset uses legacy episode-level image_targets; "
            "rerun CALVIN preprocessing to produce per-frame image_targets.",
        )

    # image_targets: one PNG per kept episode frame.
    for episode in episodes:
        traj = int(episode["episode_index"])
        expected = int(episode["length"])
        img_targets = sorted((CALVIN_PATH / "image_targets" / str(traj)).glob("*.png"))
        assert [path.name for path in img_targets] == [
            f"{base}.png" for base in range(expected)
        ]

    # point_clouds: each episode dir must have at least one (1024, 3)
    # float32 array.
    pc_root = CALVIN_PATH / "point_clouds"
    pc_episodes = sorted(p for p in pc_root.iterdir() if p.is_dir())
    assert pc_episodes, "No point_cloud episode dirs emitted"
    for ep_dir in pc_episodes:
        pc_files = sorted(ep_dir.glob("*.npy"))
        assert pc_files, f"{ep_dir} has no .npy files"
        sample = np.load(pc_files[0])
        assert sample.shape == (1024, 3), (
            f"{pc_files[0]} shape {sample.shape} != (1024, 3)"
        )
        assert sample.dtype == np.float32, (
            f"{pc_files[0]} dtype {sample.dtype} != float32"
        )


def test_meta_files_well_formed():
    """meta/ must contain modality.json, tasks.jsonl, episodes.jsonl,
    info.json — all parseable and consistent.
    """
    meta = CALVIN_PATH / "meta"

    # modality.json — same schema we registered in
    # examples/calvin/train_files/data_registry/modality.json.
    # NOTE: state/action subkeys store per-column slices (start=0..end=dim)
    # with original_key pointing at the per-column parquet name, so the
    # LeRobot reader can locate the dataset statistics. Video subkeys point
    # at the dotted info.json features key.
    mj = json.loads((meta / "modality.json").read_text())
    assert "state" in mj and "action" in mj and "video" in mj
    assert mj["state"]["robot_obs"] == {
        "start": 0, "end": 15, "original_key": "state.robot_obs",
    }
    assert mj["state"]["static_cam_trans"] == {
        "start": 0, "end": 3, "original_key": "state.static_cam_trans",
    }
    assert mj["action"]["gripper"] == {
        "start": 0, "end": 1, "original_key": "action.gripper",
    }
    assert mj["video"]["primary_image"] == {
        "original_key": "video.primary_image",
    }
    assert mj["video"]["wrist_image"] == {
        "original_key": "video.wrist_image",
    }

    # tasks.jsonl — one record per task seen.
    tasks_lines = [
        json.loads(line)
        for line in (meta / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert tasks_lines, "tasks.jsonl is empty"
    for t in tasks_lines:
        assert "task_index" in t and "task" in t

    # episodes.jsonl — one record per kept episode.
    ep_lines = [
        json.loads(line)
        for line in (meta / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert ep_lines, "episodes.jsonl is empty"
    for e in ep_lines:
        assert {"episode_index", "tasks", "length"} <= set(e.keys())
        assert isinstance(e["tasks"], list) and e["tasks"]
        assert e["length"] > 0

    # info.json — top-level keys match LeRobot v2 contract.
    info = json.loads((meta / "info.json").read_text())
    for key in (
        "codebase_version", "robot_type", "total_episodes",
        "total_frames", "total_tasks", "fps", "splits", "features",
    ):
        assert key in info, f"info.json missing {key}"
    assert info["total_episodes"] == len(ep_lines)
    assert info["total_tasks"] == len(tasks_lines)
    assert info["fps"] == 15
    assert "video.primary_image" in info["features"]
    assert "video.wrist_image" in info["features"]


def test_preprocessor_meta_uses_1000_episode_chunks(tmp_path, monkeypatch):
    """Metadata must match the preprocessor's 1000-episode chunk layout."""
    if importlib.util.find_spec("imageio") is None:
        imageio = types.ModuleType("imageio")
        imageio.__path__ = []
        imageio_v3 = types.ModuleType("imageio.v3")
        imageio.v3 = imageio_v3
        monkeypatch.setitem(sys.modules, "imageio", imageio)
        monkeypatch.setitem(sys.modules, "imageio.v3", imageio_v3)

    module_name = "tools.preprocess.calvin_preprocessor_lerobot"
    previous_module = sys.modules.pop(module_name, None)
    try:
        from tools.preprocess.calvin_preprocessor_lerobot import CalvinPreprocessorLeRobot

        preprocessor = CalvinPreprocessorLeRobot(default_scene="D")
        n_episodes = 19912
        preprocessor._emit_meta(
            output_dir=tmp_path,
            n_episodes=n_episodes,
            n_total_frames=1192122,
            tasks_seen=["move the door all the way to the right"],
            episode_index_to_task={0: 0, n_episodes - 1: 0},
            episode_lengths={0: 65, n_episodes - 1: 41},
        )

        info = json.loads((tmp_path / "meta" / "info.json").read_text())
        assert info["chunks_size"] == 1000
        assert 999 // info["chunks_size"] == 0
        assert 1000 // info["chunks_size"] == 1
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module
