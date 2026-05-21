from __future__ import annotations

import json
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


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


def _stub_imageio(monkeypatch):
    imageio = types.ModuleType("imageio")
    imageio.__path__ = []
    imageio_v3 = types.ModuleType("imageio.v3")

    def fake_imwrite(path, frames, fps, codec):
        Path(path).write_bytes(b"fake mp4")

    imageio_v3.imwrite = fake_imwrite
    imageio.v3 = imageio_v3
    monkeypatch.setitem(sys.modules, "imageio", imageio)
    monkeypatch.setitem(sys.modules, "imageio.v3", imageio_v3)


def test_writer_emits_lerobot_episode_and_sidecars(tmp_path: Path, monkeypatch):
    _stub_imageio(monkeypatch)
    module = importlib.import_module("tools.preprocess.libero_lerobot_writer")
    LiberoEpisodeBuffers = module.LiberoEpisodeBuffers
    LiberoLerobotWriter = module.LiberoLerobotWriter

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
        depth_targets=[
            np.ones((32, 32), dtype=np.float32),
            np.ones((32, 32), dtype=np.float32),
        ],
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
    assert (
        tmp_path / "videos/chunk-000/video.primary_image/episode_000000.mp4"
    ).exists()
    assert (
        tmp_path / "videos/chunk-000/video.wrist_image/episode_000000.mp4"
    ).exists()
    assert (tmp_path / "image_targets/0/0.png").exists()
    assert not (tmp_path / "image_targets/0/1.png").exists()
    assert np.load(tmp_path / "point_clouds/0/0.npy").shape == (1024, 3)
    assert np.load(tmp_path / "depths/static/0/0.npy").shape == (32, 32)
    assert np.load(tmp_path / "grounding_masks/static/0/0.npy").shape == (
        1,
        20,
        20,
    )
    with open(tmp_path / "grounding_masks/static/0/0.json") as f:
        assert json.load(f)["grounding_level"] == "object"
    table = pq.read_table(tmp_path / "data/chunk-000/episode_000000.parquet")
    assert table.num_rows == 2
    with open(tmp_path / "meta/info.json") as f:
        info = json.load(f)
    assert info["total_episodes"] == 1
    assert (
        info["video_path"]
        == "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
