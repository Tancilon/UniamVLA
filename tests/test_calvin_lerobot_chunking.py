"""Chunking regression tests for the CALVIN LeRobot preprocessor."""
from __future__ import annotations

import importlib
import json
import sys
import types

import numpy as np
import pyarrow.parquet as pq


def _stub_imageio_if_needed(monkeypatch):
    if "imageio.v3" in sys.modules:
        return
    imageio = types.ModuleType("imageio")
    imageio.__path__ = []
    imageio_v3 = types.ModuleType("imageio.v3")
    imageio.v3 = imageio_v3
    monkeypatch.setitem(sys.modules, "imageio", imageio)
    monkeypatch.setitem(sys.modules, "imageio.v3", imageio_v3)


def test_episode_files_are_chunked_every_1000_episodes(tmp_path, monkeypatch):
    _stub_imageio_if_needed(monkeypatch)

    module = importlib.import_module(
        "tools.preprocess.calvin_preprocessor_lerobot",
    )
    CalvinPreprocessorLeRobot = module.CalvinPreprocessorLeRobot

    preprocessor = CalvinPreprocessorLeRobot(default_scene="D")
    sample = {
        "episode_index": 0,
        "frame_index": 0,
        "timestamp": 0.0,
        "index": 0,
    }

    preprocessor._emit_episode_parquet([sample], tmp_path, episode_index=999)
    preprocessor._emit_episode_parquet([sample], tmp_path, episode_index=1000)

    assert (
        tmp_path / "data" / "chunk-000" / "episode_000999.parquet"
    ).exists()
    assert (
        tmp_path / "data" / "chunk-001" / "episode_001000.parquet"
    ).exists()
    assert pq.read_table(
        tmp_path / "data" / "chunk-001" / "episode_001000.parquet",
    ).num_rows == 1

    video_paths = []

    def fake_imwrite(path, frames, fps, codec):
        video_paths.append(path)

    monkeypatch.setattr(module.iio, "imwrite", fake_imwrite, raising=False)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    preprocessor._emit_episode_videos(
        [frame], [frame], tmp_path, episode_index=1000,
    )

    assert (
        tmp_path
        / "videos"
        / "chunk-001"
        / "video.primary_image"
        / "episode_001000.mp4"
    ) in video_paths
    assert (
        tmp_path
        / "videos"
        / "chunk-001"
        / "video.wrist_image"
        / "episode_001000.mp4"
    ) in video_paths


def test_episode_sidecars_write_image_target_per_frame(tmp_path, monkeypatch):
    _stub_imageio_if_needed(monkeypatch)

    module = importlib.import_module(
        "tools.preprocess.calvin_preprocessor_lerobot",
    )
    CalvinPreprocessorLeRobot = module.CalvinPreprocessorLeRobot

    preprocessor = CalvinPreprocessorLeRobot(default_scene="D")
    image_targets = [
        np.full((2, 2, 3), 11, dtype=np.uint8),
        np.full((2, 2, 3), 22, dtype=np.uint8),
    ]
    point_clouds = [
        np.zeros((module.NUM_POINTS, 3), dtype=np.float32),
        np.ones((module.NUM_POINTS, 3), dtype=np.float32),
    ]
    depth_targets = [
        np.full((4, 4), 0.5, dtype=np.float32),
        np.full((4, 4), 1.5, dtype=np.float32),
    ]
    grounding_masks = [
        np.zeros((1, 20, 20), dtype=np.float32),
        np.ones((1, 20, 20), dtype=np.float32),
    ]
    affordance_heatmaps = [
        np.full((1, 20, 20), 0.25, dtype=np.float32),
        np.full((1, 20, 20), 0.75, dtype=np.float32),
    ]
    grounding_levels = ["part", "object"]

    preprocessor._emit_episode_sidecars(
        image_targets,
        point_clouds,
        tmp_path,
        episode_index=7,
        depth_targets=depth_targets,
        grounding_masks=grounding_masks,
        grounding_levels=grounding_levels,
        affordance_heatmaps=affordance_heatmaps,
    )

    assert (tmp_path / "image_targets" / "7" / "0.png").exists()
    assert (tmp_path / "image_targets" / "7" / "1.png").exists()
    assert (tmp_path / "point_clouds" / "7" / "0.npy").exists()
    assert (tmp_path / "point_clouds" / "7" / "1.npy").exists()
    assert (tmp_path / "depths" / "static" / "7" / "0.npy").exists()
    assert (tmp_path / "depths" / "static" / "7" / "1.npy").exists()
    assert (tmp_path / "grounding_masks" / "static" / "7" / "0.npy").exists()
    assert (tmp_path / "grounding_masks" / "static" / "7" / "1.npy").exists()
    assert (tmp_path / "grounding_masks" / "static" / "7" / "0.json").exists()
    assert (tmp_path / "grounding_masks" / "static" / "7" / "1.json").exists()
    assert (tmp_path / "affordance_heatmaps" / "static" / "7" / "0.npy").exists()
    assert (tmp_path / "affordance_heatmaps" / "static" / "7" / "1.npy").exists()

    with open(tmp_path / "grounding_masks" / "static" / "7" / "0.json") as f:
        assert json.load(f)["grounding_level"] == "part"
    with open(tmp_path / "grounding_masks" / "static" / "7" / "1.json") as f:
        assert json.load(f)["grounding_level"] == "object"


def test_preprocessor_writes_uamvla_aux_coverage(tmp_path, monkeypatch):
    _stub_imageio_if_needed(monkeypatch)

    module = importlib.import_module(
        "tools.preprocess.calvin_preprocessor_lerobot",
    )
    CalvinPreprocessorLeRobot = module.CalvinPreprocessorLeRobot

    preprocessor = CalvinPreprocessorLeRobot(default_scene="D")
    coverage = preprocessor._empty_aux_coverage()
    preprocessor._merge_aux_coverage(
        coverage,
        task_name="open drawer",
        lengths={
            "image_target": 2,
            "point_cloud": 2,
            "depth": 2,
            "grounding": 2,
            "affordance": 2,
        },
    )
    preprocessor._emit_aux_coverage(tmp_path, coverage)

    written = json.loads(
        (tmp_path / "meta" / "uamvla_aux_coverage.json").read_text()
    )
    assert written["tasks"]["open drawer"]["depth"] == {"valid": 2, "total": 2}
    assert written["tasks"]["open drawer"]["grounding"] == {
        "valid": 2,
        "total": 2,
    }
    assert written["totals"]["affordance"] == {"valid": 2, "total": 2}


def test_preprocessor_meta_uses_lerobot_chunk_size_1000(tmp_path, monkeypatch):
    _stub_imageio_if_needed(monkeypatch)

    module = importlib.import_module(
        "tools.preprocess.calvin_preprocessor_lerobot",
    )
    CalvinPreprocessorLeRobot = module.CalvinPreprocessorLeRobot

    preprocessor = CalvinPreprocessorLeRobot(default_scene="D")
    n_episodes = 19912
    preprocessor._emit_meta(
        output_dir=tmp_path,
        n_episodes=n_episodes,
        n_total_frames=1192122,
        tasks_seen=["move the door all the way to the right"],
        episode_index_to_task={0: 0, 999: 0, 1000: 0, n_episodes - 1: 0},
        episode_lengths={0: 65, 999: 64, 1000: 41, n_episodes - 1: 41},
    )

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["chunks_size"] == 1000
    assert 999 // info["chunks_size"] == 0
    assert 1000 // info["chunks_size"] == 1
