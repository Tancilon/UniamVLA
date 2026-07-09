"""Tests for runners/preprocess_depth_vda.py (pure logic; GPU inference excluded)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from runners.preprocess_depth_vda import (
    depth_output_path,
    list_camera_videos,
    save_depth_npz,
)


def _make_video_tree(root: Path) -> None:
    """Minimal LeRobot-like videos/ tree with empty mp4 placeholder files."""
    for chunk, eps in [("chunk-000", [0, 1]), ("chunk-001", [1000])]:
        for cam in ["image", "wrist_image"]:
            d = root / "videos" / chunk / cam
            d.mkdir(parents=True)
            for ep in eps:
                (d / f"episode_{ep:06d}.mp4").touch()


def test_list_camera_videos_sorted_and_filtered(tmp_path):
    _make_video_tree(tmp_path)
    videos = list_camera_videos(tmp_path, "image")
    assert [v.name for v in videos] == [
        "episode_000000.mp4",
        "episode_000001.mp4",
        "episode_001000.mp4",
    ]
    assert all("wrist_image" not in str(v) for v in videos)


def test_depth_output_path_mirrors_videos_layout(tmp_path):
    video = tmp_path / "videos" / "chunk-002" / "image" / "episode_002345.mp4"
    out = depth_output_path(video, tmp_path)
    assert out == tmp_path / "depth" / "chunk-002" / "image" / "episode_002345.npz"


def test_save_depth_npz_truncates_and_casts_float16(tmp_path):
    out = tmp_path / "depth" / "chunk-000" / "image" / "episode_000000.npz"
    depths = np.random.default_rng(0).random((70, 200, 200)).astype(np.float32)
    shape = save_depth_npz(out, depths, num_frames=65)
    assert shape == (65, 200, 200)
    loaded = np.load(out)["depths"]
    assert loaded.shape == (65, 200, 200)
    assert loaded.dtype == np.float16
    np.testing.assert_allclose(
        loaded.astype(np.float32), depths[:65], atol=1e-3
    )
