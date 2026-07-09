"""Tests for runners/preprocess_depth_vda.py (pure logic; GPU inference excluded)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from runners.preprocess_depth_vda import (
    PROJECT_ROOT,
    decode_video_frames,
    depth_output_path,
    list_camera_videos,
    save_depth_npz,
)

_REAL_AV1_VIDEO = (
    PROJECT_ROOT
    / "datasets/task_ABC_D_scene_D_lerobot/videos/chunk-000/image/episode_000000.mp4"
)


def _write_synthetic_video(path: Path, n_frames: int = 7, size: int = 64) -> None:
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            img = np.full((size, size, 3), i * 30, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


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


def test_decode_video_frames_returns_all_frames(tmp_path):
    video = tmp_path / "ep.mp4"
    _write_synthetic_video(video, n_frames=7, size=64)
    frames = decode_video_frames(video)
    assert frames.shape == (7, 64, 64, 3)
    assert frames.dtype == np.uint8


@pytest.mark.skipif(not _REAL_AV1_VIDEO.exists(), reason="dataset not present")
def test_decode_real_av1_dataset_video():
    """Validates the spec's AV1-decode risk on this machine (no GPU needed)."""
    frames = decode_video_frames(_REAL_AV1_VIDEO)
    assert frames.shape == (65, 200, 200, 3)
    assert frames.dtype == np.uint8
