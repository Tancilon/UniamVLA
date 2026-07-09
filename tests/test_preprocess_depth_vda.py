"""Tests for runners/preprocess_depth_vda.py (pure logic; GPU inference excluded)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from runners.preprocess_depth_vda import (
    PROJECT_ROOT,
    decode_video_frames,
    depth_output_path,
    list_camera_videos,
    load_episode_meta,
    process_videos,
    save_depth_npz,
    verify_dataset,
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


def _make_meta(root: Path, lengths: dict[int, int], chunks_size: int = 1000) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"chunks_size": chunks_size, "fps": 10})
    )
    with open(meta / "episodes.jsonl", "w") as f:
        for ep, length in lengths.items():
            f.write(json.dumps({"episode_index": ep, "tasks": [], "length": length}) + "\n")


def _write_depth_npz(root: Path, ep: int, n_frames: int, chunks_size: int = 1000) -> None:
    out = (
        root / "depth" / f"chunk-{ep // chunks_size:03d}" / "image"
        / f"episode_{ep:06d}.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, depths=np.zeros((n_frames, 4, 4), dtype=np.float16))


def test_load_episode_meta(tmp_path):
    _make_meta(tmp_path, {0: 65, 1: 34, 1000: 60})
    lengths, chunks_size, fps = load_episode_meta(tmp_path)
    assert lengths == {0: 65, 1: 34, 1000: 60}
    assert chunks_size == 1000
    assert fps == 10


def test_verify_dataset_reports_missing_and_mismatch(tmp_path):
    _make_meta(tmp_path, {0: 65, 1: 34, 1000: 60})
    _write_depth_npz(tmp_path, 0, 65)      # ok
    _write_depth_npz(tmp_path, 1, 30)      # mismatch (expected 34)
    # episode 1000: missing
    problems = verify_dataset(tmp_path, "image")
    assert len(problems) == 2
    assert any("episode_000001.npz" in p and "mismatch" in p for p in problems)
    assert any("episode_001000.npz" in p and "missing" in p for p in problems)


def test_verify_dataset_all_ok_returns_empty(tmp_path):
    _make_meta(tmp_path, {0: 65, 1: 34})
    _write_depth_npz(tmp_path, 0, 65)
    _write_depth_npz(tmp_path, 1, 34)
    assert verify_dataset(tmp_path, "image") == []


def _make_decodable_dataset(root: Path, n_eps: int = 3, n_frames: int = 7):
    """videos/ tree whose mp4s are real decodable synthetic videos."""
    d = root / "videos" / "chunk-000" / "image"
    d.mkdir(parents=True)
    videos = []
    for ep in range(n_eps):
        p = d / f"episode_{ep:06d}.mp4"
        _write_synthetic_video(p, n_frames=n_frames, size=64)
        videos.append(p)
    return videos


def _fake_infer(frames, fps, input_size):
    # VDA pads short clips internally; emulate returning MORE frames than input.
    t, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    return np.linspace(0, 1, (t + 5) * h * w, dtype=np.float32).reshape(t + 5, h, w)


def _broken_infer(frames, fps, input_size):
    return np.zeros((frames.shape[0] - 2, frames.shape[1], frames.shape[2]), np.float32)


def test_process_videos_writes_truncated_npz(tmp_path):
    videos = _make_decodable_dataset(tmp_path, n_eps=2, n_frames=7)
    n_ok, n_skip, failures = process_videos(
        videos, _fake_infer, tmp_path, fps=10
    )
    assert (n_ok, n_skip, failures) == (2, 0, [])
    for ep in range(2):
        loaded = np.load(
            tmp_path / "depth" / "chunk-000" / "image" / f"episode_{ep:06d}.npz"
        )["depths"]
        assert loaded.shape == (7, 64, 64)  # truncated from fake's 12
        assert loaded.dtype == np.float16


def test_process_videos_skips_existing_unless_overwrite(tmp_path):
    videos = _make_decodable_dataset(tmp_path, n_eps=2, n_frames=7)
    process_videos(videos, _fake_infer, tmp_path, fps=10)
    n_ok, n_skip, _ = process_videos(videos, _fake_infer, tmp_path, fps=10)
    assert (n_ok, n_skip) == (0, 2)
    n_ok, n_skip, _ = process_videos(
        videos, _fake_infer, tmp_path, fps=10, overwrite=True
    )
    assert (n_ok, n_skip) == (2, 0)


def test_process_videos_limit(tmp_path):
    videos = _make_decodable_dataset(tmp_path, n_eps=3, n_frames=7)
    n_ok, n_skip, _ = process_videos(videos, _fake_infer, tmp_path, fps=10, limit=1)
    assert (n_ok, n_skip) == (1, 0)


def test_process_videos_records_failure_and_continues(tmp_path):
    videos = _make_decodable_dataset(tmp_path, n_eps=2, n_frames=7)
    n_ok, n_skip, failures = process_videos(
        videos, _broken_infer, tmp_path, fps=10
    )
    assert n_ok == 0
    assert len(failures) == 2
    failures_txt = (tmp_path / "depth" / "failures.txt").read_text()
    assert "episode_000000.mp4" in failures_txt
    assert "episode_000001.mp4" in failures_txt
    # no partial npz left behind
    assert not list((tmp_path / "depth").rglob("*.npz"))
