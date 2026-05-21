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
    assert (
        output_dir / "videos/chunk-000/video.primary_image/episode_000000.mp4"
    ).exists()
    assert (output_dir / "meta/uamvla_aux_coverage.json").exists()
