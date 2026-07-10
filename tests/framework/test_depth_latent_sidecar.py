"""depth_latent / depth_px sidecar loaders (UamVLAOFT), against real artifacts.

These exercise the loader methods in isolation -- no Qwen weights, no GPU --
by binding them to a stand-in object that carries only what they touch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from starVLA.model.framework.VLM4A.UamVLAOFT import UamVLAOFT

DATASET = Path("datasets/task_ABC_D_scene_D_lerobot")
GRID, LATENT_C, PX = 7, 64, 112


class _Loader:
    """Minimal stand-in exposing only what the depth loaders read."""

    _DEPTH_LATENT_CACHE_MAXSIZE = UamVLAOFT._DEPTH_LATENT_CACHE_MAXSIZE
    _lerobot_chunks_size = 1000

    _depth_sidecar_path = UamVLAOFT._depth_sidecar_path
    _load_depth_episode_memmap = UamVLAOFT._load_depth_episode_memmap
    _load_depth_latent_frame = UamVLAOFT._load_depth_latent_frame

    def __init__(self, root):
        self.sidecar_root = root


@pytest.fixture
def loader():
    if not (DATASET / "depth_latent" / "meta.json").exists():
        pytest.skip("run runners/preprocess_depth_latent.py first")
    return _Loader(DATASET)


def test_path_mirrors_the_videos_chunk_layout(loader):
    path = loader._depth_sidecar_path("depth_latent", 1234, DATASET, "image")
    assert path == DATASET / "depth_latent" / "chunk-001" / "image" / "episode_001234.npy"


def test_loads_a_single_frame_with_the_expected_shapes(loader):
    latent = loader._load_depth_latent_frame("depth_latent", 1, 0, camera="image")
    px = loader._load_depth_latent_frame("depth_px", 1, 0, camera="image")
    assert latent.shape == (LATENT_C, GRID, GRID) and latent.dtype == torch.float32
    assert px.shape == (PX, PX) and px.dtype == torch.float32
    assert 0.0 <= float(px.min()) and float(px.max()) <= 1.0


def test_missing_episode_and_out_of_range_frame_return_none(loader):
    assert loader._load_depth_latent_frame("depth_latent", 999999, 0, camera="image") is None
    assert loader._load_depth_latent_frame("depth_latent", 1, 10**6, camera="image") is None


def test_memmap_handle_is_cached_and_bounded(loader):
    path = loader._depth_sidecar_path("depth_latent", 1, DATASET, "image")
    first = loader._load_depth_episode_memmap(path)
    assert loader._load_depth_episode_memmap(path) is first, "handle not reused"

    for traj in range(2, 2 + loader._DEPTH_LATENT_CACHE_MAXSIZE + 5):
        loader._load_depth_episode_memmap(
            loader._depth_sidecar_path("depth_latent", traj, DATASET, "image")
        )
    assert len(loader._depth_memmap_cache) <= loader._DEPTH_LATENT_CACHE_MAXSIZE


def test_frames_are_not_all_identical(loader):
    """Guards against an off-by-one that returns frame 0 for every base_index."""
    a = loader._load_depth_latent_frame("depth_latent", 1, 0, camera="image")
    b = loader._load_depth_latent_frame("depth_latent", 1, 30, camera="image")
    assert not torch.allclose(a, b)


def test_collate_produces_the_mask_key_the_head_resolves_by_name():
    """Head is named 'depth_latent', so _resolve_head_mask looks for
    'depth_latent_mask' -- exactly what stack_optional_tensor_fields emits.

    Also pins the two mask regimes: partial coverage (some episodes lack the
    sidecar) and zero coverage (the key is absent entirely, so the head must
    default to all-False and take its ZeRO-aligned dummy path).
    """
    from starVLA.model.modules.uamvla.collator_helpers import stack_optional_tensor_fields

    fields = ["depth_latent", "depth_px"]
    with_depth = {"depth_latent": torch.randn(LATENT_C, GRID, GRID), "depth_px": torch.rand(PX, PX)}

    partial = stack_optional_tensor_fields([with_depth, {}], fields)
    assert partial["depth_latent"].shape == (2, LATENT_C, GRID, GRID)
    assert partial["depth_px"].shape == (2, PX, PX)
    assert partial["depth_latent_mask"].tolist() == [True, False]
    assert torch.count_nonzero(partial["depth_latent"][1]) == 0  # zero-padded row

    none = stack_optional_tensor_fields([{}, {}], fields)
    assert "depth_latent_mask" not in none, "absent field must not fabricate a mask"

    all_false = UamVLAOFT._resolve_head_mask(
        object(), "depth_latent", none, batch_size=2, device=torch.device("cpu"),
    )
    assert all_false.tolist() == [False, False]


def test_latent_and_px_come_from_the_same_frame(loader):
    """depth_latent[k] must describe depth_px[k], not some other frame.

    Cheap proxy without running the VAE: the latent's spatial mean should track
    the pixel map's mean across frames (both are monotone in overall depth).
    """
    lat = np.stack([
        loader._load_depth_latent_frame("depth_latent", 1, k, camera="image").mean().item()
        for k in range(0, 60, 6)
    ])
    px = np.stack([
        loader._load_depth_latent_frame("depth_px", 1, k, camera="image").mean().item()
        for k in range(0, 60, 6)
    ])
    corr = np.corrcoef(lat, px)[0, 1]
    assert abs(corr) > 0.5, f"latent/px frames look misaligned (corr={corr:.3f})"
