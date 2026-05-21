import json

import numpy as np


def test_calvin_aux_sidecar_writer_uses_strict_layout(tmp_path, monkeypatch):
    from tests.test_calvin_lerobot_chunking import _stub_imageio_if_needed

    _stub_imageio_if_needed(monkeypatch)

    from tools.preprocess.calvin_preprocessor_lerobot import (
        CalvinPreprocessorLeRobot,
    )

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    seg = np.zeros((8, 8), dtype=np.int32)
    seg[2:5, 3:6] = 7
    grounding = (seg == 7).astype(np.float32)
    affordance = np.full((1, 20, 20), 0.25, dtype=np.float32)

    preprocessor = CalvinPreprocessorLeRobot(default_scene="D")
    preprocessor._emit_aux_denoising_sidecars(
        output_dir=tmp_path,
        trajectory_id=3,
        base_index=5,
        depth_static=depth,
        grounding_mask=grounding,
        affordance_heatmap=affordance,
        grounding_level="object",
    )

    assert rgb.shape == (8, 8, 3)
    np.testing.assert_allclose(
        np.load(tmp_path / "depths/static/3/5.npy"),
        depth,
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "grounding_masks/static/3/5.npy"),
        grounding,
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "affordance_heatmaps/static/3/5.npy"),
        affordance,
    )
    with open(tmp_path / "grounding_masks/static/3/5.json") as f:
        assert json.load(f)["grounding_level"] == "object"


def test_libero_aux_sidecar_writer_uses_strict_layout(tmp_path):
    from tools.preprocess.libero_preprocessor import LiberoPreprocessor

    depth = np.ones((8, 8), dtype=np.float32)
    seg = np.zeros((8, 8), dtype=np.int32)
    seg[2:5, 3:6] = 7
    grounding = (seg == 7).astype(np.float32)
    affordance = np.full((1, 20, 20), 0.75, dtype=np.float32)

    preprocessor = LiberoPreprocessor()
    preprocessor._emit_aux_denoising_sidecars(
        output_dir=tmp_path,
        trajectory_id="demo_0001",
        base_index=4,
        depth_static=depth,
        grounding_mask=grounding,
        affordance_heatmap=affordance,
        grounding_level="object",
    )

    np.testing.assert_allclose(
        np.load(tmp_path / "depths/static/demo_0001/4.npy"),
        depth,
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "grounding_masks/static/demo_0001/4.npy"),
        grounding,
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "affordance_heatmaps/static/demo_0001/4.npy"),
        affordance,
    )
    with open(tmp_path / "grounding_masks/static/demo_0001/4.json") as f:
        assert json.load(f)["grounding_level"] == "object"
