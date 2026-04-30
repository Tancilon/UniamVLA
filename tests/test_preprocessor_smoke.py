"""Smoke test: libero_preprocessor module imports and class instantiates without an env."""
import pytest


def test_libero_preprocessor_imports():
    pytest.importorskip("h5py", reason="h5py not installed; runs on remote")
    pytest.importorskip("libero", reason="LIBERO benchmark not installed; runs on remote")
    from tools.preprocess.libero_preprocessor import LiberoPreprocessor, MAX_ACTION_DIM, FRANKA_ACTION_DIM
    assert MAX_ACTION_DIM == 24
    assert FRANKA_ACTION_DIM == 7


def test_libero_preprocessor_construct_no_env():
    pytest.importorskip("h5py", reason="h5py not installed; runs on remote")
    pytest.importorskip("libero", reason="LIBERO benchmark not installed; runs on remote")
    from tools.preprocess.libero_preprocessor import LiberoPreprocessor
    p = LiberoPreprocessor(suite="libero_spatial", target_object_keyword=None, target_resolver=None)
    assert p.suite == "libero_spatial"


def test_calvin_preprocessor_imports():
    """CALVIN runs only in calvin_env conda — skip on Mac local."""
    pytest.importorskip("calvin_env", reason="CALVIN runs on remote in calvin_env conda")
    from tools.preprocess.calvin_preprocessor import CalvinPreprocessor
    assert CalvinPreprocessor is not None


def test_state_stats_block_written_with_canonical_layout(tmp_path):
    """statistics.yaml must contain state_stats with canonical-shape per-field stats.

    Exercises _write_statistics directly with synthetic samples (no MuJoCo env
    required) — mirrors the module-level import style of the existing smoke tests.
    """
    import math
    import yaml
    import numpy as np
    from tools.preprocess.libero_preprocessor import (
        LiberoPreprocessor,
        FRANKA_ACTION_DIM,
        MAX_ACTION_DIM,
        STATIC_CAM,
        WRIST_CAM,
    )

    rng = np.random.default_rng(42)
    N = 20  # enough samples for meaningful quantiles

    def _make_sample(i):
        action_7d = rng.uniform(-1.0, 1.0, size=FRANKA_ACTION_DIM).tolist()
        action_24d = action_7d + [0.0] * (MAX_ACTION_DIM - FRANKA_ACTION_DIM)
        return {
            "action": action_24d,
            "robot_obs": rng.uniform(-1.0, 1.0, size=3).tolist(),
            "ee_pos": rng.uniform(-1.0, 1.0, size=3).tolist(),
            # axis-angle (3-D rotvec) — small angles so rotation is valid
            "ee_axis_angle": (rng.uniform(-0.3, 0.3, size=3)).tolist(),
            "joint_pos": rng.uniform(-math.pi, math.pi, size=7).tolist(),
            "gripper_qpos": [rng.uniform(0.0, 0.04), rng.uniform(0.0, 0.04)],
        }

    samples = [_make_sample(i) for i in range(N)]

    # Minimal camera intrinsics (3×3 flat list, same structure as real data)
    dummy_intrinsic = [1.0] * 9
    camera_intrinsics = {STATIC_CAM: dummy_intrinsic, WRIST_CAM: dummy_intrinsic}

    preprocessor = LiberoPreprocessor(
        suite="libero_spatial",
        target_object_keyword=None,
        target_resolver=None,
    )
    preprocessor._write_statistics(samples, tmp_path, camera_intrinsics)

    stats_file = tmp_path / "statistics.yaml"
    assert stats_file.exists(), "statistics.yaml was not written"

    with open(stats_file) as f:
        stats = yaml.safe_load(f)

    assert "state_stats" in stats, "statistics.yaml missing 'state_stats' block"
    franka = stats["state_stats"]["franka_libero"]

    EXPECTED = {
        "arm_0.ee_pose":   9,
        "arm_0.joint_pos": 7,
        "gripper_0":       1,
    }
    for field_path, dim in EXPECTED.items():
        assert field_path in franka, f"missing {field_path}"
        per_field = franka[field_path]
        for stat_key in ("q01", "q99", "min", "max", "mean", "std"):
            assert stat_key in per_field, f"{field_path} missing {stat_key}"
            assert len(per_field[stat_key]) == dim, (
                f"{field_path}.{stat_key} expected dim {dim}, got {len(per_field[stat_key])}"
            )
        # Quantile interpolation invariant: min <= q01 <= q99 <= max element-wise.
        for mn, q1, q9, mx in zip(per_field["min"], per_field["q01"],
                                   per_field["q99"], per_field["max"]):
            assert mn <= q1 + 1e-6, f"{field_path}: min ({mn}) > q01 ({q1})"
            assert q1 <= q9 + 1e-6, f"{field_path}: q01 ({q1}) > q99 ({q9})"
            assert q9 <= mx + 1e-6, f"{field_path}: q99 ({q9}) > max ({mx})"
