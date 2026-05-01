"""Local unit tests for the UamVLA CALVIN preprocessing + training pipeline.

These tests run on any machine without requiring calvin_env / PyBullet.
Coverage:
  - statistics.yaml schema migration (Task 1)
  - mixture registry entry (Task 2)
  - synthetic dataset → UamVLADataset chain (Task 3)
  - CLI arg parsing (Task 4)
  - training yaml loadable (Task 5)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# Task 1: _write_statistics schema migration
# ============================================================================

def _make_synthetic_calvin_samples(n: int = 6) -> list[dict]:
    """Build n CALVIN-shape sample dicts. Values are deterministic but varied
    across samples so q01 != q99 (otherwise q99 normalization would degenerate
    to identity and downstream tests can't tell stats were applied).
    """
    rng = np.random.default_rng(seed=42)
    samples = []
    for i in range(n):
        # 15-dim robot_obs: tcp(3) + euler(3) + gripper_width(1) + joints(7) + gripper_cmd(1)
        # Ranges chosen to match the values we probed from a real CALVIN frame:
        #   tcp ≈ [0.10, -0.09, 0.57], euler ≈ [3.10, 0.02, 1.50], gripper_width ≈ 0.08,
        #   joints ≈ 7-tuple within [-π, π], gripper_cmd ∈ {-1, 1}
        tcp = np.array([0.10, -0.09, 0.57], dtype=np.float64) + 0.05 * rng.standard_normal(3)
        euler = np.array([3.10, 0.02, 1.50], dtype=np.float64) + 0.05 * rng.standard_normal(3)
        gripper_width = 0.08 + 0.005 * rng.standard_normal()
        joints = np.array([-0.59, 0.90, 1.74, -1.84, -0.92, 1.60, 0.59], dtype=np.float64) \
                 + 0.1 * rng.standard_normal(7)
        gripper_cmd = 1.0
        robot_obs_15 = np.concatenate([
            tcp, euler, [gripper_width], joints, [gripper_cmd],
        ]).tolist()

        rel_actions_7 = (0.01 * rng.standard_normal(7)).tolist()
        action_24 = rel_actions_7 + [0.0] * 17

        samples.append({
            "id": f"calvin_debug_ep00000_step{i:04d}",
            "episode_id": "calvin_debug_ep00000",
            "step_idx": i,
            "total_steps": n,
            "image": [
                f"images/obs/static/calvin_debug_ep00000_step{i:04d}.jpg",
                f"images/obs/wrist/calvin_debug_ep00000_step{i:04d}.jpg",
            ],
            "instruction": "test instruction",
            "embodiment": "franka_calvin",
            "action_dim": 7,
            "action": action_24,
            "action_mask": [1] * 7 + [0] * 17,
            "robot_obs": robot_obs_15,
            "dataset_source": "calvin_debug",
        })
    return samples


def test_write_statistics_schema(tmp_path):
    """`_write_statistics` emits the new state_stats[franka_calvin] block."""
    from tools.preprocess.calvin_preprocessor import _write_statistics

    samples = _make_synthetic_calvin_samples(n=6)
    intrinsics = {
        "static": {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
        "wrist":  {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
    }

    _write_statistics(samples, tmp_path, intrinsics)

    with open(tmp_path / "statistics.yaml") as f:
        stats = yaml.safe_load(f)

    # Top-level shape
    assert set(stats.keys()) == {
        "view_names", "max_action_dim", "embodiment_stats",
        "state_stats", "cameras", "point_cloud",
    }, f"unexpected top-level keys: {sorted(stats.keys())}"

    # state_stats[franka_calvin] presence + shape
    assert "franka_calvin" in stats["state_stats"]
    fs = stats["state_stats"]["franka_calvin"]
    assert set(fs.keys()) == {"arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"}, \
        f"unexpected state field keys: {sorted(fs.keys())}"

    expected_lengths = {"arm_0.ee_pose": 9, "arm_0.joint_pos": 7, "gripper_0": 1}
    expected_substats = {"q01", "q99", "min", "max", "mean", "std"}
    for path, length in expected_lengths.items():
        assert set(fs[path].keys()) == expected_substats, (
            f"{path}: substats {sorted(fs[path].keys())} != {sorted(expected_substats)}"
        )
        for k in expected_substats:
            assert len(fs[path][k]) == length, (
                f"{path}.{k}: len={len(fs[path][k])}, expected {length}"
            )


def test_write_statistics_no_legacy_keys(tmp_path):
    """`robot_obs_mean` / `robot_obs_std` are removed from the new schema."""
    from tools.preprocess.calvin_preprocessor import _write_statistics

    samples = _make_synthetic_calvin_samples(n=4)
    intrinsics = {
        "static": {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
        "wrist":  {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
    }

    _write_statistics(samples, tmp_path, intrinsics)

    with open(tmp_path / "statistics.yaml") as f:
        stats = yaml.safe_load(f)

    assert "robot_obs_mean" not in stats, \
        "Legacy key 'robot_obs_mean' must be removed in the new schema."
    assert "robot_obs_std" not in stats, \
        "Legacy key 'robot_obs_std' must be removed in the new schema."


# ============================================================================
# Task 2: mixture registry entry
# ============================================================================

def test_calvin_uamvla_mixture_registered():
    from starVLA.dataloader.uamvla_dataset import DATASET_NAMED_MIXTURES

    assert "calvin_uamvla" in DATASET_NAMED_MIXTURES, (
        f"Missing 'calvin_uamvla' mixture; available: {list(DATASET_NAMED_MIXTURES)}"
    )
    assert DATASET_NAMED_MIXTURES["calvin_uamvla"] == [
        ("task_D_D", 1.0, "franka_calvin"),
    ]


# ============================================================================
# Task 3: synthetic-fixture dataset chain test
# ============================================================================

def _build_synthetic_calvin_dataset(tmp_path: Path, n_samples: int = 6) -> Path:
    """Build a complete CALVIN-shape UAM dataset under tmp_path.

    Produces:
      - data.jsonl with `n_samples` rows (one episode of length n_samples)
      - statistics.yaml with embodiment_stats[franka_calvin] + state_stats[franka_calvin]
      - dummy 256x256 RGB jpgs at every referenced image path
      - 1024×3 zero point clouds at every referenced point_cloud path
      - dummy 256×256 float32 depth npys at every referenced depth path

    Stats are computed from the same fixture data run through CalvinAdapter,
    keeping the fixture self-consistent (q99 != q01 → StateNormalizer
    actually transforms values, so downstream tests can detect it).
    """
    from PIL import Image

    samples = _make_synthetic_calvin_samples(n=n_samples)

    # Augment each sample with the aux fields a real CALVIN preprocessor would write.
    # Note: depth_static / depth_wrist / wrist_cam_extrinsic are NOT consumed by
    # UamVLADataset._load_aux_targets (see uamvla_dataset.py:153-186). They are
    # included here for parity with the real preprocessor's JSONL schema; future
    # tasks/heads that consume them will not need a fixture change.
    for s in samples:
        sid = s["id"]
        s["image_target"] = f"images/target/{sid}.jpg"
        s["image_future"] = f"images/future/{s['episode_id']}.jpg"
        s["depth_static"] = f"depth/static/{sid}.npy"
        s["depth_wrist"]  = f"depth/wrist/{sid}.npy"
        s["point_cloud"]  = f"point_clouds/{sid}.npy"
        s["pose_6d"] = {
            "rotation":    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # identity 6D rot
            "translation": [0.0, 0.0, 0.0],
            "object_id":   "test_object",
        }
        s["static_cam_extrinsic"] = {
            "rotation":    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
        }
        s["wrist_cam_extrinsic"] = {
            "rotation":    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
        }

    # Create directory tree.
    for sub in (
        "images/obs/static", "images/obs/wrist",
        "images/target", "images/future",
        "depth/static", "depth/wrist",
        "point_clouds",
    ):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    # Write data.jsonl.
    with open(tmp_path / "data.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Write dummy media files.
    blank_rgb = Image.new("RGB", (256, 256), color=(127, 127, 127))
    pc_zeros = np.zeros((1024, 3), dtype=np.float32)
    depth_zeros = np.zeros((256, 256), dtype=np.float32)
    futures_written: set[str] = set()
    for s in samples:
        for img_rel in s["image"] + [s["image_target"]]:
            blank_rgb.save(tmp_path / img_rel, quality=95)
        if s["image_future"] not in futures_written:
            blank_rgb.save(tmp_path / s["image_future"], quality=95)
            futures_written.add(s["image_future"])
        np.save(tmp_path / s["point_cloud"], pc_zeros)
        np.save(tmp_path / s["depth_static"], depth_zeros)
        np.save(tmp_path / s["depth_wrist"], depth_zeros)

    # Compute stats by running samples through CalvinAdapter (self-consistent).
    from tools.preprocess.calvin_preprocessor import _write_statistics
    intrinsics = {
        "static": {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
        "wrist":  {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
    }
    _write_statistics(samples, tmp_path, intrinsics)

    return tmp_path


def test_synthetic_dataset_loads(tmp_path):
    """End-to-end: synthetic CALVIN fixture → UamVLADataset → canonical_state w/ q99."""
    import torch
    from starVLA.dataloader.uamvla_dataset import UamVLADataset
    from starVLA.model.modules.uamvla.data.embodiment_adapter import CalvinAdapter

    data_root = _build_synthetic_calvin_dataset(tmp_path, n_samples=6)

    ds = UamVLADataset(
        data_root=data_root,
        embodiment="franka_calvin",
        action_horizon=8,
        normalization={
            "mode": "q99",
            "apply_to": ["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"],
        },
    )

    assert len(ds) == 6, f"expected 6 samples, got {len(ds)}"

    sample = ds[0]
    cs = sample["canonical_state"]
    assert cs["arm_0"]["ee_pose"].shape == (9,), cs["arm_0"]["ee_pose"].shape
    assert cs["arm_0"]["joint_pos"].shape == (7,), cs["arm_0"]["joint_pos"].shape
    assert cs["gripper_0"].shape == (1,), cs["gripper_0"].shape

    assert sample["action"].shape == (8, 7), sample["action"].shape
    assert sample["action_mask"].shape == (8, 7), sample["action_mask"].shape

    # State normalizer should have actually transformed the values. Recompute
    # the raw (pre-normalizer) canonical state independently and verify the
    # post-normalizer tensor differs. If they match, StateNormalizer silently
    # no-oped (e.g., mode='none' fell through, or stats were degenerate).
    raw_canonical = CalvinAdapter().to_canonical(ds.samples[0])
    raw_ee_pose = raw_canonical["arm_0"]["ee_pose"]
    assert not torch.allclose(cs["arm_0"]["ee_pose"], raw_ee_pose), (
        "Post-normalizer ee_pose equals raw canonical — StateNormalizer no-oped. "
        "Check that mode='q99' is wired and state_stats has non-degenerate q01/q99."
    )


# ============================================================================
# Task 4: CLI runner argument parsing
# ============================================================================

CLI_PATH = ROOT / "runners" / "preprocess_calvin.py"


def test_cli_help():
    """`--help` exits 0 and lists all seven expected args."""
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    for arg in (
        "--input_dir", "--output_dir", "--dataset_source",
        "--num_workers", "--default_scene",
        "--on_resolve_failure", "--on_missing_target",
    ):
        assert arg in result.stdout, f"--help output missing {arg}\n{result.stdout}"


def test_cli_required_args_missing_input():
    """Omitting --input_dir fails with non-zero exit."""
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--output_dir", "/tmp/x"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0, "Missing --input_dir should fail"
    assert "input_dir" in result.stderr.lower() or "input_dir" in result.stdout.lower()


def test_cli_required_args_missing_output():
    """Omitting --output_dir fails with non-zero exit."""
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--input_dir", "/tmp/x"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0, "Missing --output_dir should fail"
    assert "output_dir" in result.stderr.lower() or "output_dir" in result.stdout.lower()


# ============================================================================
# Task 5: training yaml loadable
# ============================================================================

def test_yaml_loadable():
    """uamvla_calvin.yaml exists, parses, and has the expected substitutions."""
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    yaml_path = ROOT / "starVLA" / "config" / "training" / "uamvla_calvin.yaml"
    assert yaml_path.exists(), f"missing {yaml_path}"

    cfg = OmegaConf.load(str(yaml_path))

    # Substitutions vs the LIBERO clone
    assert cfg.run_id == "uamvla_calvin_phase1", f"run_id = {cfg.run_id!r}"
    assert cfg.framework.embodiment.name == "franka_calvin", \
        f"framework.embodiment.name = {cfg.framework.embodiment.name!r}"
    assert cfg.datasets.vla_data.data_mix == "calvin_uamvla", \
        f"datasets.vla_data.data_mix = {cfg.datasets.vla_data.data_mix!r}"
    assert cfg.datasets.vla_data.data_root_dir == "datasets/uamvla_calvin", \
        f"datasets.vla_data.data_root_dir = {cfg.datasets.vla_data.data_root_dir!r}"

    # Identical-to-LIBERO contract checks (catch yaml drift):
    assert list(cfg.framework.state_encoder.normalization.apply_to) == [
        "arm_0.ee_pose", "arm_0.joint_pos", "gripper_0",
    ], f"unexpected state normalization apply_to: {cfg.framework.state_encoder.normalization.apply_to}"
    assert cfg.framework.name == "UamVLA"
    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.future_action_window_size == 7


# ============================================================================
# Task A (this PR): partial-aux row loads through UamVLADataset
# ============================================================================

def _build_partial_aux_calvin_dataset(
    tmp_path: Path,
    n_samples: int = 6,
    n_occluded: int = 2,
) -> Path:
    """Like _build_synthetic_calvin_dataset, but the LAST `n_occluded` rows
    omit image_target / point_cloud / pose_6d (and skip writing those media
    files), mimicking what the new preprocessor produces when seg fails.

    statistics.yaml is computed over ALL n_samples — production's
    _write_statistics(samples, ...) iterates every sample's robot_obs via
    CalvinAdapter regardless of occlusion, so stats span every frame.
    """
    from PIL import Image
    from tools.preprocess.calvin_preprocessor import _write_statistics

    samples = _make_synthetic_calvin_samples(n=n_samples)

    # Always-fields on every sample.
    for s in samples:
        sid = s["id"]
        s["image_future"] = f"images/future/{s['episode_id']}.jpg"
        s["depth_static"] = f"depth/static/{sid}.npy"
        s["depth_wrist"]  = f"depth/wrist/{sid}.npy"
        s["static_cam_extrinsic"] = {
            "rotation":    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
        }
        s["wrist_cam_extrinsic"] = {
            "rotation":    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
        }

    # Aux fields on the first (n_samples - n_occluded) rows only.
    n_visible = n_samples - n_occluded
    for s in samples[:n_visible]:
        sid = s["id"]
        s["image_target"] = f"images/target/{sid}.jpg"
        s["point_cloud"]  = f"point_clouds/{sid}.npy"
        s["pose_6d"] = {
            "rotation":    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
            "object_id":   "test_object",
        }
    # samples[n_visible:] are partial — image_target / point_cloud / pose_6d absent.

    for sub in (
        "images/obs/static", "images/obs/wrist",
        "images/target", "images/future",
        "depth/static", "depth/wrist",
        "point_clouds",
    ):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    with open(tmp_path / "data.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    blank_rgb = Image.new("RGB", (256, 256), color=(127, 127, 127))
    pc_zeros = np.zeros((1024, 3), dtype=np.float32)
    depth_zeros = np.zeros((256, 256), dtype=np.float32)
    futures_written: set[str] = set()
    for s in samples:
        # Always-fields: write obs jpgs, depth npys, future jpg.
        for img_rel in s["image"]:
            blank_rgb.save(tmp_path / img_rel, quality=95)
        if s["image_future"] not in futures_written:
            blank_rgb.save(tmp_path / s["image_future"], quality=95)
            futures_written.add(s["image_future"])
        np.save(tmp_path / s["depth_static"], depth_zeros)
        np.save(tmp_path / s["depth_wrist"], depth_zeros)
        # Seg-dependent: only when present in the sample dict.
        if "image_target" in s:
            blank_rgb.save(tmp_path / s["image_target"], quality=95)
        if "point_cloud" in s:
            np.save(tmp_path / s["point_cloud"], pc_zeros)

    intrinsics = {
        "static": {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
        "wrist":  {"fx": 100.0, "fy": 100.0, "cx": 128.0, "cy": 128.0},
    }
    _write_statistics(samples, tmp_path, intrinsics)
    return tmp_path


def test_partial_aux_row_loads(tmp_path):
    """Mixed full + partial-aux dataset loads through UamVLADataset.

    Visible rows (first 4) carry image_target / point_cloud / pose_gt.
    Occluded rows (last 2) omit them. All 6 rows must load — the legacy
    REQUIRED_OPTIONAL_FIELDS filter would have either dropped the partial
    rows (kept only the 4 truthy filtered samples) or fallen through to
    `or all_samples`. After Task 2's filter deletion, all 6 load directly.
    """
    from starVLA.dataloader.uamvla_dataset import UamVLADataset

    data_root = _build_partial_aux_calvin_dataset(
        tmp_path, n_samples=6, n_occluded=2,
    )

    ds = UamVLADataset(
        data_root=data_root,
        embodiment="franka_calvin",
        action_horizon=8,
        normalization={
            "mode": "q99",
            "apply_to": ["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"],
        },
    )

    assert len(ds) == 6, (
        f"expected all 6 rows to load (filter deleted), got {len(ds)}"
    )

    # Visible rows: aux fields present.
    assert "image_target" in ds[0], "row 0 (visible) must keep image_target"
    assert "point_cloud" in ds[0], "row 0 (visible) must keep point_cloud"
    assert "pose_gt" in ds[0], "row 0 (visible) must keep pose_gt"

    # Occluded rows: aux fields absent.
    assert "image_target" not in ds[5], "row 5 (occluded) must omit image_target"
    assert "point_cloud" not in ds[5], "row 5 (occluded) must omit point_cloud"
    assert "pose_gt" not in ds[5], "row 5 (occluded) must omit pose_gt"

    # Universal fields present on every row, including occluded ones.
    for i in range(6):
        assert ds[i]["action"].shape == (8, 7), (
            f"row {i}: action shape {ds[i]['action'].shape} != (8, 7)"
        )
        assert ds[i]["canonical_state"]["arm_0"]["ee_pose"].shape == (9,)
        assert ds[i]["canonical_state"]["arm_0"]["joint_pos"].shape == (7,)
        assert ds[i]["canonical_state"]["gripper_0"].shape == (1,)


def test_calvin_abc_d_uamvla_mixture_registered():
    """The post-fix task_ABC_D data root is reachable from a mixture key."""
    from starVLA.dataloader.uamvla_dataset import DATASET_NAMED_MIXTURES

    assert "calvin_abc_d_uamvla" in DATASET_NAMED_MIXTURES, (
        f"Missing 'calvin_abc_d_uamvla' mixture; available: "
        f"{list(DATASET_NAMED_MIXTURES)}"
    )
    assert DATASET_NAMED_MIXTURES["calvin_abc_d_uamvla"] == [
        ("task_ABC_D", 1.0, "franka_calvin"),
    ], DATASET_NAMED_MIXTURES["calvin_abc_d_uamvla"]

    # Backward-compat: the prior calvin_uamvla -> task_D_D entry must remain.
    assert DATASET_NAMED_MIXTURES["calvin_uamvla"] == [
        ("task_D_D", 1.0, "franka_calvin"),
    ], DATASET_NAMED_MIXTURES["calvin_uamvla"]


# ============================================================================
# Test B (this PR): _resolve_head_mask defaults for missing aux mask keys
# ============================================================================

def test_resolve_head_mask_defaults_for_missing_keys():
    """`_resolve_head_mask` returns all-True for `action` (universal) and
    all-False for `pose` / `recon` / `future` when their mask key is missing
    from batch_dict.

    The missing-key case is the one stack_optional_tensor_fields produces when
    every sample in the batch lacked the aux field (see collator_helpers.py:
    if no sample has the field, both the field tensor AND its `_mask` key are
    omitted entirely). Without this helper, UamVLA.forward() would default
    such missing masks to torch.ones(...) and crash inside pose / recon heads
    on `batch["image_target"]` / `assert "point_cloud" in batch`.

    All-False routes through each head's existing `not mask.any()` early exit
    in compute_loss → returns get_dummy_loss(...) — preserving the DeepSpeed
    ZeRO-2 all-reduce shape across ranks.
    """
    import torch
    from starVLA.model.framework.VLM4A.UamVLA import (
        _resolve_head_mask,
        _UNIVERSAL_HEADS,
    )

    # Sanity: the registry of universal heads is the documented one.
    assert _UNIVERSAL_HEADS == ("action",), (
        f"_UNIVERSAL_HEADS drifted from spec: {_UNIVERSAL_HEADS!r}"
    )

    B = 4
    device = torch.device("cpu")
    batch_empty: dict = {}

    action_mask = _resolve_head_mask("action", batch_empty, B, device)
    pose_mask   = _resolve_head_mask("pose",   batch_empty, B, device)
    recon_mask  = _resolve_head_mask("recon",  batch_empty, B, device)
    future_mask = _resolve_head_mask("future", batch_empty, B, device)

    # Shape & dtype contract.
    for name, m in (
        ("action", action_mask), ("pose", pose_mask),
        ("recon", recon_mask), ("future", future_mask),
    ):
        assert m.dtype == torch.bool, f"{name}: dtype {m.dtype} != bool"
        assert m.shape == (B,), f"{name}: shape {tuple(m.shape)} != ({B},)"
        assert m.device == device, f"{name}: device {m.device} != {device}"

    # Default values per universality.
    assert action_mask.all(),       "action default must be all-True (universal)"
    assert not pose_mask.any(),     "pose default must be all-False"
    assert not recon_mask.any(),    "recon default must be all-False"
    assert not future_mask.any(),   "future default must be all-False"

    # When the mask IS present in batch_dict, the helper returns it verbatim.
    explicit = torch.tensor([True, False, True, True])
    out = _resolve_head_mask("pose", {"pose_mask": explicit}, B, device)
    assert out is explicit, (
        "present mask key must be returned unchanged (no copy, no recompute)"
    )


# ============================================================================
# Test C (this PR): collator invariant — _resolve_head_mask precondition
# ============================================================================

def test_all_occluded_batch_collator_invariant():
    """Lock in the precondition the §3.3 framework patch relies on.

    When every sample in a batch lacks an aux field, stack_optional_tensor_fields
    must omit BOTH the field tensor and its `_mask` key from the output dict.
    The all-False default in _resolve_head_mask is the right choice EXACTLY
    because the helper sees the mask key as missing — if a future refactor
    started inserting an empty-tensor + all-False mask, this test would catch
    the drift (Test B's "missing key" branch would no longer be exercised in
    production, even though Test B itself would still pass).

    Mixed case is also covered: when at least one sample has the field, both
    the field tensor and the all-True (or partial-True) mask must be present.
    """
    import torch
    from starVLA.model.modules.uamvla.collator_helpers import (
        stack_optional_tensor_fields,
    )

    # Case 1: every sample lacks image_target and point_cloud.
    samples = [
        {"image_future": torch.zeros(3, 224, 224)},
        {"image_future": torch.zeros(3, 224, 224)},
    ]
    batch = stack_optional_tensor_fields(
        samples, ["image_target", "image_future", "point_cloud"],
    )
    assert "image_target"      not in batch
    assert "image_target_mask" not in batch
    assert "point_cloud"       not in batch
    assert "point_cloud_mask"  not in batch
    assert "image_future"      in batch
    assert "image_future_mask" in batch
    assert batch["image_future_mask"].dtype == torch.bool
    assert batch["image_future_mask"].all(), (
        "every sample has image_future → mask is all-True"
    )

    # Case 2: mixed — sample 0 has image_target, sample 1 does not.
    samples_mixed = [
        {"image_target": torch.zeros(3, 224, 224),
         "image_future": torch.zeros(3, 224, 224)},
        {"image_future": torch.zeros(3, 224, 224)},
    ]
    batch_mixed = stack_optional_tensor_fields(
        samples_mixed, ["image_target", "image_future"],
    )
    assert "image_target"      in batch_mixed
    assert "image_target_mask" in batch_mixed
    assert batch_mixed["image_target_mask"].tolist() == [True, False], (
        f"mixed mask must be [True, False], got "
        f"{batch_mixed['image_target_mask'].tolist()}"
    )
