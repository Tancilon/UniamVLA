"""Unit tests for StateNormalizer."""
from __future__ import annotations

import pytest
import torch

from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer


def _make_stats_dict():
    """Synthetic state_stats covering franka_libero canonical layout."""
    return {
        "state_stats": {
            "franka_libero": {
                "arm_0.ee_pose": {
                    "q01": [-1.0] * 9,
                    "q99": [ 1.0] * 9,
                    "min": [-2.0] * 9,
                    "max": [ 2.0] * 9,
                    "mean": [0.0] * 9,
                    "std": [1.0] * 9,
                },
                "arm_0.joint_pos": {
                    "q01": [-3.14] * 7,
                    "q99": [ 3.14] * 7,
                    "min": [-3.14] * 7,
                    "max": [ 3.14] * 7,
                    "mean": [0.0] * 7,
                    "std": [1.0] * 7,
                },
                "gripper_0": {
                    "q01": [0.0],
                    "q99": [1.0],
                    "min": [0.0],
                    "max": [1.0],
                    "mean": [0.5],
                    "std": [0.3],
                },
            }
        }
    }


def _make_canonical():
    return {
        "arm_0": {
            "ee_pose":   torch.zeros(9, dtype=torch.float32),
            "joint_pos": torch.zeros(7, dtype=torch.float32),
        },
        "gripper_0": torch.tensor([0.5], dtype=torch.float32),
    }


def test_q99_normalizes_zero_input_to_zero_when_symmetric():
    """With q01=-1, q99=+1, input 0 → normalized 0 (midpoint)."""
    norm = StateNormalizer(
        stats_dict=_make_stats_dict(), embodiment="franka_libero", mode="q99"
    )
    out = norm(_make_canonical())
    assert torch.allclose(out["arm_0"]["ee_pose"], torch.zeros(9), atol=1e-6)
    assert torch.allclose(out["arm_0"]["joint_pos"], torch.zeros(7), atol=1e-6)


def test_q99_clips_out_of_range_input_to_neg_one_or_one():
    """Values past q99 must be clipped to +1; values past q01 must be clipped to -1."""
    norm = StateNormalizer(
        stats_dict=_make_stats_dict(), embodiment="franka_libero", mode="q99"
    )
    canonical = _make_canonical()
    canonical["arm_0"]["ee_pose"] = torch.full((9,), 10.0)
    canonical["arm_0"]["joint_pos"] = torch.full((7,), -10.0)
    out = norm(canonical)
    assert torch.allclose(out["arm_0"]["ee_pose"], torch.ones(9))
    assert torch.allclose(out["arm_0"]["joint_pos"], -torch.ones(7))


def test_mode_none_is_identity():
    norm = StateNormalizer(
        stats_dict=_make_stats_dict(), embodiment="franka_libero", mode="none"
    )
    inp = _make_canonical()
    inp["arm_0"]["ee_pose"] = torch.full((9,), 7.0)
    out = norm(inp)
    assert torch.equal(out["arm_0"]["ee_pose"], torch.full((9,), 7.0))


def test_apply_to_filters_fields():
    """Only fields in apply_to are normalized; others pass through unchanged."""
    norm = StateNormalizer(
        stats_dict=_make_stats_dict(),
        embodiment="franka_libero",
        mode="q99",
        apply_to=["arm_0.ee_pose"],
    )
    inp = _make_canonical()
    inp["arm_0"]["joint_pos"] = torch.full((7,), 10.0)
    out = norm(inp)
    assert torch.allclose(out["arm_0"]["ee_pose"], torch.zeros(9), atol=1e-6)
    assert torch.allclose(out["arm_0"]["joint_pos"], torch.full((7,), 10.0))


def test_q01_equals_q99_passes_through():
    """Constant feature (q01==q99) must be passed through unchanged then clamped to [-1,1]."""
    stats = _make_stats_dict()
    stats["state_stats"]["franka_libero"]["gripper_0"]["q01"] = [0.5]
    stats["state_stats"]["franka_libero"]["gripper_0"]["q99"] = [0.5]
    norm = StateNormalizer(stats_dict=stats, embodiment="franka_libero", mode="q99")
    inp = _make_canonical()
    inp["gripper_0"] = torch.tensor([0.5], dtype=torch.float32)
    out = norm(inp)
    # In-range value passes through (Normalizer's degenerate-case fallback)
    assert torch.allclose(out["gripper_0"], torch.tensor([0.5]))

    # Out-of-range constant: passthrough then clamp to [-1, 1] (inherited from Normalizer.q99)
    inp_high = _make_canonical()
    inp_high["gripper_0"] = torch.tensor([5.0], dtype=torch.float32)
    out_high = norm(inp_high)
    assert torch.allclose(out_high["gripper_0"], torch.tensor([1.0]))


def test_missing_state_stats_raises_when_mode_not_none():
    with pytest.raises(KeyError, match="state_stats"):
        StateNormalizer(
            stats_dict={"embodiment_stats": {}}, embodiment="franka_libero", mode="q99"
        )


def test_missing_state_stats_ok_when_mode_none():
    norm = StateNormalizer(
        stats_dict={"embodiment_stats": {}}, embodiment="franka_libero", mode="none"
    )
    out = norm(_make_canonical())
    assert torch.allclose(out["arm_0"]["ee_pose"], torch.zeros(9))


def test_missing_embodiment_in_state_stats_raises():
    with pytest.raises(KeyError, match="franka_other"):
        StateNormalizer(
            stats_dict=_make_stats_dict(), embodiment="franka_other", mode="q99"
        )


def test_constructor_does_not_mutate_input_stats_dict():
    """Building a StateNormalizer must not turn list-valued stats into tensors in caller's dict."""
    stats = _make_stats_dict()
    StateNormalizer(stats_dict=stats, embodiment="franka_libero", mode="q99")
    # Stats dict should still contain plain Python lists, not torch.Tensors
    ee = stats["state_stats"]["franka_libero"]["arm_0.ee_pose"]
    assert isinstance(ee["q01"], list), f"q01 mutated to {type(ee['q01']).__name__}"
    assert isinstance(ee["q99"], list)
    assert isinstance(ee["min"], list)
    assert isinstance(ee["max"], list)
