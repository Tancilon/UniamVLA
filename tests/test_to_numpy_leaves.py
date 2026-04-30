"""Spec §6.3.3 — module-level helper that walks a 1-level nested dict and
converts torch.Tensor leaves to numpy.ndarray, leaving other types untouched."""
import numpy as np
import torch

from examples.LIBERO.eval_files.model2libero_interface import _to_numpy_leaves


def test_to_numpy_leaves_converts_torch_in_nested_dict():
    src = {
        "arm_0": {
            "ee_pose":   torch.tensor([1.0, 2.0, 3.0]),
            "joint_pos": torch.zeros(7),
        },
        "gripper_0": torch.tensor([0.5]),
    }

    out = _to_numpy_leaves(src)

    assert isinstance(out["arm_0"]["ee_pose"], np.ndarray)
    assert isinstance(out["arm_0"]["joint_pos"], np.ndarray)
    assert isinstance(out["gripper_0"], np.ndarray)
    np.testing.assert_array_equal(out["arm_0"]["ee_pose"], np.array([1.0, 2.0, 3.0]))


def test_to_numpy_leaves_passes_through_non_torch():
    src = {
        "a": np.array([1, 2, 3], dtype=np.float32),
        "b": {"c": [1, 2], "d": "hello"},
    }
    out = _to_numpy_leaves(src)

    assert isinstance(out["a"], np.ndarray)
    assert out["b"]["c"] == [1, 2]
    assert out["b"]["d"] == "hello"
