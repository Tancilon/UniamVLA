"""Spec §6.4 — stack_canonical must tolerate numpy-leaf canonical_state dicts
arriving via msgpack-numpy WebSocket deserialization."""
import numpy as np
import torch

from starVLA.model.modules.uamvla.collator_helpers import stack_canonical


def test_stack_canonical_accepts_numpy_leaves():
    a = {
        "arm_0": {
            "ee_pose":   np.zeros(9, dtype=np.float32),
            "joint_pos": np.zeros(7, dtype=np.float32),
        },
        "gripper_0": np.zeros(1, dtype=np.float32),
    }
    b = {
        "arm_0": {
            "ee_pose":   np.ones(9, dtype=np.float32),
            "joint_pos": np.ones(7, dtype=np.float32),
        },
        "gripper_0": np.ones(1, dtype=np.float32),
    }
    out = stack_canonical([a, b])

    assert isinstance(out["arm_0"]["ee_pose"], torch.Tensor)
    assert out["arm_0"]["ee_pose"].shape == (2, 9)
    assert out["arm_0"]["joint_pos"].shape == (2, 7)
    assert out["gripper_0"].shape == (2, 1)


def test_stack_canonical_still_accepts_torch_leaves():
    """Regression: training-side path must not break."""
    a = {
        "arm_0": {"ee_pose": torch.zeros(9), "joint_pos": torch.zeros(7)},
        "gripper_0": torch.zeros(1),
    }
    b = {
        "arm_0": {"ee_pose": torch.ones(9), "joint_pos": torch.ones(7)},
        "gripper_0": torch.ones(1),
    }
    out = stack_canonical([a, b])
    assert out["arm_0"]["ee_pose"].shape == (2, 9)
    assert torch.equal(out["gripper_0"], torch.stack([torch.zeros(1), torch.ones(1)]))
