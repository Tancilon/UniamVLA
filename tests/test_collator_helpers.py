"""Unit tests for ``starVLA/model/modules/uamvla/collator_helpers.py``.

These helpers are pure tensor ops with no backbone or HF dependencies —
they run cleanly on a Mac/CPU sandbox.
"""
import pytest
import torch

from starVLA.model.modules.uamvla.collator_helpers import (
    stack_canonical,
    stack_optional_tensor_fields,
    stack_pose_gt,
    stack_static_cam_extrinsic,
)


def test_stack_canonical_flat_and_nested():
    """stack_canonical handles both flat tensor leaves and one-level-nested dicts."""
    s0 = {
        "right_arm": torch.tensor([1.0, 2.0, 3.0]),
        "right_gripper": {
            "joint": torch.tensor([0.1, 0.2]),
            "force": torch.tensor([0.5]),
        },
    }
    s1 = {
        "right_arm": torch.tensor([4.0, 5.0, 6.0]),
        "right_gripper": {
            "joint": torch.tensor([0.3, 0.4]),
            "force": torch.tensor([0.7]),
        },
    }

    out = stack_canonical([s0, s1])

    # Flat field stacked along dim 0
    assert out["right_arm"].shape == (2, 3)
    assert torch.equal(out["right_arm"][0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(out["right_arm"][1], torch.tensor([4.0, 5.0, 6.0]))

    # Nested dict: each leaf stacked along dim 0
    assert isinstance(out["right_gripper"], dict)
    assert out["right_gripper"]["joint"].shape == (2, 2)
    assert torch.equal(
        out["right_gripper"]["joint"], torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    )
    assert out["right_gripper"]["force"].shape == (2, 1)
    assert torch.equal(
        out["right_gripper"]["force"], torch.tensor([[0.5], [0.7]])
    )


def test_stack_optional_tensor_fields_mixed_presence():
    """stack_optional_tensor_fields zero-pads missing slots and emits parallel masks."""
    img_target = torch.ones(3, 4, 4, dtype=torch.float32)
    point_cloud = torch.full((128, 3), 2.0)

    samples = [
        {"image_target": img_target, "point_cloud": point_cloud},
        {"point_cloud": point_cloud},  # image_target absent
        {"image_target": img_target},  # point_cloud absent
    ]

    out = stack_optional_tensor_fields(
        samples, ["image_target", "image_future", "point_cloud"],
    )

    # image_future is absent everywhere → omitted entirely
    assert "image_future" not in out
    assert "image_future_mask" not in out

    # image_target: shape (B,) + template.shape, zeros where missing
    assert "image_target" in out
    assert out["image_target"].shape == (3, 3, 4, 4)
    assert out["image_target"].dtype == torch.float32
    assert torch.equal(out["image_target"][0], img_target)
    assert torch.equal(out["image_target"][1], torch.zeros_like(img_target))
    assert torch.equal(out["image_target"][2], img_target)
    assert torch.equal(
        out["image_target_mask"], torch.tensor([True, False, True])
    )

    # point_cloud
    assert out["point_cloud"].shape == (3, 128, 3)
    assert torch.equal(out["point_cloud"][0], point_cloud)
    assert torch.equal(out["point_cloud"][1], point_cloud)
    assert torch.equal(out["point_cloud"][2], torch.zeros_like(point_cloud))
    assert torch.equal(
        out["point_cloud_mask"], torch.tensor([True, True, False])
    )


def test_stack_pose_gt_mixed_and_absent():
    """stack_pose_gt returns None when no sample has pose_gt; stacks with zero-pad otherwise."""
    # Case 1: nothing → None
    assert stack_pose_gt([{}, {"foo": 1}]) is None

    # Case 2: mixed presence
    rot_a = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 6D rotation
    trans_a = torch.tensor([1.0, 2.0, 3.0])
    samples = [
        {"pose_gt": {"rotation": rot_a, "translation": trans_a}},
        {},  # missing
        {"pose_gt": {"rotation": rot_a * 2, "translation": trans_a * 2}},
    ]

    out = stack_pose_gt(samples)
    assert out is not None
    assert "pose_gt" in out and "pose_mask" in out

    pose = out["pose_gt"]
    assert pose["rotation"].shape == (3, 6)
    assert torch.equal(pose["rotation"][0], rot_a)
    assert torch.equal(pose["rotation"][1], torch.zeros_like(rot_a))
    assert torch.equal(pose["rotation"][2], rot_a * 2)

    assert pose["translation"].shape == (3, 3)
    assert torch.equal(pose["translation"][0], trans_a)
    assert torch.equal(pose["translation"][1], torch.zeros(3))
    assert torch.equal(pose["translation"][2], trans_a * 2)

    assert torch.equal(out["pose_mask"], torch.tensor([True, False, True]))


def test_stack_pose_gt_missing_slot_inherits_template_device():
    """Regression: missing-slot fillers must follow the present sample's device.

    When the dataset emits partial-aux rows (CALVIN preprocess partial-aux
    retention spec), some samples lack `pose_gt`. The else-branch filler must
    use ``torch.zeros_like(template)`` — not ``torch.zeros(3)`` — so the
    filler tensor inherits the template's device. Otherwise, when the present
    samples are on GPU, ``torch.stack([cuda, cpu, cuda])`` raises:

        RuntimeError: Expected all tensors to be on the same device,
                      but found at least two devices, cuda:N and cpu

    PyTorch's `meta` device reproduces the same cross-device guard on
    macOS / CPU-only CI without needing a real GPU.
    """
    # Place the present sample's pose_gt on the meta device. Filler must
    # follow — if the implementation hardcodes torch.zeros(3) (CPU), stack
    # raises "Tensor on device cpu is not on the expected device meta".
    rot_meta = torch.zeros(6, device="meta")
    trans_meta = torch.zeros(3, device="meta")
    samples = [
        {"pose_gt": {"rotation": rot_meta, "translation": trans_meta}},
        {},  # missing — filler must inherit meta device
    ]

    out = stack_pose_gt(samples)  # Pre-fix this raises RuntimeError; post-fix succeeds.

    assert out is not None
    pose = out["pose_gt"]
    assert pose["rotation"].device == torch.device("meta"), pose["rotation"].device
    assert pose["translation"].device == torch.device("meta"), pose["translation"].device
    assert pose["rotation"].shape == (2, 6)
    assert pose["translation"].shape == (2, 3)


def test_stack_static_cam_extrinsic_mixed_and_absent():
    """stack_static_cam_extrinsic returns None when absent everywhere; stacks with zero-pad otherwise."""
    # Case 1: nothing → None
    assert stack_static_cam_extrinsic([{}, {"unrelated": "x"}]) is None

    # Case 2: mixed presence
    rot = torch.eye(3)
    trans = torch.tensor([0.1, 0.2, 0.3])
    samples = [
        {},  # missing
        {"static_cam_extrinsic": {"rotation": rot, "translation": trans}},
        {},  # missing
    ]

    out = stack_static_cam_extrinsic(samples)
    assert out is not None
    cam = out["static_cam_extrinsic"]

    assert cam["rotation"].shape == (3, 3, 3)
    assert torch.equal(cam["rotation"][0], torch.zeros(3, 3, dtype=rot.dtype))
    assert torch.equal(cam["rotation"][1], rot)
    assert torch.equal(cam["rotation"][2], torch.zeros(3, 3, dtype=rot.dtype))

    assert cam["translation"].shape == (3, 3)
    assert torch.equal(cam["translation"][0], torch.zeros(3, dtype=trans.dtype))
    assert torch.equal(cam["translation"][1], trans)
    assert torch.equal(cam["translation"][2], torch.zeros(3, dtype=trans.dtype))

    assert torch.equal(
        out["static_cam_extrinsic_mask"], torch.tensor([False, True, False])
    )
