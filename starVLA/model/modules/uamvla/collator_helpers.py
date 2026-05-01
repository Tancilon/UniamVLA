"""Pure-tensor stacking helpers for UamVLA per-sample → batch collation.

These helpers were extracted from the source UamVLA ``RobotDataCollator.__call__``
so the framework's ``UamVLA.forward`` can build a batch dict from a list of
per-sample example dicts. They are pure tensor ops with no ``uamvla.*``
dependencies — testable in isolation.

Public API:
    * ``stack_canonical(state_list)`` — stack a list of canonical_state dicts
      along batch dim 0 (supports one level of nesting).
    * ``stack_optional_tensor_fields(samples, field_names)`` — stack each
      optional tensor field with batch-zero-padding and produce a parallel
      ``f"{field}_mask"`` boolean tensor flagging real slots.
    * ``stack_pose_gt(samples)`` — stack ``pose_gt={"rotation","translation"}``
      across samples or return ``None`` if no sample provides it.
    * ``stack_static_cam_extrinsic(samples)`` — same shape as pose_gt for the
      ``static_cam_extrinsic`` field.

All four functions handle the missing-field case gracefully.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch


def stack_canonical(state_list: list[dict]) -> dict:
    """Stack list of canonical_state dicts along batch dim 0.

    Phase 1 contract: at most ONE level of nesting.
    Supported shapes:
        {"limb_id": tensor}                          # flat
        {"limb_id": {"key": tensor, "key2": tensor}} # nested (1 level)
    Each leaf is wrapped in ``torch.as_tensor`` so callers may pass numpy
    arrays (e.g. WebSocket-deserialized canonical_state from the eval
    client) without manual conversion.

    Deeper nesting (e.g., {"limb": {"sub": {"key": tensor}}}) is NOT
    supported and will produce TypeError from torch.stack on dicts.
    Update this helper if Phase 2 multi-finger gripper / per-joint
    structures need deeper nesting.
    """
    template = state_list[0]
    out: dict = {}
    for limb_id, val in template.items():
        if isinstance(val, dict):
            out[limb_id] = {
                k: torch.stack(
                    [torch.as_tensor(s[limb_id][k]) for s in state_list], dim=0
                )
                for k in val.keys()
            }
        else:
            out[limb_id] = torch.stack(
                [torch.as_tensor(s[limb_id]) for s in state_list], dim=0
            )
    return out


def stack_optional_tensor_fields(
    samples: Sequence[dict], field_names: Iterable[str]
) -> dict:
    """Stack optional per-sample tensor fields into batch tensors with masks.

    For each ``field`` in ``field_names``, builds two outputs (only if at least
    one sample provides the field):

        f"{field}":      zero-padded tensor of shape ``(B,) + template.shape``,
                         with sample-i's tensor placed at row ``i`` if present,
                         else zeros.
        f"{field}_mask": bool tensor of shape ``(B,)``, ``True`` where the
                         field was present.

    The dtype is taken from the first present sample's tensor (the
    "template"). Fields where no sample provides the value are skipped
    (omitted from the returned dict) — callers should treat absence of a
    ``f"{field}_mask"`` key as "no samples have this field at all".

    Args:
        samples: list of per-sample dicts.
        field_names: iterable of field names to consider.

    Returns:
        dict containing zero-or-more pairs of ``f"{field}"`` and
        ``f"{field}_mask"`` entries. Missing fields are simply absent from
        the returned dict.
    """
    out: dict = {}
    B = len(samples)
    for field in field_names:
        present_samples = [s for s in samples if field in s]
        if not present_samples:
            continue
        template = present_samples[0][field]
        padded = torch.zeros(
            (B,) + tuple(template.shape),
            dtype=template.dtype,
        )
        for i, s in enumerate(samples):
            if field in s:
                padded[i] = s[field]
        out[field] = padded
        out[f"{field}_mask"] = torch.tensor(
            [field in s for s in samples], dtype=torch.bool,
        )
    return out


def stack_pose_gt(samples: Sequence[dict]) -> Optional[dict]:
    """Stack ``pose_gt={"rotation","translation"}`` across samples.

    Returns ``None`` if no sample has ``pose_gt``.
    Otherwise returns:
        {
            "pose_gt": {
                "rotation":    Tensor of shape (B, ...) (zeros for missing),
                "translation": Tensor of shape (B, 3),
            },
            "pose_mask": BoolTensor of shape (B,) — True where pose_gt was present.
        }

    Missing slots use a rotation tensor of the same shape as the first
    present sample's rotation tensor (zero-filled), and a fixed translation
    of ``torch.zeros(3)``.
    """
    if not any("pose_gt" in s for s in samples):
        return None

    # Find first present rotation/translation as templates. Both fillers use
    # zeros_like(template) so dtype AND device follow the present samples —
    # critical when input tensors are on GPU (a literal `torch.zeros(3)` would
    # default to CPU, causing torch.stack to crash with cross-device error).
    present = next(s for s in samples if "pose_gt" in s)
    rot_template = present["pose_gt"]["rotation"]
    trans_template = present["pose_gt"]["translation"]

    rotations = []
    translations = []
    mask = []
    for s in samples:
        if "pose_gt" in s:
            rotations.append(s["pose_gt"]["rotation"])
            translations.append(s["pose_gt"]["translation"])
            mask.append(True)
        else:
            rotations.append(torch.zeros_like(rot_template))
            translations.append(torch.zeros_like(trans_template))
            mask.append(False)

    return {
        "pose_gt": {
            "rotation": torch.stack(rotations),
            "translation": torch.stack(translations),
        },
        "pose_mask": torch.tensor(mask, dtype=torch.bool),
    }


def stack_static_cam_extrinsic(samples: Sequence[dict]) -> Optional[dict]:
    """Stack ``static_cam_extrinsic={"rotation","translation"}`` across samples.

    Returns ``None`` if no sample has ``static_cam_extrinsic``.
    Otherwise returns:
        {
            "static_cam_extrinsic": {
                "rotation":    Tensor of shape (B, 3, 3) (zeros for missing),
                "translation": Tensor of shape (B, 3) (zeros for missing),
            },
            "static_cam_extrinsic_mask": BoolTensor of shape (B,) — True where present.
        }

    Missing slots use ``torch.zeros(3, 3, dtype=...)`` and
    ``torch.zeros(3, dtype=...)`` respectively, with dtypes copied from the
    first present sample.
    """
    if not any("static_cam_extrinsic" in s for s in samples):
        return None

    present = next(s for s in samples if "static_cam_extrinsic" in s)
    rot_dtype = present["static_cam_extrinsic"]["rotation"].dtype
    trans_dtype = present["static_cam_extrinsic"]["translation"].dtype

    rotations = []
    translations = []
    mask = []
    for s in samples:
        if "static_cam_extrinsic" in s:
            rotations.append(s["static_cam_extrinsic"]["rotation"])
            translations.append(s["static_cam_extrinsic"]["translation"])
            mask.append(True)
        else:
            rotations.append(torch.zeros(3, 3, dtype=rot_dtype))
            translations.append(torch.zeros(3, dtype=trans_dtype))
            mask.append(False)

    return {
        "static_cam_extrinsic": {
            "rotation": torch.stack(rotations),
            "translation": torch.stack(translations),
        },
        "static_cam_extrinsic_mask": torch.tensor(mask, dtype=torch.bool),
    }
