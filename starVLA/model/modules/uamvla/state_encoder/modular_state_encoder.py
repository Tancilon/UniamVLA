"""ModularStateEncoder — composes per-embodiment limb encoders.

Reads embodiment config from EmbodimentRegistry, instantiates the
appropriate limb encoders, and forwards canonical_state dict through
them in registry-defined order.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from starVLA.model.modules.uamvla.state_encoder.limb_encoders import (
    ArmEncoder,
    GripperEncoder,
    EEPoseEncoder,
    JointEncoder,
)
from starVLA.model.modules.uamvla.data.embodiment_registry import get_embodiment_config


class ModularStateEncoder(nn.Module):
    """Top-level state encoder. One instance per embodiment in Phase 1.

    Phase 2 multi-embodiment: hoist limb encoders out and share weights
    across embodiments.
    """
    def __init__(self, embodiment: str, hidden_dim: int):
        super().__init__()
        self.embodiment = embodiment
        self.hidden_dim = hidden_dim

        config = get_embodiment_config(embodiment)
        # Shallow copy to prevent in-place mutation from corrupting the shared
        # registry list (see embodiment_registry.py NOTE).
        self.limb_specs = list(config["limb_config"])

        self.limbs = nn.ModuleList()
        for spec in self.limb_specs:
            limb = self._make_limb(spec["type"], hidden_dim, spec.get("args", {}))
            self.limbs.append(limb)

    @staticmethod
    def _make_limb(limb_type: str, hidden_dim: int, args: dict) -> nn.Module:
        if limb_type == "arm":
            n_joints = args.get("n_joints", 7)
            return ArmEncoder(hidden_dim=hidden_dim, n_joints=n_joints)
        elif limb_type == "gripper":
            return GripperEncoder(hidden_dim=hidden_dim)
        elif limb_type == "ee_pose":
            return EEPoseEncoder(hidden_dim=hidden_dim)
        elif limb_type == "joint":
            n_joints = args.get("n_joints", 7)
            return JointEncoder(hidden_dim=hidden_dim, n_joints=n_joints)
        # Phase 3: BaseEncoder for mobile manipulator
        else:
            raise ValueError(f"Unknown limb_type: {limb_type}")

    def forward(self, canonical_state: dict) -> torch.Tensor:
        """canonical_state must contain all keys referenced by limb_specs.

        Returns: (B, total_tokens, hidden_dim) — concatenated in registry order.
        """
        tokens = []
        for limb, spec in zip(self.limbs, self.limb_specs):
            limb_id = spec["id"]
            limb_type = spec["type"]

            if limb_id not in canonical_state:
                raise KeyError(
                    f"canonical_state missing limb '{limb_id}'. "
                    f"Keys: {list(canonical_state.keys())}"
                )
            inputs = canonical_state[limb_id]

            if limb_type == "arm":
                # ArmEncoder needs ee_pose + joint_pos kwargs
                tokens.append(limb(ee_pose=inputs["ee_pose"], joint_pos=inputs["joint_pos"]))
            elif limb_type == "gripper":
                tokens.append(limb(inputs))  # (B, 1) directly
            elif limb_type == "ee_pose":
                tokens.append(limb(inputs["ee_pose"] if isinstance(inputs, dict) else inputs))
            elif limb_type == "joint":
                tokens.append(limb(inputs["joint_pos"] if isinstance(inputs, dict) else inputs))
            # MAINTENANCE: this dispatch must stay in sync with _make_limb above.
            # Adding a new limb_type requires updating BOTH methods. Phase 3 plan:
            # refactor so each encoder owns its own forward_from_canonical(inputs).
            else:
                raise ValueError(f"Unhandled limb_type in forward: {limb_type}")

        return torch.cat(tokens, dim=1)
