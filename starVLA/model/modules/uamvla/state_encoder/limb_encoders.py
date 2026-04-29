"""Limb encoders — atomic semantic units for state encoding.

Each encoder converts a fixed-dim canonical input to one or more hidden_dim
embeddings. Composition is done in ModularStateEncoder by chaining these
according to the embodiment's limb_config.

Phase 1 limbs:
  - EEPoseEncoder:  9-dim (xyz + 6D rotation) -> 1 token
  - JointEncoder:   N-dim (joint positions) -> 1 token
  - GripperEncoder: 1-dim (normalized [0, 1]) -> 1 token
  - ArmEncoder:     composes EEPose + Joint -> 2 tokens
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EEPoseEncoder(nn.Module):
    """End-effector pose encoder.

    Input:  (B, 9)  - [xyz(3) + 6D_rotation(6)]
    Output: (B, 1, hidden_dim)
    """
    INPUT_DIM = 9

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, ee_pose: torch.Tensor) -> torch.Tensor:
        if ee_pose.shape[-1] != self.INPUT_DIM:
            raise ValueError(
                f"EEPoseEncoder expects last dim {self.INPUT_DIM}, "
                f"got {ee_pose.shape[-1]}"
            )
        return self.mlp(ee_pose).unsqueeze(1)


class JointEncoder(nn.Module):
    """Joint position encoder.

    Input:  (B, n_joints)
    Output: (B, 1, hidden_dim)
    """
    def __init__(self, hidden_dim: int, n_joints: int = 7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_joints = n_joints
        self.mlp = nn.Sequential(
            nn.Linear(n_joints, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, joint_pos: torch.Tensor) -> torch.Tensor:
        if joint_pos.shape[-1] != self.n_joints:
            raise ValueError(
                f"JointEncoder expects last dim {self.n_joints}, "
                f"got {joint_pos.shape[-1]}"
            )
        return self.mlp(joint_pos).unsqueeze(1)


class GripperEncoder(nn.Module):
    """Gripper state encoder.

    Input:  (B, 1) - normalized to [0, 1] (works for both binary and continuous)
    Output: (B, 1, hidden_dim)
    """
    INPUT_DIM = 1

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, gripper: torch.Tensor) -> torch.Tensor:
        if gripper.ndim < 2 or gripper.shape[-1] != self.INPUT_DIM:
            raise ValueError(
                f"GripperEncoder expects shape (B, 1), got {tuple(gripper.shape)}"
            )
        return self.mlp(gripper).unsqueeze(1)


class ArmEncoder(nn.Module):
    """Composite arm encoder = EEPose + Joint.

    Input:  ee_pose (B, 9), joint_pos (B, n_joints)
    Output: (B, 2, hidden_dim) - order: [ee_token, joint_token]
    """
    def __init__(self, hidden_dim: int, n_joints: int = 7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_joints = n_joints
        self.ee_encoder = EEPoseEncoder(hidden_dim)
        self.joint_encoder = JointEncoder(hidden_dim, n_joints)

    def forward(self, ee_pose: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
        ee_token = self.ee_encoder(ee_pose)         # (B, 1, h)
        joint_token = self.joint_encoder(joint_pos)  # (B, 1, h)
        # Order is LOAD-BEARING: must match EmbodimentRegistry special_tokens
        # [<|state_ee_*|>, <|state_joint_*|>]. Do not reorder.
        return torch.cat([ee_token, joint_token], dim=1)  # (B, 2, h)
