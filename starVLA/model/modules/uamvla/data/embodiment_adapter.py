"""Embodiment adapters: convert raw obs from each benchmark to canonical state.

Canonical state schema (Phase 1):
    {
        "arm_0":  {"ee_pose": Tensor(9,), "joint_pos": Tensor(7,)},
        "gripper_0": Tensor(1,),
    }
    where ee_pose = [xyz(3) + 6D_rotation(6)] (Zhou et al. 2019 representation),
    joint_pos is in radians, gripper is normalized to [0, 1].

Each adapter is responsible for:
  1. Reading required fields from raw obs dict
  2. Converting rotations to 6D
  3. Normalizing gripper to [0, 1]
  4. Fail-fast on missing fields or non-finite values

Adding a new embodiment:
  1. Subclass EmbodimentAdapter, set embodiment_name and REQUIRED_FIELDS
  2. Implement to_canonical(raw) returning the canonical dict
  3. Register in uamvla/data/embodiment_registry.py
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import torch

from starVLA.utils.rotation import quat_to_6d, euler_to_6d, axis_angle_to_6d


class EmbodimentAdapter(ABC):
    """Abstract base — subclasses must define embodiment_name + to_canonical."""
    embodiment_name: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls) and "embodiment_name" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must define a class-level 'embodiment_name' attribute. "
                f"See LiberoAdapter / CalvinAdapter for examples."
            )

    @abstractmethod
    def to_canonical(self, raw_obs: dict) -> dict:
        """raw obs dict (from preprocessor JSONL or env) → canonical state dict."""
        ...

    @staticmethod
    def _check_required_fields(raw: dict, required: tuple, adapter_name: str):
        for f in required:
            if f not in raw:
                raise KeyError(
                    f"{adapter_name}: missing required field '{f}'. "
                    f"Available keys: {sorted(raw.keys())}"
                )

    @staticmethod
    def _check_finite(arr: np.ndarray, field_name: str, adapter_name: str):
        if not np.isfinite(arr).all():
            raise ValueError(
                f"{adapter_name}: Non-finite values in '{field_name}': {arr}"
            )


class LiberoAdapter(EmbodimentAdapter):
    embodiment_name = "franka_libero"
    REQUIRED_FIELDS = ("ee_pos", "ee_axis_angle", "joint_pos", "gripper_qpos")
    GRIPPER_MAX_WIDTH = 0.04  # Franka parallel gripper max width (meters)

    def to_canonical(self, raw_obs: dict) -> dict:
        self._check_required_fields(raw_obs, self.REQUIRED_FIELDS, "LiberoAdapter")

        ee_pos = np.asarray(raw_obs["ee_pos"], dtype=np.float32)
        ee_axis_angle = np.asarray(raw_obs["ee_axis_angle"], dtype=np.float32)
        joint_pos = np.asarray(raw_obs["joint_pos"], dtype=np.float32)
        gripper_qpos = np.asarray(raw_obs["gripper_qpos"], dtype=np.float32)

        for arr, name in [
            (ee_pos, "ee_pos"), (ee_axis_angle, "ee_axis_angle"),
            (joint_pos, "joint_pos"), (gripper_qpos, "gripper_qpos"),
        ]:
            self._check_finite(arr, name, "LiberoAdapter")

        ee_pose_9d = np.concatenate([ee_pos, axis_angle_to_6d(ee_axis_angle)])
        # Use first finger only: in Franka parallel-jaw both fingers are mechanically
        # coupled and mirror each other (qpos[0] == qpos[1] in LIBERO). For asymmetric
        # grippers, use (gripper_qpos[0] + gripper_qpos[1]) / 2 instead.
        gripper_norm = float(gripper_qpos[0]) / self.GRIPPER_MAX_WIDTH

        return {
            "arm_0": {
                "ee_pose":   torch.as_tensor(ee_pose_9d, dtype=torch.float32),
                "joint_pos": torch.as_tensor(joint_pos, dtype=torch.float32),
            },
            "gripper_0": torch.as_tensor([gripper_norm], dtype=torch.float32),
        }


class CalvinAdapter(EmbodimentAdapter):
    embodiment_name = "franka_calvin"
    REQUIRED_FIELDS = ("robot_obs",)
    GRIPPER_MAX_WIDTH = 0.077  # CALVIN gripper max width (meters)

    # CALVIN robot_obs (15-dim) layout:
    #   [0:3]   tcp position (xyz)
    #   [3:6]   tcp orientation (euler XYZ, intrinsic)
    #   [6]     gripper width
    #   [7:14]  joint positions (7-DoF Franka)
    #   [14]    gripper action command
    EXPECTED_LEN = 15

    def to_canonical(self, raw_obs: dict) -> dict:
        self._check_required_fields(raw_obs, self.REQUIRED_FIELDS, "CalvinAdapter")

        robot_obs = np.asarray(raw_obs["robot_obs"], dtype=np.float32)
        if robot_obs.shape != (self.EXPECTED_LEN,):
            raise ValueError(
                f"CalvinAdapter: robot_obs must have shape ({self.EXPECTED_LEN},), "
                f"got {robot_obs.shape}"
            )
        self._check_finite(robot_obs, "robot_obs", "CalvinAdapter")

        tcp_pos = robot_obs[0:3]
        euler = robot_obs[3:6]
        gripper_width = float(robot_obs[6])
        joint_pos = robot_obs[7:14]

        ee_pose_9d = np.concatenate([tcp_pos, euler_to_6d(euler)])
        gripper_norm = gripper_width / self.GRIPPER_MAX_WIDTH

        return {
            "arm_0": {
                "ee_pose":   torch.as_tensor(ee_pose_9d, dtype=torch.float32),
                "joint_pos": torch.as_tensor(joint_pos, dtype=torch.float32),
            },
            "gripper_0": torch.as_tensor([gripper_norm], dtype=torch.float32),
        }
