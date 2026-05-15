# examples/calvin/train_files/data_registry/data_config.py
"""CALVIN ABCD_D DataConfig for UamVLAOFT framework.

Key design choices (per design spec §4.4):
- video uses single ModalityConfig with delta_indices=[0, H-1] (dual-frame fetch);
  framework slices [0] (observation) and [1] (future) at unpack time.
- state packs 33 dims: 15 robot_obs + 9 pose + 9 cam_extrinsic.
  Only robot_obs is normalized; pose / cam_extrinsic pass through raw.
"""
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class UamVLACalvinDataConfig:
    video_keys = ["video.primary_image", "video.wrist_image"]
    state_keys = [
        "state.robot_obs",
        "state.target_pose_rot6d",
        "state.target_pose_trans",
        "state.static_cam_rot6d",
        "state.static_cam_trans",
    ]
    action_keys = [
        "action.x", "action.y", "action.z",
        "action.roll", "action.pitch", "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]

    action_horizon = 5  # must match framework's action_horizon
    action_indices = list(range(action_horizon))

    # Slice indices for unpack in framework (per spec §4.1 aux_state_slice)
    aux_state_slice = {
        "target_pose_rot6d": (15, 21),
        "target_pose_trans": (21, 24),
        "static_cam_rot6d": (24, 30),
        "static_cam_trans": (30, 33),
    }

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=[0, self.action_horizon - 1],  # = [0, 4]
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(delta_indices=[0], modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=[0], modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            # action: min_max normalize all 7 dims except gripper
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={k: "min_max" for k in self.action_keys[:-1]},
            ),
            # state: ToTensor all 5 fields, but only robot_obs gets mean_std
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=["state.robot_obs"],            # NARROWED apply_to
                normalization_modes={"state.robot_obs": "mean_std"},
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "uamvla_calvin_franka": UamVLACalvinDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "uamvla_calvin_franka": EmbodimentTag.FRANKA,
}

DATASET_NAMED_MIXTURES = {
    # Full preprocessed dataset (datasets/calvin2uam/lerobot_calvin_abcd).
    # Used for baseline B training + downstream eval.
    "uamvla_calvin_abcd": [
        ("lerobot_calvin_abcd", 1.0, "uamvla_calvin_franka"),
    ],
    # D-only preprocessed dataset (datasets/calvin2uam/lerobot_calvin_d).
    "uamvla_calvin_d": [
        ("lerobot_calvin_d", 1.0, "uamvla_calvin_franka"),
    ],
    # Smoke alias kept for the dataloader / framework smokes that ship in
    # tests/. yaml's data_root_dir is playground/Datasets when using this
    # mixture; change in test fixture if you swap to a different root.
    "uamvla_calvin_abcd_smoke": [
        ("UAMVLA_LEROBOT_CALVIN_ABCD_SMOKE", 1.0, "uamvla_calvin_franka"),
    ],
}
