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


class UamVLACalvinH8DataConfig(UamVLACalvinDataConfig):
    """CALVIN UAMVLA config with 8-step action chunks for GR00T experiments."""

    action_horizon = 8
    action_indices = list(range(action_horizon))


class CalvinABCLeRobotV21H8DataConfig:
    """HF LeRobot v2.1 CALVIN ABC-D config with 8-step action chunks."""

    video_keys = ["video.primary_image", "video.wrist_image"]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={k: "min_max" for k in self.action_keys[:-1]},
            ),
        ])


class CalvinDiTDataConfig(CalvinABCLeRobotV21H8DataConfig):
    """Configured sparse historical pairs and current images for action diffusion."""

    def modality_config(self, *, num_history_frames, history_interval):
        if any(type(value) is not int or value <= 0 for value in (num_history_frames, history_interval)):
            raise ValueError("num_history_frames and history_interval must be positive integers")
        config = super().modality_config()
        config["video"] = ModalityConfig(
            delta_indices=list(range(-num_history_frames * history_interval, 1, history_interval)),
            modality_keys=self.video_keys,
        )
        config["language"] = ModalityConfig(delta_indices=[0], modality_keys=self.language_keys)
        return config


class CalvinDT_K10DataConfig(CalvinABCLeRobotV21H8DataConfig):
    """Seer-style DT config for Calvin finetune (K=10, Seer sequence_length=10).

    Dense per-timestep prediction aligned with Seer (window_size=13):
      observation_indices = [-9,...,0, 1, 2, 3]  — 13 frames total
        positions 0..8   : delta -9..-1  (K-1 history frames)
        position  9      : delta  0      (current frame)
        positions 10..12 : delta  1..3   (future frames needed for dense targets)
      action_indices     = [-9,..., 0, 1, 2]  — 12 delta indices for dense windows
      atten_goal = 3     : supervise only K-atten_goal=7 earliest timesteps

    _pack_sample splits these into:
        sample["image"]              — current frame (delta=0)
        sample["image_history"]      — list[K-1=9] of list[num_views] PIL
        sample["future_rgb"]         — list[num_views] Tensor(3,H,W) at delta=3 (compat)
        sample["future_rgb_history"] — list[K=10] of list[num_views] Tensor (dense)
        sample["action_history"]     — ndarray (K, action_horizon, action_dim) (dense)
    """
    history_frames = 10          # K = sequence_length
    future_offset  = 3           # Seer future_steps = 3
    action_horizon = 3           # Seer action_pred_steps = 3
    atten_goal     = 3           # supervise K - atten_goal = 7 timesteps

    _history_k_minus_1 = history_frames - 1  # = 9

    # 13 frames: history (9) + current (1) + future (3) — window_size = K + future_steps
    observation_indices = list(range(-_history_k_minus_1, future_offset + 1))
    # [-9, -8, ..., 0, 1, 2, 3]

    # 12 action delta indices covering all K dense windows
    action_indices = list(range(-_history_k_minus_1, action_horizon))
    # [-9, -8, ..., 0, 1, 2]

    def modality_config(self):
        state_history_indices = list(range(-self._history_k_minus_1, 1))
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(delta_indices=state_history_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=[0], modality_keys=self.language_keys),
        }


class CalvinDT_K14DataConfig(CalvinDT_K10DataConfig):
    """Seer-style DT config for Calvin pretrain (K=14, window_size=17)."""
    history_frames     = 14
    _history_k_minus_1 = history_frames - 1                    # = 13
    observation_indices = list(range(-_history_k_minus_1, CalvinDT_K10DataConfig.future_offset + 1))
    # [-13, ..., 0, 1, 2, 3]  — 17 frames
    action_indices = list(range(-_history_k_minus_1, CalvinDT_K10DataConfig.action_horizon))
    # [-13, ..., 0, 1, 2]  — 16 delta indices


class CalvinDT_K10_H8DataConfig(CalvinDT_K10DataConfig):
    """DT config for Calvin finetune with K=10 history, action_horizon=8.

    For UamGR00T_DT: K=10 history frames, 8-step action chunk,
    future_rgb at t+8 (action_horizon steps ahead).

    Window: history(9) + current(1) + future(8) = 18 frames total.
    """
    action_horizon = 8
    future_offset  = 8           # future_rgb = t + action_horizon

    _h = CalvinDT_K10DataConfig._history_k_minus_1  # = 9

    # 18 frames: [-9,...,0,...,8]
    observation_indices = list(range(-_h, future_offset + 1))

    # 17 delta indices: [-9,...,0,...,7]
    action_indices = list(range(-_h, action_horizon))


ROBOT_TYPE_CONFIG_MAP = {
    "calvin_dit": CalvinDiTDataConfig(),
    "uamvla_calvin_franka": UamVLACalvinDataConfig(),
    "uamvla_calvin_franka_h8": UamVLACalvinH8DataConfig(),
    "calvin_abc_d_franka_h8": CalvinABCLeRobotV21H8DataConfig(),
    "calvin_dt_k10": CalvinDT_K10DataConfig(),
    "calvin_dt_k14": CalvinDT_K14DataConfig(),
    "calvin_dt_k10_h8": CalvinDT_K10_H8DataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "calvin_dit": EmbodimentTag.FRANKA,
    "uamvla_calvin_franka": EmbodimentTag.FRANKA,
    "uamvla_calvin_franka_h8": EmbodimentTag.FRANKA,
    "calvin_abc_d_franka_h8": EmbodimentTag.FRANKA,
    "calvin_dt_k10": EmbodimentTag.FRANKA,
    "calvin_dt_k14": EmbodimentTag.FRANKA,
    "calvin_dt_k10_h8": EmbodimentTag.FRANKA,
}

DATASET_NAMED_MIXTURES = {
    "calvin_abc_dit": [
        ("task_ABC_D_scene_A_lerobot_lt", 1.0, "calvin_dit"),
        ("task_ABC_D_scene_B_lerobot_lt", 1.0, "calvin_dit"),
        ("task_ABC_D_scene_C_lerobot_lt", 1.0, "calvin_dit"),
    ],
    # Three-scene CALVIN ABC LeRobot v2.1 datasets under datasets/.
    "calvin_abc_lerobot_h8": [
        ("task_ABC_D_scene_A_lerobot", 1.0, "calvin_abc_d_franka_h8"),
        ("task_ABC_D_scene_B_lerobot", 1.0, "calvin_abc_d_franka_h8"),
        ("task_ABC_D_scene_C_lerobot", 1.0, "calvin_abc_d_franka_h8"),
    ],
    # Hugging Face CollisionCode/calvin_abc_d_lerobot_v2.1 dataset.
    "calvin_abc_d": [
        ("calvin_abc_d_lerobot_v2.1", 1.0, "calvin_abc_d_franka_h8"),
    ],
    "calvin_abc_d_h8": [
        ("calvin_abc_d_lerobot_v2.1", 1.0, "calvin_abc_d_franka_h8"),
    ],
    # Full preprocessed dataset (datasets/calvin2uam/lerobot_calvin_abcd).
    # Used for baseline B training + downstream eval.
    "uamvla_calvin_abcd": [
        ("lerobot_calvin_abcd", 1.0, "uamvla_calvin_franka"),
    ],
    # D-only preprocessed dataset (datasets/calvin2uam/lerobot_calvin_d).
    "uamvla_calvin_d": [
        ("lerobot_calvin_d", 1.0, "uamvla_calvin_franka"),
    ],
    "uamvla_calvin_abcd_h8": [
        ("lerobot_calvin_abcd", 1.0, "uamvla_calvin_franka_h8"),
    ],
    "uamvla_calvin_abc_h8": [
        ("lerobot_calvin_abc", 1.0, "uamvla_calvin_franka_h8"),
    ],
    "uamvla_calvin_d_h8": [
        ("lerobot_calvin_d", 1.0, "uamvla_calvin_franka_h8"),
    ],
    # Smoke alias kept for the dataloader / framework smokes that ship in
    # tests/. yaml's data_root_dir is playground/Datasets when using this
    # mixture; change in test fixture if you swap to a different root.
    "uamvla_calvin_abcd_smoke": [
        ("UAMVLA_LEROBOT_CALVIN_ABCD_SMOKE", 1.0, "uamvla_calvin_franka"),
    ],

    # -----------------------------------------------------------------------
    # Seer-style DT mixes (history K frames + future_rgb at t+3)
    # -----------------------------------------------------------------------
    # Finetune mix: CALVIN ABC language-annotated data, K=10, action_horizon=3.
    # Dataset names reuse the same LeRobot v2.1 data as calvin_abc_lerobot_h8;
    # robot_type "calvin_dt_k10" selects CalvinDT_K10DataConfig which sets the
    # correct delta_indices and signals _pack_sample via num_history_frames/future_offset.
    # Points to the _lt datasets (scene_A/B/C_lerobot_lt) which already exist
    # on disk. CalvinDT_K10DataConfig only reads the 8D state + 7D action +
    # 2 video views — the extra _lt sidecar columns are ignored.
    "calvin_abc_dt_k10": [
        ("task_ABC_D_scene_A_lerobot_lt", 1.0, "calvin_dt_k10"),
        ("task_ABC_D_scene_B_lerobot_lt", 1.0, "calvin_dt_k10"),
        ("task_ABC_D_scene_C_lerobot_lt", 1.0, "calvin_dt_k10"),
    ],

    # K=10, action_horizon=8, future_offset=8 — for UamGR00T_DT (new architecture)
    "calvin_abc_dt_k10_h8": [
        ("task_ABC_D_scene_A_lerobot_lt", 1.0, "calvin_dt_k10_h8"),
        ("task_ABC_D_scene_B_lerobot_lt", 1.0, "calvin_dt_k10_h8"),
        ("task_ABC_D_scene_C_lerobot_lt", 1.0, "calvin_dt_k10_h8"),
    ],

    # K=14 counterpart used when finetuning the K=14 pretrained architecture
    # without changing its temporal parameter shapes.
    "calvin_abc_dt_k14": [
        ("task_ABC_D_scene_A_lerobot_lt", 1.0, "calvin_dt_k14"),
        ("task_ABC_D_scene_B_lerobot_lt", 1.0, "calvin_dt_k14"),
        ("task_ABC_D_scene_C_lerobot_lt", 1.0, "calvin_dt_k14"),
    ],

    # Pretrain mix: CALVIN play data (no language annotations), K=14.
    # Expects datasets/task_ABC_D_play_lerobot (or similar play split).
    # Language field will be empty string; trainer should set loss_scale.vla=0
    # to focus on future reconstruction only during pretraining.
    "calvin_play_dt_k14": [
        ("task_ABC_D_play_lerobot", 1.0, "calvin_dt_k14"),
    ],
}
