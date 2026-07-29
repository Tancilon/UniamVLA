"""RobotWin benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig — Agilex (RobotWin, action_indices=16)
# ---------------------------------------------------------------------------
class AgilexDataConfig:
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.left_joints", "state.right_joints", "state.left_gripper", "state.right_gripper"]
    action_keys = ["action.left_joints", "action.right_joints", "action.left_gripper", "action.right_gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max", "state.right_joints": "min_max",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max", "action.right_joints": "min_max",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — Agilex 50 (action_indices=50)
# ---------------------------------------------------------------------------
class AgilexData50Config(AgilexDataConfig):
    action_indices = list(range(50))

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "state.left_joints": "min_max", "state.right_joints": "min_max",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "action.left_joints": "min_max", "action.right_joints": "min_max",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — ARX X5
# ---------------------------------------------------------------------------
class ArxX5DataConfig:
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.left_joints", "state.right_joints", "state.left_gripper", "state.right_gripper"]
    action_keys = ["action.left_joints", "action.right_joints", "action.left_gripper", "action.right_gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max", "state.right_joints": "min_max",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max", "action.right_joints": "min_max",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — RoboTwin DT (K-frame history + future pixel prediction)
#
# Mirrors CalvinDT_K10/K14DataConfig but for Agilex bimanual (14D, 3 views).
# delta_indices encodes K-1 history frames + current frame + 1 future frame:
#   [-(K-1), ..., -1, 0, future_offset]
# _pack_sample reads num_history_frames and future_offset to split into:
#   sample["image"]          — current frame (num_views PIL images)
#   sample["image_history"]  — list[K-1] of list[num_views] PIL images
#   sample["future_rgb"]     — list[num_views] Tensor(3,H,W) in [0,1]
# ---------------------------------------------------------------------------
class RobotwinDT_K10DataConfig(AgilexDataConfig):
    """Seer-style DT config for RoboTwin finetune (K=10 history, future_offset=3, action_horizon=16).

    Shorter history window keeps memory manageable; action_horizon=16 matches
    the existing RoboTwin LT baseline to reuse normalization stats.
    """
    history_frames = 10
    future_offset = 3
    # Keep action_horizon=16 (RoboTwin convention); distinct from Calvin's 3.
    action_indices = list(range(16))

    _history_k_minus_1 = history_frames - 1
    # Combined delta_indices: [-(K-1), ..., -1, 0, future_offset]
    observation_indices = list(range(-_history_k_minus_1, 1)) + [future_offset]
    # e.g. [-9,-8,-7,-6,-5,-4,-3,-2,-1, 0, 3]

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(delta_indices=[0], modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=[0], modality_keys=self.language_keys),
        }


class RobotwinDT_K14DataConfig(RobotwinDT_K10DataConfig):
    """Seer-style DT config for RoboTwin pretrain (K=14 history, future_offset=3).

    Longer history window for the pretrain phase; action_model is frozen so
    action_horizon is unused in the loss but kept for DataConfig consistency.
    """
    history_frames = 14
    _history_k_minus_1 = history_frames - 1
    observation_indices = list(range(-_history_k_minus_1, 1)) + [RobotwinDT_K10DataConfig.future_offset]
    # [-13,-12,...,-1, 0, 3]  — 15 total frames loaded


ROBOT_TYPE_CONFIG_MAP = {
    "robotwin": AgilexDataConfig(),
    "robotwin50": AgilexData50Config(),
    "arx_x5": ArxX5DataConfig(),
    "robotwin_dt_k10": RobotwinDT_K10DataConfig(),
    "robotwin_dt_k14": RobotwinDT_K14DataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Uses NEW_EMBODIMENT fallback (not in base embodiment_tags.py)
}

# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "robotwin_all": [
        ("Clean/adjust_bottle", 1.0, "robotwin"), ("Randomized/adjust_bottle", 1.0, "robotwin"),
        ("Clean/beat_block_hammer", 1.0, "robotwin"), ("Randomized/beat_block_hammer", 1.0, "robotwin"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin"), ("Randomized/blocks_ranking_size", 1.0, "robotwin"),
        ("Clean/click_alarmclock", 1.0, "robotwin"), ("Randomized/click_alarmclock", 1.0, "robotwin"),
        ("Clean/click_bell", 1.0, "robotwin"), ("Randomized/click_bell", 1.0, "robotwin"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin"),
        ("Clean/grab_roller", 1.0, "robotwin"), ("Randomized/grab_roller", 1.0, "robotwin"),
        ("Clean/handover_block", 1.0, "robotwin"), ("Randomized/handover_block", 1.0, "robotwin"),
        ("Clean/handover_mic", 1.0, "robotwin"), ("Randomized/handover_mic", 1.0, "robotwin"),
        ("Clean/hanging_mug", 1.0, "robotwin"), ("Randomized/hanging_mug", 1.0, "robotwin"),
        ("Clean/lift_pot", 1.0, "robotwin"), ("Randomized/lift_pot", 1.0, "robotwin"),
        ("Clean/move_can_pot", 1.0, "robotwin"), ("Randomized/move_can_pot", 1.0, "robotwin"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin"),
        ("Clean/move_playingcard_away", 1.0, "robotwin"), ("Randomized/move_playingcard_away", 1.0, "robotwin"),
        ("Clean/move_stapler_pad", 1.0, "robotwin"), ("Randomized/move_stapler_pad", 1.0, "robotwin"),
        ("Clean/open_laptop", 1.0, "robotwin"), ("Randomized/open_laptop", 1.0, "robotwin"),
        ("Clean/open_microwave", 1.0, "robotwin"), ("Randomized/open_microwave", 1.0, "robotwin"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin"), ("Randomized/pick_dual_bottles", 1.0, "robotwin"),
        ("Clean/place_a2b_left", 1.0, "robotwin"), ("Randomized/place_a2b_left", 1.0, "robotwin"),
        ("Clean/place_a2b_right", 1.0, "robotwin"), ("Randomized/place_a2b_right", 1.0, "robotwin"),
        ("Clean/place_bread_basket", 1.0, "robotwin"), ("Randomized/place_bread_basket", 1.0, "robotwin"),
        ("Clean/place_bread_skillet", 1.0, "robotwin"), ("Randomized/place_bread_skillet", 1.0, "robotwin"),
        ("Clean/place_burger_fries", 1.0, "robotwin"), ("Randomized/place_burger_fries", 1.0, "robotwin"),
        ("Clean/place_can_basket", 1.0, "robotwin"), ("Randomized/place_can_basket", 1.0, "robotwin"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin"),
        ("Clean/place_container_plate", 1.0, "robotwin"), ("Randomized/place_container_plate", 1.0, "robotwin"),
        ("Clean/place_dual_shoes", 1.0, "robotwin"), ("Randomized/place_dual_shoes", 1.0, "robotwin"),
        ("Clean/place_empty_cup", 1.0, "robotwin"), ("Randomized/place_empty_cup", 1.0, "robotwin"),
        ("Clean/place_fan", 1.0, "robotwin"), ("Randomized/place_fan", 1.0, "robotwin"),
        ("Clean/place_mouse_pad", 1.0, "robotwin"), ("Randomized/place_mouse_pad", 1.0, "robotwin"),
        ("Clean/place_object_basket", 1.0, "robotwin"), ("Randomized/place_object_basket", 1.0, "robotwin"),
        ("Clean/place_object_scale", 1.0, "robotwin"), ("Randomized/place_object_scale", 1.0, "robotwin"),
        ("Clean/place_object_stand", 1.0, "robotwin"), ("Randomized/place_object_stand", 1.0, "robotwin"),
        ("Clean/place_phone_stand", 1.0, "robotwin"), ("Randomized/place_phone_stand", 1.0, "robotwin"),
        ("Clean/place_shoe", 1.0, "robotwin"), ("Randomized/place_shoe", 1.0, "robotwin"),
        ("Clean/press_stapler", 1.0, "robotwin"), ("Randomized/press_stapler", 1.0, "robotwin"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin"),
        ("Clean/put_object_cabinet", 1.0, "robotwin"), ("Randomized/put_object_cabinet", 1.0, "robotwin"),
        ("Clean/rotate_qrcode", 1.0, "robotwin"), ("Randomized/rotate_qrcode", 1.0, "robotwin"),
        ("Clean/scan_object", 1.0, "robotwin"), ("Randomized/scan_object", 1.0, "robotwin"),
        ("Clean/shake_bottle", 1.0, "robotwin"), ("Randomized/shake_bottle", 1.0, "robotwin"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin"),
        ("Clean/stack_blocks_three", 1.0, "robotwin"), ("Randomized/stack_blocks_three", 1.0, "robotwin"),
        ("Clean/stack_blocks_two", 1.0, "robotwin"), ("Randomized/stack_blocks_two", 1.0, "robotwin"),
        ("Clean/stack_bowls_three", 1.0, "robotwin"), ("Randomized/stack_bowls_three", 1.0, "robotwin"),
        ("Clean/stack_bowls_two", 1.0, "robotwin"), ("Randomized/stack_bowls_two", 1.0, "robotwin"),
        ("Clean/stamp_seal", 1.0, "robotwin"), ("Randomized/stamp_seal", 1.0, "robotwin"),
        ("Clean/turn_switch", 1.0, "robotwin"), ("Randomized/turn_switch", 1.0, "robotwin"),
    ],
    "robotwin_all_50": [
        ("Clean/adjust_bottle", 1.0, "robotwin50"), ("Randomized/adjust_bottle", 1.0, "robotwin50"),
        ("Clean/beat_block_hammer", 1.0, "robotwin50"), ("Randomized/beat_block_hammer", 1.0, "robotwin50"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin50"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin50"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin50"), ("Randomized/blocks_ranking_size", 1.0, "robotwin50"),
        ("Clean/click_alarmclock", 1.0, "robotwin50"), ("Randomized/click_alarmclock", 1.0, "robotwin50"),
        ("Clean/click_bell", 1.0, "robotwin50"), ("Randomized/click_bell", 1.0, "robotwin50"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin50"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin50"),
        ("Clean/grab_roller", 1.0, "robotwin50"), ("Randomized/grab_roller", 1.0, "robotwin50"),
        ("Clean/handover_block", 1.0, "robotwin50"), ("Randomized/handover_block", 1.0, "robotwin50"),
        ("Clean/handover_mic", 1.0, "robotwin50"), ("Randomized/handover_mic", 1.0, "robotwin50"),
        ("Clean/hanging_mug", 1.0, "robotwin50"), ("Randomized/hanging_mug", 1.0, "robotwin50"),
        ("Clean/lift_pot", 1.0, "robotwin50"), ("Randomized/lift_pot", 1.0, "robotwin50"),
        ("Clean/move_can_pot", 1.0, "robotwin50"), ("Randomized/move_can_pot", 1.0, "robotwin50"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin50"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin50"),
        ("Clean/move_playingcard_away", 1.0, "robotwin50"), ("Randomized/move_playingcard_away", 1.0, "robotwin50"),
        ("Clean/move_stapler_pad", 1.0, "robotwin50"), ("Randomized/move_stapler_pad", 1.0, "robotwin50"),
        ("Clean/open_laptop", 1.0, "robotwin50"), ("Randomized/open_laptop", 1.0, "robotwin50"),
        ("Clean/open_microwave", 1.0, "robotwin50"), ("Randomized/open_microwave", 1.0, "robotwin50"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin50"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin50"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin50"), ("Randomized/pick_dual_bottles", 1.0, "robotwin50"),
        ("Clean/place_a2b_left", 1.0, "robotwin50"), ("Randomized/place_a2b_left", 1.0, "robotwin50"),
        ("Clean/place_a2b_right", 1.0, "robotwin50"), ("Randomized/place_a2b_right", 1.0, "robotwin50"),
        ("Clean/place_bread_basket", 1.0, "robotwin50"), ("Randomized/place_bread_basket", 1.0, "robotwin50"),
        ("Clean/place_bread_skillet", 1.0, "robotwin50"), ("Randomized/place_bread_skillet", 1.0, "robotwin50"),
        ("Clean/place_burger_fries", 1.0, "robotwin50"), ("Randomized/place_burger_fries", 1.0, "robotwin50"),
        ("Clean/place_can_basket", 1.0, "robotwin50"), ("Randomized/place_can_basket", 1.0, "robotwin50"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin50"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin50"),
        ("Clean/place_container_plate", 1.0, "robotwin50"), ("Randomized/place_container_plate", 1.0, "robotwin50"),
        ("Clean/place_dual_shoes", 1.0, "robotwin50"), ("Randomized/place_dual_shoes", 1.0, "robotwin50"),
        ("Clean/place_empty_cup", 1.0, "robotwin50"), ("Randomized/place_empty_cup", 1.0, "robotwin50"),
        ("Clean/place_fan", 1.0, "robotwin50"), ("Randomized/place_fan", 1.0, "robotwin50"),
        ("Clean/place_mouse_pad", 1.0, "robotwin50"), ("Randomized/place_mouse_pad", 1.0, "robotwin50"),
        ("Clean/place_object_basket", 1.0, "robotwin50"), ("Randomized/place_object_basket", 1.0, "robotwin50"),
        ("Clean/place_object_scale", 1.0, "robotwin50"), ("Randomized/place_object_scale", 1.0, "robotwin50"),
        ("Clean/place_object_stand", 1.0, "robotwin50"), ("Randomized/place_object_stand", 1.0, "robotwin50"),
        ("Clean/place_phone_stand", 1.0, "robotwin50"), ("Randomized/place_phone_stand", 1.0, "robotwin50"),
        ("Clean/place_shoe", 1.0, "robotwin50"), ("Randomized/place_shoe", 1.0, "robotwin50"),
        ("Clean/press_stapler", 1.0, "robotwin50"), ("Randomized/press_stapler", 1.0, "robotwin50"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin50"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin50"),
        ("Clean/put_object_cabinet", 1.0, "robotwin50"), ("Randomized/put_object_cabinet", 1.0, "robotwin50"),
        ("Clean/rotate_qrcode", 1.0, "robotwin50"), ("Randomized/rotate_qrcode", 1.0, "robotwin50"),
        ("Clean/scan_object", 1.0, "robotwin50"), ("Randomized/scan_object", 1.0, "robotwin50"),
        ("Clean/shake_bottle", 1.0, "robotwin50"), ("Randomized/shake_bottle", 1.0, "robotwin50"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin50"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin50"),
        ("Clean/stack_blocks_three", 1.0, "robotwin50"), ("Randomized/stack_blocks_three", 1.0, "robotwin50"),
        ("Clean/stack_blocks_two", 1.0, "robotwin50"), ("Randomized/stack_blocks_two", 1.0, "robotwin50"),
        ("Clean/stack_bowls_three", 1.0, "robotwin50"), ("Randomized/stack_bowls_three", 1.0, "robotwin50"),
        ("Clean/stack_bowls_two", 1.0, "robotwin50"), ("Randomized/stack_bowls_two", 1.0, "robotwin50"),
        ("Clean/stamp_seal", 1.0, "robotwin50"), ("Randomized/stamp_seal", 1.0, "robotwin50"),
        ("Clean/turn_switch", 1.0, "robotwin50"), ("Randomized/turn_switch", 1.0, "robotwin50"),
    ],
    "robotwin": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),
        ("click_alarmclock", 1.0, "robotwin"),
        ("click_bell", 1.0, "robotwin"),
        ("dump_bin_bigbin", 1.0, "robotwin"),
        ("grab_roller", 1.0, "robotwin"),
        ("handover_block", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),
        ("hanging_mug", 1.0, "robotwin"),
        ("lift_pot", 1.0, "robotwin"),
        ("move_can_pot", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),
        ("move_playingcard_away", 1.0, "robotwin"),
        ("move_stapler_pad", 1.0, "robotwin"),
        ("open_laptop", 1.0, "robotwin"),
        ("open_microwave", 1.0, "robotwin"),
        ("pick_diverse_bottles", 1.0, "robotwin"),
        ("pick_dual_bottles", 1.0, "robotwin"),
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
        ("place_bread_basket", 1.0, "robotwin"),
        ("place_bread_skillet", 1.0, "robotwin"),
        ("place_burger_fries", 1.0, "robotwin"),
        ("place_can_basket", 1.0, "robotwin"),
        ("place_cans_plasticbox", 1.0, "robotwin"),
        ("place_container_plate", 1.0, "robotwin"),
        ("place_dual_shoes", 1.0, "robotwin"),
        ("place_empty_cup", 1.0, "robotwin"),
        ("place_fan", 1.0, "robotwin"),
        ("place_mouse_pad", 1.0, "robotwin"),
        ("place_object_basket", 1.0, "robotwin"),
        ("place_object_scale", 1.0, "robotwin"),
        ("place_object_stand", 1.0, "robotwin"),
        ("place_phone_stand", 1.0, "robotwin"),
        ("place_shoe", 1.0, "robotwin"),
        ("press_stapler", 1.0, "robotwin"),
        ("put_bottles_dustbin", 1.0, "robotwin"),
        ("put_object_cabinet", 1.0, "robotwin"),
        ("rotate_qrcode", 1.0, "robotwin"),
        ("scan_object", 1.0, "robotwin"),
        ("shake_bottle", 1.0, "robotwin"),
        ("shake_bottle_horizontally", 1.0, "robotwin"),
        ("stack_blocks_three", 1.0, "robotwin"),
        ("stack_blocks_two", 1.0, "robotwin"),
        ("stack_bowls_three", 1.0, "robotwin"),
        ("stack_bowls_two", 1.0, "robotwin"),
        ("stamp_seal", 1.0, "robotwin"),
        ("turn_switch", 1.0, "robotwin"),
    ],
    "robotwin_task1": [("adjust_bottle", 1.0, "robotwin")],
    "robotwin_task2": [("place_a2b_left", 1.0, "robotwin"), ("place_a2b_right", 1.0, "robotwin")],
    "arx_x5": [("arx_x5", 1.0, "arx_x5")],

    # -----------------------------------------------------------------------
    # Seer-style DT mixes (K-frame history + future_rgb at t+3)
    # -----------------------------------------------------------------------
    # Pretrain mix: all RoboTwin tasks (Clean + Randomized), K=14.
    # RoboTwin has no unlabeled play data; using full task data for pretrain
    # is valid — the future reconstruction objective does not require labels.
    # action_model is frozen during pretrain (see YAML freeze_modules).
    "robotwin_all_dt_k14": [
        ("Clean/adjust_bottle", 1.0, "robotwin_dt_k14"), ("Randomized/adjust_bottle", 1.0, "robotwin_dt_k14"),
        ("Clean/beat_block_hammer", 1.0, "robotwin_dt_k14"), ("Randomized/beat_block_hammer", 1.0, "robotwin_dt_k14"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin_dt_k14"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin_dt_k14"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin_dt_k14"), ("Randomized/blocks_ranking_size", 1.0, "robotwin_dt_k14"),
        ("Clean/click_alarmclock", 1.0, "robotwin_dt_k14"), ("Randomized/click_alarmclock", 1.0, "robotwin_dt_k14"),
        ("Clean/click_bell", 1.0, "robotwin_dt_k14"), ("Randomized/click_bell", 1.0, "robotwin_dt_k14"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin_dt_k14"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin_dt_k14"),
        ("Clean/grab_roller", 1.0, "robotwin_dt_k14"), ("Randomized/grab_roller", 1.0, "robotwin_dt_k14"),
        ("Clean/handover_block", 1.0, "robotwin_dt_k14"), ("Randomized/handover_block", 1.0, "robotwin_dt_k14"),
        ("Clean/handover_mic", 1.0, "robotwin_dt_k14"), ("Randomized/handover_mic", 1.0, "robotwin_dt_k14"),
        ("Clean/hanging_mug", 1.0, "robotwin_dt_k14"), ("Randomized/hanging_mug", 1.0, "robotwin_dt_k14"),
        ("Clean/lift_pot", 1.0, "robotwin_dt_k14"), ("Randomized/lift_pot", 1.0, "robotwin_dt_k14"),
        ("Clean/move_can_pot", 1.0, "robotwin_dt_k14"), ("Randomized/move_can_pot", 1.0, "robotwin_dt_k14"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin_dt_k14"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin_dt_k14"),
        ("Clean/move_playingcard_away", 1.0, "robotwin_dt_k14"), ("Randomized/move_playingcard_away", 1.0, "robotwin_dt_k14"),
        ("Clean/move_stapler_pad", 1.0, "robotwin_dt_k14"), ("Randomized/move_stapler_pad", 1.0, "robotwin_dt_k14"),
        ("Clean/open_laptop", 1.0, "robotwin_dt_k14"), ("Randomized/open_laptop", 1.0, "robotwin_dt_k14"),
        ("Clean/open_microwave", 1.0, "robotwin_dt_k14"), ("Randomized/open_microwave", 1.0, "robotwin_dt_k14"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin_dt_k14"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin_dt_k14"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin_dt_k14"), ("Randomized/pick_dual_bottles", 1.0, "robotwin_dt_k14"),
        ("Clean/place_a2b_left", 1.0, "robotwin_dt_k14"), ("Randomized/place_a2b_left", 1.0, "robotwin_dt_k14"),
        ("Clean/place_a2b_right", 1.0, "robotwin_dt_k14"), ("Randomized/place_a2b_right", 1.0, "robotwin_dt_k14"),
        ("Clean/place_bread_basket", 1.0, "robotwin_dt_k14"), ("Randomized/place_bread_basket", 1.0, "robotwin_dt_k14"),
        ("Clean/place_bread_skillet", 1.0, "robotwin_dt_k14"), ("Randomized/place_bread_skillet", 1.0, "robotwin_dt_k14"),
        ("Clean/place_burger_fries", 1.0, "robotwin_dt_k14"), ("Randomized/place_burger_fries", 1.0, "robotwin_dt_k14"),
        ("Clean/place_can_basket", 1.0, "robotwin_dt_k14"), ("Randomized/place_can_basket", 1.0, "robotwin_dt_k14"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin_dt_k14"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin_dt_k14"),
        ("Clean/place_container_plate", 1.0, "robotwin_dt_k14"), ("Randomized/place_container_plate", 1.0, "robotwin_dt_k14"),
        ("Clean/place_dual_shoes", 1.0, "robotwin_dt_k14"), ("Randomized/place_dual_shoes", 1.0, "robotwin_dt_k14"),
        ("Clean/place_empty_cup", 1.0, "robotwin_dt_k14"), ("Randomized/place_empty_cup", 1.0, "robotwin_dt_k14"),
        ("Clean/place_fan", 1.0, "robotwin_dt_k14"), ("Randomized/place_fan", 1.0, "robotwin_dt_k14"),
        ("Clean/place_mouse_pad", 1.0, "robotwin_dt_k14"), ("Randomized/place_mouse_pad", 1.0, "robotwin_dt_k14"),
        ("Clean/place_object_basket", 1.0, "robotwin_dt_k14"), ("Randomized/place_object_basket", 1.0, "robotwin_dt_k14"),
        ("Clean/place_object_scale", 1.0, "robotwin_dt_k14"), ("Randomized/place_object_scale", 1.0, "robotwin_dt_k14"),
        ("Clean/place_object_stand", 1.0, "robotwin_dt_k14"), ("Randomized/place_object_stand", 1.0, "robotwin_dt_k14"),
        ("Clean/place_phone_stand", 1.0, "robotwin_dt_k14"), ("Randomized/place_phone_stand", 1.0, "robotwin_dt_k14"),
        ("Clean/place_shoe", 1.0, "robotwin_dt_k14"), ("Randomized/place_shoe", 1.0, "robotwin_dt_k14"),
        ("Clean/press_stapler", 1.0, "robotwin_dt_k14"), ("Randomized/press_stapler", 1.0, "robotwin_dt_k14"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin_dt_k14"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin_dt_k14"),
        ("Clean/put_object_cabinet", 1.0, "robotwin_dt_k14"), ("Randomized/put_object_cabinet", 1.0, "robotwin_dt_k14"),
        ("Clean/rotate_qrcode", 1.0, "robotwin_dt_k14"), ("Randomized/rotate_qrcode", 1.0, "robotwin_dt_k14"),
        ("Clean/scan_object", 1.0, "robotwin_dt_k14"), ("Randomized/scan_object", 1.0, "robotwin_dt_k14"),
        ("Clean/shake_bottle", 1.0, "robotwin_dt_k14"), ("Randomized/shake_bottle", 1.0, "robotwin_dt_k14"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin_dt_k14"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin_dt_k14"),
        ("Clean/stack_blocks_three", 1.0, "robotwin_dt_k14"), ("Randomized/stack_blocks_three", 1.0, "robotwin_dt_k14"),
        ("Clean/stack_blocks_two", 1.0, "robotwin_dt_k14"), ("Randomized/stack_blocks_two", 1.0, "robotwin_dt_k14"),
        ("Clean/stack_bowls_three", 1.0, "robotwin_dt_k14"), ("Randomized/stack_bowls_three", 1.0, "robotwin_dt_k14"),
        ("Clean/stack_bowls_two", 1.0, "robotwin_dt_k14"), ("Randomized/stack_bowls_two", 1.0, "robotwin_dt_k14"),
        ("Clean/stamp_seal", 1.0, "robotwin_dt_k14"), ("Randomized/stamp_seal", 1.0, "robotwin_dt_k14"),
        ("Clean/turn_switch", 1.0, "robotwin_dt_k14"), ("Randomized/turn_switch", 1.0, "robotwin_dt_k14"),
    ],

    # Finetune mix: same tasks, K=10 (shorter window for efficiency).
    "robotwin_all_dt_k10": [
        ("Clean/adjust_bottle", 1.0, "robotwin_dt_k10"), ("Randomized/adjust_bottle", 1.0, "robotwin_dt_k10"),
        ("Clean/beat_block_hammer", 1.0, "robotwin_dt_k10"), ("Randomized/beat_block_hammer", 1.0, "robotwin_dt_k10"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin_dt_k10"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin_dt_k10"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin_dt_k10"), ("Randomized/blocks_ranking_size", 1.0, "robotwin_dt_k10"),
        ("Clean/click_alarmclock", 1.0, "robotwin_dt_k10"), ("Randomized/click_alarmclock", 1.0, "robotwin_dt_k10"),
        ("Clean/click_bell", 1.0, "robotwin_dt_k10"), ("Randomized/click_bell", 1.0, "robotwin_dt_k10"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin_dt_k10"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin_dt_k10"),
        ("Clean/grab_roller", 1.0, "robotwin_dt_k10"), ("Randomized/grab_roller", 1.0, "robotwin_dt_k10"),
        ("Clean/handover_block", 1.0, "robotwin_dt_k10"), ("Randomized/handover_block", 1.0, "robotwin_dt_k10"),
        ("Clean/handover_mic", 1.0, "robotwin_dt_k10"), ("Randomized/handover_mic", 1.0, "robotwin_dt_k10"),
        ("Clean/hanging_mug", 1.0, "robotwin_dt_k10"), ("Randomized/hanging_mug", 1.0, "robotwin_dt_k10"),
        ("Clean/lift_pot", 1.0, "robotwin_dt_k10"), ("Randomized/lift_pot", 1.0, "robotwin_dt_k10"),
        ("Clean/move_can_pot", 1.0, "robotwin_dt_k10"), ("Randomized/move_can_pot", 1.0, "robotwin_dt_k10"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin_dt_k10"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin_dt_k10"),
        ("Clean/move_playingcard_away", 1.0, "robotwin_dt_k10"), ("Randomized/move_playingcard_away", 1.0, "robotwin_dt_k10"),
        ("Clean/move_stapler_pad", 1.0, "robotwin_dt_k10"), ("Randomized/move_stapler_pad", 1.0, "robotwin_dt_k10"),
        ("Clean/open_laptop", 1.0, "robotwin_dt_k10"), ("Randomized/open_laptop", 1.0, "robotwin_dt_k10"),
        ("Clean/open_microwave", 1.0, "robotwin_dt_k10"), ("Randomized/open_microwave", 1.0, "robotwin_dt_k10"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin_dt_k10"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin_dt_k10"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin_dt_k10"), ("Randomized/pick_dual_bottles", 1.0, "robotwin_dt_k10"),
        ("Clean/place_a2b_left", 1.0, "robotwin_dt_k10"), ("Randomized/place_a2b_left", 1.0, "robotwin_dt_k10"),
        ("Clean/place_a2b_right", 1.0, "robotwin_dt_k10"), ("Randomized/place_a2b_right", 1.0, "robotwin_dt_k10"),
        ("Clean/place_bread_basket", 1.0, "robotwin_dt_k10"), ("Randomized/place_bread_basket", 1.0, "robotwin_dt_k10"),
        ("Clean/place_bread_skillet", 1.0, "robotwin_dt_k10"), ("Randomized/place_bread_skillet", 1.0, "robotwin_dt_k10"),
        ("Clean/place_burger_fries", 1.0, "robotwin_dt_k10"), ("Randomized/place_burger_fries", 1.0, "robotwin_dt_k10"),
        ("Clean/place_can_basket", 1.0, "robotwin_dt_k10"), ("Randomized/place_can_basket", 1.0, "robotwin_dt_k10"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin_dt_k10"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin_dt_k10"),
        ("Clean/place_container_plate", 1.0, "robotwin_dt_k10"), ("Randomized/place_container_plate", 1.0, "robotwin_dt_k10"),
        ("Clean/place_dual_shoes", 1.0, "robotwin_dt_k10"), ("Randomized/place_dual_shoes", 1.0, "robotwin_dt_k10"),
        ("Clean/place_empty_cup", 1.0, "robotwin_dt_k10"), ("Randomized/place_empty_cup", 1.0, "robotwin_dt_k10"),
        ("Clean/place_fan", 1.0, "robotwin_dt_k10"), ("Randomized/place_fan", 1.0, "robotwin_dt_k10"),
        ("Clean/place_mouse_pad", 1.0, "robotwin_dt_k10"), ("Randomized/place_mouse_pad", 1.0, "robotwin_dt_k10"),
        ("Clean/place_object_basket", 1.0, "robotwin_dt_k10"), ("Randomized/place_object_basket", 1.0, "robotwin_dt_k10"),
        ("Clean/place_object_scale", 1.0, "robotwin_dt_k10"), ("Randomized/place_object_scale", 1.0, "robotwin_dt_k10"),
        ("Clean/place_object_stand", 1.0, "robotwin_dt_k10"), ("Randomized/place_object_stand", 1.0, "robotwin_dt_k10"),
        ("Clean/place_phone_stand", 1.0, "robotwin_dt_k10"), ("Randomized/place_phone_stand", 1.0, "robotwin_dt_k10"),
        ("Clean/place_shoe", 1.0, "robotwin_dt_k10"), ("Randomized/place_shoe", 1.0, "robotwin_dt_k10"),
        ("Clean/press_stapler", 1.0, "robotwin_dt_k10"), ("Randomized/press_stapler", 1.0, "robotwin_dt_k10"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin_dt_k10"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin_dt_k10"),
        ("Clean/put_object_cabinet", 1.0, "robotwin_dt_k10"), ("Randomized/put_object_cabinet", 1.0, "robotwin_dt_k10"),
        ("Clean/rotate_qrcode", 1.0, "robotwin_dt_k10"), ("Randomized/rotate_qrcode", 1.0, "robotwin_dt_k10"),
        ("Clean/scan_object", 1.0, "robotwin_dt_k10"), ("Randomized/scan_object", 1.0, "robotwin_dt_k10"),
        ("Clean/shake_bottle", 1.0, "robotwin_dt_k10"), ("Randomized/shake_bottle", 1.0, "robotwin_dt_k10"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin_dt_k10"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin_dt_k10"),
        ("Clean/stack_blocks_three", 1.0, "robotwin_dt_k10"), ("Randomized/stack_blocks_three", 1.0, "robotwin_dt_k10"),
        ("Clean/stack_blocks_two", 1.0, "robotwin_dt_k10"), ("Randomized/stack_blocks_two", 1.0, "robotwin_dt_k10"),
        ("Clean/stack_bowls_three", 1.0, "robotwin_dt_k10"), ("Randomized/stack_bowls_three", 1.0, "robotwin_dt_k10"),
        ("Clean/stack_bowls_two", 1.0, "robotwin_dt_k10"), ("Randomized/stack_bowls_two", 1.0, "robotwin_dt_k10"),
        ("Clean/stamp_seal", 1.0, "robotwin_dt_k10"), ("Randomized/stamp_seal", 1.0, "robotwin_dt_k10"),
        ("Clean/turn_switch", 1.0, "robotwin_dt_k10"), ("Randomized/turn_switch", 1.0, "robotwin_dt_k10"),
    ],
}
