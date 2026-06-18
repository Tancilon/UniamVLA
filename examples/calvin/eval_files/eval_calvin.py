"""
Calvin Multi-Step Evaluation Script

Based on RoboFlamingo's evaluation protocol:
https://github.com/RoboFlamingo/RoboFlamingo/blob/main/robot_flamingo/eval/eval_utils.py

Evaluates a policy server on Calvin's long-horizon multi-task benchmark.
Measures success rate on chains of 1-5 consecutive tasks.

Usage:
    python examples/calvin/eval_calvin.py \
        --args.host 0.0.0.0 \
        --args.port 8000 \
        --args.dataset_path /path/to/calvin/task_D_D \
        --args.num_sequences 1000
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import hydra
import numpy as np
try:
    import tyro
except ModuleNotFoundError:
    tyro = None

# # Add Calvin to path
# CALVIN_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "calvin"
# sys.path.insert(0, str(CALVIN_ROOT))
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm

from deployment.model_server.tools import image_tools
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

# from calvin_env.envs.play_table_env import get_env

# Set OpenGL platform for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EP_LEN = 360  # Max steps per task


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5  # 0 means use the model checkpoint's default action horizon
    pretrained_path: str = ""
    unnorm_key: str = ""
    
    use_train_renderer: bool = False
    gripper_binarize_threshold: float = 0.0
    action_normalization_mode: str = "min_max"

    #################################################################################################################
    # Calvin environment-specific parameters
    #################################################################################################################
    dataset_path: str = "/path/to/calvin/task_D_D"  # Path to Calvin dataset
    calvin_config_path: str = "/path/to/calvin/calvin_models/conf"
    eval_sequences_path: str = "/path/to/calvin/eval_sequences.json"
    num_sequences: int = 1000  # Number of evaluation sequences
    num_workers: int = 1  # For future multi-process support
    seed: int = 0
    create_plan_tsne: bool = False

    #################################################################################################################
    # Evaluation settings
    #################################################################################################################
    debug: bool = False  # Save debug videos
    eval_log_dir: str = "tmp/calvin/eval_logs"  # Path to save evaluation logs and videos
    reset: bool = False  # If True, reset robot state between tasks (easier)
    diverse_inst: bool = False  # Use diverse instructions (zero-shot generalization)


class CalvinPolicyClient:
    """Wrapper around websocket client with Calvin-specific preprocessing."""

    def __init__(
        self,
        host: str,
        port: int,
        resize_size: int = 256,
        replan_steps: int = 0,
        pretrained_path: str = "",
        unnorm_key: str = "",
        train_renderer=None,
        gripper_binarize_threshold: float = 0.0,
        action_normalization_mode: str = "auto",
    ):
        self.client = ModelClient(
            policy_ckpt_path=pretrained_path,
            host=host,
            port=port,
            image_size=[resize_size, resize_size],
            unnorm_key=(unnorm_key or None),
            gripper_binarize_threshold=gripper_binarize_threshold,
            action_normalization_mode=action_normalization_mode,
            action_query_interval=replan_steps if replan_steps > 0 else None,
        )
        self.resize_size = resize_size
        self.replan_steps = replan_steps
        self.step_count = 0
        self.train_renderer = train_renderer
        self._uamvla_gr00t_state_indices = self._gr00t_state_indices_from_config(
            getattr(self.client, "model_config", None)
        )
        self.send_uamvla_gr00t_state = self._client_uses_uamvla_gr00t_state(self.client)
        self._uamvla_gr00t_state_norm_mode = None
        self._uamvla_gr00t_state_stat_a = None
        self._uamvla_gr00t_state_stat_b = None
        if self.send_uamvla_gr00t_state:
            stats_key = getattr(self.client, "unnorm_key", None) or unnorm_key or None
            (
                self._uamvla_gr00t_state_norm_mode,
                self._uamvla_gr00t_state_stat_a,
                self._uamvla_gr00t_state_stat_b,
            ) = self._load_uamvla_gr00t_state_stats(
                pretrained_path,
                stats_key,
                self._uamvla_gr00t_state_indices,
            )

    @staticmethod
    def _nested_config_get(config, *keys):
        current = config
        for key in keys:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = getattr(current, key, None)
        return current

    @classmethod
    def _client_uses_uamvla_gr00t_state(cls, client) -> bool:
        config = getattr(client, "model_config", None)
        framework_name = cls._nested_config_get(config, "framework", "name")
        state_dim = cls._nested_config_get(config, "framework", "action_model", "state_dim")
        indices = cls._gr00t_state_indices_from_config(config)
        try:
            state_dim = int(state_dim or 0)
        except (TypeError, ValueError):
            state_dim = 0
        return framework_name in {"UamGR00T", "UamVLAGR00T"} and state_dim >= len(indices)

    @classmethod
    def _gr00t_state_indices_from_config(cls, config) -> list[int]:
        indices = cls._nested_config_get(config, "datasets", "vla_data", "gr00t_state_indices")
        if indices is None:
            indices = cls._nested_config_get(
                config,
                "framework",
                "action_model",
                "gr00t_state_indices",
            )
        if indices is None:
            return list(range(7))
        return [int(i) for i in indices]

    @staticmethod
    def _load_uamvla_gr00t_state_stats(
        pretrained_path: str,
        unnorm_key: Optional[str],
        state_indices: list[int],
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        checkpoint_path = Path(pretrained_path)
        if len(checkpoint_path.parents) < 2:
            raise FileNotFoundError(
                "UamGR00T CALVIN eval requires a checkpoint path inside a "
                f"training run directory, got {pretrained_path!r}."
            )

        stats_path = checkpoint_path.parents[1] / "dataset_statistics.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                "UamGR00T CALVIN eval requires state normalization stats at "
                f"{stats_path}. Expected dataset_statistics.json from the "
                "training run directory."
            )

        with open(stats_path) as f:
            dataset_stats = json.load(f)

        stats_key = unnorm_key
        if stats_key is None:
            if len(dataset_stats) != 1:
                raise KeyError(
                    "dataset_statistics.json has multiple entries; pass "
                    f"--args.unnorm-key. Available keys: {list(dataset_stats.keys())}"
                )
            stats_key = next(iter(dataset_stats.keys()))

        if stats_key not in dataset_stats:
            raise KeyError(
                f"State stats key {stats_key!r} not found in {stats_path}. "
                f"Available keys: {list(dataset_stats.keys())}"
            )

        state_stats = dataset_stats[stats_key].get("state")
        if not isinstance(state_stats, dict):
            raise KeyError(
                f"{stats_path} entry {stats_key!r} must contain "
                "state normalization stats for UamGR00T CALVIN eval."
            )

        if "q01" in state_stats and "q99" in state_stats:
            mode = "q99"
            stat_a = np.asarray(state_stats["q01"], dtype=np.float32).reshape(-1)
            stat_b = np.asarray(state_stats["q99"], dtype=np.float32).reshape(-1)
        elif "mean" in state_stats and "std" in state_stats:
            mode = "mean_std"
            stat_a = np.asarray(state_stats["mean"], dtype=np.float32).reshape(-1)
            stat_b = np.asarray(state_stats["std"], dtype=np.float32).reshape(-1)
        else:
            raise KeyError(
                f"{stats_path} entry {stats_key!r} must contain either "
                "state.q01/state.q99 or state.mean/state.std."
            )

        if not state_indices:
            raise ValueError("gr00t_state_indices must contain at least one index.")
        max_index = max(state_indices)
        if stat_a.shape[0] <= max_index or stat_b.shape[0] <= max_index:
            raise ValueError(
                f"Expected state stats with at least {max_index + 1} values in "
                f"{stats_path} for key {stats_key!r}, got "
                f"stat_a={stat_a.shape}, stat_b={stat_b.shape}."
            )
        return mode, stat_a[state_indices], stat_b[state_indices]

    @staticmethod
    def _extract_raw_uamvla_gr00t_state(obs: dict, state_indices: list[int]) -> np.ndarray:
        robot_obs = np.asarray(obs["robot_obs"], dtype=np.float32).reshape(-1)
        if not state_indices:
            raise ValueError("gr00t_state_indices must contain at least one index.")
        max_index = max(state_indices)
        if robot_obs.shape[0] <= max_index:
            raise ValueError(
                f"Expected CALVIN robot_obs with at least {max_index + 1} dims, "
                f"got {robot_obs.shape}."
            )
        return robot_obs[state_indices].reshape(1, len(state_indices))

    def _extract_uamvla_gr00t_state(self, obs: dict) -> np.ndarray:
        state = self._extract_raw_uamvla_gr00t_state(
            obs,
            self._uamvla_gr00t_state_indices,
        ).reshape(-1)
        if (
            self._uamvla_gr00t_state_norm_mode is None
            or self._uamvla_gr00t_state_stat_a is None
            or self._uamvla_gr00t_state_stat_b is None
        ):
            raise RuntimeError(
                "UamGR00T state normalization stats were not initialized."
            )

        normalized = np.zeros_like(state, dtype=np.float32)
        if self._uamvla_gr00t_state_norm_mode == "q99":
            q01 = self._uamvla_gr00t_state_stat_a
            q99 = self._uamvla_gr00t_state_stat_b
            mask = q01 != q99
            normalized[mask] = 2 * (state[mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1
            normalized[~mask] = state[~mask]
            normalized = np.clip(normalized, -1, 1)
        elif self._uamvla_gr00t_state_norm_mode == "mean_std":
            mean = self._uamvla_gr00t_state_stat_a
            std = self._uamvla_gr00t_state_stat_b
            mask = std != 0
            normalized[mask] = (state[mask] - mean[mask]) / std[mask]
            normalized[~mask] = state[~mask]
        else:
            raise ValueError(
                f"Unsupported UamGR00T state normalization mode: "
                f"{self._uamvla_gr00t_state_norm_mode!r}"
            )
        return normalized.reshape(1, len(self._uamvla_gr00t_state_indices))

    def reset(self):
        """Reset action plan buffer."""
        self.step_count = 0

    def step(self, obs: dict, lang_annotation: str) -> np.ndarray:
        """
        Query policy for action given observation and language instruction.

        Args:
            obs: Calvin observation dict with keys:
                - rgb_obs: dict with 'rgb_static' (200x200x3) and 'rgb_gripper' (84x84x3)
                - robot_obs: (15,) proprioceptive state [ee_pos(3), ee_ori(3), gripper(2), joint_pos(7)]
            lang_annotation: Natural language task description
            get_action: If True, query model for new action chunk

        Returns:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Prefer the training-time CALVIN renderer path for UamVLA checkpoints:
        # render both views at 256x256, then let the backbone perform the only
        # resize to Qwen3-VL's 640x640 target.
        if self.train_renderer is not None:
            rendered = self.train_renderer.render_cameras(
                width=self.resize_size,
                height=self.resize_size,
            )
            image = image_tools.convert_to_uint8(rendered["rgb_static"])
            wrist_image = image_tools.convert_to_uint8(rendered["rgb_wrist"])
        else:
            rgb_static = obs["rgb_obs"]["rgb_static"]  # (200, 200, 3) uint8
            rgb_gripper = obs["rgb_obs"]["rgb_gripper"]  # (84, 84, 3) uint8

            image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rgb_static, self.resize_size, self.resize_size)
            )
            wrist_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(rgb_gripper, self.resize_size, self.resize_size)
            )

        # Prepare input for policy server (aligned with eval_libero)
        example = {
            "image": [image, wrist_image],
            "lang": lang_annotation,
        }
        if self.send_uamvla_gr00t_state:
            example["state"] = self._extract_uamvla_gr00t_state(obs)

        # Spec parallel of LIBERO state-passthrough: hand the inner ModelClient
        # the raw 15-D CALVIN robot_obs only when ModelClient successfully
        # initialised the CalvinAdapter + StateNormalizer (gated by
        # uamvla_state_enabled). CalvinAdapter.REQUIRED_FIELDS == ("robot_obs",)
        # so this single key is sufficient; ModelClient.step() converts it to
        # canonical_state and pops uamvla_raw_state before going on the wire.
        if getattr(self.client, "uamvla_state_enabled", False):
            example["uamvla_raw_state"] = {
                "robot_obs": np.asarray(obs["robot_obs"], dtype=np.float32),
            }

        # Query model
        model_output = self.client.step(example=example, step=self.step_count)
        raw_action = model_output["raw_action"]
        world_vector = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
        rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
        open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)

        action = np.concatenate([world_vector, rotation_delta, open_gripper], axis=0).astype(np.float32)
        self.step_count += 1
        return action


def make_env(dataset_path: str):
    """Initialize Calvin environment without tactile sensor (to avoid OpenGL issues)."""
    val_folder = Path(dataset_path) / "validation"

    # Load config and disable tactile sensor to avoid pyrender/OpenGL conflicts
    from omegaconf import OmegaConf

    config_path = val_folder / ".hydra" / "merged_config.yaml"
    cfg = OmegaConf.load(config_path)

    # Remove tactile sensor from camera list if it exists
    if hasattr(cfg.env, "cameras") and "tactile" in cfg.env.cameras:
        # Create a new camera dict without tactile
        new_cameras = OmegaConf.create({k: v for k, v in cfg.env.cameras.items() if k != "tactile"})
        cfg.env.cameras = new_cameras

    # Initialize environment with modified config
    import hydra

    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

    return env


def load_lang_task(dataset_path: str) -> dict:
    """Load language annotations and task oracle for Calvin validation set."""
    conf_dir = Path(dataset_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return val_annotations, task_oracle


def _limit_eval_sequences(eval_sequences: list, num_sequences: int) -> list:
    """Keep the first num_sequences entries; values <=0 mean no limit."""
    if num_sequences <= 0:
        return eval_sequences
    return eval_sequences[:num_sequences]


def evaluate_policy_ddp(
    policy,
    env,
    epoch,
    calvin_conf_path,
    eval_sequences_path,
    num_sequences,
    eval_log_dir=None,
    debug=False,
    create_plan_tsne=False,
    reset=False,
    diverse_inst=False,
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open("/mnt/bn/robotics/lxh/robot-flamingo/lang_annotation_cache.json", "r") as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    with open(eval_sequences_path, "r") as f:
        eval_sequences = json.load(f)
    eval_sequences = _limit_eval_sequences(eval_sequences, num_sequences)
    # device_num = int(torch.distributed.get_world_size())
    # device_id = torch.distributed.get_rank()
    # assert num_sequences % device_num == 0
    # interval_len = int(num_sequences // device_num)
    # eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, num_sequences)]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = 0  # device_id * interval_len

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(
            env,
            policy,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            plans,
            debug,
            eval_log_dir,
            base_sequence_i + local_sequence_i,
            reset=reset,
            diverse_inst=diverse_inst,
        )
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        local_sequence_i += 1

    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]

    # if create_plan_tsne:
    #     create_tsne(plans, eval_log_dir, epoch)

    eval_sequences = extract_iter_from_tqdm(eval_sequences)

    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(
    env,
    policy,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    plans,
    debug,
    eval_log_dir="",
    sequence_i=-1,
    reset=False,
    diverse_inst=False,
):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            success = rollout(
                env,
                policy,
                task_checker,
                subtask,
                val_annotations,
                plans,
                debug,
                eval_log_dir,
                subtask_i,
                sequence_i,
                robot_obs=robot_obs,
                scene_obs=scene_obs,
                diverse_inst=diverse_inst,
            )
        else:
            success = rollout(
                env,
                policy,
                task_checker,
                subtask,
                val_annotations,
                plans,
                debug,
                eval_log_dir,
                subtask_i,
                sequence_i,
                diverse_inst=diverse_inst,
            )
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(
    env,
    policy,
    task_oracle,
    subtask,
    val_annotations,
    plans,
    debug,
    eval_log_dir="",
    subtask_i=-1,
    sequence_i=-1,
    robot_obs=None,
    scene_obs=None,
    diverse_inst=False,
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split("\n")[0]
    if "\u2019" in lang_annotation:
        lang_annotation.replace("\u2019", "'")
    policy.reset()
    start_info = env.get_info()

    if debug:
        img_queue = []

    for step in range(EP_LEN):

        action = policy.step(obs, lang_annotation)

        # Ensure action is writable (Calvin env modifies it in-place)
        if not action.flags.writeable:
            action = np.array(action, copy=True)
        action[-1] = 1 if action[-1] > 0 else -1

        obs, _, _, current_info = env.step(action)
        if debug:
            img_copy = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
            img_queue.append(img_copy)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(policy, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                img_clip = ImageSequenceClip(img_queue, fps=30)
                img_clip.write_gif(os.path.join(eval_log_dir, f"{sequence_i}-{subtask_i}-{subtask}-succ.gif"), fps=30)
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        img_clip = ImageSequenceClip(img_queue, fps=30)
        img_clip.write_gif(os.path.join(eval_log_dir, f"{sequence_i}-{subtask_i}-{subtask}-fail.gif"), fps=30)
    return False


def main(args: Args):
    # args = tyro.cli(Args)

    env = make_env(args.dataset_path)
    train_renderer = None
    if args.use_train_renderer:
        from tools.preprocess.calvin_env_adapter import CalvinEnvAdapter

        train_renderer = CalvinEnvAdapter(env)

    policy = CalvinPolicyClient(
        args.host,
        args.port,
        args.resize_size,
        args.replan_steps,
        pretrained_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
        train_renderer=train_renderer,
        gripper_binarize_threshold=args.gripper_binarize_threshold,
        action_normalization_mode=args.action_normalization_mode,
    )

    evaluate_policy_ddp(
        policy,
        env,
        0,
        args.calvin_config_path,
        args.eval_sequences_path,
        args.num_sequences,
        args.eval_log_dir,
        args.debug,
        args.create_plan_tsne,
        args.reset,
        args.diverse_inst,
    )


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


# def _parse_args_with_argparse() -> Args:
#     """Small tyro-compatible fallback for older CALVIN envs without tyro."""
#     import argparse

#     parser = argparse.ArgumentParser(description="Evaluate a policy server on CALVIN.")
#     for field in dataclasses.fields(Args):
#         default = field.default
#         dashed = field.name.replace("_", "-")
#         flags = (f"--args.{dashed}", f"--args.{field.name}")
#         if isinstance(default, bool):
#             group = parser.add_mutually_exclusive_group()
#             group.add_argument(
#                 *flags,
#                 dest=field.name,
#                 nargs="?",
#                 const=True,
#                 default=default,
#                 type=_parse_bool,
#             )
#             group.add_argument(f"--args.no-{dashed}", dest=field.name, action="store_false")
#         else:
#             parser.add_argument(*flags, dest=field.name, default=default, type=type(default))

#     return Args(**vars(parser.parse_args()))


if __name__ == "__main__":
    if tyro is not None:
        tyro.cli(main)
    # else:
    #     main(_parse_args_with_argparse())
