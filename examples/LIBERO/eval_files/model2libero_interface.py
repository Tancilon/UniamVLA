# PEP 585 generic-alias syntax (`list[int]`, `tuple[dict[str, ...], ...]`)
# appears below at module scope. Eval clients may run on Python 3.8 (e.g.,
# the `calvin_env` conda env), where evaluating those annotations at import
# time raises TypeError. Make annotations lazy (PEP 563, Python 3.7+).
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import yaml

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler


def read_mode_config(pretrained_checkpoint):
    checkpoint_pt = Path(pretrained_checkpoint)
    if not checkpoint_pt.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {checkpoint_pt}")

    run_dir = checkpoint_pt.parents[1]
    config_yaml = run_dir / "config.yaml"
    dataset_statistics_json = run_dir / "dataset_statistics.json"
    if not config_yaml.exists():
        raise FileNotFoundError(f"Missing config.yaml for run dir: {run_dir}")
    if not dataset_statistics_json.exists():
        raise FileNotFoundError(f"Missing dataset_statistics.json for run dir: {run_dir}")

    with open(config_yaml) as f:
        model_config = yaml.safe_load(f)
    with open(dataset_statistics_json) as f:
        norm_stats = json.load(f)

    action_model = model_config.get("framework", {}).get("action_model", {})
    if "future_action_window_size" not in action_model and "action_horizon" in action_model:
        action_model["future_action_window_size"] = int(action_model["action_horizon"]) - 1

    return model_config, norm_stats


def _to_numpy_leaves(d: dict) -> dict:
    """Walk a 1-level nested dict; convert torch.Tensor leaves to numpy.

    Used at the WebSocket boundary: msgpack-numpy serializes numpy.ndarray
    natively but not torch.Tensor. Server-side stack_canonical accepts numpy
    after the §6.4 torch.as_tensor wrap.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {
                kk: vv.detach().cpu().numpy() if hasattr(vv, "detach") else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v.detach().cpu().numpy() if hasattr(v, "detach") else v
    return out


def _debug_uamvla_actions_enabled() -> bool:
    return os.getenv("UAMVLA_DEBUG_ACTIONS", "").lower() in {"1", "true", "yes", "on"}


def _debug_print_action_array(name: str, arr: np.ndarray) -> None:
    arr_np = np.asarray(arr, dtype=np.float32)
    flat = arr_np.reshape(-1, arr_np.shape[-1])
    per_dim_min = np.round(flat.min(axis=0), 5).tolist()
    per_dim_max = np.round(flat.max(axis=0), 5).tolist()
    per_dim_mean = np.round(flat.mean(axis=0), 5).tolist()
    first = np.round(arr_np.reshape(-1, arr_np.shape[-1])[0], 5).tolist()
    print(
        f"*** UamVLA action debug(client): {name} "
        f"shape={arr_np.shape}, first={first}, "
        f"min={per_dim_min}, max={per_dim_max}, mean={per_dim_mean} ***",
        flush=True,
    )


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble=True,
        action_ensemble_horizon: Optional[int] = 3,  # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        gripper_binarize_threshold: float = 0.5,
        gripper_interpretation: str = "positive_open",
        action_normalization_mode: str = "min_max",
        action_query_interval: Optional[int] = None,
        host="0.0.0.0",
        port=10095,
    ) -> None:

        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup

        # ----- Resolve unnorm_key once and write back to self (fixes B1).
        # eval_libero.py does not pass unnorm_key; the staticmethod in
        # get_action_stats resolved it locally but did NOT update self.
        # Doing it here makes self.unnorm_key authoritative for both
        # action stats and state stats lookups.
        _model_config, _norm_stats = read_mode_config(policy_ckpt_path)
        self.model_config = _model_config
        self.unnorm_key = self._check_unnorm_key(_norm_stats, unnorm_key)
        self.action_norm_stats = _norm_stats[self.unnorm_key]["action"]

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {self.unnorm_key} ***")

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon  # 0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.gripper_binarize_threshold = float(gripper_binarize_threshold)
        if gripper_interpretation not in {"positive_open", "libero_sign"}:
            raise ValueError(
                "gripper_interpretation must be one of "
                "{'positive_open', 'libero_sign'}"
            )
        self.gripper_interpretation = gripper_interpretation
        self.action_normalization_mode = self._resolve_action_normalization_mode(
            action_normalization_mode,
            self.model_config,
            self.action_norm_stats,
        )
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        self.action_query_interval = int(action_query_interval or self.action_chunk_size)
        if self.action_query_interval <= 0:
            raise ValueError("action_query_interval must be positive")
        if self.action_query_interval > self.action_chunk_size:
            raise ValueError(
                "action_query_interval cannot exceed action_chunk_size "
                f"({self.action_chunk_size})"
            )

        # ----- UamVLA opt-in state-passthrough setup (fixes B2 + spec §6.3.1).
        # statistics.yaml lives at the run dir level (mirrors dataset_statistics.json).
        # read_mode_config resolves run_dir = checkpoint_pt.parents[1] so we use the
        # same logic here.
        self.uamvla_state_enabled = False  # public; gates eval_libero.py assembly
        run_dir = Path(policy_ckpt_path).parents[1]
        stats_yaml_path = run_dir / "statistics.yaml"
        if stats_yaml_path.exists():
            with open(stats_yaml_path) as f:
                stats_dict = yaml.safe_load(f)
            if (
                isinstance(stats_dict, dict)
                and "state_stats" in stats_dict
                and self.unnorm_key in stats_dict["state_stats"]
            ):
                # Lazy import: the state-passthrough path pulls in
                # gr00t_lerobot.schema -> numpydantic, which is not available
                # in the CALVIN eval conda env (Python 3.8) — but CALVIN's
                # state_normalizer path itself is the same module chain, so
                # if the deps are missing the whole passthrough degrades to
                # disabled and the model runs in stateless mode (still
                # functional, with the documented training/eval state gap).
                try:
                    from starVLA.model.modules.uamvla.data.embodiment_registry import (
                        get_embodiment_config,
                    )
                    from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer
                except ImportError as e:
                    print(
                        f"*** UamVLA state passthrough disabled "
                        f"(state_stats present but deps missing in this env: {e}) ***"
                    )
                else:
                    try:
                        adapter = get_embodiment_config(self.unnorm_key)["adapter"]
                    except KeyError as e:
                        print(
                            f"*** UamVLA state passthrough disabled "
                            f"(no adapter registered for unnorm_key={self.unnorm_key!r}: {e}) ***"
                        )
                    else:
                        self._adapter = adapter
                        # Pin apply_to to the training-side field list so future
                        # state_stats schema additions cannot silently diverge between
                        # train (uamvla_libero.yaml / uamvla_calvin{,_abcd}.yaml:
                        # state_encoder.normalization.apply_to) and eval. Today
                        # this matches the canonical fields written by both the
                        # franka_libero and franka_calvin preprocessors.
                        self._state_normalizer = StateNormalizer(
                            stats_dict=stats_dict,
                            embodiment=self.unnorm_key,
                            mode="q99",
                            apply_to=["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"],
                        )
                        self.uamvla_state_enabled = True
                        print(
                            f"*** UamVLA state passthrough enabled "
                            f"(stats: {stats_yaml_path}, adapter: {type(self._adapter).__name__}) ***"
                        )

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(self, example: dict, step: int = 0, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one step of inference
        :param image: Input image in the format (H, W, 3), type uint8
        :param task_description: Task description text
        :return: (raw action, processed action)
        """

        task_description = example.get("lang", None)
        images = example["image"]  # list of images for history

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = [self._resize_image(image) for image in images]
        example["image"] = images

        # Spec §6.3.2: always pop the UamVLA raw-state key so it never reaches
        # the wire. Conversion to canonical_state is gated on
        # uamvla_state_enabled (see __init__).
        raw = example.pop("uamvla_raw_state", None)
        if self.uamvla_state_enabled and raw is not None:
            canonical = self._adapter.to_canonical(raw)
            canonical = self._state_normalizer(canonical)
            example["canonical_state"] = _to_numpy_leaves(canonical)

        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        action_query_interval = self.action_query_interval
        if step % action_query_interval == 0 or not hasattr(self, "raw_actions"):
            response = self.client.predict_action(vla_input)
            if isinstance(response, dict) and response.get("ok") is False:
                error = response.get("error", {})
                if isinstance(error, dict):
                    message = error.get("message") or repr(error)
                    tb = error.get("traceback")
                else:
                    message = str(error)
                    tb = None
                details = f"Policy server inference failed: {message}"
                if tb:
                    details = f"{details}\n{tb}"
                raise RuntimeError(details)

            try:
                normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
            except KeyError as e:
                print(f"Response data: {response}")
                data = response.get("data") if isinstance(response, dict) else None
                data_keys = list(data.keys()) if isinstance(data, dict) else None
                response_keys = list(response.keys()) if isinstance(response, dict) else None
                raise KeyError(
                    "Malformed policy server response: "
                    f"missing {e.args[0]!r}; "
                    f"response_keys={response_keys}; data_keys={data_keys}"
                ) from e

            normalized_actions = normalized_actions[0]
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions,
                action_norm_stats=self.action_norm_stats,
                normalization_mode=self.action_normalization_mode,
                gripper_binarize_threshold=self.gripper_binarize_threshold,
                gripper_interpretation=self.gripper_interpretation,
            )
            if _debug_uamvla_actions_enabled():
                _debug_print_action_array("normalized_actions(client)", normalized_actions)
                _debug_print_action_array("raw_actions(client)", self.raw_actions)

        raw_actions = self.raw_actions[step % action_query_interval][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        return {"raw_action": raw_action}

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
        gripper_binarize_threshold: float = 0.5,
        gripper_interpretation: str = "positive_open",
    ) -> np.ndarray:
        action_high, action_low = ModelClient._get_normalization_bounds(
            action_norm_stats,
            normalization_mode=normalization_mode,
        )
        mask = np.asarray(
            action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool)),
            dtype=bool,
        )
        normalized_actions = np.clip(normalized_actions, -1, 1)
        if gripper_interpretation == "positive_open":
            normalized_actions[:, 6] = np.where(
                normalized_actions[:, 6] < gripper_binarize_threshold,
                0,
                1,
            )
        elif gripper_interpretation == "libero_sign":
            normalized_actions[:, 6] = np.where(
                normalized_actions[:, 6] < gripper_binarize_threshold,
                1,
                0,
            )
        else:
            raise ValueError(
                "gripper_interpretation must be one of "
                "{'positive_open', 'libero_sign'}"
            )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _get_normalization_bounds(
        norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> tuple[np.ndarray, np.ndarray]:
        if normalization_mode == "q99":
            if "q01" not in norm_stats or "q99" not in norm_stats:
                raise KeyError(
                    "Normalization mode `q99` requires statistics keys `q01` and `q99`."
                )
            return np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
        if normalization_mode == "min_max":
            if "min" not in norm_stats or "max" not in norm_stats:
                raise KeyError(
                    "Normalization mode `min_max` requires statistics keys `min` and `max`."
                )
            return np.array(norm_stats["max"]), np.array(norm_stats["min"])
        raise ValueError(
            f"Unsupported normalization_mode: {normalization_mode}. "
            "Expected one of ['min_max', 'q99']."
        )

    @staticmethod
    def _resolve_action_normalization_mode(
        requested_mode: str,
        model_config: dict,
        action_norm_stats: Dict[str, np.ndarray],
    ) -> str:
        if requested_mode != "auto":
            # Validate eagerly so a typo fails before the long-running eval loop.
            ModelClient._get_normalization_bounds(
                action_norm_stats,
                normalization_mode=requested_mode,
            )
            return requested_mode

        datasets_cfg = (model_config or {}).get("datasets", {})
        vla_cfg = datasets_cfg.get("vla_data", {}) if isinstance(datasets_cfg, dict) else {}
        data_mix = str(vla_cfg.get("data_mix", ""))
        if "starvla_uam_state_h8" in data_mix:
            ModelClient._get_normalization_bounds(action_norm_stats, normalization_mode="min_max")
            return "min_max"
        return "min_max"

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        # import ipdb; ipdb.set_trace()
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        target_w, target_h = tuple(self.image_size)
        if image.shape[1] == target_w and image.shape[0] == target_h:
            return image
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """Resolve unnorm_key against norm_stats: pick the sole entry if None,
        otherwise validate the requested key is present."""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
