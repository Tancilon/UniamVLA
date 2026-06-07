#!/usr/bin/env python3
"""
Convert an extracted / non-zip CALVIN dataset directory into a StarVLA-first
LeRobot dataset, while adding only the extra UamVLA state columns.

This is the directory-based version of convert_to_lerobot_starvla_with_uam_state.py.

Input can be either:
  1) dataset root:
       /path/to/task_D_D
       /path/to/task_D_D/training
       /path/to/task_D_D/validation

     Run with:
       --input-dir /path/to/task_D_D --splits training

  2) split dir directly:
       /path/to/task_D_D/training

     Run with:
       --input-dir /path/to/task_D_D/training

Schema policy
-------------
Keep original StarVLA fields:
  image
  wrist_image
  state      # 8D by default
  actions    # 7D

Add only these extra state fields:
  state.target_pose_rot6d
  state.target_pose_trans
  state.static_cam_rot6d
  state.static_cam_trans

Do NOT write:
  state.robot_obs
  action.x/action.y/...
  video.primary_image/video.wrist_image

Real target/camera state requires calvin_env:
  --extra-state-mode env

Schema-only debug mode:
  --extra-state-mode lite
"""

from __future__ import annotations

import json
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

# Make project root importable when this script is placed at examples/calvin/.
# For /repo/examples/calvin/script.py, parents[2] is /repo.
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except Exception:
    pass

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


MERGED_MODALITY_JSON = {
    "state": {
        # Original StarVLA state mapping.
        "x": {"start": 0, "end": 1, "original_key": "state"},
        "y": {"start": 1, "end": 2, "original_key": "state"},
        "z": {"start": 2, "end": 3, "original_key": "state"},
        "roll": {"start": 3, "end": 4, "original_key": "state"},
        "pitch": {"start": 4, "end": 5, "original_key": "state"},
        "yaw": {"start": 5, "end": 6, "original_key": "state"},
        "pad": {"start": 6, "end": 7, "original_key": "state"},
        "gripper": {"start": 7, "end": 8, "original_key": "state"},

        # Extra UamVLA state information.
        # state.robot_obs is intentionally omitted.
        "target_pose_rot6d": {
            "start": 0,
            "end": 6,
            "original_key": "state.target_pose_rot6d",
        },
        "target_pose_trans": {
            "start": 0,
            "end": 3,
            "original_key": "state.target_pose_trans",
        },
        "static_cam_rot6d": {
            "start": 0,
            "end": 6,
            "original_key": "state.static_cam_rot6d",
        },
        "static_cam_trans": {
            "start": 0,
            "end": 3,
            "original_key": "state.static_cam_trans",
        },
    },
    "action": {
        # Keep original StarVLA action-vector slicing.
        "x": {"start": 0, "end": 1, "original_key": "actions"},
        "y": {"start": 1, "end": 2, "original_key": "actions"},
        "z": {"start": 2, "end": 3, "original_key": "actions"},
        "roll": {"start": 3, "end": 4, "original_key": "actions"},
        "pitch": {"start": 4, "end": 5, "original_key": "actions"},
        "yaw": {"start": 5, "end": 6, "original_key": "actions"},
        "gripper": {"start": 6, "end": 7, "original_key": "actions"},
    },
    "video": {
        # Keep original StarVLA video keys.
        "primary_image": {"original_key": "image"},
        "wrist_image": {"original_key": "wrist_image"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


@dataclass(frozen=True)
class Args:
    input_dir: str

    output_root: str = "/mnt/data/jiangnan/lerobot"
    repo_id: str | None = None
    fps: int = 10
    splits: Literal["training", "validation", "both"] = "training"

    action_key: Literal["rel_actions", "actions"] = "rel_actions"
    state_format: Literal["starvla8", "raw15"] = "starvla8"

    robot_type: str = "panda"
    max_episodes: int | None = None
    overwrite: bool = True

    # env: real target/camera state via calvin_env
    # lite: zero-filled target/camera state for smoke tests only
    # none: original StarVLA-only conversion
    extra_state_mode: Literal["env", "lite", "none"] = "env"

    # Required for stack/unstack target resolution in env mode.
    # For task_D_D use D. For multiscene data, split by scene first.
    default_scene: str | None = None

    on_resolve_failure: Literal["skip", "abort"] = "abort"
    on_missing_target: Literal["skip", "abort"] = "abort"

    render_width: int = 256
    render_height: int = 256

    write_modality_json: bool = True


def _as_1d_float32(x, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.shape[0] != dim:
        raise ValueError(f"{name} expected {dim} dims, got shape {arr.shape}")
    return arr


def _normalize_task(task) -> str:
    return str(task).strip().split("\n")[0]


def _load_npy(path: Path):
    return np.load(path, allow_pickle=True)


def _is_split_dir(path: Path) -> bool:
    return (
        (path / "lang_annotations" / "auto_lang_ann.npy").exists()
        and any(path.glob("episode_*.npz"))
    )


def _resolve_split_dirs(
    input_dir: Path,
    splits: Literal["training", "validation", "both"],
) -> list[tuple[str, Path]]:
    """Return [(split_name, split_dir), ...]."""
    input_dir = input_dir.resolve()

    if _is_split_dir(input_dir):
        split_name = input_dir.name if input_dir.name in {"training", "validation"} else "training"
        if splits != "both" and input_dir.name in {"training", "validation"} and splits != input_dir.name:
            warnings.warn(
                f"--splits={splits!r} was requested, but --input-dir points to "
                f"a {input_dir.name!r} split dir. Using {input_dir.name!r}.",
                RuntimeWarning,
            )
        return [(split_name, input_dir)]

    requested = ["training", "validation"] if splits == "both" else [splits]
    out: list[tuple[str, Path]] = []
    for split in requested:
        split_dir = input_dir / split
        if not _is_split_dir(split_dir):
            raise FileNotFoundError(
                f"Could not find a valid CALVIN split dir at {split_dir}. "
                "Pass --input-dir as either the dataset root containing training/validation "
                "or the split dir itself."
            )
        out.append((split, split_dir))
    return out


def _load_language_annotations(split_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    lang_data = _load_npy(path).item()
    ranges = lang_data["info"]["indx"]
    instructions = lang_data["language"]["ann"]
    task_labels = lang_data["language"].get("task", None)
    return ranges, instructions, task_labels


def _load_step_npz(split_dir: Path, step_id: int) -> dict[str, np.ndarray]:
    path = split_dir / f"episode_{step_id:07d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing CALVIN step file: {path}")
    npz = np.load(path, allow_pickle=True)
    try:
        return {k: npz[k] for k in npz.files}
    finally:
        npz.close()


def _build_starvla_state(robot_obs: np.ndarray, state_format: str) -> np.ndarray:
    robot_obs = _as_1d_float32(robot_obs, 15, "robot_obs")
    if state_format == "raw15":
        return robot_obs

    # CALVIN 15D robot_obs:
    # [eef_xyz(3), eef_rpy(3), gripper_width(1), joint_pos(7), gripper_action(1)]
    #
    # StarVLA 8D state:
    # [x, y, z, roll, pitch, yaw, pad, gripper]
    return np.concatenate(
        [
            robot_obs[:6],
            robot_obs[6:7],    # state.pad = gripper width
            robot_obs[14:15],  # state.gripper = gripper action/state
        ],
        axis=0,
    ).astype(np.float32)


def _mat_to_6d(R: np.ndarray) -> np.ndarray:
    """Use UamVLA convention: first two columns, flattened as [col0, col1]."""
    R = np.asarray(R, dtype=np.float32)
    if R.shape == (4, 4):
        R = R[:3, :3]
    if R.shape != (3, 3):
        raise ValueError(f"Expected R shape (3,3) or (4,4), got {R.shape}")
    return R[:, :2].T.flatten().astype(np.float32)


def _zero_extra_state() -> dict[str, np.ndarray]:
    return {
        "state.target_pose_rot6d": np.zeros((6,), dtype=np.float32),
        "state.target_pose_trans": np.zeros((3,), dtype=np.float32),
        "state.static_cam_rot6d": np.zeros((6,), dtype=np.float32),
        "state.static_cam_trans": np.zeros((3,), dtype=np.float32),
    }


class CalvinEnvStateExtractor:
    """Extract target pose and static camera pose from calvin_env.

    This class does not write image sidecars and does not replace original images.
    It only computes:
      state.target_pose_rot6d
      state.target_pose_trans
      state.static_cam_rot6d
      state.static_cam_trans
    """

    def __init__(
        self,
        *,
        split_dir: Path,
        default_scene: str | None,
        on_resolve_failure: str,
        render_width: int,
        render_height: int,
    ) -> None:
        self.default_scene = default_scene
        self.on_resolve_failure = on_resolve_failure
        self.render_width = render_width
        self.render_height = render_height

        try:
            from tools.preprocess.calvin_env_adapter import make_calvin_env_adapter
            from tools.preprocess.calvin_task_map import (
                STACK_TASKS,
                infer_stack_block,
                resolve_target_object,
            )
        except ImportError as exc:
            raise ImportError(
                "--extra-state-mode env requires UniamVLA tools and calvin_env. "
                "Run this script inside the UniamVLA repo environment, or use "
                "--extra-state-mode lite for schema-only testing."
            ) from exc

        self.STACK_TASKS = STACK_TASKS
        self.infer_stack_block = infer_stack_block
        self.resolve_target_object = resolve_target_object
        self.env = make_calvin_env_adapter(dataset_path=str(split_dir))

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def resolve_target_object_id(
        self,
        *,
        task_label: str,
        start_scene_obs: np.ndarray,
        end_scene_obs: np.ndarray,
    ) -> str | None:
        if task_label in self.STACK_TASKS:
            if self.default_scene is None:
                if self.on_resolve_failure == "skip":
                    return None
                raise ValueError(
                    "Stack/unstack task needs --default-scene A/B/C/D. "
                    "For multiscene data, split by scene first."
                )
            return self.infer_stack_block(
                task_label,
                self.default_scene,
                start_scene_obs,
                end_scene_obs,
            )

        try:
            return self.resolve_target_object(task_label)
        except KeyError:
            if self.on_resolve_failure == "skip":
                return None
            raise

    def extract(
        self,
        *,
        robot_obs: np.ndarray,
        scene_obs: np.ndarray,
        target_object_id: str,
    ) -> dict[str, np.ndarray]:
        robot_obs = _as_1d_float32(robot_obs, 15, "robot_obs")
        scene_obs = np.asarray(scene_obs, dtype=np.float32)

        self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        rendered = self.env.render_cameras(
            width=self.render_width,
            height=self.render_height,
        )
        obj_pos, obj_R = self.env.get_object_pose(target_object_id)

        return {
            "state.target_pose_rot6d": _mat_to_6d(obj_R),
            "state.target_pose_trans": _as_1d_float32(
                obj_pos,
                3,
                "state.target_pose_trans",
            ),
            "state.static_cam_rot6d": _mat_to_6d(rendered["static_cam_R"]),
            "state.static_cam_trans": _as_1d_float32(
                rendered["static_cam_t"],
                3,
                "state.static_cam_trans",
            ),
        }


def _get_features(state_format: str, extra_state_mode: str) -> dict:
    state_dim = 8 if state_format == "starvla8" else 15

    features = {
        "image": {
            "dtype": "video",
            "shape": (200, 200, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "video",
            "shape": (84, 84, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }

    if extra_state_mode != "none":
        features.update(
            {
                "state.target_pose_rot6d": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["state.target_pose_rot6d"],
                },
                "state.target_pose_trans": {
                    "dtype": "float32",
                    "shape": (3,),
                    "names": ["state.target_pose_trans"],
                },
                "state.static_cam_rot6d": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["state.static_cam_rot6d"],
                },
                "state.static_cam_trans": {
                    "dtype": "float32",
                    "shape": (3,),
                    "names": ["state.static_cam_trans"],
                },
            }
        )

    return features


def _write_modality_json(dataset_path: Path, extra_state_mode: str) -> None:
    modality = json.loads(json.dumps(MERGED_MODALITY_JSON))
    if extra_state_mode == "none":
        for key in [
            "target_pose_rot6d",
            "target_pose_trans",
            "static_cam_rot6d",
            "static_cam_trans",
        ]:
            modality["state"].pop(key, None)

    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "modality.json").write_text(
        json.dumps(modality, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _default_repo_id(input_dir: Path) -> str:
    p = input_dir.resolve()
    if p.name in {"training", "validation"}:
        return f"{p.parent.name}_{p.name}_starvla_uam_state"
    return f"{p.name}_starvla_uam_state"


def main(args: Args) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if args.state_format != "starvla8":
        warnings.warn(
            "state_format='raw15' does not match the StarVLA-first 8D modality "
            "mapping unless you edit meta/modality.json accordingly.",
            RuntimeWarning,
        )

    if args.extra_state_mode == "lite":
        warnings.warn(
            "--extra-state-mode lite writes zero target/camera fields. "
            "Use only for schema smoke tests, not real training.",
            RuntimeWarning,
        )

    split_dirs = _resolve_split_dirs(input_dir, args.splits)

    repo_id = args.repo_id or _default_repo_id(input_dir)
    dataset_path = Path(args.output_root) / repo_id

    if dataset_path.exists():
        if args.overwrite:
            shutil.rmtree(dataset_path)
        else:
            raise FileExistsError(
                f"Output dataset already exists: {dataset_path}. "
                "Use --overwrite true to replace it."
            )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_path,
        robot_type=args.robot_type,
        fps=args.fps,
        features=_get_features(args.state_format, args.extra_state_mode),
    )

    total_saved_episodes = 0

    for split_name, split_dir in split_dirs:
        ranges, instructions, task_labels = _load_language_annotations(split_dir)

        extractor: CalvinEnvStateExtractor | None = None
        if args.extra_state_mode == "env":
            extractor = CalvinEnvStateExtractor(
                split_dir=split_dir,
                default_scene=args.default_scene,
                on_resolve_failure=args.on_resolve_failure,
                render_width=args.render_width,
                render_height=args.render_height,
            )

        try:
            for episode_idx, (start_idx, end_idx) in enumerate(ranges):
                start_idx = int(start_idx)
                end_idx = int(end_idx)

                instruction = _normalize_task(instructions[episode_idx])
                task_label = (
                    str(task_labels[episode_idx])
                    if task_labels is not None
                    else instruction
                )

                target_object_id: str | None = None
                if extractor is not None:
                    start_step = _load_step_npz(split_dir, start_idx)
                    end_step = _load_step_npz(split_dir, end_idx)

                    target_object_id = extractor.resolve_target_object_id(
                        task_label=task_label,
                        start_scene_obs=_as_1d_float32(
                            start_step["scene_obs"],
                            24,
                            "start scene_obs",
                        ),
                        end_scene_obs=_as_1d_float32(
                            end_step["scene_obs"],
                            24,
                            "end scene_obs",
                        ),
                    )

                    if target_object_id is None:
                        print(
                            f"[{split_name}] Skipping episode {episode_idx}: "
                            f"unresolved target for task_label={task_label!r}"
                        )
                        continue

                frames: list[dict] = []
                drop_episode = False

                for step_id in range(start_idx, end_idx + 1):
                    step = _load_step_npz(split_dir, step_id)

                    robot_obs = _as_1d_float32(step["robot_obs"], 15, "robot_obs")
                    action = _as_1d_float32(step[args.action_key], 7, args.action_key)

                    frame = {
                        "image": np.asarray(step["rgb_static"], dtype=np.uint8),
                        "wrist_image": np.asarray(step["rgb_gripper"], dtype=np.uint8),
                        "state": _build_starvla_state(robot_obs, args.state_format),
                        "actions": action,
                    }

                    if args.extra_state_mode == "lite":
                        frame.update(_zero_extra_state())

                    elif args.extra_state_mode == "env":
                        assert extractor is not None
                        assert target_object_id is not None
                        try:
                            frame.update(
                                extractor.extract(
                                    robot_obs=robot_obs,
                                    scene_obs=step["scene_obs"],
                                    target_object_id=target_object_id,
                                )
                            )
                        except Exception as exc:
                            if args.on_missing_target == "skip":
                                print(
                                    f"[{split_name}] Skipping episode {episode_idx}: "
                                    f"failed extra state at frame {step_id}: {exc}"
                                )
                                drop_episode = True
                                break
                            raise

                    frames.append(frame)

                if drop_episode:
                    continue

                for frame in frames:
                    dataset.add_frame(frame, task=instruction)
                dataset.save_episode()

                total_saved_episodes += 1
                if (
                    args.max_episodes is not None
                    and total_saved_episodes >= args.max_episodes
                ):
                    break

            if (
                args.max_episodes is not None
                and total_saved_episodes >= args.max_episodes
            ):
                break

        finally:
            if extractor is not None:
                extractor.close()

    if args.write_modality_json:
        _write_modality_json(dataset_path, args.extra_state_mode)

    print("Conversion finished.")
    print(f"Input dir: {input_dir}")
    print("Resolved splits:")
    for split_name, split_dir in split_dirs:
        print(f"  {split_name}: {split_dir}")
    print(f"Output dataset: {dataset_path}")
    print(f"Saved episodes: {total_saved_episodes}")
    print("Schema:")
    print("  StarVLA primary fields: image, wrist_image, state, actions")
    if args.extra_state_mode != "none":
        print(
            "  Extra state fields: state.target_pose_rot6d, "
            "state.target_pose_trans, state.static_cam_rot6d, state.static_cam_trans"
        )
    print("  meta/modality.json: StarVLA-first merged mapping")


if __name__ == "__main__":
    main(tyro.cli(Args))
