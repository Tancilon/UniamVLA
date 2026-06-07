#!/usr/bin/env python3
"""
Convert a CALVIN zip dataset to a StarVLA-first LeRobot dataset, while adding
only the extra state columns needed by UamVLA-style CALVIN training.

Schema policy
-------------
Keep original StarVLA columns as the primary interface:
  image          : video, original CALVIN rgb_static
  wrist_image    : video, original CALVIN rgb_gripper
  state          : 8D StarVLA state by default
  actions        : 7D CALVIN action vector

Add only extra UamVLA state columns:
  state.robot_obs
  state.target_pose_rot6d
  state.target_pose_trans
  state.static_cam_rot6d
  state.static_cam_trans

Do NOT add:
  action.x/action.y/...      # action still maps from the original actions vector
  video.primary_image        # video still maps from original image/wrist_image

The written meta/modality.json is the merged StarVLA-first JSON:
  - original StarVLA state/action/video mapping remains intact
  - extra state.* mappings are appended

Real target/camera state requires calvin_env. Use --extra-state-mode env.
Use --extra-state-mode lite only for schema/debug smoke tests; it fills zeros
for target/camera fields and is not semantically valid for UamVLA training.

Example
-------
python examples/calvin/convert_to_lerobot_starvla_with_uam_state.py \
  --zip-path /mnt/data/jiangnan/calvin/task_D_D.zip \
  --output-root /mnt/data/jiangnan/lerobot \
  --repo-id task_D_D_starvla_uam_state \
  --splits training \
  --extra-state-mode env \
  --default-scene D \
  --env-split-dir /mnt/data/jiangnan/calvin/task_D_D/training \
  --overwrite true
"""

from __future__ import annotations

import json
import shutil
import tempfile
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
import tyro

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


# StarVLA-first merged modality.
# This should be copied to <dataset>/meta/modality.json.
MERGED_MODALITY_JSON = {
    "state": {
        # Original StarVLA 8D state vector.
        "x": {"start": 0, "end": 1, "original_key": "state"},
        "y": {"start": 1, "end": 2, "original_key": "state"},
        "z": {"start": 2, "end": 3, "original_key": "state"},
        "roll": {"start": 3, "end": 4, "original_key": "state"},
        "pitch": {"start": 4, "end": 5, "original_key": "state"},
        "yaw": {"start": 5, "end": 6, "original_key": "state"},
        "pad": {"start": 6, "end": 7, "original_key": "state"},
        "gripper": {"start": 7, "end": 8, "original_key": "state"},

        # Extra UamVLA state information.
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
        # LeRobotDataset.add_frame(..., task=instruction) writes task_index.
        "human.action.task_description": {"original_key": "task_index"},
    },
}


@dataclass(frozen=True)
class Args:
    zip_path: str

    output_root: str = "/mnt/data/jiangnan/lerobot"
    repo_id: str | None = None
    fps: int = 10
    splits: Literal["training", "validation", "both"] = "training"

    # StarVLA action vector. rel_actions is the usual CALVIN training target.
    action_key: Literal["rel_actions", "actions"] = "rel_actions"

    # Keep starvla8 for compatibility with the original StarVLA modality.
    state_format: Literal["starvla8", "raw15"] = "starvla8"

    robot_type: str = "panda"
    max_episodes: int | None = None
    overwrite: bool = False

    # env: real target/camera state via calvin_env
    # lite: schema-only zero target/camera state, useful for smoke tests only
    # none: no extra state columns, original StarVLA conversion
    extra_state_mode: Literal["env", "lite", "none"] = "env"

    # Required/recommended for --extra-state-mode env.
    # This should point to the extracted CALVIN split dir containing .hydra/.
    # Example: /path/to/task_D_D/training
    env_split_dir: str | None = None

    # Required for stack/unstack target resolution in env mode.
    # For task_D_D use D. For multiscene data, run per-scene conversion.
    default_scene: str | None = None

    on_resolve_failure: Literal["skip", "abort"] = "abort"
    on_missing_target: Literal["skip", "abort"] = "abort"

    # Only used by env mode for camera pose extraction; output videos still use
    # original rgb_static/rgb_gripper, not re-rendered images.
    render_width: int = 256
    render_height: int = 256

    # Write merged StarVLA-first modality.json.
    write_modality_json: bool = True


def _load_npy_from_zip(zf: zipfile.ZipFile, member: str):
    with zf.open(member, "r") as f:
        return np.load(f, allow_pickle=True)


def _infer_dataset_root(zf: zipfile.ZipFile) -> str:
    for member in zf.namelist():
        parts = PurePosixPath(member).parts
        if len(parts) >= 3 and parts[1] in {"training", "validation"}:
            return parts[0]
    raise ValueError("Failed to infer dataset root from zip contents.")


def _load_language_annotations(
    zf: zipfile.ZipFile,
    dataset_root: str,
    split: Literal["training", "validation"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    member = f"{dataset_root}/{split}/lang_annotations/auto_lang_ann.npy"
    lang_data = _load_npy_from_zip(zf, member).item()
    ranges = lang_data["info"]["indx"]
    instructions = lang_data["language"]["ann"]
    task_labels = lang_data["language"].get("task", None)
    return ranges, instructions, task_labels


def _load_step_npz(
    zf: zipfile.ZipFile,
    dataset_root: str,
    split: Literal["training", "validation"],
    step_id: int,
) -> dict[str, np.ndarray]:
    member = f"{dataset_root}/{split}/episode_{step_id:07d}.npz"
    with zf.open(member, "r") as f:
        npz = np.load(f, allow_pickle=True)
        try:
            return {k: npz[k] for k in npz.files}
        finally:
            npz.close()


def _as_1d_float32(x, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.shape[0] != dim:
        raise ValueError(f"{name} expected {dim} dims, got shape {arr.shape}")
    return arr


def _normalize_task(task) -> str:
    return str(task).strip().split("\n")[0]


def _build_starvla_state(robot_obs: np.ndarray, state_format: str) -> np.ndarray:
    robot_obs = _as_1d_float32(robot_obs, 15, "robot_obs")
    if state_format == "raw15":
        return robot_obs

    # CALVIN 15D proprioception:
    # [eef_xyz(3), eef_rpy(3), gripper_width(1), joint_pos(7), gripper_action(1)]
    #
    # StarVLA 8D state:
    # [x, y, z, roll, pitch, yaw, pad, gripper]
    return np.concatenate(
        [
            robot_obs[:6],
            robot_obs[6:7],    # state.pad keeps gripper width as extra proprio
            robot_obs[14:15],  # state.gripper aligns with action.gripper
        ],
        axis=0,
    ).astype(np.float32)


def _mat_to_6d(R: np.ndarray) -> np.ndarray:
    """Same convention as UamVLA: first two columns, concatenated column-wise."""
    R = np.asarray(R, dtype=np.float32)
    if R.shape == (4, 4):
        R = R[:3, :3]
    if R.shape != (3, 3):
        raise ValueError(f"Expected R shape (3,3) or (4,4), got {R.shape}")
    return R[:, :2].T.flatten().astype(np.float32)


def _zero_extra_state(robot_obs: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "state.target_pose_rot6d": np.zeros((6,), dtype=np.float32),
        "state.target_pose_trans": np.zeros((3,), dtype=np.float32),
        "state.static_cam_rot6d": np.zeros((6,), dtype=np.float32),
        "state.static_cam_trans": np.zeros((3,), dtype=np.float32),
    }


class CalvinEnvStateExtractor:
    """Extract only the extra UamVLA state fields from calvin_env.

    This intentionally does not re-render or write replacement image columns.
    It uses render_cameras only to obtain static_cam_R/static_cam_t.
    """

    def __init__(
        self,
        *,
        dataset_path: Path,
        default_scene: str | None,
        on_resolve_failure: str,
        render_width: int,
        render_height: int,
    ) -> None:
        self.default_scene = default_scene
        self.on_resolve_failure = on_resolve_failure
        self.render_width = render_width
        self.render_height = render_height

        # Prefer the repo's official helper if available.
        try:
            from tools.preprocess.calvin_env_adapter import make_calvin_env_adapter
            from tools.preprocess.calvin_task_map import (
                STACK_TASKS,
                infer_stack_block,
                resolve_target_object,
            )

            self.STACK_TASKS = STACK_TASKS
            self.infer_stack_block = infer_stack_block
            self.resolve_target_object = resolve_target_object
            self.env = make_calvin_env_adapter(dataset_path=str(dataset_path))
        except ImportError as exc:
            raise ImportError(
                "extra_state_mode='env' requires UniamVLA tools and calvin_env. "
                "Run inside the repo environment or use --extra-state-mode lite "
                "for schema-only testing."
            ) from exc

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
                    "stack/unstack task needs --default-scene A/B/C/D. "
                    "For multi-scene data, split by scene first."
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
        # Original StarVLA fields.
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
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    modality = json.loads(json.dumps(MERGED_MODALITY_JSON))
    if extra_state_mode == "none":
        # Remove extra UamVLA state entries if requested.
        for key in [
            "robot_obs",
            "target_pose_rot6d",
            "target_pose_trans",
            "static_cam_rot6d",
            "static_cam_trans",
        ]:
            modality["state"].pop(key, None)

    (meta_dir / "modality.json").write_text(
        json.dumps(modality, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _extract_minimal_env_split_from_zip(
    *,
    zf: zipfile.ZipFile,
    dataset_root: str,
    split: Literal["training", "validation"],
) -> tempfile.TemporaryDirectory[str]:
    """Best-effort fallback for calvin_env when --env-split-dir is not provided.

    Prefer passing --env-split-dir explicitly. calvin_env mainly needs .hydra/.
    Different CALVIN dumps may have slightly different config layouts.
    """
    tmp = tempfile.TemporaryDirectory(prefix="calvin_env_split_")
    tmp_path = Path(tmp.name)

    prefix = f"{dataset_root}/{split}/"
    for member in zf.namelist():
        if not member.startswith(prefix):
            continue
        rel = member[len(prefix):]
        if rel.startswith(".hydra/") or rel == "scene_info.npy":
            out = tmp_path / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    if not (tmp_path / ".hydra").exists():
        warnings.warn(
            "No .hydra directory was found inside the zip split. "
            "extra_state_mode='env' will likely fail unless --env-split-dir is set.",
            RuntimeWarning,
        )
    return tmp


def main(args: Args) -> None:
    if args.state_format != "starvla8":
        warnings.warn(
            "state_format='raw15' no longer matches the StarVLA-first "
            "state.x/y/... mapping in MERGED_MODALITY_JSON. Use starvla8 unless "
            "you also edit modality.json.",
            RuntimeWarning,
        )

    if args.extra_state_mode == "lite":
        warnings.warn(
            "extra_state_mode='lite' writes zero target/camera fields. "
            "This is only for schema smoke tests, not real UamVLA training.",
            RuntimeWarning,
        )

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        dataset_root = _infer_dataset_root(zf)

        repo_id = args.repo_id or f"{dataset_root}_starvla_uam_state"
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

        splits: list[Literal["training", "validation"]]
        if args.splits == "both":
            splits = ["training", "validation"]
        else:
            splits = [args.splits]

        total_saved_episodes = 0

        for split in splits:
            ranges, instructions, task_labels = _load_language_annotations(
                zf,
                dataset_root,
                split,
            )

            env_tmp: tempfile.TemporaryDirectory[str] | None = None
            extractor: CalvinEnvStateExtractor | None = None

            if args.extra_state_mode == "env":
                env_split_dir = Path(args.env_split_dir) if args.env_split_dir else None
                if env_split_dir is None:
                    env_tmp = _extract_minimal_env_split_from_zip(
                        zf=zf,
                        dataset_root=dataset_root,
                        split=split,
                    )
                    env_split_dir = Path(env_tmp.name)

                extractor = CalvinEnvStateExtractor(
                    dataset_path=env_split_dir,
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
                        start_step = _load_step_npz(zf, dataset_root, split, start_idx)
                        end_step = _load_step_npz(zf, dataset_root, split, end_idx)
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
                                f"Skipping episode {episode_idx}: unresolved target "
                                f"for task_label={task_label!r}"
                            )
                            continue

                    # Buffer whole episode first so skip-on-error never writes
                    # a partial LeRobot episode.
                    frames: list[dict] = []
                    drop_episode = False

                    for step_id in range(start_idx, end_idx + 1):
                        step = _load_step_npz(zf, dataset_root, split, step_id)

                        robot_obs = _as_1d_float32(step["robot_obs"], 15, "robot_obs")
                        action = _as_1d_float32(step[args.action_key], 7, args.action_key)

                        frame = {
                            # Original StarVLA fields.
                            "image": np.asarray(step["rgb_static"], dtype=np.uint8),
                            "wrist_image": np.asarray(step["rgb_gripper"], dtype=np.uint8),
                            "state": _build_starvla_state(robot_obs, args.state_format),
                            "actions": action,
                        }

                        if args.extra_state_mode == "lite":
                            frame.update(_zero_extra_state(robot_obs))

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
                                        f"Skipping episode {episode_idx}: "
                                        f"failed to extract extra state at frame "
                                        f"{step_id}: {exc}"
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
                if env_tmp is not None:
                    env_tmp.cleanup()

    if args.write_modality_json:
        _write_modality_json(dataset_path, args.extra_state_mode)

    print("Conversion finished.")
    print(f"Input zip: {zip_path}")
    print(f"Detected dataset root: {dataset_root}")
    print(f"Output dataset: {dataset_path}")
    print(f"Saved episodes: {total_saved_episodes}")
    print("Schema:")
    print("  StarVLA primary fields: image, wrist_image, state, actions")
    if args.extra_state_mode != "none":
        print(
            "  Extra state fields:  state.target_pose_rot6d, "
            "state.target_pose_trans, state.static_cam_rot6d, state.static_cam_trans"
        )
    print("  meta/modality.json: StarVLA-first merged mapping")


if __name__ == "__main__":
    main(tyro.cli(Args))
