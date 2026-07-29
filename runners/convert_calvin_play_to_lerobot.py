#!/usr/bin/env python3
"""
Convert raw CALVIN play episodes to LeRobot format for DT pretrain.

Unlike convert_calvin_dir_to_lerobot_starvla_with_uam_state.py which uses
language-annotated windows (auto_lang_ann.npy, ~17K short episodes), this
script uses ep_start_end_ids.npy — the raw, unsegmented play episodes
(147 long trajectories, ~1.79M total steps) — matching Seer's pretrain setup.

Key differences from the language-annotated converter:
  - Episode source: ep_start_end_ids.npy (not lang_annotations/)
  - Language annotation: empty string "" for every episode
  - No calvin_env needed: no target-pose or camera-extrinsic columns
  - Schema: StarVLA-8D state + 7D action + 2 video views only
  - Faster: ~30-90 min single-threaded (disk I/O limited)

Output is compatible with CalvinDT_K14DataConfig / CalvinDT_K10DataConfig
which only reads state.x/y/z/... (8D) + actions (7D) + 2 video views.

Usage
-----
# Single dataset from task_ABC_D (all scenes mixed):
python runners/convert_calvin_play_to_lerobot.py \\
    --input-dir datasets/calvin/task_ABC_D \\
    --output-root datasets \\
    --repo-id task_ABC_D_play_lerobot

# Limit to first N episodes for smoke-test:
python runners/convert_calvin_play_to_lerobot.py \\
    --input-dir datasets/calvin/task_ABC_D \\
    --output-root datasets \\
    --repo-id task_ABC_D_play_lerobot \\
    --max-episodes 5

# Override action key to absolute actions:
python runners/convert_calvin_play_to_lerobot.py \\
    --input-dir datasets/calvin/task_ABC_D \\
    --output-root datasets \\
    --repo-id task_ABC_D_play_lerobot \\
    --action-key actions
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

# Make project root importable when this script is placed at runners/.
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


# ---------------------------------------------------------------------------
# modality.json — StarVLA-8D state only (no target_pose / static_cam)
# This matches CalvinABCLeRobotV21H8DataConfig / CalvinDT_K10/K14DataConfig.
# ---------------------------------------------------------------------------
PLAY_MODALITY_JSON = {
    "state": {
        "x":       {"start": 0, "end": 1, "original_key": "state"},
        "y":       {"start": 1, "end": 2, "original_key": "state"},
        "z":       {"start": 2, "end": 3, "original_key": "state"},
        "roll":    {"start": 3, "end": 4, "original_key": "state"},
        "pitch":   {"start": 4, "end": 5, "original_key": "state"},
        "yaw":     {"start": 5, "end": 6, "original_key": "state"},
        "pad":     {"start": 6, "end": 7, "original_key": "state"},
        "gripper": {"start": 7, "end": 8, "original_key": "state"},
    },
    "action": {
        "x":       {"start": 0, "end": 1, "original_key": "actions"},
        "y":       {"start": 1, "end": 2, "original_key": "actions"},
        "z":       {"start": 2, "end": 3, "original_key": "actions"},
        "roll":    {"start": 3, "end": 4, "original_key": "actions"},
        "pitch":   {"start": 4, "end": 5, "original_key": "actions"},
        "yaw":     {"start": 5, "end": 6, "original_key": "actions"},
        "gripper": {"start": 6, "end": 7, "original_key": "actions"},
    },
    "video": {
        "primary_image": {"original_key": "image"},
        "wrist_image":   {"original_key": "wrist_image"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


@dataclass(frozen=True)
class Args:
    # Path to dataset root (containing training/ and validation/) or directly
    # to the training/ split directory.
    input_dir: str

    # Root directory where the output dataset will be created.
    output_root: str = "datasets"

    # LeRobot repo_id (and output directory name).
    # Defaults to "<input_dir_name>_play_lerobot".
    repo_id: str | None = None

    fps: int = 10

    # Which action key to use from the .npz files.
    action_key: Literal["rel_actions", "actions"] = "rel_actions"

    # Overwrite the output directory if it already exists.
    overwrite: bool = False

    # Limit total episodes written (for smoke-tests).
    max_episodes: int | None = None

    robot_type: str = "panda"

    # Splits to convert. Normally only "training" since validation/ has no
    # scene_info.npy and much fewer steps; set "both" to include it.
    splits: Literal["training", "validation", "both"] = "training"

    # Log progress every N episodes.
    log_interval: int = 10

    # Placeholder language string for all episodes.
    # Cannot be empty — the StarVLA dataloader skips trajectories with empty
    # language instructions. This value is never used for gradient computation
    # when loss_scale.vla: 0.0 (pretrain mode).
    language: str = "play"

    # Resume a previously interrupted conversion.
    # Reads <output_dir>/_checkpoint.json to find the last completed episode
    # index, then continues from the next episode without re-doing prior work.
    # Incompatible with --overwrite (they are mutually exclusive).
    resume: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_split_dir(path: Path) -> bool:
    return (
        (path / "ep_start_end_ids.npy").exists()
        and any(path.glob("episode_*.npz"))
    )


def _resolve_split_dirs(
    input_dir: Path,
    splits: Literal["training", "validation", "both"],
) -> list[tuple[str, Path]]:
    input_dir = input_dir.resolve()

    if _is_split_dir(input_dir):
        split_name = input_dir.name if input_dir.name in {"training", "validation"} else "training"
        return [(split_name, input_dir)]

    requested = ["training", "validation"] if splits == "both" else [splits]
    out: list[tuple[str, Path]] = []
    for split in requested:
        split_dir = input_dir / split
        if not _is_split_dir(split_dir):
            raise FileNotFoundError(
                f"No valid CALVIN split dir found at {split_dir}. "
                "Expected ep_start_end_ids.npy and episode_*.npz files."
            )
        out.append((split, split_dir))
    return out


def _load_step_npz(split_dir: Path, step_id: int) -> dict[str, np.ndarray]:
    path = split_dir / f"episode_{step_id:07d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing CALVIN step file: {path}")
    npz = np.load(path, allow_pickle=True)
    try:
        return {k: npz[k] for k in npz.files}
    finally:
        npz.close()


def _build_starvla_state(robot_obs: np.ndarray) -> np.ndarray:
    """Map 15D CALVIN robot_obs → 8D StarVLA state.

    CALVIN 15D layout: eef_xyz(3) | eef_rpy(3) | gripper_width(1) | joints(7) | gripper_action(1)
    StarVLA 8D layout: x y z roll pitch yaw pad gripper
    """
    robot_obs = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
    return np.concatenate([
        robot_obs[:6],       # x, y, z, roll, pitch, yaw
        robot_obs[6:7],      # pad = gripper_width
        robot_obs[14:15],    # gripper = gripper_action
    ]).astype(np.float32)


def _get_features() -> dict:
    return {
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
            "shape": (8,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }


_CHECKPOINT_FILENAME = "_checkpoint.json"


def _write_checkpoint(dataset_path: Path, last_ep_i: int, total_episodes: int, total_frames: int) -> None:
    """Persist progress after each successfully saved episode."""
    cp = {
        "last_completed_ep_i": last_ep_i,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
    }
    (dataset_path / _CHECKPOINT_FILENAME).write_text(
        json.dumps(cp, indent=2), encoding="utf-8"
    )


def _read_checkpoint(dataset_path: Path) -> dict | None:
    """Return checkpoint dict, or None if none exists."""
    cp_path = dataset_path / _CHECKPOINT_FILENAME
    if not cp_path.exists():
        return None
    return json.loads(cp_path.read_text(encoding="utf-8"))


def _open_dataset_for_append(dataset_path: Path, repo_id: str, fps: int) -> "LeRobotDataset":
    """Re-open an existing LeRobot v3 dataset for continued writing.

    LeRobotDataset.create() refuses to open an existing directory
    (exist_ok=False).  This function bypasses that by loading the existing
    metadata through the read-mode __init__ path, then wiring up the
    write-mode attributes that add_frame / save_episode need.
    """
    try:
        from lerobot.common.datasets.lerobot_dataset import (
            LeRobotDataset, LeRobotDatasetMetadata,
        )
    except ImportError:
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset, LeRobotDatasetMetadata,
        )

    # Load existing metadata (read path — no mkdir, no overwrite).
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_path)

    # Construct write-mode object without calling create().
    obj = LeRobotDataset.__new__(LeRobotDataset)
    obj.meta = meta
    obj.repo_id = meta.repo_id
    obj.root = meta.root
    obj.revision = None
    obj.tolerance_s = 1e-4
    obj.image_writer = None
    obj.batch_encoding_size = 1
    obj.episodes_since_last_encoding = 0
    obj.vcodec = "libsvtav1"
    obj._encoder_threads = None
    obj._streaming_encoder = None
    obj.episodes = None
    obj.image_transforms = None
    obj.delta_timestamps = None
    obj.delta_indices = None
    obj._absolute_to_relative_idx = None
    obj._lazy_loading = False
    obj._recorded_frames = 0
    obj._writer_closed_for_reading = False
    obj.writer = None
    obj.latest_episode = None
    obj._current_file_start_frame = None

    # hf_dataset starts empty; add_frame appends to episode_buffer only.
    obj.hf_dataset = obj.create_hf_dataset()
    obj.episode_buffer = obj.create_episode_buffer()

    try:
        from lerobot.datasets.lerobot_dataset import get_safe_default_codec
        obj.video_backend = get_safe_default_codec()
    except ImportError:
        obj.video_backend = "torchvision_av"

    return obj


def _write_modality_json(dataset_path: Path) -> None:
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "modality.json").write_text(
        json.dumps(PLAY_MODALITY_JSON, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _default_repo_id(input_dir: Path) -> str:
    p = input_dir.resolve()
    if p.name in {"training", "validation"}:
        return f"{p.parent.name}_play_lerobot"
    return f"{p.name}_play_lerobot"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args) -> None:
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive.")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    split_dirs = _resolve_split_dirs(input_dir, args.splits)

    repo_id = args.repo_id or _default_repo_id(input_dir)
    dataset_path = Path(args.output_root) / repo_id

    # ------------------------------------------------------------------
    # Dataset creation or resume
    # ------------------------------------------------------------------
    skip_ep_i: int = -1   # episodes with ep_i <= skip_ep_i are already done

    if args.resume:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"--resume requested but output dataset not found: {dataset_path}"
            )
        cp = _read_checkpoint(dataset_path)
        if cp is None:
            raise FileNotFoundError(
                f"--resume requested but no checkpoint found at "
                f"{dataset_path / _CHECKPOINT_FILENAME}. "
                "Run without --resume to start from scratch."
            )
        skip_ep_i = int(cp["last_completed_ep_i"])
        total_episodes = int(cp["total_episodes"])
        total_frames = int(cp["total_frames"])
        print(
            f"Resuming from checkpoint: last_completed_ep_i={skip_ep_i}, "
            f"total_episodes={total_episodes}, total_frames={total_frames}"
        )
        dataset = _open_dataset_for_append(dataset_path, repo_id, args.fps)

    else:
        if dataset_path.exists():
            if args.overwrite:
                print(f"Removing existing dataset at {dataset_path}")
                shutil.rmtree(dataset_path)
            else:
                raise FileExistsError(
                    f"Output dataset already exists: {dataset_path}. "
                    "Use --overwrite to replace it, or --resume to continue."
                )
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=dataset_path,
            robot_type=args.robot_type,
            fps=args.fps,
            features=_get_features(),
        )
        total_episodes = 0
        total_frames = 0

    # ------------------------------------------------------------------
    # Conversion loop
    # ------------------------------------------------------------------
    for split_name, split_dir in split_dirs:
        ep_ids = np.load(split_dir / "ep_start_end_ids.npy", allow_pickle=True)
        num_episodes = len(ep_ids)
        print(f"\n[{split_name}] {num_episodes} play episodes found in {split_dir}")

        for ep_i, (start_idx, end_idx) in enumerate(ep_ids):
            # Skip episodes already completed in a prior run.
            if ep_i <= skip_ep_i:
                continue

            start_idx = int(start_idx)
            end_idx = int(end_idx)
            ep_len = end_idx - start_idx + 1

            if ep_i % args.log_interval == 0:
                print(
                    f"  [{split_name}] episode {ep_i}/{num_episodes}  "
                    f"steps {start_idx}–{end_idx} (len={ep_len})"
                )

            for step_id in range(start_idx, end_idx + 1):
                step = _load_step_npz(split_dir, step_id)

                frame = {
                    "image":       np.asarray(step["rgb_static"],  dtype=np.uint8),
                    "wrist_image": np.asarray(step["rgb_gripper"], dtype=np.uint8),
                    "state":       _build_starvla_state(step["robot_obs"]),
                    "actions":     np.asarray(step[args.action_key], dtype=np.float32).reshape(-1),
                    # Non-empty placeholder so the StarVLA dataloader does not
                    # filter this trajectory out. Language is not used during
                    # pretrain (loss_scale.vla: 0.0 in pretrain config).
                    "task":        args.language,
                }
                dataset.add_frame(frame)

            dataset.save_episode()
            total_frames += ep_len
            total_episodes += 1

            # Persist progress immediately after save so an interrupt here
            # loses at most the current episode's write time.
            _write_checkpoint(dataset_path, ep_i, total_episodes, total_frames)

            if args.max_episodes is not None and total_episodes >= args.max_episodes:
                print(f"  Reached max_episodes={args.max_episodes}, stopping.")
                break

        if args.max_episodes is not None and total_episodes >= args.max_episodes:
            break

    _write_modality_json(dataset_path)

    print("\n" + "=" * 60)
    print("Conversion finished.")
    print(f"  Input:      {input_dir}")
    print(f"  Output:     {dataset_path}")
    print(f"  Episodes:   {total_episodes}")
    print(f"  Frames:     {total_frames}")
    print(f"  Action key: {args.action_key}")
    print(
        "\nRegister in data_registry/data_config.py as:\n"
        f'    "calvin_play_dt_k14": [\n'
        f'        ("{repo_id}", 1.0, "calvin_dt_k14"),\n'
        f'    ]'
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
