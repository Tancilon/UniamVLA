"""Merge per-scene CALVIN LeRobot outputs into one dataset directory."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class MergeResult:
    output_dir: Path
    total_episodes: int
    total_frames: int
    total_tasks: int


class CalvinLeRobotMergeError(RuntimeError):
    pass


class ExistingOutputError(CalvinLeRobotMergeError):
    pass


class SceneDatasetMismatchError(CalvinLeRobotMergeError):
    pass


@dataclass(frozen=True)
class _SceneMeta:
    root: Path
    info: dict
    episodes: list[dict]
    tasks: list[dict]
    episodes_stats: list[dict]
    task_index_to_global: dict[int, int]
    episode_index_to_global: dict[int, int]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise ExistingOutputError(
                f"Output directory already exists: {output_dir}. "
                "Pass overwrite=True to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _check_output_does_not_overlap_sources(
    scene_dirs: Sequence[Path],
    output_dir: Path,
) -> None:
    output_resolved = output_dir.resolve(strict=False)
    for scene_dir in scene_dirs:
        scene_resolved = scene_dir.resolve(strict=False)
        output_inside_scene = _path_is_relative_to(output_resolved, scene_resolved)
        scene_inside_output = _path_is_relative_to(scene_resolved, output_resolved)
        if output_resolved == scene_resolved or output_inside_scene or scene_inside_output:
            raise CalvinLeRobotMergeError(
                f"Output directory overlaps scene input: output={output_dir}, "
                f"scene={scene_dir}"
            )


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _same_json_or_bytes(left: Path, right: Path) -> bool:
    left_bytes = left.read_bytes()
    right_bytes = right.read_bytes()
    if left_bytes == right_bytes:
        return True
    try:
        return json.loads(left_bytes) == json.loads(right_bytes)
    except json.JSONDecodeError:
        return False


def _validate_matching_file(scene_dirs: Sequence[Path], relative_path: Path) -> Path | None:
    first_path = scene_dirs[0] / relative_path
    if not first_path.exists():
        for scene_dir in scene_dirs[1:]:
            if (scene_dir / relative_path).exists():
                raise SceneDatasetMismatchError(
                    f"{relative_path.as_posix()} exists in {scene_dir} but not in "
                    f"{scene_dirs[0]}"
                )
        return None

    for scene_dir in scene_dirs[1:]:
        candidate = scene_dir / relative_path
        if not candidate.exists():
            raise SceneDatasetMismatchError(
                f"{relative_path.as_posix()} exists in {scene_dirs[0]} but not in "
                f"{scene_dir}"
            )
        if not _same_json_or_bytes(first_path, candidate):
            raise SceneDatasetMismatchError(
                f"Mismatched {relative_path.as_posix()}: {first_path} vs {candidate}"
            )
    return first_path


def _validate_matching_info_fields(scenes: Sequence[_SceneMeta]) -> None:
    stable_fields = (
        "codebase_version",
        "fps",
        "data_path",
        "video_path",
        "chunks_size",
        "features",
    )
    template = scenes[0].info
    for scene in scenes[1:]:
        for field in stable_fields:
            if scene.info.get(field) != template.get(field):
                raise SceneDatasetMismatchError(
                    f"Mismatched meta/info.json field {field!r}: "
                    f"{scenes[0].root} vs {scene.root}"
                )


def _copy_validated_file(
    scene_dirs: Sequence[Path],
    output_dir: Path,
    relative_path: Path,
) -> None:
    source = _validate_matching_file(scene_dirs, relative_path)
    if source is None:
        return
    destination = output_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _parse_episode_stem(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _extract_task_texts(row: dict) -> list[str]:
    tasks = row.get("tasks", [])
    if isinstance(tasks, str):
        return [tasks]
    return list(tasks)


def _episode_length(episode_row: dict) -> int:
    return int(episode_row["length"])


def _collect_scene_meta(scene_dirs: Sequence[Path]) -> tuple[list[_SceneMeta], list[dict]]:
    global_tasks: list[dict] = []
    task_text_to_global: dict[str, int] = {}
    global_episode_index = 0
    scenes: list[_SceneMeta] = []

    for scene_dir in scene_dirs:
        meta_dir = scene_dir / "meta"
        info = _read_json(meta_dir / "info.json")
        episodes = _read_jsonl(meta_dir / "episodes.jsonl")
        tasks = _read_jsonl(meta_dir / "tasks.jsonl")
        episodes_stats = _read_jsonl(meta_dir / "episodes_stats.jsonl")

        task_index_to_text: dict[int, str] = {}
        for task_row in tasks:
            task_index = int(task_row["task_index"])
            task_text = str(task_row["task"])
            task_index_to_text[task_index] = task_text
            if task_text not in task_text_to_global:
                task_text_to_global[task_text] = len(global_tasks)
                global_tasks.append(
                    {"task_index": task_text_to_global[task_text], "task": task_text}
                )

        for episode_row in episodes:
            for task_text in _extract_task_texts(episode_row):
                if task_text not in task_text_to_global:
                    task_text_to_global[task_text] = len(global_tasks)
                    global_tasks.append(
                        {
                            "task_index": task_text_to_global[task_text],
                            "task": task_text,
                        }
                    )

        task_index_to_global = {
            local_index: task_text_to_global[task_text]
            for local_index, task_text in task_index_to_text.items()
        }
        episode_index_to_global: dict[int, int] = {}
        for episode_row in episodes:
            local_episode = int(episode_row["episode_index"])
            episode_index_to_global[local_episode] = global_episode_index
            global_episode_index += 1

        scenes.append(
            _SceneMeta(
                root=scene_dir,
                info=info,
                episodes=episodes,
                tasks=tasks,
                episodes_stats=episodes_stats,
                task_index_to_global=task_index_to_global,
                episode_index_to_global=episode_index_to_global,
            )
        )

    return scenes, global_tasks


def _video_feature_keys(info: dict) -> list[str]:
    features = info.get("features", {})
    return [
        feature_key
        for feature_key, feature_spec in features.items()
        if feature_key.startswith("video.")
        or (
            isinstance(feature_spec, dict)
            and feature_spec.get("dtype") == "video"
        )
    ]


def _source_video_path(scene: _SceneMeta, video_key: str, local_episode: int) -> Path:
    matches = sorted(
        (scene.root / "videos").glob(
            f"chunk-*/{video_key}/episode_{local_episode:06d}.mp4"
        )
    )
    if not matches:
        raise CalvinLeRobotMergeError(
            f"Missing video for episode {local_episode}: "
            f"{scene.root}/videos/chunk-*/{video_key}/episode_{local_episode:06d}.mp4"
        )
    return matches[0]


def _copy_videos(scene: _SceneMeta, output_dir: Path, chunks_size: int) -> None:
    for episode_row in scene.episodes:
        local_episode = int(episode_row["episode_index"])
        global_episode = scene.episode_index_to_global[local_episode]
        chunk = _episode_chunk(global_episode, chunks_size)
        for video_key in _video_feature_keys(scene.info):
            src_path = _source_video_path(scene, video_key, local_episode)
            dst_path = (
                output_dir
                / "videos"
                / f"chunk-{chunk:03d}"
                / video_key
                / f"episode_{global_episode:06d}.mp4"
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)


def _copy_image_targets(scene: _SceneMeta, output_dir: Path) -> None:
    image_targets_dir = scene.root / "image_targets"
    for episode_row in scene.episodes:
        local_episode = int(episode_row["episode_index"])
        global_episode = scene.episode_index_to_global[local_episode]
        src_path = image_targets_dir / f"{local_episode}.png"
        if not src_path.exists():
            raise CalvinLeRobotMergeError(
                f"Missing image target for episode {local_episode}: {src_path}"
            )
        dst_path = output_dir / "image_targets" / f"{global_episode}.png"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def _copy_point_clouds(scene: _SceneMeta, output_dir: Path) -> None:
    point_clouds_dir = scene.root / "point_clouds"
    for episode_row in scene.episodes:
        local_episode = int(episode_row["episode_index"])
        global_episode = scene.episode_index_to_global[local_episode]
        src_dir = point_clouds_dir / str(local_episode)
        dst_dir = output_dir / "point_clouds" / str(global_episode)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for base_index in range(_episode_length(episode_row)):
            src_path = src_dir / f"{base_index}.npy"
            if not src_path.exists():
                raise CalvinLeRobotMergeError(
                    f"Missing point cloud for episode {local_episode}, "
                    f"frame {base_index}: {src_path}"
                )
            shutil.copy2(src_path, dst_dir / src_path.name)


def _read_scene_frames(scene: _SceneMeta) -> dict[int, list[pd.DataFrame]]:
    frames_by_episode: dict[int, list[pd.DataFrame]] = {
        local_episode: []
        for local_episode in scene.episode_index_to_global
    }
    data_dir = scene.root / "data"
    if not data_dir.exists():
        return frames_by_episode

    for src_path in sorted(data_dir.glob("chunk-*/*.parquet")):
        frame = pd.read_parquet(src_path)
        if "episode_index" in frame.columns:
            for local_episode, episode_frame in frame.groupby("episode_index", sort=False):
                local_episode = int(local_episode)
                if local_episode in frames_by_episode:
                    frames_by_episode[local_episode].append(episode_frame.copy())
        else:
            local_episode = _parse_episode_stem(src_path)
            if local_episode in frames_by_episode:
                frames_by_episode[local_episode].append(frame.copy())

    return frames_by_episode


def _validate_scene_assets(scenes: Sequence[_SceneMeta]) -> None:
    for scene in scenes:
        frames_by_episode = _read_scene_frames(scene)
        for episode_row in scene.episodes:
            local_episode = int(episode_row["episode_index"])
            expected_length = _episode_length(episode_row)

            episode_frames = frames_by_episode.get(local_episode, [])
            if not episode_frames:
                raise CalvinLeRobotMergeError(
                    f"Missing parquet rows for episode {local_episode} in {scene.root}"
                )
            frame_count = sum(len(frame) for frame in episode_frames)
            if frame_count != expected_length:
                raise CalvinLeRobotMergeError(
                    f"Episode {local_episode} in {scene.root} has {frame_count} "
                    f"parquet rows, expected {expected_length}"
                )

            for video_key in _video_feature_keys(scene.info):
                _source_video_path(scene, video_key, local_episode)

            image_target = scene.root / "image_targets" / f"{local_episode}.png"
            if not image_target.exists():
                raise CalvinLeRobotMergeError(
                    f"Missing image target for episode {local_episode}: "
                    f"{image_target}"
                )

            point_cloud_dir = scene.root / "point_clouds" / str(local_episode)
            for base_index in range(expected_length):
                point_cloud = point_cloud_dir / f"{base_index}.npy"
                if not point_cloud.exists():
                    raise CalvinLeRobotMergeError(
                        f"Missing point cloud for episode {local_episode}, "
                        f"frame {base_index}: {point_cloud}"
                    )


def _map_task_index(value: object, task_index_to_global: dict[int, int]) -> int:
    return task_index_to_global[int(value)]


def _rewrite_episode_frame(
    frame: pd.DataFrame,
    *,
    global_episode: int,
    task_index_to_global: dict[int, int],
    frame_start: int,
) -> pd.DataFrame:
    frame = frame.copy()
    length = len(frame)

    if "episode_index" in frame.columns:
        frame["episode_index"] = global_episode
    if "trajectory_id" in frame.columns:
        frame["trajectory_id"] = global_episode
    if "frame_index" in frame.columns:
        frame["frame_index"] = range(length)
    if "base_index" in frame.columns:
        frame["base_index"] = range(length)
    if "task_index" in frame.columns:
        frame["task_index"] = frame["task_index"].map(
            lambda value: _map_task_index(value, task_index_to_global)
        )
    if "index" in frame.columns:
        frame["index"] = range(frame_start, frame_start + length)

    return frame


def _rewrite_parquet_files(
    scene: _SceneMeta,
    output_dir: Path,
    chunks_size: int,
    frame_start: int,
) -> int:
    frames_by_episode = _read_scene_frames(scene)
    frame_cursor = frame_start

    for episode_row in scene.episodes:
        local_episode = int(episode_row["episode_index"])
        global_episode = scene.episode_index_to_global[local_episode]
        episode_frames = frames_by_episode.get(local_episode, [])
        if not episode_frames:
            raise CalvinLeRobotMergeError(
                f"Missing parquet rows for episode {local_episode} in {scene.root}"
            )

        frame = pd.concat(episode_frames, ignore_index=True)
        expected_length = _episode_length(episode_row)
        if len(frame) != expected_length:
            raise CalvinLeRobotMergeError(
                f"Episode {local_episode} in {scene.root} has {len(frame)} "
                f"parquet rows, expected {expected_length}"
            )

        frame = _rewrite_episode_frame(
            frame,
            global_episode=global_episode,
            task_index_to_global=scene.task_index_to_global,
            frame_start=frame_cursor,
        )
        frame_cursor += len(frame)

        chunk = _episode_chunk(global_episode, chunks_size)
        dst_path = (
            output_dir
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{global_episode:06d}.parquet"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(dst_path, index=False)

    return frame_cursor - frame_start


def _merged_episodes(scenes: Sequence[_SceneMeta]) -> list[dict]:
    rows: list[dict] = []
    for scene in scenes:
        for episode_row in scene.episodes:
            local_episode = int(episode_row["episode_index"])
            row = dict(episode_row)
            row["episode_index"] = scene.episode_index_to_global[local_episode]
            rows.append(row)
    return rows


def _merged_episode_stats(scenes: Sequence[_SceneMeta]) -> list[dict]:
    rows: list[dict] = []
    for scene in scenes:
        for stats_row in scene.episodes_stats:
            local_episode = int(stats_row["episode_index"])
            if local_episode not in scene.episode_index_to_global:
                continue
            row = dict(stats_row)
            row["episode_index"] = scene.episode_index_to_global[local_episode]
            rows.append(row)
    return rows


def _merged_info(
    template: dict,
    *,
    robot_type: str,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> dict:
    info = dict(template)
    info["robot_type"] = robot_type
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = total_tasks
    info["splits"] = {"train": f"0:{total_episodes}"}
    info.pop("total_videos", None)
    return info


def compute_lerobot_stats(
    dataset_dir: Path,
    *,
    robot_type: str = "uamvla_calvin_franka",
    action_mode: str = "abs",
) -> None:
    """Build GR00T/LeRobot stats using the StarVLA dataset loader."""
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
    from starVLA.dataloader.gr00t_lerobot.registry import (
        ROBOT_TYPE_CONFIG_MAP,
        ROBOT_TYPE_TO_EMBODIMENT_TAG,
    )

    if robot_type not in ROBOT_TYPE_CONFIG_MAP:
        raise CalvinLeRobotMergeError(
            f"Unknown robot_type {robot_type!r}; not found in ROBOT_TYPE_CONFIG_MAP."
        )
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        raise CalvinLeRobotMergeError(
            f"Unknown robot_type {robot_type!r}; not found in "
            "ROBOT_TYPE_TO_EMBODIMENT_TAG."
        )

    normalized_action_mode = str(action_mode).lower()
    action_mode_aliases = {
        "absolute": "abs",
        "raw": "abs",
        "delta_qpos": "delta",
        "relative": "rel",
    }
    normalized_action_mode = action_mode_aliases.get(
        normalized_action_mode,
        normalized_action_mode,
    )
    if normalized_action_mode not in {"abs", "delta", "rel"}:
        raise CalvinLeRobotMergeError(
            f"Unsupported action_mode {action_mode!r}; expected one of "
            "'abs', 'absolute', 'raw', 'delta', 'delta_qpos', 'relative', or 'rel'."
        )

    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    LeRobotSingleDataset(
        dataset_path=dataset_dir,
        modality_configs=data_config.modality_config(),
        embodiment_tag=ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type],
        transforms=data_config.transform(),
        data_cfg={"action_mode": normalized_action_mode},
    )


def merge_lerobot_scene_outputs(
    scene_dirs: Sequence[Path | str],
    output_dir: Path | str,
    *,
    overwrite: bool = False,
    skip_stats: bool = False,
    robot_type: str = "uamvla_calvin_franka",
    action_mode: str = "abs",
) -> MergeResult:
    """Merge per-scene LeRobot datasets into one CALVIN LeRobot dataset."""
    scene_paths = [Path(scene_dir) for scene_dir in scene_dirs]
    if not scene_paths:
        raise ValueError("merge_lerobot_scene_outputs requires at least one scene dir.")

    output_path = Path(output_dir)
    _check_output_does_not_overlap_sources(scene_paths, output_path)
    if output_path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Output directory already exists: {output_path}. "
            "Pass overwrite=True to replace it."
        )

    scenes, global_tasks = _collect_scene_meta(scene_paths)
    _validate_matching_info_fields(scenes)
    _validate_matching_file(scene_paths, Path("camera_params.json"))
    _validate_matching_file(scene_paths, Path("meta") / "modality.json")
    _validate_scene_assets(scenes)

    _prepare_output_dir(output_path, overwrite)
    _copy_validated_file(scene_paths, output_path, Path("camera_params.json"))
    _copy_validated_file(scene_paths, output_path, Path("meta") / "modality.json")

    chunks_size = int(scenes[0].info.get("chunks_size", 1000))
    total_frames = 0
    for scene in scenes:
        total_frames += _rewrite_parquet_files(
            scene,
            output_path,
            chunks_size,
            total_frames,
        )
        _copy_videos(scene, output_path, chunks_size)
        _copy_image_targets(scene, output_path)
        _copy_point_clouds(scene, output_path)

    total_episodes = sum(len(scene.episodes) for scene in scenes)
    total_tasks = len(global_tasks)

    _write_jsonl(output_path / "meta" / "tasks.jsonl", global_tasks)
    _write_jsonl(output_path / "meta" / "episodes.jsonl", _merged_episodes(scenes))
    _write_jsonl(
        output_path / "meta" / "episodes_stats.jsonl",
        _merged_episode_stats(scenes),
    )
    _write_json(
        output_path / "meta" / "info.json",
        _merged_info(
            scenes[0].info,
            robot_type=robot_type,
            total_episodes=total_episodes,
            total_frames=total_frames,
            total_tasks=total_tasks,
        ),
    )

    if not skip_stats:
        compute_lerobot_stats(
            output_path,
            robot_type=robot_type,
            action_mode=action_mode,
        )

    return MergeResult(
        output_dir=output_path,
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=total_tasks,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge per-scene CALVIN LeRobot outputs into one dataset.",
    )
    parser.add_argument(
        "--scene-dir",
        "--scene_dir",
        action="append",
        required=True,
        type=Path,
        dest="scene_dir",
        help="Input scene dataset directory. Repeat for multiple scenes.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        required=True,
        type=Path,
        dest="output_dir",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    parser.add_argument(
        "--skip-stats",
        "--skip_stats",
        action="store_true",
        dest="skip_stats",
        help="Skip GR00T/LeRobot stats generation after merging.",
    )
    parser.add_argument(
        "--robot-type",
        "--robot_type",
        default="uamvla_calvin_franka",
        dest="robot_type",
        help="Robot type key used for merged metadata and stats.",
    )
    parser.add_argument(
        "--action-mode",
        "--action_mode",
        default="abs",
        dest="action_mode",
        help=(
            "Action stats mode: abs, absolute, raw, delta, delta_qpos, relative, "
            "or rel."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    merge_lerobot_scene_outputs(
        args.scene_dir,
        args.output_dir,
        overwrite=args.overwrite,
        skip_stats=args.skip_stats,
        robot_type=args.robot_type,
        action_mode=args.action_mode,
    )


if __name__ == "__main__":
    main()
