import json
from pathlib import Path

import pandas as pd
import pytest

from tools.preprocess.calvin_lerobot_merger import (
    ExistingOutputError,
    SceneDatasetMismatchError,
    merge_lerobot_scene_outputs,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_scene_dataset(
    root: Path,
    scene: str,
    *,
    episode_start: int,
    task_names: list[str],
    episode_lengths: list[int],
    camera_params: dict | None = None,
) -> Path:
    scene_root = root / f"lerobot_calvin_{scene}"
    scene_root.mkdir(parents=True)

    total_frames = sum(episode_lengths)
    episode_rows = []
    frame_cursor = 0
    for local_episode, length in enumerate(episode_lengths):
        local_episode_id = episode_start + local_episode
        task_index = local_episode % len(task_names)
        episode_rows.append(
            {
                "episode_index": local_episode_id,
                "tasks": [task_names[task_index]],
                "length": length,
            }
        )
        frame_rows = [
            {
                "episode_index": local_episode_id,
                "trajectory_id": local_episode_id,
                "frame_index": frame_cursor + offset,
                "base_index": offset,
                "timestamp": float(offset) / 30.0,
                "task_index": task_index,
                "index": frame_cursor + offset,
                "action": [float(offset), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            }
            for offset in range(length)
        ]
        data_dir = scene_root / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(frame_rows).to_parquet(
            data_dir / f"episode_{local_episode_id:06d}.parquet"
        )
        frame_cursor += length

    for video_key in ("video.primary_image", "video.wrist_image"):
        video_dir = scene_root / "videos" / "chunk-000" / video_key
        video_dir.mkdir(parents=True)
        for row in episode_rows:
            (video_dir / f"episode_{row['episode_index']:06d}.mp4").write_bytes(
                b"fake mp4 bytes"
            )

    _write_json(
        scene_root / "meta" / "info.json",
        {
            "codebase_version": "v2.0",
            "robot_type": "uamvla_calvin_franka",
            "total_episodes": len(episode_lengths),
            "total_frames": total_frames,
            "total_tasks": len(task_names),
            "total_videos": len(episode_lengths) * 2,
            "splits": {"train": f"0:{len(episode_lengths)}"},
            "fps": 30,
            "features": {
                "video.primary_image": {"dtype": "video"},
                "video.wrist_image": {"dtype": "video"},
            },
        },
    )
    _write_jsonl(scene_root / "meta" / "episodes.jsonl", episode_rows)
    _write_jsonl(
        scene_root / "meta" / "tasks.jsonl",
        [{"task_index": i, "task": name} for i, name in enumerate(task_names)],
    )
    _write_jsonl(
        scene_root / "meta" / "episodes_stats.jsonl",
        [
            {
                "episode_index": row["episode_index"],
                "stats": {"action": {"mean": [0.0] * 7, "std": [1.0] * 7}},
            }
            for row in episode_rows
        ],
    )
    _write_json(scene_root / "meta" / "modality.json", {"state": {}, "action": {}})
    _write_json(scene_root / "camera_params.json", camera_params or {"static": 1})
    return scene_root


def test_merge_renumbers_parquet_rows_meta_and_videos(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=10,
        task_names=["open drawer"],
        episode_lengths=[2, 3],
    )
    scene_b = _write_scene_dataset(
        tmp_path,
        "B",
        episode_start=50,
        task_names=["close drawer", "push block"],
        episode_lengths=[1, 2],
    )
    out_dir = tmp_path / "lerobot_calvin_abcd"

    result = merge_lerobot_scene_outputs(
        [scene_a, scene_b],
        out_dir,
        overwrite=False,
        skip_stats=True,
    )

    assert result.output_dir == out_dir
    info = json.loads((out_dir / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 4
    assert info["total_frames"] == 8
    assert info["total_tasks"] == 3
    assert info["total_videos"] == 8
    assert info["splits"] == {"train": "0:4"}

    episodes = [
        json.loads(line)
        for line in (out_dir / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert [row["episode_index"] for row in episodes] == [0, 1, 2, 3]
    assert [row["length"] for row in episodes] == [2, 3, 1, 2]

    tasks = [
        json.loads(line)
        for line in (out_dir / "meta" / "tasks.jsonl").read_text().splitlines()
    ]
    assert tasks == [
        {"task_index": 0, "task": "open drawer"},
        {"task_index": 1, "task": "close drawer"},
        {"task_index": 2, "task": "push block"},
    ]

    parquet_paths = sorted((out_dir / "data").glob("*/*.parquet"))
    assert [path.name for path in parquet_paths] == [
        "episode_000000.parquet",
        "episode_000001.parquet",
        "episode_000002.parquet",
        "episode_000003.parquet",
    ]
    frames = pd.concat(pd.read_parquet(path) for path in parquet_paths)
    assert sorted(frames["episode_index"].unique().tolist()) == [0, 1, 2, 3]
    assert sorted(frames["task_index"].unique().tolist()) == [0, 1, 2]
    assert frames["index"].tolist() == list(range(8))
    assert [
        frame["trajectory_id"].unique().tolist()
        for _, frame in frames.groupby("episode_index", sort=True)
    ] == [[0], [1], [2], [3]]
    assert [
        frame["base_index"].tolist()
        for _, frame in frames.groupby("episode_index", sort=True)
    ] == [[0, 1], [0, 1, 2], [0], [0, 1]]

    videos = sorted((out_dir / "videos").glob("*/*/*.mp4"))
    assert len(videos) == 8
    assert (
        out_dir
        / "videos"
        / "chunk-000"
        / "video.primary_image"
        / "episode_000003.mp4"
    ).exists()
    assert json.loads((out_dir / "camera_params.json").read_text()) == {"static": 1}


def test_merge_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=0,
        task_names=["open drawer"],
        episode_lengths=[1],
    )
    out_dir = tmp_path / "existing"
    out_dir.mkdir()

    with pytest.raises(ExistingOutputError):
        merge_lerobot_scene_outputs([scene_a], out_dir, overwrite=False, skip_stats=True)


def test_merge_overwrite_replaces_existing_output(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=0,
        task_names=["open drawer"],
        episode_lengths=[1],
    )
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("old", encoding="utf-8")

    merge_lerobot_scene_outputs([scene_a], out_dir, overwrite=True, skip_stats=True)

    assert not (out_dir / "stale.txt").exists()
    assert (out_dir / "meta" / "info.json").exists()


def test_merge_rejects_mismatched_camera_params(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=0,
        task_names=["open drawer"],
        episode_lengths=[1],
        camera_params={"static": 1},
    )
    scene_b = _write_scene_dataset(
        tmp_path,
        "B",
        episode_start=0,
        task_names=["close drawer"],
        episode_lengths=[1],
        camera_params={"static": 2},
    )

    with pytest.raises(SceneDatasetMismatchError):
        merge_lerobot_scene_outputs(
            [scene_a, scene_b],
            tmp_path / "merged",
            overwrite=False,
            skip_stats=True,
        )
