import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.preprocess.calvin_lerobot_merger import (
    CalvinLeRobotMergeError,
    ExistingOutputError,
    SceneDatasetMismatchError,
    compute_lerobot_stats,
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

        image_target_dir = scene_root / "image_targets"
        image_target_dir.mkdir(parents=True, exist_ok=True)
        (image_target_dir / f"{local_episode_id}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n"
        )

        point_cloud_dir = scene_root / "point_clouds" / str(local_episode_id)
        point_cloud_dir.mkdir(parents=True, exist_ok=True)
        for offset in range(length):
            np.save(point_cloud_dir / f"{offset}.npy", np.zeros((1, 3)))

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
            "splits": {"train": f"0:{len(episode_lengths)}"},
            "fps": 15,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
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
    assert info["splits"] == {"train": "0:4"}
    assert info["chunks_size"] == 1000
    assert (
        info["data_path"]
        == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    assert (
        info["video_path"]
        == "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    assert "video.primary_image" in info["features"]
    assert "video.wrist_image" in info["features"]
    assert json.loads((out_dir / "meta" / "modality.json").read_text()) == {
        "state": {},
        "action": {},
    }

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

    episodes_stats = [
        json.loads(line)
        for line in (out_dir / "meta" / "episodes_stats.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["episode_index"] for row in episodes_stats] == [0, 1, 2, 3]
    assert [row["stats"] for row in episodes_stats] == [
        {"action": {"mean": [0.0] * 7, "std": [1.0] * 7}},
        {"action": {"mean": [0.0] * 7, "std": [1.0] * 7}},
        {"action": {"mean": [0.0] * 7, "std": [1.0] * 7}},
        {"action": {"mean": [0.0] * 7, "std": [1.0] * 7}},
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
    assert [
        frame["frame_index"].tolist()
        for _, frame in frames.groupby("episode_index", sort=True)
    ] == [[0, 1], [0, 1, 2], [0], [0, 1]]

    videos = sorted((out_dir / "videos").glob("*/*/*.mp4"))
    assert [path.relative_to(out_dir).as_posix() for path in videos] == [
        f"videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"
        for video_key in ("video.primary_image", "video.wrist_image")
        for episode_index in range(4)
    ]

    image_targets = sorted((out_dir / "image_targets").glob("*.png"))
    assert [path.name for path in image_targets] == ["0.png", "1.png", "2.png", "3.png"]

    point_cloud_dirs = sorted((out_dir / "point_clouds").iterdir())
    assert [path.name for path in point_cloud_dirs] == ["0", "1", "2", "3"]
    point_cloud_files = [
        [path.name for path in sorted(point_cloud_dir.glob("*.npy"))]
        for point_cloud_dir in point_cloud_dirs
    ]
    assert point_cloud_files == [
        ["0.npy", "1.npy"],
        ["0.npy", "1.npy", "2.npy"],
        ["0.npy"],
        ["0.npy", "1.npy"],
    ]

    assert json.loads((out_dir / "camera_params.json").read_text()) == {"static": 1}


def test_merge_deduplicates_tasks_first_seen(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=10,
        task_names=["open drawer"],
        episode_lengths=[1],
    )
    scene_b = _write_scene_dataset(
        tmp_path,
        "B",
        episode_start=50,
        task_names=["open drawer", "push block"],
        episode_lengths=[1, 1],
    )
    out_dir = tmp_path / "lerobot_calvin_ab"

    merge_lerobot_scene_outputs(
        [scene_a, scene_b],
        out_dir,
        overwrite=False,
        skip_stats=True,
    )

    tasks = [
        json.loads(line)
        for line in (out_dir / "meta" / "tasks.jsonl").read_text().splitlines()
    ]
    assert tasks == [
        {"task_index": 0, "task": "open drawer"},
        {"task_index": 1, "task": "push block"},
    ]

    parquet_paths = sorted((out_dir / "data").glob("*/*.parquet"))
    frames = pd.concat(pd.read_parquet(path) for path in parquet_paths)
    task_ids_by_episode = [
        frame["task_index"].unique().tolist()
        for _, frame in frames.groupby("episode_index", sort=True)
    ]
    assert task_ids_by_episode == [[0], [0], [1]]


def test_merge_generates_stats_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=0,
        task_names=["open drawer"],
        episode_lengths=[1],
    )
    out_dir = tmp_path / "merged"
    calls = []

    def fake_compute_lerobot_stats(
        dataset_dir: Path,
        *,
        robot_type: str,
        action_mode: str,
    ) -> None:
        calls.append((dataset_dir, robot_type, action_mode))
        _write_json(dataset_dir / "meta" / "stats_gr00t.json", {"marker": "stats"})

    monkeypatch.setattr(
        "tools.preprocess.calvin_lerobot_merger.compute_lerobot_stats",
        fake_compute_lerobot_stats,
    )

    merge_lerobot_scene_outputs(
        [scene_a],
        out_dir,
        overwrite=False,
        skip_stats=False,
        robot_type="uamvla_calvin_franka",
        action_mode="delta",
    )

    assert calls == [(out_dir, "uamvla_calvin_franka", "delta")]
    assert json.loads((out_dir / "meta" / "stats_gr00t.json").read_text()) == {
        "marker": "stats"
    }


def test_merge_skip_stats_does_not_generate_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=0,
        task_names=["open drawer"],
        episode_lengths=[1],
    )
    out_dir = tmp_path / "merged"
    calls = []

    def fake_compute_lerobot_stats(
        dataset_dir: Path,
        *,
        robot_type: str,
        action_mode: str,
    ) -> None:
        calls.append((dataset_dir, robot_type, action_mode))
        _write_json(dataset_dir / "meta" / "stats_gr00t.json", {"marker": "stats"})

    monkeypatch.setattr(
        "tools.preprocess.calvin_lerobot_merger.compute_lerobot_stats",
        fake_compute_lerobot_stats,
    )

    merge_lerobot_scene_outputs(
        [scene_a],
        out_dir,
        overwrite=False,
        skip_stats=True,
        robot_type="uamvla_calvin_franka",
        action_mode="delta",
    )

    assert calls == []
    assert not (out_dir / "meta" / "stats_gr00t.json").exists()


def test_cli_parses_skip_stats_and_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.preprocess.calvin_lerobot_merger import main

    scene_a = tmp_path / "lerobot_calvin_A"
    out_dir = tmp_path / "merged"
    calls = []

    def fake_merge_lerobot_scene_outputs(
        scene_dirs: list[Path],
        output_dir: Path,
        *,
        overwrite: bool,
        skip_stats: bool,
        robot_type: str,
        action_mode: str,
    ) -> None:
        calls.append(
            {
                "scene_dirs": scene_dirs,
                "output_dir": output_dir,
                "overwrite": overwrite,
                "skip_stats": skip_stats,
                "robot_type": robot_type,
                "action_mode": action_mode,
            }
        )

    monkeypatch.setattr(
        "tools.preprocess.calvin_lerobot_merger.merge_lerobot_scene_outputs",
        fake_merge_lerobot_scene_outputs,
    )

    main(
        [
            "--scene-dir",
            str(scene_a),
            "--output-dir",
            str(out_dir),
            "--overwrite",
            "--skip-stats",
            "--robot-type",
            "uamvla_calvin_franka",
            "--action-mode",
            "delta",
        ]
    )

    assert calls == [
        {
            "scene_dirs": [scene_a],
            "output_dir": out_dir,
            "overwrite": True,
            "skip_stats": True,
            "robot_type": "uamvla_calvin_franka",
            "action_mode": "delta",
        }
    ]


def test_cli_accepts_multiple_scene_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.preprocess.calvin_lerobot_merger import main

    scene_a = tmp_path / "lerobot_calvin_A"
    scene_b = tmp_path / "lerobot_calvin_B"
    out_dir = tmp_path / "merged"
    calls = []

    def fake_merge_lerobot_scene_outputs(
        scene_dirs: list[Path],
        output_dir: Path,
        *,
        overwrite: bool,
        skip_stats: bool,
        robot_type: str,
        action_mode: str,
    ) -> None:
        calls.append((scene_dirs, output_dir, overwrite, skip_stats, robot_type, action_mode))

    monkeypatch.setattr(
        "tools.preprocess.calvin_lerobot_merger.merge_lerobot_scene_outputs",
        fake_merge_lerobot_scene_outputs,
    )

    main(
        [
            "--scene-dir",
            str(scene_a),
            "--scene-dir",
            str(scene_b),
            "--output-dir",
            str(out_dir),
        ]
    )

    assert calls == [
        (
            [scene_a, scene_b],
            out_dir,
            False,
            False,
            "uamvla_calvin_franka",
            "abs",
        )
    ]


def test_cli_accepts_underscore_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.preprocess.calvin_lerobot_merger import main

    scene_a = tmp_path / "lerobot_calvin_A"
    out_dir = tmp_path / "merged"
    calls = []

    def fake_merge_lerobot_scene_outputs(
        scene_dirs: list[Path],
        output_dir: Path,
        *,
        overwrite: bool,
        skip_stats: bool,
        robot_type: str,
        action_mode: str,
    ) -> None:
        calls.append(
            {
                "scene_dirs": scene_dirs,
                "output_dir": output_dir,
                "overwrite": overwrite,
                "skip_stats": skip_stats,
                "robot_type": robot_type,
                "action_mode": action_mode,
            }
        )

    monkeypatch.setattr(
        "tools.preprocess.calvin_lerobot_merger.merge_lerobot_scene_outputs",
        fake_merge_lerobot_scene_outputs,
    )

    main(
        [
            "--scene_dir",
            str(scene_a),
            "--output_dir",
            str(out_dir),
            "--overwrite",
            "--skip_stats",
            "--robot_type",
            "fake_robot",
            "--action_mode",
            "raw",
        ]
    )

    assert calls == [
        {
            "scene_dirs": [scene_a],
            "output_dir": out_dir,
            "overwrite": True,
            "skip_stats": True,
            "robot_type": "fake_robot",
            "action_mode": "raw",
        }
    ]


def test_compute_lerobot_stats_normalizes_raw_action_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeDataConfig:
        def modality_config(self) -> str:
            return "fake_modality_config"

        def transform(self) -> str:
            return "fake_transform"

    def fake_lerobot_single_dataset(**kwargs: object) -> None:
        calls.append(kwargs)

    fake_datasets = types.ModuleType("starVLA.dataloader.gr00t_lerobot.datasets")
    fake_datasets.LeRobotSingleDataset = fake_lerobot_single_dataset
    fake_registry = types.ModuleType("starVLA.dataloader.gr00t_lerobot.registry")
    fake_registry.ROBOT_TYPE_CONFIG_MAP = {"fake_robot": FakeDataConfig()}
    fake_registry.ROBOT_TYPE_TO_EMBODIMENT_TAG = {"fake_robot": "fake_embodiment"}
    monkeypatch.setitem(
        sys.modules,
        "starVLA.dataloader.gr00t_lerobot.datasets",
        fake_datasets,
    )
    monkeypatch.setitem(
        sys.modules,
        "starVLA.dataloader.gr00t_lerobot.registry",
        fake_registry,
    )

    compute_lerobot_stats(tmp_path, robot_type="fake_robot", action_mode="raw")

    assert calls == [
        {
            "dataset_path": tmp_path,
            "modality_configs": "fake_modality_config",
            "embodiment_tag": "fake_embodiment",
            "transforms": "fake_transform",
            "data_cfg": {"action_mode": "abs"},
        }
    ]


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


def test_merge_rejects_output_that_overlaps_scene_input(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=0,
        task_names=["open drawer"],
        episode_lengths=[1],
    )

    with pytest.raises(CalvinLeRobotMergeError):
        merge_lerobot_scene_outputs(
            [scene_a],
            scene_a,
            overwrite=True,
            skip_stats=True,
        )

    assert (scene_a / "meta" / "info.json").exists()


def test_merge_rejects_missing_episode_parquet(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=0,
        task_names=["open drawer"],
        episode_lengths=[1, 1],
    )
    (scene_a / "data" / "chunk-000" / "episode_000001.parquet").unlink()

    with pytest.raises(CalvinLeRobotMergeError):
        merge_lerobot_scene_outputs(
            [scene_a],
            tmp_path / "merged",
            overwrite=False,
            skip_stats=True,
        )


def test_merge_rejects_parquet_row_count_mismatch(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=0,
        task_names=["open drawer"],
        episode_lengths=[2],
    )
    parquet_path = scene_a / "data" / "chunk-000" / "episode_000000.parquet"
    pd.read_parquet(parquet_path).iloc[:1].to_parquet(parquet_path)

    with pytest.raises(CalvinLeRobotMergeError):
        merge_lerobot_scene_outputs(
            [scene_a],
            tmp_path / "merged",
            overwrite=False,
            skip_stats=True,
        )


def test_merge_rejects_missing_video(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=0,
        task_names=["open drawer"],
        episode_lengths=[1],
    )
    (
        scene_a
        / "videos"
        / "chunk-000"
        / "video.primary_image"
        / "episode_000000.mp4"
    ).unlink()

    with pytest.raises(CalvinLeRobotMergeError):
        merge_lerobot_scene_outputs(
            [scene_a],
            tmp_path / "merged",
            overwrite=False,
            skip_stats=True,
        )


def test_merge_rejects_missing_image_target(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=0,
        task_names=["open drawer"],
        episode_lengths=[1],
    )
    (scene_a / "image_targets" / "0.png").unlink()

    with pytest.raises(CalvinLeRobotMergeError):
        merge_lerobot_scene_outputs(
            [scene_a],
            tmp_path / "merged",
            overwrite=False,
            skip_stats=True,
        )


def test_merge_rejects_missing_point_cloud(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=0,
        task_names=["open drawer"],
        episode_lengths=[2],
    )
    (scene_a / "point_clouds" / "0" / "1.npy").unlink()

    with pytest.raises(CalvinLeRobotMergeError):
        merge_lerobot_scene_outputs(
            [scene_a],
            tmp_path / "merged",
            overwrite=False,
            skip_stats=True,
        )


def test_merge_rejects_mismatched_info_schema(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=0,
        task_names=["open drawer"],
        episode_lengths=[1],
    )
    scene_b = _write_scene_dataset(
        tmp_path,
        "B",
        episode_start=10,
        task_names=["close drawer"],
        episode_lengths=[1],
    )
    info_path = scene_b / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["fps"] = 30
    _write_json(info_path, info)

    with pytest.raises(SceneDatasetMismatchError):
        merge_lerobot_scene_outputs(
            [scene_a, scene_b],
            tmp_path / "merged",
            overwrite=False,
            skip_stats=True,
        )
