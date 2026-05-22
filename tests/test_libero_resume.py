from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.preprocess.libero_resume import (
    DoneMarker,
    PlanMismatchError,
    PreprocessOptions,
    TaskFailureRecord,
    build_preprocess_plan,
    done_marker_path,
    failed_tasks_path,
    load_done_markers,
    load_or_create_plan,
    planned_episode_paths,
    write_done_marker,
    write_failed_tasks,
)


def test_load_or_create_plan_writes_stable_episode_indices(tmp_path: Path):
    output_dir = tmp_path / "lerobot_libero_goal"
    frame_counts = {
        "task_a.hdf5": [3, 2],
        "task_b.hdf5": [4],
    }
    options = PreprocessOptions(
        suite="libero_goal",
        min_segment_len=3,
        active_target_score_window=8,
        max_tasks=None,
        max_demos_per_task=None,
        max_frames_per_demo=None,
    )

    plan = load_or_create_plan(output_dir, frame_counts, options, resume=False)

    assert plan.suite == "libero_goal"
    assert plan.frame_counts == frame_counts
    assert plan.episode_plan["task_a.hdf5"]["0"]["episode_index"] == 0
    assert plan.episode_plan["task_a.hdf5"]["1"]["row_start"] == 3
    assert (output_dir / "meta/preprocess_plan.json").exists()


def test_load_or_create_plan_rejects_mismatch_on_resume(tmp_path: Path):
    output_dir = tmp_path / "lerobot_libero_goal"
    options = PreprocessOptions(
        suite="libero_goal",
        min_segment_len=3,
        active_target_score_window=8,
        max_tasks=None,
        max_demos_per_task=None,
        max_frames_per_demo=None,
    )
    load_or_create_plan(output_dir, {"task_a.hdf5": [3]}, options, resume=False)

    with pytest.raises(PlanMismatchError, match="frame_counts"):
        load_or_create_plan(output_dir, {"task_a.hdf5": [4]}, options, resume=True)


def test_done_marker_round_trip(tmp_path: Path):
    marker = DoneMarker(
        task_stem="task_a",
        task_filename="task_a.hdf5",
        task_name="task a",
        task_index=0,
        demo_indices=[0, 1],
        episode_indices=[0, 1],
        episode_lengths={"0": 3, "1": 2},
        episode_to_task={"0": 0, "1": 0},
        frame_count=5,
        coverage={"depth": {"valid": 5, "total": 5}},
        camera_params={"width": 256},
        elapsed_sec=12.5,
        frames_per_sec=0.4,
        retry_count=1,
        options_hash="abc",
    )

    write_done_marker(tmp_path, marker)

    assert done_marker_path(tmp_path, "task_a").exists()
    loaded = load_done_markers(tmp_path)
    assert loaded["task_a"].frame_count == 5
    assert loaded["task_a"].retry_count == 1


def test_write_failed_tasks_replaces_report(tmp_path: Path):
    records = [
        TaskFailureRecord(
            task_stem="task_a",
            task_filename="task_a.hdf5",
            task_name="task a",
            exception_type="RuntimeError",
            message="boom",
            traceback="RuntimeError: boom",
            retry_count=1,
            gpu_id="0",
            worker_pid=123,
        )
    ]

    write_failed_tasks(tmp_path, records)

    payload = json.loads(failed_tasks_path(tmp_path).read_text())
    assert payload["failed_count"] == 1
    assert payload["failed_tasks"][0]["message"] == "boom"


def test_planned_episode_paths_for_force_task_cleanup(tmp_path: Path):
    frame_counts = {"task_a.hdf5": [3, 2]}
    options = PreprocessOptions(
        suite="libero_goal",
        min_segment_len=3,
        active_target_score_window=8,
        max_tasks=None,
        max_demos_per_task=None,
        max_frames_per_demo=None,
    )
    plan = build_preprocess_plan(frame_counts, options)

    paths = planned_episode_paths(tmp_path, plan, "task_a.hdf5")

    assert tmp_path / "data/chunk-000/episode_000000.parquet" in paths
    assert tmp_path / "videos/chunk-000/video.primary_image/episode_000001.mp4" in paths
    assert tmp_path / "point_clouds/1" in paths
