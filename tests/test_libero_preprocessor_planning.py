from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from runners.preprocess_libero import build_suite_jobs, default_output_name, parse_args
from tools.preprocess.libero_preprocessor import _TaskReplayWorker
from tools.preprocess.libero_target_mapping import PartGroundingSpec


def test_default_output_name():
    assert default_output_name("libero_spatial") == "lerobot_libero_spatial"
    assert default_output_name("libero_10") == "lerobot_libero_10"


def test_build_suite_jobs_for_all_suites(tmp_path: Path):
    input_root = tmp_path / "libero"
    output_root = tmp_path / "libero2uam"
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        (input_root / suite).mkdir(parents=True)
    jobs = build_suite_jobs(input_root, output_root, "all")
    assert [(j.suite, j.input_dir.name, j.output_dir.name) for j in jobs] == [
        ("libero_spatial", "libero_spatial", "lerobot_libero_spatial"),
        ("libero_object", "libero_object", "lerobot_libero_object"),
        ("libero_goal", "libero_goal", "lerobot_libero_goal"),
        ("libero_10", "libero_10", "lerobot_libero_10"),
    ]


def test_parse_args_accepts_parallel_options():
    args = parse_args(
        [
            "--input-root",
            "datasets/libero",
            "--output-root",
            "datasets/libero2uam",
            "--suite",
            "libero_goal",
            "--num-workers",
            "2",
            "--render-gpus",
            "0,1",
            "--max-tasks",
            "1",
            "--max-demos-per-task",
            "2",
            "--max-frames-per-demo",
            "3",
            "--active-target-score-window",
            "6",
            "--overwrite",
        ]
    )
    assert args.input_root == "datasets/libero"
    assert args.output_root == "datasets/libero2uam"
    assert args.suite == "libero_goal"
    assert args.num_workers == 2
    assert args.render_gpus == "0,1"
    assert args.max_tasks == 1
    assert args.max_demos_per_task == 2
    assert args.max_frames_per_demo == 3
    assert args.active_target_score_window == 6
    assert args.overwrite is True


def test_parse_args_accepts_resume_throughput_options():
    args = parse_args(
        [
            "--input-root",
            "datasets/libero",
            "--output-root",
            "datasets/libero2uam",
            "--suite",
            "libero_10",
            "--num-workers",
            "4",
            "--render-gpus",
            "0",
            "--resume",
            "--force-task",
            "task_a",
            "--force-task",
            "task_b",
            "--fail-fast",
            "--profile",
            "--max-retries",
            "1",
        ]
    )
    assert args.resume is True
    assert args.force_task == ["task_a", "task_b"]
    assert args.fail_fast is True
    assert args.profile is True
    assert args.max_retries == 1


def test_parse_args_rejects_overwrite_with_resume():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input-root",
                "datasets/libero",
                "--output-root",
                "datasets/libero2uam",
                "--suite",
                "libero_goal",
                "--overwrite",
                "--resume",
            ]
        )


def test_libero_preprocessor_stores_resume_options():
    from tools.preprocess.libero_preprocessor import LiberoPreprocessor

    pre = LiberoPreprocessor(
        suite="libero_10",
        resume=True,
        force_tasks=("task_a",),
        fail_fast=True,
        profile=True,
        max_retries=2,
    )

    assert pre.resume is True
    assert pre.force_tasks == {"task_a"}
    assert pre.fail_fast is True
    assert pre.profile is True
    assert pre.max_retries == 2


def test_resume_filters_done_tasks(tmp_path: Path):
    from tools.preprocess.libero_preprocessor import LiberoPreprocessor
    from tools.preprocess.libero_resume import DoneMarker, write_done_marker

    input_dir = tmp_path / "raw" / "libero_goal"
    output_dir = tmp_path / "out" / "lerobot_libero_goal"
    input_dir.mkdir(parents=True)
    for name in ["task_a.hdf5", "task_b.hdf5"]:
        (input_dir / name).write_bytes(b"hdf5 test bytes")

    marker = DoneMarker(
        task_stem="task_a",
        task_filename="task_a.hdf5",
        task_name="task a",
        task_index=0,
        demo_indices=[0],
        episode_indices=[0],
        episode_lengths={"0": 3},
        episode_to_task={"0": 0},
        frame_count=3,
        coverage={},
        camera_params=None,
        elapsed_sec=1.0,
        frames_per_sec=3.0,
        retry_count=0,
        options_hash="abc",
    )
    write_done_marker(output_dir, marker)

    pre = LiberoPreprocessor(suite="libero_goal", resume=True)
    jobs = [
        SimpleNamespace(hdf5_path=input_dir / "task_a.hdf5"),
        SimpleNamespace(hdf5_path=input_dir / "task_b.hdf5"),
    ]

    remaining = pre._filter_resume_jobs(output_dir, jobs, {"task_a": marker})

    assert [job.hdf5_path.name for job in remaining] == ["task_b.hdf5"]


def test_run_task_job_with_retries_reports_failure(monkeypatch):
    from tools.preprocess.libero_preprocessor import _run_task_job_with_retries

    job = SimpleNamespace(
        hdf5_path=Path("bad_task_demo.hdf5"),
        gpu_id="0",
    )

    def boom(_job):
        raise RuntimeError("bad target")

    monkeypatch.setattr("tools.preprocess.libero_preprocessor._run_task_job", boom)

    result = _run_task_job_with_retries(job, max_retries=1)

    assert result["ok"] is False
    assert result["retry_count"] == 1
    assert result["task_filename"] == "bad_task_demo.hdf5"
    assert result["exception_type"] == "RuntimeError"
    assert "bad target" in result["message"]


def test_replay_worker_maps_main_body_to_instance_id():
    worker = object.__new__(_TaskReplayWorker)

    class Model:
        instances_to_ids = {
            "robot0": {},
            "akita_black_bowl_1": {},
            "plate_1": {},
        }

    class InnerEnv:
        model = Model()

    class Env:
        env = InnerEnv()

    worker.env = Env()
    assert worker._instance_name_for_body("akita_black_bowl_1_main") == "akita_black_bowl_1"
    assert worker._instance_id_for_body("akita_black_bowl_1_main") == 2


def test_grounding_falls_back_to_object_mask_when_active_body_is_not_part_body():
    worker = object.__new__(_TaskReplayWorker)
    worker.policy = SimpleNamespace(
        part_grounding=PartGroundingSpec(
            enabled=True,
            body_patterns=("wooden_cabinet",),
            geom_patterns=("drawer",),
            grounding_level="part",
        )
    )
    worker._part_mask = lambda seg_geom, active_body: np.ones((8, 8), dtype=np.uint8)
    object_mask = np.zeros((8, 8), dtype=np.uint8)
    object_mask[2:4, 2:4] = 1

    mask, level = worker._grounding_mask_for_frame(
        {"seg_geom": np.zeros((8, 8), dtype=np.int32)},
        active_body="akita_black_bowl_1_main",
        object_mask=object_mask,
    )

    assert level == "object"
    assert mask.shape == (1, 20, 20)
    assert float(mask.mean()) < 1.0


def test_grounding_uses_part_mask_when_active_body_matches_part_body():
    worker = object.__new__(_TaskReplayWorker)
    worker.policy = SimpleNamespace(
        part_grounding=PartGroundingSpec(
            enabled=True,
            body_patterns=("wooden_cabinet",),
            geom_patterns=("drawer",),
            grounding_level="part",
        )
    )
    worker._part_mask = lambda seg_geom, active_body: np.ones((8, 8), dtype=np.uint8)
    object_mask = np.zeros((8, 8), dtype=np.uint8)
    object_mask[2:4, 2:4] = 1

    mask, level = worker._grounding_mask_for_frame(
        {"seg_geom": np.zeros((8, 8), dtype=np.int32)},
        active_body="wooden_cabinet_1_main",
        object_mask=object_mask,
    )

    assert level == "part"
    assert mask.shape == (1, 20, 20)
    assert np.isclose(float(mask.mean()), 1.0)
