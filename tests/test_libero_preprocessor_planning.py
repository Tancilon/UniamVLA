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
