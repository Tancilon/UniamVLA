from __future__ import annotations

from pathlib import Path

from runners.preprocess_libero import build_suite_jobs, default_output_name, parse_args
from tools.preprocess.libero_preprocessor import _TaskReplayWorker


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
            "--affordance-local-window-size",
            "4",
            "--affordance-action-chunk-horizon",
            "8",
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
    assert args.affordance_local_window_size == 4
    assert args.affordance_action_chunk_horizon == 8
    assert args.overwrite is True


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
