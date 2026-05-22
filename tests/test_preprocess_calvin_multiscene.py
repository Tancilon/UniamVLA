import argparse

import pytest

import runners.preprocess_calvin_multiscene as runner


def test_runner_rejects_existing_output_without_overwrite(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    output_dir.mkdir()
    work_dir.mkdir()

    called = []

    def fail_if_called(*args, **kwargs):
        called.append((args, kwargs))
        pytest.fail("runner should reject existing output_dir before doing work")

    monkeypatch.setattr(runner, "split_calvin_by_scene", fail_if_called)
    monkeypatch.setattr(runner, "_run_preprocessor", fail_if_called)
    monkeypatch.setattr(
        runner,
        "merge_lerobot_scene_outputs",
        fail_if_called,
        raising=False,
    )

    with pytest.raises(SystemExit):
        runner.main(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                "--work_dir",
                str(work_dir),
                "--scenes",
                "A,B",
                "--skip_stats",
                "--robot_type",
                "uamvla_calvin_franka",
                "--action_mode",
                "delta",
            ]
        )

    assert called == []


def test_runner_rejects_clean_work_that_contains_output(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "work" / "final"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    work_dir.mkdir()

    called = []

    def fail_if_called(*args, **kwargs):
        called.append((args, kwargs))
        pytest.fail("runner should reject output_dir inside cleaned work_dir")

    monkeypatch.setattr(runner, "split_calvin_by_scene", fail_if_called)
    monkeypatch.setattr(runner, "_run_preprocessor", fail_if_called)
    monkeypatch.setattr(
        runner,
        "merge_lerobot_scene_outputs",
        fail_if_called,
        raising=False,
    )

    with pytest.raises(SystemExit, match="output_dir.*work_dir"):
        runner.main(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                "--work_dir",
                str(work_dir),
                "--scenes",
                "A,B",
                "--clean_work",
            ]
        )

    assert called == []


def test_runner_clears_scene_outputs_and_calls_lerobot_merger(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    work_dir.mkdir()

    stale_scene_output = work_dir / "preprocessed" / "A"
    stale_scene_output.mkdir(parents=True)
    (stale_scene_output / "stale.txt").write_text("old shard\n")
    stale_split_file = work_dir / "split" / "stale.txt"
    stale_split_file.parent.mkdir(parents=True)
    stale_split_file.write_text("old split\n")

    split_calls = []
    preprocess_calls = []
    merge_calls = []

    def fake_split(*, input_dir, output_dir, scenes, scene_config_dir=None):
        scene_tuple = tuple(scenes)
        split_calls.append((input_dir, output_dir, scene_tuple, scene_config_dir))
        scene_inputs = {}
        for scene in scene_tuple:
            scene_input_dir = output_dir / scene
            scene_input_dir.mkdir(parents=True)
            scene_inputs[scene] = scene_input_dir
        return scene_inputs

    def fake_preprocess(scene, scene_input_dir, scene_output_dir, args):
        assert not (scene_output_dir / "stale.txt").exists()
        preprocess_calls.append((scene, scene_input_dir, scene_output_dir, args))
        scene_output_dir.mkdir(parents=True, exist_ok=True)

    def fake_merge(
        scene_dirs,
        output_dir,
        *,
        overwrite,
        skip_stats,
        robot_type,
        action_mode,
    ):
        merge_calls.append(
            (
                list(scene_dirs),
                output_dir,
                overwrite,
                skip_stats,
                robot_type,
                action_mode,
            )
        )

    monkeypatch.setattr(runner, "split_calvin_by_scene", fake_split)
    monkeypatch.setattr(runner, "_run_preprocessor", fake_preprocess)
    monkeypatch.setattr(runner, "merge_lerobot_scene_outputs", fake_merge, raising=False)

    runner.main(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--work_dir",
            str(work_dir),
            "--scenes",
            "A,B",
            "--overwrite",
            "--skip_stats",
            "--robot_type",
            "uamvla_calvin_franka",
            "--action_mode",
            "delta",
        ]
    )

    assert split_calls == [
        (input_dir, work_dir / "split", ("A", "B"), None),
    ]
    assert not stale_split_file.exists()
    assert [(scene, scene_output_dir) for scene, _, scene_output_dir, _ in preprocess_calls] == [
        ("A", work_dir / "preprocessed" / "A"),
        ("B", work_dir / "preprocessed" / "B"),
    ]
    assert [scene_input_dir for _, scene_input_dir, _, _ in preprocess_calls] == [
        work_dir / "split" / "A",
        work_dir / "split" / "B",
    ]
    assert merge_calls == [
        (
            [work_dir / "preprocessed" / "A", work_dir / "preprocessed" / "B"],
            output_dir,
            True,
            True,
            "uamvla_calvin_franka",
            "delta",
        )
    ]


def test_run_preprocessor_passes_scene_specific_dataset_source(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, *, check):
        calls.append((cmd, check))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    args = argparse.Namespace(
        num_workers=3,
        on_resolve_failure="skip",
        on_missing_target="abort",
    )

    runner._run_preprocessor(
        "A",
        tmp_path / "input_scene",
        tmp_path / "output_scene",
        args,
    )

    assert len(calls) == 1
    cmd, check = calls[0]
    assert check is True
    assert cmd[cmd.index("--dataset_source") + 1] == "calvin_scene_A"
    assert cmd[cmd.index("--input_dir") + 1] == str(tmp_path / "input_scene")
    assert cmd[cmd.index("--output_dir") + 1] == str(tmp_path / "output_scene")
    assert cmd[cmd.index("--num_workers") + 1] == "3"
    assert cmd[cmd.index("--default_scene") + 1] == "A"
    assert cmd[cmd.index("--on_resolve_failure") + 1] == "skip"
    assert cmd[cmd.index("--on_missing_target") + 1] == "abort"


def test_parse_args_accepts_scene_throughput_flags(tmp_path):
    args = runner.parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--work_dir",
            str(tmp_path / "work"),
            "--output_dir",
            str(tmp_path / "output"),
            "--scenes",
            "A,B,C,D",
            "--scene-workers",
            "3",
            "--resume",
            "--force-scene",
            "B",
            "--force-scene",
            "D",
            "--max-retries",
            "2",
            "--profile",
            "--fail-fast",
        ]
    )

    assert args.scene_workers == 3
    assert args.resume is True
    assert args.force_scene == ["B", "D"]
    assert args.max_retries == 2
    assert args.profile is True
    assert args.fail_fast is True


def test_parse_args_rejects_overwrite_with_resume(tmp_path):
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--work_dir",
                str(tmp_path / "work"),
                "--output_dir",
                str(tmp_path / "output"),
                "--overwrite",
                "--resume",
            ]
        )
