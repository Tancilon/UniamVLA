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

    returned_cmd = runner._run_preprocessor(
        "A",
        tmp_path / "input_scene",
        tmp_path / "output_scene",
        args,
    )

    assert len(calls) == 1
    cmd, check = calls[0]
    assert returned_cmd == cmd
    assert check is True
    assert cmd[cmd.index("--dataset_source") + 1] == "calvin_scene_A"
    assert cmd[cmd.index("--input_dir") + 1] == str(tmp_path / "input_scene")
    assert cmd[cmd.index("--output_dir") + 1] == str(tmp_path / "output_scene")
    assert cmd[cmd.index("--num_workers") + 1] == "3"
    assert cmd[cmd.index("--default_scene") + 1] == "A"
    assert cmd[cmd.index("--on_resolve_failure") + 1] == "skip"
    assert cmd[cmd.index("--on_missing_target") + 1] == "abort"


def test_build_preprocessor_cmd_contains_scene_specific_arguments(tmp_path):
    args = argparse.Namespace(
        num_workers=4,
        on_resolve_failure="skip",
        on_missing_target="abort",
    )

    cmd = runner._build_preprocessor_cmd(
        "C",
        tmp_path / "split" / "C",
        tmp_path / "preprocessed" / "C",
        args,
    )

    assert cmd[cmd.index("--dataset_source") + 1] == "calvin_scene_C"
    assert cmd[cmd.index("--input_dir") + 1] == str(tmp_path / "split" / "C")
    assert cmd[cmd.index("--output_dir") + 1] == str(tmp_path / "preprocessed" / "C")
    assert cmd[cmd.index("--num_workers") + 1] == "4"
    assert cmd[cmd.index("--default_scene") + 1] == "C"
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


def test_run_scene_preprocessor_with_retries_records_success(tmp_path, monkeypatch):
    calls = []

    def fake_run_preprocessor(scene, scene_input_dir, scene_output_dir, args):
        calls.append(scene)
        scene_output_dir.mkdir(parents=True)
        return ["python", "runner.py", scene]

    monkeypatch.setattr(runner, "_run_preprocessor", fake_run_preprocessor)
    args = argparse.Namespace(
        max_retries=1,
        profile=True,
        num_workers=1,
        on_resolve_failure="abort",
        on_missing_target="skip",
    )

    result = runner._run_scene_preprocessor_with_retries(
        "A",
        tmp_path / "split" / "A",
        tmp_path / "preprocessed" / "A",
        args,
    )

    assert calls == ["A"]
    assert result.status == "done"
    assert result.return_code == 0
    assert result.attempt_count == 1
    assert result.retry_count == 0
    assert result.command == ["python", "runner.py", "A"]


def test_run_scene_preprocessor_with_retries_retries_called_process_error(tmp_path, monkeypatch):
    calls = []

    def flaky_run_preprocessor(scene, scene_input_dir, scene_output_dir, args):
        calls.append(scene)
        if len(calls) == 1:
            raise runner.subprocess.CalledProcessError(
                returncode=9,
                cmd=["python", "runner.py", scene],
            )
        scene_output_dir.mkdir(parents=True)
        return ["python", "runner.py", scene]

    monkeypatch.setattr(runner, "_run_preprocessor", flaky_run_preprocessor)
    args = argparse.Namespace(
        max_retries=1,
        profile=True,
        num_workers=1,
        on_resolve_failure="abort",
        on_missing_target="skip",
    )

    result = runner._run_scene_preprocessor_with_retries(
        "B",
        tmp_path / "split" / "B",
        tmp_path / "preprocessed" / "B",
        args,
    )

    assert calls == ["B", "B"]
    assert result.status == "done"
    assert result.attempt_count == 2
    assert result.retry_count == 1


def test_run_scene_preprocessor_with_retries_returns_failure_after_exhaustion(
    tmp_path, monkeypatch
):
    def failing_run_preprocessor(scene, scene_input_dir, scene_output_dir, args):
        raise runner.subprocess.CalledProcessError(
            returncode=11,
            cmd=["python", "runner.py", scene],
        )

    monkeypatch.setattr(runner, "_run_preprocessor", failing_run_preprocessor)
    args = argparse.Namespace(
        max_retries=1,
        profile=True,
        num_workers=1,
        on_resolve_failure="abort",
        on_missing_target="skip",
    )

    result = runner._run_scene_preprocessor_with_retries(
        "C",
        tmp_path / "split" / "C",
        tmp_path / "preprocessed" / "C",
        args,
    )

    assert result.status == "failed"
    assert result.return_code == 11
    assert result.attempt_count == 2
    assert result.retry_count == 1
    assert result.exception_type == "CalledProcessError"


def test_runner_resume_skips_done_scene_and_merges_all_requested_scenes(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    output_dir.mkdir()
    (work_dir / "split" / "A").mkdir(parents=True)
    (work_dir / "split" / "B").mkdir(parents=True)
    (work_dir / "preprocessed" / "A").mkdir(parents=True)

    from tools.preprocess.calvin_multiscene_resume import (
        SceneDoneMarker,
        write_scene_done_marker,
    )

    write_scene_done_marker(
        work_dir,
        SceneDoneMarker(
            scene="A",
            scene_input_dir=str(work_dir / "split" / "A"),
            scene_output_dir=str(work_dir / "preprocessed" / "A"),
            command=["python", "runner.py", "A"],
            return_code=0,
            elapsed_sec=1.0,
            attempt_count=1,
            retry_count=0,
            output_summary={},
        ),
    )

    preprocess_calls = []
    merge_calls = []

    def fake_split(*, input_dir, output_dir, scenes, scene_config_dir=None):
        return {scene: output_dir / scene for scene in scenes}

    def fake_run_with_retries(scene, scene_input_dir, scene_output_dir, args):
        preprocess_calls.append(scene)
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        return runner.SceneRunResult(
            scene=scene,
            scene_input_dir=str(scene_input_dir),
            scene_output_dir=str(scene_output_dir),
            command=["python", "runner.py", scene],
            return_code=0,
            elapsed_sec=1.0,
            attempt_count=1,
            retry_count=0,
            status="done",
            output_summary={},
        )

    def fake_merge(scene_dirs, output_dir, *, overwrite, skip_stats, robot_type, action_mode):
        merge_calls.append((list(scene_dirs), output_dir, overwrite))

    monkeypatch.setattr(runner, "split_calvin_by_scene", fake_split)
    monkeypatch.setattr(
        runner,
        "_run_scene_preprocessor_with_retries",
        fake_run_with_retries,
    )
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
            "--resume",
            "--skip_stats",
        ]
    )

    assert preprocess_calls == ["B"]
    assert merge_calls == [
        (
            [work_dir / "preprocessed" / "A", work_dir / "preprocessed" / "B"],
            output_dir,
            True,
        )
    ]


def test_runner_force_scene_reruns_done_scene(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    output_dir.mkdir()
    (work_dir / "split" / "A").mkdir(parents=True)
    stale_scene_output = work_dir / "preprocessed" / "A"
    stale_scene_output.mkdir(parents=True)
    (stale_scene_output / "stale.txt").write_text("stale")

    from tools.preprocess.calvin_multiscene_resume import (
        SceneDoneMarker,
        write_scene_done_marker,
    )

    write_scene_done_marker(
        work_dir,
        SceneDoneMarker(
            scene="A",
            scene_input_dir=str(work_dir / "split" / "A"),
            scene_output_dir=str(stale_scene_output),
            command=["python", "runner.py", "A"],
            return_code=0,
            elapsed_sec=1.0,
            attempt_count=1,
            retry_count=0,
            output_summary={},
        ),
    )

    preprocess_calls = []

    def fake_split(*, input_dir, output_dir, scenes, scene_config_dir=None):
        return {"A": output_dir / "A"}

    def fake_run_with_retries(scene, scene_input_dir, scene_output_dir, args):
        assert not (scene_output_dir / "stale.txt").exists()
        preprocess_calls.append(scene)
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        return runner.SceneRunResult(
            scene=scene,
            scene_input_dir=str(scene_input_dir),
            scene_output_dir=str(scene_output_dir),
            command=["python", "runner.py", scene],
            return_code=0,
            elapsed_sec=1.0,
            attempt_count=1,
            retry_count=0,
            status="done",
            output_summary={},
        )

    monkeypatch.setattr(runner, "split_calvin_by_scene", fake_split)
    monkeypatch.setattr(
        runner,
        "_run_scene_preprocessor_with_retries",
        fake_run_with_retries,
    )
    monkeypatch.setattr(
        runner,
        "merge_lerobot_scene_outputs",
        lambda *args, **kwargs: None,
        raising=False,
    )

    runner.main(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--work_dir",
            str(work_dir),
            "--scenes",
            "A",
            "--resume",
            "--force-scene",
            "A",
            "--skip_stats",
        ]
    )

    assert preprocess_calls == ["A"]


def test_runner_uses_thread_pool_when_scene_workers_exceeds_one(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    work_dir.mkdir()
    submitted = []
    merge_calls = []

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, scene, scene_input_dir, scene_output_dir, args):
            submitted.append((self.max_workers, scene))
            return FakeFuture(fn(scene, scene_input_dir, scene_output_dir, args))

    def fake_as_completed(futures):
        return list(futures)

    def fake_split(*, input_dir, output_dir, scenes, scene_config_dir=None):
        return {scene: output_dir / scene for scene in scenes}

    def fake_run_with_retries(scene, scene_input_dir, scene_output_dir, args):
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        return runner.SceneRunResult(
            scene=scene,
            scene_input_dir=str(scene_input_dir),
            scene_output_dir=str(scene_output_dir),
            command=["python", "runner.py", scene],
            return_code=0,
            elapsed_sec=1.0,
            attempt_count=1,
            retry_count=0,
            status="done",
            output_summary={},
        )

    monkeypatch.setattr(runner, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(runner, "as_completed", fake_as_completed)
    monkeypatch.setattr(runner, "split_calvin_by_scene", fake_split)
    monkeypatch.setattr(
        runner,
        "_run_scene_preprocessor_with_retries",
        fake_run_with_retries,
    )
    monkeypatch.setattr(
        runner,
        "merge_lerobot_scene_outputs",
        lambda *args, **kwargs: merge_calls.append(args),
        raising=False,
    )

    runner.main(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--work_dir",
            str(work_dir),
            "--scenes",
            "A,B,C",
            "--overwrite",
            "--scene-workers",
            "2",
            "--skip_stats",
        ]
    )

    assert submitted == [(2, "A"), (2, "B"), (2, "C")]
    assert len(merge_calls) == 1


def test_runner_records_failed_scene_and_skips_merge(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    work_dir.mkdir()
    merge_calls = []

    def fake_split(*, input_dir, output_dir, scenes, scene_config_dir=None):
        return {scene: output_dir / scene for scene in scenes}

    def fake_run_with_retries(scene, scene_input_dir, scene_output_dir, args):
        if scene == "B":
            return runner.SceneRunResult(
                scene=scene,
                scene_input_dir=str(scene_input_dir),
                scene_output_dir=str(scene_output_dir),
                command=["python", "runner.py", scene],
                return_code=5,
                elapsed_sec=1.0,
                attempt_count=1,
                retry_count=0,
                status="failed",
                output_summary={},
                exception_type="CalledProcessError",
                message="exit status 5",
            )
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        return runner.SceneRunResult(
            scene=scene,
            scene_input_dir=str(scene_input_dir),
            scene_output_dir=str(scene_output_dir),
            command=["python", "runner.py", scene],
            return_code=0,
            elapsed_sec=1.0,
            attempt_count=1,
            retry_count=0,
            status="done",
            output_summary={},
        )

    monkeypatch.setattr(runner, "split_calvin_by_scene", fake_split)
    monkeypatch.setattr(
        runner,
        "_run_scene_preprocessor_with_retries",
        fake_run_with_retries,
    )
    monkeypatch.setattr(
        runner,
        "merge_lerobot_scene_outputs",
        lambda *args, **kwargs: merge_calls.append(args),
        raising=False,
    )

    with pytest.raises(SystemExit, match="incomplete"):
        runner.main(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                "--work_dir",
                str(work_dir),
                "--scenes",
                "A,B,C",
                "--overwrite",
                "--scene-workers",
                "2",
                "--skip_stats",
            ]
        )

    assert merge_calls == []
