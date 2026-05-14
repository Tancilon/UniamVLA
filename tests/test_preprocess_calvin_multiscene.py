import pytest

import runners.preprocess_calvin_multiscene as runner


def test_runner_rejects_existing_output_without_overwrite(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    output_dir.mkdir()
    work_dir.mkdir()

    with pytest.raises(SystemExit):
        runner.main(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                "--work_dir",
                str(work_dir),
            ]
        )


def test_runner_clears_scene_outputs_and_calls_lerobot_merger(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    work_dir.mkdir()

    stale_scene_output = work_dir / "preprocessed" / "A"
    stale_scene_output.mkdir(parents=True)
    (stale_scene_output / "stale.txt").write_text("old shard\n")

    split_calls = []
    preprocess_calls = []
    merge_calls = []

    def fake_split(input_dir, output_dir, scenes, overwrite):
        scene_tuple = tuple(scenes)
        split_calls.append((input_dir, output_dir, scene_tuple, overwrite))
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
    monkeypatch.setattr(runner, "merge_lerobot_scene_outputs", fake_merge)

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
        (input_dir, work_dir / "split", ("A", "B"), True),
    ]
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
