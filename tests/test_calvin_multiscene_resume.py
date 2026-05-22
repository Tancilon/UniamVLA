import json

from tools.preprocess.calvin_multiscene_resume import (
    SceneDoneMarker,
    SceneFailureRecord,
    failed_scenes_path,
    load_scene_done_markers,
    remove_scene_done_marker,
    scene_done_marker_path,
    summarize_scene_output,
    write_failed_scenes,
    write_scene_done_marker,
)


def test_scene_done_marker_roundtrip(tmp_path):
    marker = SceneDoneMarker(
        scene="A",
        scene_input_dir=str(tmp_path / "split" / "A"),
        scene_output_dir=str(tmp_path / "preprocessed" / "A"),
        command=["python", "runners/preprocess_calvin.py"],
        return_code=0,
        elapsed_sec=1.25,
        attempt_count=2,
        retry_count=1,
        output_summary={"parquet_files": 3, "primary_videos": 3},
    )

    write_scene_done_marker(tmp_path, marker)

    assert scene_done_marker_path(tmp_path, "A").exists()
    loaded = load_scene_done_markers(tmp_path)
    assert loaded == {"A": marker}

    remove_scene_done_marker(tmp_path, "A")
    assert load_scene_done_markers(tmp_path) == {}


def test_failed_scene_report_roundtrip(tmp_path):
    record = SceneFailureRecord(
        scene="B",
        command=["python", "runners/preprocess_calvin.py"],
        return_code=7,
        exception_type="CalledProcessError",
        message="exit status 7",
        elapsed_sec=2.5,
        attempt_count=3,
        retry_count=2,
    )

    write_failed_scenes(tmp_path, [record])

    payload = json.loads(failed_scenes_path(tmp_path).read_text())
    assert payload["failed_count"] == 1
    assert payload["failed_scenes"][0]["scene"] == "B"
    assert payload["failed_scenes"][0]["retry_count"] == 2


def test_summarize_scene_output_counts_expected_files(tmp_path):
    scene_root = tmp_path / "preprocessed" / "A"
    (scene_root / "data" / "chunk-000").mkdir(parents=True)
    (scene_root / "data" / "chunk-000" / "episode_000000.parquet").write_text("x")
    (scene_root / "videos" / "chunk-000" / "video.primary_image").mkdir(parents=True)
    (scene_root / "videos" / "chunk-000" / "video.primary_image" / "episode_000000.mp4").write_text("x")
    (scene_root / "videos" / "chunk-000" / "video.wrist_image").mkdir(parents=True)
    (scene_root / "videos" / "chunk-000" / "video.wrist_image" / "episode_000000.mp4").write_text("x")
    for root in (
        "image_targets",
        "point_clouds",
        "depths/static",
        "grounding_masks/static",
        "affordance_heatmaps/static",
    ):
        (scene_root / root / "0").mkdir(parents=True)

    assert summarize_scene_output(scene_root) == {
        "parquet_files": 1,
        "primary_videos": 1,
        "wrist_videos": 1,
        "image_target_episodes": 1,
        "point_cloud_episodes": 1,
        "depth_static_episodes": 1,
        "grounding_static_episodes": 1,
        "affordance_static_episodes": 1,
    }
