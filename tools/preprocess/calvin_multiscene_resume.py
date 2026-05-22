from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(eq=True)
class SceneDoneMarker:
    scene: str
    scene_input_dir: str
    scene_output_dir: str
    command: list[str]
    return_code: int
    elapsed_sec: float
    attempt_count: int
    retry_count: int
    output_summary: dict[str, int]


@dataclass(eq=True)
class SceneFailureRecord:
    scene: str
    command: list[str]
    return_code: int | None
    exception_type: str
    message: str
    elapsed_sec: float
    attempt_count: int
    retry_count: int


@dataclass(eq=True)
class SceneRunResult:
    scene: str
    scene_input_dir: str
    scene_output_dir: str
    command: list[str]
    return_code: int | None
    elapsed_sec: float
    attempt_count: int
    retry_count: int
    status: str
    output_summary: dict[str, int]
    exception_type: str | None = None
    message: str | None = None

    def to_done_marker(self) -> SceneDoneMarker:
        if self.status != "done" or self.return_code != 0:
            raise ValueError(f"cannot create done marker from status={self.status!r}")
        return SceneDoneMarker(
            scene=self.scene,
            scene_input_dir=self.scene_input_dir,
            scene_output_dir=self.scene_output_dir,
            command=self.command,
            return_code=0,
            elapsed_sec=self.elapsed_sec,
            attempt_count=self.attempt_count,
            retry_count=self.retry_count,
            output_summary=self.output_summary,
        )

    def to_failure_record(self) -> SceneFailureRecord:
        return SceneFailureRecord(
            scene=self.scene,
            command=self.command,
            return_code=self.return_code,
            exception_type=self.exception_type or "RuntimeError",
            message=self.message or f"scene {self.scene} failed",
            elapsed_sec=self.elapsed_sec,
            attempt_count=self.attempt_count,
            retry_count=self.retry_count,
        )


def scene_marker_dir(work_dir: str | Path) -> Path:
    return Path(work_dir) / "meta" / "preprocess_scenes"


def scene_done_marker_path(work_dir: str | Path, scene: str) -> Path:
    return scene_marker_dir(work_dir) / f"{scene.upper()}.done.json"


def failed_scenes_path(work_dir: str | Path) -> Path:
    return Path(work_dir) / "meta" / "preprocess_failed_scenes.json"


def write_scene_done_marker(work_dir: str | Path, marker: SceneDoneMarker) -> None:
    path = scene_done_marker_path(work_dir, marker.scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(marker), indent=2, sort_keys=True) + "\n")


def load_scene_done_markers(work_dir: str | Path) -> dict[str, SceneDoneMarker]:
    root = scene_marker_dir(work_dir)
    if not root.exists():
        return {}
    markers: dict[str, SceneDoneMarker] = {}
    for path in sorted(root.glob("*.done.json")):
        payload = json.loads(path.read_text())
        marker = SceneDoneMarker(**payload)
        markers[marker.scene.upper()] = marker
    return markers


def remove_scene_done_marker(work_dir: str | Path, scene: str) -> None:
    scene_done_marker_path(work_dir, scene).unlink(missing_ok=True)


def write_failed_scenes(
    work_dir: str | Path,
    records: list[SceneFailureRecord],
) -> None:
    path = failed_scenes_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "failed_count": len(records),
        "failed_scenes": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.glob(pattern) if path.is_file())


def _count_episode_dirs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.iterdir() if path.is_dir())


def summarize_scene_output(scene_output_dir: str | Path) -> dict[str, int]:
    root = Path(scene_output_dir)
    return {
        "parquet_files": _count_files(root / "data", "chunk-*/episode_*.parquet"),
        "primary_videos": _count_files(
            root / "videos",
            "chunk-*/video.primary_image/episode_*.mp4",
        ),
        "wrist_videos": _count_files(
            root / "videos",
            "chunk-*/video.wrist_image/episode_*.mp4",
        ),
        "image_target_episodes": _count_episode_dirs(root / "image_targets"),
        "point_cloud_episodes": _count_episode_dirs(root / "point_clouds"),
        "depth_static_episodes": _count_episode_dirs(root / "depths" / "static"),
        "grounding_static_episodes": _count_episode_dirs(
            root / "grounding_masks" / "static"
        ),
        "affordance_static_episodes": _count_episode_dirs(
            root / "affordance_heatmaps" / "static"
        ),
    }
