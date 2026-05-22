from __future__ import annotations

import hashlib
import json
import traceback as traceback_module
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.preprocess.libero_preprocess_utils import compute_episode_plan


PLAN_VERSION = 1
LEROBOT_CHUNK_SIZE = 1000


class PlanMismatchError(RuntimeError):
    """Raised when an existing resume plan does not match current inputs."""


@dataclass(frozen=True)
class PreprocessOptions:
    suite: str
    min_segment_len: int
    active_target_score_window: int
    max_tasks: int | None
    max_demos_per_task: int | None
    max_frames_per_demo: int | None

    def stable_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stable_hash(self) -> str:
        payload = json.dumps(self.stable_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class PreprocessPlan:
    version: int
    suite: str
    created_at: str
    options: dict[str, Any]
    options_hash: str
    frame_counts: dict[str, list[int]]
    episode_plan: dict[str, dict[str, dict[str, int]]]


@dataclass
class DoneMarker:
    task_stem: str
    task_filename: str
    task_name: str
    task_index: int
    demo_indices: list[int]
    episode_indices: list[int]
    episode_lengths: dict[str, int]
    episode_to_task: dict[str, int]
    frame_count: int
    coverage: dict[str, Any]
    camera_params: dict[str, Any] | None
    elapsed_sec: float
    frames_per_sec: float
    retry_count: int
    options_hash: str


@dataclass
class TaskFailureRecord:
    task_stem: str
    task_filename: str
    task_name: str
    exception_type: str
    message: str
    traceback: str
    retry_count: int
    gpu_id: str
    worker_pid: int | None


def plan_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "meta" / "preprocess_plan.json"


def failed_tasks_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "meta" / "preprocess_failed_tasks.json"


def done_marker_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "meta" / "preprocess_tasks"


def done_marker_path(output_dir: str | Path, task_stem: str) -> Path:
    return done_marker_dir(output_dir) / f"{task_stem}.done.json"


def build_preprocess_plan(
    frame_counts: dict[str, list[int]],
    options: PreprocessOptions,
) -> PreprocessPlan:
    raw_plan = compute_episode_plan(frame_counts)
    serial_plan: dict[str, dict[str, dict[str, int]]] = {}
    for (filename, demo_idx), item in raw_plan.items():
        serial_plan.setdefault(filename, {})[str(demo_idx)] = {
            "episode_index": int(item.episode_index),
            "row_start": int(item.row_start),
            "length": int(item.length),
        }
    return PreprocessPlan(
        version=PLAN_VERSION,
        suite=options.suite,
        created_at=datetime.now(timezone.utc).isoformat(),
        options=options.stable_dict(),
        options_hash=options.stable_hash(),
        frame_counts={k: [int(v) for v in vals] for k, vals in frame_counts.items()},
        episode_plan=serial_plan,
    )


def load_or_create_plan(
    output_dir: str | Path,
    frame_counts: dict[str, list[int]],
    options: PreprocessOptions,
    resume: bool,
) -> PreprocessPlan:
    path = plan_path(output_dir)
    if path.exists():
        plan = _plan_from_dict(json.loads(path.read_text()))
        _validate_plan(plan, frame_counts, options)
        return plan
    plan = build_preprocess_plan(frame_counts, options)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(plan), indent=2, sort_keys=True) + "\n")
    return plan


def _plan_from_dict(payload: dict[str, Any]) -> PreprocessPlan:
    return PreprocessPlan(
        version=int(payload["version"]),
        suite=str(payload["suite"]),
        created_at=str(payload["created_at"]),
        options=dict(payload["options"]),
        options_hash=str(payload["options_hash"]),
        frame_counts={
            str(k): [int(v) for v in vals]
            for k, vals in dict(payload["frame_counts"]).items()
        },
        episode_plan={
            str(filename): {
                str(demo_idx): {
                    "episode_index": int(item["episode_index"]),
                    "row_start": int(item["row_start"]),
                    "length": int(item["length"]),
                }
                for demo_idx, item in demos.items()
            }
            for filename, demos in dict(payload["episode_plan"]).items()
        },
    )


def _validate_plan(
    plan: PreprocessPlan,
    frame_counts: dict[str, list[int]],
    options: PreprocessOptions,
) -> None:
    if plan.version != PLAN_VERSION:
        raise PlanMismatchError(f"version mismatch: {plan.version} != {PLAN_VERSION}")
    if plan.suite != options.suite:
        raise PlanMismatchError(f"suite mismatch: {plan.suite} != {options.suite}")
    expected_counts = {k: [int(v) for v in vals] for k, vals in frame_counts.items()}
    if plan.frame_counts != expected_counts:
        raise PlanMismatchError("frame_counts mismatch")
    if plan.options != options.stable_dict():
        raise PlanMismatchError("options mismatch")


def write_done_marker(output_dir: str | Path, marker: DoneMarker) -> None:
    path = done_marker_path(output_dir, marker.task_stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(marker), indent=2, sort_keys=True) + "\n")


def load_done_markers(output_dir: str | Path) -> dict[str, DoneMarker]:
    root = done_marker_dir(output_dir)
    if not root.exists():
        return {}
    markers: dict[str, DoneMarker] = {}
    for path in sorted(root.glob("*.done.json")):
        payload = json.loads(path.read_text())
        marker = DoneMarker(**payload)
        markers[marker.task_stem] = marker
    return markers


def write_failed_tasks(
    output_dir: str | Path,
    records: list[TaskFailureRecord],
) -> None:
    path = failed_tasks_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "failed_count": len(records),
        "failed_tasks": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def traceback_text(exc: BaseException) -> str:
    return "".join(
        traceback_module.format_exception(type(exc), exc, exc.__traceback__)
    )


def planned_episode_paths(
    output_dir: str | Path,
    plan: PreprocessPlan,
    task_filename: str,
) -> set[Path]:
    output = Path(output_dir)
    paths: set[Path] = set()
    for item in plan.episode_plan.get(task_filename, {}).values():
        episode_index = int(item["episode_index"])
        chunk = episode_index // LEROBOT_CHUNK_SIZE
        ep_name = f"episode_{episode_index:06d}"
        paths.add(output / "data" / f"chunk-{chunk:03d}" / f"{ep_name}.parquet")
        paths.add(
            output
            / "videos"
            / f"chunk-{chunk:03d}"
            / "video.primary_image"
            / f"{ep_name}.mp4"
        )
        paths.add(
            output
            / "videos"
            / f"chunk-{chunk:03d}"
            / "video.wrist_image"
            / f"{ep_name}.mp4"
        )
        for root in (
            "image_targets",
            "point_clouds",
            "depths/static",
            "grounding_masks/static",
            "affordance_heatmaps/static",
        ):
            paths.add(output / root / str(episode_index))
    return paths
