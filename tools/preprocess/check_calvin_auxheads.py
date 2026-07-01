#!/usr/bin/env python3
"""Check CALVIN LeRobot UamVLA aux-head sidecar completeness.

The checker uses ``meta/episodes.jsonl`` as the source of truth for expected
episode/frame indices, then verifies that every aux-head sidecar file exists
and is readable with the expected shape. It also checks that the StarVLA-first
parquet files contain the extra target/camera state columns written by
``convert_calvin_dir_to_lerobot_starvla_with_uam_state.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised in environments without pyarrow.
    pq = None


DEFAULT_DATASETS = (
    Path("datasets/task_ABC_D_scene_A_lerobot_copy"),
    Path("datasets/task_ABC_D_scene_B_lerobot_copy"),
    Path("datasets/task_ABC_D_scene_C_lerobot_copy"),
)


@dataclass(frozen=True)
class AuxSpec:
    label: str
    root: Path
    suffix: str
    kind: str
    shape: tuple[int, ...] | None


@dataclass
class CheckStats:
    expected: int = 0
    ok: int = 0
    missing: int = 0
    bad: int = 0
    extra: int = 0
    zero_like: int = 0

    @property
    def hard_errors(self) -> int:
        return self.missing + self.bad


AUX_SPECS = (
    AuxSpec("recon:image_target", Path("image_targets"), ".png", "png", (224, 224)),
    AuxSpec("pose:point_cloud", Path("point_clouds"), ".npy", "point_cloud", (1024, 3)),
    AuxSpec("depth:depth_target", Path("depths") / "static", ".npy", "array", (256, 256)),
    AuxSpec("grounding:mask", Path("grounding_masks") / "static", ".npy", "map", (1, 20, 20)),
    AuxSpec("grounding:metadata", Path("grounding_masks") / "static", ".json", "grounding_json", None),
    AuxSpec("affordance:heatmap", Path("affordance_heatmaps") / "static", ".npy", "map", (1, 20, 20)),
)

STATE_COLUMN_DIMS = {
    "state": 8,
    "actions": 7,
    "state.target_pose_rot6d": 6,
    "state.target_pose_trans": 3,
    "state.static_cam_rot6d": 6,
    "state.static_cam_trans": 3,
}

PRIMARY_VIDEO_KEYS = ("image", "video.primary_image", "primary_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check CALVIN LeRobot aux-head sidecar completeness against "
            "meta/episodes.jsonl."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        type=Path,
        default=list(DEFAULT_DATASETS),
        help="Dataset roots to check. Defaults to the scene A/B/C _copy datasets.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=256,
        help=(
            "Maximum existing files per aux kind to open for shape/dtype checks. "
            "Use 0 to validate every existing file."
        ),
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Skip parquet extra-state/action column checks.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip primary-video existence checks for future aux heads.",
    )
    parser.add_argument(
        "--fail-on-extra",
        action="store_true",
        help="Treat orphan sidecar files not referenced by episodes.jsonl as errors.",
    )
    parser.add_argument(
        "--fail-on-zero-like",
        action="store_true",
        help="Treat all-zero npy arrays as errors instead of warnings.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def episode_index(episode: dict[str, Any], fallback: int) -> int:
    return int(episode.get("episode_index", fallback))


def episode_length(episode: dict[str, Any]) -> int:
    return int(episode["length"])


def relative_sort_key(value: str) -> tuple[int, int]:
    episode, filename = value.split("/", 1)
    return int(episode), int(Path(filename).stem)


def expected_sidecar_files(episodes: list[dict[str, Any]], suffix: str) -> set[str]:
    expected: set[str] = set()
    for fallback, episode in enumerate(episodes):
        ep_idx = episode_index(episode, fallback)
        for frame_idx in range(episode_length(episode)):
            expected.add(f"{ep_idx}/{frame_idx}{suffix}")
    return expected


def existing_sidecar_files(root: Path, suffix: str) -> dict[str, Path]:
    if not root.exists():
        return {}
    return {
        f"{path.parent.name}/{path.name}": path
        for path in root.glob(f"*/*{suffix}")
        if path.is_file()
    }


def sample_relpaths(relpaths: list[str], sample_limit: int) -> list[str]:
    if sample_limit == 0 or len(relpaths) <= sample_limit:
        return relpaths
    step = max(1, len(relpaths) // sample_limit)
    return relpaths[::step][:sample_limit]


def is_zero_like_array(array: np.ndarray) -> bool:
    return bool(array.size) and not bool(np.any(array))


def validate_png(path: Path, shape: tuple[int, ...]) -> tuple[str | None, bool]:
    with Image.open(path) as image:
        image.load()
        if image.size != shape:
            return f"size={image.size}, expected={shape}", False
        if image.mode != "RGB":
            return f"mode={image.mode}, expected=RGB", False
    return None, False


def validate_array(path: Path, spec: AuxSpec) -> tuple[str | None, bool]:
    array = np.load(path)
    if spec.shape is not None and array.shape != spec.shape:
        return f"shape={array.shape}, expected={spec.shape}", False
    if array.dtype != np.float32:
        return f"dtype={array.dtype}, expected=float32", False
    if not np.isfinite(array).all():
        return "contains NaN/Inf", False
    if spec.kind == "map" and (float(array.min()) < -1e-5 or float(array.max()) > 1.00001):
        return f"value_range=({float(array.min())}, {float(array.max())}), expected=[0,1]", False
    return None, is_zero_like_array(array)


def validate_grounding_json(path: Path) -> tuple[str | None, bool]:
    payload = read_json(path)
    level = payload.get("grounding_level", payload.get("level"))
    if level not in {"object", "part"}:
        return f"grounding_level={level!r}, expected object/part", False
    return None, False


def validate_sidecar(path: Path, spec: AuxSpec) -> tuple[str | None, bool]:
    if spec.kind == "png":
        assert spec.shape is not None
        return validate_png(path, spec.shape)
    if spec.kind == "grounding_json":
        return validate_grounding_json(path)
    return validate_array(path, spec)


def check_sidecar(
    dataset: Path,
    episodes: list[dict[str, Any]],
    spec: AuxSpec,
    *,
    sample_limit: int,
) -> tuple[CheckStats, list[str], list[str], list[str]]:
    expected = expected_sidecar_files(episodes, spec.suffix)
    existing = existing_sidecar_files(dataset / spec.root, spec.suffix)

    missing = sorted(expected - set(existing), key=relative_sort_key)
    extra = sorted(set(existing) - expected, key=relative_sort_key)
    present = sorted(expected & set(existing), key=relative_sort_key)
    checked = sample_relpaths(present, sample_limit)

    bad: list[str] = []
    zero_like = 0
    for relpath in checked:
        error, is_zero_like = validate_sidecar(existing[relpath], spec)
        if error is not None:
            bad.append(f"{relpath}: {error}")
        if is_zero_like:
            zero_like += 1

    stats = CheckStats(
        expected=len(expected),
        ok=len(expected) - len(missing) - len(bad),
        missing=len(missing),
        bad=len(bad),
        extra=len(extra),
        zero_like=zero_like,
    )
    return stats, missing, bad, extra


def print_examples(prefix: str, values: list[str], max_items: int = 8) -> None:
    if values:
        print(f"    first {prefix}: {values[:max_items]}")


def check_info_counts(dataset: Path, episodes: list[dict[str, Any]], info: dict[str, Any]) -> int:
    errors = 0
    total_frames = sum(episode_length(episode) for episode in episodes)
    total_episodes = len(episodes)

    info_episodes = info.get("total_episodes")
    info_frames = info.get("total_frames")
    if info_episodes is not None and int(info_episodes) != total_episodes:
        print(f"  meta/info FAIL total_episodes={info_episodes}, expected={total_episodes}")
        errors += 1
    if info_frames is not None and int(info_frames) != total_frames:
        print(f"  meta/info FAIL total_frames={info_frames}, expected={total_frames}")
        errors += 1

    print(f"  episodes={total_episodes}, frames={total_frames}, info_total_frames={info_frames}")
    return errors


def parquet_path_by_episode(dataset: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in sorted((dataset / "data").glob("chunk-*/episode_*.parquet")):
        try:
            episode = int(path.stem.removeprefix("episode_"))
        except ValueError:
            continue
        paths[episode] = path
    return paths


def check_parquet(dataset: Path, episodes: list[dict[str, Any]]) -> int:
    if pq is None:
        print("  parquet FAIL pyarrow is not installed; pass --skip-parquet to ignore this check")
        return 1

    errors = 0
    parquet_paths = parquet_path_by_episode(dataset)
    for fallback, episode in enumerate(episodes):
        ep_idx = episode_index(episode, fallback)
        length = episode_length(episode)
        path = parquet_paths.get(ep_idx)
        if path is None:
            print(f"  parquet FAIL missing episode_{ep_idx:06d}.parquet")
            errors += 1
            continue

        schema_names = set(pq.read_schema(path).names)
        missing_cols = [column for column in STATE_COLUMN_DIMS if column not in schema_names]
        if missing_cols:
            print(f"  parquet FAIL episode {ep_idx}: missing columns {missing_cols}")
            errors += 1
            continue

        table = pq.read_table(path, columns=list(STATE_COLUMN_DIMS))
        if table.num_rows != length:
            print(f"  parquet FAIL episode {ep_idx}: rows={table.num_rows}, expected={length}")
            errors += 1

        if table.num_rows == 0:
            print(f"  parquet FAIL episode {ep_idx}: empty parquet")
            errors += 1
            continue

        row0 = table.slice(0, 1).to_pylist()[0]
        for column, expected_dim in STATE_COLUMN_DIMS.items():
            value = row0[column]
            if len(value) != expected_dim:
                print(f"  parquet FAIL episode {ep_idx}: {column} dim={len(value)}, expected={expected_dim}")
                errors += 1
            array = np.asarray(value, dtype=np.float32)
            if not np.isfinite(array).all():
                print(f"  parquet FAIL episode {ep_idx}: {column} contains NaN/Inf")
                errors += 1

    if errors == 0:
        print("  parquet extra-state/action check: ok")
    else:
        print(f"  parquet extra-state/action check: errors={errors}")
    return errors


def primary_video_candidates(dataset: Path, episode: int) -> list[Path]:
    episode_name = f"episode_{episode:06d}.mp4"
    candidates: list[Path] = []

    for key in PRIMARY_VIDEO_KEYS:
        candidates.extend(
            sorted((dataset / "videos").glob(f"chunk-*/{key}/{episode_name}"))
        )

    if not candidates:
        for path in sorted((dataset / "videos").glob(f"**/{episode_name}")):
            parent_name = path.parent.name.lower()
            if "wrist" not in parent_name and "gripper" not in parent_name:
                candidates.append(path)

    return candidates


def check_primary_videos(dataset: Path, episodes: list[dict[str, Any]]) -> int:
    errors = 0
    missing: list[str] = []
    empty: list[str] = []

    for fallback, episode in enumerate(episodes):
        ep_idx = episode_index(episode, fallback)
        candidates = primary_video_candidates(dataset, ep_idx)
        if not candidates:
            missing.append(f"episode_{ep_idx:06d}.mp4")
            continue
        if all(path.stat().st_size == 0 for path in candidates):
            empty.append(f"episode_{ep_idx:06d}.mp4")

    errors += len(missing) + len(empty)
    ok = len(episodes) - len(missing) - len(empty)
    print(
        "  future/action_conditioned_future:primary_video "
        f"{ok}/{len(episodes)} ok, missing={len(missing)}, empty={len(empty)}"
    )
    print_examples("missing videos", missing)
    print_examples("empty videos", empty)
    return errors


def check_dataset(args: argparse.Namespace, dataset: Path) -> int:
    print(f"\n== {dataset} ==")
    if not dataset.exists():
        print("  FAIL dataset path does not exist")
        return 1

    episodes_path = dataset / "meta" / "episodes.jsonl"
    info_path = dataset / "meta" / "info.json"
    if not episodes_path.exists():
        print(f"  FAIL missing {episodes_path}")
        return 1
    if not info_path.exists():
        print(f"  FAIL missing {info_path}")
        return 1

    episodes = read_jsonl(episodes_path)
    info = read_json(info_path)
    errors = check_info_counts(dataset, episodes, info)

    for spec in AUX_SPECS:
        stats, missing, bad, extra = check_sidecar(
            dataset,
            episodes,
            spec,
            sample_limit=args.sample_limit,
        )
        print(
            f"  {spec.label:32s} "
            f"{stats.ok}/{stats.expected} ok, missing={stats.missing}, "
            f"bad={stats.bad}, extra={stats.extra}, zero_like_checked={stats.zero_like}"
        )
        print_examples("missing", missing)
        print_examples("bad", bad)
        print_examples("extra", extra)
        errors += stats.hard_errors
        if args.fail_on_extra:
            errors += stats.extra
        if args.fail_on_zero_like:
            errors += stats.zero_like

    if not args.skip_video:
        errors += check_primary_videos(dataset, episodes)
    if not args.skip_parquet:
        errors += check_parquet(dataset, episodes)

    return errors


def main() -> int:
    args = parse_args()
    if args.sample_limit < 0:
        raise ValueError("--sample-limit must be >= 0")

    errors = 0
    for dataset in args.datasets:
        errors += check_dataset(args, dataset)

    print(f"\nRESULT: {'PASS' if errors == 0 else f'FAIL errors={errors}'}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
