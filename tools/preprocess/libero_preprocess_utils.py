from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EpisodePlan:
    episode_index: int
    row_start: int
    length: int


def smooth_active_targets(
    raw_targets: Sequence[str | None],
    min_segment_len: int = 3,
) -> list[str | None]:
    values = list(raw_targets)
    if min_segment_len <= 1 or len(values) < 3:
        return values

    segments: list[tuple[int, int, str | None]] = []
    start = 0
    for idx in range(1, len(values) + 1):
        if idx == len(values) or values[idx] != values[start]:
            segments.append((start, idx, values[start]))
            start = idx

    out = values[:]
    for seg_idx, (start, end, value) in enumerate(segments):
        length = end - start
        if length >= min_segment_len or value is None:
            continue
        if seg_idx == 0 or seg_idx == len(segments) - 1:
            continue
        prev_value = segments[seg_idx - 1][2]
        next_value = segments[seg_idx + 1][2]
        if prev_value is not None and prev_value == next_value:
            for i in range(start, end):
                out[i] = prev_value
    return out


def mask_to_token_grid(mask: np.ndarray, target_size: int = 20) -> np.ndarray:
    arr = (np.asarray(mask) > 0).astype(np.float32)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img = img.resize((target_size, target_size), Image.BILINEAR)
    grid = np.asarray(img, dtype=np.float32) / 255.0
    return grid[None, ...].astype(np.float32)


def gaussian_heatmap_from_pixel(
    pixel_xy: np.ndarray,
    image_width: int,
    image_height: int,
    target_size: int = 20,
    sigma: float = 1.5,
) -> np.ndarray:
    x, y = float(pixel_xy[0]), float(pixel_xy[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return np.zeros((1, target_size, target_size), dtype=np.float32)
    cx = (x + 0.5) * target_size / max(1, image_width) - 0.5
    cy = (y + 0.5) * target_size / max(1, image_height) - 0.5
    yy, xx = np.mgrid[0:target_size, 0:target_size].astype(np.float32)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    peak = float(heatmap.max())
    if peak > 0:
        heatmap /= peak
    return heatmap[None, ...].astype(np.float32)


def pointcloud_to_tcp_distance(
    point_cloud: np.ndarray,
    tcp_positions: np.ndarray,
) -> float:
    points = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
    tcp = np.asarray(tcp_positions, dtype=np.float32).reshape(-1, 3)
    if points.size == 0 or tcp.size == 0:
        return float("inf")
    distances = np.linalg.norm(points[:, None, :] - tcp[None, :, :], axis=-1)
    return float(distances.min())


def select_segment_aware_future_tcp(
    tcp_positions: np.ndarray,
    active_targets: Sequence[str | None],
    frame_idx: int,
    local_window_size: int = 4,
    action_chunk_horizon: int = 8,
) -> np.ndarray:
    tcp = np.asarray(tcp_positions, dtype=np.float32).reshape(-1, 3)
    active = list(active_targets)
    if len(tcp) != len(active):
        raise ValueError(
            f"tcp_positions length {len(tcp)} != active_targets length {len(active)}"
        )
    if frame_idx < 0 or frame_idx >= len(tcp):
        raise IndexError(f"frame_idx {frame_idx} outside episode length {len(tcp)}")

    local_window_size = max(1, int(local_window_size))
    action_chunk_horizon = max(1, int(action_chunk_horizon))

    current_target = active[frame_idx]
    segment_end = frame_idx + 1
    while segment_end < len(active) and active[segment_end] == current_target:
        segment_end += 1

    horizon_end = min(len(tcp), frame_idx + action_chunk_horizon)
    preferred_end = min(segment_end, horizon_end)
    if preferred_end - frame_idx >= local_window_size:
        return tcp[frame_idx:preferred_end]

    fallback_end = min(len(tcp), frame_idx + local_window_size, horizon_end)
    if fallback_end <= frame_idx:
        fallback_end = min(len(tcp), frame_idx + 1)
    return tcp[frame_idx:fallback_end]


def pack_robot_obs(
    ee_pos: np.ndarray,
    ee_ori: np.ndarray,
    joint_states: np.ndarray,
    gripper_states: np.ndarray,
) -> np.ndarray:
    gripper = np.asarray(gripper_states, dtype=np.float32).reshape(-1)
    primary_gripper = gripper[:1] if gripper.size else np.zeros(1, dtype=np.float32)
    secondary_gripper = (
        gripper[1:2] if gripper.size > 1 else np.zeros(1, dtype=np.float32)
    )
    obs = np.concatenate(
        [
            np.asarray(ee_pos, dtype=np.float32).reshape(3),
            np.asarray(ee_ori, dtype=np.float32).reshape(3),
            primary_gripper,
            np.asarray(joint_states, dtype=np.float32).reshape(7),
            secondary_gripper,
        ]
    )
    return obs.astype(np.float32)


def compute_episode_plan(
    frame_counts: Mapping[str, Sequence[int]],
) -> dict[tuple[str, int], EpisodePlan]:
    plans: dict[tuple[str, int], EpisodePlan] = {}
    episode_index = 0
    row_start = 0
    for task_file in sorted(frame_counts):
        for demo_idx, length in enumerate(frame_counts[task_file]):
            plans[(task_file, demo_idx)] = EpisodePlan(
                episode_index=episode_index,
                row_start=row_start,
                length=int(length),
            )
            episode_index += 1
            row_start += int(length)
    return plans


def select_render_gpus(
    cuda_visible_devices: str | None,
    render_gpus: str | None,
) -> list[str]:
    if render_gpus:
        return [v.strip() for v in render_gpus.split(",") if v.strip()]
    if cuda_visible_devices:
        values = [v.strip() for v in cuda_visible_devices.split(",") if v.strip()]
        return values or ["0"]
    return ["0"]


def task_name_from_hdf5(path: str | Path) -> str:
    stem = Path(path).name
    if stem.endswith(".hdf5"):
        stem = stem[:-5]
    if stem.endswith("_demo"):
        stem = stem[:-5]
    return stem
