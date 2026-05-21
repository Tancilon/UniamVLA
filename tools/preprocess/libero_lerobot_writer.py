from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


LEROBOT_CHUNK_SIZE = 1000
FPS = 10


@dataclass
class LiberoEpisodeBuffers:
    episode_index: int
    task_index: int
    task_name: str
    rows: list[dict[str, Any]]
    primary_frames: list[np.ndarray]
    wrist_frames: list[np.ndarray]
    image_targets: list[np.ndarray | None]
    point_clouds: list[np.ndarray | None]
    depth_targets: list[np.ndarray | None]
    grounding_masks: list[np.ndarray | None]
    grounding_levels: list[str | None]
    affordance_heatmaps: list[np.ndarray | None]


class LiberoLerobotWriter:
    def __init__(self, output_dir: str | Path, fps: int = FPS):
        self.output_dir = Path(output_dir)
        self.fps = int(fps)

    def write_episode(self, buffers: LiberoEpisodeBuffers) -> None:
        self._write_parquet(buffers.rows, buffers.episode_index)
        self._write_videos(
            buffers.primary_frames,
            buffers.wrist_frames,
            buffers.episode_index,
        )
        self._write_sidecars(buffers)

    def _write_parquet(
        self,
        rows: list[dict[str, Any]],
        episode_index: int,
    ) -> None:
        chunk = episode_index // LEROBOT_CHUNK_SIZE
        path = (
            self.output_dir
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)

    def _write_videos(
        self,
        primary_frames: list[np.ndarray],
        wrist_frames: list[np.ndarray],
        episode_index: int,
    ) -> None:
        chunk = episode_index // LEROBOT_CHUNK_SIZE
        base = self.output_dir / "videos" / f"chunk-{chunk:03d}"
        primary_path = base / "video.primary_image" / f"episode_{episode_index:06d}.mp4"
        wrist_path = base / "video.wrist_image" / f"episode_{episode_index:06d}.mp4"
        primary_path.parent.mkdir(parents=True, exist_ok=True)
        wrist_path.parent.mkdir(parents=True, exist_ok=True)
        _write_video(primary_path, np.stack(primary_frames), fps=self.fps)
        _write_video(wrist_path, np.stack(wrist_frames), fps=self.fps)

    def _write_sidecars(self, buffers: LiberoEpisodeBuffers) -> None:
        self._validate_optional_lengths(buffers)
        for frame_idx, target in enumerate(buffers.image_targets):
            if target is not None:
                path = (
                    self.output_dir
                    / "image_targets"
                    / str(buffers.episode_index)
                    / f"{frame_idx}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.asarray(target, dtype=np.uint8)).save(path)
        for frame_idx, pc in enumerate(buffers.point_clouds):
            if pc is not None:
                arr = np.asarray(pc, dtype=np.float32)
                if arr.shape != (1024, 3):
                    raise RuntimeError(f"point cloud shape {arr.shape} != (1024, 3)")
                path = (
                    self.output_dir
                    / "point_clouds"
                    / str(buffers.episode_index)
                    / f"{frame_idx}.npy"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, arr)
        for frame_idx, depth in enumerate(buffers.depth_targets):
            if depth is not None:
                self.write_aux_denoising_sidecar(
                    output_dir=self.output_dir,
                    trajectory_id=buffers.episode_index,
                    base_index=frame_idx,
                    depth_static=depth,
                )
        for frame_idx, mask in enumerate(buffers.grounding_masks):
            if mask is not None:
                self.write_aux_denoising_sidecar(
                    output_dir=self.output_dir,
                    trajectory_id=buffers.episode_index,
                    base_index=frame_idx,
                    grounding_mask=mask,
                    grounding_level=buffers.grounding_levels[frame_idx] or "object",
                )
        for frame_idx, heatmap in enumerate(buffers.affordance_heatmaps):
            if heatmap is not None:
                self.write_aux_denoising_sidecar(
                    output_dir=self.output_dir,
                    trajectory_id=buffers.episode_index,
                    base_index=frame_idx,
                    affordance_heatmap=heatmap,
                )

    @staticmethod
    def _validate_optional_lengths(buffers: LiberoEpisodeBuffers) -> None:
        expected = len(buffers.rows)
        for name in (
            "primary_frames",
            "wrist_frames",
            "image_targets",
            "point_clouds",
            "depth_targets",
            "grounding_masks",
            "grounding_levels",
            "affordance_heatmaps",
        ):
            values = getattr(buffers, name)
            if len(values) != expected:
                raise RuntimeError(f"{name} length {len(values)} != rows {expected}")

    @staticmethod
    def write_aux_denoising_sidecar(
        output_dir: str | Path,
        trajectory_id: int | str,
        base_index: int,
        depth_static: np.ndarray | None = None,
        grounding_mask: np.ndarray | None = None,
        affordance_heatmap: np.ndarray | None = None,
        grounding_level: str = "object",
    ) -> None:
        output_dir = Path(output_dir)
        trajectory = str(trajectory_id)
        base = f"{int(base_index)}.npy"

        if depth_static is not None:
            depth_dir = output_dir / "depths" / "static" / trajectory
            depth_dir.mkdir(parents=True, exist_ok=True)
            np.save(depth_dir / base, np.asarray(depth_static, dtype=np.float32))

        if grounding_mask is not None:
            grounding_dir = output_dir / "grounding_masks" / "static" / trajectory
            grounding_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                grounding_dir / base,
                np.asarray(grounding_mask, dtype=np.float32),
            )
            with open(grounding_dir / f"{int(base_index)}.json", "w") as f:
                json.dump({"grounding_level": str(grounding_level)}, f)

        if affordance_heatmap is not None:
            affordance_dir = output_dir / "affordance_heatmaps" / "static" / trajectory
            affordance_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                affordance_dir / base,
                np.asarray(affordance_heatmap, dtype=np.float32),
            )

    def write_camera_params(self, camera_params: dict[str, float | int | str]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "camera_params.json", "w") as f:
            json.dump(camera_params, f, indent=2)

    def write_meta(
        self,
        tasks: list[str],
        episode_lengths: dict[int, int],
        episode_to_task: dict[int, int],
        total_frames: int,
        coverage: dict[str, Any],
    ) -> None:
        meta_dir = self.output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        self._write_modality(meta_dir)
        with open(meta_dir / "tasks.jsonl", "w") as f:
            for idx, task in enumerate(tasks):
                f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for ep_idx in sorted(episode_lengths):
                f.write(
                    json.dumps(
                        {
                            "episode_index": ep_idx,
                            "tasks": [tasks[episode_to_task[ep_idx]]],
                            "length": int(episode_lengths[ep_idx]),
                        }
                    )
                    + "\n"
                )
        info = {
            "codebase_version": "v2.0",
            "robot_type": "uamvla_libero_franka_h8",
            "total_episodes": len(episode_lengths),
            "total_frames": int(total_frames),
            "total_tasks": len(tasks),
            "fps": self.fps,
            "splits": {"train": f"0:{len(episode_lengths)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "chunks_size": LEROBOT_CHUNK_SIZE,
            "features": {
                "video.primary_image": {
                    "dtype": "video",
                    "shape": [256, 256, 3],
                    "names": ["height", "width", "channel"],
                    "info": {"video.fps": self.fps, "video.channels": 3},
                },
                "video.wrist_image": {
                    "dtype": "video",
                    "shape": [256, 256, 3],
                    "names": ["height", "width", "channel"],
                    "info": {"video.fps": self.fps, "video.channels": 3},
                },
            },
        }
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        with open(meta_dir / "uamvla_aux_coverage.json", "w") as f:
            json.dump(coverage, f, indent=2)

    @staticmethod
    def _write_modality(meta_dir: Path) -> None:
        modality = {
            "state": {
                "robot_obs": {
                    "start": 0,
                    "end": 15,
                    "original_key": "state.robot_obs",
                },
                "target_pose_rot6d": {
                    "start": 0,
                    "end": 6,
                    "original_key": "state.target_pose_rot6d",
                },
                "target_pose_trans": {
                    "start": 0,
                    "end": 3,
                    "original_key": "state.target_pose_trans",
                },
                "static_cam_rot6d": {
                    "start": 0,
                    "end": 6,
                    "original_key": "state.static_cam_rot6d",
                },
                "static_cam_trans": {
                    "start": 0,
                    "end": 3,
                    "original_key": "state.static_cam_trans",
                },
            },
            "action": {
                "x": {"start": 0, "end": 1, "original_key": "action.x"},
                "y": {"start": 0, "end": 1, "original_key": "action.y"},
                "z": {"start": 0, "end": 1, "original_key": "action.z"},
                "roll": {"start": 0, "end": 1, "original_key": "action.roll"},
                "pitch": {"start": 0, "end": 1, "original_key": "action.pitch"},
                "yaw": {"start": 0, "end": 1, "original_key": "action.yaw"},
                "gripper": {
                    "start": 0,
                    "end": 1,
                    "original_key": "action.gripper",
                },
            },
            "video": {
                "primary_image": {"original_key": "video.primary_image"},
                "wrist_image": {"original_key": "video.wrist_image"},
            },
            "annotation": {
                "human.action.task_description": {"original_key": "task_index"},
            },
        }
        with open(meta_dir / "modality.json", "w") as f:
            json.dump(modality, f, indent=2)


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    arr = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
    try:
        import imageio.v3 as iio

        iio.imwrite(path, arr, fps=fps, codec="libx264")
        return
    except ModuleNotFoundError:
        pass

    import cv2

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB video frames with shape (T,H,W,3), got {arr.shape}"
        )
    height, width = int(arr.shape[1]), int(arr.shape[2])
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV failed to open video writer for {path}")
    try:
        for frame in arr:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
