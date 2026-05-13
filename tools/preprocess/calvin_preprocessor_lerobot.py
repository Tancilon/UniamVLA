"""CALVIN → LeRobot v2 parquet + sidecar preprocessor.

Replaces the JSONL writer in :mod:`tools.preprocess.calvin_preprocessor`.

Output layout (spec §4.3):

    <output_dir>/
    ├── data/chunk-000/episode_NNNNNN.parquet
    ├── videos/chunk-000/video.primary_image/episode_NNNNNN.mp4
    ├── videos/chunk-000/video.wrist_image/episode_NNNNNN.mp4
    ├── image_targets/<trajectory_id>.png
    ├── point_clouds/<trajectory_id>/<base_index>.npy
    ├── camera_params.json
    └── meta/{modality.json, tasks.jsonl, episodes.jsonl, info.json}

The per-frame extraction (npz parsing, scene_obs, target object, point cloud)
reuses the existing :class:`CalvinWorker` setup logic — only the writing
layer is replaced. The old JSONL preprocessor lives alongside until PR 9
cleans it up.

Spec: docs/superpowers/specs/2026-05-13-uamvla-on-qwenoft-design.md §4.3.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from tools.preprocess.base_preprocessor import BasePreprocessor
from tools.preprocess.calvin_preprocessor import (
    EMBODIMENT,
    FRANKA_ACTION_DIM,
    NUM_POINTS,
    RENDER_H,
    RENDER_W,
    CalvinWorker,
    LangWindow,
    SceneResolveError,
    SceneResolver,
    collect_lang_windows,
)
from tools.preprocess.calvin_task_map import (
    STACK_TASKS,
    infer_stack_block,
    resolve_target_object,
)
from starVLA.utils.geometry import (
    depth_to_world_points,
    mat_to_6d,
)
from starVLA.utils.point_cloud import clean_point_cloud

logger = logging.getLogger(__name__)


# LeRobot v2 video / state defaults — agentview RGB-only, 256x256, 15 Hz.
FPS = 15
ACTION_DIM = FRANKA_ACTION_DIM  # 7
ROBOT_OBS_DIM = 15
TARGET_POSE_ROT6D_DIM = 6
TARGET_POSE_TRANS_DIM = 3
STATIC_CAM_ROT6D_DIM = 6
STATIC_CAM_TRANS_DIM = 3
STATE_DIM = (
    ROBOT_OBS_DIM
    + TARGET_POSE_ROT6D_DIM + TARGET_POSE_TRANS_DIM
    + STATIC_CAM_ROT6D_DIM + STATIC_CAM_TRANS_DIM
)  # 33


@dataclass
class _EpisodeBuffers:
    """In-memory buffers populated frame-by-frame for one episode."""
    samples: list[dict]
    primary_frames: list[np.ndarray]   # uint8 (H, W, 3)
    wrist_frames: list[np.ndarray]     # uint8 (H, W, 3)
    point_clouds: list[np.ndarray]     # (1024, 3) float32 after cleaning
    image_target: np.ndarray | None    # uint8 (h, w, 3) — taken from first valid frame
    instruction: str


class CalvinPreprocessorLeRobot(BasePreprocessor):
    """CALVIN → LeRobot v2 parquet + sidecar (point_cloud, image_target,
    camera_params.json). Replaces calvin_preprocessor.py's JSONL output.

    Spec: docs/superpowers/specs/2026-05-13-uamvla-on-qwenoft-design.md §4.3
    """

    def __init__(
        self,
        dataset_source: str = "calvin",
        default_scene: str | None = None,
        num_workers: int = 1,
        on_resolve_failure: str = "abort",
        on_missing_target: str = "abort",
        max_episodes: int | None = None,
    ):
        if on_resolve_failure not in {"skip", "abort"}:
            raise ValueError(
                f"on_resolve_failure must be 'skip' or 'abort', "
                f"got {on_resolve_failure!r}"
            )
        if on_missing_target not in {"skip", "abort"}:
            raise ValueError(
                f"on_missing_target must be 'skip' or 'abort', "
                f"got {on_missing_target!r}"
            )
        if num_workers != 1:
            # Per spec §4.3 / smoke scope: keep PR 2 single-process. The
            # old preprocessor's spawn-pool path was JSONL-specific; the
            # new path collects frames in-memory per episode so the
            # per-window pool isn't needed for the smoke run. Full ABCD_D
            # parallelism can be added later if profiling demands it.
            logger.warning(
                "num_workers=%d ignored — LeRobot preprocessor runs single-"
                "process. (See spec §4.3.)", num_workers,
            )
        self.dataset_source = dataset_source
        self.default_scene = default_scene
        self.num_workers = 1
        self.on_resolve_failure = on_resolve_failure
        self.on_missing_target = on_missing_target
        self.max_episodes = max_episodes

    # ------------------------------------------------------------------
    # Top-level driver
    # ------------------------------------------------------------------

    def process(self, input_dir, output_dir):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        windows = collect_lang_windows(input_dir)
        if not windows:
            raise RuntimeError(
                f"No language windows found in {input_dir}; nothing to do."
            )
        if self.max_episodes is not None:
            windows = windows[: self.max_episodes]
        logger.info(
            "Processing %d language windows (max_episodes=%s)",
            len(windows), self.max_episodes,
        )

        scene_resolver = SceneResolver(
            split_dir=input_dir, default_scene=self.default_scene,
        )
        try:
            resolved = {
                scene_resolver.resolve_for_window(w) for w in windows
            }
            logger.info(
                "Scene letters covered (informational): %s", sorted(resolved),
            )
        except SceneResolveError:
            raise

        worker = self._build_worker(
            rank=0, dataset_path=str(input_dir),
            scene_resolver=scene_resolver,
        )
        try:
            # Capture camera intrinsics once from the env. Static + wrist
            # intrinsics are camera-state-independent, so any frame
            # works. We delay the render call until after the first window
            # has run reset() to ensure the env is in a valid state, but
            # rendering before that also works on PyBullet — the env was
            # constructed with a default scene. We use a dummy render now.
            intrinsics = self._read_intrinsics(worker)

            tasks_seen: list[str] = []
            task_to_index: dict[str, int] = {}
            episode_index_to_task: dict[int, int] = {}
            episode_lengths: dict[int, int] = {}
            n_total_frames = 0
            kept_episodes = 0

            for episode_index, window in enumerate(windows):
                logger.info(
                    "Episode %d/%d: window_idx=%d frames=[%d, %d] task=%r",
                    episode_index, len(windows) - 1,
                    window.window_idx, window.ep_start, window.ep_end,
                    window.task_label,
                )
                buffers = self._process_window_into_buffers(
                    worker=worker, window=window,
                    input_dir=input_dir, episode_index=episode_index,
                )
                if buffers is None:
                    logger.warning(
                        "Episode %d skipped (no valid frames produced).",
                        episode_index,
                    )
                    continue

                # Register task index once per unique instruction.
                instr = buffers.instruction
                if instr not in task_to_index:
                    task_to_index[instr] = len(tasks_seen)
                    tasks_seen.append(instr)
                task_index = task_to_index[instr]
                episode_index_to_task[episode_index] = task_index
                episode_lengths[episode_index] = len(buffers.samples)
                n_total_frames += len(buffers.samples)

                # Stamp task_index into every row before writing parquet.
                for row in buffers.samples:
                    row["task_index"] = task_index

                self._emit_episode_parquet(
                    buffers.samples, output_dir, episode_index,
                )
                self._emit_episode_videos(
                    buffers.primary_frames, buffers.wrist_frames,
                    output_dir, episode_index, fps=FPS,
                )
                if buffers.image_target is not None and buffers.point_clouds:
                    self._emit_episode_sidecars(
                        buffers.image_target, buffers.point_clouds,
                        output_dir, episode_index,
                    )
                else:
                    logger.warning(
                        "Episode %d missing sidecar (image_target=%s, "
                        "point_clouds=%d) — skipping sidecar emission.",
                        episode_index,
                        buffers.image_target is not None,
                        len(buffers.point_clouds),
                    )
                kept_episodes += 1
        finally:
            try:
                worker.env.close()
            except Exception as exc:  # pragma: no cover
                logger.debug("env.close() raised: %s", exc)

        if kept_episodes == 0:
            raise RuntimeError("No episodes produced; nothing to write.")

        # Dataset-level outputs.
        static_intr = intrinsics["static"]
        self._emit_camera_params(
            fx=static_intr["fx"], fy=static_intr["fy"],
            cx=static_intr["cx"], cy=static_intr["cy"],
            output_dir=output_dir, width=RENDER_W, height=RENDER_H,
        )
        self._emit_meta(
            output_dir=output_dir,
            n_episodes=kept_episodes,
            n_total_frames=n_total_frames,
            tasks_seen=tasks_seen,
            episode_index_to_task=episode_index_to_task,
            episode_lengths=episode_lengths,
            fps=FPS,
        )
        logger.info(
            "Done. %d episodes, %d frames → %s",
            kept_episodes, n_total_frames, output_dir,
        )

    # ------------------------------------------------------------------
    # Per-window extraction (reuses CalvinWorker setup, in-memory output)
    # ------------------------------------------------------------------

    def _build_worker(
        self, rank: int, dataset_path: str, scene_resolver: SceneResolver,
    ) -> CalvinWorker:
        """Construct the underlying CalvinWorker. Tests monkeypatch this."""
        from tools.preprocess.calvin_env_adapter import (
            make_calvin_env_adapter,
        )
        env = make_calvin_env_adapter(dataset_path=dataset_path)
        return CalvinWorker(
            env=env,
            dataset_source=self.dataset_source,
            rank=rank,
            scene_resolver=scene_resolver,
            on_resolve_failure=self.on_resolve_failure,
            on_missing_target=self.on_missing_target,
        )

    def _read_intrinsics(self, worker: CalvinWorker) -> dict:
        rendered = worker.env.render_cameras(width=RENDER_W, height=RENDER_H)
        return {
            "static": rendered["static_intrinsic"],
            "wrist": rendered["wrist_intrinsic"],
        }

    def _process_window_into_buffers(
        self,
        worker: CalvinWorker,
        window: LangWindow,
        input_dir: Path,
        episode_index: int,
    ) -> _EpisodeBuffers | None:
        """Replay a single language window into in-memory buffers.

        Mirrors the npz/env loop in CalvinWorker.process_window (lines
        222-372 of calvin_preprocessor.py) but skips disk writes for the
        per-frame .jpg/.npy artifacts — we keep the raw arrays so the
        episode-level writers can emit one parquet + two mp4 + one
        point_cloud dir + one image_target.png.
        """
        # Target object resolution (mirrors CalvinWorker.process_window
        # lines 222-240).
        if window.task_label in STACK_TASKS:
            scene_letter = worker.scene_resolver.resolve_for_window(window)
            start_so = CalvinWorker._load_scene_obs(input_dir, window.ep_start)
            end_so = CalvinWorker._load_scene_obs(input_dir, window.ep_end)
            target_object_id = infer_stack_block(
                window.task_label, scene_letter, start_so, end_so,
            )
        else:
            try:
                target_object_id = resolve_target_object(window.task_label)
            except KeyError:
                if self.on_resolve_failure == "skip":
                    logger.warning(
                        "Skipping window %d: unknown task_label %r",
                        window.window_idx, window.task_label,
                    )
                    return None
                raise
        target_seg_id = worker.env.get_target_seg_id(target_object_id)

        frames = list(range(window.ep_start, window.ep_end + 1))
        n_frames = len(frames)

        samples: list[dict] = []
        primary_frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []
        point_clouds: list[np.ndarray] = []
        image_target: np.ndarray | None = None

        for step_idx, t in enumerate(frames):
            with np.load(input_dir / f"episode_{t:07d}.npz") as npz:
                rel_actions = np.asarray(npz["rel_actions"], dtype=np.float32)
                robot_obs = np.asarray(npz["robot_obs"], dtype=np.float32)
                scene_obs = np.asarray(npz["scene_obs"], dtype=np.float32)

            worker.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            rendered = worker.env.render_cameras(
                width=RENDER_W, height=RENDER_H,
            )
            obj_pos, obj_mat = worker.env.get_object_pose(target_object_id)

            # Always keep the rendered images (parquet has VideoBackend
            # entries that reference these mp4 frames by index).
            primary_frames.append(
                np.ascontiguousarray(rendered["rgb_static"], dtype=np.uint8)
            )
            wrist_frames.append(
                np.ascontiguousarray(rendered["rgb_wrist"], dtype=np.uint8)
            )

            # Try to extract point cloud + image_target. On miss, the frame
            # is still recorded in parquet but with sidecar omission (the
            # framework falls back to defaults at training time).
            pts = depth_to_world_points(
                depth=rendered["depth_static"],
                seg_mask=rendered["seg_static"],
                intrinsic=rendered["static_intrinsic"],
                cam_R=rendered["static_cam_R"],
                cam_t=rendered["static_cam_t"],
                target_id=target_seg_id,
                num_points=NUM_POINTS,
            )
            if pts is not None:
                pc_clean = clean_point_cloud(pts).astype(np.float32)
                if pc_clean.shape != (NUM_POINTS, 3):
                    raise RuntimeError(
                        f"point cloud shape {pc_clean.shape} != "
                        f"({NUM_POINTS}, 3) — clean_point_cloud invariant"
                    )
                point_clouds.append(pc_clean)
            elif self.on_missing_target == "abort":
                raise RuntimeError(
                    f"Target {target_object_id!r} not visible in frame {t} "
                    f"(window {window.window_idx})"
                )

            if image_target is None:
                # We follow spec §4.2 — one image_target per trajectory,
                # taken from the first frame where the target is visible.
                from starVLA.utils.geometry import crop_target_from_seg
                target_img = crop_target_from_seg(
                    rendered["rgb_static"], rendered["seg_static"],
                    target_id=target_seg_id,
                )
                if target_img is not None:
                    image_target = np.asarray(target_img, dtype=np.uint8)

            # State vector (33 dims): robot_obs (15) || target_pose_rot6d (6)
            # || target_pose_trans (3) || static_cam_rot6d (6) ||
            # static_cam_trans (3).
            target_pose_rot6d = mat_to_6d(np.asarray(obj_mat))
            target_pose_trans = np.asarray(obj_pos, dtype=np.float32).tolist()
            static_cam_rot6d = mat_to_6d(
                np.asarray(rendered["static_cam_R"]),
            )
            static_cam_trans = np.asarray(
                rendered["static_cam_t"], dtype=np.float32,
            ).tolist()

            action = rel_actions.tolist()  # 7 dims

            sample = {
                "episode_index": int(episode_index),
                "frame_index": int(step_idx),
                "timestamp": float(step_idx) / float(FPS),
                "index": int(step_idx),
                "task_index": 0,
                # state.* columns (per LeRobot convention, dotted-key flat)
                "state.robot_obs": robot_obs.astype(np.float32).tolist(),
                "state.target_pose_rot6d":
                    [float(v) for v in target_pose_rot6d],
                "state.target_pose_trans":
                    [float(v) for v in target_pose_trans],
                "state.static_cam_rot6d":
                    [float(v) for v in static_cam_rot6d],
                "state.static_cam_trans":
                    [float(v) for v in static_cam_trans],
                # action.* (split by component per modality.json)
                "action.x": float(action[0]),
                "action.y": float(action[1]),
                "action.z": float(action[2]),
                "action.roll": float(action[3]),
                "action.pitch": float(action[4]),
                "action.yaw": float(action[5]),
                "action.gripper": float(action[6]),
                # Language column (free-form string; tasks.jsonl indexes it)
                "annotation.human.action.task_description":
                    window.instruction,
                # Provenance + sidecar lookup
                "trajectory_id": int(episode_index),
                "base_index": int(step_idx),
            }
            samples.append(sample)

        if not samples:
            return None

        return _EpisodeBuffers(
            samples=samples,
            primary_frames=primary_frames,
            wrist_frames=wrist_frames,
            point_clouds=point_clouds,
            image_target=image_target,
            instruction=window.instruction,
        )

    # ------------------------------------------------------------------
    # Episode-level writers (from spec §4.3, Task 2.3 Steps 2-6)
    # ------------------------------------------------------------------

    def _emit_episode_parquet(
        self, episode_samples: list[dict],
        output_dir: Path, episode_index: int,
    ) -> None:
        chunk_dir = output_dir / "data" / "chunk-000"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_dir / f"episode_{episode_index:06d}.parquet"
        table = pa.Table.from_pylist(episode_samples)
        pq.write_table(table, parquet_path)

    def _emit_episode_videos(
        self, primary_frames: list, wrist_frames: list,
        output_dir: Path, episode_index: int, fps: int = FPS,
    ) -> None:
        video_dir = output_dir / "videos" / "chunk-000"
        primary_dir = video_dir / "video.primary_image"
        wrist_dir = video_dir / "video.wrist_image"
        primary_dir.mkdir(parents=True, exist_ok=True)
        wrist_dir.mkdir(parents=True, exist_ok=True)
        primary_path = primary_dir / f"episode_{episode_index:06d}.mp4"
        wrist_path = wrist_dir / f"episode_{episode_index:06d}.mp4"
        iio.imwrite(
            primary_path, np.stack(primary_frames), fps=fps, codec='libx264',
        )
        iio.imwrite(
            wrist_path, np.stack(wrist_frames), fps=fps, codec='libx264',
        )

    def _emit_episode_sidecars(
        self, image_target: np.ndarray, point_clouds: list,
        output_dir: Path, episode_index: int,
    ) -> None:
        img_target_dir = output_dir / "image_targets"
        img_target_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image_target).save(
            img_target_dir / f"{episode_index}.png",
        )
        pc_episode_dir = output_dir / "point_clouds" / str(episode_index)
        pc_episode_dir.mkdir(parents=True, exist_ok=True)
        for base_index, pc in enumerate(point_clouds):
            # Defensive — clean_point_cloud is already applied upstream,
            # but we re-assert the shape here per spec §4.3 (sidecar must
            # be (1024, 3) float32).
            arr = np.asarray(pc, dtype=np.float32)
            if arr.shape != (NUM_POINTS, 3):
                raise RuntimeError(
                    f"point cloud shape {arr.shape} != ({NUM_POINTS}, 3)"
                )
            np.save(pc_episode_dir / f"{base_index}.npy", arr)

    def _emit_camera_params(
        self, fx, fy, cx, cy, output_dir: Path,
        width: int = RENDER_W, height: int = RENDER_H,
    ) -> None:
        params = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": width,
            "height": height,
            "camera_name": "agentview",
        }
        with open(output_dir / "camera_params.json", "w") as f:
            json.dump(params, f, indent=2)

    def _emit_meta(
        self, output_dir: Path, n_episodes: int, n_total_frames: int,
        tasks_seen: list[str],
        episode_index_to_task: dict[int, int],
        episode_lengths: dict[int, int],
        fps: int = FPS,
    ) -> None:
        meta_dir = output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # Copy the canonical modality.json into the dataset meta dir.
        # Use the package-rooted path so the runner is cwd-agnostic.
        repo_root = Path(__file__).resolve().parents[2]
        src_modality = (
            repo_root
            / "examples" / "calvin" / "train_files"
            / "data_registry" / "modality.json"
        )
        shutil.copy(src_modality, meta_dir / "modality.json")

        with open(meta_dir / "tasks.jsonl", "w") as f:
            for task_index, task in enumerate(tasks_seen):
                f.write(json.dumps(
                    {"task_index": task_index, "task": task}
                ) + "\n")

        with open(meta_dir / "episodes.jsonl", "w") as f:
            for ep_idx in sorted(episode_lengths.keys()):
                f.write(json.dumps({
                    "episode_index": ep_idx,
                    "tasks": [
                        tasks_seen[episode_index_to_task[ep_idx]],
                    ],
                    "length": int(episode_lengths[ep_idx]),
                }) + "\n")

        info = {
            "codebase_version": "v2.0",
            "robot_type": EMBODIMENT,
            "total_episodes": n_episodes,
            "total_frames": n_total_frames,
            "total_tasks": len(tasks_seen),
            "fps": fps,
            "splits": {"train": f"0:{n_episodes}"},
            "features": {
                "video.primary_image": {
                    "dtype": "video",
                    "shape": [RENDER_H, RENDER_W, 3],
                },
                "video.wrist_image": {
                    "dtype": "video",
                    "shape": [RENDER_H, RENDER_W, 3],
                },
            },
        }
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)
