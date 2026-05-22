"""CALVIN → LeRobot v2 parquet + sidecar preprocessor.

Replaces the JSONL writer in :mod:`tools.preprocess.calvin_preprocessor`.

Output layout (spec §4.3):

    <output_dir>/
    ├── data/chunk-XXX/episode_NNNNNN.parquet
    ├── videos/chunk-XXX/video.primary_image/episode_NNNNNN.mp4
    ├── videos/chunk-XXX/video.wrist_image/episode_NNNNNN.mp4
    ├── image_targets/<trajectory_id>/<base_index>.png
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
from tools.preprocess.calvin_task_map import (
    STACK_TASKS,
    infer_stack_block,
    resolve_target_object,
)
from starVLA.utils.geometry import (
    crop_target_from_seg,
    depth_to_world_points,
    mat_to_6d,
)
from starVLA.utils.point_cloud import clean_point_cloud

logger = logging.getLogger(__name__)


# ---------- Inlined from legacy tools/preprocess/calvin_preprocessor.py -----
# PR 9 deletes the legacy module. The symbols below are absorbed here so this
# (LeRobot-format) preprocessor stays self-contained.

@dataclass(frozen=True)
class LangWindow:
    """One 64-frame language-annotated window — the UAM episode unit."""
    window_idx: int
    ep_start: int
    ep_end: int        # inclusive
    instruction: str
    task_label: str


def collect_lang_windows(split_dir: Path) -> list[LangWindow]:
    """Parse a CALVIN split dir into the list of language windows.

    Reads `lang_annotations/auto_lang_ann.npy` which is the authoritative
    source of (instruction, task, frame_range) triples.  `ep_start_end_ids.npy`
    is not consulted here — windows already live inside it by construction,
    and we drop non-language frames per the spec.
    """
    split_dir = Path(split_dir)
    lang_path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not lang_path.exists():
        raise FileNotFoundError(
            f"Missing language annotations: {lang_path}. "
            f"A CALVIN split without lang_annotations cannot be converted."
        )
    lang_ann = np.load(lang_path, allow_pickle=True).item()
    anns = lang_ann["language"]["ann"]
    tasks = lang_ann["language"]["task"]
    indx = lang_ann["info"]["indx"]

    if not (len(anns) == len(tasks) == len(indx)):
        raise ValueError(
            f"Malformed auto_lang_ann.npy in {split_dir}: "
            f"ann/task/indx lengths {len(anns)}/{len(tasks)}/{len(indx)} must match"
        )

    windows = []
    for i, (ann, task, rng) in enumerate(zip(anns, tasks, indx)):
        start, end = int(rng[0]), int(rng[1])
        windows.append(LangWindow(
            window_idx=i,
            ep_start=start,
            ep_end=end,
            instruction=str(ann),
            task_label=str(task),
        ))
    if not windows:
        logger.warning(
            "collect_lang_windows: no language windows found in %s", split_dir,
        )
    return windows


class SceneResolveError(RuntimeError):
    """Raised when a window's scene cannot be determined."""


class SceneResolver:
    """Resolve the CALVIN scene letter (A/B/C/D) for each window.

    Single-scene splits (calvin_debug_dataset, task_D_D) use the CLI
    --default_scene argument.  Multi-scene splits (task_ABCD_D, task_ABC_D) ship
    a scene_info.npy mapping frame ranges to scene letters.  The resolver
    picks the latter when present, else falls back to the default.
    """

    def __init__(self, split_dir: Path, default_scene: str | None):
        self.split_dir = Path(split_dir)
        self.default_scene = default_scene
        self._scene_ranges: list[tuple[str, int, int]] | None = None
        info_path = self.split_dir / "scene_info.npy"
        if info_path.exists():
            # Both legacy CALVIN dumps (`scene_A`) and newer ones
            # (`calvin_scene_A`) appear in the wild. Reuse the splitter's
            # regex-based parser so both forms route to letters A/B/C/D
            # — `infer_stack_block` only accepts those four.
            from tools.preprocess.calvin_scene_splitter import (
                load_scene_ranges,
            )
            self._scene_ranges = [
                (letter, lo, hi)
                for letter, (lo, hi) in load_scene_ranges(self.split_dir).items()
            ]
        elif default_scene is None:
            raise SceneResolveError(
                f"{self.split_dir} has no scene_info.npy and no "
                f"default_scene was provided. One is required."
            )

    def resolve_for_window(self, window: LangWindow) -> str:
        if self._scene_ranges is None:
            return self.default_scene  # type: ignore[return-value]
        for letter, start, end in self._scene_ranges:
            if start <= window.ep_start <= end and start <= window.ep_end <= end:
                return letter
        raise SceneResolveError(
            f"Window {window.window_idx} frames "
            f"[{window.ep_start},{window.ep_end}] not covered by any "
            f"scene range in {self.split_dir}/scene_info.npy"
        )


# --- Format constants (shared with LIBERO preprocessor semantics) -----------
MAX_ACTION_DIM = 24
FRANKA_ACTION_DIM = 7
ACTION_MASK = [1] * FRANKA_ACTION_DIM + [0] * (MAX_ACTION_DIM - FRANKA_ACTION_DIM)
RENDER_W = 256
RENDER_H = 256
NUM_POINTS = 1024
EMBODIMENT = "franka_calvin"


class CalvinWorker:
    """Consumes LangWindows, emits shard_<rank>_win<idx>.jsonl + files.

    The env is constructor-injected so tests can swap in a pure-Python fake.
    """

    def __init__(
        self,
        env,
        dataset_source: str,
        rank: int,
        scene_resolver: "SceneResolver",
        on_resolve_failure: str = "abort",
        on_missing_target: str = "abort",
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
        self.env = env
        self.dataset_source = dataset_source
        self.rank = rank
        # `scene_resolver` is required even on single-scene splits — its
        # per-window letter lookup drives the scene-dependent slot order
        # in `infer_stack_block`. For single-scene splits it just returns
        # the default scene letter every time.
        self.scene_resolver = scene_resolver
        self.on_resolve_failure = on_resolve_failure
        self.on_missing_target = on_missing_target


def _load_scene_obs(input_dir: Path, frame: int) -> np.ndarray:
    """Read scene_obs from one episode npz.

    Inlined here so this module does not depend on a private staticmethod
    of the legacy :class:`CalvinWorker` (PR 9 deletes that file).
    """
    npz_path = input_dir / f"episode_{frame:07d}.npz"
    with np.load(npz_path) as npz:
        return np.asarray(npz["scene_obs"], dtype=np.float32)


# LeRobot v2 video / state defaults — agentview RGB-only, 256x256, 15 Hz.
FPS = 15
LEROBOT_CHUNK_SIZE = 1000
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
    image_targets: list[np.ndarray]    # one uint8 (h, w, 3) target crop per frame
    depth_targets: list[np.ndarray]    # raw static metric depth per frame
    grounding_masks: list[np.ndarray]  # [1, 20, 20] object/part mask per frame
    grounding_levels: list[str]        # "part" or "object" per frame
    affordance_heatmaps: list[np.ndarray]  # [1, 20, 20] future TCP affordance
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
            # Dataset-global cumulative counter (LeRobot v2 invariant: the
            # `index` column must be contiguous across all parquets so that
            # `index - frame_index` recovers each episode's start). It is
            # advanced ONLY when an episode is actually emitted (so dropped
            # episodes do not leave gaps).
            global_index = 0
            aux_coverage = self._empty_aux_coverage()

            for window in windows:
                logger.info(
                    "Window idx=%d frames=[%d, %d] task=%r",
                    window.window_idx, window.ep_start, window.ep_end,
                    window.task_label,
                )
                # Tentatively assign the next kept_episodes slot. We only
                # commit it (kept_episodes += 1) once we know the episode
                # will be emitted — otherwise dropped episodes would leave
                # holes in `episode_index`.
                episode_index = kept_episodes
                buffers = self._process_window_into_buffers(
                    worker=worker, window=window,
                    input_dir=input_dir, episode_index=episode_index,
                    global_index_start=global_index,
                )
                if buffers is None:
                    logger.warning(
                        "Window %d: dropped (no valid frames produced).",
                        window.window_idx,
                    )
                    continue
                if len(buffers.image_targets) != len(buffers.samples):
                    logger.warning(
                        "Episode %d (window %d): dropped (%d image_targets "
                        "for %d frames).",
                        episode_index,
                        window.window_idx,
                        len(buffers.image_targets),
                        len(buffers.samples),
                    )
                    continue
                if len(buffers.point_clouds) != len(buffers.samples):
                    logger.warning(
                        "Episode %d (window %d): dropped (%d point clouds "
                        "for %d frames).",
                        episode_index,
                        window.window_idx,
                        len(buffers.point_clouds),
                        len(buffers.samples),
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
                self._emit_episode_sidecars(
                    buffers.image_targets,
                    buffers.point_clouds,
                    output_dir,
                    episode_index,
                    depth_targets=buffers.depth_targets,
                    grounding_masks=buffers.grounding_masks,
                    grounding_levels=buffers.grounding_levels,
                    affordance_heatmaps=buffers.affordance_heatmaps,
                )
                self._merge_aux_coverage(
                    aux_coverage,
                    task_name=instr,
                    lengths={
                        "image_target": len(buffers.image_targets),
                        "point_cloud": len(buffers.point_clouds),
                        "depth": len(buffers.depth_targets),
                        "grounding": len(buffers.grounding_masks),
                        "affordance": len(buffers.affordance_heatmaps),
                    },
                )
                # Commit the global counter and episode slot.
                global_index += len(buffers.samples)
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
        self._emit_aux_coverage(output_dir, aux_coverage)
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
        global_index_start: int = 0,
    ) -> _EpisodeBuffers | None:
        """Replay a single language window into in-memory buffers.

        Mirrors the npz/env loop in CalvinWorker.process_window (lines
        222-372 of calvin_preprocessor.py) but skips disk writes for the
        per-frame .jpg/.npy artifacts — we keep the raw arrays so the
        episode-level writers can emit one parquet + two mp4 + one
        point_cloud dir + one image_target PNG per frame.
        """
        # Target object resolution (mirrors CalvinWorker.process_window
        # lines 222-240).
        if window.task_label in STACK_TASKS:
            scene_letter = worker.scene_resolver.resolve_for_window(window)
            start_so = _load_scene_obs(input_dir, window.ep_start)
            end_so = _load_scene_obs(input_dir, window.ep_end)
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
        grounding_level = self._grounding_level_for_target_id(target_object_id)
        target_seg_id = worker.env.get_target_seg_id(target_object_id)

        frames = list(range(window.ep_start, window.ep_end + 1))
        n_frames = len(frames)

        samples: list[dict] = []
        primary_frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []
        point_clouds: list[np.ndarray] = []
        image_targets: list[np.ndarray] = []
        depth_targets: list[np.ndarray] = []
        grounding_masks: list[np.ndarray] = []
        grounding_levels: list[str] = []
        tcp_positions: list[np.ndarray] = []
        affordance_contexts: list[dict] = []
        # Dataset-global row counter; advances 1-per-sample and is written
        # to the `index` column (LeRobot v2 invariant).
        global_index = global_index_start

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
            depth_targets.append(
                np.asarray(rendered["depth_static"], dtype=np.float32)
            )
            grounding_masks.append(
                self._segmentation_to_token_grid_mask(
                    rendered["seg_static"], target_seg_id,
                )
            )
            grounding_levels.append(grounding_level)

            # Always keep the rendered images (parquet has VideoBackend
            # entries that reference these mp4 frames by index).
            primary_frames.append(
                np.ascontiguousarray(rendered["rgb_static"], dtype=np.uint8)
            )
            wrist_frames.append(
                np.ascontiguousarray(rendered["rgb_wrist"], dtype=np.uint8)
            )

            # Try to extract point cloud + per-frame image_target. On miss,
            # drop/abort the whole window so parquet rows and sidecars stay
            # one-to-one.
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
                tcp_positions.append(
                    np.asarray(robot_obs[:3], dtype=np.float32)
                )
                affordance_contexts.append({
                    "point_cloud": pc_clean,
                    "intrinsic": rendered["static_intrinsic"],
                    "cam_R": np.asarray(rendered["static_cam_R"], dtype=np.float32),
                    "cam_t": np.asarray(rendered["static_cam_t"], dtype=np.float32),
                })
            elif self.on_missing_target == "abort":
                raise RuntimeError(
                    f"Target {target_object_id!r} not visible in frame {t} "
                    f"(window {window.window_idx})"
                )
            else:
                # on_missing_target == "skip": drop the ENTIRE window for
                # dataset consistency. primary_frames / wrist_frames are
                # appended unconditionally above, so silently skipping just
                # this frame leaves point_clouds shorter than parquet rows —
                # dataloader would FileNotFoundError on the missing sidecar
                # at training time.
                logger.warning(
                    "Target %r not visible in frame %d (window %d); "
                    "dropping entire window per --on_missing_target=skip.",
                    target_object_id, t, window.window_idx,
                )
                return None

            target_img = crop_target_from_seg(
                rendered["rgb_static"], rendered["seg_static"],
                target_id=target_seg_id,
            )
            if target_img is None:
                if self.on_missing_target == "abort":
                    raise RuntimeError(
                        f"Target {target_object_id!r} crop missing in frame {t} "
                        f"(window {window.window_idx})"
                    )
                logger.warning(
                    "Target %r crop missing in frame %d (window %d); dropping "
                    "entire window per --on_missing_target=skip.",
                    target_object_id, t, window.window_idx,
                )
                return None
            image_targets.append(np.asarray(target_img, dtype=np.uint8))

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

            # CALVIN rel_actions layout: (dx, dy, dz, droll, dpitch, dyaw,
            # dgripper). See third_party/calvin/dataset/README.md for the
            # canonical definition. Assert the length so a future upstream
            # format change fails loudly here instead of silently miswiring
            # the action columns.
            assert len(rel_actions) == 7, (
                f"CALVIN rel_actions expected 7 dims, got {len(rel_actions)}"
            )
            action = rel_actions.tolist()  # 7 dims

            sample = {
                "episode_index": int(episode_index),
                "frame_index": int(step_idx),
                "timestamp": float(step_idx) / float(FPS),
                # Dataset-global index (LeRobot v2 invariant: contiguous
                # across every parquet in the dataset).
                "index": int(global_index),
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
                # action.* (split by component per modality.json).
                # Stored as 1-element lists so the LeRobot reader (which calls
                # np.stack on the per-step values) gets the expected 2D (T, D)
                # shape rather than a 1D (T,) scalar array.
                "action.x": [float(action[0])],
                "action.y": [float(action[1])],
                "action.z": [float(action[2])],
                "action.roll": [float(action[3])],
                "action.pitch": [float(action[4])],
                "action.yaw": [float(action[5])],
                "action.gripper": [float(action[6])],
                # Language column (free-form string; tasks.jsonl indexes it)
                "annotation.human.action.task_description":
                    window.instruction,
                # Provenance + sidecar lookup
                "trajectory_id": int(episode_index),
                "base_index": int(step_idx),
            }
            samples.append(sample)
            global_index += 1

        if not samples:
            return None

        tcp_trajectory = np.stack(tcp_positions).astype(np.float32)
        affordance_heatmaps = [
            self._target_point_affordance_heatmap(
                point_cloud=ctx["point_cloud"],
                tcp_positions=tcp_trajectory[i:],
                intrinsic=ctx["intrinsic"],
                cam_R=ctx["cam_R"],
                cam_t=ctx["cam_t"],
            )
            for i, ctx in enumerate(affordance_contexts)
        ]

        return _EpisodeBuffers(
            samples=samples,
            primary_frames=primary_frames,
            wrist_frames=wrist_frames,
            point_clouds=point_clouds,
            image_targets=image_targets,
            depth_targets=depth_targets,
            grounding_masks=grounding_masks,
            grounding_levels=grounding_levels,
            affordance_heatmaps=affordance_heatmaps,
            instruction=window.instruction,
        )

    # ------------------------------------------------------------------
    # Episode-level writers (from spec §4.3, Task 2.3 Steps 2-6)
    # ------------------------------------------------------------------

    def _emit_episode_parquet(
        self, episode_samples: list[dict],
        output_dir: Path, episode_index: int,
    ) -> None:
        episode_chunk = episode_index // LEROBOT_CHUNK_SIZE
        chunk_dir = output_dir / "data" / f"chunk-{episode_chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_dir / f"episode_{episode_index:06d}.parquet"
        table = pa.Table.from_pylist(episode_samples)
        pq.write_table(table, parquet_path)

    def _emit_episode_videos(
        self, primary_frames: list, wrist_frames: list,
        output_dir: Path, episode_index: int, fps: int = FPS,
    ) -> None:
        episode_chunk = episode_index // LEROBOT_CHUNK_SIZE
        video_dir = output_dir / "videos" / f"chunk-{episode_chunk:03d}"
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
        self, image_targets: list[np.ndarray], point_clouds: list,
        output_dir: Path, episode_index: int,
        depth_targets: list[np.ndarray] | None = None,
        grounding_masks: list[np.ndarray] | None = None,
        grounding_levels: list[str] | None = None,
        affordance_heatmaps: list[np.ndarray] | None = None,
    ) -> None:
        if len(image_targets) != len(point_clouds):
            raise RuntimeError(
                f"image_targets length {len(image_targets)} != "
                f"point_clouds length {len(point_clouds)}"
            )
        for name, values in [
            ("depth_targets", depth_targets),
            ("grounding_masks", grounding_masks),
            ("grounding_levels", grounding_levels),
            ("affordance_heatmaps", affordance_heatmaps),
        ]:
            if values is not None and len(values) != len(point_clouds):
                raise RuntimeError(
                    f"{name} length {len(values)} != point_clouds length "
                    f"{len(point_clouds)}"
                )
        img_target_dir = output_dir / "image_targets" / str(episode_index)
        img_target_dir.mkdir(parents=True, exist_ok=True)
        for base_index, image_target in enumerate(image_targets):
            Image.fromarray(np.asarray(image_target, dtype=np.uint8)).save(
                img_target_dir / f"{base_index}.png",
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
            if (
                depth_targets is not None
                or grounding_masks is not None
                or affordance_heatmaps is not None
            ):
                self._emit_aux_denoising_sidecars(
                    output_dir=output_dir,
                    trajectory_id=episode_index,
                    base_index=base_index,
                    depth_static=(
                        depth_targets[base_index]
                        if depth_targets is not None
                        else None
                    ),
                    grounding_mask=(
                        grounding_masks[base_index]
                        if grounding_masks is not None
                        else None
                    ),
                    affordance_heatmap=(
                        affordance_heatmaps[base_index]
                        if affordance_heatmaps is not None
                        else None
                    ),
                    grounding_level=(
                        grounding_levels[base_index]
                        if grounding_levels is not None
                        else "object"
                    ),
                )

    @staticmethod
    def _emit_aux_denoising_sidecars(
        output_dir: Path,
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

    @staticmethod
    def _grounding_level_for_target_id(target_id: str) -> str:
        return "part" if str(target_id).startswith("table__") else "object"

    @staticmethod
    def _empty_aux_coverage() -> dict[str, dict]:
        metrics = ("image_target", "point_cloud", "depth", "grounding", "affordance")
        return {
            "tasks": {},
            "totals": {
                name: {"valid": 0, "total": 0}
                for name in metrics
            },
        }

    @classmethod
    def _merge_aux_coverage(
        cls,
        coverage: dict[str, dict],
        task_name: str,
        lengths: dict[str, int],
    ) -> None:
        metrics = ("image_target", "point_cloud", "depth", "grounding", "affordance")
        task_bucket = coverage["tasks"].setdefault(
            task_name,
            {
                name: {"valid": 0, "total": 0}
                for name in metrics
            },
        )
        for name in metrics:
            count = int(lengths.get(name, 0))
            task_bucket[name]["valid"] += count
            task_bucket[name]["total"] += count
            coverage["totals"][name]["valid"] += count
            coverage["totals"][name]["total"] += count

    @staticmethod
    def _emit_aux_coverage(output_dir: Path, coverage: dict[str, dict]) -> None:
        meta_dir = Path(output_dir) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "uamvla_aux_coverage.json", "w") as f:
            json.dump(coverage, f, indent=2, sort_keys=True)

    @staticmethod
    def _segmentation_to_token_grid_mask(
        seg_mask: np.ndarray,
        target_id: int,
        target_size: int = 20,
    ) -> np.ndarray:
        seg = np.asarray(seg_mask)
        if seg.ndim == 3:
            seg = seg[..., -1]
        mask = (seg == target_id).astype(np.float32)
        img = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L")
        img = img.resize((target_size, target_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr[None, ...].astype(np.float32)

    @staticmethod
    def _gaussian_heatmap_from_pixel(
        pixel_xy: np.ndarray,
        image_width: int = RENDER_W,
        image_height: int = RENDER_H,
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
        max_value = float(heatmap.max())
        if max_value > 0.0:
            heatmap = heatmap / max_value
        return heatmap[None, ...].astype(np.float32)

    @classmethod
    def _target_point_affordance_heatmap(
        cls,
        point_cloud: np.ndarray,
        tcp_positions: np.ndarray,
        intrinsic: dict,
        cam_R: np.ndarray,
        cam_t: np.ndarray,
        target_size: int = 20,
    ) -> np.ndarray:
        points = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
        tcp = np.asarray(tcp_positions, dtype=np.float32).reshape(-1, 3)
        if points.size == 0 or tcp.size == 0:
            return np.zeros((1, target_size, target_size), dtype=np.float32)

        distances = np.linalg.norm(points[:, None, :] - tcp[None, :, :], axis=-1)
        target_point = points[np.argmin(distances.min(axis=1))]
        from starVLA.utils.vis_draw import world_to_pixel

        pixel = world_to_pixel(
            target_point[None, :],
            intrinsic=intrinsic,
            cam_R=np.asarray(cam_R, dtype=np.float32),
            cam_t=np.asarray(cam_t, dtype=np.float32),
        )[0]
        return cls._gaussian_heatmap_from_pixel(target_size=target_size, pixel_xy=pixel)

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
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "chunks_size": LEROBOT_CHUNK_SIZE,
            "features": {
                "video.primary_image": {
                    "dtype": "video",
                    "shape": [RENDER_H, RENDER_W, 3],
                    "names": ["height", "width", "channel"],
                    "info": {"video.fps": fps, "video.channels": 3},
                },
                "video.wrist_image": {
                    "dtype": "video",
                    "shape": [RENDER_H, RENDER_W, 3],
                    "names": ["height", "width", "channel"],
                    "info": {"video.fps": fps, "video.channels": 3},
                },
            },
        }
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)
