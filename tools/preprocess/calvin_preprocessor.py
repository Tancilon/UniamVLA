"""Convert CALVIN .npz-per-frame datasets to starVLA unified format.

Relies on calvin_env (PyBullet) for scene replay + rendering.  Structurally
mirrors LiberoPreprocessor — see that file for the equivalent HDF5-based
implementation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
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

logger = logging.getLogger(__name__)


# ---------- Language window collector ---------------------------------------

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


# ---------- Scene config resolver -------------------------------------------

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


# ---------- CalvinWorker -------------------------------------------------------


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

    def process_window(
        self,
        window: LangWindow,
        input_dir: Path,
        output_dir: Path,
    ) -> Path | None:
        # Caller is responsible for creating the output subdirs
        # (images/obs/static, images/obs/wrist, images/target, images/future,
        # depth/static, depth/wrist, point_clouds).  process_window creates
        # only the shards/ subdir lazily.
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        shard_dir = output_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / (
            f"shard_{self.rank:04d}_win{window.window_idx:04d}.jsonl"
        )

        # Resolve target object before entering the frame loop — one check
        # per window.
        #
        # Two paths:
        #   1. `stack_block` / `unstack_block` are color-agnostic in CALVIN
        #      — the manipulated block varies per trajectory. Infer from
        #      the window's start/end scene_obs (largest xyz delta = the
        #      block the robot moved). Scene-letter context is required
        #      because each CALVIN scene yaml lists `movable_objects` in
        #      a different order. Failures here (missing npz, corrupt
        #      scene_obs, unknown scene letter) propagate as-is — they
        #      indicate dataset corruption, not the vocabulary gap that
        #      `on_resolve_failure="skip"` is for.
        #   2. All other tasks: the deterministic static map. Unknown
        #      labels raise KeyError, which `on_resolve_failure="skip"`
        #      converts to a dropped window.
        if window.task_label in STACK_TASKS:
            scene_letter = self.scene_resolver.resolve_for_window(window)
            start_so = self._load_scene_obs(input_dir, window.ep_start)
            end_so = self._load_scene_obs(input_dir, window.ep_end)
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
                        window.window_idx,
                        window.task_label,
                    )
                    return None
                raise

        # Ask the env for the PyBullet seg-mask id belonging to our target.
        # One call per window — body/link→id is static across a scene.
        target_seg_id = self.env.get_target_seg_id(target_object_id)

        frames = list(range(window.ep_start, window.ep_end + 1))
        total_steps = len(frames)
        episode_id = f"{self.dataset_source}_ep{window.window_idx:05d}"
        future_rel = f"images/future/{episode_id}.jpg"

        rows: list[dict] = []
        last_rgb_static: np.ndarray | None = None

        for step_idx, t in enumerate(frames):
            with np.load(input_dir / f"episode_{t:07d}.npz") as npz:
                rel_actions = np.asarray(npz["rel_actions"], dtype=np.float32)
                robot_obs   = np.asarray(npz["robot_obs"],   dtype=np.float32)
                scene_obs   = np.asarray(npz["scene_obs"],   dtype=np.float32)

            self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            rendered = self.env.render_cameras(width=RENDER_W, height=RENDER_H)
            obj_pos, obj_mat = self.env.get_object_pose(target_object_id)

            sample_id = f"{episode_id}_step{step_idx:04d}"

            # --- Always-write fields (no seg dependency) ---
            static_rel = f"images/obs/static/{sample_id}.jpg"
            wrist_rel  = f"images/obs/wrist/{sample_id}.jpg"
            Image.fromarray(rendered["rgb_static"]).save(output_dir / static_rel, quality=95)
            Image.fromarray(rendered["rgb_wrist"]).save(output_dir / wrist_rel, quality=95)

            depth_static_rel = f"depth/static/{sample_id}.npy"
            depth_wrist_rel  = f"depth/wrist/{sample_id}.npy"
            np.save(output_dir / depth_static_rel, rendered["depth_static"].astype(np.float32))
            np.save(output_dir / depth_wrist_rel,  rendered["depth_wrist"].astype(np.float32))

            # G2 fix: anchor image_future regardless of seg outcome.
            last_rgb_static = rendered["rgb_static"]

            # --- Seg-dependent fields: try, omit on failure ---
            pc_rel_present:     str | None = None
            target_rel_present: str | None = None

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
                pc_rel = f"point_clouds/{sample_id}.npy"
                np.save(output_dir / pc_rel, pts)
                pc_rel_present = pc_rel
            elif self.on_missing_target == "abort":
                raise RuntimeError(
                    f"Target {target_object_id!r} not visible in frame {t} "
                    f"(window {window.window_idx})"
                )
            # else (skip mode): leave pc_rel_present = None

            target_img = crop_target_from_seg(
                rendered["rgb_static"], rendered["seg_static"], target_id=target_seg_id,
            )
            if target_img is not None:
                target_rel = f"images/target/{sample_id}.jpg"
                target_img.save(output_dir / target_rel, quality=95)
                target_rel_present = target_rel
            elif self.on_missing_target == "abort":
                raise RuntimeError(
                    f"Seg crop failed for frame {t} (window {window.window_idx})"
                )

            # Replace the old "Skipping frame ..." log with a wording that reflects
            # the new behaviour (we now retain the frame, just drop the seg-dependent
            # fields). Operators see one line per partial-aux frame; verification in
            # §4.2/§4.3 cross-checks this against JSONL counts but does not require
            # log-count to match (avoids coupling validation to log format).
            if pc_rel_present is None or target_rel_present is None:
                logger.info(
                    "frame %d in window %d: seg failed for target %r — emitting "
                    "partial-aux row (image_target=%s, point_cloud=%s, pose_6d=%s)",
                    t, window.window_idx, target_object_id,
                    "kept" if target_rel_present else "omitted",
                    "kept" if pc_rel_present     else "omitted",
                    "kept" if pc_rel_present     else "omitted",  # pose_6d gated on pc
                )

            # --- Assemble row (always-fields + present aux-fields) ---
            action_7d  = rel_actions.tolist()
            action_24d = action_7d + [0.0] * (MAX_ACTION_DIM - FRANKA_ACTION_DIM)
            row = {
                "id":           sample_id,
                "episode_id":   episode_id,
                "step_idx":     step_idx,
                "total_steps":  total_steps,
                "image":        [static_rel, wrist_rel],
                "instruction":  window.instruction,
                "embodiment":   EMBODIMENT,
                "action_dim":   FRANKA_ACTION_DIM,
                "action":       action_24d,
                "action_mask":  ACTION_MASK,
                "robot_obs":    robot_obs.tolist(),
                "dataset_source": self.dataset_source,
                "image_future": future_rel,
                "depth_static": depth_static_rel,
                "depth_wrist":  depth_wrist_rel,
                "static_cam_extrinsic": {
                    "rotation":    np.asarray(rendered["static_cam_R"]).flatten().tolist(),
                    "translation": np.asarray(rendered["static_cam_t"]).tolist(),
                },
                "wrist_cam_extrinsic": {
                    "rotation":    np.asarray(rendered["wrist_cam_R"]).flatten().tolist(),
                    "translation": np.asarray(rendered["wrist_cam_t"]).tolist(),
                },
            }

            # Seg-dependent. pose_6d is gated on point_cloud because pose head's
            # forward asserts batch["point_cloud"] is present (pose_head.py:166).
            if pc_rel_present is not None:
                row["point_cloud"] = pc_rel_present
                row["pose_6d"] = {
                    "rotation":    mat_to_6d(np.asarray(obj_mat)),
                    "translation": np.asarray(obj_pos).tolist(),
                    "object_id":   target_object_id,
                }
            if target_rel_present is not None:
                row["image_target"] = target_rel_present

            rows.append(row)

        # image_future: write last frame's static RGB once per window.
        # If every frame was skipped (on_missing_target="skip" + full-window
        # occlusion), emit no shard and no future image — there's nothing
        # to anchor either to.
        if not rows:
            logger.warning(
                "Window %d produced no samples (all frames skipped); "
                "dropping window.", window.window_idx,
            )
            return None
        assert last_rgb_static is not None
        Image.fromarray(last_rgb_static).save(
            output_dir / future_rel, quality=95,
        )

        with open(shard_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return shard_path

    @staticmethod
    def _load_scene_obs(input_dir: Path, frame: int) -> np.ndarray:
        """Load only the `scene_obs` array from one episode npz.

        Wrapped in a context manager so the underlying file handle and
        decompression buffers are released before we open the next one
        (np.load on .npz returns an NpzFile that defers extraction).
        """
        path = input_dir / f"episode_{frame:07d}.npz"
        with np.load(path) as npz:
            return np.asarray(npz["scene_obs"], dtype=np.float32)


# ---------- Shard merger + statistics writer --------------------------------


def merge_shards_and_write_stats(
    output_dir: Path,
    camera_intrinsics: dict,
) -> None:
    """Concat output_dir/shards/*.jsonl → output_dir/data.jsonl (sorted by id)
    and write output_dir/statistics.yaml over the merged sample list.
    """
    output_dir = Path(output_dir)
    shard_dir = output_dir / "shards"
    shard_files = sorted(shard_dir.glob("shard_*.jsonl"))
    if not shard_files:
        raise RuntimeError(f"No shard files found in {shard_dir}")

    all_samples: list[dict] = []
    for sp in shard_files:
        with open(sp) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json.loads(line))
    all_samples.sort(key=lambda r: r["id"])

    with open(output_dir / "data.jsonl", "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    _write_statistics(all_samples, output_dir, camera_intrinsics)

    logger.info(
        f"Merged {len(shard_files)} shards into "
        f"{output_dir / 'data.jsonl'} ({len(all_samples)} samples)"
    )


def _write_statistics(
    samples: list[dict],
    output_dir: Path,
    camera_intrinsics: dict,
) -> None:
    from starVLA.model.modules.uamvla.data.embodiment_adapter import CalvinAdapter

    if len(samples) < 2:
        logger.warning(
            f"Computing statistics over only {len(samples)} sample(s); "
            f"q01/q99/std will be degenerate. Consider preprocessing more episodes."
        )

    actions_7d = np.array(
        [s["action"][:FRANKA_ACTION_DIM] for s in samples], dtype=np.float64,
    )

    # Per-field stats over the CANONICAL representation produced by CalvinAdapter,
    # keyed by dotted path matching the canonical_state nested dict.
    adapter = CalvinAdapter()
    field_buffers: dict[str, list[np.ndarray]] = {
        "arm_0.ee_pose":   [],
        "arm_0.joint_pos": [],
        "gripper_0":       [],
    }
    for s in samples:
        canonical = adapter.to_canonical(s)  # reads s["robot_obs"] (15-dim)
        field_buffers["arm_0.ee_pose"].append(canonical["arm_0"]["ee_pose"].numpy())
        field_buffers["arm_0.joint_pos"].append(canonical["arm_0"]["joint_pos"].numpy())
        field_buffers["gripper_0"].append(canonical["gripper_0"].numpy())

    franka_state_stats: dict[str, dict] = {}
    for path, vals in field_buffers.items():
        arr = np.stack(vals).astype(np.float64)  # (N, D)
        franka_state_stats[path] = {
            "q01":  np.quantile(arr, 0.01, axis=0).tolist(),
            "q99":  np.quantile(arr, 0.99, axis=0).tolist(),
            "min":  arr.min(axis=0).tolist(),
            "max":  arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).tolist(),
        }

    stats = {
        "view_names": ["static", "wrist"],
        "max_action_dim": MAX_ACTION_DIM,
        "embodiment_stats": {
            EMBODIMENT: {
                "action_dim": FRANKA_ACTION_DIM,
                "action_min_bound": actions_7d.min(axis=0).tolist(),
                "action_max_bound": actions_7d.max(axis=0).tolist(),
            },
        },
        "state_stats": {EMBODIMENT: franka_state_stats},
        "cameras": {
            "static": {"intrinsic": camera_intrinsics["static"]},
            "wrist":  {"intrinsic": camera_intrinsics["wrist"]},
        },
        "point_cloud": {"num_points": NUM_POINTS, "frame": "world"},
    }
    with open(output_dir / "statistics.yaml", "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Wrote statistics.yaml ({len(samples)} samples)")


# ---------- CalvinPreprocessor orchestrator ----------------------------------


class CalvinPreprocessor(BasePreprocessor):
    """Top-level CALVIN → starVLA conversion driver.

    The actual per-frame work lives in CalvinWorker.  This class decides
    how windows are assigned to workers and how shard outputs are merged.
    """

    # Subdirs under output_dir that every sample writes into.
    _OUTPUT_SUBDIRS = (
        "images/obs/static",
        "images/obs/wrist",
        "images/target",
        "images/future",
        "depth/static",
        "depth/wrist",
        "point_clouds",
        "shards",
    )

    # multiprocessing.get_context name. MUST be "spawn" for any run that
    # touches GPU/EGL: after the main-process intrinsics probe builds a
    # PyBullet env, fork()ed children inherit the parent's GPU handles and
    # the NVIDIA driver rejects the child's eglMakeCurrent with
    # EGL_BAD_ALLOC (0x3003). Spawn gives each child a clean interpreter
    # with zero inherited GPU state. Tests override to "fork" so
    # monkeypatches applied in the main process propagate to children.
    _POOL_CONTEXT = "spawn"

    def __init__(
        self,
        dataset_source: str,
        default_scene: str | None,
        num_workers: int = 1,
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
        self.dataset_source = dataset_source
        self.default_scene = default_scene
        self.num_workers = num_workers
        self.on_resolve_failure = on_resolve_failure
        self.on_missing_target = on_missing_target

    def process(self, input_dir, output_dir):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        for sub in self._OUTPUT_SUBDIRS:
            (output_dir / sub).mkdir(parents=True, exist_ok=True)

        windows = collect_lang_windows(input_dir)
        if not windows:
            raise RuntimeError(
                f"No language windows found in {input_dir}; nothing to do."
            )
        logger.info(f"Collected {len(windows)} language windows")

        # SceneResolver is invoked for its side effect of validating
        # scene_info.npy / default_scene, and its output is logged for
        # provenance. It does NOT drive env construction any more —
        # PlayTableSimEnv picks its scene from the dataset path's hydra
        # conf.
        scene_resolver = SceneResolver(
            split_dir=input_dir, default_scene=self.default_scene,
        )
        try:
            resolved = {
                scene_resolver.resolve_for_window(w) for w in windows
            }
            logger.info(
                "Scene letters covered by this split (informational): %s",
                sorted(resolved),
            )
        except SceneResolveError:
            # Re-raise — a window not covered by scene_info.npy is a
            # dataset malformation we want to fail fast on.
            raise

        dataset_path = str(input_dir)

        camera_intrinsics = None
        if self.num_workers == 1:
            # Inline path — also used by unit tests. One env per worker,
            # reused across all windows via env.reset(robot_obs, scene_obs).
            # Rebuilding PyBullet/EGL per window triggers a reconnect path
            # that silently corrupts the physics client and surfaces as
            # "Not connected to physics server" on window 1+.
            worker = self._build_worker(rank=0, dataset_path=dataset_path)
            try:
                for window in windows:
                    shard = worker.process_window(
                        window=window,
                        input_dir=input_dir,
                        output_dir=output_dir,
                    )
                    if shard is not None and camera_intrinsics is None:
                        camera_intrinsics = self._intrinsics_from_env(worker.env)
            finally:
                worker.env.close()
        else:
            camera_intrinsics = self._dispatch_multiprocess(
                windows=windows,
                dataset_path=dataset_path,
                input_dir=input_dir,
                output_dir=output_dir,
            )

        if camera_intrinsics is None:
            raise RuntimeError("No samples produced; cannot write statistics.")

        merge_shards_and_write_stats(output_dir, camera_intrinsics)
        logger.info(f"Done. Output: {output_dir}")

    # --- Hooks that tests can monkeypatch ---------------------------------

    def _build_worker(self, rank: int, dataset_path: str) -> "CalvinWorker":
        """Real impl: construct PlayTableSimEnv via the dataset hydra config
        stored at `dataset_path`, wrap it in a CalvinEnvAdapter exposing
        the {reset, render_cameras, get_object_pose, get_target_seg_id,
        close} surface, return a CalvinWorker.

        Tests monkeypatch this to inject a fake env.
        """
        from tools.preprocess.calvin_env_adapter import (
            make_calvin_env_adapter,
        )
        env = make_calvin_env_adapter(dataset_path=dataset_path)
        scene_resolver = SceneResolver(
            split_dir=Path(dataset_path), default_scene=self.default_scene,
        )
        return CalvinWorker(
            env=env,
            dataset_source=self.dataset_source,
            rank=rank,
            scene_resolver=scene_resolver,
            on_resolve_failure=self.on_resolve_failure,
            on_missing_target=self.on_missing_target,
        )

    def _intrinsics_from_env(self, env) -> dict:
        """Extract the static+wrist intrinsics once from a live env.

        Intrinsics are camera-state-independent; we render once to read
        them. The env is typically in its post-last-reset state at this
        point, which is fine.
        """
        rendered = env.render_cameras(width=RENDER_W, height=RENDER_H)
        return {
            "static": rendered["static_intrinsic"],
            "wrist": rendered["wrist_intrinsic"],
        }

    def _dispatch_multiprocess(
        self,
        windows,
        dataset_path,
        input_dir,
        output_dir,
    ) -> dict:
        """multiprocessing.Pool fan-out, one persistent env per child.

        Each Pool child builds its CalvinEnvAdapter once in the `initializer`
        callback and reuses it across every window the imap queue feeds it.
        Rebuilding per task is what triggers the PyBullet
        reconnect-in-process crash — see `_init_worker_env` for the failure
        mode.

        Camera intrinsics are read from a short-lived probe env in the main
        process before the pool starts, so we don't need IPC for them.
        """
        from multiprocessing import get_context
        # Intrinsics probe: spin up one env transiently, read intrinsics, close.
        probe_worker = self._build_worker(rank=0, dataset_path=dataset_path)
        try:
            intrinsics = self._intrinsics_from_env(probe_worker.env)
        finally:
            probe_worker.env.close()

        tasks = [(w, str(input_dir), str(output_dir)) for w in windows]
        ctx = get_context(self._POOL_CONTEXT)
        with ctx.Pool(
            processes=self.num_workers,
            initializer=_init_worker_env,
            initargs=(
                dataset_path,
                self.dataset_source,
                self.default_scene,
                self.on_resolve_failure,
                self.on_missing_target,
            ),
        ) as pool:
            for shard in pool.imap_unordered(_worker_entrypoint, tasks):
                if shard:
                    logger.info(f"Finished shard {shard}")
                else:
                    logger.debug("Worker returned None shard (window skipped)")
        return intrinsics


# Module-level state populated once per child by `_init_worker_env`.
# Each Pool child has its own copy of these globals; the main process's
# values stay None.
_WORKER_ENV = None
_WORKER_META: tuple | None = None


def _init_worker_env(
    dataset_path: str,
    dataset_source: str,
    default_scene: str | None,
    on_resolve_failure: str,
    on_missing_target: str,
) -> None:
    """Pool initializer: build one PyBullet env per child process.

    Why not per task: closing+reopening a CalvinEnvAdapter inside the same
    process reloads the EGL plugin, which silently corrupts the PyBullet
    physics client. The corruption surfaces on the next
    `p.resetJointState` as 'Not connected to physics server' — the exact
    bug we fixed for num_workers=1. Per-child-once avoids it by never
    tearing down the env within a child's lifetime; the child exits when
    the pool terminates and the OS reclaims everything.

    A per-child `SceneResolver` is constructed here from the split's
    scene_info.npy (or the CLI default_scene). It's pure-Python and tied
    to the dataset path, so we build it locally rather than pickling
    across the spawn boundary.
    """
    global _WORKER_ENV, _WORKER_META
    from tools.preprocess.calvin_env_adapter import (
        make_calvin_env_adapter,
    )
    _WORKER_ENV = make_calvin_env_adapter(dataset_path=dataset_path)
    scene_resolver = SceneResolver(
        split_dir=Path(dataset_path), default_scene=default_scene,
    )
    _WORKER_META = (
        dataset_source, scene_resolver, on_resolve_failure, on_missing_target,
    )


def _worker_entrypoint(task_tuple) -> str:
    """Per-task callable for Pool.imap_unordered. Reuses the module-level
    env initialized once by `_init_worker_env`.

    rank is hardcoded to 0; multiprocess shard uniqueness comes from the
    per-window index in the filename (shard_0000_winNNNN).

    Returns str(shard_path) on success, or "" if the window was skipped
    (on_resolve_failure == "skip" and task_label was unknown).
    """
    window, input_dir, output_dir = task_tuple
    (
        dataset_source, scene_resolver, on_resolve_failure, on_missing_target,
    ) = _WORKER_META
    worker = CalvinWorker(
        env=_WORKER_ENV,
        dataset_source=dataset_source,
        rank=0,
        scene_resolver=scene_resolver,
        on_resolve_failure=on_resolve_failure,
        on_missing_target=on_missing_target,
    )
    shard = worker.process_window(
        window=window,
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
    )
    return str(shard) if shard is not None else ""
