"""Replay official LIBERO HDF5 demos into UamVLA LeRobot datasets.

The module intentionally avoids importing LIBERO or robosuite at import time:
unit tests and training-side registry checks should run in the normal
``uamvla`` environment, while replay/render execution happens in
``libero_env``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tools.preprocess.base_preprocessor import BasePreprocessor
from tools.preprocess.libero_lerobot_writer import (
    FPS,
    LiberoEpisodeBuffers,
    LiberoLerobotWriter,
)
from tools.preprocess.libero_preprocess_utils import (
    compute_episode_plan,
    gaussian_heatmap_from_pixel,
    mask_to_token_grid,
    pack_robot_obs,
    pointcloud_to_tcp_distance,
    select_active_target_score_tcp,
    select_segment_aware_future_tcp,
    select_render_gpus,
    smooth_active_targets,
    task_name_from_hdf5,
)
from tools.preprocess.libero_target_mapping import (
    TaskTargetPolicy,
    get_task_policy,
    validate_policy_table,
)
from starVLA.utils.geometry import (
    crop_target_from_seg,
    depth_to_world_points,
    extract_instruction_from_filename,
    get_camera_intrinsic_from_fovy,
    linearize_depth,
    mat_to_6d,
)
from starVLA.utils.point_cloud import clean_point_cloud


logger = logging.getLogger(__name__)

STATIC_CAM = "agentview"
WRIST_CAM = "robot0_eye_in_hand"
RENDER_W = 256
RENDER_H = 256
NUM_POINTS = 1024


@dataclass(frozen=True)
class TaskJob:
    suite: str
    hdf5_path: Path
    output_dir: Path
    task_index: int
    demo_episode_indices: dict[int, int]
    demo_row_starts: dict[int, int]
    gpu_id: str
    min_segment_len: int
    active_target_score_window: int
    debug_rgb_check_frames: int
    max_demos_per_task: int | None = None
    max_frames_per_demo: int | None = None


class LiberoPreprocessor(BasePreprocessor):
    def __init__(
        self,
        suite: str = "libero_spatial",
        num_workers: int | None = None,
        render_gpus: str | None = None,
        min_segment_len: int = 3,
        active_target_score_window: int = 8,
        debug_rgb_check_frames: int = 3,
        max_tasks: int | None = None,
        max_demos_per_task: int | None = None,
        max_frames_per_demo: int | None = None,
    ):
        self.suite = suite
        self.num_workers = num_workers
        self.render_gpus = render_gpus
        self.min_segment_len = int(min_segment_len)
        self.active_target_score_window = int(active_target_score_window)
        self.debug_rgb_check_frames = int(debug_rgb_check_frames)
        self.max_tasks = max_tasks
        self.max_demos_per_task = max_demos_per_task
        self.max_frames_per_demo = max_frames_per_demo

    def process(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"LIBERO input directory not found: {input_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        missing = validate_policy_table(input_path.parent)
        suite_missing = [m for m in missing if m.startswith(f"{self.suite}/")]
        if suite_missing:
            raise RuntimeError(f"Missing curated target policies for {suite_missing}")

        hdf5_files = sorted(input_path.glob("*.hdf5"))
        if self.max_tasks is not None:
            hdf5_files = hdf5_files[: self.max_tasks]
        if not hdf5_files:
            raise FileNotFoundError(f"No .hdf5 files found in {input_path}")

        frame_counts = self._scan_frame_counts(hdf5_files)
        episode_plan = compute_episode_plan(frame_counts)
        render_gpus = select_render_gpus(
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            self.render_gpus,
        )
        worker_count = self.num_workers if self.num_workers is not None else len(render_gpus)
        worker_count = max(1, int(worker_count))
        jobs = self._build_jobs(hdf5_files, output_path, episode_plan, render_gpus)

        logger.info(
            "Processing %d LIBERO task files with %d worker(s) on render GPUs %s",
            len(jobs),
            worker_count,
            render_gpus,
        )
        if worker_count == 1:
            results = [_run_task_job(job) for job in jobs]
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=worker_count) as pool:
                results = pool.map(_run_task_job, jobs)

        writer = LiberoLerobotWriter(output_path, fps=FPS)
        tasks = [task_name_from_hdf5(p) for p in hdf5_files]
        episode_lengths: dict[int, int] = {}
        episode_to_task: dict[int, int] = {}
        total_frames = 0
        coverage: dict[str, Any] = {"tasks": {}, "totals": {}}
        camera_params = None
        for result in results:
            total_frames += int(result["total_frames"])
            episode_lengths.update(
                {int(k): int(v) for k, v in result["episode_lengths"].items()}
            )
            episode_to_task.update(
                {int(k): int(v) for k, v in result["episode_to_task"].items()}
            )
            coverage["tasks"][result["task_name"]] = result["coverage"]
            camera_params = camera_params or result.get("camera_params")
        coverage["totals"] = _sum_coverage(coverage["tasks"])
        writer.write_meta(tasks, episode_lengths, episode_to_task, total_frames, coverage)
        if camera_params is not None:
            writer.write_camera_params(camera_params)

    def _scan_frame_counts(self, hdf5_files: list[Path]) -> dict[str, list[int]]:
        counts: dict[str, list[int]] = {}
        for h5_path in hdf5_files:
            with h5py.File(h5_path, "r") as f:
                demo_keys = _sorted_demo_keys(f)
                if self.max_demos_per_task is not None:
                    demo_keys = demo_keys[: self.max_demos_per_task]
                lengths = []
                for demo_key in demo_keys:
                    length = int(f[f"data/{demo_key}/actions"].shape[0])
                    if self.max_frames_per_demo is not None:
                        length = min(length, int(self.max_frames_per_demo))
                    lengths.append(length)
                counts[h5_path.name] = lengths
        return counts

    def _build_jobs(
        self,
        hdf5_files: list[Path],
        output_path: Path,
        episode_plan: dict[tuple[str, int], Any],
        render_gpus: list[str],
    ) -> list[TaskJob]:
        jobs = []
        for task_idx, h5_path in enumerate(hdf5_files):
            demo_episode_indices = {}
            demo_row_starts = {}
            for (filename, demo_idx), plan in episode_plan.items():
                if filename == h5_path.name:
                    demo_episode_indices[demo_idx] = plan.episode_index
                    demo_row_starts[demo_idx] = plan.row_start
            jobs.append(
                TaskJob(
                    suite=self.suite,
                    hdf5_path=h5_path,
                    output_dir=output_path,
                    task_index=task_idx,
                    demo_episode_indices=demo_episode_indices,
                    demo_row_starts=demo_row_starts,
                    gpu_id=render_gpus[task_idx % len(render_gpus)],
                    min_segment_len=self.min_segment_len,
                    active_target_score_window=self.active_target_score_window,
                    debug_rgb_check_frames=self.debug_rgb_check_frames,
                    max_demos_per_task=self.max_demos_per_task,
                    max_frames_per_demo=self.max_frames_per_demo,
                )
            )
        return jobs

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
        LiberoLerobotWriter.write_aux_denoising_sidecar(
            output_dir=output_dir,
            trajectory_id=trajectory_id,
            base_index=base_index,
            depth_static=depth_static,
            grounding_mask=grounding_mask,
            affordance_heatmap=affordance_heatmap,
            grounding_level=grounding_level,
        )

    def probe_replay_environment(self) -> dict[str, bool]:
        gpu = select_render_gpus(
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            self.render_gpus,
        )[0]
        _configure_worker_env(gpu)
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        import robosuite

        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = suite.get_task(0)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            robots=["Panda"],
            controller="OSC_POSE",
            camera_names=[STATIC_CAM],
            camera_heights=64,
            camera_widths=64,
            camera_depths=True,
        )
        try:
            env.reset()
            rgb = env.sim.render(camera_name=STATIC_CAM, width=64, height=64)
            return {
                "libero_import": True,
                "robosuite_import": robosuite is not None,
                "render_ok": rgb is not None,
            }
        finally:
            env.close()


def _sorted_demo_keys(h5_file: h5py.File) -> list[str]:
    return sorted(h5_file["data"].keys(), key=lambda x: int(x.split("_")[1]))


def _configure_worker_env(gpu_id: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    # Editable robosuite installs may fail while creating numba cache locators.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


def _run_task_job(job: TaskJob) -> dict[str, Any]:
    _configure_worker_env(job.gpu_id)
    from libero.libero.envs import OffScreenRenderEnv

    policy = get_task_policy(job.suite, job.hdf5_path.name)
    bddl_path = _resolve_bddl_path(job.suite, job.hdf5_path)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        robots=["Panda"],
        controller="OSC_POSE",
        camera_names=[STATIC_CAM, WRIST_CAM],
        camera_heights=RENDER_H,
        camera_widths=RENDER_W,
        camera_depths=True,
        camera_segmentations="instance",
    )
    env.reset()
    try:
        worker = _TaskReplayWorker(job, env, policy)
        return worker.run()
    finally:
        env.close()


def _resolve_bddl_path(suite_name: str, hdf5_file: Path) -> str:
    from libero.libero import benchmark, get_libero_path

    task_name = task_name_from_hdf5(hdf5_file)
    bench_dict = benchmark.get_benchmark_dict()
    suite = bench_dict[suite_name]()
    bddl_root = get_libero_path("bddl_files")
    for idx in range(suite.n_tasks):
        task = suite.get_task(idx)
        if task.name == task_name:
            return str(Path(bddl_root) / task.problem_folder / task.bddl_file)
    raise ValueError(f"Task {task_name!r} not found in benchmark suite {suite_name!r}")


class _TaskReplayWorker:
    def __init__(self, job: TaskJob, env, policy: TaskTargetPolicy):
        self.job = job
        self.env = env
        self.policy = policy
        self.writer = LiberoLerobotWriter(job.output_dir, fps=FPS)
        self.task_name = task_name_from_hdf5(job.hdf5_path)
        self.instruction = extract_instruction_from_filename(job.hdf5_path.name)
        self._intrinsics = self._extract_camera_intrinsics()
        self._candidate_bodies = self._resolve_candidate_bodies()
        self._fallback_body = self._resolve_fallback_body()

    def run(self) -> dict[str, Any]:
        results = {
            "task_name": self.task_name,
            "total_frames": 0,
            "episode_lengths": {},
            "episode_to_task": {},
            "coverage": self._empty_coverage(),
            "camera_params": self._camera_params(),
        }
        with h5py.File(self.job.hdf5_path, "r") as f:
            demo_keys = _sorted_demo_keys(f)
            if self.job.max_demos_per_task is not None:
                demo_keys = demo_keys[: self.job.max_demos_per_task]
            for demo_key in demo_keys:
                demo_idx = int(demo_key.split("_")[1])
                buffers = self._process_demo(f[f"data/{demo_key}"], demo_idx)
                self.writer.write_episode(buffers)
                results["total_frames"] += len(buffers.rows)
                results["episode_lengths"][buffers.episode_index] = len(buffers.rows)
                results["episode_to_task"][buffers.episode_index] = self.job.task_index
                self._merge_coverage(results["coverage"], buffers)
        return results

    @staticmethod
    def _empty_coverage() -> dict[str, dict[str, int]]:
        return {
            "image_target": {"valid": 0, "total": 0},
            "point_cloud": {"valid": 0, "total": 0},
            "depth": {"valid": 0, "total": 0},
            "grounding": {"valid": 0, "total": 0},
            "affordance": {"valid": 0, "total": 0},
            "debug_rgb": {"valid": 0, "total": 0},
        }

    def _merge_coverage(
        self,
        coverage: dict[str, dict[str, int]],
        buffers: LiberoEpisodeBuffers,
    ) -> None:
        self._accumulate(coverage, "image_target", buffers.image_targets)
        self._accumulate(coverage, "point_cloud", buffers.point_clouds)
        self._accumulate(coverage, "depth", buffers.depth_targets)
        self._accumulate(coverage, "grounding", buffers.grounding_masks)
        self._accumulate(coverage, "affordance", buffers.affordance_heatmaps)

    @staticmethod
    def _accumulate(
        coverage: dict[str, dict[str, int]],
        name: str,
        values: list[np.ndarray | None],
    ) -> None:
        coverage[name]["total"] += len(values)
        coverage[name]["valid"] += sum(v is not None for v in values)

    def _camera_params(self) -> dict[str, float | int | str]:
        intr = self._intrinsics[STATIC_CAM]
        return {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
            "width": RENDER_W,
            "height": RENDER_H,
            "camera_name": STATIC_CAM,
        }

    def _extract_camera_intrinsics(self) -> dict[str, dict[str, float]]:
        result = {}
        for cam_name in [STATIC_CAM, WRIST_CAM]:
            cam_id = self.env.sim.model.camera_name2id(cam_name)
            fovy = self.env.sim.model.cam_fovy[cam_id]
            result[cam_name] = get_camera_intrinsic_from_fovy(fovy, RENDER_W, RENDER_H)
        return result

    def _process_demo(self, demo_grp, demo_idx: int) -> LiberoEpisodeBuffers:
        actions = np.asarray(demo_grp["actions"][()], dtype=np.float32)
        states = np.asarray(demo_grp["states"][()], dtype=np.float32)
        obs = demo_grp["obs"]
        ee_pos = np.asarray(obs["ee_pos"][()], dtype=np.float32)
        ee_ori = np.asarray(obs["ee_ori"][()], dtype=np.float32)
        joint_states = np.asarray(obs["joint_states"][()], dtype=np.float32)
        gripper_states = np.asarray(obs["gripper_states"][()], dtype=np.float32)

        length = min(actions.shape[0], states.shape[0], ee_pos.shape[0])
        if self.job.max_frames_per_demo is not None:
            length = min(length, int(self.job.max_frames_per_demo))
        if length <= 0:
            raise RuntimeError(f"{self.job.hdf5_path.name} demo_{demo_idx} has no frames")

        frame_payloads = []
        active_raw: list[str | None] = []
        for t in range(length):
            score_tcp = select_active_target_score_tcp(
                ee_pos[:length],
                frame_idx=t,
                window_size=self.job.active_target_score_window,
            )
            payload = self._extract_frame_payload(
                states[t],
                future_tcp_positions=score_tcp,
            )
            frame_payloads.append(payload)
            active_raw.append(payload["active_body"])
            self._maybe_log_rgb_debug(demo_grp, payload, t)

        active_smoothed = smooth_active_targets(
            active_raw,
            min_segment_len=self.job.min_segment_len,
        )
        active_for_affordance = [body or self._fallback_body for body in active_smoothed]
        episode_index = self.job.demo_episode_indices[demo_idx]
        global_index = self.job.demo_row_starts[demo_idx]

        rows: list[dict[str, Any]] = []
        primary_frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []
        image_targets: list[np.ndarray | None] = []
        point_clouds: list[np.ndarray | None] = []
        depth_targets: list[np.ndarray | None] = []
        grounding_masks: list[np.ndarray | None] = []
        grounding_levels: list[str | None] = []
        affordance_heatmaps: list[np.ndarray | None] = []

        for frame_idx, payload in enumerate(frame_payloads):
            active_body = active_smoothed[frame_idx] or self._fallback_body
            target = payload["bodies"].get(active_body)
            if target is None:
                target = payload["bodies"].get(self._fallback_body)
            if target is None:
                target = next(iter(payload["bodies"].values()))

            pc = target.get("point_cloud")
            heatmap = None
            if pc is not None:
                tcp_window = select_segment_aware_future_tcp(
                    ee_pos[:length],
                    active_targets=active_for_affordance,
                    frame_idx=frame_idx,
                )
                heatmap = self._target_point_affordance_heatmap(
                    point_cloud=pc,
                    tcp_positions=tcp_window,
                    intrinsic=payload["intrinsic"],
                    cam_R=payload["static_cam_mat"],
                    cam_t=payload["static_cam_pos"],
                )

            grounding_mask, grounding_level = self._grounding_mask_for_frame(
                payload,
                active_body,
                target["mask"],
            )
            image_target = crop_target_from_seg(
                payload["rgb_static_aligned"],
                target["mask"],
                target_id=1,
            )

            robot_obs = pack_robot_obs(
                ee_pos=ee_pos[frame_idx],
                ee_ori=ee_ori[frame_idx],
                joint_states=joint_states[frame_idx],
                gripper_states=gripper_states[frame_idx],
            )
            action = np.asarray(actions[frame_idx], dtype=np.float32).reshape(-1)
            if action.shape[0] != 7:
                raise RuntimeError(f"LIBERO action expected 7 dims, got {action.shape[0]}")

            row = {
                "episode_index": int(episode_index),
                "frame_index": int(frame_idx),
                "timestamp": float(frame_idx) / float(FPS),
                "index": int(global_index + frame_idx),
                "task_index": int(self.job.task_index),
                "state.robot_obs": robot_obs.tolist(),
                "state.target_pose_rot6d": [
                    float(v) for v in mat_to_6d(np.asarray(target["body_mat"]))
                ],
                "state.target_pose_trans": [
                    float(v) for v in np.asarray(target["body_pos"], dtype=np.float32)
                ],
                "state.static_cam_rot6d": [
                    float(v) for v in mat_to_6d(np.asarray(payload["static_cam_mat"]))
                ],
                "state.static_cam_trans": [
                    float(v) for v in np.asarray(payload["static_cam_pos"], dtype=np.float32)
                ],
                "action.x": [float(action[0])],
                "action.y": [float(action[1])],
                "action.z": [float(action[2])],
                "action.roll": [float(action[3])],
                "action.pitch": [float(action[4])],
                "action.yaw": [float(action[5])],
                "action.gripper": [float(action[6])],
                "annotation.human.action.task_description": self.instruction,
                "trajectory_id": int(episode_index),
                "base_index": int(frame_idx),
            }
            rows.append(row)
            primary_frames.append(np.asarray(payload["rgb_static"], dtype=np.uint8))
            wrist_frames.append(np.asarray(payload["rgb_wrist"], dtype=np.uint8))
            image_targets.append(
                np.asarray(image_target, dtype=np.uint8)
                if image_target is not None
                else None
            )
            point_clouds.append(pc)
            depth_targets.append(np.asarray(payload["depth_static"], dtype=np.float32))
            grounding_masks.append(grounding_mask)
            grounding_levels.append(grounding_level)
            affordance_heatmaps.append(heatmap)

        return LiberoEpisodeBuffers(
            episode_index=episode_index,
            task_index=self.job.task_index,
            task_name=self.task_name,
            rows=rows,
            primary_frames=primary_frames,
            wrist_frames=wrist_frames,
            image_targets=image_targets,
            point_clouds=point_clouds,
            depth_targets=depth_targets,
            grounding_masks=grounding_masks,
            grounding_levels=grounding_levels,
            affordance_heatmaps=affordance_heatmaps,
        )

    def _extract_frame_payload(
        self,
        state: np.ndarray,
        future_tcp_positions: np.ndarray,
    ) -> dict[str, Any]:
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()

        rgb_static = self._render_rgb(STATIC_CAM)
        rgb_static_aligned = self._render_rgb_aligned(STATIC_CAM)
        rgb_wrist = self._render_rgb(WRIST_CAM)
        depth_static = self._render_depth(STATIC_CAM)
        seg_instance = self._render_segmentation_instance(STATIC_CAM)
        seg_geom = self._render_segmentation_geom(STATIC_CAM)

        static_cam_id = self.env.sim.model.camera_name2id(STATIC_CAM)
        static_cam_pos = self.env.sim.data.cam_xpos[static_cam_id].copy()
        static_cam_mat = self.env.sim.data.cam_xmat[static_cam_id].reshape(3, 3).copy()

        bodies: dict[str, dict[str, Any]] = {}
        best_body = None
        best_score = float("inf")
        for body_name in self._candidate_bodies:
            body_payload = self._body_payload(
                body_name=body_name,
                seg_instance=seg_instance,
                depth_static=depth_static,
                rgb_static=rgb_static_aligned,
                intrinsic=self._intrinsics[STATIC_CAM],
                static_cam_mat=static_cam_mat,
                static_cam_pos=static_cam_pos,
                future_tcp_positions=future_tcp_positions,
            )
            bodies[body_name] = body_payload
            score = body_payload["score"]
            if score < best_score:
                best_body = body_name
                best_score = score

        if best_body is None or not np.isfinite(best_score):
            best_body = self._fallback_body

        return {
            "rgb_static": rgb_static,
            "rgb_static_aligned": rgb_static_aligned,
            "rgb_wrist": rgb_wrist,
            "depth_static": depth_static,
            "seg_instance": seg_instance,
            "seg_geom": seg_geom,
            "static_cam_pos": static_cam_pos,
            "static_cam_mat": static_cam_mat,
            "intrinsic": self._intrinsics[STATIC_CAM],
            "bodies": bodies,
            "active_body": best_body,
        }

    def _body_payload(
        self,
        body_name: str,
        seg_instance: np.ndarray,
        depth_static: np.ndarray,
        rgb_static: np.ndarray,
        intrinsic: dict[str, float],
        static_cam_mat: np.ndarray,
        static_cam_pos: np.ndarray,
        future_tcp_positions: np.ndarray,
    ) -> dict[str, Any]:
        body_id = self.env.sim.model.body_name2id(body_name)
        body_pos = self.env.sim.data.body_xpos[body_id].copy()
        body_mat = self.env.sim.data.body_xmat[body_id].reshape(3, 3).copy()
        geom_ids = self._geom_ids_for_body(body_name)
        instance_id = self._instance_id_for_body(body_name)
        if instance_id is None:
            mask = np.zeros(seg_instance.shape, dtype=np.uint8)
        else:
            mask = (seg_instance == instance_id).astype(np.uint8)
        point_cloud = self._point_cloud_from_mask(
            depth=depth_static,
            mask=mask,
            intrinsic=intrinsic,
            cam_R=static_cam_mat,
            cam_t=static_cam_pos,
        )
        score = pointcloud_to_tcp_distance(point_cloud, future_tcp_positions) if point_cloud is not None else float("inf")
        return {
            "body_name": body_name,
            "body_pos": body_pos,
            "body_mat": body_mat,
            "instance_id": instance_id,
            "geom_ids": geom_ids,
            "mask": mask,
            "point_cloud": point_cloud,
            "score": score,
            "image_target_visible": crop_target_from_seg(rgb_static, mask, target_id=1)
            is not None,
        }

    def _point_cloud_from_mask(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsic: dict[str, float],
        cam_R: np.ndarray,
        cam_t: np.ndarray,
    ) -> np.ndarray | None:
        pts = depth_to_world_points(
            depth=depth,
            seg_mask=mask,
            intrinsic=intrinsic,
            cam_R=cam_R,
            cam_t=cam_t,
            target_id=1,
            num_points=NUM_POINTS,
        )
        if pts is None:
            return None
        return clean_point_cloud(pts, num_points=NUM_POINTS).astype(np.float32)

    def _grounding_mask_for_frame(
        self,
        payload: dict[str, Any],
        active_body: str,
        object_mask: np.ndarray,
    ) -> tuple[np.ndarray | None, str | None]:
        part = self.policy.part_grounding
        if part.enabled and self._part_grounding_matches_active_body(part, active_body):
            part_mask = self._part_mask(payload["seg_geom"], active_body)
            if part_mask is not None and np.any(part_mask):
                return mask_to_token_grid(part_mask, target_size=20), part.grounding_level
        if object_mask is not None and np.any(object_mask):
            return mask_to_token_grid(object_mask, target_size=20), "object"
        return None, None

    @staticmethod
    def _part_grounding_matches_active_body(part, active_body: str) -> bool:
        return not part.body_patterns or any(
            pattern in active_body for pattern in part.body_patterns
        )

    def _part_mask(self, seg_geom: np.ndarray, active_body: str) -> np.ndarray | None:
        part = self.policy.part_grounding
        if not self._part_grounding_matches_active_body(part, active_body):
            return None
        body_patterns = part.body_patterns or (active_body,)
        geom_ids = []
        for geom_id, geom_name in enumerate(self.env.sim.model.geom_names):
            body_id = int(self.env.sim.model.geom_bodyid[geom_id])
            body_name = self.env.sim.model.body_id2name(body_id)
            if not any(pattern in body_name for pattern in body_patterns):
                continue
            if any(pattern in geom_name for pattern in part.geom_patterns):
                geom_ids.append(geom_id)
        if not geom_ids:
            return None
        return np.isin(seg_geom, geom_ids).astype(np.uint8)

    def _resolve_candidate_bodies(self) -> list[str]:
        bodies: list[str] = []
        for target in self.policy.candidate_targets:
            matches = self._match_body_names(target.body_pattern)
            for body in matches:
                if body not in bodies:
                    bodies.append(body)
        if not bodies:
            raise RuntimeError(f"No MuJoCo bodies match policy {self.policy.key}")
        return bodies

    def _resolve_fallback_body(self) -> str:
        matches = self._match_body_names(self.policy.fallback_target_pattern)
        if matches:
            return matches[0]
        return self._candidate_bodies[0]

    def _match_body_names(self, pattern: str) -> list[str]:
        excluded = ("robot", "gripper", "mount", "base", "world")
        matches = [
            str(name)
            for name in self.env.sim.model.body_names
            if pattern in str(name)
            and not any(str(name).startswith(prefix) for prefix in excluded)
        ]
        main_matches = [m for m in matches if m.endswith("_main")]
        return main_matches or matches

    def _geom_ids_for_body(self, body_name: str) -> list[int]:
        body_id = self.env.sim.model.body_name2id(body_name)
        geom_bodyids = self.env.sim.model.geom_bodyid
        return [i for i in range(len(geom_bodyids)) if int(geom_bodyids[i]) == body_id]

    def _instance_id_for_body(self, body_name: str) -> int | None:
        instance_name = self._instance_name_for_body(body_name)
        if instance_name is None:
            return None
        instance_names = list(self.env.env.model.instances_to_ids.keys())
        return instance_names.index(instance_name) + 1

    def _instance_name_for_body(self, body_name: str) -> str | None:
        instance_names = list(self.env.env.model.instances_to_ids.keys())
        candidates = [body_name]
        if body_name.endswith("_main"):
            candidates.append(body_name[: -len("_main")])
        for candidate in candidates:
            if candidate in instance_names:
                return candidate
        for instance_name in instance_names:
            if body_name.startswith(f"{instance_name}_"):
                return instance_name
        return None

    def _render_rgb(self, camera_name: str) -> np.ndarray:
        rgb = self.env.sim.render(
            camera_name=camera_name,
            width=RENDER_W,
            height=RENDER_H,
            depth=False,
        )
        if isinstance(rgb, tuple):
            rgb = rgb[0]
        return np.asarray(rgb[::-1].copy(), dtype=np.uint8)

    def _render_rgb_aligned(self, camera_name: str) -> np.ndarray:
        return self._render_rgb(camera_name)

    def _render_depth(self, camera_name: str) -> np.ndarray:
        extent = self.env.sim.model.stat.extent
        znear = self.env.sim.model.vis.map.znear * extent
        zfar = self.env.sim.model.vis.map.zfar * extent
        depth = self.env.sim.render(
            camera_name=camera_name,
            width=RENDER_W,
            height=RENDER_H,
            depth=True,
        )
        if isinstance(depth, tuple):
            depth = depth[1]
        depth = np.asarray(depth[::-1].copy(), dtype=np.float32)
        return linearize_depth(depth, znear, zfar).astype(np.float32)

    def _render_segmentation_instance(self, camera_name: str) -> np.ndarray:
        obs = self._get_observations(force_update=True)
        key = f"{camera_name}_segmentation_instance"
        if key not in obs:
            raise RuntimeError(f"Missing LIBERO segmentation observable: {key}")
        seg = np.asarray(obs[key])
        if seg.ndim == 3:
            seg = seg[..., 0]
        seg = self._align_observable_segmentation(seg)
        return np.asarray(seg, dtype=np.int32)

    def _render_segmentation_geom(self, camera_name: str) -> np.ndarray:
        seg = self.env.sim.render(
            camera_name=camera_name,
            width=RENDER_W,
            height=RENDER_H,
            segmentation=True,
        )
        if isinstance(seg, tuple):
            seg = seg[-1]
        seg = np.asarray(seg[::-1].copy())
        return seg[:, :, 1] if seg.ndim == 3 else seg

    def _get_observations(self, force_update: bool = False) -> dict[str, Any]:
        if hasattr(self.env, "env") and hasattr(self.env.env, "_get_observations"):
            try:
                return self.env.env._get_observations(force_update=force_update)
            except TypeError:
                return self.env.env._get_observations()
        return self.env._get_observations()

    @staticmethod
    def _align_observable_segmentation(seg: np.ndarray) -> np.ndarray:
        try:
            from robosuite import macros
        except ImportError:
            return seg
        if getattr(macros, "IMAGE_CONVENTION", "opengl") == "opengl":
            return seg[::-1].copy()
        return seg.copy()

    def _target_point_affordance_heatmap(
        self,
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
        return gaussian_heatmap_from_pixel(
            pixel,
            image_width=RENDER_W,
            image_height=RENDER_H,
            target_size=target_size,
        )

    def _maybe_log_rgb_debug(self, demo_grp, payload: dict[str, Any], frame_idx: int) -> None:
        if frame_idx >= self.job.debug_rgb_check_frames:
            return
        obs = demo_grp["obs"]
        if "agentview_rgb" not in obs:
            return
        hdf5_rgb = np.asarray(obs["agentview_rgb"][frame_idx], dtype=np.uint8)
        replay_rgb = payload["rgb_static"]
        if hdf5_rgb.shape != replay_rgb.shape:
            from PIL import Image

            replay_small = np.asarray(
                Image.fromarray(replay_rgb).resize(
                    (hdf5_rgb.shape[1], hdf5_rgb.shape[0])
                ),
                dtype=np.uint8,
            )
        else:
            replay_small = replay_rgb
        mae = float(
            np.abs(
                replay_small.astype(np.float32) - hdf5_rgb.astype(np.float32)
            ).mean()
        )
        logger.info(
            "RGB debug %s frame %d replay-vs-hdf5 MAE %.3f",
            self.task_name,
            frame_idx,
            mae,
        )


def _sum_coverage(tasks: dict[str, dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for task_coverage in tasks.values():
        for name, counts in task_coverage.items():
            if name not in totals:
                totals[name] = {"valid": 0, "total": 0}
            totals[name]["valid"] += int(counts.get("valid", 0))
            totals[name]["total"] += int(counts.get("total", 0))
    return totals
