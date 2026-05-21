"""Convert LIBERO HDF5 datasets to UamVLA unified format.

Requires: mujoco, robosuite, bddl, libero (pip install -e third_party/LIBERO)
"""
import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from tools.preprocess.base_preprocessor import BasePreprocessor
from tools.preprocess.target_object_resolver import (
    LLMTargetResolver,
    TargetResolveError,
)
from starVLA.utils.geometry import (
    crop_target_from_seg,
    depth_to_world_points,
    extract_instruction_from_filename,
    get_camera_intrinsic_from_fovy,
    linearize_depth,
    mat_to_6d,
)

logger = logging.getLogger(__name__)

# Cross-embodiment constants (from dataset format spec)
MAX_ACTION_DIM = 24
FRANKA_ACTION_DIM = 7
ACTION_MASK = [1] * FRANKA_ACTION_DIM + [0] * (MAX_ACTION_DIM - FRANKA_ACTION_DIM)

# LIBERO camera names
STATIC_CAM = "agentview"
WRIST_CAM = "robot0_eye_in_hand"

# Rendering resolution
RENDER_W = 256
RENDER_H = 256

# Point cloud config
NUM_POINTS = 1024


class LiberoPreprocessor(BasePreprocessor):
    """Convert LIBERO HDF5 demos to UamVLA unified dataset format."""

    def __init__(
        self,
        suite: str = "libero_spatial",
        target_object_keyword: str | None = None,
        target_resolver: LLMTargetResolver | None = None,
        skip_on_resolve_failure: bool = False,
    ):
        self.suite = suite
        self.target_object_keyword = target_object_keyword
        self._target_resolver = target_resolver
        self._skip_on_resolve_failure = skip_on_resolve_failure
        # Resolved at runtime per-task from MuJoCo body names
        self._target_body_name: str | None = None
        # Counters and skip log populated by process()
        self._resolve_stats = {"llm_hit": 0, "keyword_hit": 0, "skipped": 0}
        self._skipped_samples: list[tuple[str, str]] = []  # (filename, reason)

    def process(self, input_dir: str, output_dir: str):
        """Read LIBERO HDF5 files, append to UamVLA unified format dataset.

        Supports incremental processing: multiple suites can be processed
        sequentially into the same output_dir. New samples are appended to
        data.jsonl, files are added to existing directories, and
        statistics.yaml is recomputed over all accumulated data.

        Raises FileExistsError if any output file already exists (duplicate
        sample IDs across suites indicate a naming collision).
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Create output directories (idempotent)
        for subdir in [
            "images/obs/static",
            "images/obs/wrist",
            "images/target",
            "images/future",
            "point_clouds",
            "depth/static",
            "depth/wrist",
            "depths/static",
            "grounding_masks/static",
            "affordance_heatmaps/static",
        ]:
            (output_path / subdir).mkdir(parents=True, exist_ok=True)

        hdf5_files = sorted(input_path.glob("*.hdf5"))
        if not hdf5_files:
            raise FileNotFoundError(f"No .hdf5 files found in {input_path}")

        logger.info(f"Found {len(hdf5_files)} HDF5 files in {input_path}")

        # Load existing sample IDs to detect duplicates
        existing_ids = self._load_existing_ids(output_path)
        if existing_ids:
            logger.info(
                f"Found {len(existing_ids)} existing samples in {output_path}"
            )

        new_samples = []
        camera_intrinsics = None
        # Reset per-run counters
        self._resolve_stats = {"llm_hit": 0, "keyword_hit": 0, "skipped": 0}
        self._skipped_samples = []

        for task_idx, hdf5_file in enumerate(hdf5_files):
            logger.info(
                f"Processing task {task_idx}/{len(hdf5_files)}: {hdf5_file.name}"
            )
            instruction = extract_instruction_from_filename(hdf5_file.name)

            # Create LIBERO environment for MuJoCo replay
            env = self._create_env(hdf5_file)
            try:
                self._target_body_name = self._resolve_target_body(
                    env, instruction
                )
            except TargetResolveError as e:
                env.close()
                if self._skip_on_resolve_failure:
                    self._resolve_stats["skipped"] += 1
                    self._skipped_samples.append((hdf5_file.name, str(e)))
                    logger.warning(
                        f"  SKIPPED {hdf5_file.name}: {e}"
                    )
                    continue
                raise

            try:
                # Extract camera intrinsics once (intrinsic is constant in LIBERO;
                # static cam extrinsic is now stored per-sample, see _replay_and_extract)
                if camera_intrinsics is None:
                    camera_intrinsics = self._extract_camera_intrinsics(env)

                with h5py.File(hdf5_file, "r") as f:
                    demo_keys = sorted(
                        f["data"].keys(), key=lambda x: int(x.split("_")[1])
                    )

                    for demo_key in demo_keys:
                        demo_idx = int(demo_key.split("_")[1])
                        demo_grp = f[f"data/{demo_key}"]

                        actions = demo_grp["actions"][()]
                        states = demo_grp["states"][()]
                        ee_pos = demo_grp["obs/ee_pos"][()]                 # (T, 3)
                        ee_axis_angle = demo_grp["obs/ee_ori"][()]          # (T, 3) axis-angle rotvec
                        joint_pos = demo_grp["obs/joint_states"][()]        # (T, 7)
                        gripper_qpos = demo_grp["obs/gripper_states"][()]   # (T, 2)

                        T = actions.shape[0]
                        episode_id = (
                            f"{self.suite}_{task_idx:02d}_ep{demo_idx:04d}"
                        )

                        for t in range(T):
                            sample_id = f"{episode_id}_step{t:04d}"

                            # Duplicate check
                            if sample_id in existing_ids:
                                raise FileExistsError(
                                    f"Duplicate sample ID '{sample_id}' — "
                                    f"this suite may have been processed already"
                                )

                            # All rendering from MuJoCo replay (256x256, unified resolution)
                            extracted = self._replay_and_extract(
                                env,
                                states[t],
                                camera_intrinsics,
                                future_tcp_positions=ee_pos[t:],
                            )

                            # Save rendered RGB images
                            static_img_path = (
                                f"images/obs/static/{sample_id}.jpg"
                            )
                            wrist_img_path = (
                                f"images/obs/wrist/{sample_id}.jpg"
                            )
                            self._save_file(
                                output_path / static_img_path,
                                lambda p: Image.fromarray(
                                    extracted["rgb_static"]
                                ).save(p, quality=95),
                            )
                            self._save_file(
                                output_path / wrist_img_path,
                                lambda p: Image.fromarray(
                                    extracted["rgb_wrist"]
                                ).save(p, quality=95),
                            )

                            # Save image_future (last frame rendered from MuJoCo)
                            future_path = f"images/future/{episode_id}.jpg"
                            if t == T - 1:
                                self._save_file(
                                    output_path / future_path,
                                    lambda p: Image.fromarray(
                                        extracted["rgb_static"]
                                    ).save(p, quality=95),
                                )

                            # Action: pad to 24D
                            action_7d = actions[t].tolist()
                            action_24d = action_7d + [0.0] * (
                                MAX_ACTION_DIM - FRANKA_ACTION_DIM
                            )

                            sample = {
                                "id": sample_id,
                                "episode_id": episode_id,
                                "step_idx": t,
                                "total_steps": T,
                                "image": [static_img_path, wrist_img_path],
                                "instruction": instruction,
                                "embodiment": "franka_libero",
                                "action_dim": FRANKA_ACTION_DIM,
                                "action": action_24d,
                                "action_mask": ACTION_MASK,
                                "robot_obs": ee_pos[t].tolist(),  # kept for backward compat
                                "ee_pos": ee_pos[t].tolist(),
                                "ee_axis_angle": ee_axis_angle[t].tolist(),
                                "joint_pos": joint_pos[t].tolist(),
                                "gripper_qpos": gripper_qpos[t].tolist(),
                                "dataset_source": self.suite,
                                "image_future": future_path,
                            }

                            # Save depth maps
                            depth_static_path = (
                                f"depth/static/{sample_id}.npy"
                            )
                            depth_wrist_path = f"depth/wrist/{sample_id}.npy"
                            self._save_file(
                                output_path / depth_static_path,
                                lambda p: np.save(
                                    p, extracted["depth_static"]
                                ),
                            )
                            self._save_file(
                                output_path / depth_wrist_path,
                                lambda p: np.save(
                                    p, extracted["depth_wrist"]
                                ),
                            )
                            sample["depth_static"] = depth_static_path
                            sample["depth_wrist"] = depth_wrist_path
                            self._emit_aux_denoising_sidecars(
                                output_dir=output_path,
                                trajectory_id=episode_id,
                                base_index=t,
                                depth_static=extracted["depth_static"],
                                grounding_mask=extracted.get("grounding_mask"),
                                affordance_heatmap=extracted.get("affordance_heatmap"),
                                grounding_level="object",
                            )
                            sample["depth_target"] = (
                                f"depths/static/{episode_id}/{t}.npy"
                            )
                            if extracted.get("grounding_mask") is not None:
                                sample["grounding_mask"] = (
                                    f"grounding_masks/static/{episode_id}/{t}.npy"
                                )
                                sample["grounding_level"] = "object"
                            if extracted.get("affordance_heatmap") is not None:
                                sample["affordance_heatmap"] = (
                                    f"affordance_heatmaps/static/{episode_id}/{t}.npy"
                                )

                            # Wrist camera extrinsic
                            sample["wrist_cam_extrinsic"] = {
                                "rotation": extracted[
                                    "wrist_mat"
                                ].flatten().tolist(),
                                "translation": extracted[
                                    "wrist_pos"
                                ].tolist(),
                            }

                            # Static camera extrinsic (per-sample — varies by
                            # task scene XML, so cannot be a global statistic)
                            sample["static_cam_extrinsic"] = {
                                "rotation": extracted[
                                    "static_cam_mat"
                                ].flatten().tolist(),
                                "translation": extracted[
                                    "static_cam_pos"
                                ].tolist(),
                            }

                            # Pose 6D
                            sample["pose_6d"] = {
                                "rotation": mat_to_6d(extracted["obj_mat"]),
                                "translation": extracted["obj_pos"].tolist(),
                                "object_id": self._target_body_name,
                            }

                            # Point cloud (world frame)
                            pts = extracted.get("point_cloud")
                            if pts is not None:
                                pc_path = f"point_clouds/{sample_id}.npy"
                                self._save_file(
                                    output_path / pc_path,
                                    lambda p: np.save(p, pts),
                                )
                                sample["point_cloud"] = pc_path

                            # Image target (seg crop)
                            target_img = extracted.get("image_target")
                            if target_img is not None:
                                target_path = (
                                    f"images/target/{sample_id}.jpg"
                                )
                                self._save_file(
                                    output_path / target_path,
                                    lambda p: target_img.save(
                                        p, quality=95
                                    ),
                                )
                                sample["image_target"] = target_path

                            new_samples.append(sample)

                        logger.info(f"  {demo_key}: {T} steps processed")
            finally:
                env.close()

        # Append new samples to data.jsonl
        jsonl_path = output_path / "data.jsonl"
        with open(jsonl_path, "a") as f:
            for s in new_samples:
                f.write(json.dumps(s) + "\n")
        logger.info(
            f"Appended {len(new_samples)} samples to {jsonl_path} "
            f"(total: {len(existing_ids) + len(new_samples)})"
        )

        # Recompute statistics over ALL samples (existing + new)
        # Skip if no samples were produced at all (e.g. all tasks skipped)
        if camera_intrinsics is not None:
            all_samples = self._load_all_samples(output_path)
            if all_samples:
                self.write_statistics(
                    all_samples, output_path, camera_intrinsics
                )
            else:
                logger.warning(
                    "No samples available; skipping statistics.yaml write."
                )

        # Resolver summary
        stats = self._resolve_stats
        logger.info(
            f"[target-resolver] llm-hit: {stats['llm_hit']}  "
            f"keyword-hit: {stats['keyword_hit']}  "
            f"skipped: {stats['skipped']}"
        )
        if self._skipped_samples:
            logger.info(f"[skipped samples] ({len(self._skipped_samples)})")
            for fname, reason in self._skipped_samples:
                logger.info(f"  - {fname}  reason: {reason}")

        logger.info(f"Done. Output: {output_path}")

    @staticmethod
    def _load_existing_ids(output_path: Path) -> set[str]:
        """Load sample IDs from existing data.jsonl for duplicate detection."""
        jsonl_path = output_path / "data.jsonl"
        if not jsonl_path.exists():
            return set()
        ids = set()
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["id"])
        return ids

    @staticmethod
    def _load_all_samples(output_path: Path) -> list[dict]:
        """Load all samples from data.jsonl for statistics recomputation."""
        jsonl_path = output_path / "data.jsonl"
        samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    @staticmethod
    def _save_file(path: Path, write_fn):
        """Write a file, raising FileExistsError if it already exists."""
        if path.exists():
            raise FileExistsError(
                f"Output file already exists: {path}. "
                f"This indicates a duplicate sample ID collision."
            )
        write_fn(path)

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

    def _create_env(self, hdf5_file: Path):
        """Create LIBERO OffScreenRenderEnv from HDF5 env_args.

        The BDDL path stored in HDF5 metadata uses object-specific names
        (e.g. "pick_the_akita_black_bowl_...") that don't match the actual
        BDDL files on disk ("pick_up_the_black_bowl_..."). We use the
        benchmark API to resolve the correct BDDL file path by matching
        the HDF5 filename to the benchmark task name.
        """
        from libero.libero.envs import OffScreenRenderEnv

        bddl_file_name = self._resolve_bddl_path(hdf5_file)

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_name,
            robots=["Panda"],
            controller="OSC_POSE",
            camera_names=[STATIC_CAM, WRIST_CAM],
            camera_heights=RENDER_H,
            camera_widths=RENDER_W,
            camera_depths=True,
            camera_segmentations="instance",
        )
        env.reset()
        return env

    EXCLUDED_BODY_PREFIXES = ("robot", "gripper", "mount")

    def _build_candidate_list(self, env) -> list[dict]:
        """Collect (_main) object bodies and convert to LLM-friendly form.

        Returns a list of {"id": <body_name>, "name": <space-separated>}
        suitable for passing to LLMTargetResolver.resolve().
        """
        candidates = []
        for body in env.sim.model.body_names:
            if not body.endswith("_main"):
                continue
            if any(body.startswith(p) for p in self.EXCLUDED_BODY_PREFIXES):
                continue
            stem = body[: -len("_main")]
            human = stem.replace("_", " ")
            candidates.append({"id": body, "name": human})
        return candidates

    def _resolve_target_body(self, env, instruction: str) -> str:
        """Find the MuJoCo body name for the target object of this task.

        Strategy chain:
        1. If a target_resolver (LLM) is configured, call it first. On hit
           (returned id is in the candidate set) use it.
        2. Otherwise, or if the LLM returned None, fall back to the explicit
           target_object_keyword substring match (only if the keyword is set).
        3. If neither produces a result, raise TargetResolveError.
        """
        candidates = self._build_candidate_list(env)

        # Strategy 1: LLM
        if self._target_resolver is not None:
            llm_id = self._target_resolver.resolve(instruction, candidates)
            valid_ids = {c["id"] for c in candidates}
            if llm_id is not None and llm_id in valid_ids:
                self._resolve_stats["llm_hit"] += 1
                logger.info(
                    f"  Target body resolved: '{llm_id}' [llm-hit]"
                )
                return llm_id
            if llm_id is not None:
                logger.warning(
                    f"  LLM resolver returned id '{llm_id}' not in candidate "
                    f"set for instruction '{instruction}'; falling through"
                )
            else:
                logger.warning(
                    f"  LLM resolver returned no match for instruction "
                    f"'{instruction}'; falling through to keyword strategy"
                )

        # Strategy 2: explicit keyword substring match
        if self.target_object_keyword:
            kw_matches = [
                c["id"] for c in candidates
                if self.target_object_keyword in c["id"]
            ]
            if kw_matches:
                chosen = kw_matches[0]
                self._resolve_stats["keyword_hit"] += 1
                logger.info(
                    f"  Target body resolved: '{chosen}' [keyword-hit] "
                    f"(keyword='{self.target_object_keyword}')"
                )
                return chosen

        raise TargetResolveError(
            f"Cannot resolve target object for instruction '{instruction}'. "
            f"Candidates: {[c['id'] for c in candidates]}"
        )

    def _resolve_bddl_path(self, hdf5_file: Path) -> str:
        """Resolve the absolute BDDL file path for a given HDF5 demo file.

        Uses LIBERO benchmark API: HDF5 filename (minus _demo.hdf5) matches
        the benchmark task name, which gives the correct BDDL path.
        """
        from libero.libero import benchmark, get_libero_path

        # HDF5 filename like "pick_up_the_black_bowl_from_table_center_..._demo.hdf5"
        # Task name is the same without "_demo.hdf5"
        task_name = hdf5_file.stem  # removes .hdf5
        if task_name.endswith("_demo"):
            task_name = task_name[: -len("_demo")]

        # Look up suite from benchmark API
        suite_name = self.suite.replace("_test", "")  # handle test directories
        bench_dict = benchmark.get_benchmark_dict()
        if suite_name not in bench_dict:
            raise ValueError(
                f"Suite '{suite_name}' not found in LIBERO benchmark. "
                f"Available: {list(bench_dict.keys())}"
            )
        suite = bench_dict[suite_name]()

        bddl_files_root = get_libero_path("bddl_files")
        for i in range(suite.n_tasks):
            task = suite.get_task(i)
            if task.name == task_name:
                return os.path.join(
                    bddl_files_root, task.problem_folder, task.bddl_file
                )

        raise ValueError(
            f"Task '{task_name}' not found in suite '{suite_name}'. "
            f"Available tasks: {[suite.get_task(i).name for i in range(suite.n_tasks)]}"
        )

    def _extract_camera_intrinsics(self, env) -> dict:
        """Extract camera intrinsic parameters from MuJoCo model."""
        result = {}
        for cam_name in [STATIC_CAM, WRIST_CAM]:
            cam_id = env.sim.model.camera_name2id(cam_name)
            fovy = env.sim.model.cam_fovy[cam_id]
            result[cam_name] = get_camera_intrinsic_from_fovy(
                fovy, RENDER_W, RENDER_H
            )
        return result

    def _replay_and_extract(
        self,
        env,
        state: np.ndarray,
        camera_intrinsics: dict,
        future_tcp_positions: np.ndarray | None = None,
    ) -> dict:
        """Replay MuJoCo state and extract depth, seg, pose, camera, point cloud."""
        env.sim.set_state_from_flattened(state)
        env.sim.forward()

        # Object pose
        obj_body_id = env.sim.model.body_name2id(self._target_body_name)
        obj_pos = env.sim.data.body_xpos[obj_body_id].copy()
        obj_mat = env.sim.data.body_xmat[obj_body_id].reshape(3, 3).copy()

        # Wrist camera extrinsic
        wrist_cam_id = env.sim.model.camera_name2id(WRIST_CAM)
        wrist_pos = env.sim.data.cam_xpos[wrist_cam_id].copy()
        wrist_mat = env.sim.data.cam_xmat[wrist_cam_id].reshape(3, 3).copy()

        # Render RGB from MuJoCo for both cameras (256x256, pixel-aligned
        # with depth/seg — replaces HDF5 128x128 originals for consistency)
        rgb_static = self._render_rgb(env, STATIC_CAM)
        rgb_wrist = self._render_rgb(env, WRIST_CAM)

        # Render depth for both cameras.
        # MuJoCo sim.render(depth=True) returns the raw OpenGL depth buffer
        # (nonlinear, [0, 1]).  We must linearize to metric depth (meters).
        extent = env.sim.model.stat.extent
        znear = env.sim.model.vis.map.znear * extent
        zfar = env.sim.model.vis.map.zfar * extent

        depth_static = env.sim.render(
            camera_name=STATIC_CAM,
            width=RENDER_W,
            height=RENDER_H,
            depth=True,
        )
        if isinstance(depth_static, tuple):
            depth_static = depth_static[1]
        depth_static = depth_static[::-1].copy()
        depth_static = linearize_depth(depth_static, znear, zfar)

        depth_wrist = env.sim.render(
            camera_name=WRIST_CAM,
            width=RENDER_W,
            height=RENDER_H,
            depth=True,
        )
        if isinstance(depth_wrist, tuple):
            depth_wrist = depth_wrist[1]
        depth_wrist = depth_wrist[::-1].copy()
        depth_wrist = linearize_depth(depth_wrist, znear, zfar)

        # Render segmentation for static camera
        seg_static = env.sim.render(
            camera_name=STATIC_CAM,
            width=RENDER_W,
            height=RENDER_H,
            segmentation=True,
        )
        if isinstance(seg_static, tuple):
            seg_static = seg_static[-1]
        seg_static = seg_static[::-1].copy()

        # MuJoCo segmentation returns (type, geom_id) per pixel.
        # Convert geom-level IDs to a binary mask for the target body.
        target_geom_ids = self._get_target_geom_ids(env)
        seg_geom = seg_static[:, :, 1] if seg_static.ndim == 3 else seg_static
        # Binary mask: 1 where target object, 0 elsewhere
        target_mask = np.isin(seg_geom, target_geom_ids).astype(np.int32)
        grounding_mask = self._mask_to_token_grid(target_mask)

        # Static camera extrinsic for point cloud transform
        static_cam_id = env.sim.model.camera_name2id(STATIC_CAM)
        static_cam_pos = env.sim.data.cam_xpos[static_cam_id].copy()
        static_cam_mat = (
            env.sim.data.cam_xmat[static_cam_id].reshape(3, 3).copy()
        )

        # Point cloud (world frame) — target_id=1 matches the binary mask
        point_cloud = depth_to_world_points(
            depth_static,
            target_mask,
            camera_intrinsics[STATIC_CAM],
            static_cam_mat,
            static_cam_pos,
            target_id=1,
            num_points=NUM_POINTS,
        )
        affordance_heatmap = None
        if point_cloud is not None and future_tcp_positions is not None:
            affordance_heatmap = self._target_point_affordance_heatmap(
                point_cloud=point_cloud,
                tcp_positions=future_tcp_positions,
                intrinsic=camera_intrinsics[STATIC_CAM],
                cam_R=static_cam_mat,
                cam_t=static_cam_pos,
            )

        # Image target (seg crop from MuJoCo-rendered RGB, pixel-aligned with seg)
        image_target = crop_target_from_seg(
            rgb_static, target_mask, target_id=1
        )

        return {
            "rgb_static": rgb_static,
            "rgb_wrist": rgb_wrist,
            "obj_pos": obj_pos,
            "obj_mat": obj_mat,
            "wrist_pos": wrist_pos,
            "wrist_mat": wrist_mat,
            "static_cam_pos": static_cam_pos,
            "static_cam_mat": static_cam_mat,
            "depth_static": depth_static.astype(np.float32),
            "depth_wrist": depth_wrist.astype(np.float32),
            "point_cloud": point_cloud,
            "image_target": image_target,
            "grounding_mask": grounding_mask,
            "affordance_heatmap": affordance_heatmap,
        }

    @staticmethod
    def _mask_to_token_grid(mask: np.ndarray, target_size: int = 20) -> np.ndarray:
        mask_arr = (np.asarray(mask) > 0).astype(np.float32)
        img = Image.fromarray((mask_arr * 255.0).astype(np.uint8), mode="L")
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

    @staticmethod
    def _render_rgb(env, camera_name: str) -> np.ndarray:
        """Render RGB image from MuJoCo, flipped to image convention."""
        rgb = env.sim.render(
            camera_name=camera_name,
            width=RENDER_W,
            height=RENDER_H,
            depth=False,
        )
        if isinstance(rgb, tuple):
            rgb = rgb[0]
        return rgb[::-1, ::-1].copy()

    def _get_target_geom_ids(self, env) -> list[int]:
        """Get all MuJoCo geom IDs belonging to the target body.

        MuJoCo segmentation renders geom IDs, not body IDs. A single body
        (e.g. akita_black_bowl_1_main) can have many geoms (g0..g40).
        """
        body_id = env.sim.model.body_name2id(self._target_body_name)
        geom_bodyids = env.sim.model.geom_bodyid
        return [i for i in range(len(geom_bodyids)) if geom_bodyids[i] == body_id]

    def write_statistics(
        self,
        samples: list[dict],
        output_path: Path,
        camera_intrinsics: dict,
    ):
        """Compute normalization statistics and write statistics.yaml.

        DEPRECATED post 2026-05-13 (spec §2.2): the canonical-state pipeline
        (`LiberoAdapter`, `statistics.yaml`) was removed when uamvla migrated
        to UamVLAOFT (LeRobot computes stats automatically into
        `meta/stats_gr00t.json`). This method raises NotImplementedError —
        rewrite to emit LeRobot v2 stats if you need it.
        """
        raise NotImplementedError(
            "LiberoPreprocessor.compute_statistics() depended on "
            "starVLA.model.modules.uamvla.data.embodiment_adapter.LiberoAdapter "
            "and statistics.yaml schema, both removed in 2026-05-13 cleanup "
            "(spec §2.2). LeRobot now auto-computes stats into "
            "meta/stats_gr00t.json on first dataset load. "
            "Rewrite this method against LeRobot's pipeline if you need it."
        )

        # Original implementation (preserved as reference for any rewrite):
        import yaml

        from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter

        if len(samples) < 2:
            logger.warning(
                f"Computing statistics over only {len(samples)} sample(s); "
                f"q01/q99/std will be degenerate. Consider preprocessing more episodes."
            )

        actions_7d = np.array(
            [s["action"][:FRANKA_ACTION_DIM] for s in samples]
        )

        # Canonical-space state stats: run adapter per sample, accumulate per field.
        adapter = LiberoAdapter()
        field_buffers: dict[str, list[np.ndarray]] = {
            "arm_0.ee_pose": [],
            "arm_0.joint_pos": [],
            "gripper_0": [],
        }
        for s in samples:
            canonical = adapter.to_canonical(s)
            field_buffers["arm_0.ee_pose"].append(
                canonical["arm_0"]["ee_pose"].numpy()
            )
            field_buffers["arm_0.joint_pos"].append(
                canonical["arm_0"]["joint_pos"].numpy()
            )
            field_buffers["gripper_0"].append(canonical["gripper_0"].numpy())

        franka_state_stats: dict[str, dict] = {}
        for field_path, vals in field_buffers.items():
            arr = np.stack(vals).astype(np.float64)  # (N, D)
            franka_state_stats[field_path] = {
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
                "franka_libero": {
                    "action_dim": FRANKA_ACTION_DIM,
                    "action_min_bound": actions_7d.min(axis=0).tolist(),
                    "action_max_bound": actions_7d.max(axis=0).tolist(),
                },
            },
            "state_stats": {"franka_libero": franka_state_stats},
            "cameras": {
                "static": {"intrinsic": camera_intrinsics[STATIC_CAM]},
                "wrist":  {"intrinsic": camera_intrinsics[WRIST_CAM]},
            },
            "point_cloud": {
                "num_points": NUM_POINTS,
                "frame": "world",
            },
        }

        with open(output_path / "statistics.yaml", "w") as f:
            yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Wrote statistics.yaml ({len(samples)} samples)")
