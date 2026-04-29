"""PyBullet-backed adapter around calvin_env's PlayTableSimEnv.

CalvinWorker only touches five methods on the env; this module wraps the
real simulator to match that surface (see `CalvinEnvAdapter` below).

Runtime requirements (server only — macOS cannot satisfy these):
  - The `calvin_env` conda environment must be active.
  - calvin_env and pybullet must be importable.
  - For headless rendering on a GPU box WITHOUT an X server, set
    `export PYOPENGL_PLATFORM=egl` before launching python. This is done
    by `scripts/preprocess_calvin.sh`. If PYOPENGL_PLATFORM is unset
    PyBullet will attempt to open a GLX window and silently fall back to
    CPU-only TinyRenderer (≈20× slower).
  - calvin_env ships a tactile_sensor.py that imports a GL extension
    which segfaults on headless H200s. The server-side `calvin_env`
    setup replaces that module with a stub (see Task 9a). Keep the stub
    in place whenever preprocessing runs on the server.

The `make_calvin_env_adapter` factory imports calvin_env lazily so this
module stays importable on macOS (the development platform).
"""
from __future__ import annotations

import logging

import numpy as np

from starVLA.utils.geometry import (
    get_camera_intrinsic_from_fovy,
    linearize_depth,
)

logger = logging.getLogger(__name__)


def make_calvin_env_adapter(dataset_path: str) -> "CalvinEnvAdapter":
    """Construct a CalvinEnvAdapter from a CALVIN dataset path.

    `calvin_env.envs.play_table_env.get_env` reads the dataset's hydra
    `.hydra/` directory to pick the right scene config (A/B/C/D) — so
    the caller does NOT pass a scene letter. The default_scene CLI
    argument still exists but only drives SceneResolver logging now.

    The import is inside the function so this module loads cleanly on
    macOS where calvin_env is absent.
    """
    from calvin_env.envs.play_table_env import get_env  # noqa: WPS433

    env = get_env(str(dataset_path), show_gui=False)
    try:
        return CalvinEnvAdapter(env)
    except Exception:
        # Adapter construction failed mid-setup — close the PyBullet
        # client so we don't leak a physics connection. Swallow any
        # secondary close() error so the original exception propagates.
        try:
            env.close()
        except Exception:
            pass
        raise


class CalvinEnvAdapter:
    """Wraps PlayTableSimEnv behind the 5-method surface CalvinWorker uses.

    Duck-typed — no ABC. The fake env in tests implements the same
    methods, so CalvinWorker can swap between them transparently.
    """

    def __init__(self, env):
        self._env = env
        self._closed = False

        # PyBullet physics client id (needed for direct p.getCameraImage /
        # p.getLinkState calls that the env doesn't expose).
        self._cid = env.cid

        # Cameras: calvin_env populates env.cameras = [StaticCamera,
        # GripperCamera] (order defined by hydra conf).
        self._static_cam = env.cameras[0]
        self._wrist_cam = env.cameras[1]

        # Build the object_id → (body_uid, link_idx) resolver once.
        # - Movable objects: env.scene.movable_objects — each has `.name`
        #   (matches our CALVIN_TASK_TO_OBJECT values like "block_red")
        #   and `.uid` (PyBullet body uid). link_idx = -1.
        # - Fixed-scene links: env.scene.fixed_objects[0] is the table
        #   (name "base" in URDF). Its links are the composite parts —
        #   "slide", "drawer", "switch", "button", etc. We enumerate
        #   them via p.getJointInfo.
        import pybullet as p  # lazy — only available in adapter env

        self._p = p

        self._resolver: dict[str, tuple[int, int]] = {}
        for mov in env.scene.movable_objects:
            self._resolver[mov.name] = (int(mov.uid), -1)

        fixed = env.scene.fixed_objects[0]  # table
        self._table_uid = int(fixed.uid)
        n_joints = p.getNumJoints(self._table_uid, physicsClientId=self._cid)
        for j in range(n_joints):
            info = p.getJointInfo(self._table_uid, j, physicsClientId=self._cid)
            link_name = info[12].decode()  # linkName field
            # Key format: "{body_name}__{link_name}", e.g. "table__slide_link".
            # Must match values in CALVIN_TASK_TO_OBJECT.
            key = f"{fixed.name}__{link_name}"
            self._resolver[key] = (self._table_uid, j)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, robot_obs, scene_obs) -> None:
        """Restore the env to the state stored in (robot_obs, scene_obs)."""
        self._env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    def render_cameras(self, width: int, height: int) -> dict:
        """Render both cameras at (width, height). Returns rgb/depth/seg/
        intrinsics/extrinsics for static + wrist, keyed as CalvinWorker
        expects.

        We call p.getCameraImage directly (rather than cam.render()) for
        two reasons:
          1. We need the segmentation mask, which cam.render() discards.
          2. We want to honor caller-requested (width, height) instead
             of the cam's baked-in defaults.
        """
        rgb_s, depth_s, seg_s, K_s, R_s, t_s = self._render_static(width, height)
        rgb_w, depth_w, seg_w, K_w, R_w, t_w = self._render_wrist(width, height)

        return {
            "rgb_static":        rgb_s,
            "rgb_wrist":         rgb_w,
            "depth_static":      depth_s,
            "depth_wrist":       depth_w,
            "seg_static":        seg_s,
            "seg_wrist":         seg_w,
            "static_intrinsic":  K_s,
            "wrist_intrinsic":   K_w,
            "static_cam_R":      R_s,
            "static_cam_t":      t_s,
            "wrist_cam_R":       R_w,
            "wrist_cam_t":       t_w,
        }

    def get_object_pose(self, object_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (position (3,), rotation_matrix (3, 3)) in world frame.

        Position is the AABB center of the target's mesh (link_idx ==
        -1 for base, else the specified link). For CALVIN composite
        parts the URDF link frame origin is at the joint pivot — often
        far from the visual center — so using AABB center keeps the
        pose aligned with the point cloud's bounding box. For movable
        blocks the base pose and AABB center nearly coincide, so the
        change is a no-op there.

        Rotation still comes from the link/base orientation so the OBB
        orientation in compute_obb_corners remains meaningful.
        """
        body_uid, link_idx = self._resolve(object_id)
        p = self._p
        if link_idx == -1:
            _pos, orn = p.getBasePositionAndOrientation(
                body_uid, physicsClientId=self._cid
            )
        else:
            ls = p.getLinkState(
                body_uid, link_idx,
                computeForwardKinematics=True,
                physicsClientId=self._cid,
            )
            # ls[1] = linkWorldOrientation
            orn = ls[1]

        aabb_min, aabb_max = p.getAABB(
            body_uid, link_idx, physicsClientId=self._cid,
        )
        center = (np.asarray(aabb_min) + np.asarray(aabb_max)) * 0.5
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return center.astype(np.float32), R.astype(np.float32)

    def get_target_seg_id(self, object_id: str) -> int:
        """Return the integer value that appears in the seg mask for
        this object under PyBullet's ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        encoding:

            seg_value = body_uid + ((link_idx + 1) << 24)

        For a movable object (link_idx == -1), this collapses to body_uid.
        """
        body_uid, link_idx = self._resolve(object_id)
        return int(body_uid + ((link_idx + 1) << 24))

    def close(self) -> None:
        """Close the underlying env. Idempotent — guards against
        PlayTableSimEnv.__del__ double-close (upstream bug raises
        RuntimeError on already-disconnected client).
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._env.close()
        except RuntimeError as e:
            logger.debug("Ignoring RuntimeError on env.close(): %s", e)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve(self, object_id: str) -> tuple[int, int]:
        try:
            return self._resolver[object_id]
        except KeyError:
            raise KeyError(
                f"object_id {object_id!r} not found in scene. "
                f"Known ids: {sorted(self._resolver.keys())}"
            )

    def _render_static(self, width: int, height: int):
        cam = self._static_cam
        p = self._p

        view_matrix = cam.viewMatrix
        # Always recompute the projection matrix from (fov, aspect, near,
        # far) instead of reading `cam.projectionMatrix`. Upstream only
        # caches that attribute on StaticCamera; GripperCamera does not
        # set it at all. Recomputing is cheap and keeps both branches on
        # a single code path (see _render_wrist).
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=cam.fov,
            aspect=width / height,
            nearVal=cam.nearval,
            farVal=cam.farval,
        )
        return self._capture(
            cam, width, height, view_matrix, proj_matrix,
        )

    def _render_wrist(self, width: int, height: int):
        cam = self._wrist_cam
        p = self._p

        # GripperCamera recomputes its viewMatrix every frame from the
        # robot link pose. Mirror the logic in calvin_env's
        # GripperCamera.render().
        link_state = p.getLinkState(
            cam.robot_uid,
            cam.gripper_cam_link,
            computeForwardKinematics=True,
            physicsClientId=self._cid,
        )
        # link_state[0]=pos, link_state[1]=orn (world frame)
        cam_pos, cam_orn = link_state[0], link_state[1]
        cam_rot = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        target, up = self._compute_wrist_view_axes(cam_rot, np.asarray(cam_pos))
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
        )

        # See _render_static for why we always recompute rather than
        # read cam.projectionMatrix (GripperCamera doesn't have it).
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=cam.fov,
            aspect=width / height,
            nearVal=cam.nearval,
            farVal=cam.farval,
        )
        return self._capture(
            cam, width, height, view_matrix, proj_matrix,
        )

    def _capture(
        self,
        cam,
        width: int,
        height: int,
        view_matrix,
        proj_matrix,
    ):
        """Common p.getCameraImage path. Returns:
        (rgb uint8 HxWx3, depth float32 HxW in meters, seg int32 HxW,
         intrinsic dict, cam_R float32 3x3, cam_t float32 3).
        """
        p = self._p
        w, h, rgb_px, depth_px, seg_px = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=self._cid,
        )

        # rgb: PyBullet gives RGBA uint8
        rgb = np.asarray(rgb_px, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        rgb = np.ascontiguousarray(rgb)

        # depth: GL depth buffer in [0, 1] → linearize to meters
        depth_buf = np.asarray(depth_px, dtype=np.float32).reshape(h, w)
        depth = linearize_depth(
            depth_buf, znear=cam.nearval, zfar=cam.farval
        )

        # seg: int32 HxW — composite (body_uid | (link_idx+1)<<24)
        seg = np.asarray(seg_px, dtype=np.int32).reshape(h, w)

        intrinsic = get_camera_intrinsic_from_fovy(
            fovy=cam.fov, width=w, height=h,
        )

        cam_R, cam_t = self._extrinsics_from_view_matrix(view_matrix)
        return rgb, depth, seg, intrinsic, cam_R, cam_t

    @staticmethod
    def _compute_wrist_view_axes(cam_rot: np.ndarray, cam_pos: np.ndarray):
        """Return (target, up) for PyBullet's computeViewMatrix, matching
        upstream calvin_env GripperCamera: forward = +Y of link frame,
        up = -Z of link frame.

        cam_rot is the 3x3 world-frame rotation of the gripper_cam link
        (columns = link basis vectors in world coords). cam_pos is the
        link position in world coords. target = cam_pos + forward.
        """
        forward = cam_rot[:, 1]
        up = -cam_rot[:, 2]
        target = np.asarray(cam_pos, dtype=np.float64) + forward
        return target, up

    @staticmethod
    def _extrinsics_from_view_matrix(view_matrix):
        """PyBullet viewMatrix is a column-major, world→camera transform
        in OpenGL convention (camera looks down -Z, +Y up).

        Returns (R_wc, t_wc) as float32, where R_wc is the rotation
        mapping camera-frame vectors to world-frame (columns are the
        camera basis vectors in world coords) and t_wc is the camera
        position in world coords. This matches the convention
        depth_to_world_points expects.
        """
        V = np.array(view_matrix, dtype=np.float64).reshape(4, 4).T
        # V: world→camera. Invert (rotation transpose, translation negated).
        R_wc = V[:3, :3].T
        t_wc = -R_wc @ V[:3, 3]
        return R_wc.astype(np.float32), t_wc.astype(np.float32)
