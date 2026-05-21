"""UamVLAOFT framework: QwenOFT + pose/future/recon perception aux heads.

Inherits L1 action regression + Qwen3-VL backbone from ``Qwenvl_OFT``;
adds three aux heads consuming ``hidden_states[-1]`` alongside the L1
action loss.  No state encoder — robot state is inferred from images only.
Sidecar IO (point_cloud / image_target) happens in
:meth:`_unpack_lerobot_sample`.

PR 4 of the UamVLA-on-QwenOFT migration keeps all aux heads OFF — this
is the "baseline B" run per spec §5.7. PR 5/6/7 will activate the
pose/future/recon heads in turn.

Spec: ``docs/superpowers/specs/2026-05-13-uamvla-on-qwenoft-design.md`` §5
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT
from starVLA.model.modules.uamvla.collator_helpers import (
    stack_optional_string_fields,
    stack_optional_tensor_fields,
    stack_pose_gt,
    stack_static_cam_extrinsic,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY

# Aux heads whose mask defaults to all-True when their ``<head>_mask`` key
# is missing from batch_dict.  Currently empty: all three perception heads
# are non-universal (image_target / image_future / pose_gt / point_cloud
# may be missing on some samples).
_UNIVERSAL_HEADS: tuple[str, ...] = ()

_DEFAULT_LEROBOT_VIDEO_PATH_PATTERN = (
    "videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"
)
_DEFAULT_LEROBOT_CHUNKS_SIZE = 1_000_000_000


def _move_to_device(value, device):
    """Recursively move tensors inside dicts/lists onto ``device``."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    return value


@FRAMEWORK_REGISTRY.register("UamVLAOFT")
class UamVLAOFT(Qwenvl_OFT):
    """UamVLA framework rebuilt on top of ``Qwenvl_OFT``.

    PR 4 contract:
      * No aux heads are constructed (``self.aux_heads`` is empty).
      * ``forward`` produces only the L1 action loss.
      * ``_unpack_lerobot_sample`` already loads ``point_cloud`` /
        ``image_target`` sidecar files and slices ``pose_gt`` /
        ``static_cam_extrinsic`` from ``state`` so PR 5-7 only need to
        wire heads, not data.
    """

    def __init__(self, config) -> None:
        super().__init__(config)  # builds _QWen3_VL_Interface + L1RegressionActionHead

        # ── 🔍 emoji sanity check ────────────────────────────────────
        # Action token must tokenize to exactly 1 token per copy.
        sanity_ids = self.qwen_vl_interface.processor.tokenizer(
            "🔍" * self.chunk_len, add_special_tokens=False,
        )["input_ids"]
        if len(sanity_ids) != self.chunk_len:
            raise RuntimeError(
                f"🔍 must tokenize to 1 token each, got {len(sanity_ids)} for "
                f"chunk_len={self.chunk_len}. Check Qwen3-VL tokenizer version "
                f"or replace 🔍 with a registered special token."
            )

        # ── Sidecar root for point_cloud / image_target lookup ──────
        # Resolve via the mixture registry so the framework tracks
        # whichever dataset directory was registered (e.g. *_SMOKE vs
        # full ABCD).
        from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
        mixture = DATASET_NAMED_MIXTURES[self.config.datasets.vla_data.data_mix]
        dataset_name = mixture[0][0] # name: lerobot_calvin_abcd
        self.sidecar_root = Path(self.config.datasets.vla_data.data_root_dir) / dataset_name
        self._init_lerobot_video_path_config()

        # Per-frame image_target cache (one PNG per trajectory/base index).
        # OrderedDict + LRU eviction so worker RAM stays bounded on full-scale
        # datasets (many frame crops per trajectory; cap bounds worker memory).
        # 256 entries ≈ 120 MB per worker — comfortable headroom even for 16 workers.
        from collections import OrderedDict
        self._image_target_cache: "OrderedDict[tuple[int, int], torch.Tensor]" = (
            OrderedDict()
        )
        self._image_target_cache_maxsize = int(
            self.config.datasets.vla_data.get("image_target_cache_maxsize", 256)
        )

        # Aux state slice indices — mirrors DataConfig.aux_state_slice so
        # the framework is decoupled from DataConfig's import path.
        self.aux_state_slice = self.config.datasets.vla_data.get(
            "aux_state_slice",
            {
                "target_pose_rot6d": [15, 21],
                "target_pose_trans": [21, 24],
                "static_cam_rot6d":  [24, 30],
                "static_cam_trans":  [30, 33],
            },
        )

        # Aux heads — empty in PR 4. PR 5/6/7 populate via
        # `_maybe_build_aux_heads`.
        self.aux_heads = nn.ModuleDict()
        self._maybe_build_aux_heads()
        self._maybe_build_aux_loss_control()

    # ──────────────────────────────────────────────────────────────────
    #  Aux heads (no-op in PR 4)
    # ──────────────────────────────────────────────────────────────────
    def _maybe_build_aux_heads(self) -> None:
        """Construct enabled aux heads on ``self.aux_heads``.

        PR 5 wires the PoseHead; PR 6/7 will add FutureHead / ReconHead.
        Heads are gated by ``config.framework.aux_heads.<name>.enabled``;
        when ``false`` (the baseline B path) this is a no-op.
        """
        from starVLA.model.modules.uamvla.aux_heads.pose_head import PoseHead

        cfg_heads = self.config.framework.aux_heads
        hidden_size = self.qwen_vl_interface.model.config.hidden_size

        if cfg_heads.get("pose", {}).get("enabled", False):
            pose_kwargs = {
                k: v for k, v in cfg_heads.pose.items()
                if k not in ("enabled", "lr")
            }
            # camera_params_path: framework auto-derives from sidecar_root.
            if "stats_path" not in pose_kwargs:
                cam_params_path = self.sidecar_root / "camera_params.json"
                if cam_params_path.exists():
                    pose_kwargs["stats_path"] = str(cam_params_path)
            self.aux_heads["pose"] = PoseHead(
                hidden_size=hidden_size, **pose_kwargs,
            )

        if cfg_heads.get("future", {}).get("enabled", False):
            from starVLA.model.modules.uamvla.aux_heads.future_head import FutureHead
            from starVLA.model.modules.uamvla.components.pixel_decoder.vae import (
                VAEPixelDecoder,
            )

            # VAE is shared between future and recon heads — construct once.
            if not hasattr(self, "vae") or self.vae is None:
                self.vae = VAEPixelDecoder(self.config.framework.vae.path)

            vision_extra = {
                "image_mean": [0.5, 0.5, 0.5],
                "image_std":  [0.5, 0.5, 0.5],
                "image_token_id": getattr(
                    self.qwen_vl_interface, "image_token_id",
                    self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids(
                        "<|image_pad|>"
                    ),
                ),
                "patches_per_view": 400,
                "n_patches": 400,
                "target_resize": 320,
            }
            future_cfg = {
                k: v for k, v in cfg_heads.future.items()
                if k not in ("enabled", "lr")
            }
            self.aux_heads["future"] = FutureHead(
                hidden_size=hidden_size, vae=self.vae,
                **{**vision_extra, **future_cfg},
            )

            # Sanity check decord availability ONCE here, not per-sample.
            # _load_image_future falls back to None when decord is missing,
            # which silently degrades FutureHead to its dummy_loss path on
            # every step — invisible in training logs. Logging once at init
            # lets ops spot the misconfig before launching a multi-hour run.
            try:
                import decord  # noqa: F401
            except ImportError:
                logger.warning(
                    "[FutureHead] decord is not installed in this environment. "
                    "_load_image_future will return None for every sample, "
                    "causing FutureHead to always run its dummy_loss path. "
                    "Install via `pip install decord==0.6.0` (matches requirements.txt) "
                    "to get a non-dummy future_loss."
                )

        if cfg_heads.get("recon", {}).get("enabled", False):
            from starVLA.model.modules.uamvla.aux_heads.recon_head import ReconHead
            from starVLA.model.modules.uamvla.components.pixel_decoder.vae import (
                VAEPixelDecoder,
            )

            # VAE is shared between future and recon heads — construct once.
            if not hasattr(self, "vae") or self.vae is None:
                self.vae = VAEPixelDecoder(self.config.framework.vae.path)

            vision_extra = {
                "image_mean": [0.5, 0.5, 0.5],
                "image_std":  [0.5, 0.5, 0.5],
                "image_token_id": getattr(
                    self.qwen_vl_interface, "image_token_id",
                    self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids(
                        "<|image_pad|>"
                    ),
                ),
                "patches_per_view": 400,
                "n_patches": 400,
                "target_resize": 320,
            }
            recon_cfg = {
                k: v for k, v in cfg_heads.recon.items()
                if k not in ("enabled", "lr")
            }
            self.aux_heads["recon"] = ReconHead(
                hidden_size=hidden_size, vae=self.vae,
                **{**vision_extra, **recon_cfg},
            )

        map_vision_extra = {
            "image_token_id": getattr(
                self.qwen_vl_interface,
                "image_token_id",
                self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids(
                    "<|image_pad|>"
                ),
            ),
            "patches_per_view": 400,
        }

        if cfg_heads.get("depth", {}).get("enabled", False):
            from starVLA.model.modules.uamvla.aux_heads.depth_head import DepthDenoisingHead

            depth_cfg = {
                k: v for k, v in cfg_heads.depth.items()
                if k not in ("enabled", "lr")
            }
            self.aux_heads["depth"] = DepthDenoisingHead(
                hidden_size=hidden_size,
                **{**map_vision_extra, **depth_cfg},
            )

        if cfg_heads.get("grounding", {}).get("enabled", False):
            from starVLA.model.modules.uamvla.aux_heads.grounding_head import (
                GroundingMaskDenoisingHead,
            )

            grounding_cfg = {
                k: v for k, v in cfg_heads.grounding.items()
                if k not in ("enabled", "lr")
            }
            self.aux_heads["grounding"] = GroundingMaskDenoisingHead(
                hidden_size=hidden_size,
                **{**map_vision_extra, **grounding_cfg},
            )

        if cfg_heads.get("affordance", {}).get("enabled", False):
            from starVLA.model.modules.uamvla.aux_heads.affordance_head import (
                AffordanceHeatmapDenoisingHead,
            )

            affordance_cfg = {
                k: v for k, v in cfg_heads.affordance.items()
                if k not in ("enabled", "lr")
            }
            self.aux_heads["affordance"] = AffordanceHeatmapDenoisingHead(
                hidden_size=hidden_size,
                **{**map_vision_extra, **affordance_cfg},
            )

        if cfg_heads.get("action_conditioned_future", {}).get("enabled", False):
            from starVLA.model.modules.uamvla.aux_heads.action_conditioned_future_head import (
                ActionConditionedFutureHead,
            )
            from starVLA.model.modules.uamvla.components.pixel_decoder.vae import (
                VAEPixelDecoder,
            )

            if not hasattr(self, "vae") or self.vae is None:
                self.vae = VAEPixelDecoder(self.config.framework.vae.path)

            action_future_cfg = {
                k: v for k, v in cfg_heads.action_conditioned_future.items()
                if k not in ("enabled", "lr")
            }
            action_model_cfg = self.config.framework.action_model
            action_dim = int(action_model_cfg.get("action_dim", 7))
            action_horizon = int(
                getattr(self, "action_horizon", action_model_cfg.get("action_horizon"))
            )
            vision_extra = {
                "image_mean": [0.5, 0.5, 0.5],
                "image_std":  [0.5, 0.5, 0.5],
                "image_token_id": map_vision_extra["image_token_id"],
                "patches_per_view": 400,
                "n_patches": 400,
                "target_resize": 320,
            }
            self.aux_heads["action_conditioned_future"] = ActionConditionedFutureHead(
                hidden_size=hidden_size,
                vae=self.vae,
                action_dim=action_dim,
                action_horizon=action_horizon,
                **{**vision_extra, **action_future_cfg},
            )

    def _maybe_build_aux_loss_control(self) -> None:
        """Construct the global aux-loss controller when configured."""
        cfg = getattr(self.config.framework, "aux_loss_control", None)
        if cfg is None:
            return

        from starVLA.model.modules.uamvla.aux_loss_control import AuxDenoisingSuite

        allowed = {
            "enabled",
            "aux_budget",
            "warmup_steps",
            "aux_ratio_cap",
            "action_loss_ema_beta",
            "eps",
        }
        suite_kwargs = {
            key: value
            for key, value in cfg.items()
            if key in allowed
        }
        self.aux_suite = AuxDenoisingSuite(
            heads=self.aux_heads,
            **suite_kwargs,
        )

    # ──────────────────────────────────────────────────────────────────
    #  Image resize — shared between training and inference
    # ──────────────────────────────────────────────────────────────────
    def _force_resize_640(self, image_list: list) -> list:
        """Resize each image to 640x640 via ``Image.BICUBIC``, idempotent.

        Qwen3VLProcessor produces ``image_grid_thw=(1, 40, 40)``
        (i.e. ppv=400) at this resolution, which matches the
        image_pad token count invariant in :meth:`forward`. Both training
        and inference must call this to keep tokens-per-view consistent.
        See spec §5.2.

        Idempotent: if the dataloader already produced 640×640 (e.g. via
        ``datasets.vla_data.image_resize: 640`` driving
        ``LeRobotSingleDataset._pack_sample``), this is a no-op for that
        image. Inference paths that feed PIL of arbitrary size still get
        resized. Together this collapses the training-time double resize
        from 200→224→640 to 200→640 (codex I-5).
        """
        out: list = []
        for img in image_list:
            if not isinstance(img, Image.Image):
                img = to_pil_preserve(img)
            if img.size != (640, 640):
                img = img.resize((640, 640), Image.BICUBIC)
            out.append(img)
        return out

    # ──────────────────────────────────────────────────────────────────
    #  Sample unpacking — LeRobot keys → framework keys + sidecar IO
    # ──────────────────────────────────────────────────────────────────
    def _unpack_lerobot_sample(self, sample: dict) -> dict:
        """Convert a LeRobot-style sample dict into framework-style.

        Input (LeRobot ``_pack_sample`` output):
          * ``image``: ``List[PIL]`` — observation frames (primary, wrist).
          * ``state``: ndarray/tensor shape ``(1, 33)`` or ``(33,)``.
          * ``action``: ndarray/tensor shape ``(H, 7)``.
          * ``lang``: ``str``.
          * ``__trajectory_id``: ``int``.
          * ``__base_index``: ``int``.

        Output (framework-side):
          * ``image``: ``List[PIL]``.
          * ``lang``: ``str``.
          * ``action``: ``Tensor (H, 7)``.
          * ``pose_gt``: dict ``{"rotation": (3,3), "translation": (3,)}``.
          * ``static_cam_extrinsic``: dict like ``pose_gt``.
          * ``point_cloud`` (optional): ``Tensor (1024, 3)`` — loaded from
            sidecar ``<sidecar_root>/point_clouds/<traj>/<base>.npy`` if
            present.
          * ``image_target`` (optional): ``Tensor (C, H, W)`` — loaded from
            sidecar ``<sidecar_root>/image_targets/<traj>/<base>.png`` if
            present (cached by trajectory_id and base_index).

        ``image_future`` is deferred to PR 6 (FutureHead is the only
        consumer; this field is intentionally absent here).
        """
        from starVLA.model.modules.uamvla.components.pose.pose_utils import (
            rotation_6d_to_matrix,
        )

        image = sample["image"]  # already List[PIL]
        lang = sample["lang"]

        action = sample["action"]
        if not torch.is_tensor(action):
            action = torch.as_tensor(np.asarray(action), dtype=torch.float32)
        else:
            action = action.to(dtype=torch.float32)

        # ── state slice → pose_gt + static_cam_extrinsic ────────────
        state = sample["state"]
        if not torch.is_tensor(state):
            state = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        else:
            state = state.to(dtype=torch.float32)
        # _pack_sample concatenates state.* into (T=1, 33); flatten the
        # leading dim before slicing.
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)

        s = self.aux_state_slice
        pose_rot6d = state[s["target_pose_rot6d"][0]:s["target_pose_rot6d"][1]]
        pose_trans = state[s["target_pose_trans"][0]:s["target_pose_trans"][1]]
        cam_rot6d = state[s["static_cam_rot6d"][0]:s["static_cam_rot6d"][1]]
        cam_trans = state[s["static_cam_trans"][0]:s["static_cam_trans"][1]]

        pose_gt = {
            "rotation": rotation_6d_to_matrix(pose_rot6d),  # (3, 3)
            "translation": pose_trans,                       # (3,)
        }
        static_cam_extrinsic = {
            "rotation": rotation_6d_to_matrix(cam_rot6d),
            "translation": cam_trans,
        }

        out = {
            "image": image,
            "lang": lang,
            "action": action,
            "pose_gt": pose_gt,
            "static_cam_extrinsic": static_cam_extrinsic,
        }

        # ── Sidecar IO ──────────────────────────────────────────────
        traj = int(sample["__trajectory_id"])
        base = int(sample["__base_index"])

        pc_path = self.sidecar_root / "point_clouds" / str(traj) / f"{base}.npy"
        if pc_path.exists():
            pc = np.load(pc_path)
            out["point_cloud"] = torch.as_tensor(pc, dtype=torch.float32)

        image_target_key = (traj, base)
        if image_target_key not in self._image_target_cache:
            it_path = self.sidecar_root / "image_targets" / str(traj) / f"{base}.png"
            if it_path.exists():
                arr = np.array(Image.open(it_path).convert("RGB"), dtype=np.uint8)
                self._image_target_cache[image_target_key] = (
                    torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                )
                # LRU eviction: drop least-recently-used entries past maxsize.
                while len(self._image_target_cache) > self._image_target_cache_maxsize:
                    self._image_target_cache.popitem(last=False)
        if image_target_key in self._image_target_cache:
            # Mark as recently used so subsequent evictions skip it.
            self._image_target_cache.move_to_end(image_target_key)
            out["image_target"] = self._image_target_cache[image_target_key]

        # image_future: load the terminal primary frame of this task episode.
        # Each CALVIN LeRobot episode corresponds to one language task window.
        future_tensor = self._load_image_future(traj, base)
        if future_tensor is not None:
            out["image_future"] = future_tensor

        depth = self._load_depth_target(traj, base)
        if depth is not None:
            out["depth_target"] = depth

        grounding = self._load_grounding_mask(traj, base)
        if grounding is not None:
            out["grounding_mask"] = grounding["mask"]
            out["grounding_level"] = grounding["level"]

        affordance = self._load_affordance_heatmap(traj, base)
        if affordance is not None:
            out["affordance_heatmap"] = affordance

        image_action_future = self._load_image_action_future(traj, base)
        if image_action_future is not None:
            out["image_action_future"] = image_action_future
        return out

    # ──────────────────────────────────────────────────────────────────
    #  image_future loader (decord random seek)
    # ──────────────────────────────────────────────────────────────────
    def _init_lerobot_video_path_config(self) -> None:
        """Cache LeRobot video path metadata for sidecar future-frame IO."""
        self._lerobot_video_path_pattern = _DEFAULT_LEROBOT_VIDEO_PATH_PATTERN
        self._lerobot_chunks_size = _DEFAULT_LEROBOT_CHUNKS_SIZE

        info_path = self.sidecar_root / "meta" / "info.json"
        if not info_path.exists():
            logger.warning(
                "LeRobot info metadata not found at %s; falling back to "
                "legacy chunk-000 video path lookup for image_future.",
                info_path,
            )
            return

        with open(info_path, "r") as f:
            info = json.load(f)

        chunks_size = int(info.get("chunks_size", _DEFAULT_LEROBOT_CHUNKS_SIZE))
        if chunks_size <= 0:
            raise ValueError(
                f"Invalid chunks_size in {info_path}: {chunks_size!r}"
            )

        self._lerobot_chunks_size = chunks_size
        self._lerobot_video_path_pattern = str(
            info.get("video_path", _DEFAULT_LEROBOT_VIDEO_PATH_PATTERN)
        )

    def _image_future_video_path(self, trajectory_id: int) -> Path:
        episode_chunk = int(trajectory_id) // self._lerobot_chunks_size
        video_filename = self._lerobot_video_path_pattern.format(
            episode_chunk=episode_chunk,
            episode_index=int(trajectory_id),
            video_key="video.primary_image",
        )
        return self.sidecar_root / video_filename

    @staticmethod
    def _image_future_frame_index(video_length: int) -> int | None:
        if video_length <= 0:
            return None
        return video_length - 1

    @staticmethod
    def _load_npy_sidecar(path: Path) -> torch.Tensor | None:
        if not path.exists():
            return None
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[-1] == 1 and arr.shape[0] != 1:
            arr = np.moveaxis(arr, -1, 0)
        return torch.as_tensor(arr, dtype=torch.float32)

    def _load_depth_target(self, trajectory_id: int, base_index: int) -> torch.Tensor | None:
        path = self.sidecar_root / "depths" / "static" / str(trajectory_id) / f"{base_index}.npy"
        return self._load_npy_sidecar(path)

    def _load_grounding_mask(self, trajectory_id: int, base_index: int) -> dict | None:
        mask_path = (
            self.sidecar_root
            / "grounding_masks"
            / "static"
            / str(trajectory_id)
            / f"{base_index}.npy"
        )
        mask = self._load_npy_sidecar(mask_path)
        if mask is None:
            return None
        level = "object"
        meta_path = mask_path.with_suffix(".json")
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                level = str(meta.get("grounding_level", meta.get("level", level)))
            except Exception as exc:
                logger.warning("Failed to load grounding metadata %s: %s", meta_path, exc)
        return {"mask": mask, "level": level}

    def _load_affordance_heatmap(self, trajectory_id: int, base_index: int) -> torch.Tensor | None:
        path = (
            self.sidecar_root
            / "affordance_heatmaps"
            / "static"
            / str(trajectory_id)
            / f"{base_index}.npy"
        )
        return self._load_npy_sidecar(path)

    def _image_action_future_frame_index(self, base_index: int, video_length: int) -> int | None:
        if video_length <= 0:
            return None
        action_model_cfg = self.config.framework.action_model
        offset = int(action_model_cfg.get("future_action_window_size", 0))
        frame_index = int(base_index) + offset
        if frame_index < 0 or frame_index >= video_length:
            return None
        return frame_index

    @staticmethod
    def _to_rgb_pil(image) -> Image.Image:
        """Convert PIL / numpy / tensor image values into an RGB PIL image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if torch.is_tensor(image):
            arr = image.detach().float().cpu().numpy()
        else:
            arr = np.asarray(image)

        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            finite = arr[np.isfinite(arr)]
            if finite.size and finite.min() < 0.0:
                arr = (arr + 1.0) / 2.0
            elif finite.size and finite.max() > 1.5:
                arr = arr / 255.0
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

        return Image.fromarray(arr).convert("RGB")

    @staticmethod
    def _to_action_numpy(action) -> np.ndarray:
        """Convert action arrays/tensors to ``float32`` numpy."""
        if torch.is_tensor(action):
            return action.detach().float().cpu().numpy()
        return np.asarray(action, dtype=np.float32)

    @staticmethod
    def _fit_for_viz(image: Image.Image, height: int = 160) -> Image.Image:
        scale = height / max(1, image.height)
        width = max(1, int(round(image.width * scale)))
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        return image.resize((width, height), resample)

    @staticmethod
    def _draw_action_comparison(pred_action: np.ndarray, gt_action: np.ndarray | None) -> Image.Image:
        """Draw predicted-vs-ground-truth normalized action curves."""
        pred = np.asarray(pred_action, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred[:, None]

        gt = None
        if gt_action is not None:
            gt = np.asarray(gt_action, dtype=np.float32)
            if gt.ndim == 1:
                gt = gt[:, None]

        dims = pred.shape[-1]
        if gt is not None:
            dims = min(dims, gt.shape[-1])
        dims = min(dims, 7)

        width = 560
        row_h = 34
        top = 30
        left = 48
        right = width - 16
        height = top + max(1, dims) * row_h + 14
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), "actions: pred red / gt blue", fill=(32, 32, 32))

        for dim in range(dims):
            y_mid = top + dim * row_h + row_h // 2
            y0 = y_mid - 11
            y1 = y_mid + 11
            draw.text((10, y_mid - 7), f"a{dim}", fill=(55, 65, 81))
            draw.line((left, y_mid, right, y_mid), fill=(229, 231, 235), width=1)
            draw.rectangle((left, y0, right, y1), outline=(229, 231, 235))

            series = [pred[:, dim]]
            if gt is not None:
                series.append(gt[:, dim])
            finite = np.concatenate([
                s[np.isfinite(s)] for s in series if np.isfinite(s).any()
            ]) if any(np.isfinite(s).any() for s in series) else np.array([0.0])
            lo = float(finite.min())
            hi = float(finite.max())
            if abs(hi - lo) < 1e-6:
                lo -= 0.5
                hi += 0.5
            pad = 0.05 * (hi - lo)
            lo -= pad
            hi += pad

            def _points(values: np.ndarray) -> list[tuple[int, int]]:
                points = []
                denom = max(1, len(values) - 1)
                for idx, value in enumerate(values):
                    if not np.isfinite(value):
                        continue
                    x = int(round(left + idx * (right - left) / denom))
                    y = int(round(y1 - (float(value) - lo) * (y1 - y0) / (hi - lo)))
                    points.append((x, y))
                return points

            if gt is not None:
                pts = _points(gt[:, dim])
                if len(pts) >= 2:
                    draw.line(pts, fill=(37, 99, 235), width=2)
                elif pts:
                    x, y = pts[0]
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(37, 99, 235))

            pts = _points(pred[:, dim])
            if len(pts) >= 2:
                draw.line(pts, fill=(220, 38, 38), width=2)
            elif pts:
                x, y = pts[0]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(220, 38, 38))

        return canvas

    @classmethod
    def _make_visualization_canvas(
        cls,
        images: list,
        pred_action: np.ndarray,
        gt_action: np.ndarray | None,
    ) -> Image.Image:
        view_imgs = [
            cls._fit_for_viz(cls._to_rgb_pil(img))
            for img in images[:4]
            if img is not None
        ]
        chart = cls._draw_action_comparison(pred_action, gt_action)

        gap = 8
        if view_imgs:
            view_width = sum(img.width for img in view_imgs) + gap * (len(view_imgs) - 1)
            view_height = max(img.height for img in view_imgs)
        else:
            view_width = 0
            view_height = 0

        width = max(chart.width, view_width)
        height = chart.height + (view_height + gap if view_imgs else 0)
        canvas = Image.new("RGB", (width, height), "white")

        x = 0
        for img in view_imgs:
            canvas.paste(img, (x, 0))
            x += img.width + gap
        canvas.paste(chart, (0, view_height + gap if view_imgs else 0))
        return canvas

    @classmethod
    def _pil_to_normalized_chw(cls, image, device: torch.device | None = None) -> torch.Tensor:
        """Convert a visualization image to CHW tensor in [-1, 1]."""
        pil = cls._to_rgb_pil(image)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        tensor = (tensor - 0.5) / 0.5
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    @classmethod
    def _make_visualization_image_batch(
        cls,
        examples: list[dict],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build ``[B, V, C, H, W]`` normalized image tensor for aux visualizers."""
        per_sample: list[list[torch.Tensor]] = []
        max_views = 0
        for example in examples:
            images = example.get("image", [])
            if isinstance(images, Image.Image) or not isinstance(images, (list, tuple)):
                images = [images]
            tensors = [
                cls._pil_to_normalized_chw(img, device=device)
                for img in images
                if img is not None
            ]
            per_sample.append(tensors)
            max_views = max(max_views, len(tensors))

        if max_views == 0:
            raise ValueError("visualize_batch requires at least one image view")

        template = next(t for tensors in per_sample for t in tensors)
        padded_samples = []
        for tensors in per_sample:
            if not tensors:
                tensors = [torch.zeros_like(template)]
            while len(tensors) < max_views:
                tensors.append(torch.zeros_like(tensors[0]))
            padded_samples.append(torch.stack(tensors[:max_views], dim=0))
        return torch.stack(padded_samples, dim=0)

    def _visualization_forward_context(
        self,
        selected: list[dict],
    ) -> tuple[list[dict], np.ndarray, torch.Tensor, dict]:
        """Run one inference forward pass and build shared aux visualization inputs."""
        examples = [
            self._unpack_lerobot_sample(e) if "__trajectory_id" in e else e
            for e in selected
        ]

        batch_images = [self._force_resize_640(e["image"]) for e in examples]
        examples = [
            {**example, "image": images}
            for example, images in zip(examples, batch_images)
        ]
        instructions = [e["lang"] for e in examples]

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = (
            f" Please predict the next {self.chunk_len} robot actions: "
            f"<action>{action_tokens}<action>."
        )
        instructions = [s + prompt_suffix for s in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                hidden, qwen_inputs["input_ids"], action_token_id=self.action_token_id,
            )
            pred_actions = self.action_model.predict_action(action_queries)

        batch_dict = self._collate_aux(examples, qwen_inputs)
        batch_dict["image"] = self._make_visualization_image_batch(
            examples, device=qwen_inputs["input_ids"].device,
        )
        batch_dict["instruction"] = [e["lang"] for e in examples]

        return examples, pred_actions.detach().cpu().numpy(), hidden, batch_dict

    def _collect_aux_head_visualizations(
        self,
        hidden_states: torch.Tensor,
        batch_dict: dict,
        num_samples: int,
        outputs: dict,
    ) -> dict:
        """Append enabled aux-head visualizations to ``outputs``."""
        aux_heads = getattr(self, "aux_heads", {})
        if not aux_heads:
            return outputs

        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        for name, head in aux_heads.items():
            if not hasattr(head, "visualize"):
                continue
            try:
                mask = self._resolve_head_mask(name, batch_dict, batch_size, device)
                kwargs = {}
                if name == "pose":
                    kwargs["camera_params"] = getattr(head, "camera_params", None)
                images = head.visualize(
                    hidden_states,
                    batch_dict,
                    mask=mask,
                    num_samples=num_samples,
                    **kwargs,
                )
                for idx, image in enumerate(images or []):
                    outputs[f"viz/uamvla_oft/{name}_{idx}"] = image
            except Exception as exc:
                logger.warning("UamVLAOFT %s visualization failed: %s", name, exc)
        return outputs

    def _load_image_future(self, trajectory_id: int, base_index: int) -> torch.Tensor | None:
        """Read the task episode's terminal primary video frame.

        Returns CHW float in [-1, 1] (matching VAE input normalization spec
        in :meth:`FutureHead._normalize_for_vae`).
        """
        video_path = self._image_future_video_path(trajectory_id)
        if not video_path.exists():
            return None
        try:
            import decord
            # Do NOT call decord.bridge.set_bridge("torch") — it's a global setting
            # that corrupts LeRobot's gr00t_lerobot.video.get_frames_by_timestamps
            # (it calls frames.asnumpy() which only exists under the default
            # 'native' bridge). Use the default bridge and convert manually.
            vr = decord.VideoReader(str(video_path))
            future_idx = self._image_future_frame_index(len(vr))
            if future_idx is None:
                return None
            frame_np = vr[future_idx].asnumpy()          # (H, W, C) uint8 ndarray
            chw = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0  # (C,H,W) in [0,1]
            return (chw - 0.5) / 0.5                      # → [-1, 1]
        except Exception as e:
            logger.warning(
                f"Failed to load image_future for traj={trajectory_id} "
                f"base={base_index}: {e}"
            )
            return None

    def _load_image_action_future(self, trajectory_id: int, base_index: int) -> torch.Tensor | None:
        """Read the local future frame aligned to the current action chunk."""
        video_path = self._image_future_video_path(trajectory_id)
        if not video_path.exists():
            return None
        try:
            import decord

            vr = decord.VideoReader(str(video_path))
            future_idx = self._image_action_future_frame_index(base_index, len(vr))
            if future_idx is None:
                return None
            frame_np = vr[future_idx].asnumpy()
            chw = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
            return (chw - 0.5) / 0.5
        except Exception as e:
            logger.warning(
                f"Failed to load image_action_future for traj={trajectory_id} "
                f"base={base_index}: {e}"
            )
            return None

    # ──────────────────────────────────────────────────────────────────
    #  Aux-head helper utilities (used by PR 5+)
    # ──────────────────────────────────────────────────────────────────
    def _resolve_head_mask(
        self,
        head_name: str,
        batch_dict: dict,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-sample boolean mask for an aux head.

        Universal heads (none currently) default to all-True when their
        ``<head>_mask`` key is missing.  Non-universal heads default to
        all-False — the head's ``compute_loss`` early-exits via
        ``not mask.any()`` and returns a dummy loss, preserving DeepSpeed
        ZeRO-2 all-reduce shape across ranks.
        """
        mask = batch_dict.get(f"{head_name}_mask")
        if mask is None:
            fill = head_name in _UNIVERSAL_HEADS
            return torch.full((batch_size,), fill, dtype=torch.bool, device=device)
        return mask

    @staticmethod
    def _aux_metric_log_key(head_name: str, metric_name: str) -> str:
        prefix = f"{head_name}_"
        metric_core = (
            metric_name[len(prefix):]
            if metric_name.startswith(prefix)
            else metric_name
        )
        return f"{head_name}_{metric_core}_raw"

    @staticmethod
    def _global_step_from_kwargs(kwargs: dict) -> int:
        value = kwargs.get("global_step", 0)
        if torch.is_tensor(value):
            return int(value.detach().item())
        return int(value or 0)

    def _compute_aux_training_losses(
        self,
        total: torch.Tensor,
        hidden: torch.Tensor,
        batch_dict: dict,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        """Run aux heads through the budget suite, falling back to legacy direct sums."""
        log_metrics: dict = {}
        aux_heads = getattr(self, "aux_heads", {})

        if hasattr(self, "aux_suite"):
            masks = {
                name: self._resolve_head_mask(name, batch_dict, hidden.shape[0], hidden.device)
                for name in aux_heads
            }
            aux_loss, aux_metrics = self.aux_suite(
                action_loss=total,
                hidden_states=hidden,
                batch=batch_dict,
                masks=masks,
                global_step=global_step,
            )
            return total + aux_loss, aux_metrics

        for name, head in aux_heads.items():
            mask = self._resolve_head_mask(name, batch_dict, hidden.shape[0], hidden.device)
            out = head.compute_loss(hidden, batch_dict, mask=mask)
            if out.loss is not None:
                total = total + out.loss
                log_metrics[f"{name}_loss_weighted"] = out.loss.detach()
            for metric_name, metric_value in out.metrics.items():
                log_metrics[
                    self._aux_metric_log_key(name, metric_name)
                ] = metric_value

        return total, log_metrics

    def _collate_aux(self, examples: List[dict], qwen_inputs: dict) -> dict:
        """Stack per-sample optional fields into batch tensors.

        REQUIREMENT (spec §4.5, codex Showstopper #3): the returned
        ``batch_dict`` MUST contain ``input_ids`` because the future and
        recon heads use it to locate ``<|image_pad|>`` positions.
        """
        batch_dict: dict = {
            "input_ids": qwen_inputs["input_ids"],
        }
        batch_dict.update(stack_optional_tensor_fields(
            examples,
            [
                "image_target",
                "image_future",
                "point_cloud",
                "depth_target",
                "grounding_mask",
                "affordance_heatmap",
                "image_action_future",
                "action",
            ],
        ))
        batch_dict.update(stack_optional_string_fields(examples, ["grounding_level"]))
        if "image_target_mask" in batch_dict:
            batch_dict["recon_mask"] = batch_dict["image_target_mask"]
        if "image_future_mask" in batch_dict:
            batch_dict["future_mask"] = batch_dict["image_future_mask"]
        if "depth_target_mask" in batch_dict:
            batch_dict["depth_mask"] = batch_dict["depth_target_mask"]
        if "affordance_heatmap_mask" in batch_dict:
            batch_dict["affordance_mask"] = batch_dict["affordance_heatmap_mask"]
        if "image_action_future_mask" in batch_dict:
            batch_dict["action_conditioned_future_mask"] = batch_dict["image_action_future_mask"]

        pose_out = stack_pose_gt(examples)
        if pose_out is not None:
            batch_dict["pose_gt"] = pose_out["pose_gt"]
            batch_dict["pose_mask"] = pose_out["pose_mask"]

        cam_out = stack_static_cam_extrinsic(examples)
        if cam_out is not None:
            batch_dict["static_cam_extrinsic"] = cam_out["static_cam_extrinsic"]
            batch_dict["static_cam_extrinsic_mask"] = cam_out["static_cam_extrinsic_mask"]

        device = qwen_inputs["input_ids"].device
        return _move_to_device(batch_dict, device)

    # ──────────────────────────────────────────────────────────────────
    #  Training forward
    # ──────────────────────────────────────────────────────────────────
    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Training forward pass.

        Steps:
          1. ``_unpack_lerobot_sample`` each example.
          2. ``_force_resize_640`` each image list.
          3. Append the 🔍-emoji prompt suffix (mirrors parent QwenOFT).
          4. ``build_qwenvl_inputs`` and run Qwen3-VL with
             ``output_hidden_states=True``.
          5. Assert ``image_pad`` token count equals ``ppv * num_views``.
          6. Gather 🔍 query positions and L1-regress actions.
          7. Run aux heads (no-op in PR 4) and accumulate losses.
        """
        # ① Unpack LeRobot → framework
        examples = [self._unpack_lerobot_sample(e) for e in examples]

        batch_images = [self._force_resize_640(e["image"]) for e in examples]
        instructions = [e["lang"] for e in examples]
        gt_actions = [e["action"] for e in examples]

        # ② Prompt suffix with 🔍 placeholders (same pattern as parent)
        action_tokens = self.action_token * self.chunk_len  # "🔍" * H
        prompt_suffix = (
            f" Please predict the next {self.chunk_len} robot actions: "
            f"<action>{action_tokens}<action>."
        )
        instructions = [s + prompt_suffix for s in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = qwenvl_outputs.hidden_states[-1]  # (B, L, H)

        # ③ image_pad token count invariant
        input_ids = qwen_inputs["input_ids"]
        image_token_id = getattr(
            self.qwen_vl_interface,
            "image_token_id",
            self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        )
        ppv = 400
        num_views = len(examples[0]["image"])  # 2 for CALVIN (primary + wrist)
        img_token_count = (input_ids == image_token_id).sum(dim=1)
        expected = ppv * num_views
        if not (img_token_count == expected).all():
            raise RuntimeError(
                f"image_pad token count mismatch: expected {expected} per sample, "
                f"got {img_token_count.tolist()}. Check _force_resize_640 + Qwen3VLProcessor."
            )

        # ④ Action loss (L1 regression on 🔍 queries)
        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                hidden, input_ids, action_token_id=self.action_token_id,
            )
            pred_actions = self.action_model.predict_action(action_queries)
            gt_actions_t = torch.as_tensor(
                np.array([
                    a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
                    for a in gt_actions
                ]),
                device=pred_actions.device,
                dtype=pred_actions.dtype,
            )
            gt_actions_t = gt_actions_t[:, -self.action_horizon:, :]
            total = self.l1_loss(pred_actions, gt_actions_t)
        log_metrics = {"action_loss_l1": total.detach()}

        # ⑤ Aux head losses (no-op when no aux heads are enabled)
        batch_dict = self._collate_aux(examples, qwen_inputs)
        assert "input_ids" in batch_dict, "future/recon heads require input_ids"
        total, aux_metrics = self._compute_aux_training_losses(
            total,
            hidden,
            batch_dict,
            global_step=self._global_step_from_kwargs(kwargs),
        )
        log_metrics.update(aux_metrics)

        return {"action_loss": total, **log_metrics}

    @torch.inference_mode()
    def visualize_batch(self, batch: List[dict], n_samples: int = 1) -> dict:
        """Visualize action predictions plus any enabled aux heads."""
        if not isinstance(batch, list):
            batch = [batch]
        limit = min(max(int(n_samples), 0), len(batch))
        if limit == 0:
            return {}

        selected = batch[:limit]
        was_training = bool(getattr(self, "training", False))
        self.eval()
        try:
            examples, pred_actions, hidden_states, batch_dict = (
                self._visualization_forward_context(selected)
            )
            if pred_actions.ndim == 2:
                pred_actions = pred_actions[None, ...]

            try:
                import wandb
            except ImportError:
                wandb = None

            outputs = {}
            for idx, sample in enumerate(examples):
                pred = pred_actions[idx]
                horizon = int(getattr(self, "action_horizon", pred.shape[0]))

                gt_action = None
                if "action" in sample:
                    gt_action = self._to_action_numpy(sample["action"])
                    if gt_action.ndim >= 2:
                        gt_action = gt_action[-horizon:, :pred.shape[-1]]

                images = sample.get("image", [])
                if isinstance(images, Image.Image) or not isinstance(images, (list, tuple)):
                    images = [images]

                canvas = self._make_visualization_canvas(
                    images=list(images),
                    pred_action=pred,
                    gt_action=gt_action,
                )
                caption = str(sample.get("lang", ""))
                value = wandb.Image(canvas, caption=caption) if wandb is not None else canvas
                outputs[f"viz/uamvla_oft/action_{idx}"] = value

            return self._collect_aux_head_visualizations(
                hidden_states,
                batch_dict,
                num_samples=limit,
                outputs=outputs,
            )
        finally:
            if was_training:
                self.train()

    # ──────────────────────────────────────────────────────────────────
    #  Inference
    # ──────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        """Inference path.

        Must apply ``_force_resize_640`` so the image_grid_thw matches
        training (codex Significant gap #3 fix).  Parent's
        ``predict_action`` only resizes when ``config.framework.obs_image_size``
        is set; here we resize unconditionally to keep tokens-per-view
        invariant.
        """
        if not isinstance(examples, list):
            examples = [examples]

        # Eval-side examples may be raw framework dicts already; unpack
        # only when the LeRobot sidecar keys are present.
        examples = [
            self._unpack_lerobot_sample(e) if "__trajectory_id" in e else e
            for e in examples
        ]

        batch_images = [self._force_resize_640(e["image"]) for e in examples]
        instructions = [e["lang"] for e in examples]

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = (
            f" Please predict the next {self.chunk_len} robot actions: "
            f"<action>{action_tokens}<action>."
        )
        instructions = [s + prompt_suffix for s in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                hidden, qwen_inputs["input_ids"], action_token_id=self.action_token_id,
            )
            pred_actions = self.action_model.predict_action(action_queries)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
