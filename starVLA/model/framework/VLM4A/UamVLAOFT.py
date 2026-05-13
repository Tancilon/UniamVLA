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

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT
from starVLA.model.modules.uamvla.collator_helpers import (
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
        dataset_name = mixture[0][0]
        self.sidecar_root = Path(self.config.datasets.vla_data.data_root_dir) / dataset_name

        # Per-trajectory image_target cache (single PNG per traj on disk).
        self._image_target_cache: dict[int, torch.Tensor] = {}

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

    # ──────────────────────────────────────────────────────────────────
    #  Image resize — shared between training and inference
    # ──────────────────────────────────────────────────────────────────
    def _force_resize_640(self, image_list: list) -> list:
        """Resize each PIL image to 640x640 via ``Image.BICUBIC``.

        Qwen3VLProcessor produces ``image_grid_thw=(1, 40, 40)``
        (i.e. ppv=400) at this resolution, which matches the
        image_pad token count invariant in :meth:`forward`. Both training
        and inference must call this to keep tokens-per-view consistent.
        See spec §5.2.
        """
        return [
            img.resize((640, 640), Image.BICUBIC) if isinstance(img, Image.Image) else img
            for img in image_list
        ]

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
            sidecar ``<sidecar_root>/image_targets/<traj>.png`` if present
            (cached by trajectory_id).

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

        if traj not in self._image_target_cache:
            it_path = self.sidecar_root / "image_targets" / f"{traj}.png"
            if it_path.exists():
                arr = np.array(Image.open(it_path).convert("RGB"), dtype=np.uint8)
                self._image_target_cache[traj] = (
                    torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                )
        if traj in self._image_target_cache:
            out["image_target"] = self._image_target_cache[traj]

        # image_future deferred — PR 6 (FutureHead) will populate.
        return out

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
            examples, ["image_target", "image_future", "point_cloud"],
        ))
        if "image_target_mask" in batch_dict:
            batch_dict["recon_mask"] = batch_dict["image_target_mask"]
        if "image_future_mask" in batch_dict:
            batch_dict["future_mask"] = batch_dict["image_future_mask"]

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

        # ⑤ Aux head losses (no-op in PR 4 — self.aux_heads is empty)
        batch_dict = self._collate_aux(examples, qwen_inputs)
        assert "input_ids" in batch_dict, "future/recon heads require input_ids"
        for name, head in self.aux_heads.items():
            mask = self._resolve_head_mask(name, batch_dict, hidden.shape[0], hidden.device)
            out = head.compute_loss(hidden, batch_dict, mask=mask)
            if out.loss is not None:
                total = total + out.loss
                log_metrics[f"{name}_loss"] = out.loss.detach()
            log_metrics.update({f"{name}_{k}": v for k, v in out.metrics.items()})

        return {"action_loss": total, **log_metrics}

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
