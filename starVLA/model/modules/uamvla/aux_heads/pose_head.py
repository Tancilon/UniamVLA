"""PoseHead auxiliary branch for 6D pose estimation.

Estimates 6D object pose (rotation + translation) via score-based diffusion in
SE(3) space, conditioned on LLM hidden states (via multi-query cross-attention)
and point cloud features (via PointNet2). All weights are trained from scratch
as part of UamVLA pretraining; gradient flows back to the LLM through the
semantic conditioning path.
"""

import logging

import torch
import torch.nn as nn

from starVLA.model.modules.uamvla.aux_heads.base import AuxHead, HeadOutput
from starVLA.model.modules.uamvla.components.pose.pts_encoder import PointNet2Wrapper
from starVLA.model.modules.uamvla.components.pose.score_net import PoseScoreNet
from starVLA.model.modules.uamvla.components.pose.sde import init_sde
from starVLA.model.modules.uamvla.components.pose.sampler import cond_pc_sampler
from starVLA.model.modules.uamvla.components.pose.pose_utils import (
    get_pose_dim,
    rotation_6d_to_matrix,
)
from starVLA.model.modules.uamvla.components.query_reader import TaskQueryReader

logger = logging.getLogger(__name__)


class PoseHead(AuxHead):
    """6D pose estimation head using score-based diffusion in SE(3) space."""

    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 4,
        semantic_dim: int = 512,
        pose_mode: str = "rot_matrix",
        sde_mode: str = "ve",
        repeat_num: int = 4,
        sampling_steps: int = 500,
        pointnet_input_channels: int = 0,
        loss_weight: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.repeat_num = repeat_num
        self.sampling_steps = sampling_steps
        self.pose_mode = pose_mode
        self.pose_dim = get_pose_dim(pose_mode)

        # SDE components (not nn.Module)
        prior_fn, marginal_prob_fn, sde_fn, eps, T = init_sde(sde_mode)
        self.prior_fn = prior_fn
        self.marginal_prob_fn = marginal_prob_fn
        self.sde_fn = sde_fn
        self.sde_eps = eps
        self.sde_T = T

        # Semantic conditioning path: LLM hidden_states → 4 learnable queries
        # via cross-attention → concat → Linear → semantic_feat [B, semantic_dim].
        # This is the only gradient path reaching the LLM from pose loss.
        self.query_reader = TaskQueryReader(num_queries=num_queries, hidden_dim=hidden_size)
        self.semantic_proj = nn.Linear(num_queries * hidden_size, semantic_dim)

        # Geometry path — decoupled from LLM on purpose. Point cloud goes
        # through a plain PointNet2 (no DINO fusion).
        self.pts_encoder = PointNet2Wrapper(
            input_channels=pointnet_input_channels,
            output_dim=1024,
        )

        self.score_net = PoseScoreNet(
            marginal_prob_func=marginal_prob_fn,
            semantic_dim=semantic_dim,
            pose_mode=pose_mode,
            regression_head="RT",
        )

    def _get_score_dtype(self) -> torch.dtype:
        """Return the score network's parameter dtype, or float32 if the
        module has no parameters (e.g. MagicMock in unit tests).
        """
        param = next(self.score_net.parameters(), None)
        return param.dtype if param is not None else torch.float32

    def _prepare_score_inputs(
        self,
        hidden_states: torch.Tensor,
        point_cloud: torch.Tensor,
        mask: torch.Tensor | None,
        score_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute semantic_feat, pts_feat, pts_center, and valid_index.

        Dtype contract (audit finding H1, fixed 2026-04-15):

        - ``semantic_feat`` and ``pts_feat`` are returned in the score
          network's parameter dtype. Under DeepSpeed BF16 this is bf16;
          on CPU/float32 it is float32. This is what the caller must feed
          into ``self.score_net``; PyTorch does NOT auto-upcast a float32
          input through a bf16 Linear — it raises RuntimeError.
        - ``pts_center`` stays float32 regardless. It is only used for
          world-frame de-centering at ``sampler.py:55-56`` and during GT
          residual computation, where meter-scale precision matters.
        - ``pts_encoder`` (PointNet2) always receives float32 input because
          its CUDA kernels only support float32.

        Args:
            hidden_states: [B, L, H] LLM hidden states.
            point_cloud: [B, N, 3] world-frame object point cloud. May
                contain zero-padded rows for samples that lack pose data.
            mask: [B] bool mask over the batch dimension. If None, all
                samples are treated as valid.

        Returns:
            semantic_feat: [B', semantic_dim] in score_dtype
            pts_feat: [B', 1024] in score_dtype
            pts_center: [B', 3] float32 — centroid of the masked point cloud
            valid_index: [B'] long — indices of masked samples in the
                original batch, used by predict() to scatter outputs back.
        """
        if mask is None:
            mask = torch.ones(
                hidden_states.shape[0],
                dtype=torch.bool,
                device=hidden_states.device,
            )

        valid_index = mask.nonzero(as_tuple=True)[0]
        h_sub = hidden_states[mask]                    # [B', L, H]
        pc_sub = point_cloud[mask]                     # [B', N, 3]

        if score_dtype is None:
            score_dtype = self._get_score_dtype()

        # Semantic path (LLM gradient path). Cast hidden_states to match the
        # query_reader's parameter dtype so BF16 LLM outputs can flow through
        # a float32 module (CPU tests) or a BF16 module (DeepSpeed train)
        # without a dtype mismatch in the in-projection matmul.
        query_reader_dtype = next(self.query_reader.parameters()).dtype
        h_sub = h_sub.to(query_reader_dtype)
        queries = self.query_reader(h_sub)             # [B', num_queries, H]
        B_sub = queries.shape[0]
        semantic_feat = self.semantic_proj(queries.reshape(B_sub, -1))
        semantic_feat = semantic_feat.to(score_dtype)

        # Geometry path — PointNet2 CUDA needs float32 input; cast the
        # feature back to score_dtype afterwards.
        pts_center = pc_sub.mean(dim=1).float()                    # [B', 3]
        pc_centered = (pc_sub - pts_center.unsqueeze(1)).float()   # [B', N, 3]
        pts_feat = self.pts_encoder(pc_centered).to(score_dtype)   # [B', 1024]

        return (semantic_feat, pts_feat, pts_center, valid_index)

    def compute_loss(
        self, hidden_states: torch.Tensor, batch: dict, mask: torch.Tensor,
    ) -> HeadOutput:
        if not mask.any():
            return HeadOutput(
                loss=self.get_dummy_loss(),
                metrics={"pose_loss": 0.0, "pose_translation_residual_mean": 0.0},
                predictions=None,
            )
        assert batch.get("point_cloud") is not None, (
            "PoseHead.compute_loss requires batch['point_cloud']; "
            "got None with non-empty mask."
        )

        device = hidden_states.device
        score_dtype = self._get_score_dtype()

        # 1. Shared feature prep — pts_center float32, other features in score_dtype.
        semantic_feat, pts_feat, pts_center, _ = self._prepare_score_inputs(
            hidden_states=hidden_states,
            point_cloud=batch["point_cloud"],
            mask=mask,
            score_dtype=score_dtype,
        )
        B_sub = semantic_feat.shape[0]

        # 2. GT pose in the object-local frame. pts_center is float32 so the
        #    residual metric below is computed in float32 before we cast into
        #    score_dtype for the score matching loss.
        pose_gt_rotation = batch["pose_gt"]["rotation"][mask]
        pose_gt_translation = batch["pose_gt"]["translation"][mask]
        if pose_gt_rotation.ndim == 3:
            pose_gt_rotation = torch.cat(
                [pose_gt_rotation[:, :, 0], pose_gt_rotation[:, :, 1]], dim=-1
            )
        pose_gt_rotation = pose_gt_rotation.float()
        pose_gt_translation = pose_gt_translation.float()

        translation_residual = pose_gt_translation - pts_center          # [B', 3] float32

        # 3. Debug metric — sub-cm on clean segmentation.
        residual_norm_mean = (
            translation_residual.norm(dim=-1).mean().detach().item()
        )

        gt_pose = torch.cat(
            [pose_gt_rotation, translation_residual], dim=-1
        ).to(score_dtype)                                                  # [B', 9]

        # 4. Score matching loss. All tensors reaching score_net are in
        #    score_dtype — PyTorch does NOT auto-upcast a float32 input
        #    through a bf16 Linear, it raises RuntimeError (audit H1).
        semantic_feat_rep = semantic_feat.repeat(self.repeat_num, 1)
        pts_feat_rep = pts_feat.repeat(self.repeat_num, 1)
        gt_pose_rep = gt_pose.repeat(self.repeat_num, 1)
        B_rep = B_sub * self.repeat_num

        random_t = (
            torch.rand(B_rep, 1, device=device, dtype=score_dtype)
            * (self.sde_T - self.sde_eps)
            + self.sde_eps
        )
        mean, std = self.marginal_prob_fn(gt_pose_rep, random_t)
        if std.ndim == 1:
            std = std.unsqueeze(-1)
        z = torch.randn_like(gt_pose_rep)
        perturbed_pose = mean + z * std

        data = {
            "semantic_feat": semantic_feat_rep,
            "pts_feat": pts_feat_rep,
            "sampled_pose": perturbed_pose,
            "t": random_t,
        }
        score = self.score_net(data)

        raw_loss = torch.sum((std * score + z) ** 2, dim=-1).mean()
        weighted_loss = self.loss_weight * raw_loss

        # 5. Cast loss back to LLM dtype so Lightning total_loss stays single-dtype.
        weighted_loss = weighted_loss.to(hidden_states.dtype)

        return HeadOutput(
            loss=weighted_loss,
            metrics={
                "pose_loss": raw_loss.detach().float().item(),
                "pose_translation_residual_mean": residual_norm_mean,
            },
            predictions=None,
        )

    def predict(
        self,
        hidden_states: torch.Tensor,
        batch: dict,
        mask: torch.Tensor | None = None,
    ) -> HeadOutput:
        assert batch.get("point_cloud") is not None, (
            "PoseHead.predict requires batch['point_cloud']"
        )
        device = hidden_states.device
        B_full = hidden_states.shape[0]

        if mask is None:
            # Backward-compat: caller asserts all samples are valid (e.g.
            # visualize() pre-filters with valid_indices).
            mask = torch.ones(B_full, dtype=torch.bool, device=device)

        if not mask.any():
            nan_rot = torch.full(
                (B_full, 3, 3), float("nan"), dtype=torch.float32, device=device,
            )
            nan_trans = torch.full(
                (B_full, 3), float("nan"), dtype=torch.float32, device=device,
            )
            return HeadOutput(
                loss=None,
                metrics={},
                predictions={
                    "pose_rotation": nan_rot,
                    "pose_translation": nan_trans,
                    "pose_mask": mask,
                },
            )

        # Shared feature prep — score_dtype features, float32 pts_center,
        # filtered by mask. pts_center is computed only over valid samples,
        # so padded rows cannot leak into the world-frame recovery at
        # sampler.py:55-56.
        score_dtype = self._get_score_dtype()
        semantic_feat, pts_feat, pts_center, valid_index = self._prepare_score_inputs(
            hidden_states=hidden_states,
            point_cloud=batch["point_cloud"],
            mask=mask,
            score_dtype=score_dtype,
        )

        data = {
            "semantic_feat": semantic_feat,
            "pts_feat": pts_feat,
            "pts_center": pts_center,
        }
        result = cond_pc_sampler(
            score_model=self.score_net,
            data=data,
            prior=self.prior_fn,
            sde_coeff=self.sde_fn,
            num_steps=self.sampling_steps,
            eps=self.sde_eps,
            snr=0.16,
            pose_mode=self.pose_mode,
            device=str(device),
        )  # [B_sub, 9]

        rot_6d_sub = result[:, :-3]
        translation_sub = result[:, -3:]
        rotation_sub = rotation_6d_to_matrix(rot_6d_sub)  # [B_sub, 3, 3]

        # Scatter back into full batch shape. Masked-out slots are NaN — a
        # loud failure mode, not a silent zero that could render a wrong bbox.
        rotation_full = torch.full(
            (B_full, 3, 3), float("nan"), dtype=torch.float32, device=device,
        )
        translation_full = torch.full(
            (B_full, 3), float("nan"), dtype=torch.float32, device=device,
        )
        rotation_full[valid_index] = rotation_sub.float()
        translation_full[valid_index] = translation_sub.float()

        return HeadOutput(
            loss=None,
            metrics={},
            predictions={
                "pose_rotation": rotation_full,
                "pose_translation": translation_full,
                "pose_mask": mask,
            },
        )

    def visualize(
        self,
        hidden_states: torch.Tensor,
        batch: dict,
        mask: torch.Tensor,
        num_samples: int = 1,
        camera_params: dict | None = None,
        **kwargs,
    ) -> list:
        """Generate side-by-side pose visualization: GT Pose | Predicted Pose."""
        skip_reason = None
        if camera_params is None:
            skip_reason = "camera_params is None (stats_path not configured?)"
        elif "intrinsic" not in camera_params:
            skip_reason = "camera_params missing 'intrinsic' key"
        elif "static_cam_extrinsic" not in batch:
            skip_reason = "batch missing 'static_cam_extrinsic' (dataset not preprocessed with per-sample extrinsic?)"
        elif not mask.any():
            skip_reason = "pose_mask all False for this batch"
        if skip_reason is not None:
            logger.info("PoseHead.visualize skipped: %s", skip_reason)
            return []

        import wandb
        from starVLA.utils.vis_draw import tensor_to_pil, concat_images_h
        from starVLA.utils.geometry import rotation_6d_to_matrix_np

        n = min(num_samples, mask.sum().item())
        valid_indices = mask.nonzero(as_tuple=True)[0][:n]

        # Run pose prediction
        h_subset = hidden_states[valid_indices]
        batch_subset = {"point_cloud": batch["point_cloud"][valid_indices]}
        with torch.no_grad():
            output = self.predict(h_subset, batch_subset)
        pred_rot_tensor = output.predictions["pose_rotation"]
        pred_trans_tensor = output.predictions["pose_translation"]
        assert not torch.isnan(pred_rot_tensor).any(), (
            "PoseHead.visualize got NaN rotation from predict — this should "
            "never happen when batch_subset is pre-filtered."
        )
        assert not torch.isnan(pred_trans_tensor).any(), (
            "PoseHead.visualize got NaN translation from predict."
        )
        pred_rot = pred_rot_tensor.detach().float().cpu().numpy()  # (n, 3, 3)
        pred_trans = pred_trans_tensor.detach().float().cpu().numpy()  # (n, 3)

        import numpy as _np_diag
        gt_trans_diag = batch["pose_gt"]["translation"][valid_indices].detach().float().cpu().numpy()
        logger.info(
            "PoseHead.visualize DIAG pred_trans=%s pred_trans_nan=%s pred_trans_abs_max=%.4f "
            "gt_trans=%s pred_rot_det=%s",
            pred_trans.tolist(),
            bool(_np_diag.isnan(pred_trans).any()),
            float(_np_diag.nanmax(_np_diag.abs(pred_trans))) if pred_trans.size else 0.0,
            gt_trans_diag.tolist(),
            [float(_np_diag.linalg.det(r)) for r in pred_rot],
        )

        intrinsic = camera_params["intrinsic"]
        ext_rot = batch["static_cam_extrinsic"]["rotation"]      # (B, 3, 3)
        ext_trans = batch["static_cam_extrinsic"]["translation"] # (B, 3)

        results = []
        for i, idx in enumerate(valid_indices):
            idx_int = idx.item()
            input_pil = tensor_to_pil(batch["image"][idx_int, 0])
            pts = batch["point_cloud"][idx_int].detach().float().cpu().numpy()

            cam_R = ext_rot[idx_int].detach().float().cpu().numpy()
            cam_t = ext_trans[idx_int].detach().float().cpu().numpy()

            gt_R_6d = batch["pose_gt"]["rotation"][idx_int].detach().float().cpu().numpy()
            gt_R = rotation_6d_to_matrix_np(gt_R_6d)
            gt_t = batch["pose_gt"]["translation"][idx_int].detach().float().cpu().numpy()

            gt_img = input_pil.copy()
            self._draw_pose_on_image(gt_img, pts, gt_R, gt_t, intrinsic, cam_R, cam_t)

            pred_img = input_pil.copy()
            self._draw_pose_on_image(pred_img, pts, pred_rot[i], pred_trans[i], intrinsic, cam_R, cam_t)

            combined = concat_images_h([gt_img, pred_img])
            caption = batch["instruction"][idx_int] if "instruction" in batch else ""
            results.append(wandb.Image(combined, caption=caption))
        return results

    @staticmethod
    def _draw_pose_on_image(img, pts, R, t, intrinsic, cam_R, cam_t):
        """Draw bbox + rotation axes on a PIL image in-place."""
        import numpy as _np
        from PIL import ImageDraw as _ImageDraw
        from starVLA.utils.vis_draw import (
            world_to_pixel, compute_obb_corners, draw_bbox_colorful, draw_axes_3d,
        )

        # Camera intrinsic is calibrated for the MuJoCo render resolution
        # (e.g. 256x256); the image tensor shown in wandb has been resized by
        # the data transforms (e.g. 384x384). Scale intrinsic to match.
        scale = img.size[0] / intrinsic["width"]
        intrinsic_scaled = {
            "fx": intrinsic["fx"] * scale,
            "fy": intrinsic["fy"] * scale,
            "cx": intrinsic["cx"] * scale,
            "cy": intrinsic["cy"] * scale,
        }

        draw = _ImageDraw.Draw(img)
        box_corners = compute_obb_corners(pts, R, t)
        box_pixels = world_to_pixel(box_corners, intrinsic_scaled, cam_R, cam_t)

        # White halo (width 5) then bright colored edges (width 2) on top —
        # ensures the bbox is visible on top of textured target objects.
        draw_bbox_colorful(draw, box_pixels, width=5, halo=True)
        draw_bbox_colorful(draw, box_pixels, width=2)

        # Fixed axis length (10 cm) so axes are visible regardless of object
        # size — was previously derived from local_extent which made axes
        # invisible on small objects.
        axis_length = 0.10

        axes_world = {
            "origin": t,
            "x": t + R[:, 0] * axis_length,
            "y": t + R[:, 1] * axis_length,
            "z": t + R[:, 2] * axis_length,
        }
        axes_pts = _np.array([axes_world["origin"], axes_world["x"], axes_world["y"], axes_world["z"]])
        axes_px = world_to_pixel(axes_pts, intrinsic_scaled, cam_R, cam_t)
        draw_axes_3d(draw, axes_px, width=3, tip_radius=4)
