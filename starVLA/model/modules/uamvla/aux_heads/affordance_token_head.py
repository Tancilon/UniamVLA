"""Affordance-token head: per-token regression onto a 2D affordance heatmap.

Same family as ``LatentDepthHead`` -- no denoiser, a pointwise projection from
the affordance tokens (a duplicated primary vision block) straight onto the
precomputed target.  The target is the VRB contact heatmap downsampled to
``afford_px_size`` (112) by runners/preprocess_affordance_px.py; each of the
49 tokens regresses its own 16x16 pixel patch.

The head is a plain Linear output (no sigmoid): the target lives in [0, 1] but
is zero-dominated, and a sigmoid would saturate around 0.  ``predict`` clamps
to [0, 1] instead.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from starVLA.model.modules.uamvla.aux_heads.base import AuxHead, HeadOutput
from starVLA.model.modules.uamvla.components.spatial_reader import slice_image_tokens


class AffordanceTokenHead(AuxHead):
    """Per-token regression from affordance tokens onto heatmap pixel patches.

    Args:
        hidden_size: Backbone hidden dimension (2560 for Qwen3-VL-4B).
        image_token_id: ``<|image_pad|>`` id -- the affordance block reuses it.
        patches_per_view: Tokens per vision block (49 at qwen_image_size=224).
        view_idx: Which vision block holds the affordance tokens
            (== num_views + 1 when the depth block is also enabled, see
            UamGR00T_LT._maybe_build_aux_heads).
        afford_px_size: Side of the stored ``affordance_px`` map (112).
        head_kind: ``pointwise_mlp`` (default) or ``linear``. 
        loss_weight: Scales the head loss.
    """

    def __init__(
        self,
        hidden_size: int,
        image_token_id: int,
        patches_per_view: int,
        view_idx: int,
        afford_px_size: int = 112,
        head_kind: str = "pointwise_mlp",
        loss_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.image_token_id = int(image_token_id)
        self.patches_per_view = int(patches_per_view)
        self.view_idx = int(view_idx)
        self.loss_weight = float(loss_weight)

        self.grid = int(math.sqrt(self.patches_per_view))
        if self.grid * self.grid != self.patches_per_view:
            raise ValueError(f"patches_per_view must be a square grid, got {patches_per_view}")
        if afford_px_size % self.grid:
            raise ValueError(f"afford_px_size {afford_px_size} not divisible by grid {self.grid}")
        self.patch = afford_px_size // self.grid  # 112 // 7 == 16

        self.ln_pre = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.head = self._build_head(head_kind, hidden_size, self.patch * self.patch)

    @staticmethod
    def _build_head(kind: str, hidden_size: int, out_dim: int) -> nn.Module:
        if kind == "linear":
            return nn.Linear(hidden_size, out_dim)
        if kind == "pointwise_mlp":
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, out_dim),
            )
        raise ValueError(f"head_kind must be 'linear' or 'pointwise_mlp', got {kind!r}")

    # ──────────────────────────────────────────────────────────────────
    #  Metrics
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
        """1 - mse / var(target).  0.0 means "no better than the batch mean"."""
        var = target.var(unbiased=False)
        if var <= 0:
            return 0.0
        return float(1.0 - F.mse_loss(pred, target) / var)

    # ──────────────────────────────────────────────────────────────────
    #  Forward paths
    # ──────────────────────────────────────────────────────────────────
    def _afford_tokens(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        tokens = slice_image_tokens(
            hidden, input_ids, self.image_token_id, self.patches_per_view, self.view_idx,
        )  # (B, patches_per_view, H)
        # fp32 for the same reason as LatentDepthHead: aux heads run outside
        # autocast, so their fp32 Linear weights would reject a bf16 input.
        return tokens.float()

    def _pixel_target(self, afford_px: torch.Tensor) -> torch.Tensor:
        return rearrange(
            afford_px, "b (h ph) (w pw) -> b (h w) (ph pw)", ph=self.patch, pw=self.patch,
        )

    def compute_loss(self, hidden_states: torch.Tensor, batch: dict, mask: torch.Tensor) -> HeadOutput:
        valid_ratio = float(mask.float().mean()) if mask.numel() else 0.0
        if not mask.any():
            return HeadOutput(
                loss=self._zero_aligned_loss(hidden_states, batch),
                metrics={"r2": 0.0, "fg_mse": 0.0, "valid_ratio": 0.0},
                predictions=None,
            )

        h = self._afford_tokens(hidden_states[mask], batch["input_ids"][mask])
        with torch.amp.autocast("cuda", dtype=torch.float32):
            pred = self.head(self.ln_pre(h))  # (B, 49, patch*patch)
        gt = self._pixel_target(batch["affordance_px"][mask]).to(pred.dtype)
        head_loss = F.mse_loss(pred, gt)
        loss = self.loss_weight * head_loss

        # The heatmap is zero-dominated: an all-zero prediction already gets a
        # low global MSE.  fg_mse watches the peaks so collapse-to-zero shows.
        fg = gt > 0.1
        fg_mse = float(F.mse_loss(pred[fg], gt[fg])) if fg.any() else 0.0

        metrics = {
            "r2": self._r2(pred, gt),
            "var_ratio": float(pred.var(unbiased=False) / gt.var(unbiased=False).clamp_min(1e-8)),
            "cosine": float(F.cosine_similarity(pred.flatten(1), gt.flatten(1), dim=-1).mean()),
            "valid_ratio": valid_ratio,
            "head_mse": float(head_loss),
            "fg_mse": fg_mse,
        }
        return HeadOutput(loss=loss, metrics=metrics, predictions=None)

    def _zero_aligned_loss(self, hidden_states: torch.Tensor, batch: dict) -> torch.Tensor:
        """Run a zero-weight dummy forward so ZeRO-3 collectives stay aligned."""
        if hidden_states.shape[0] == 0:
            return self.get_dummy_loss()
        h = self._afford_tokens(hidden_states[:1], batch["input_ids"][:1])
        with torch.amp.autocast("cuda", dtype=torch.float32):
            loss = self.head(self.ln_pre(h)).sum()
        return loss * 0.0

    # ──────────────────────────────────────────────────────────────────
    #  Inference / visualization
    # ──────────────────────────────────────────────────────────────────
    def _patches_to_map(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, N, patch*patch) -> (B, R, R) heatmap in [0, 1]."""
        maps = rearrange(
            patches, "b (h w) (ph pw) -> b (h ph) (w pw)",
            h=self.grid, w=self.grid, ph=self.patch, pw=self.patch,
        )
        return maps.clamp(0, 1)

    def predict(self, hidden_states: torch.Tensor, batch: dict) -> HeadOutput:
        h = self._afford_tokens(hidden_states, batch["input_ids"])
        with torch.amp.autocast("cuda", dtype=torch.float32):
            pred = self.head(self.ln_pre(h))
        return HeadOutput(loss=None, metrics={},
                          predictions={"affordance_maps": self._patches_to_map(pred.float())})

    @staticmethod
    def _rgb_to_pil(rgb: torch.Tensor):
        import numpy as np
        from PIL import Image

        if rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError(f"Expected RGB tensor [3,H,W], got {tuple(rgb.shape)}")
        image = ((rgb.float() * 0.5 + 0.5).clamp(0, 1) * 255).round()
        arr = image.to(torch.uint8).cpu().permute(1, 2, 0).numpy()
        return Image.fromarray(arr, mode="RGB")

    @staticmethod
    def _overlay_map_on_rgb(rgb: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.75):
        import numpy as np
        import torch.nn.functional as F
        from PIL import Image

        if rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError(f"Expected RGB tensor [3,H,W], got {tuple(rgb.shape)}")
        if heatmap.ndim == 3:
            heatmap = heatmap.squeeze(0)
        if heatmap.ndim != 2:
            raise ValueError(f"Expected 2-D heatmap, got {tuple(heatmap.shape)}")

        rgb_01 = (rgb.float() * 0.5 + 0.5).clamp(0, 1)
        gray = (
            0.299 * rgb_01[0]
            + 0.587 * rgb_01[1]
            + 0.114 * rgb_01[2]
        ).unsqueeze(0).expand_as(rgb_01)
        base = (0.85 * gray + 0.15 * rgb_01) * 0.75
        heat = F.interpolate(
            heatmap[None, None].float(),
            size=base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0].clamp(0, 1)
        red = torch.zeros_like(base)
        red[0] = 1.0
        overlay = base * (1.0 - alpha * heat[None]) + red * (alpha * heat[None])
        arr = (overlay.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        return Image.fromarray(arr.permute(1, 2, 0).numpy().astype(np.uint8), mode="RGB")

    def visualize(self, hidden_states, batch, mask, num_samples: int = 1, **kwargs) -> list:
        """RGB | GT overlay | prediction overlay."""
        mask = mask.to(device=hidden_states.device, dtype=torch.bool)
        if num_samples <= 0 or not mask.any():
            return []

        from starVLA.utils.vis_draw import concat_images_h
        from starVLA.model.modules.uamvla.aux_heads.spatial_map_denoising_head import (
            SpatialMapDenoisingHead as _S,
        )

        n = min(int(num_samples), int(mask.sum()))
        idx = mask.nonzero(as_tuple=True)[0][:n]
        was_training = bool(self.training)
        self.eval()
        try:
            with torch.no_grad():
                pred = self.predict(hidden_states[idx], {"input_ids": batch["input_ids"][idx]})
        finally:
            if was_training:
                self.train()

        maps = pred.predictions["affordance_maps"]
        gt = batch["affordance_px"][idx].float().clamp(0, 1)
        rgb_batch = batch.get("image")
        instructions = batch.get("instruction", [])
        results = []
        for i, b in enumerate(idx):
            b = int(b)
            if rgb_batch is not None:
                rgb = rgb_batch[b, 0].detach().cpu()
                combined = concat_images_h([
                    self._rgb_to_pil(rgb),
                    self._overlay_map_on_rgb(rgb, gt[i].detach().cpu()),
                    self._overlay_map_on_rgb(rgb, maps[i].detach().cpu()),
                ])
                caption = "affordance_px: RGB | GT overlay | Pred overlay"
            else:
                combined = concat_images_h([_S._map_to_pil(gt[i]), _S._map_to_pil(maps[i])])
                caption = "affordance_px: GT vs Pred"
            if b < len(instructions):
                caption = f"{caption} | {instructions[b]}"
            results.append(_S._maybe_wandb_image(combined, caption=caption))
        return results
