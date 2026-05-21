from __future__ import annotations

import torch

from starVLA.model.modules.uamvla.aux_heads.spatial_map_denoising_head import SpatialMapDenoisingHead


class DepthDenoisingHead(SpatialMapDenoisingHead):
    """Relative inverse depth denoising on the visual-token grid."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("target_key", "depth_target")
        kwargs.setdefault("mask_key", "depth_mask")
        kwargs.setdefault("metric_prefix", "depth")
        super().__init__(*args, **kwargs)

    @staticmethod
    def relative_inverse_depth_per_frame(
        depth: torch.Tensor,
        eps: float = 1.0e-6,
        max_depth: float = 10.0,
    ) -> torch.Tensor:
        depth = depth.float()
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        out = torch.zeros_like(depth)
        for i in range(depth.shape[0]):
            d = depth[i]
            valid = torch.isfinite(d) & (d > 0)
            if not valid.any():
                continue
            inv = torch.zeros_like(d)
            inv[valid] = 1.0 / torch.clamp(d[valid], min=eps, max=max_depth)
            vals = inv[valid]
            lo = torch.quantile(vals, 0.01)
            hi = torch.quantile(vals, 0.99)
            denom = torch.clamp(hi - lo, min=eps)
            rel = torch.clamp((inv - lo) / denom, 0.0, 1.0)
            rel = torch.where(valid, rel, torch.zeros_like(rel))
            out[i] = rel
        return out

    def _prepare_target_01(self, target: torch.Tensor) -> torch.Tensor:
        depth_01 = self.relative_inverse_depth_per_frame(target)
        return super()._prepare_target_01(depth_01)
