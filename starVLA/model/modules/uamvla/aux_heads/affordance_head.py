from __future__ import annotations

from starVLA.model.modules.uamvla.aux_heads.spatial_map_denoising_head import SpatialMapDenoisingHead


class AffordanceHeatmapDenoisingHead(SpatialMapDenoisingHead):
    """Actionable interaction heatmap denoising on the token grid."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("target_key", "affordance_heatmap")
        kwargs.setdefault("mask_key", "affordance_mask")
        kwargs.setdefault("metric_prefix", "affordance")
        super().__init__(*args, **kwargs)
