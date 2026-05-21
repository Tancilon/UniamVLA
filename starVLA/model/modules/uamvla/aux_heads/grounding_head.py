from __future__ import annotations

from starVLA.model.modules.uamvla.aux_heads.spatial_map_denoising_head import SpatialMapDenoisingHead


class GroundingMaskDenoisingHead(SpatialMapDenoisingHead):
    """Instruction-relevant part/object mask denoising on the token grid."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("target_key", "grounding_mask")
        kwargs.setdefault("mask_key", "grounding_mask_mask")
        kwargs.setdefault("metric_prefix", "grounding")
        super().__init__(*args, **kwargs)
