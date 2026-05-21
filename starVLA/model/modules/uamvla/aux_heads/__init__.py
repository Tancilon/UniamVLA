"""aux_heads package — auxiliary task heads (action, pose, future, recon).

Heads are constructed directly by UamVLAFramework; no registry indirection.
"""

from starVLA.model.modules.uamvla.aux_heads.spatial_map_denoising_head import SpatialMapDenoisingHead
from starVLA.model.modules.uamvla.aux_heads.depth_head import DepthDenoisingHead
from starVLA.model.modules.uamvla.aux_heads.grounding_head import GroundingMaskDenoisingHead
from starVLA.model.modules.uamvla.aux_heads.affordance_head import AffordanceHeatmapDenoisingHead
from starVLA.model.modules.uamvla.aux_heads.action_conditioned_future_head import ActionConditionedFutureHead
