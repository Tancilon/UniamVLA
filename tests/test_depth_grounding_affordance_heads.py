import torch

from starVLA.model.modules.uamvla.aux_heads.affordance_head import AffordanceHeatmapDenoisingHead
from starVLA.model.modules.uamvla.aux_heads.depth_head import DepthDenoisingHead
from starVLA.model.modules.uamvla.aux_heads.grounding_head import GroundingMaskDenoisingHead


def test_depth_relative_inverse_depth_per_frame():
    depth = torch.tensor(
        [[[[1.0, 2.0], [4.0, 0.0]]]],
        dtype=torch.float32,
    )
    out = DepthDenoisingHead.relative_inverse_depth_per_frame(depth)
    assert out.shape == (1, 1, 2, 2)
    assert out[0, 0, 0, 0] > out[0, 0, 0, 1]
    assert out[0, 0, 0, 1] > out[0, 0, 1, 0]
    assert out[0, 0, 1, 1] == 0.0
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_wrapper_target_keys_are_stable():
    common = dict(hidden_size=4, image_token_id=99, patches_per_view=4, denoiser=None)
    depth = DepthDenoisingHead(**common)
    grounding = GroundingMaskDenoisingHead(**common)
    affordance = AffordanceHeatmapDenoisingHead(**common)
    assert depth.target_key == "depth_target"
    assert depth.mask_key == "depth_mask"
    assert grounding.target_key == "grounding_mask"
    assert grounding.mask_key == "grounding_mask_mask"
    assert affordance.target_key == "affordance_heatmap"
    assert affordance.mask_key == "affordance_mask"
