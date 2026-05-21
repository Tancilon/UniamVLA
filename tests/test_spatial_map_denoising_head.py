import sys

import torch
import torch.nn as nn
from PIL import Image

from starVLA.model.modules.uamvla.aux_heads.spatial_map_denoising_head import (
    SpatialMapDenoisingHead,
)


class _FakeDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.last_z_shape = None
        self.last_target = None

    def forward(self, z, target):
        self.last_z_shape = tuple(z.shape)
        self.last_target = target.detach().clone()
        return ((target * self.weight) ** 2).mean(dim=(1, 2, 3))

    def sample(self, z):
        return torch.zeros(z.shape[0], 1, z.shape[-2], z.shape[-1], device=z.device)


def test_spatial_map_head_scales_target_and_reports_metrics():
    head = SpatialMapDenoisingHead(
        hidden_size=4,
        image_token_id=99,
        patches_per_view=4,
        target_key="depth_target",
        mask_key="depth_mask",
        metric_prefix="depth",
        loss_weight=0.5,
        denoiser=_FakeDenoiser(),
    )
    hidden = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    input_ids = torch.full((2, 4), 99, dtype=torch.long)
    batch = {
        "input_ids": input_ids,
        "depth_target": torch.tensor(
            [
                [[[0.0, 0.5], [1.0, 0.25]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        ),
    }
    mask = torch.tensor([True, False])

    out = head.compute_loss(hidden, batch, mask)

    expected_target = torch.tensor([[[[-1.0, 0.0], [1.0, -0.5]]]])
    assert torch.equal(head.denoiser.last_target, expected_target)
    assert head.denoiser.last_z_shape == (1, 4, 2, 2)
    assert out.loss is not None
    assert out.metrics["loss_raw"] > 0.0
    assert out.metrics["valid_ratio"] == 0.5


def test_spatial_map_head_empty_mask_returns_dummy_loss():
    head = SpatialMapDenoisingHead(
        hidden_size=4,
        image_token_id=99,
        patches_per_view=4,
        target_key="grounding_mask",
        mask_key="grounding_mask_mask",
        metric_prefix="grounding",
        loss_weight=0.1,
        denoiser=_FakeDenoiser(),
    )
    hidden = torch.zeros(2, 4, 4)
    batch = {
        "input_ids": torch.full((2, 4), 99, dtype=torch.long),
        "grounding_mask": torch.zeros(2, 1, 2, 2),
    }
    out = head.compute_loss(hidden, batch, torch.tensor([False, False]))
    assert out.loss is not None
    assert out.loss.item() == 0.0
    assert out.metrics["loss_raw"] == 0.0
    assert out.metrics["valid_ratio"] == 0.0


def test_spatial_map_head_visualizes_gt_vs_pred(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    head = SpatialMapDenoisingHead(
        hidden_size=4,
        image_token_id=99,
        patches_per_view=4,
        target_key="depth_target",
        mask_key="depth_mask",
        metric_prefix="depth",
        denoiser=_FakeDenoiser(),
    )
    hidden = torch.zeros(2, 4, 4)
    batch = {
        "input_ids": torch.full((2, 4), 99, dtype=torch.long),
        "depth_target": torch.rand(2, 1, 2, 2),
        "instruction": ["open", "close"],
    }

    images = head.visualize(
        hidden,
        batch,
        mask=torch.tensor([False, True]),
        num_samples=1,
    )

    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
