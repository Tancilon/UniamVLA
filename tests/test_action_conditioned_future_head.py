import sys

import pytest
import torch
import torch.nn as nn
from PIL import Image

from starVLA.model.modules.uamvla.aux_heads.action_conditioned_future_head import (
    ActionConditionedFutureHead,
)
from starVLA.model.modules.uamvla.components.action_chunk_encoder import ActionChunkEncoder


def test_action_chunk_encoder_outputs_hidden_size():
    encoder = ActionChunkEncoder(
        action_dim=7,
        hidden_size=16,
        action_embed_dim=8,
        num_layers=1,
        num_heads=2,
        max_horizon=8,
    )
    actions = torch.randn(3, 8, 7)
    out = encoder(actions)
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()


def test_action_chunk_encoder_matches_parameter_dtype():
    encoder = ActionChunkEncoder(
        action_dim=7,
        hidden_size=16,
        action_embed_dim=8,
        num_layers=1,
        num_heads=2,
        max_horizon=8,
    ).to(torch.bfloat16)
    actions = torch.randn(3, 8, 7, dtype=torch.float32)
    out = encoder(actions)
    assert out.shape == (3, 16)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_action_chunk_encoder_uses_learnable_sincos_position_init():
    encoder = ActionChunkEncoder(
        action_dim=7,
        hidden_size=16,
        action_embed_dim=6,
        num_layers=1,
        num_heads=2,
        max_horizon=4,
    )

    position = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, 6, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / 6)
    )
    expected = torch.zeros(1, 4, 6)
    expected[0, :, 0::2] = torch.sin(position * div_term)
    expected[0, :, 1::2] = torch.cos(position * div_term)

    assert isinstance(encoder.pos_embed, nn.Parameter)
    assert encoder.pos_embed.requires_grad
    assert torch.allclose(encoder.pos_embed.detach(), expected)


class _FakeVAE(nn.Module):
    latent_channels = 2
    scaling_factor = 1.0
    shift_factor = 0.0

    def encode(self, images):
        class _Posterior:
            def __init__(self, x):
                self.x = x

            def sample(self):
                return self.x[:, :2, ::8, ::8]

        return type("Encoded", (), {"latent_dist": _Posterior(images)})()

    def decode(self, latents):
        return torch.zeros(latents.shape[0], 3, latents.shape[-2] * 8, latents.shape[-1] * 8)


class _FakeDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.last_z = None

    def forward(self, z, target):
        self.last_z = z.detach().clone()
        return (target * self.weight).pow(2).mean(dim=(1, 2, 3))

    def sample(self, z):
        return torch.zeros(z.shape[0], 8, z.shape[-2], z.shape[-1], device=z.device)


def test_action_conditioned_future_uses_action_film_condition():
    head = ActionConditionedFutureHead(
        hidden_size=4,
        vae=_FakeVAE(),
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        image_token_id=99,
        patches_per_view=4,
        action_dim=7,
        action_horizon=8,
        target_resize=32,
        action_encoder_layers=1,
        action_embed_dim=8,
        action_encoder_heads=2,
        action_dropout=0.0,
        denoiser=_FakeDenoiser(),
    )
    hidden = torch.randn(2, 4, 4)
    batch = {
        "input_ids": torch.full((2, 4), 99, dtype=torch.long),
        "action": torch.randn(2, 8, 7),
        "image_action_future": torch.rand(2, 3, 32, 32),
    }
    mask = torch.tensor([True, True])
    out = head.compute_loss(hidden, batch, mask)
    assert out.loss is not None
    assert out.metrics["valid_ratio"] == 1.0
    assert head.denoiser.last_z.shape == (2, 4, 2, 2)


def test_action_conditioned_future_rejects_grid_mismatch():
    head = ActionConditionedFutureHead(
        hidden_size=4,
        vae=_FakeVAE(),
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        image_token_id=99,
        patches_per_view=4,
        action_dim=7,
        action_horizon=8,
        target_resize=16,
        action_encoder_layers=1,
        action_embed_dim=8,
        action_encoder_heads=2,
        action_dropout=0.0,
        denoiser=_FakeDenoiser(),
    )
    hidden = torch.randn(1, 4, 4)
    batch = {
        "input_ids": torch.full((1, 4), 99, dtype=torch.long),
        "action": torch.randn(1, 8, 7),
        "image_action_future": torch.rand(1, 3, 32, 32),
    }
    with pytest.raises(RuntimeError, match="condition/target grid mismatch"):
        head.compute_loss(hidden, batch, torch.tensor([True]))


def test_action_conditioned_future_empty_mask_returns_dummy_loss():
    denoiser = _FakeDenoiser()
    head = ActionConditionedFutureHead(
        hidden_size=4,
        vae=_FakeVAE(),
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        image_token_id=99,
        patches_per_view=4,
        action_dim=7,
        action_horizon=8,
        target_resize=32,
        action_encoder_layers=1,
        action_embed_dim=8,
        action_encoder_heads=2,
        denoiser=denoiser,
    )
    out = head.compute_loss(
        torch.zeros(2, 4, 4),
        {
            "input_ids": torch.full((2, 4), 99, dtype=torch.long),
            "action": torch.zeros(2, 8, 7),
            "image_action_future": torch.zeros(2, 3, 32, 32),
        },
        torch.tensor([False, False]),
    )
    assert out.loss is not None
    assert out.loss.item() == 0.0
    assert out.metrics["loss_raw"] == 0.0
    assert out.metrics["valid_ratio"] == 0.0
    assert denoiser.last_z is not None
    assert denoiser.last_z.shape == (1, 4, 2, 2)


def test_action_conditioned_future_visualizes_gt_vs_pred(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    head = ActionConditionedFutureHead(
        hidden_size=4,
        vae=_FakeVAE(),
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        image_token_id=99,
        patches_per_view=4,
        action_dim=7,
        action_horizon=8,
        target_resize=32,
        action_encoder_layers=1,
        action_embed_dim=8,
        action_encoder_heads=2,
        action_dropout=0.0,
        denoiser=_FakeDenoiser(),
    )
    hidden = torch.zeros(2, 4, 4)
    batch = {
        "input_ids": torch.full((2, 4), 99, dtype=torch.long),
        "action": torch.zeros(2, 8, 7),
        "image_action_future": torch.rand(2, 3, 32, 32),
        "instruction": ["open", "close"],
    }

    images = head.visualize(
        hidden,
        batch,
        mask=torch.tensor([True, False]),
        num_samples=1,
    )

    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
