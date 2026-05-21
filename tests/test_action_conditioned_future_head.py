import torch
import torch.nn as nn

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


def test_action_conditioned_future_empty_mask_returns_dummy_loss():
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
        denoiser=_FakeDenoiser(),
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
