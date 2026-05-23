import torch
import torch.nn as nn

from starVLA.model.modules.uamvla.aux_heads.future_head import FutureHead
from starVLA.model.modules.uamvla.aux_heads.recon_head import ReconHead


class _FakeVAE(nn.Module):
    latent_channels = 2
    scaling_factor = 1.0
    shift_factor = 0.0

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))

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
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.last_z_shape = None
        self.last_target_shape = None

    def forward(self, z, target):
        self.last_z_shape = tuple(z.shape)
        self.last_target_shape = tuple(target.shape)
        return (target * self.weight).pow(2).mean(dim=(1, 2, 3))

    def sample(self, z):
        return torch.zeros(z.shape[0], 8, z.shape[-2], z.shape[-1], device=z.device)


def _head(head_cls):
    head = head_cls.__new__(head_cls)
    nn.Module.__init__(head)
    head.vae = _FakeVAE()
    head.loss_weight = 0.5
    head.repeat_factor = 4
    head.image_token_id = 99
    head.patches_per_view = 4
    head.view_idx = 0
    head.target_resize = 32
    head.register_buffer(
        "image_mean",
        torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1),
        persistent=False,
    )
    head.register_buffer(
        "image_std",
        torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1),
        persistent=False,
    )
    head.ln_pre = nn.LayerNorm(4, elementwise_affine=False)
    head.denoiser = _FakeDenoiser()
    return head


def test_recon_empty_mask_runs_zero_aligned_denoiser_forward():
    head = _head(ReconHead)
    out = head.compute_loss(
        torch.zeros(2, 4, 4),
        {
            "input_ids": torch.full((2, 4), 99, dtype=torch.long),
            "image_target": torch.zeros(2, 3, 32, 32),
        },
        torch.tensor([False, False]),
    )

    assert out.loss is not None
    assert out.loss.item() == 0.0
    assert out.metrics["recon_loss"] == 0.0
    assert head.denoiser.last_z_shape == (1, 4, 2, 2)
    assert head.denoiser.last_target_shape == (1, 8, 2, 2)


def test_future_empty_mask_runs_zero_aligned_denoiser_forward():
    head = _head(FutureHead)
    out = head.compute_loss(
        torch.zeros(2, 4, 4),
        {
            "input_ids": torch.full((2, 4), 99, dtype=torch.long),
            "image_future": torch.zeros(2, 3, 32, 32),
        },
        torch.tensor([False, False]),
    )

    assert out.loss is not None
    assert out.loss.item() == 0.0
    assert out.metrics["future_loss"] == 0.0
    assert head.denoiser.last_z_shape == (1, 4, 2, 2)
    assert head.denoiser.last_target_shape == (1, 8, 2, 2)
