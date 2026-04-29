"""Diffusion scheduler wrapping DiT for training and sampling."""

import math

import torch
import torch.nn as nn

from starVLA.model.modules.uamvla.components.denoiser.dit import DiT
from starVLA.utils.diffusion_utils import create_diffusion


class ReconDenoiser(nn.Module):
    def __init__(
        self,
        x_channel: int,
        z_channel: int,
        embed_dim: int,
        depth: int,
        learn_sigma: bool = False,
        timesteps: str = "1000",
        n_patches: int = 576,
    ):
        super().__init__()
        self.in_channels = x_channel

        self.ln_pre = nn.LayerNorm(z_channel, elementwise_affine=False)

        self.net = DiT(
            input_size=int(math.sqrt(n_patches)),
            patch_size=1,
            in_channels=x_channel,
            hidden_size=embed_dim,
            z_channel=z_channel,
            depth=depth,
            learn_sigma=learn_sigma,
        )
        self.train_diffusion = create_diffusion(
            timestep_respacing="", noise_schedule="cosine", learn_sigma=learn_sigma
        )
        self.gen_diffusion = create_diffusion(
            timestep_respacing=timesteps,
            noise_schedule="cosine",
            learn_sigma=learn_sigma,
        )

    def forward(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = torch.randint(
            self.train_diffusion.num_timesteps,
            size=(target.shape[0],),
            device=target.device,
        ).long()
        model_kwargs = dict(context=z)
        loss_dict = self.train_diffusion.training_losses(
            self.net, target, t, model_kwargs
        )
        return loss_dict["loss"]

    @torch.no_grad()
    def sample(
        self, z: torch.Tensor, temperature: float = 1.0, cfg: float = 1.0,
    ) -> torch.Tensor:
        noise = torch.randn(
            z.shape[0], self.in_channels, z.shape[-2], z.shape[-1],
            device=z.device,
        )
        model_kwargs = dict(context=z)
        sample_fn = self.net.forward

        sampled = self.gen_diffusion.p_sample_loop(
            sample_fn,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            device=z.device,
            progress=False,
            temperature=temperature,
        )
        return sampled
