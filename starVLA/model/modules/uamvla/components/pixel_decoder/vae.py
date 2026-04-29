"""Frozen VAE pixel decoder wrapping diffusers AutoencoderKL."""

import torch.nn as nn
from diffusers import AutoencoderKL


class VAEPixelDecoder(nn.Module):
    """Frozen AutoencoderKL wrapper for encoding images to latent space and decoding back.

    Args:
        model_path: Path or HuggingFace model ID for the pretrained VAE.
    """

    def __init__(self, model_path: str, **kwargs) -> None:
        super().__init__()
        self.pixel_decoder = AutoencoderKL.from_pretrained(model_path)
        self.pixel_decoder.requires_grad_(False)
        self.pixel_decoder.float()
        self.pixel_decoder.eval()

    @property
    def scaling_factor(self) -> float:
        return self.pixel_decoder.config.scaling_factor

    @property
    def shift_factor(self) -> float:
        return self.pixel_decoder.config.shift_factor

    @property
    def latent_channels(self) -> int:
        return self.pixel_decoder.config.latent_channels

    def encode(self, x):
        """Encode images to latent distribution.

        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            Encoder output with latent_dist for sampling.
        """
        return self.pixel_decoder.encode(x)

    def decode(self, z):
        """Decode latent to pixel images.

        Args:
            z: Latent tensor.

        Returns:
            Decoded image tensor of shape (B, C, H, W).
        """
        return self.pixel_decoder.decode(z).sample
