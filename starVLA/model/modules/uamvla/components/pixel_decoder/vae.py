"""Frozen VAE pixel decoder wrapping diffusers AutoencoderKL."""

import re
from pathlib import Path

import torch.nn as nn
from diffusers import AutoencoderKL


class VAEPixelDecoder(nn.Module):
    """Frozen AutoencoderKL wrapper for encoding images to latent space and decoding back.

    Args:
        model_path: Path or HuggingFace model ID for the pretrained VAE.
    """

    def __init__(self, model_path: str, **kwargs) -> None:
        super().__init__()
        self.pixel_decoder = self._load_autoencoder(model_path)
        self.pixel_decoder.requires_grad_(False)
        self.pixel_decoder.float()
        self.pixel_decoder.eval()

    @classmethod
    def _load_autoencoder(cls, model_path: str):
        model_dir = Path(model_path)
        weights = model_dir / "diffusion_pytorch_model.safetensors"
        if weights.exists() and cls._looks_like_legacy_diffusers_vae(weights):
            return cls._load_legacy_diffusers_vae(model_dir, weights)
        return AutoencoderKL.from_pretrained(model_path, low_cpu_mem_usage=False)

    @staticmethod
    def _looks_like_legacy_diffusers_vae(weights: Path) -> bool:
        from safetensors.torch import safe_open

        with safe_open(str(weights), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        return "encoder.down.0.block.0.conv1.weight" in keys

    @classmethod
    def _load_legacy_diffusers_vae(cls, model_dir: Path, weights: Path):
        from safetensors.torch import load_file

        vae = AutoencoderKL.from_config(str(model_dir / "config.json"))
        state_dict = {
            cls._convert_legacy_key(key): value
            for key, value in load_file(str(weights)).items()
        }
        for key, value in list(state_dict.items()):
            if ".attentions.0." in key and value.ndim == 4 and value.shape[-2:] == (1, 1):
                state_dict[key] = value.squeeze(-1).squeeze(-1)
        vae.load_state_dict(state_dict, strict=True)
        return vae

    @staticmethod
    def _convert_legacy_key(key: str) -> str:
        key = key.replace("encoder.down.", "encoder.down_blocks.")
        key = re.sub(
            r"decoder\.up\.(\d+)\.",
            lambda match: f"decoder.up_blocks.{3 - int(match.group(1))}.",
            key,
        )
        key = re.sub(
            r"down_blocks\.(\d+)\.block\.(\d+)\.",
            r"down_blocks.\1.resnets.\2.",
            key,
        )
        key = re.sub(
            r"up_blocks\.(\d+)\.block\.(\d+)\.",
            r"up_blocks.\1.resnets.\2.",
            key,
        )
        key = key.replace("downsample.conv", "downsamplers.0.conv")
        key = key.replace("upsample.conv", "upsamplers.0.conv")
        key = key.replace("nin_shortcut", "conv_shortcut")
        key = key.replace("encoder.mid.block_1.", "encoder.mid_block.resnets.0.")
        key = key.replace("encoder.mid.block_2.", "encoder.mid_block.resnets.1.")
        key = key.replace("decoder.mid.block_1.", "decoder.mid_block.resnets.0.")
        key = key.replace("decoder.mid.block_2.", "decoder.mid_block.resnets.1.")
        key = key.replace("encoder.mid.attn_1.norm.", "encoder.mid_block.attentions.0.group_norm.")
        key = key.replace("decoder.mid.attn_1.norm.", "decoder.mid_block.attentions.0.group_norm.")
        for old, new in (("q", "to_q"), ("k", "to_k"), ("v", "to_v")):
            key = key.replace(f"encoder.mid.attn_1.{old}.", f"encoder.mid_block.attentions.0.{new}.")
            key = key.replace(f"decoder.mid.attn_1.{old}.", f"decoder.mid_block.attentions.0.{new}.")
        key = key.replace("encoder.mid.attn_1.proj_out.", "encoder.mid_block.attentions.0.to_out.0.")
        key = key.replace("decoder.mid.attn_1.proj_out.", "decoder.mid_block.attentions.0.to_out.0.")
        key = key.replace("encoder.norm_out.", "encoder.conv_norm_out.")
        key = key.replace("decoder.norm_out.", "decoder.conv_norm_out.")
        return key

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
