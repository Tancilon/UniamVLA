from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from starVLA.model.modules.uamvla.aux_heads.base import AuxHead, HeadOutput
from starVLA.model.modules.uamvla.components.denoiser.scheduler import ReconDenoiser
from starVLA.model.modules.uamvla.components.spatial_reader import slice_image_tokens


class SpatialMapDenoisingHead(AuxHead):
    """Shared DDPM denoising head for 20x20 single-channel spatial maps."""

    def __init__(
        self,
        hidden_size: int,
        image_token_id: int,
        patches_per_view: int,
        target_key: str,
        mask_key: str,
        metric_prefix: str,
        view_idx: int = 0,
        target_size: int | None = None,
        loss_weight: float = 0.1,
        denoiser_depth: int = 3,
        denoiser_embed_dim: int = 512,
        gen_timesteps: str = "1000",
        denoiser: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.image_token_id = int(image_token_id)
        self.patches_per_view = int(patches_per_view)
        self.view_idx = int(view_idx)
        self.target_key = target_key
        self.mask_key = mask_key
        self.metric_prefix = metric_prefix
        self.grid_size = int(math.sqrt(self.patches_per_view))
        if self.grid_size * self.grid_size != self.patches_per_view:
            raise ValueError(
                f"patches_per_view must be a square grid, got {self.patches_per_view}"
            )
        self.target_size = int(target_size or self.grid_size)
        self.ln_pre = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.denoiser = denoiser or ReconDenoiser(
            x_channel=1,
            z_channel=hidden_size,
            embed_dim=denoiser_embed_dim,
            depth=denoiser_depth,
            n_patches=self.target_size * self.target_size,
            timesteps=gen_timesteps,
        )

    def _spatial_condition(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        image_h = slice_image_tokens(
            hidden_states,
            input_ids,
            self.image_token_id,
            self.patches_per_view,
            self.view_idx,
        )
        image_h = self.ln_pre(image_h)
        spatial = rearrange(
            image_h,
            "b (h w) c -> b c h w",
            h=self.grid_size,
            w=self.grid_size,
        ).contiguous()
        if spatial.shape[-1] != self.target_size:
            spatial = F.interpolate(
                spatial,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        return spatial

    def _prepare_target_01(self, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.ndim != 4:
            raise ValueError(
                f"{self.target_key} must have shape [B, 1, H, W], got {tuple(target.shape)}"
            )
        target = target.float()
        if target.shape[1] != 1:
            raise ValueError(
                f"{self.target_key} must be single-channel, got {target.shape[1]} channels"
            )
        if target.shape[-2:] != (self.target_size, self.target_size):
            target = F.interpolate(
                target,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        return target.clamp(0.0, 1.0)

    @staticmethod
    def _target_01_to_diffusion(target_01: torch.Tensor) -> torch.Tensor:
        return target_01 * 2.0 - 1.0

    @staticmethod
    def _diffusion_to_target_01(target: torch.Tensor) -> torch.Tensor:
        return ((target + 1.0) / 2.0).clamp(0.0, 1.0)

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        batch: dict,
        mask: torch.Tensor,
    ) -> HeadOutput:
        valid_ratio = float(mask.float().mean().item()) if mask.numel() else 0.0
        if not mask.any():
            return HeadOutput(
                loss=self.get_dummy_loss(),
                metrics={"loss_raw": 0.0, "valid_ratio": 0.0},
                predictions=None,
            )

        hidden_valid = hidden_states[mask]
        input_ids_valid = batch["input_ids"][mask]
        spatial_cond = self._spatial_condition(hidden_valid, input_ids_valid)

        target_valid = batch[self.target_key][mask]
        target_01 = self._prepare_target_01(target_valid)
        target = self._target_01_to_diffusion(target_01).to(
            device=spatial_cond.device,
            dtype=spatial_cond.dtype,
        )

        loss = self.denoiser(z=spatial_cond.float(), target=target.float())
        raw_loss = loss.mean()
        weighted_loss = self.loss_weight * raw_loss
        return HeadOutput(
            loss=weighted_loss,
            metrics={"loss_raw": raw_loss.detach().item(), "valid_ratio": valid_ratio},
            predictions=None,
        )

    def predict(
        self,
        hidden_states: torch.Tensor,
        batch: dict,
    ) -> HeadOutput:
        spatial_cond = self._spatial_condition(hidden_states, batch["input_ids"])
        sampled = self.denoiser.sample(z=spatial_cond)
        maps = self._diffusion_to_target_01(sampled)
        return HeadOutput(
            loss=None,
            metrics={},
            predictions={f"{self.metric_prefix}_maps": maps},
        )
