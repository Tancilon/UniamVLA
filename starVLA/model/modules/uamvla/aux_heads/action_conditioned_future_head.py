from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from starVLA.model.modules.uamvla.aux_heads.base import AuxHead, HeadOutput
from starVLA.model.modules.uamvla.components.action_chunk_encoder import ActionChunkEncoder
from starVLA.model.modules.uamvla.components.denoiser.scheduler import ReconDenoiser
from starVLA.model.modules.uamvla.components.spatial_reader import slice_image_tokens


class ActionConditionedFutureHead(AuxHead):
    """Local future-frame denoising conditioned on visual tokens and actions."""

    def __init__(
        self,
        hidden_size: int,
        vae,
        image_mean: list[float],
        image_std: list[float],
        image_token_id: int,
        patches_per_view: int,
        action_dim: int,
        action_horizon: int,
        view_idx: int = 0,
        target_resize: int = 320,
        denoiser_depth: int = 3,
        denoiser_embed_dim: int = 1024,
        n_patches: int | None = None,
        loss_weight: float = 1.0,
        repeat_factor: int = 1,
        gen_timesteps: str = "1000",
        action_encoder_layers: int = 2,
        action_embed_dim: int = 512,
        action_encoder_heads: int = 8,
        action_dropout: float = 0.1,
        denoiser: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.loss_weight = float(loss_weight)
        self.repeat_factor = int(repeat_factor)
        self.image_token_id = int(image_token_id)
        self.patches_per_view = int(patches_per_view)
        self.view_idx = int(view_idx)
        self.target_resize = int(target_resize)
        self.action_horizon = int(action_horizon)
        self.action_dropout = float(action_dropout)

        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std).view(1, -1, 1, 1),
            persistent=False,
        )

        self.ln_pre = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.action_encoder = ActionChunkEncoder(
            action_dim=action_dim,
            hidden_size=hidden_size,
            action_embed_dim=action_embed_dim,
            num_layers=action_encoder_layers,
            num_heads=action_encoder_heads,
            max_horizon=action_horizon,
        )
        self.film = nn.Linear(hidden_size, 2 * hidden_size)

        x_channel = vae.latent_channels * 4
        self.denoiser = denoiser or ReconDenoiser(
            x_channel=x_channel,
            z_channel=hidden_size,
            embed_dim=denoiser_embed_dim,
            depth=denoiser_depth,
            n_patches=n_patches or patches_per_view,
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
        h = w = int(self.patches_per_view ** 0.5)
        spatial = rearrange(image_h, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        return spatial

    def _condition_with_action(
        self,
        spatial_cond: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        action_cond = self.action_encoder(action_chunk[:, -self.action_horizon :, :])
        action_cond = action_cond.to(device=spatial_cond.device, dtype=spatial_cond.dtype)
        if self.training and self.action_dropout > 0.0:
            keep = torch.rand(
                action_cond.shape[0],
                1,
                device=action_cond.device,
                dtype=action_cond.dtype,
            ) >= self.action_dropout
            action_cond = action_cond * keep
        scale, shift = self.film(action_cond).chunk(2, dim=-1)
        return spatial_cond * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

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
        fused_cond = self._condition_with_action(spatial_cond, batch["action"][mask])

        future_images = batch["image_action_future"][mask]
        future_images = F.interpolate(
            future_images,
            size=self.target_resize,
            mode="bilinear",
            align_corners=False,
        )
        with torch.no_grad():
            future_vae = self._normalize_for_vae(future_images)
            z_q = self._encode_to_latent(future_vae)

        repeated_cond = fused_cond.repeat(self.repeat_factor, 1, 1, 1).contiguous().float()
        repeated_target = z_q.repeat(self.repeat_factor, 1, 1, 1).contiguous().float()
        loss = self.denoiser(z=repeated_cond, target=repeated_target)
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
        fused_cond = self._condition_with_action(spatial_cond, batch["action"])
        sampled_latent = self.denoiser.sample(z=fused_cond)
        pixels = self._latent_to_pixels(sampled_latent)
        return HeadOutput(
            loss=None,
            metrics={},
            predictions={"action_conditioned_future_images": pixels},
        )

    def _normalize_for_vae(self, images: torch.Tensor) -> torch.Tensor:
        image_std = self.image_std.to(device=images.device, dtype=images.dtype)
        image_mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        if images.detach().amin() >= 0.0 and images.detach().amax() <= 1.0:
            return ((images - 0.5) / 0.5).clamp(-1.0, 1.0)
        return ((images * image_std + image_mean - 0.5) / 0.5).clamp(-1.0, 1.0)

    def _vae_dtype(self) -> torch.dtype:
        param = next(self.vae.parameters(), None)
        return param.dtype if param is not None else torch.float32

    def _encode_to_latent(self, images_vae: torch.Tensor) -> torch.Tensor:
        vae_dtype = self._vae_dtype()
        images_vae = images_vae.to(vae_dtype)
        posterior = self.vae.encode(images_vae).latent_dist
        z_q = (posterior.sample() - self.vae.shift_factor) * self.vae.scaling_factor
        z_q = z_q.unfold(2, 2, 2).unfold(3, 2, 2)
        return rearrange(z_q, "b c h w p1 p2 -> b (c p1 p2) h w").contiguous()

    def _latent_to_pixels(self, sampled: torch.Tensor) -> torch.Tensor:
        sampled = rearrange(
            sampled,
            "b (c p1 p2) h w -> b c h w p1 p2",
            p1=2,
            p2=2,
        )
        sampled = rearrange(sampled, "b c h w p1 p2 -> b c (h p1) (w p2)")
        sampled = sampled / self.vae.scaling_factor + self.vae.shift_factor
        sampled = sampled.to(self._vae_dtype())
        pixels = self.vae.decode(sampled)
        pixels = (pixels / 2 + 0.5).clamp(0, 1)
        return pixels

    @staticmethod
    def _tensor_image_to_pil(image: torch.Tensor):
        from starVLA.utils.vis_draw import tensor_to_pil

        image = image.detach().float()
        if image.amin().item() < 0.0:
            return tensor_to_pil(image, mean=0.5, std=0.5)
        return tensor_to_pil(image, mean=0.0, std=1.0)

    @staticmethod
    def _maybe_wandb_image(image, caption: str):
        try:
            import wandb
        except Exception:
            return image
        return wandb.Image(image, caption=caption)

    def visualize(
        self,
        hidden_states: torch.Tensor,
        batch: dict,
        mask: torch.Tensor,
        num_samples: int = 1,
        **kwargs,
    ) -> list:
        """Side-by-side GT and predicted local future frames for valid samples."""
        mask = mask.to(device=hidden_states.device, dtype=torch.bool)
        if num_samples <= 0 or not mask.any():
            return []

        from starVLA.utils.vis_draw import concat_images_h

        n = min(int(num_samples), int(mask.sum().item()))
        valid_indices = mask.nonzero(as_tuple=True)[0][:n]
        batch_subset = {
            "input_ids": batch["input_ids"][valid_indices],
            "action": batch["action"][valid_indices],
        }

        was_training = bool(self.training)
        self.eval()
        try:
            with torch.no_grad():
                output = self.predict(hidden_states[valid_indices], batch_subset)
        finally:
            if was_training:
                self.train()

        pred_images = output.predictions["action_conditioned_future_images"].detach()
        gt_images = batch["image_action_future"][valid_indices]
        if pred_images.shape[-2:] != gt_images.shape[-2:]:
            pred_images = F.interpolate(
                pred_images,
                size=gt_images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        instructions = batch.get("instruction", [])

        results = []
        for i, idx in enumerate(valid_indices):
            gt_img = self._tensor_image_to_pil(gt_images[i])
            pred_img = self._tensor_image_to_pil(pred_images[i])
            combined = concat_images_h([gt_img, pred_img])
            idx_int = int(idx.item())
            instruction = instructions[idx_int] if idx_int < len(instructions) else ""
            caption = "action_conditioned_future: GT vs Pred"
            if instruction:
                caption = f"{caption} | {instruction}"
            results.append(self._maybe_wandb_image(combined, caption=caption))
        return results
