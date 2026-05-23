"""FutureHead auxiliary branch for future-frame prediction.

Predicts the episode's final agentview frame via DiT-based diffusion
conditioned on LLM image-token hidden states (spatial 2D layout preserved,
ReconVLA-style).

Spec ref: docs/superpowers/specs/2026-04-24-recon-spatial-reader-design.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from starVLA.model.modules.uamvla.aux_heads.base import AuxHead, HeadOutput
from starVLA.model.modules.uamvla.components.denoiser.scheduler import ReconDenoiser
from starVLA.model.modules.uamvla.components.spatial_reader import slice_image_tokens


class FutureHead(AuxHead):
    """Future-frame prediction head using spatial image-token conditioning.

    Args:
        hidden_size: Backbone hidden dimension.
        vae: Shared frozen VAEPixelDecoder reference.
        image_mean: Vision tower normalization mean (list of 3 floats).
        image_std: Vision tower normalization std (list of 3 floats).
        image_token_id: Token ID of the <image> placeholder in input_ids.
        patches_per_view: Number of image tokens per view (e.g. 729).
        view_idx: Which view's tokens to use as condition (0 = agentview).
        target_resize: Square size to resize image_future to before VAE encode.
        denoiser_depth: Number of DiT layers.
        denoiser_embed_dim: DiT hidden dimension.
        n_patches: DiT spatial grid (= patches_per_view).
        loss_weight: Loss scaling factor.
        repeat_factor: Training repeat multiplier for stability.
        gen_timesteps: Timestep respacing string for inference sampling.
    """

    def __init__(
        self,
        hidden_size: int,
        vae,
        image_mean: list[float],
        image_std: list[float],
        image_token_id: int,
        patches_per_view: int,
        view_idx: int = 0,
        target_resize: int = 432,
        denoiser_depth: int = 3,
        denoiser_embed_dim: int = 1024,
        n_patches: int = 729,
        loss_weight: float = 0.1,
        repeat_factor: int = 4,
        gen_timesteps: str = "1000",
        **kwargs,
    ):
        super().__init__()
        self.vae = vae
        self.loss_weight = loss_weight
        self.repeat_factor = repeat_factor
        self.image_token_id = image_token_id
        self.patches_per_view = patches_per_view
        self.view_idx = view_idx
        self.target_resize = target_resize

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

        x_channel = vae.latent_channels * 4
        self.denoiser = ReconDenoiser(
            x_channel=x_channel,
            z_channel=hidden_size,
            embed_dim=denoiser_embed_dim,
            depth=denoiser_depth,
            n_patches=n_patches,
            timesteps=gen_timesteps,
        )

    def _spatial_condition(self, hidden_states: torch.Tensor,
                           input_ids: torch.Tensor) -> torch.Tensor:
        image_h = slice_image_tokens(
            hidden_states, input_ids,
            self.image_token_id, self.patches_per_view, self.view_idx,
        )
        image_h = self.ln_pre(image_h)
        h = w = int(self.patches_per_view ** 0.5)
        spatial = rearrange(image_h, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        return spatial

    def compute_loss(
        self, hidden_states: torch.Tensor, batch: dict, mask: torch.Tensor,
    ) -> HeadOutput:
        if not mask.any():
            return HeadOutput(
                loss=self._zero_aligned_loss(hidden_states, batch),
                metrics={"future_loss": 0.0},
                predictions=None,
            )

        hidden_valid = hidden_states[mask]
        input_ids_valid = batch["input_ids"][mask]
        spatial_cond = self._spatial_condition(hidden_valid, input_ids_valid)

        future_images = batch["image_future"][mask]
        future_images = F.interpolate(
            future_images, size=self.target_resize, mode="bilinear",
            align_corners=False,
        )
        with torch.no_grad():
            future_vae = self._normalize_for_vae(future_images)
            z_q = self._encode_to_latent(future_vae)

        with torch.amp.autocast("cuda", dtype=torch.float32):
            repeated_cond = spatial_cond.repeat(self.repeat_factor, 1, 1, 1).contiguous().float()
            repeated_target = z_q.repeat(self.repeat_factor, 1, 1, 1).contiguous().float()
            loss = self.denoiser(z=repeated_cond, target=repeated_target)

        raw_loss = loss.mean()
        weighted_loss = self.loss_weight * raw_loss

        return HeadOutput(
            loss=weighted_loss,
            metrics={"future_loss": raw_loss.detach().item()},
            predictions=None,
        )

    def _zero_aligned_loss(self, hidden_states: torch.Tensor, batch: dict) -> torch.Tensor:
        """Run a zero-weight dummy path so ZeRO-3 collectives stay aligned."""
        if hidden_states.shape[0] == 0:
            return self.get_dummy_loss()

        spatial_cond = self._spatial_condition(hidden_states[:1], batch["input_ids"][:1])
        dummy_images = torch.zeros(
            1,
            3,
            self.target_resize,
            self.target_resize,
            device=spatial_cond.device,
            dtype=spatial_cond.dtype,
        )
        with torch.no_grad():
            z_q = self._encode_to_latent(self._normalize_for_vae(dummy_images))

        loss = self.denoiser(
            z=spatial_cond.contiguous().float(),
            target=z_q.contiguous().float(),
        )
        return loss.mean() * 0.0

    def predict(
        self, hidden_states: torch.Tensor, batch: dict,
    ) -> HeadOutput:
        spatial_cond = self._spatial_condition(hidden_states, batch["input_ids"])
        sampled_latent = self.denoiser.sample(z=spatial_cond)
        pixels = self._latent_to_pixels(sampled_latent)
        return HeadOutput(
            loss=None,
            metrics={},
            predictions={"future_images": pixels},
        )

    def visualize(
        self,
        hidden_states: torch.Tensor,
        batch: dict,
        mask: torch.Tensor,
        num_samples: int = 1,
        **kwargs,
    ) -> list:
        """Side-by-side: Input (agentview) | GT (image_future) | Pred (resized back)."""
        if not mask.any():
            return []

        import wandb
        from starVLA.utils.vis_draw import tensor_to_pil, concat_images_h

        n = min(num_samples, mask.sum().item())
        valid_indices = mask.nonzero(as_tuple=True)[0][:n]

        h_subset = hidden_states[valid_indices]
        ids_subset = batch["input_ids"][valid_indices]
        batch_subset = {
            "input_ids": ids_subset,
            "image_future": batch["image_future"][valid_indices],
        }
        with torch.no_grad():
            output = self.predict(h_subset, batch_subset)
        pred_images = output.predictions["future_images"]

        gt_hw = batch["image_future"].shape[-1]
        pred_downscaled = F.interpolate(
            pred_images, size=gt_hw, mode="bilinear", align_corners=False,
        )

        results = []
        for i, idx in enumerate(valid_indices):
            idx_int = idx.item()
            input_img = tensor_to_pil(batch["image"][idx_int, 0])
            gt_img = tensor_to_pil(batch["image_future"][idx_int])
            pred_img = tensor_to_pil(pred_downscaled[i], mean=0.0, std=1.0)
            combined = concat_images_h([input_img, gt_img, pred_img])
            caption = batch["instruction"][idx_int] if "instruction" in batch else ""
            results.append(wandb.Image(combined, caption=caption))
        return results

    def _normalize_for_vae(self, images: torch.Tensor) -> torch.Tensor:
        image_std = self.image_std.to(device=images.device, dtype=images.dtype)
        image_mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        images_vae = (
            (images * image_std + image_mean - 0.5) / 0.5
        ).clamp(-1.0, 1.0)
        return images_vae

    def _encode_to_latent(self, images_vae: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Cast to VAE's parameter dtype before conv2d to avoid the
            # "Input type (float) and bias type (BFloat16) should be the same"
            # mismatch when training under bf16. Mirrors _latent_to_pixels.
            vae_dtype = next(self.vae.parameters()).dtype
            images_vae = images_vae.to(vae_dtype)
            posterior = self.vae.encode(images_vae).latent_dist
            z_q = (
                posterior.sample() - self.vae.shift_factor
            ) * self.vae.scaling_factor
            z_q = z_q.unfold(2, 2, 2).unfold(3, 2, 2)
            z_q = rearrange(z_q, "b c h w p1 p2 -> b (c p1 p2) h w").contiguous()
        return z_q

    def _latent_to_pixels(self, sampled: torch.Tensor) -> torch.Tensor:
        sampled = rearrange(
            sampled, "b (c p1 p2) h w -> b c h w p1 p2", p1=2, p2=2
        )
        sampled = rearrange(sampled, "b c h w p1 p2 -> b c (h p1) (w p2)")
        sampled = sampled / self.vae.scaling_factor + self.vae.shift_factor
        vae_dtype = next(self.vae.parameters()).dtype
        sampled = sampled.to(vae_dtype)
        pixels = self.vae.decode(sampled)
        pixels = (pixels / 2 + 0.5).clamp(0, 1)
        return pixels
