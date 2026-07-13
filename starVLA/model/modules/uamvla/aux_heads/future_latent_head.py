"""Future-latent head: regress the frozen-VAE latent of the t+N frame, no denoiser.

Seer-style foresight (arXiv:2412.15109) adapted to the LT convention.  Seer
appends learnable foresight tokens per timestep, decodes them into the frame
``future_steps`` ahead, and lets the action token attend to them -- "predict
the future, then infer the action that reaches it".  Here the foresight tokens
are the last duplicated-primary vision block of the LLM sequence (so they get
a real 2D M-RoPE grid), and instead of Seer's MAE decoder each token is
projected straight onto the precomputed VAE latent of the frame
``future_offset`` steps ahead.  The GR00T action head cross-attends the same
hidden states, so the action is conditioned on this predicted-future
representation without any extra wiring.

Same family as ``LatentDepthHead`` -- pointwise projection, no denoiser, no
probe (there is no pixel-space sidecar for RGB; the VAE round-trip is already
verified at preprocess time by --report_psnr).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from starVLA.model.modules.uamvla.aux_heads.base import AuxHead, HeadOutput
from starVLA.model.modules.uamvla.components.spatial_reader import slice_image_tokens


class FutureLatentHead(AuxHead):
    """Per-token regression from foresight tokens onto the t+N frame's VAE latent.

    Args:
        hidden_size: Backbone hidden dimension (2560 for Qwen3-VL-4B).
        image_token_id: ``<|image_pad|>`` id -- the foresight block reuses it.
        patches_per_view: Tokens per vision block (49 at qwen_image_size=224).
        view_idx: Which vision block holds the foresight tokens (== num_views
            + enabled depth block + enabled affordance block, see
            UamGR00T_LT._maybe_build_aux_heads).
        latent_channels: Channels of the 2x2-grouped VAE latent (16 * 4 = 64).
        head_kind: ``pointwise_mlp`` (default) or ``linear``.  Never anything
            with cross-token mixing.
        loss_weight: Scales the head loss.
        vae: Shared frozen VAEPixelDecoder, used by ``visualize`` only.
    """

    def __init__(
        self,
        hidden_size: int,
        image_token_id: int,
        patches_per_view: int,
        view_idx: int,
        latent_channels: int = 64,
        head_kind: str = "pointwise_mlp",
        loss_weight: float = 1.0,
        vae=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.image_token_id = int(image_token_id)
        self.patches_per_view = int(patches_per_view)
        self.view_idx = int(view_idx)
        self.latent_channels = int(latent_channels)
        self.loss_weight = float(loss_weight)
        self.vae = vae  # shared reference, not owned

        self.grid = int(math.sqrt(self.patches_per_view))
        if self.grid * self.grid != self.patches_per_view:
            raise ValueError(f"patches_per_view must be a square grid, got {patches_per_view}")

        self.ln_pre = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.head = self._build_head(head_kind, hidden_size, self.latent_channels)

    @staticmethod
    def _build_head(kind: str, hidden_size: int, out_dim: int) -> nn.Module:
        if kind == "linear":
            return nn.Linear(hidden_size, out_dim)
        if kind == "pointwise_mlp":
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, out_dim),
            )
        raise ValueError(f"head_kind must be 'linear' or 'pointwise_mlp', got {kind!r}")

    # ──────────────────────────────────────────────────────────────────
    #  Metrics
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
        """1 - mse / var(target).  0.0 means "no better than the batch mean"."""
        var = target.var(unbiased=False)
        if var <= 0:
            return 0.0
        return float(1.0 - F.mse_loss(pred, target) / var)

    # ──────────────────────────────────────────────────────────────────
    #  Forward paths
    # ──────────────────────────────────────────────────────────────────
    def _foresight_tokens(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        tokens = slice_image_tokens(
            hidden, input_ids, self.image_token_id, self.patches_per_view, self.view_idx,
        )  # (B, patches_per_view, H)
        # fp32 for the same reason as LatentDepthHead: aux heads run outside
        # autocast, so their fp32 Linear weights would reject a bf16 input.
        return tokens.float()

    def _latent_target(self, future_latent: torch.Tensor) -> torch.Tensor:
        return rearrange(future_latent, "b c h w -> b (h w) c")

    def compute_loss(self, hidden_states: torch.Tensor, batch: dict, mask: torch.Tensor) -> HeadOutput:
        valid_ratio = float(mask.float().mean()) if mask.numel() else 0.0
        if not mask.any():
            return HeadOutput(
                loss=self._zero_aligned_loss(hidden_states, batch),
                metrics={"r2": 0.0, "valid_ratio": 0.0},
                predictions=None,
            )

        h = self._foresight_tokens(hidden_states[mask], batch["input_ids"][mask])
        with torch.amp.autocast("cuda", dtype=torch.float32):
            z_pred = self.head(self.ln_pre(h))
        z_gt = self._latent_target(batch["future_latent"][mask]).to(z_pred.dtype)
        head_loss = F.mse_loss(z_pred, z_gt)  # latent is already ~unit scale

        loss = self.loss_weight * head_loss
        metrics = {
            "r2": self._r2(z_pred, z_gt),
            "var_ratio": float(z_pred.var(unbiased=False) / z_gt.var(unbiased=False).clamp_min(1e-8)),
            "cosine": float(F.cosine_similarity(z_pred.flatten(1), z_gt.flatten(1), dim=-1).mean()),
            "valid_ratio": valid_ratio,
            "head_mse": float(head_loss),
        }
        return HeadOutput(loss=loss, metrics=metrics, predictions=None)

    def _zero_aligned_loss(self, hidden_states: torch.Tensor, batch: dict) -> torch.Tensor:
        """Run a zero-weight dummy forward so ZeRO-3 collectives stay aligned.

        Every rank must execute the same module forwards each step, even the
        ones whose batch has no future target.
        """
        if hidden_states.shape[0] == 0:
            return self.get_dummy_loss()

        h = self._foresight_tokens(hidden_states[:1], batch["input_ids"][:1])
        with torch.amp.autocast("cuda", dtype=torch.float32):
            loss = self.head(self.ln_pre(h)).sum()
        return loss * 0.0

    # ──────────────────────────────────────────────────────────────────
    #  Inference / visualization (VAE decoder)
    # ──────────────────────────────────────────────────────────────────
    def _latent_to_rgb(self, z: torch.Tensor) -> torch.Tensor:
        """(B, N, C) -> (B, 3, R, R) RGB in [0, 1] via the frozen VAE."""
        if self.vae is None:
            raise RuntimeError("FutureLatentHead.vae is None; set aux_heads.future_latent.visualize")
        z = rearrange(z, "b (h w) c -> b c h w", h=self.grid, w=self.grid)
        z = rearrange(z, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2)
        z = z / self.vae.scaling_factor + self.vae.shift_factor
        pixels = self.vae.decode(z.to(next(self.vae.parameters()).dtype))
        return (pixels / 2 + 0.5).clamp(0, 1)

    @staticmethod
    def _rgb_to_pil(rgb: torch.Tensor):
        """(3, R, R) in [0, 1] -> PIL RGB image."""
        import numpy as np
        from PIL import Image

        arr = (rgb.detach().float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def predict(self, hidden_states: torch.Tensor, batch: dict) -> HeadOutput:
        h = self._foresight_tokens(hidden_states, batch["input_ids"])
        with torch.amp.autocast("cuda", dtype=torch.float32):
            z_pred = self.head(self.ln_pre(h))
        return HeadOutput(loss=None, metrics={},
                          predictions={"future_frames": self._latent_to_rgb(z_pred.float())})

    def visualize(self, hidden_states, batch, mask, num_samples: int = 1, **kwargs) -> list:
        """GT | prediction, both decoded through the frozen VAE."""
        mask = mask.to(device=hidden_states.device, dtype=torch.bool)
        if num_samples <= 0 or not mask.any() or self.vae is None:
            return []

        from starVLA.utils.vis_draw import concat_images_h
        from starVLA.model.modules.uamvla.aux_heads.spatial_map_denoising_head import (
            SpatialMapDenoisingHead as _S,
        )

        n = min(int(num_samples), int(mask.sum()))
        idx = mask.nonzero(as_tuple=True)[0][:n]
        was_training = bool(self.training)
        self.eval()
        try:
            with torch.no_grad():
                pred = self.predict(hidden_states[idx], {"input_ids": batch["input_ids"][idx]})
                gt = self._latent_to_rgb(self._latent_target(batch["future_latent"][idx]).float())
        finally:
            if was_training:
                self.train()

        frames = pred.predictions["future_frames"]
        instructions = batch.get("instruction", [])
        results = []
        for i, b in enumerate(idx):
            combined = concat_images_h([self._rgb_to_pil(gt[i]), self._rgb_to_pil(frames[i])])
            caption = "future_latent: GT vs Pred"
            b = int(b)
            if b < len(instructions):
                caption = f"{caption} | {instructions[b]}"
            results.append(_S._maybe_wandb_image(combined, caption=caption))
        return results
