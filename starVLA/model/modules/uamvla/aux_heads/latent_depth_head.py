"""Latent-token depth head: regress a frozen-VAE depth latent, no denoiser.

Unlike ``DepthDenoisingHead``, there is no diffusion model between the backbone
and the loss.  The depth tokens (block 2 of the LLM sequence -- the duplicated
primary image) are projected per-token straight onto the precomputed VAE latent.

Two modules read the same hidden states, for two different purposes:

  head   pointwise MLP -> VAE latent.  Gradients reach the backbone; this is
         what shapes the representation.
  probe  linear -> raw 16x16 depth pixels, applied to ``hidden.detach()``.
         Gradients never reach the backbone; this only *measures* how linearly
         decodable depth is from ``h``.  Reported as ``probe_r2_pixel``.

"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from starVLA.model.modules.uamvla.aux_heads.base import AuxHead, HeadOutput
from starVLA.model.modules.uamvla.components.spatial_reader import slice_image_tokens


class LatentDepthHead(AuxHead):
    """Per-token regression from depth tokens onto a frozen-VAE depth latent.

    Args:
        hidden_size: Backbone hidden dimension (2560 for Qwen3-VL-4B).
        image_token_id: ``<|image_pad|>`` id -- the depth block reuses it.
        patches_per_view: Tokens per vision block (49 at qwen_image_size=224).
        view_idx: Which vision block holds the depth tokens (== num_views).
        latent_channels: Channels of the 2x2-grouped VAE latent (16 * 4 = 64).
        depth_px_size: Side of the stored ``depth_px`` map (112).
        head_kind: ``pointwise_mlp`` (default) or ``linear``.  Never anything
            with cross-token mixing.
        probe_enabled: Attach the detached linear probe.
        loss_weight: Scales the training-head loss only, never the probe.
        vae: Shared frozen VAEPixelDecoder, used by ``visualize`` only.
    """

    def __init__(
        self,
        hidden_size: int,
        image_token_id: int,
        patches_per_view: int,
        view_idx: int,
        latent_channels: int = 64,
        depth_px_size: int = 112,
        head_kind: str = "pointwise_mlp",
        probe_enabled: bool = True,
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
        if depth_px_size % self.grid:
            raise ValueError(f"depth_px_size {depth_px_size} not divisible by grid {self.grid}")
        self.patch = depth_px_size // self.grid  # 112 // 7 == 16

        self.ln_pre = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.head = self._build_head(head_kind, hidden_size, self.latent_channels)

        # Separate LN so the probe's normalisation can never affect the head's.
        self.probe = None
        if probe_enabled:
            self.ln_pre_probe = nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.probe = nn.Linear(hidden_size, self.patch * self.patch)

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
    def _depth_tokens(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        tokens = slice_image_tokens(
            hidden, input_ids, self.image_token_id, self.patches_per_view, self.view_idx,
        )  # (B, patches_per_view, H)
        # The backbone emits bf16 under autocast, but aux heads run outside any
        # autocast block, so their fp32 Linear weights would reject a bf16 input.
        # Regressing a unit-scale latent in fp32 is also the better numerics.
        return tokens.float()

    def _latent_target(self, depth_latent: torch.Tensor) -> torch.Tensor:
        return rearrange(depth_latent, "b c h w -> b (h w) c")

    def _pixel_target(self, depth_px: torch.Tensor) -> torch.Tensor:
        return rearrange(
            depth_px, "b (h ph) (w pw) -> b (h w) (ph pw)", ph=self.patch, pw=self.patch,
        )

    def compute_loss(self, hidden_states: torch.Tensor, batch: dict, mask: torch.Tensor) -> HeadOutput:
        valid_ratio = float(mask.float().mean()) if mask.numel() else 0.0
        if not mask.any():
            return HeadOutput(
                loss=self._zero_aligned_loss(hidden_states, batch),
                metrics={"r2": 0.0, "probe_r2_pixel": 0.0, "valid_ratio": 0.0},
                predictions=None,
            )

        h = self._depth_tokens(hidden_states[mask], batch["input_ids"][mask])

        # DeepSpeed bf16 safety, as in ReconHead: under bf16 the head's Linear
        # weights are bf16 while `h` is fp32, and nn.Linear refuses the mix.
        # autocast(fp32) promotes both, and fp32 is the right precision for
        # regressing a unit-scale latent anyway.
        with torch.amp.autocast("cuda", dtype=torch.float32):
            z_pred = self.head(self.ln_pre(h))
        z_gt = self._latent_target(batch["depth_latent"][mask]).to(z_pred.dtype)
        head_loss = F.mse_loss(z_pred, z_gt)  # latent is already ~unit scale

        loss = self.loss_weight * head_loss
        metrics = {
            "r2": self._r2(z_pred, z_gt),
            "var_ratio": float(z_pred.var(unbiased=False) / z_gt.var(unbiased=False).clamp_min(1e-8)),
            "cosine": float(F.cosine_similarity(z_pred.flatten(1), z_gt.flatten(1), dim=-1).mean()),
            "valid_ratio": valid_ratio,
            "head_mse": float(head_loss),
            "probe_mse": 0.0,
            "probe_r2_pixel": 0.0,
        }

        if self.probe is not None and "depth_px" in batch:
            with torch.amp.autocast("cuda", dtype=torch.float32):
                d_pred = self.probe(self.ln_pre_probe(h.detach()))
            d_gt = self._pixel_target(batch["depth_px"][mask]).to(d_pred.dtype)
            probe_loss = F.mse_loss(d_pred, d_gt)
            # Added so the probe's own parameters get a gradient; h is detached,
            # so this term contributes nothing to the backbone. It does inflate
            # the summed loss, hence head_mse / probe_mse are logged separately.
            loss = loss + probe_loss
            metrics["probe_mse"] = float(probe_loss)
            metrics["probe_r2_pixel"] = self._r2(d_pred, d_gt)

        return HeadOutput(loss=loss, metrics=metrics, predictions=None)

    def _zero_aligned_loss(self, hidden_states: torch.Tensor, batch: dict) -> torch.Tensor:
        """Run a zero-weight dummy forward so ZeRO-3 collectives stay aligned.

        Every rank must execute the same module forwards each step, even the
        ones whose batch has no depth target.
        """
        if hidden_states.shape[0] == 0:
            return self.get_dummy_loss()

        h = self._depth_tokens(hidden_states[:1], batch["input_ids"][:1])
        with torch.amp.autocast("cuda", dtype=torch.float32):
            loss = self.head(self.ln_pre(h)).sum()
            if self.probe is not None:
                loss = loss + self.probe(self.ln_pre_probe(h.detach())).sum()
        return loss * 0.0

    # ──────────────────────────────────────────────────────────────────
    #  Inference / visualization (VAE decoder, not the probe)
    # ──────────────────────────────────────────────────────────────────
    def _latent_to_depth(self, z: torch.Tensor) -> torch.Tensor:
        """(B, N, C) -> (B, R, R) grayscale depth in [0, 1] via the frozen VAE."""
        if self.vae is None:
            raise RuntimeError("LatentDepthHead.vae is None; set aux_heads.depth_latent.visualize")
        z = rearrange(z, "b (h w) c -> b c h w", h=self.grid, w=self.grid)
        z = rearrange(z, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2)
        z = z / self.vae.scaling_factor + self.vae.shift_factor
        pixels = self.vae.decode(z.to(next(self.vae.parameters()).dtype))
        return (pixels / 2 + 0.5).clamp(0, 1).mean(1)

    def predict(self, hidden_states: torch.Tensor, batch: dict) -> HeadOutput:
        h = self._depth_tokens(hidden_states, batch["input_ids"])
        with torch.amp.autocast("cuda", dtype=torch.float32):
            z_pred = self.head(self.ln_pre(h))
        return HeadOutput(loss=None, metrics={},
                          predictions={"depth_maps": self._latent_to_depth(z_pred.float())})

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
                gt = self._latent_to_depth(self._latent_target(batch["depth_latent"][idx]).float())
        finally:
            if was_training:
                self.train()

        maps = pred.predictions["depth_maps"]
        instructions = batch.get("instruction", [])
        results = []
        for i, b in enumerate(idx):
            combined = concat_images_h([_S._map_to_pil(gt[i]), _S._map_to_pil(maps[i])])
            caption = "depth_latent: GT vs Pred"
            b = int(b)
            if b < len(instructions):
                caption = f"{caption} | {instructions[b]}"
            results.append(_S._maybe_wandb_image(combined, caption=caption))
        return results
