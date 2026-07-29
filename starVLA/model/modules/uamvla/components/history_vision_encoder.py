"""HistoryVisionEncoder: encode K-1 historical frames via Qwen's visual tower + PerceiverResampler.

Design rationale:
  - Running the full Qwen3-VL LLM on K=10 frames is prohibitively expensive.  Instead we
    reuse Qwen's already-loaded visual tower (patch_embed + transformer blocks + merger) and
    skip the LLM entirely.  This mirrors Seer's choice of a frozen lightweight ViT for history
    but avoids introducing a second set of large vision weights.
  - PerceiverResampler compresses each (view, frame) from ~49-64 tokens down to ``num_latents``
    tokens, keeping the KV budget in FutureCrossAttnBranch manageable.
  - The output is in the LLM hidden-state dimension (D_llm) so it concatenates directly with
    the Qwen hidden states that serve as KV for the future cross-attention branch.

Expected dataloader contract:
  ``example["image_history"]`` — list of length K-1, each entry is a list of PIL images
  (one per camera view, same order as ``example["image"]``).  If the key is absent or the
  list is empty the encoder returns an empty tensor and is a no-op.
"""
from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image

from starVLA.model.modules.uamvla.components.perceiver_resampler import PerceiverResampler

logger = logging.getLogger(__name__)


class HistoryVisionEncoder(nn.Module):
    """Encode historical frames into compressed token sequences.

    Args:
        qwen_vl_interface: The ``_QWen3_VL_Interface`` instance already built by the framework.
            We borrow its ``processor.image_processor`` and ``model.model.get_image_features``.
        d_llm: LLM hidden dimension.  Visual-tower output is already projected to this size.
        num_latents: Tokens per (view, frame) after PerceiverResampler compression.
        resampler_depth: Number of Perceiver cross-attention layers.
        freeze_visual: If True, freeze the Qwen visual tower weights used here (recommended).
        max_history_frames: Maximum K-1 value.
        num_views: Number of camera views per frame.  The PerceiverResampler
            receives T = (K-1) × num_views slots, so its position embeddings
            must be sized accordingly.
    """

    def __init__(
        self,
        qwen_vl_interface,
        d_llm: int,
        num_latents: int = 10,
        resampler_depth: int = 3,
        freeze_visual: bool = True,
        max_history_frames: int = 16,
        num_views: int = 1,
    ) -> None:
        super().__init__()

        self._image_processor = qwen_vl_interface.processor.image_processor
        # Borrow the visual tower — a Qwen3VLVisionModel (patch_embed + blocks + merger)
        self._visual = qwen_vl_interface.model.model.visual
        self._get_image_features = qwen_vl_interface.model.model.get_image_features
        self._spatial_merge_size = self._visual.spatial_merge_size

        if freeze_visual:
            for p in self._visual.parameters():
                p.requires_grad_(False)

        # T fed to resampler = (K-1) × num_views; size the position embeddings accordingly.
        self.resampler = PerceiverResampler(
            dim=d_llm,
            depth=resampler_depth,
            num_latents=num_latents,
            max_num_media=max_history_frames * num_views,
        )

    # ------------------------------------------------------------------
    #  PIL → pixel_values / image_grid_thw helper
    # ------------------------------------------------------------------

    def _preprocess_images(self, images: List[Image.Image], device: torch.device):
        """Run Qwen's image processor on a flat list of PIL images.

        Returns:
            pixel_values: FloatTensor on ``device``.
            image_grid_thw: LongTensor (num_images, 3) on ``device``.
        """
        proc_out = self._image_processor(images=images, return_tensors="pt")
        pixel_values = proc_out["pixel_values"].to(device)
        image_grid_thw = proc_out["image_grid_thw"].to(device)
        return pixel_values, image_grid_thw

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        image_history: List[List[List[Image.Image]]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Encode K-1 historical frames into compressed tokens.

        Args:
            image_history: Nested list — [batch][frame_t-k][view].
                Shape convention: (B, K-1, num_views).
            device / dtype: Target device and dtype for the output.

        Returns:
            Tensor of shape (B, (K-1)*num_views*num_latents, d_llm), or
            None if ``image_history`` is empty / None.
        """
        if not image_history or not image_history[0]:
            return None

        B = len(image_history)
        K_minus_1 = len(image_history[0])
        num_views = len(image_history[0][0])

        # Flatten all (batch, frame, view) images into one list for the image processor.
        flat_images: List[Image.Image] = []
        for b in range(B):
            for k in range(K_minus_1):
                for v in range(num_views):
                    flat_images.append(image_history[b][k][v])

        with torch.no_grad() if not self.resampler.training else torch.enable_grad():
            pixel_values, image_grid_thw = self._preprocess_images(flat_images, device)
            pixel_values = pixel_values.to(dtype)

            # get_image_features already splits by image and returns a tuple of
            # per-image tensors (Qwen3-VL behaviour).  The second return value is
            # deepstack_image_embeds which we don't use here.
            per_image_embeds, _ = self._get_image_features(pixel_values, image_grid_thw)

        # per_image_embeds is already a tuple[Tensor] — one (n_tok, d_llm) per image.
        tokens_per_image = [t.shape[0] for t in per_image_embeds]

        # Verify all images produced the same token count (required for batching).
        n_tok = tokens_per_image[0]
        if any(t != n_tok for t in tokens_per_image):
            raise RuntimeError(
                f"HistoryVisionEncoder: unequal token counts across images {tokens_per_image}. "
                "Ensure all historical frames are the same size."
            )

        # Stack → (total_images, n_tok, d_llm), then reshape for resampler.
        stacked = torch.stack(list(per_image_embeds), dim=0)     # (total_images, n_tok, d_llm)
        stacked = stacked.view(B, K_minus_1 * num_views, n_tok, -1)

        # PerceiverResampler: (B, T, n_tok, d_llm) → (B, T, num_latents, d_llm)
        compressed = self.resampler(stacked)                      # (B, T, num_latents, d_llm)

        # Flatten temporal dimension → (B, (K-1)*num_views*num_latents, d_llm)
        compressed = rearrange(compressed, "b t n d -> b (t n) d")
        return compressed.to(dtype)
