"""UamGR00T_LT: latent-token depth supervision, no diffusion denoiser.

Sequence layout (qwen_image_size=224, 49 tokens per block):

    [instruction] -> [primary x49] -> [wrist x49] -> [primary x49]
                                                      ^ the depth block

The depth block is the primary image passed a *second* time.  That is not a
trick to save code -- it is the only way the depth tokens get a real 2D M-RoPE
grid.  
"""
from __future__ import annotations

import json
import logging
from typing import List

import torch

from starVLA.model.framework.VLM4A.UamGR00T import UamVLAGR00T
from starVLA.model.modules.uamvla.aux_heads.latent_depth_head import LatentDepthHead
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)

# Consumed by _maybe_build_aux_heads; everything else is forwarded to the head.
_NON_HEAD_KEYS = frozenset({"enabled", "lr", "camera", "visualize", "latent_root", "px_root"})


@FRAMEWORK_REGISTRY.register("UamGR00T_LT")
class UamVLAGR00T_LT(UamVLAGR00T):
    """UamVLAGR00T with a duplicated-primary depth block and a latent depth head.

    ``UamVLAGR00T.forward`` already feeds the full ``hidden`` to both the action
    model and the aux heads, so forward / predict_action need no override.
    """

    def _num_views_from_config(self) -> int:
        obs = self.config.datasets.vla_data.get("obs", [])
        views = [key for key in obs if str(key).startswith("video.")]
        if not views:
            raise RuntimeError("datasets.vla_data.obs contains no video.* keys")
        return len(views)

    def _assert_depth_latent_meta(self, layout: dict, latent_channels: int) -> None:
        """Fail loudly when the precomputed artifacts disagree with this config.

        A silent mismatch here (wrong grid, wrong VAE) trains the head against
        targets that do not correspond to the tokens it reads.
        """
        meta_path = self.sidecar_root / "depth_latent" / "meta.json"
        if not meta_path.exists():
            raise RuntimeError(
                f"{meta_path} not found. Run:\n"
                f"  python runners/preprocess_depth_latent.py "
                f"--dataset_root {self.sidecar_root}"
            )
        meta = json.loads(meta_path.read_text())
        expected = {
            "grid": layout["target_size"],
            "target_resize": layout["target_resize"],
            "latent_channels": latent_channels,
        }
        mismatched = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
        if mismatched:
            raise RuntimeError(
                f"{meta_path} disagrees with the current config "
                f"(found, expected): {mismatched}. Re-run preprocess_depth_latent.py "
                f"with --qwen_image_size {self._qwen_image_size()}."
            )
        vae_path = str(self.config.framework.vae.path)
        if meta.get("vae_path") != vae_path:
            logger.warning(
                "depth_latent was encoded with vae_path=%r but framework.vae.path=%r; "
                "visualization would decode with a different VAE than the targets.",
                meta.get("vae_path"), vae_path,
            )

    # ──────────────────────────────────────────────────────────────────
    #  Aux heads: only the latent depth head, never a denoiser
    # ──────────────────────────────────────────────────────────────────
    def _maybe_build_aux_heads(self) -> None:
        cfg = self.config.framework.aux_heads.get("depth_latent", {})
        self._depth_block_enabled = bool(cfg.get("enabled", False))
        if not self._depth_block_enabled:
            return

        layout = self._qwen_vision_layout()
        latent_channels = int(cfg.get("latent_channels", 64))
        self._assert_depth_latent_meta(layout, latent_channels)

        vae = None
        if cfg.get("visualize", False):
            # ZeRO-3 caveat: this VAE is only touched inside visualize(), so its
            # params are never gathered during a training forward.  Safe under
            # ZeRO-2 (no partitioning); leave off under ZeRO-3.
            from starVLA.model.modules.uamvla.components.pixel_decoder.vae import VAEPixelDecoder

            self.vae = VAEPixelDecoder(self.config.framework.vae.path)
            vae = self.vae

        head_kwargs = {k: v for k, v in cfg.items() if k not in _NON_HEAD_KEYS}
        head_kwargs.pop("latent_channels", None)

        self.aux_heads["depth_latent"] = LatentDepthHead(
            hidden_size=self.qwen_vl_interface.model.config.hidden_size,
            image_token_id=getattr(
                self.qwen_vl_interface, "image_token_id",
                self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            ),
            patches_per_view=layout["patches_per_view"],
            # The depth block is appended last, so its index == number of real
            # views.  Never hard-code 2: a third camera would silently make the
            # head regress the wrong block's hidden states.
            view_idx=self._num_views_from_config(),
            latent_channels=latent_channels,
            depth_px_size=layout["target_resize"],
            vae=vae,
            **head_kwargs,
        )

    # ──────────────────────────────────────────────────────────────────
    #  Encode: append the primary image as a third vision block
    # ──────────────────────────────────────────────────────────────────
    def _encode_qwen_hidden(self, examples: List[dict]):
        batch_images = [self._force_resize_640(example["image"]) for example in examples]
        if self._depth_block_enabled:
            expected_views = self._num_views_from_config()
            if len(batch_images[0]) != expected_views:
                raise RuntimeError(
                    f"depth head was built for view_idx={expected_views} but this batch "
                    f"has {len(batch_images[0])} views; the head would read the wrong block."
                )
            batch_images = [images + [images[0]] for images in batch_images]

        examples = [
            {**example, "image": images}
            for example, images in zip(examples, batch_images)
        ]
        instructions = [example["lang"] for example in examples]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = qwenvl_outputs.hidden_states[-1]
        # ``examples`` now carry 3 images, so the expected count auto-adapts to 147.
        self._assert_image_token_count(qwen_inputs["input_ids"], examples)
        return examples, qwen_inputs, hidden


if __name__ == "__main__":
    import argparse

    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml", type=str,
        default="./examples/calvin/train_files/run_uamgr00t_LT_depth_train.yaml",
    )
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)

    model = UamVLAGR00T_LT(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    grid = model._qwen_vision_layout()["target_size"]
    ppv = model._qwen_vision_layout()["patches_per_view"]
    num_views = model._num_views_from_config()

    def fake_image():
        return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    sample = {
        # torch tensors, like the real dataloader: _collate_aux reads
        # template.dtype straight into torch.zeros.
        "action": torch.rand(16, 7) * 2 - 1,
        "image": [fake_image() for _ in range(num_views)],
        "lang": "pick up the blue block",
        # (1, state_dim): _state_batch_or_none stacks these and repeats along
        # dim 0, so a bare (state_dim,) would silently transpose the batch.
        "state": np.zeros((1, 7), dtype=np.float32),
        "depth_latent": torch.randn(64, grid, grid),
        "depth_px": torch.rand(grid * 16, grid * 16),
    }
    batch = [sample, {**sample, "lang": "open the drawer"}]

    image_token_id = model.aux_heads["depth_latent"].image_token_id
    _, qwen_inputs, hidden = model._encode_qwen_hidden(model._prepare_examples(list(batch)))
    counts = (qwen_inputs["input_ids"] == image_token_id).sum(dim=1)
    expected = ppv * (num_views + 1)
    assert (counts == expected).all(), f"expected {expected} image_pad, got {counts.tolist()}"
    print(f"[ok] image_pad tokens per sample: {counts.tolist()} == {ppv} x {num_views + 1}")
    print(f"[ok] hidden {tuple(hidden.shape)} -> action model sees the depth block (not stripped)")

    out = model(batch)
    print({k: (float(v) if torch.is_tensor(v) and v.numel() == 1 else v)
           for k, v in out.items() if not torch.is_tensor(v) or v.numel() == 1})
    print(f"[ok] action_loss {out['action_loss'].item():.4f}")

    pred = model.predict_action(examples=[sample])
    print(f"[ok] predict_action -> {pred['normalized_actions'].shape}")
    print("Finished")
