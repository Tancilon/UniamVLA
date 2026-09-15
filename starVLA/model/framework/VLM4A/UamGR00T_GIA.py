"""UamGR00T_GIA: Geometry–Interaction–Action three-dimensional auxiliary framework.

Three auxiliary supervision tasks, each with a **dedicated emoji token block**
injected in the LLM text-suffix (ReconVLA §3.2 style):

  Geometry      depth       → LatentDepthHead     Video-Depth-Anything V2 VAE latent regression
  Interaction   affordance  → AffordanceTokenHead  contact-point heatmap regression
  Temporal      future      → FutureLatentHead     Seer-style future VAE latent regression

Sequence layout (N=2 cameras, patches_per_view=49 at obs_image_size=224):

    [instruction] → [primary×49] → [wrist×49] → [✨×49] → [🟩×49] → [🟧×49]
                                                  ^depth    ^afford   ^future

Token order follows a **causal dependency chain** (Strategy B):
  depth(✨) → affordance(🟩) → future(🟧)
  • affordance tokens attend to depth: geometry-constrained interaction detection
  • future tokens attend to depth + affordance: geometry+interaction informed prediction

All three token blocks are stripped before the GR00T DiT action model, so the
action head sees only [instruction + image] hidden states.

Compared with UamGR00T_hR_LT:
  • No recon head (keeps the sequence shorter)
  • FutureLatentHead instead of ActionConditionedFutureHead: simpler Seer-style
    per-token regression, no action conditioning encoder
  • Compatible with flash_attention_2 (standard causal mask, no 4-D block mask)

YAML keys (under framework.aux_heads):
    depth       — LatentDepthHead fields (latent_channels, head_kind,
                  probe_enabled, loss_weight, camera, visualize)
    affordance  — AffordanceTokenHead fields (head_kind, loss_weight, camera)
    future      — FutureLatentHead fields (latent_channels, head_kind,
                  loss_weight, future_offset, visualize)

Sidecar prerequisites (per dataset scene):
    python runners/preprocess_depth_vda.py       --dataset_root <scene_root>
    python runners/preprocess_depth_latent.py    --dataset_root <scene_root>
    python runners/preprocess_affordance_vrb.py  --dataset_root <scene_root>
    python runners/preprocess_affordance_px.py   --dataset_root <scene_root>
    python runners/preprocess_rgb_latent.py      --dataset_root <scene_root>

Registered framework name: UamGR00T_GIA
"""
from __future__ import annotations

import json
import logging
from typing import List

import torch

from starVLA.model.framework.VLM4A.UamGR00T_hR_multi import _MultiHeadBase
from starVLA.model.modules.uamvla.aux_heads.affordance_token_head import AffordanceTokenHead
from starVLA.model.modules.uamvla.aux_heads.future_latent_head import FutureLatentHead
from starVLA.model.modules.uamvla.aux_heads.latent_depth_head import LatentDepthHead
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)

# Keys consumed at the framework level that must not be forwarded to head constructors.
_GIA_NON_HEAD_KEYS = frozenset({
    "enabled",
    "lr",
    "camera",       # which camera view to validate sidecar against
    "visualize",    # attach frozen VAE for visualization (off under ZeRO-3)
    "latent_root",  # depth/future: sidecar subdirectory override
    "px_root",      # depth/affordance: pixel-sidecar subdirectory override
    "depth_px_size",   # derived from layout; filtered to avoid double-passing
    "afford_px_size",  # same
    "future_offset",   # consumed by the dataloader, not the head constructor
})


@FRAMEWORK_REGISTRY.register("UamGR00T_GIA")
class UamVLAGR00T_GIA(_MultiHeadBase):
    """Geometry–Interaction–Action framework (three dedicated per-head token blocks).

    Token hierarchy (Strategy B, standard causal mask):
        depth(✨) → affordance(🟩) → future(🟧)

    Later heads attend to earlier heads through the causal mask:
      - affordance(🟩) sees depth(✨): geometry-grounded interaction detection
      - future(🟧) sees depth + affordance: full scene + interaction context

    Compatible with flash_attention_2 (no 4-D block mask needed).
    """

    # ──────────────────────────────────────────────────────────────────
    #  Token order (causal dependency chain: geometry→interaction→temporal)
    # ──────────────────────────────────────────────────────────────────
    def _head_order(self) -> list[str]:
        return ["depth", "affordance", "future"]

    # ──────────────────────────────────────────────────────────────────
    #  Sidecar validation helpers
    # ──────────────────────────────────────────────────────────────────
    def _num_views_from_config(self) -> int:
        obs = self.config.datasets.vla_data.get("obs", [])
        views = [k for k in obs if str(k).startswith("video.")]
        if not views:
            raise RuntimeError("datasets.vla_data.obs contains no video.* keys")
        return len(views)

    def _assert_depth_latent_meta(self, layout: dict, latent_channels: int) -> None:
        """Fail loudly when depth_latent sidecar disagrees with this config."""
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
                f"{meta_path} disagrees with config (found, expected): {mismatched}. "
                f"Re-run preprocess_depth_latent.py --qwen_image_size {self._qwen_image_size()}."
            )
        vae_path = str(self.config.framework.vae.path)
        if meta.get("vae_path") != vae_path:
            logger.warning(
                "depth_latent encoded with vae_path=%r but framework.vae.path=%r; "
                "visualization would use a different VAE than the targets.",
                meta.get("vae_path"), vae_path,
            )

    def _assert_affordance_px_meta(self, layout: dict, camera: str) -> None:
        """Fail loudly when affordance_px sidecar disagrees with this config."""
        meta_path = self.sidecar_root / "affordance_px" / "meta.json"
        if not meta_path.exists():
            raise RuntimeError(
                f"{meta_path} not found. Run:\n"
                f"  python runners/preprocess_affordance_px.py "
                f"--dataset_root {self.sidecar_root}"
            )
        meta = json.loads(meta_path.read_text())
        expected = {
            "grid": layout["target_size"],
            "target_resize": layout["target_resize"],
            "camera": camera,
        }
        mismatched = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
        if mismatched:
            raise RuntimeError(
                f"{meta_path} disagrees with config (found, expected): {mismatched}. "
                f"Re-run preprocess_affordance_px.py --qwen_image_size {self._qwen_image_size()}."
            )

    def _assert_future_latent_meta(self, layout: dict, latent_channels: int) -> None:
        """Fail loudly when rgb_latent sidecar disagrees with this config."""
        meta_path = self.sidecar_root / "rgb_latent" / "meta.json"
        if not meta_path.exists():
            raise RuntimeError(
                f"{meta_path} not found. Run:\n"
                f"  python runners/preprocess_rgb_latent.py "
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
                f"{meta_path} disagrees with config (found, expected): {mismatched}. "
                f"Re-run preprocess_rgb_latent.py --qwen_image_size {self._qwen_image_size()}."
            )
        vae_path = str(self.config.framework.vae.path)
        if meta.get("vae_path") != vae_path:
            logger.warning(
                "rgb_latent encoded with vae_path=%r but framework.vae.path=%r; "
                "visualization would use a different VAE than the targets.",
                meta.get("vae_path"), vae_path,
            )

    # ──────────────────────────────────────────────────────────────────
    #  Aux head construction
    # ──────────────────────────────────────────────────────────────────
    def _maybe_build_aux_heads(self) -> None:  # noqa: C901
        """Build three paper-aligned aux heads for the GIA dimensions.

        Registration key → cfg_key mapping (via _MultiHeadBase._CFG_TO_HEAD_KEY):
            "depth"      ← aux_heads.depth      → LatentDepthHead
            "affordance" ← aux_heads.affordance  → AffordanceTokenHead
            "future"     ← aux_heads.future      → FutureLatentHead

        All heads use view_idx=0: each emoji token appears exactly once in the
        text-suffix, so slice_image_tokens always selects the correct block.
        """
        cfg_heads = self.config.framework.aux_heads
        hidden_size = self.qwen_vl_interface.model.config.hidden_size
        layout = self._qwen_vision_layout()
        image_token_id = getattr(
            self.qwen_vl_interface, "image_token_id",
            self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        )

        # Shared frozen VAE (depth/future visualization, built lazily).
        vae = None

        def _get_or_build_vae():
            nonlocal vae
            if vae is None:
                from starVLA.model.modules.uamvla.components.pixel_decoder.vae import VAEPixelDecoder
                vae = VAEPixelDecoder(self.config.framework.vae.path)
                self.vae = vae
            return vae

        # ── 1. LatentDepthHead (geometry) ─────────────────────────────
        if cfg_heads.get("depth", {}).get("enabled", False):
            depth_cfg = cfg_heads.get("depth", {})
            latent_channels = int(depth_cfg.get("latent_channels", 64))
            self._assert_depth_latent_meta(layout, latent_channels)

            depth_vae = _get_or_build_vae() if depth_cfg.get("visualize", False) else None
            head_kwargs = {k: v for k, v in depth_cfg.items() if k not in _GIA_NON_HEAD_KEYS}
            self.aux_heads["depth"] = LatentDepthHead(
                hidden_size=hidden_size,
                image_token_id=image_token_id,
                patches_per_view=layout["patches_per_view"],
                view_idx=0,
                depth_px_size=layout["target_resize"],
                vae=depth_vae,
                **head_kwargs,
            )

        # ── 2. AffordanceTokenHead (interaction) ──────────────────────
        if cfg_heads.get("affordance", {}).get("enabled", False):
            afford_cfg = cfg_heads.get("affordance", {})
            afford_camera = str(afford_cfg.get("camera", "image"))
            self._assert_affordance_px_meta(layout, afford_camera)

            head_kwargs = {k: v for k, v in afford_cfg.items() if k not in _GIA_NON_HEAD_KEYS}
            self.aux_heads["affordance"] = AffordanceTokenHead(
                hidden_size=hidden_size,
                image_token_id=image_token_id,
                patches_per_view=layout["patches_per_view"],
                view_idx=0,
                afford_px_size=layout["target_resize"],
                **head_kwargs,
            )

        # ── 3. FutureLatentHead (temporal) ────────────────────────────
        if cfg_heads.get("future", {}).get("enabled", False):
            future_cfg = cfg_heads.get("future", {})
            latent_channels = int(future_cfg.get("latent_channels", 64))
            self._assert_future_latent_meta(layout, latent_channels)

            future_vae = _get_or_build_vae() if future_cfg.get("visualize", False) else None
            head_kwargs = {k: v for k, v in future_cfg.items() if k not in _GIA_NON_HEAD_KEYS}
            self.aux_heads["future"] = FutureLatentHead(
                hidden_size=hidden_size,
                image_token_id=image_token_id,
                patches_per_view=layout["patches_per_view"],
                view_idx=0,
                vae=future_vae,
                **head_kwargs,
            )


# ──────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml", type=str,
        default="./examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml",
    )
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)

    model = UamVLAGR00T_GIA(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ppv = model._qwen_vision_layout()["patches_per_view"]
    num_views = model._num_views_from_config()
    grid = model._qwen_vision_layout()["target_size"]

    def fake_rgb(size: int = 224):
        return Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))

    sample = {
        "action": torch.rand(16, 7) * 2 - 1,
        "image": [fake_rgb() for _ in range(num_views)],
        "lang": "pick up the blue block",
        "state": np.zeros((1, 7), dtype=np.float32),
        # Sidecar targets (zeros are valid shapes for the forward pass)
        "depth_latent": torch.randn(64, grid, grid),
        "depth_px":     torch.rand(grid * 16, grid * 16),
        "affordance_px": torch.rand(grid * 16, grid * 16),
        "future_latent": torch.randn(64, grid, grid),
    }
    batch = [sample, {**sample, "lang": "open the drawer"}]

    _, qwen_inputs, hidden = model._encode_qwen_hidden(model._prepare_examples(list(batch)))
    print(f"[ok] sequence length: {hidden.shape[1]}")
    print(f"[ok] hidden: {tuple(hidden.shape)}")
    print(f"[ok] enabled head tokens: {list(model._head_token_str.items())}")

    out = model(batch)
    print({k: float(v) for k, v in out.items() if torch.is_tensor(v) and v.numel() == 1})
    print(f"[ok] action_loss: {out['action_loss'].item():.4f}")

    pred = model.predict_action(examples=[sample])
    print(f"[ok] predict_action -> {pred['normalized_actions'].shape}")
    print("Finished UamGR00T_GIA smoke test.")
