"""UamGR00T_hR_LT: _MultiHeadBase with paper-aligned per-token aux heads.

Four auxiliary supervision tasks, each with a **dedicated emoji token block**
injected in the LLM text-suffix (ReconVLA §3.2 style):

  recon        → ReconHead                        visual reconstruction
  depth        → LatentDepthHead                  Video-Depth-Anything V2 VAE latent regression
  affordance   → AffordanceTokenHead              Splat-MOVER contact-point heatmap regression
  acf          → ActionConditionedFutureHead      SEER action-conditioned future prediction

Sequence layout (example: N=2 cameras, patches_per_view=49 at obs_image_size=224):

    [instruction] → [primary×49] → [wrist×49] →
    [🎯×49] [✨×49] [🟩×49] [🔶×49]
    ^recon   ^depth  ^afford  ^acf

All four token blocks are stripped before the GR00T DiT action model.

Design differences from UamGR00T_LT
────────────────────────────────────
• Token source: text-suffix emoji placeholders, NOT duplicated vision blocks.
  _MultiHeadBase._register_all_head_tokens() patches each head's image_token_id
  to its own emoji, so slice_image_tokens always uses view_idx=0 (there is only
  one "block" of each emoji in the sequence — no positional ambiguity).

• All four heads read tokens that have full causal attention over [instruction +
  image] context.  Later heads (depth → affordance → acf) can additionally attend
  to earlier heads in the suffix (standard causal mask), forming a light semantic
  hierarchy without an explicit block mask.

• Depth and affordance heads require precomputed sidecar artefacts:
    <sidecar_root>/depth_latent/   (from runners/preprocess_depth_latent.py)
    <sidecar_root>/affordance_px/  (from runners/preprocess_affordance_px.py)
  The framework validates meta.json at init so a stale sidecar fails loudly
  instead of silently training at zero coverage.

YAML keys (under framework.aux_heads):
    recon        — same fields as UamVLAGR00T / UamGR00T_hR
    depth        — LatentDepthHead fields (latent_channels, depth_px_size,
                   head_kind, probe_enabled, loss_weight, visualize)
    affordance   — AffordanceTokenHead fields (afford_px_size, head_kind,
                   loss_weight, camera)
    action_conditioned_future — ActionConditionedFutureHead fields
                   (action_encoder_layers, action_embed_dim, film_hidden_dim,
                   denoiser_depth, denoiser_embed_dim, loss_weight, repeat_factor,
                   gen_timesteps, action_dropout, visualize)

Registered framework name:  UamGR00T_hR_LT
"""
from __future__ import annotations

import json
import logging
from typing import List

import torch

from starVLA.model.framework.VLM4A.UamGR00T_hR_multi import _MultiHeadBase
from starVLA.model.modules.uamvla.aux_heads.affordance_token_head import AffordanceTokenHead
from starVLA.model.modules.uamvla.aux_heads.latent_depth_head import LatentDepthHead
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)

# Keys consumed at the framework level; not forwarded to head constructors.
# "enabled" / "lr" are always stripped by _aux_head_cfg_without_layout_keys;
# the ones here are specific to the LT-style config surface.
_LT_NON_HEAD_KEYS = frozenset({
    "enabled",
    "lr",
    "camera",           # affordance: which camera's heatmap to validate against
    "visualize",        # depth / acf: attach frozen VAE for visualization
    "latent_root",      # depth: sidecar subdirectory override
    "px_root",          # depth / affordance: pixel-sidecar subdirectory override
    # Spatial dims derived from layout — explicitly passed by _maybe_build_aux_heads;
    # filtering prevents TypeError from double-passing if they appear in YAML.
    "depth_px_size",
    "afford_px_size",
})


@FRAMEWORK_REGISTRY.register("UamGR00T_hR_LT")
class UamVLAGR00T_hR_LT(_MultiHeadBase):
    """UamGR00T with dedicated per-head tokens + paper-aligned aux-head implementations.

    Token order (semantic hierarchy):
        recon → depth → affordance → action_conditioned_future

    Each head attends to the full image context; later heads can additionally
    attend to earlier heads via the standard causal mask.  All head tokens are
    stripped before the GR00T flow-matching action model.
    """

    # ──────────────────────────────────────────────────────────────────────
    #  Token order (cfg_keys consumed by _MultiHeadBase._build_all_head_suffix)
    # ──────────────────────────────────────────────────────────────────────
    def _head_order(self) -> list[str]:
        return ["recon", "depth", "affordance", "action_conditioned_future"]

    # ──────────────────────────────────────────────────────────────────────
    #  Sidecar validation helpers (adapted from UamGR00T_LT)
    # ──────────────────────────────────────────────────────────────────────
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
                f"{meta_path} disagrees with the current config "
                f"(found, expected): {mismatched}. Re-run preprocess_affordance_px.py "
                f"with --qwen_image_size {self._qwen_image_size()}."
            )

    # ──────────────────────────────────────────────────────────────────────
    #  Aux head construction
    # ──────────────────────────────────────────────────────────────────────
    def _maybe_build_aux_heads(self) -> None:  # noqa: C901
        """Build the four paper-aligned aux heads.

        Heads are registered under the keys that _MultiHeadBase._CFG_TO_HEAD_KEY
        maps to, so token-id patching in _register_all_head_tokens() finds them
        automatically:

            cfg_key                   aux_heads key                head class
            ─────────────────────────────────────────────────────────────────
            recon               →   "recon"               ReconHead
            depth               →   "depth"               LatentDepthHead
            affordance          →   "affordance"          AffordanceTokenHead
            action_conditioned_future →
                                    "action_conditioned_future"  ActionConditionedFutureHead

        view_idx=0 for LatentDepthHead and AffordanceTokenHead:
            Unlike UamGR00T_LT (which duplicates a full vision block), this
            framework uses dedicated emoji tokens in the text suffix.  Each emoji
            appears exactly once in the sequence with patches_per_view tokens, so
            slice_image_tokens(view_idx=0) always selects the right block.
        """
        cfg_heads = self.config.framework.aux_heads
        hidden_size = self.qwen_vl_interface.model.config.hidden_size
        layout = self._qwen_vision_layout()
        image_token_id = getattr(
            self.qwen_vl_interface, "image_token_id",
            self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        )

        # ── Shared VAE (recon / acf / optionally depth visualisation) ────────
        # Constructed lazily: only when at least one head that needs it is enabled.
        vae = None

        def _get_or_build_vae():
            nonlocal vae
            if vae is None:
                from starVLA.model.modules.uamvla.components.pixel_decoder.vae import VAEPixelDecoder
                vae = VAEPixelDecoder(self.config.framework.vae.path)
                self.vae = vae
            return vae

        # ── Vision extras shared by recon / future heads ──────────────────────
        vision_extra = {
            "image_mean": [0.5, 0.5, 0.5],
            "image_std":  [0.5, 0.5, 0.5],
            "image_token_id": image_token_id,
            "patches_per_view": layout["patches_per_view"],
            "n_patches": layout["patches_per_view"],
            "target_resize": layout["target_resize"],
        }

        # ── 1. ReconHead ──────────────────────────────────────────────────────
        if cfg_heads.get("recon", {}).get("enabled", False):
            from starVLA.model.modules.uamvla.aux_heads.recon_head import ReconHead
            recon_cfg = self._aux_head_cfg_without_layout_keys(cfg_heads.recon)
            self.aux_heads["recon"] = ReconHead(
                hidden_size=hidden_size, vae=_get_or_build_vae(),
                **{**vision_extra, **recon_cfg},
            )

        # ── 2. LatentDepthHead (Video-Depth-Anything V2 style) ────────────────
        if cfg_heads.get("depth", {}).get("enabled", False):
            depth_cfg = cfg_heads.get("depth", {})
            latent_channels = int(depth_cfg.get("latent_channels", 64))
            self._assert_depth_latent_meta(layout, latent_channels)

            depth_vae = None
            if depth_cfg.get("visualize", False):
                # ZeRO-3 note: the VAE is only touched inside visualize(), never
                # during a training forward.  Safe under ZeRO-2; leave off under ZeRO-3.
                depth_vae = _get_or_build_vae()

            head_kwargs = {
                k: v for k, v in depth_cfg.items()
                if k not in _LT_NON_HEAD_KEYS
            }
            # view_idx=0: the dedicated depth emoji token appears exactly once in the
            # suffix; no extra vision block is appended as in UamGR00T_LT.
            self.aux_heads["depth"] = LatentDepthHead(
                hidden_size=hidden_size,
                image_token_id=image_token_id,   # will be patched to depth emoji
                patches_per_view=layout["patches_per_view"],
                view_idx=0,
                depth_px_size=layout["target_resize"],
                vae=depth_vae,
                **head_kwargs,
            )

        # ── 3. AffordanceTokenHead (Splat-MOVER contact-point style) ─────────
        if cfg_heads.get("affordance", {}).get("enabled", False):
            afford_cfg = cfg_heads.get("affordance", {})
            afford_camera = str(afford_cfg.get("camera", "image"))
            self._assert_affordance_px_meta(layout, afford_camera)

            head_kwargs = {
                k: v for k, v in afford_cfg.items()
                if k not in _LT_NON_HEAD_KEYS
            }
            # view_idx=0: same rationale as LatentDepthHead above.
            self.aux_heads["affordance"] = AffordanceTokenHead(
                hidden_size=hidden_size,
                image_token_id=image_token_id,   # will be patched to affordance emoji
                patches_per_view=layout["patches_per_view"],
                view_idx=0,
                afford_px_size=layout["target_resize"],
                **head_kwargs,
            )

        # ── 4. ActionConditionedFutureHead (SEER style) ───────────────────────
        if cfg_heads.get("action_conditioned_future", {}).get("enabled", False):
            from starVLA.model.modules.uamvla.aux_heads.action_conditioned_future_head import (
                ActionConditionedFutureHead,
            )
            action_model_cfg = self.config.framework.action_model
            action_dim = int(action_model_cfg.get("action_dim", 7))
            action_horizon = int(
                getattr(self, "action_horizon", action_model_cfg.get("action_horizon", 8))
            )
            acf_cfg = self._aux_head_cfg_without_layout_keys(
                cfg_heads.action_conditioned_future
            )
            self.aux_heads["action_conditioned_future"] = ActionConditionedFutureHead(
                hidden_size=hidden_size,
                vae=_get_or_build_vae(),
                action_dim=action_dim,
                action_horizon=action_horizon,
                **{**vision_extra, **acf_cfg},
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
        default="./starVLA/config/training/uamvla_gr00t_libero.yaml",
    )
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)

    model = UamVLAGR00T_hR_LT(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ppv = model._qwen_vision_layout()["patches_per_view"]
    num_views = model._num_views_from_config()
    n_heads = len(model._head_token_str)  # enabled heads

    def fake_rgb(size: int = 224):
        return Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))

    grid = model._qwen_vision_layout()["target_size"]
    sample = {
        "action": torch.rand(16, 7) * 2 - 1,
        "image": [fake_rgb() for _ in range(num_views)],
        "lang": "pick up the blue block",
        "state": np.zeros((1, 7), dtype=np.float32),
        # Sidecar targets (zeros are valid shapes for the forward pass)
        "depth_latent": torch.randn(64, grid, grid),
        "depth_px": torch.rand(grid * 16, grid * 16),
        "affordance_px": torch.rand(grid * 16, grid * 16),
        "image_action_future": torch.rand(3, grid * 16, grid * 16),
        "image_future": torch.rand(3, grid * 16, grid * 16),
    }
    batch = [sample, {**sample, "lang": "open the drawer"}]

    _, qwen_inputs, hidden = model._encode_qwen_hidden(model._prepare_examples(list(batch)))
    counts = (qwen_inputs["input_ids"] == list(model._head_token_id.values())[0]).sum(dim=1)
    expected_per_head = ppv
    print(f"[ok] sequence length: {hidden.shape[1]}")
    print(f"[ok] hidden: {tuple(hidden.shape)}")
    print(f"[ok] enabled head tokens: {list(model._head_token_str.items())}")
    print(f"[ok] tokens per head block: {int(counts[0])} (expected {expected_per_head})")

    out = model(batch)
    print({k: float(v) for k, v in out.items() if torch.is_tensor(v) and v.numel() == 1})
    print(f"[ok] action_loss: {out['action_loss'].item():.4f}")

    pred = model.predict_action(examples=[sample])
    print(f"[ok] predict_action -> {pred['normalized_actions'].shape}")
    print("Finished UamGR00T_hR_LT smoke test.")
