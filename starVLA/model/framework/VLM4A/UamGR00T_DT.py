"""UamGR00T_DT: Seer-style future prediction branch on top of Qwen3-VL + GR00T.

Sequence layout (standard, no duplicate images unlike _LT):

    [instruction] -> [primary x49] -> [wrist x49]

Future branch runs *after* Qwen encoding:
  1. Learnable future_query_tokens cross-attend to Qwen hidden states
     + compressed history tokens (K-1 frames via HistoryVisionEncoder)
     + robot state token.
  2. future_tokens appended to vl_embs → GR00T DiT cross-attends to both.
  3. Pixel decoder reconstructs t+N future frame patches (MSE loss).

Key differences from UamGR00T_LT:
  - No auxhead design (no _maybe_build_aux_heads, no duplicate image blocks).
  - Future supervision is not VAE latent.
  - Historical context via a separate lightweight HistoryVisionEncoder.

Dataloader contract:
  example["image"]          — list of PIL images, current frame (num_views).
  example["image_history"]  — list[list[PIL]] shape (K-1, num_views); optional.
  example["future_rgb"]     — list[Tensor(3,H,W)] shape (num_views,) in [0,1]; train only.
  example["state"]          — (1, state_dim) numpy array; optional.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
from PIL import Image as PILImage

from starVLA.model.framework.VLM4A.UamGR00T import UamVLAGR00T
from starVLA.model.modules.uamvla.components.future_cross_attn import FutureCrossAttnBranch
from starVLA.model.modules.uamvla.components.history_vision_encoder import HistoryVisionEncoder
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)


@FRAMEWORK_REGISTRY.register("UamGR00T_DT")
class UamVLAGR00T_DT(UamVLAGR00T):
    """GR00T + Seer-style future prediction branch; no auxhead design."""

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(self, config) -> None:
        # Builds Qwen3-VL backbone + GR00T action head + sidecar helpers.
        # No call to _maybe_build_aux_heads.
        super().__init__(config)

        dt_cfg = self.config.framework.get("future_branch", {})
        history_cfg = self.config.framework.get("history_encoder", {})

        d_llm = self.qwen_vl_interface.model.config.hidden_size
        num_views = self._num_views_from_config()
        state_dim = int(self.config.framework.action_model.get("state_dim", 0) or 0)

        # Future cross-attention branch (DiT diffusion decoder).
        self.future_branch = FutureCrossAttnBranch(
            d_model=d_llm,
            num_views=num_views,
            num_obs_tokens=int(dt_cfg.get("num_obs_tokens", 49)),
            num_cross_attn_layers=int(dt_cfg.get("num_cross_attn_layers", 2)),
            num_heads=int(dt_cfg.get("num_heads", 8)),
            state_dim=state_dim,
            future_weight=float(dt_cfg.get("future_weight", 0.5)),
            z_channel=int(dt_cfg.get("z_channel", 512)),
            dit_embed_dim=int(dt_cfg.get("dit_embed_dim", 512)),
            dit_depth=int(dt_cfg.get("dit_depth", 3)),
            dit_timesteps=str(dt_cfg.get("dit_timesteps", "1000")),
        )

        # History vision encoder (skipped when history_frames <= 1).
        self._history_frames = int(self.config.framework.get("history_frames", 1))
        self.history_encoder: HistoryVisionEncoder | None = None
        if self._history_frames > 1:
            self.history_encoder = HistoryVisionEncoder(
                qwen_vl_interface=self.qwen_vl_interface,
                d_llm=d_llm,
                num_latents=int(history_cfg.get("num_latents", 10)),
                resampler_depth=int(history_cfg.get("resampler_depth", 3)),
                freeze_visual=bool(history_cfg.get("freeze_qwen_vit", True)),
                max_history_frames=self._history_frames - 1,
                num_views=num_views,
            )

        # exec_horizon: how many actions to return at inference time.
        # Predict action_horizon steps but only execute exec_horizon of them
        # before re-querying (Seer-style: predict 3, execute 1).
        # -1 (default) means return all action_horizon steps.
        self._exec_horizon = int(self.config.framework.get("exec_horizon", -1))

        logger.info(
            "[UamGR00T_DT] history_frames=%d  num_views=%d  state_dim=%d  "
            "future_weight=%.2f  exec_horizon=%d",
            self._history_frames, num_views, state_dim,
            float(dt_cfg.get("future_weight", 0.5)),
            self._exec_horizon,
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _num_views_from_config(self) -> int:
        obs = self.config.datasets.vla_data.get("obs", [])
        views = [k for k in obs if str(k).startswith("video.")]
        if not views:
            raise RuntimeError("datasets.vla_data.obs contains no video.* keys")
        return len(views)

    # DT-specific keys that _unpack_lerobot_sample does not carry through.
    _DT_PASSTHROUGH_KEYS = ("future_rgb", "image_history")

    def _prepare_examples(self, examples: List[dict]) -> List[dict]:
        """Unpack LeRobot samples and restore DT-specific future_rgb / image_history.

        The parent's _unpack_lerobot_sample builds a new dict with a fixed set
        of keys and silently drops future_rgb and image_history.  We splice them
        back in so FutureCrossAttnBranch.compute_loss() receives its targets.
        """
        unpacked = super()._prepare_examples(examples)
        for orig, out in zip(examples, unpacked):
            for key in self._DT_PASSTHROUGH_KEYS:
                if key in orig:
                    out[key] = orig[key]
        return unpacked

    @staticmethod
    def _collect_history(examples: List[dict]):
        """Return (B, K-1, num_views) nested list of PIL images, or None."""
        if not examples or "image_history" not in examples[0]:
            return None
        hist = [e.get("image_history") for e in examples]
        if not hist[0]:
            return None
        return hist

    @staticmethod
    def _collect_future_rgb(examples: List[dict], device, dtype):
        """Stack per-sample future_rgb into a list[Tensor(B,3,H,W)], or None."""
        if not examples or "future_rgb" not in examples[0]:
            return None
        num_views = len(examples[0]["future_rgb"])
        result = []
        for v in range(num_views):
            frames = torch.stack([
                (e["future_rgb"][v].float() if torch.is_tensor(e["future_rgb"][v])
                 else torch.as_tensor(np.asarray(e["future_rgb"][v]), dtype=torch.float32))
                for e in examples
            ]).to(device=device, dtype=dtype)
            result.append(frames)
        return result

    def _encode_history(self, examples: List[dict], device, dtype):
        """Run HistoryVisionEncoder if history is available, else return None."""
        if self.history_encoder is None:
            return None
        hist = self._collect_history(examples)
        if hist is None:
            return None
        return self.history_encoder(hist, device=device, dtype=dtype)

    @torch.no_grad()
    def _encode_future_vlm_features(
        self,
        future_rgb_list: List[torch.Tensor] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor] | None:
        """Encode future frames through Qwen's frozen ViT → VLM patch tokens.

        Runs only the visual encoder (not the full LLM), so the cost is much
        lower than a full _encode_qwen_hidden forward pass.

        Args:
            future_rgb_list: list of (B, 3, H, W) float tensors in [0, 1], one per view.
                             None or empty → returns None.
            device, dtype:   Target device/dtype for the returned tensors.

        Returns:
            list of (B, num_obs_tokens, D_llm) tensors, one per view; or None.
        """
        if not future_rgb_list:
            return None

        B = future_rgb_list[0].shape[0]
        num_obs = self.future_branch.num_obs_tokens
        results: List[torch.Tensor] = []

        for rgb in future_rgb_list:                     # (B, 3, H, W), float [0, 1]
            # Tensor → list of PIL images (processor expects PIL or numpy).
            rgb_u8 = (rgb.detach().cpu().clamp(0.0, 1.0) * 255).byte()
            pil_imgs = [
                PILImage.fromarray(rgb_u8[i].permute(1, 2, 0).numpy())
                for i in range(B)
            ]

            # Qwen image processor → pixel_values + image_grid_thw.
            proc_out = self.qwen_vl_interface.processor.image_processor(
                images=pil_imgs, return_tensors="pt"
            )
            pixel_values = proc_out["pixel_values"].to(device=device, dtype=dtype)
            image_grid_thw = proc_out["image_grid_thw"].to(device=device)

            # Forward through frozen Qwen visual tower only (no LLM layers).
            vl_out = self.qwen_vl_interface.model.visual(
                pixel_values, grid_thw=image_grid_thw
            )
            # model.visual may return a raw tensor or a BaseModelOutput / tuple.
            if isinstance(vl_out, torch.Tensor):
                vl_tokens = vl_out
            elif hasattr(vl_out, "last_hidden_state"):
                vl_tokens = vl_out.last_hidden_state
            else:
                vl_tokens = vl_out[0]
            # vl_tokens: (B * tokens_per_frame, D_llm)

            tokens_per_frame = vl_tokens.shape[0] // B
            if tokens_per_frame != num_obs:
                raise RuntimeError(
                    f"_encode_future_vlm_features: expected {num_obs} tokens/frame "
                    f"(future_branch.num_obs_tokens={num_obs}), "
                    f"got {tokens_per_frame}.  "
                    f"Adjust num_obs_tokens in config or check Qwen image preprocessing."
                )

            results.append(vl_tokens.view(B, num_obs, -1).to(dtype=dtype))

        return results

    # ------------------------------------------------------------------
    #  Training forward
    # ------------------------------------------------------------------

    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Training forward: future VLM feature DDPM loss + (optionally) GR00T action loss.

        When action_model is frozen (freeze_modules: "action_model" in pretrain config),
        action_loss is logged but excluded from backward to prevent noisy gradients from
        the uninitialised DiT polluting the Qwen backbone.  During finetune the action_model
        is unfrozen and action_loss is re-included automatically.
        """
        examples = self._prepare_examples(examples)
        examples, qwen_inputs, hidden = self._encode_qwen_hidden(examples)

        device, dtype = hidden.device, hidden.dtype

        # --- History tokens (optional) ---
        hist_tokens = self._encode_history(examples, device, dtype)

        # --- Robot state for future branch ---
        state = self._state_batch_or_none(examples, device, dtype)
        state_for_future = state[:, 0, :] if state is not None else None  # (B, state_dim)

        # --- Future branch: tokens + VLM feature-space DDPM loss ---
        batch_dict = self._collate_aux(examples, qwen_inputs)
        future_rgb_list = self._collect_future_rgb(examples, device, dtype)
        batch_dict["future_rgb"] = future_rgb_list

        # Encode future frames through frozen Qwen ViT → VLM feature targets.
        future_vlm_feat = self._encode_future_vlm_features(future_rgb_list, device, dtype)

        with torch.autocast("cuda", dtype=torch.float32):
            future_tokens, future_loss = self.future_branch.compute_loss(
                hidden=hidden,
                batch=batch_dict,
                hist_tokens=hist_tokens,
                state=state_for_future,
                future_vlm_features=future_vlm_feat,
            )

        # --- Extend vl_embs with future tokens for GR00T ---
        vl_embs = torch.cat([hidden, future_tokens], dim=1)   # (B, seq+N_f, D)

        # Build extended encoder attention mask (pad mask extended for future tokens).
        base_mask = self._encoder_attention_mask(qwen_inputs)
        if base_mask is not None:
            future_mask_ext = base_mask.new_ones(base_mask.shape[0], future_tokens.shape[1])
            ext_mask = torch.cat([base_mask, future_mask_ext], dim=1)
        else:
            ext_mask = None

        # --- GR00T flow-matching action loss ---
        gt_actions = [e["action"] for e in examples]
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.as_tensor(
                np.asarray([
                    a.detach().float().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
                    for a in gt_actions
                ]),
                device=device, dtype=dtype,
            )
            actions_target = actions[:, -self.action_horizon:, :]
            if actions_target.shape[1] != self.action_horizon:
                raise RuntimeError(
                    f"Expected at least {self.action_horizon} action steps, "
                    f"got {actions.shape[1]}."
                )

            repeated_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            actions_rep = actions_target.repeat(repeated_steps, 1, 1)
            vl_embs_rep = vl_embs.repeat(repeated_steps, 1, 1)
            state_rep = state.repeat(repeated_steps, 1, 1) if state is not None else None
            ext_mask_rep = ext_mask.repeat(repeated_steps, 1) if ext_mask is not None else None

            action_loss = self.action_model(
                vl_embs_rep, actions_rep, state_rep,
                encoder_attention_mask=ext_mask_rep,
            )

        log_metrics = {
            "action_loss_fm": action_loss.detach(),
            "future_recon_loss": future_loss.detach(),
        }

        # When action_model is frozen (pretrain), exclude its loss from backward to
        # prevent noisy gradients from the uninitialised DiT polluting the backbone.
        # freeze_modules sets requires_grad=False on all action_model params; we detect
        # this here so the pretrain→finetune transition requires no config change.
        action_model_trainable = any(p.requires_grad for p in self.action_model.parameters())
        total = (action_loss + future_loss) if action_model_trainable else future_loss

        return {"action_loss": total, **log_metrics}

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        """Inference: Qwen → history encode → future branch → GR00T sample."""
        if not isinstance(examples, list):
            examples = [examples]

        examples = self._prepare_examples(examples)
        examples, qwen_inputs, hidden = self._encode_qwen_hidden(examples)

        device, dtype = hidden.device, hidden.dtype

        hist_tokens = self._encode_history(examples, device, dtype)
        state = self._state_batch_or_none(examples, device, dtype)
        state_for_future = state[:, 0, :] if state is not None else None

        with torch.autocast("cuda", dtype=torch.float32):
            future_tokens = self.future_branch(
                hidden=hidden,
                hist_tokens=hist_tokens,
                state=state_for_future,
            )

        vl_embs = torch.cat([hidden, future_tokens], dim=1)

        base_mask = self._encoder_attention_mask(qwen_inputs)
        if base_mask is not None:
            future_mask_ext = base_mask.new_ones(base_mask.shape[0], future_tokens.shape[1])
            ext_mask = torch.cat([base_mask, future_mask_ext], dim=1)
        else:
            ext_mask = None

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs, state,
                encoder_attention_mask=ext_mask,
            )

        pred_np = pred_actions.detach().cpu().numpy()
        # Apply exec_horizon: return only the first exec_horizon steps so the
        # eval loop can re-query after each step (Seer-style: predict 3, exec 1).
        if self._exec_horizon > 0:
            pred_np = pred_np[:, :self._exec_horizon, :]
        return {"normalized_actions": pred_np}


# ---------------------------------------------------------------------------
#  Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from PIL import Image
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml", type=str,
        default="./examples/calvin/train_files/run_uamgr00t_LT_depth_train.yaml",
    )
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)

    # Inject DT-specific config for the smoke test.
    from omegaconf import OmegaConf as OC
    dt_patch = OC.create({
        "framework": {
            "future_branch": {
                "num_obs_tokens": 49,
                "num_cross_attn_layers": 2,
                "num_heads": 8,
                "future_weight": 0.5,
                "z_channel": 512,
                "dit_embed_dim": 512,
                "dit_depth": 3,
                "dit_timesteps": "1000",
            },
            "history_frames": 1,
        }
    })
    cfg = OC.merge(cfg, dt_patch)

    model = UamVLAGR00T_DT(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_views = model._num_views_from_config()

    def fake_img():
        return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    sample = {
        "action": torch.rand(16, 7) * 2 - 1,
        "image": [fake_img() for _ in range(num_views)],
        "lang": "pick up the blue block",
        "state": np.zeros((1, 7), dtype=np.float32),
        "future_rgb": [torch.rand(3, 224, 224) for _ in range(num_views)],
    }
    batch = [sample, {**sample, "lang": "open the drawer"}]

    out = model(batch)
    print({k: float(v) if torch.is_tensor(v) and v.numel() == 1 else v
           for k, v in out.items() if not torch.is_tensor(v) or v.numel() == 1})
    print(f"[ok] action_loss={out['action_loss'].item():.4f}  "
          f"future_recon_loss={out['future_recon_loss'].item():.4f}")

    pred = model.predict_action([sample])
    print(f"[ok] predict_action → {pred['normalized_actions'].shape}")
    print("Finished")
