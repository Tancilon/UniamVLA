"""UamGR00T_DT: Qwen3-VL backbone + Seer-style future decoder + lightweight MLP action.

Accurate description (after Seer alignment work):
  "Qwen3-VL 主干 + Seer-style future decoder/action loss"
  (not a full Seer replication — backbone topology and attention masking differ)

Pipeline:
  1. Qwen3-VL processes ALL K frames as a single multi-image sequence
     (K-1 history + current, oldest→newest, in one forward pass).
     Qwen's causal attention provides native temporal context.
     NOTE: Qwen3 chat template places instruction text BEFORE image tokens.

  2. FutureCrossAttnBranch: learnable future_query_tokens cross-attend to
     [Qwen hidden | K-frame state tokens] → future_tokens (9 per view).
       - Seer MAE decoder: obs_tokens + mask_tokens → 2×ViT Block → normalized patch MSE.

  3. SeerMLPActionHead: action_pred_tokens cross-attend to
     [future_tokens | Qwen hidden] → MLP → arm (Tanh) + gripper (±1 at inference).
       - Loss: SmoothL1(arm) + 0.01 × BCEWithLogits(gripper).
       - Gripper inference: logit sign → {+1 open, -1 close} (matches eval threshold-at-0).

Seer alignment status:
  ✅ Future reconstruction: Seer MAE ViT decoder, normalized patch MSE
  ✅ Future/action token interaction: cross-attn to [future | qwen_hidden]
  ✅ Action loss: SmoothL1 + 0.01×BCE (Seer train_utils.py)
  ✅ Action model: lightweight MLP (no diffusion)
  ✅ History: K frames in one Qwen causal forward pass
  ✅ K-frame state: (K, state_dim) preserved in _prepare_examples, bypasses parent squeeze
  ✅ Training protocol: future + action losses jointly (no pretrain freeze)
  ✅ Gripper inference: {-1,+1} binarised from logit sign
  ⚠  Backbone topology: Qwen+cross-attn (not GPT2 internal token coupling)
  ⚠  Attention mask: Qwen causal (not Seer block-level temporal mask)
  ⚠  Prediction granularity: current-anchor chunk (not Seer dense per-timestep)

Dataloader contract:
  example["image"]          — list[PIL], current frame (num_views).
  example["image_history"]  — list[list[PIL]] shape (K-1, num_views); optional.
                              Combined with current → K*num_views images for Qwen.
  example["future_rgb"]     — list[Tensor(3,H,W)] shape (num_views,) in [0,1]; train only.
  example["state"]          — (K, state_dim) after _prepare_examples re-extraction;
                              (1, state_dim) if history absent or K=1.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.UamGR00T import UamVLAGR00T
from starVLA.model.modules.uamvla.components.seer_joint_decoder import SeerJointDecoder
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)


@FRAMEWORK_REGISTRY.register("UamGR00T_DT")
class UamVLAGR00T_DT(UamVLAGR00T):
    """GR00T + Seer-style future prediction branch; no auxhead design."""

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(self, config) -> None:
        # Builds Qwen3-VL backbone via parent chain.
        super().__init__(config)

        sjd_cfg = self.config.framework.get("seer_joint_decoder", {})

        d_llm     = self.qwen_vl_interface.model.config.hidden_size
        num_views = self._num_views_from_config()
        state_dim = int(sjd_cfg.get("state_dim", 0) or 0)

        # SeerJointDecoder: unified self-attn + Seer block mask.
        # Replaces independent FutureCrossAttnBranch + SeerMLPActionHead.
        self.seer_joint_decoder = SeerJointDecoder(
            d_model=d_llm,
            num_views=num_views,
            num_obs_tokens=int(sjd_cfg.get("num_obs_tokens", 9)),
            action_dim=int(sjd_cfg.get("action_dim", 7)),
            action_pred_steps=int(sjd_cfg.get("action_horizon", 3)),
            num_joint_layers=int(sjd_cfg.get("num_joint_layers", 2)),
            num_heads=int(sjd_cfg.get("num_heads", 16)),
            state_dim=state_dim,
            hidden_dim=int(sjd_cfg.get("hidden_dim", d_llm // 2)) or None,
            gripper_loss_ratio=float(sjd_cfg.get("gripper_loss_ratio", 0.01)),
            patch_size=int(sjd_cfg.get("patch_size", 16)),
            image_size=int(sjd_cfg.get("image_size", 224)),
            decoder_dim=int(sjd_cfg.get("decoder_dim", d_llm)) or None,
            num_decoder_heads=int(sjd_cfg.get("num_decoder_heads", 16)),
            num_decoder_blocks=int(sjd_cfg.get("num_decoder_blocks", 2)),
            future_weight=float(sjd_cfg.get("future_weight", 1.0)),
        )

        # K-frame state projection: appends proprio tokens to Qwen hidden so the
        # joint decoder can see state via the block mask (Seer per-timestep state).
        self.state_proj_to_llm = (
            nn.Linear(state_dim, d_llm) if state_dim > 0 else None
        )

        self._history_frames = int(self.config.framework.get("history_frames", 1))
        self._exec_horizon   = int(self.config.framework.get("exec_horizon", -1))

        logger.info(
            "[UamGR00T_DT] history_frames=%d  num_views=%d  state_dim=%d  "
            "num_obs_tokens=%d  num_joint_layers=%d  exec_horizon=%d",
            self._history_frames, num_views, state_dim,
            int(sjd_cfg.get("num_obs_tokens", 9)),
            int(sjd_cfg.get("num_joint_layers", 2)),
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

    def _encode_qwen_hidden(self, examples: List[dict]):
        """Override: pass temporal frame markers when use_temporal_markers is set.

        Temporal markers insert text labels ("Frame t-9:", "Current frame:", etc.)
        between image groups in the Qwen input sequence, providing explicit
        timestep structure as a prompt-based approximation of Seer's token blocks.
        Controlled by ``framework.use_temporal_markers: true`` in YAML (default false).
        """
        batch_images = [self._force_resize_640(example["image"]) for example in examples]
        examples = [
            {**example, "image": images}
            for example, images in zip(examples, batch_images)
        ]
        instructions = [example["lang"] for example in examples]

        use_markers = bool(self.config.framework.get("use_temporal_markers", False))
        num_vpf = self._num_views_from_config() if use_markers else None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            num_views_per_frame=num_vpf,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = qwenvl_outputs.hidden_states[-1]
        self._assert_image_token_count(qwen_inputs["input_ids"], examples)
        return examples, qwen_inputs, hidden

    # DT-specific key that _unpack_lerobot_sample does not carry through.
    _DT_PASSTHROUGH_KEYS = ("future_rgb", "image_history")

    def _prepare_examples(self, examples: List[dict]) -> List[dict]:
        """Unpack samples and build Seer-aligned K-frame image list for Qwen.

        Restores future_rgb for pixel MSE supervision, then flattens
        image_history + current frame into a single ordered list so that
        Qwen processes all K frames in one forward pass (Seer-faithful causal
        temporal context without a separate history encoder):

            example["image"] = [
                hist_0_view0, hist_0_view1,   ← oldest frame
                hist_1_view0, hist_1_view1,
                ...
                hist_{K-2}_view0, hist_{K-2}_view1,
                cur_view0,    cur_view1,       ← current frame (last)
            ]  length = K * num_views

        When image_history is absent the list stays as [cur_view0, cur_view1]
        (single-frame mode, K=1).
        """
        unpacked = super()._prepare_examples(examples)
        for orig, out in zip(examples, unpacked):
            # Always restore future_rgb for training supervision.
            if "future_rgb" in orig:
                out["future_rgb"] = orig["future_rgb"]

            # Re-extract K-frame state, bypassing parent's squeeze.
            # Parent's _extract_gr00t_state_by_indices does reshape(...)[0] for ndim>1,
            # which picks the oldest history frame (row 0).  We preserve all K frames so
            # that FutureCrossAttnBranch._build_kv receives (K, state_dim) context.
            raw_state = orig.get("state")
            if raw_state is not None:
                arr = np.asarray(raw_state, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] > 1:          # K-frame state (K, full_dim)
                    indices = self._configured_gr00t_state_indices()
                    if indices is None:
                        indices = list(range(7))                 # default Calvin gr00t dims
                    if max(indices) < arr.shape[-1]:
                        # (K, state_dim) — all frames, configured dims only
                        out["state"] = torch.as_tensor(arr[:, indices])

            # Build multi-frame image list (oldest → newest → current).
            history = orig.get("image_history")  # list[list[PIL]], shape (K-1, num_views)
            if history:
                current_images = out["image"]   # list[PIL], shape (num_views,)
                all_images: List = []
                for frame_views in history:     # iterate K-1 frames, chronological order
                    all_images.extend(frame_views)
                all_images.extend(current_images)
                out["image"] = all_images       # K * num_views images total
        return unpacked

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

    # ------------------------------------------------------------------
    #  Training forward
    # ------------------------------------------------------------------

    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Training forward: pixel-space future MSE + Seer-style MLP action MSE.

        All K frames are processed by Qwen in _prepare_examples; hidden contains
        the full temporal context.  No separate history encoder needed.

        When action_model is frozen (freeze_modules: "action_model" in pretrain config),
        action_loss is excluded from backward so the future branch can pretrain
        the Qwen backbone without noisy action gradients.
        """
        examples = self._prepare_examples(examples)
        examples, qwen_inputs, hidden = self._encode_qwen_hidden(examples)

        device, dtype = hidden.device, hidden.dtype

        # --- Robot state (K-frame: all history + current) ---
        # With K-frame state config, state shape is (B, K, state_dim).
        state = self._state_batch_or_none(examples, device, dtype)

        # Inject K-frame state tokens into Qwen hidden so the joint decoder
        # sees proprio via the block mask (Seer per-timestep state token coupling).
        if state is not None and self.state_proj_to_llm is not None:
            s = state.float()
            if s.ndim == 2:
                s = s.unsqueeze(1)
            state_tokens = self.state_proj_to_llm(s).to(dtype)
            hidden = torch.cat([hidden, state_tokens], dim=1)  # (B, L+K, d_llm)

        # qwen_pad_mask: (B, L+K) — 1=valid, 0=pad (Qwen left-pads; state tokens valid)
        qwen_pad_mask = qwen_inputs.get("attention_mask")
        if qwen_pad_mask is not None and state is not None and self.state_proj_to_llm is not None:
            K_state = state.shape[1] if state.ndim == 3 else 1
            extra   = qwen_pad_mask.new_ones(qwen_pad_mask.shape[0], K_state)
            qwen_pad_mask = torch.cat([qwen_pad_mask, extra], dim=1)

        # --- Future reconstruction + action prediction (unified Seer block mask) ---
        batch_dict      = self._collate_aux(examples, qwen_inputs)
        future_rgb_list = self._collect_future_rgb(examples, device, dtype)

        gt_actions = [e["action"] for e in examples]
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.as_tensor(
                np.asarray([
                    a.detach().float().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
                    for a in gt_actions
                ]),
                device=device, dtype=dtype,
            )
            T_act = self.seer_joint_decoder.N_act
            actions_target = actions[:, -T_act:, :]
            if actions_target.shape[1] != T_act:
                raise RuntimeError(
                    f"Expected at least {T_act} action steps, got {actions.shape[1]}."
                )

            future_loss, action_loss = self.seer_joint_decoder.compute_loss(
                qwen_hidden=hidden,
                qwen_pad_mask=qwen_pad_mask,
                state=state,
                gt_actions=actions_target,
                future_rgb_list=future_rgb_list,
            )

        log_metrics = {
            "action_loss_mse": action_loss.detach(),
            "future_recon_loss": future_loss.detach(),
        }

        # When joint decoder is frozen (pretrain), exclude action loss from backward.
        decoder_trainable = any(p.requires_grad for p in self.seer_joint_decoder.parameters())
        total = (action_loss + future_loss) if decoder_trainable else future_loss

        return {"action_loss": total, **log_metrics}

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        """Inference: Qwen → state inject → SeerJointDecoder → actions."""
        if not isinstance(examples, list):
            examples = [examples]

        examples = self._prepare_examples(examples)
        examples, qwen_inputs, hidden = self._encode_qwen_hidden(examples)

        device, dtype = hidden.device, hidden.dtype
        state = self._state_batch_or_none(examples, device, dtype)

        # Inject K-frame state tokens (same as forward).
        if state is not None and self.state_proj_to_llm is not None:
            s = state.float()
            if s.ndim == 2:
                s = s.unsqueeze(1)
            state_tokens = self.state_proj_to_llm(s).to(dtype)
            hidden = torch.cat([hidden, state_tokens], dim=1)

        qwen_pad_mask = qwen_inputs.get("attention_mask")
        if qwen_pad_mask is not None and state is not None and self.state_proj_to_llm is not None:
            K_state = state.shape[1] if state.ndim == 3 else 1
            extra   = qwen_pad_mask.new_ones(qwen_pad_mask.shape[0], K_state)
            qwen_pad_mask = torch.cat([qwen_pad_mask, extra], dim=1)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.seer_joint_decoder.predict_action(
                qwen_hidden=hidden,
                qwen_pad_mask=qwen_pad_mask,
                state=state,
            )

        pred_np = pred_actions.detach().cpu().numpy()
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
            "seer_joint_decoder": {
                "num_obs_tokens": 9,
                "future_weight": 1.0,
                "patch_size": 16,
                "decoder_dim": 512,
                "num_decoder_heads": 8,
                "num_decoder_blocks": 2,
                "num_joint_layers": 2,
                "num_heads": 8,
                "action_dim": 7,
                "state_dim": 7,
                "action_horizon": 3,
                "hidden_dim": 512,
                "gripper_loss_ratio": 0.01,
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
