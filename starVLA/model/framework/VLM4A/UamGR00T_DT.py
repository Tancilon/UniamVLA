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


# ---------------------------------------------------------------------------
#  Forward hook: replaces pad-token embeddings at insertion positions with
#  per-timestep state embeddings (gradient-safe via scatter_ + torch.where).
# ---------------------------------------------------------------------------

class _UamGR00T_DT_StateHook:
    """One-shot embed_tokens hook for temporal state injection.

    After Qwen's embed_tokens maps token IDs to embeddings, this hook
    replaces the pad-token embeddings that were inserted at each frame's
    vision_end boundary with the corresponding state_proj_to_llm(state_t)
    output.  torch.where preserves autograd through both paths.

    Args:
        insert_positions: list[B] of list[K] — positions in the extended
                          sequence where pad tokens were inserted.
        state_embeds:     Tensor (B, K, d_llm) — projected state embeddings.
    """

    def __init__(self, insert_positions, state_embeds: torch.Tensor) -> None:
        self.insert_positions = insert_positions   # list[B][K] int
        self.state_embeds = state_embeds           # (B, K, D)

    def __call__(self, module, input_tuple, output: torch.Tensor) -> torch.Tensor:
        B, T_new, D = output.shape
        K = len(self.insert_positions[0])
        device = output.device

        # Build index tensor (B, K)
        idx = torch.tensor(
            self.insert_positions, dtype=torch.long, device=device
        )

        # Scatter state embeddings into a (B, T_new, D) zero tensor.
        # scatter_ is autograd-safe when the source tensor requires grad.
        state_full = output.new_zeros(B, T_new, D)
        state_full.scatter_(
            1,
            idx.unsqueeze(-1).expand(-1, -1, D),
            self.state_embeds.to(dtype=output.dtype, device=device),
        )

        # Boolean replacement mask (B, T_new)
        mask = torch.zeros(B, T_new, dtype=torch.bool, device=device)
        mask.scatter_(1, idx, True)

        # torch.where: gradient flows through state_full at True positions
        # and through original output at False positions.
        return torch.where(mask.unsqueeze(-1), state_full, output)

logger = logging.getLogger(__name__)


@FRAMEWORK_REGISTRY.register("UamGR00T_DT")
class UamVLAGR00T_DT(UamVLAGR00T):
    """GR00T + Seer-style future prediction branch; no auxhead design."""

    # Qwen3-VL token IDs (from config.json)
    _VISION_START_TOKEN_ID: int = 151652
    _VISION_END_TOKEN_ID:   int = 151653

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
            history_frames=int(sjd_cfg.get("history_frames", 1)),
            atten_goal=int(sjd_cfg.get("atten_goal", 0)),
            # FutureDiTBranch params (active when future_decoder_type="dit")
            future_decoder_type=str(sjd_cfg.get("future_decoder_type", "vit")),
            dit_hidden_dim=int(sjd_cfg.get("dit_hidden_dim", 256)),
            dit_depth=int(sjd_cfg.get("dit_depth", 2)),
            dit_num_heads=int(sjd_cfg.get("dit_num_heads", 8)),
            dit_num_img_tokens=int(sjd_cfg.get("dit_num_img_tokens", 49)),
            dit_num_inference_steps=int(sjd_cfg.get("dit_num_inference_steps", 4)),
            token_loss_weight=float(sjd_cfg.get("token_loss_weight", 1.0)),
            pixel_loss_weight=float(sjd_cfg.get("pixel_loss_weight", 0.1)),
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

    def _encode_qwen_hidden(
        self,
        examples: List[dict],
        state: "torch.Tensor | None" = None,
    ):
        """Qwen forward with optional per-timestep state injection (Option A).

        When ``state`` is provided and ``state_proj_to_llm`` exists, one state
        token is inserted into Qwen's input sequence immediately after each
        frame's last vision_end token.  This aligns with Seer's per-timestep
        A-block state token (text | state | image_emb | cls_token).

        The insertion is done via a one-shot forward hook on embed_tokens so
        that the projected state embeddings replace the pad-token placeholders
        inserted into input_ids.  Gradients flow through state_proj_to_llm
        via scatter_ + torch.where inside the hook.

        Falls back to plain Qwen forward when state is None or K == 1.
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

        # --- Option A: temporal state injection into Qwen input_embeds ---
        # State is inserted BEFORE each frame's first vision_start token so that
        # image tokens (which come after vision_start) can attend to state via
        # Qwen causal attention — mirroring Seer's [text|state|image] A-block order.
        inject = (
            state is not None
            and self.state_proj_to_llm is not None
            and self._history_frames > 1
        )
        if inject:
            num_views = self._num_views_from_config()
            K         = self._history_frames
            pad_id    = self.qwen_vl_interface.processor.tokenizer.pad_token_id

            # Find first vision_start per frame; insert state token BEFORE it.
            # We pass (frame_start - 1) as the "end_pos" to _extend_inputs_for_state,
            # which inserts the pad token after that position = before vision_start.
            frame_starts = self._find_frame_start_positions(
                qwen_inputs["input_ids"], num_views, K, self._VISION_START_TOKEN_ID
            )
            before_starts = [[s - 1 for s in starts] for starts in frame_starts]
            new_input_ids, new_attn_mask, insert_pos = self._extend_inputs_for_state(
                qwen_inputs["input_ids"],
                qwen_inputs["attention_mask"],
                before_starts,
                pad_id,
            )

            # Project state → (B, K, d_llm).
            # Cast to the Linear layer's dtype (BF16 under DeepSpeed) to avoid
            # dtype mismatch: state arrives as float32 from the dataloader path.
            s = state.to(dtype=self.state_proj_to_llm.weight.dtype)
            if s.ndim == 2:
                s = s.unsqueeze(1)
            state_embeds = self.state_proj_to_llm(s)

            # Register hook on the input embedding layer.
            # Use get_input_embeddings() (standard HF API) instead of hard-coded
            # .model.embed_tokens — Qwen3VL stores it at .model.language_model.embed_tokens.
            hook = self.qwen_vl_interface.model.get_input_embeddings().register_forward_hook(
                _UamGR00T_DT_StateHook(insert_pos, state_embeds)
            )
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    qwenvl_outputs = self.qwen_vl_interface(
                        input_ids=new_input_ids,
                        attention_mask=new_attn_mask,
                        pixel_values=qwen_inputs.get("pixel_values"),
                        image_grid_thw=qwen_inputs.get("image_grid_thw"),
                        output_attentions=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
            finally:
                hook.remove()

            # Update both attention_mask and input_ids in qwen_inputs so that
            # downstream consumers (_collate_aux, _assert_image_token_count) see
            # the extended sequence with injected state tokens.
            qwen_inputs = {
                **qwen_inputs,
                "attention_mask": new_attn_mask,
                "input_ids":      new_input_ids,
            }
            state_token_positions: "list[list[int]] | None" = insert_pos  # list[B][K]
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

        hidden = qwenvl_outputs.hidden_states[-1]
        self._assert_image_token_count(qwen_inputs.get("input_ids",
                                                        new_input_ids if inject else None),
                                       examples)

        # qwen_frame_ends: list[B][K] int — per-sample position of each frame's
        # last vision_end token in qwen_hidden (inclusive).
        # Used by _build_block_mask to prevent future_t from attending qwen positions
        # that belong to frame t+1..K-1 of that specific sample.
        # Must be per-sample because Qwen uses left-padding: different prompt lengths
        # shift image token positions by varying amounts across batch samples.
        _V = self._num_views_from_config()
        K  = self._history_frames
        if inject:
            # Scan each sample's new_input_ids for vision_end positions.
            qwen_frame_ends: "list[list[int]] | None" = []
            for b in range(new_input_ids.shape[0]):
                ve_b = (new_input_ids[b] == self._VISION_END_TOKEN_ID).nonzero(
                    as_tuple=False
                ).squeeze(-1)
                qwen_frame_ends.append([int(ve_b[(t + 1) * _V - 1].item()) for t in range(K)])
            # state_token_positions: list[B][K] — position of each injected state token.
            # Used by _build_block_mask to implement atten_goal_state (Seer pretrain.sh:42):
            # future_t may additionally attend the state token of frame t+atten_goal.
            state_token_positions: "list[list[int]] | None" = insert_pos
        elif K > 1:
            qwen_frame_ends = self._find_frame_end_positions(
                qwen_inputs["input_ids"], _V, K, self._VISION_END_TOKEN_ID
            )
            state_token_positions = None  # no injected state tokens in this path
        else:
            qwen_frame_ends = None
            state_token_positions = None

        return examples, qwen_inputs, hidden, qwen_frame_ends, state_token_positions

    # ------------------------------------------------------------------
    #  Static helpers for state injection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_frame_start_positions(
        input_ids: torch.Tensor,
        num_views: int,
        K: int,
        vision_start_id: int,
    ) -> "list[list[int]]":
        """Return the position of the FIRST vision_start token for each frame.

        With V views per frame, the sequence contains K*V vision_start tokens.
        Frame t's first view starts at the t*V-th occurrence (0-indexed).
        State tokens are inserted just before this position so that image tokens
        (which come after vision_start) can attend to state via causal attention.

        Returns list[B] of list[K] int positions.
        """
        result = []
        for b in range(input_ids.shape[0]):
            vs_pos = (input_ids[b] == vision_start_id).nonzero(as_tuple=False).squeeze(-1)
            frame_starts = [int(vs_pos[t * num_views].item()) for t in range(K)]
            result.append(frame_starts)
        return result

    @staticmethod
    def _find_frame_end_positions(
        input_ids: torch.Tensor,
        num_views: int,
        K: int,
        vision_end_id: int,
    ) -> "list[list[int]]":
        """Return the position of the last vision_end token for each frame.

        With V views per frame, the sequence contains K*V vision_end tokens.
        Frame t's last token is the (t+1)*V-th occurrence (0-indexed: (t+1)*V-1).

        Returns list[B] of list[K] int positions.
        """
        result = []
        for b in range(input_ids.shape[0]):
            ve_pos = (input_ids[b] == vision_end_id).nonzero(as_tuple=False).squeeze(-1)
            frame_ends = [
                int(ve_pos[(t + 1) * num_views - 1].item()) for t in range(K)
            ]
            result.append(frame_ends)
        return result

    @staticmethod
    def _extend_inputs_for_state(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        frame_end_positions: "list[list[int]]",
        pad_token_id: int,
    ) -> "tuple[torch.Tensor, torch.Tensor, list[list[int]]]":
        """Insert one pad token after each frame's last vision_end.

        Returns:
            new_input_ids:    (B, T+K)
            new_attention_mask: (B, T+K)
            insert_positions: list[B] of list[K] — positions in new sequence
                              where the inserted tokens landed.
        """
        new_ids_list, new_mask_list, all_insert_pos = [], [], []

        for b, ends in enumerate(frame_end_positions):
            row_ids  = input_ids[b]
            row_mask = attention_mask[b]
            parts_ids, parts_mask, insert_pos = [], [], []
            prev = 0

            for end_pos in sorted(ends):
                parts_ids.append(row_ids[prev : end_pos + 1])
                parts_mask.append(row_mask[prev : end_pos + 1])
                # Record the position AFTER the inserted token lands.
                insert_pos.append(sum(p.shape[0] for p in parts_ids))
                parts_ids.append(row_ids.new_full((1,), pad_token_id))
                parts_mask.append(row_mask.new_ones(1))
                prev = end_pos + 1

            parts_ids.append(row_ids[prev:])
            parts_mask.append(row_mask[prev:])

            new_ids_list.append(torch.cat(parts_ids))
            new_mask_list.append(torch.cat(parts_mask))
            all_insert_pos.append(insert_pos)

        return (
            torch.stack(new_ids_list),
            torch.stack(new_mask_list),
            all_insert_pos,
        )

    # DT-specific keys that _unpack_lerobot_sample does not carry through.
    _DT_PASSTHROUGH_KEYS = ("future_rgb", "image_history", "future_rgb_history", "action_history")

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
            # Restore DT-specific keys stripped by the parent unpack step.
            for key in self._DT_PASSTHROUGH_KEYS:
                if key in orig:
                    out[key] = orig[key]

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
                        state_arr = arr[:, indices].copy()
                        # Align gripper state with Seer (train_utils.py:109):
                        # CALVIN gripper is {-1, 1}; Seer converts to {0, 1} via (x+1)//2.
                        # Apply to the last dimension (gripper is always the last index).
                        state_arr[:, -1] = (state_arr[:, -1] + 1) / 2
                        out["state"] = torch.as_tensor(state_arr)

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

    @staticmethod
    def _collect_future_rgb_history(examples: List[dict], device, dtype):
        """Batch dense future_rgb_history for compute_loss dense mode.

        Input per sample: list[K] of list[num_views] of Tensor(3,H,W).
        Output: list[K] of list[num_views] of Tensor(B,3,H,W).
        Returns None when field is absent (single-step fallback).
        """
        if not examples or "future_rgb_history" not in examples[0]:
            return None
        K = len(examples[0]["future_rgb_history"])
        V = len(examples[0]["future_rgb_history"][0])
        result = []
        for t in range(K):
            views = []
            for v in range(V):
                frames = torch.stack([
                    (e["future_rgb_history"][t][v].float()
                     if torch.is_tensor(e["future_rgb_history"][t][v])
                     else torch.as_tensor(
                         np.asarray(e["future_rgb_history"][t][v]), dtype=torch.float32))
                    for e in examples
                ]).to(device=device, dtype=dtype)
                views.append(frames)
            result.append(views)
        return result  # list[K] of list[V] of Tensor(B,3,H,W)

    @staticmethod
    def _collect_action_history(examples: List[dict]):
        """Batch dense action_history for compute_loss dense mode.

        Input per sample: ndarray (K, T_act, action_dim).
        Output: ndarray (B, K, T_act, action_dim).
        Returns None when field is absent.
        """
        if not examples or "action_history" not in examples[0]:
            return None
        return np.stack(
            [
                np.asarray(e["action_history"], dtype=np.float32)
                for e in examples
            ],
            axis=0,
        )  # (B, K, T_act, action_dim)

    # ------------------------------------------------------------------
    #  GT image token extraction (for FutureDiTBranch)
    # ------------------------------------------------------------------

    def _extract_gt_image_tokens(
        self,
        rgb_list: "List[torch.Tensor]",   # list[V] of Tensor(B, 3, H, W) float [0,1]
        device: torch.device,
        dtype: torch.dtype,
    ) -> "List[torch.Tensor] | None":
        """Extract GT image tokens per view from future_rgb using frozen Qwen ViT.

        Only active when seer_joint_decoder.future_decoder_type = "dit".
        Runs the visual encoder (no LLM), so the overhead is small.

        Returns list[V] of Tensor(B, N_img, d_llm) or None.
        """
        if self.seer_joint_decoder._future_decoder_type != "dit":
            return None
        if not rgb_list:
            return None

        from torchvision.transforms.functional import to_pil_image

        V = len(rgb_list)
        B = rgb_list[0].shape[0]
        results = []

        visual = self.qwen_vl_interface.model.model.visual
        proc   = self.qwen_vl_interface.processor.image_processor
        vis_dtype = next(visual.parameters()).dtype

        for v in range(V):
            rgb_v = rgb_list[v]  # (B, 3, H, W) float [0,1]

            # Convert tensors to PIL for the Qwen image processor
            pils = [
                to_pil_image(
                    (rgb_v[b].float().clamp(0, 1) * 255).byte().cpu()
                )
                for b in range(B)
            ]

            # Process through Qwen image processor
            with torch.no_grad():
                inputs = proc(images=pils, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(
                    device=device, dtype=vis_dtype
                )
                grid_thw = inputs["image_grid_thw"].to(device=device)

                # Visual encoder (frozen): (B*N_tokens, d_llm)
                tokens = visual(pixel_values, grid_thw=grid_thw)
                # Qwen3VL visual encoder may return (features, ...) tuple
                if isinstance(tokens, tuple):
                    tokens = tokens[0]

            N_tokens = tokens.shape[0] // B
            gt_v = tokens.reshape(B, N_tokens, -1).to(dtype=dtype)  # (B, N_img, d_llm)
            results.append(gt_v)

        return results  # list[V] of (B, N_img, d_llm)

    def _extract_gt_image_tokens_history(
        self,
        rgb_history: "List | None",   # list[K] of list[V](B, 3, H, W)
        device: torch.device,
        dtype: torch.dtype,
    ) -> "List | None":
        """Extract GT tokens for dense supervision history.

        Returns list[K] of list[V](B, N_img, d_llm) or None.
        """
        if rgb_history is None:
            return None
        if self.seer_joint_decoder._future_decoder_type != "dit":
            return None
        return [
            self._extract_gt_image_tokens(rgb_k, device, dtype)
            for rgb_k in rgb_history
        ]

    # ------------------------------------------------------------------
    #  Training forward
    # ------------------------------------------------------------------

    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Training forward: pixel-space future MSE + Seer-style MLP action MSE.

        State tokens are injected into Qwen's input sequence (Option A: input-side
        temporal alignment) so that all K frames' state information is processed
        by Qwen's full depth before reaching the SeerJointDecoder.
        """
        examples = self._prepare_examples(examples)

        # Pre-extract state for Qwen temporal injection (before hidden exists).
        # Use parameter device; will be re-cast to hidden's dtype after forward.
        _dev = next(self.parameters()).device
        state_pre = self._state_batch_or_none(examples, _dev, torch.float32)

        examples, qwen_inputs, hidden, qwen_frame_ends, state_token_positions = (
            self._encode_qwen_hidden(examples, state=state_pre)
        )

        device, dtype = hidden.device, hidden.dtype
        state = state_pre.to(device=device, dtype=dtype) if state_pre is not None else None

        qwen_pad_mask = qwen_inputs.get("attention_mask")

        # --- Future reconstruction + action prediction (unified Seer block mask) ---
        batch_dict          = self._collate_aux(examples, qwen_inputs)
        future_rgb_list     = self._collect_future_rgb(examples, device, dtype)
        future_rgb_history  = self._collect_future_rgb_history(examples, device, dtype)
        action_history      = self._collect_action_history(examples)

        # GT image tokens for FutureDiTBranch (frozen Qwen ViT, no gradient here)
        gt_img_tokens         = self._extract_gt_image_tokens(
            future_rgb_list or [], device, dtype
        )
        gt_img_tokens_history = self._extract_gt_image_tokens_history(
            future_rgb_history, device, dtype
        )

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
                gt_actions=actions_target,
                future_rgb_list=future_rgb_list,
                action_history=action_history,
                future_rgb_history=future_rgb_history,
                qwen_frame_ends=qwen_frame_ends,
                state_token_positions=state_token_positions,
                gt_img_tokens=gt_img_tokens_history if future_rgb_history else gt_img_tokens,
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
        """Inference: Qwen (with state injection) → SeerJointDecoder → actions."""
        if not isinstance(examples, list):
            examples = [examples]

        examples = self._prepare_examples(examples)

        _dev = next(self.parameters()).device
        state_pre = self._state_batch_or_none(examples, _dev, torch.float32)

        examples, qwen_inputs, hidden, qwen_frame_ends, state_token_positions = (
            self._encode_qwen_hidden(examples, state=state_pre)
        )

        device, dtype = hidden.device, hidden.dtype
        state = state_pre.to(device=device, dtype=dtype) if state_pre is not None else None

        qwen_pad_mask = qwen_inputs.get("attention_mask")

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.seer_joint_decoder.predict_action(
                qwen_hidden=hidden,
                qwen_pad_mask=qwen_pad_mask,
                qwen_frame_ends=qwen_frame_ends,
                state_token_positions=state_token_positions,
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
