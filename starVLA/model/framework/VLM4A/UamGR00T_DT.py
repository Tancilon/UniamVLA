"""UamGR00T_DT: Future Query Tokens in Qwen Sequence + GR00T Action Model.

Architecture:
  1. K history frames + state tokens (injected before each frame via Option-A hook)
     are fed into Qwen3-VL as a single multi-image sequence.
  2. N_future = num_views × future_tokens_per_view learnable future_query_tokens
     (nn.Parameter) are appended at sequence end via a second embed_tokens hook.
     With causal attention they attend to the full K-frame context.
  3. Qwen full hidden (B, T_orig + N_future, D) is passed directly to the GR00T
     FlowmatchingActionHead for action prediction.
  4. The last N_future positions serve as conditions for FutureDiTBranch, supervised
     by GT image tokens from a frozen Qwen visual encoder + pixel-level MSE.

Registered as: ``UamGR00T_DT``
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.UamGR00T import UamVLAGR00T
from starVLA.model.modules.uamvla.components.future_dit_branch import FutureDiTBranch
from starVLA.model.modules.uamvla.components.perceiver_resampler import PerceiverResampler
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State injection hook (gradient-safe, per-frame)
# ---------------------------------------------------------------------------


class _UamGR00T_DT_StateHook:
    """embed_tokens hook: replace pad placeholders with state_proj_to_llm outputs."""

    def __init__(self, insert_positions, state_embeds: torch.Tensor) -> None:
        self.insert_positions = insert_positions  # list[B][K] int
        self.state_embeds = state_embeds  # (B, K, D)

    def __call__(self, module, input_tuple, output: torch.Tensor) -> torch.Tensor:
        B, T_new, D = output.shape
        device = output.device
        idx = torch.tensor(self.insert_positions, dtype=torch.long, device=device)
        state_full = output.new_zeros(B, T_new, D)
        state_full.scatter_(
            1, idx.unsqueeze(-1).expand(-1, -1, D), self.state_embeds.to(dtype=output.dtype, device=device)
        )
        mask = torch.zeros(B, T_new, dtype=torch.bool, device=device)
        mask.scatter_(1, idx, True)
        return torch.where(mask.unsqueeze(-1), state_full, output)


# ---------------------------------------------------------------------------
# Future query token hook (gradient-safe, fixed tail)
# ---------------------------------------------------------------------------


class _FutureQueryHook:
    """embed_tokens hook: replace tail pad tokens with future_query_tokens."""

    def __init__(self, future_query_tokens: torch.Tensor, insert_start: int, n_future: int) -> None:
        self.fq = future_query_tokens
        self.start = insert_start
        self.n = n_future

    def __call__(self, module, input_tuple, output: torch.Tensor) -> torch.Tensor:
        B, T, D = output.shape
        device, dtype = output.device, output.dtype
        fq = self.fq.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype, device=device)
        fq_full = torch.zeros_like(output)
        fq_full[:, self.start : self.start + self.n, :] = fq
        mask = torch.zeros(T, dtype=torch.bool, device=device)
        mask[self.start : self.start + self.n] = True
        return torch.where(mask.unsqueeze(0).unsqueeze(-1), fq_full, output)


# ---------------------------------------------------------------------------
# Future visualization helper
# ---------------------------------------------------------------------------


def _make_future_strip(
    groups: "list[list]",
    height: int = 160,
    gap: int = 4,
    group_gap: int = 20,
) -> "Image.Image":
    """Stitch image groups into a horizontal strip with group separators.

    Args:
        groups:    Each inner list is one display group (curr / pred / gt).
                   Images can be PIL.Image, np.ndarray, or torch.Tensor.
        height:    Uniform height for all panels in pixels.
        gap:       Pixel gap between panels within a group.
        group_gap: Pixel gap between groups (the visual separator).
    Returns:
        A single PIL.Image.Image.
    """
    from PIL import Image as PILImage

    resample = getattr(getattr(PILImage, "Resampling", PILImage), "LANCZOS")

    def _resize(img: "PILImage.Image") -> "PILImage.Image":
        scale = height / max(1, img.height)
        w = max(1, int(round(img.width * scale)))
        return img.resize((w, height), resample)

    panels: list[list] = [[_resize(im) for im in grp] for grp in groups]

    total_w = (
        sum(im.width for grp in panels for im in grp)
        + gap * sum(max(0, len(grp) - 1) for grp in panels)
        + group_gap * max(0, len(panels) - 1)
    )

    canvas = PILImage.new("RGB", (total_w, height), (40, 40, 40))
    x = 0
    for g_idx, grp in enumerate(panels):
        for p_idx, im in enumerate(grp):
            canvas.paste(im, (x, 0))
            x += im.width
            if p_idx < len(grp) - 1:
                x += gap
        if g_idx < len(panels) - 1:
            x += group_gap
    return canvas


# ---------------------------------------------------------------------------
# Main framework class
# ---------------------------------------------------------------------------


@FRAMEWORK_REGISTRY.register("UamGR00T_DT")
class UamVLAGR00T_DT(UamVLAGR00T):
    """GR00T backbone + future query tokens in Qwen + FutureDiTBranch."""

    _VISION_START_TOKEN_ID: int = 151652
    _VISION_END_TOKEN_ID: int = 151653
    _DT_PASSTHROUGH_KEYS = ("future_rgb", "image_history")

    def __init__(self, config) -> None:
        super().__init__(config)  # builds Qwen3-VL + GR00T action_model

        d_llm = self.qwen_vl_interface.model.config.hidden_size
        num_views = self._num_views_from_config()
        fq_cfg = self.config.framework.get("future_branch", {})
        vs_cfg = self.config.framework.get("visual_resampler", {})
        state_dim = int(self.config.framework.get("state_dim", 0) or 0)

        self.state_proj_to_llm = nn.Linear(state_dim, d_llm) if state_dim > 0 else None
        self._history_frames = int(self.config.framework.get("history_frames", 1))

        fut_per_view = int(fq_cfg.get("future_tokens_per_view", 49))
        self._n_future = num_views * fut_per_view
        self._fut_per_view = fut_per_view
        self.future_query_tokens = nn.Parameter(torch.randn(self._n_future, d_llm) * 0.02)
        self.future_dit_branches = nn.ModuleList(
            [
                FutureDiTBranch(
                    d_model=d_llm,
                    num_obs_tokens=fut_per_view,
                    num_img_tokens=int(fq_cfg.get("dit_num_img_tokens", 49)),
                    image_size=int(fq_cfg.get("image_size", 224)),
                    dit_hidden=int(fq_cfg.get("dit_hidden_dim", 512)),
                    dit_depth=int(fq_cfg.get("dit_depth", 2)),
                    dit_num_heads=int(fq_cfg.get("dit_num_heads", 8)),
                    num_inference_steps=int(fq_cfg.get("dit_num_inference_steps", 4)),
                    token_loss_weight=float(fq_cfg.get("token_loss_weight", 1.0)),
                    pixel_loss_weight=float(fq_cfg.get("pixel_loss_weight", 0.1)),
                )
                for _ in range(num_views)
            ]
        )

        # ── PerceiverResampler: compress n_vis_orig → n_vis_compressed per view ──
        # Enabled via YAML: framework.visual_resampler.enabled: true
        self._n_vis_orig = int(vs_cfg.get("num_orig_tokens", 49))
        self._n_vis_compressed = int(vs_cfg.get("num_compressed_tokens", 16))
        if bool(vs_cfg.get("enabled", False)):
            self.visual_resampler = PerceiverResampler(
                dim=d_llm,
                depth=int(vs_cfg.get("depth", 3)),
                dim_head=int(vs_cfg.get("dim_head", 64)),
                heads=int(vs_cfg.get("heads", 8)),
                num_latents=self._n_vis_compressed,
                max_num_media=None,
            )
        else:
            self.visual_resampler = None

        logger.info(
            "[UamGR00T_DT] K=%d  views=%d  future/view=%d  state_dim=%d  " "vis_resampler=%s  vis_tokens=%d→%d",
            self._history_frames,
            num_views,
            fut_per_view,
            state_dim,
            "ON" if self.visual_resampler is not None else "OFF",
            self._n_vis_orig,
            self._n_vis_compressed,
        )

    # ------------------------------------------------------------------
    #  Config helpers
    # ------------------------------------------------------------------

    def _num_views_from_config(self) -> int:
        obs = self.config.datasets.vla_data.get("obs", [])
        views = [k for k in obs if str(k).startswith("video.")]
        if not views:
            raise RuntimeError("datasets.vla_data.obs contains no video.* keys")
        return len(views)

    # ------------------------------------------------------------------
    #  Qwen forward with state + future token injection
    # ------------------------------------------------------------------

    def _encode_qwen_hidden(
        self,
        examples: List[dict],
        state: "torch.Tensor | None" = None,
    ):
        """Qwen forward: two paths depending on whether visual_resampler is enabled.

        Hook path  (visual_resampler=None):  original embed_tokens-hook approach.
        Resampler path (visual_resampler set): run ViT manually → compress →
            build inputs_embeds directly → call Qwen LLM with inputs_embeds only.

        Returns (examples, qwen_inputs, hidden, future_hidden) where:
          hidden:        (B, T_seq + N_future, d_llm)
          future_hidden: (B, N_future, d_llm)
        """
        batch_images = [self._force_resize_640(example["image"]) for example in examples]
        examples = [{**example, "image": images} for example, images in zip(examples, batch_images)]
        instructions = [example["lang"] for example in examples]
        use_markers = bool(self.config.framework.get("use_temporal_markers", False))
        num_vpf = self._num_views_from_config() if use_markers else None
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            num_views_per_frame=num_vpf,
        )

        if self.visual_resampler is not None:
            return self._encode_qwen_hidden_resampler(examples, qwen_inputs, state)
        else:
            return self._encode_qwen_hidden_hooks(examples, qwen_inputs, state)

    def _encode_qwen_hidden_hooks(self, examples, qwen_inputs, state):
        """Original hook-based path (visual_resampler disabled)."""
        inject_state = state is not None and self.state_proj_to_llm is not None and self._history_frames > 1
        if inject_state:
            nv = self._num_views_from_config()
            pad_id = self.qwen_vl_interface.processor.tokenizer.pad_token_id
            fs = self._find_frame_start_positions(
                qwen_inputs["input_ids"], nv, self._history_frames, self._VISION_START_TOKEN_ID
            )
            before = [[s - 1 for s in row] for row in fs]
            new_ids, new_mask, s_pos = self._extend_inputs_for_state(
                qwen_inputs["input_ids"], qwen_inputs["attention_mask"], before, pad_id
            )
            sv = state.to(dtype=self.state_proj_to_llm.weight.dtype)
            if sv.ndim == 2:
                sv = sv.unsqueeze(1)
            s_emb = self.state_proj_to_llm(sv)
            s_hook = self.qwen_vl_interface.model.get_input_embeddings().register_forward_hook(
                _UamGR00T_DT_StateHook(s_pos, s_emb)
            )
        else:
            new_ids, new_mask, s_hook = (qwen_inputs["input_ids"], qwen_inputs["attention_mask"], None)

        B, T0 = new_ids.shape
        pad_id = self.qwen_vl_interface.processor.tokenizer.pad_token_id
        ext_ids = torch.cat([new_ids, new_ids.new_full((B, self._n_future), pad_id)], 1)
        ext_mask = torch.cat([new_mask, new_mask.new_ones((B, self._n_future))], 1)
        fq_hook = self.qwen_vl_interface.model.get_input_embeddings().register_forward_hook(
            _FutureQueryHook(self.future_query_tokens, T0, self._n_future)
        )

        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.qwen_vl_interface(
                    input_ids=ext_ids,
                    attention_mask=ext_mask,
                    pixel_values=qwen_inputs.get("pixel_values"),
                    image_grid_thw=qwen_inputs.get("image_grid_thw"),
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            fq_hook.remove()
            if s_hook is not None:
                s_hook.remove()

        hidden = out.hidden_states[-1]
        qwen_inputs = {**qwen_inputs, "attention_mask": ext_mask, "input_ids": ext_ids}
        future_hidden = hidden[:, -self._n_future :, :]
        return examples, qwen_inputs, hidden, future_hidden

    def _encode_qwen_hidden_resampler(self, examples, qwen_inputs, state):
        """PerceiverResampler path: ViT → compress → inputs_embeds → Qwen LLM.

        Sequence length is reduced from T_orig (with n_vis_orig tokens per image)
        to T_comp (with n_vis_compressed tokens per image), cutting Qwen attention
        cost by up to (n_vis_orig / n_vis_compressed)² per image slot.
        """
        B = len(examples)
        device = next(self.parameters()).device
        d_llm = self.qwen_vl_interface.model.config.hidden_size

        # ── Step 1: run Qwen ViT independently ──
        # visual = ViT + merger; returns (N_total_vis_tokens, d_llm)
        visual = self.qwen_vl_interface.model.model.visual
        vis_dtype = next(visual.parameters()).dtype
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vis_flat = visual(
                qwen_inputs["pixel_values"].to(device=device, dtype=vis_dtype),
                grid_thw=qwen_inputs["image_grid_thw"].to(device=device),
            )  # (B * N_img * n_vis_orig, d_llm)

        # ── Step 2: reshape → PerceiverResampler → (B, N_img, n_comp, d_llm) ──
        n_orig = self._n_vis_orig
        n_comp = self._n_vis_compressed
        total_vis = vis_flat.shape[0]
        n_img_per_sample = total_vis // (B * n_orig)
        vis_4d = vis_flat.reshape(B, n_img_per_sample, n_orig, d_llm)

        rs_dtype = next(self.visual_resampler.parameters()).dtype
        with torch.autocast("cuda", dtype=torch.bfloat16):
            compressed = self.visual_resampler(vis_4d.to(dtype=rs_dtype))  # (B, N_img, n_comp, d_llm)

        # ── Step 3: optional state embeddings ──
        inject_state = state is not None and self.state_proj_to_llm is not None and self._history_frames > 1
        state_embs = None
        if inject_state:
            sv = state.to(dtype=self.state_proj_to_llm.weight.dtype)
            if sv.ndim == 2:
                sv = sv.unsqueeze(1)  # (B, 1, state_dim)
            state_embs = self.state_proj_to_llm(sv)  # (B, K, d_llm)

        # ── Step 4: build inputs_embeds from text tokens + compressed visual ──
        embed_fn = self.qwen_vl_interface.model.get_input_embeddings()
        inputs_embeds, new_mask = self._build_compressed_inputs_embeds(
            input_ids=qwen_inputs["input_ids"].to(device),
            attention_mask=qwen_inputs["attention_mask"].to(device),
            compressed_vis=compressed.to(dtype=torch.bfloat16),
            embed_fn=embed_fn,
            n_orig=n_orig,
            n_comp=n_comp,
            num_views=self._num_views_from_config(),
            state_embs=state_embs,
            device=device,
        )

        # ── Step 5: append future_query_tokens at tail ──
        fq = self.future_query_tokens.unsqueeze(0).expand(B, -1, -1)
        fq = fq.to(dtype=inputs_embeds.dtype, device=device)
        inputs_embeds = torch.cat([inputs_embeds, fq], dim=1)
        new_mask = torch.cat([new_mask, new_mask.new_ones(B, self._n_future)], dim=1)

        # ── Step 6: run Qwen LLM only (no pixel_values) ──
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface.model(
                inputs_embeds=inputs_embeds,
                attention_mask=new_mask,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden = out.hidden_states[-1]  # (B, T_comp + N_future, d_llm)
        future_hidden = hidden[:, -self._n_future :, :]
        qwen_inputs_out = {**qwen_inputs, "attention_mask": new_mask}
        return examples, qwen_inputs_out, hidden, future_hidden

    # ------------------------------------------------------------------
    #  Static helpers (state injection infrastructure)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_frame_start_positions(input_ids, num_views, K, vision_start_id):
        result = []
        for b in range(input_ids.shape[0]):
            vs = (input_ids[b] == vision_start_id).nonzero(as_tuple=False).squeeze(-1)
            result.append([int(vs[t * num_views].item()) for t in range(K)])
        return result

    @staticmethod
    def _extend_inputs_for_state(input_ids, attention_mask, frame_end_positions, pad_id):
        new_ids_list, new_mask_list, all_pos = [], [], []
        for b, ends in enumerate(frame_end_positions):
            ri, rm = input_ids[b], attention_mask[b]
            parts_i, parts_m, pos, prev = [], [], [], 0
            for ep in sorted(ends):
                parts_i.append(ri[prev : ep + 1])
                parts_m.append(rm[prev : ep + 1])
                pos.append(sum(p.shape[0] for p in parts_i))
                parts_i.append(ri.new_full((1,), pad_id))
                parts_m.append(rm.new_ones(1))
                prev = ep + 1
            parts_i.append(ri[prev:])
            parts_m.append(rm[prev:])
            new_ids_list.append(torch.cat(parts_i))
            new_mask_list.append(torch.cat(parts_m))
            all_pos.append(pos)
        return torch.stack(new_ids_list), torch.stack(new_mask_list), all_pos

    @staticmethod
    def _build_compressed_inputs_embeds(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        compressed_vis: torch.Tensor,
        embed_fn: "nn.Embedding",
        n_orig: int,
        n_comp: int,
        num_views: int,
        state_embs: "torch.Tensor | None",
        device: torch.device,
    ):
        """Rebuild the Qwen input sequence with compressed visual tokens.

        Replaces every n_orig-token vision span (between <|vision_start|> and
        <|vision_end|>) with the corresponding n_comp compressed tokens from
        ``compressed_vis``.  Optionally injects one state embedding before the
        first view of each history/current frame.

        Args:
            input_ids:      (B, T)     original tokenizer output.
            attention_mask: (B, T)     1 for real tokens, 0 for left-padding.
            compressed_vis: (B, N_img, n_comp, d_llm)  resampled visual tokens.
            embed_fn:       nn.Embedding — the LLM's token embedding table.
            n_orig:         original visual tokens per image slot (e.g. 49).
            n_comp:         compressed tokens per image slot (e.g. 16).
            num_views:      number of camera views per time-step.
            state_embs:     (B, K, d_llm) or None — state embeddings to inject
                            before the first view of each frame.
            device:         target device.

        Returns:
            inputs_embeds: (B, T_new, d_llm)
            new_mask:      (B, T_new)  long tensor, all 1 (left-pad already stripped).
        """
        VIS_START = 151652  # <|vision_start|>
        VIS_END = 151653  # <|vision_end|>
        emb_dtype = next(embed_fn.parameters()).dtype

        B = input_ids.shape[0]
        new_embeds_list: list = []
        new_mask_list: list = []

        for b in range(B):
            ids = input_ids[b]
            mask = attention_mask[b]
            T = ids.shape[0]

            parts_e: list = []  # embedding tensors
            parts_m: list = []  # mask tensors (long)

            img_idx = 0  # which compressed image slot we are on
            i = 0
            while i < T:
                # Skip left-padding (mask == 0 outside any vision span)
                if mask[i].item() == 0:
                    i += 1
                    continue

                tok_id = ids[i].item()

                if tok_id == VIS_START:
                    # ── optional state injection before frame boundary ──
                    if state_embs is not None:
                        frame_idx = img_idx // num_views
                        view_idx = img_idx % num_views
                        if view_idx == 0 and frame_idx < state_embs.shape[1]:
                            s = state_embs[b, frame_idx].unsqueeze(0)  # (1, d)
                            parts_e.append(s.to(device=device, dtype=emb_dtype))
                            parts_m.append(ids.new_ones(1))

                    # vision_start embedding
                    vs_emb = embed_fn(ids[i : i + 1])  # (1, d)
                    parts_e.append(vs_emb)
                    parts_m.append(ids.new_ones(1))
                    i += 1

                    # skip original n_orig vision tokens in input_ids
                    i += n_orig

                    # insert compressed tokens for this image
                    comp = compressed_vis[b, img_idx].to(device=device, dtype=emb_dtype)
                    parts_e.append(comp)  # (n_comp, d)
                    parts_m.append(ids.new_ones(n_comp))
                    img_idx += 1

                    # vision_end (next real token after the skipped span)
                    if i < T and ids[i].item() == VIS_END:
                        ve_emb = embed_fn(ids[i : i + 1])
                        parts_e.append(ve_emb)
                        parts_m.append(ids.new_ones(1))
                        i += 1

                else:
                    # regular text token
                    tok_emb = embed_fn(ids[i : i + 1])
                    parts_e.append(tok_emb)
                    parts_m.append(mask[i : i + 1])
                    i += 1

            new_embeds_list.append(torch.cat(parts_e, dim=0))  # (T_new_b, d)
            new_mask_list.append(torch.cat(parts_m, dim=0))  # (T_new_b,)

        # Left-pad to uniform length so the batch can be stacked
        max_len = max(e.shape[0] for e in new_embeds_list)
        d_llm = new_embeds_list[0].shape[-1]
        dtype = new_embeds_list[0].dtype

        padded_e = torch.zeros(B, max_len, d_llm, device=device, dtype=dtype)
        padded_m = torch.zeros(B, max_len, dtype=torch.long, device=device)

        for b in range(B):
            L = new_embeds_list[b].shape[0]
            padded_e[b, max_len - L :] = new_embeds_list[b]
            padded_m[b, max_len - L :] = new_mask_list[b]

        return padded_e, padded_m

    @staticmethod
    def _collect_future_rgb(examples, device, dtype):
        if not examples or "future_rgb" not in examples[0]:
            return None
        V = len(examples[0]["future_rgb"])
        return [
            torch.stack(
                [
                    (
                        e["future_rgb"][v].float()
                        if torch.is_tensor(e["future_rgb"][v])
                        else torch.as_tensor(np.asarray(e["future_rgb"][v]), dtype=torch.float32)
                    )
                    for e in examples
                ]
            ).to(device=device, dtype=dtype)
            for v in range(V)
        ]

    def _extract_gt_image_tokens(self, rgb_list, device, dtype):
        """Extract GT image tokens via frozen Qwen ViT (always active in this class)."""
        if not rgb_list:
            return None
        from torchvision.transforms.functional import to_pil_image

        V, B = len(rgb_list), rgb_list[0].shape[0]
        visual = self.qwen_vl_interface.model.model.visual
        proc = self.qwen_vl_interface.processor.image_processor
        vis_dtype = next(visual.parameters()).dtype
        results = []
        for v in range(V):
            pils = [to_pil_image((rgb_list[v][b].float().clamp(0, 1) * 255).byte().cpu()) for b in range(B)]
            with torch.no_grad():
                inp = proc(images=pils, return_tensors="pt")
                pv = inp["pixel_values"].to(device=device, dtype=vis_dtype)
                gt = inp["image_grid_thw"].to(device=device)
                tok = visual(pv, grid_thw=gt)
                if isinstance(tok, tuple):
                    tok = tok[0]
            results.append(tok.reshape(B, tok.shape[0] // B, -1).to(dtype=dtype))
        return results

    # ------------------------------------------------------------------
    #  Training forward
    # ------------------------------------------------------------------

    def forward(self, examples: List[dict], **kwargs) -> dict:
        examples = self._prepare_examples(examples)
        _dev = next(self.parameters()).device
        state_pre = self._state_batch_or_none(examples, _dev, torch.float32)
        examples, qwen_inputs, hidden, future_hidden = self._encode_qwen_hidden(examples, state=state_pre)
        device, dtype = hidden.device, hidden.dtype
        state = state_pre.to(device=device, dtype=dtype) if state_pre is not None else None

        # GR00T action loss
        gt_actions = [e["action"] for e in examples]
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.as_tensor(
                np.asarray(
                    [a.detach().float().cpu().numpy() if torch.is_tensor(a) else np.asarray(a) for a in gt_actions]
                ),
                device=device,
                dtype=dtype,
            )
            actions_target = actions[:, -self.action_horizon :, :]
            rep = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            enc_mask = self._encoder_attention_mask(qwen_inputs)
            action_loss = self.action_model(
                hidden.repeat(rep, 1, 1),
                actions_target.repeat(rep, 1, 1),
                state.repeat(rep, 1, 1) if state is not None else None,
                encoder_attention_mask=(enc_mask.repeat(rep, 1) if enc_mask is not None else None),
            )

        # FutureDiTBranch loss
        future_rgb_list = self._collect_future_rgb(examples, device, dtype)
        gt_tokens = self._extract_gt_image_tokens(future_rgb_list or [], device, dtype)
        future_loss = hidden.new_zeros(())
        if future_rgb_list and gt_tokens:
            for v, branch in enumerate(self.future_dit_branches):
                obs_v = future_hidden[:, v * self._fut_per_view : (v + 1) * self._fut_per_view]
                future_loss = future_loss + branch.compute_loss(
                    obs_tokens=obs_v,
                    future_rgb=future_rgb_list[v],
                    gt_img_tokens=gt_tokens[v],
                )
            future_loss = future_loss / len(self.future_dit_branches)

        return {
            "action_loss": action_loss + future_loss,
            "action_loss_fm": action_loss.detach(),
            "future_recon_loss": future_loss.detach(),
        }

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        examples = self._prepare_examples(examples)
        _dev = next(self.parameters()).device
        state_pre = self._state_batch_or_none(examples, _dev, torch.float32)
        examples, qwen_inputs, hidden, _ = self._encode_qwen_hidden(examples, state=state_pre)
        device, dtype = hidden.device, hidden.dtype
        state = state_pre.to(device=device, dtype=dtype) if state_pre is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred = self.action_model.predict_action(
                hidden,
                state,
                encoder_attention_mask=self._encoder_attention_mask(qwen_inputs),
            )
        pred_np = pred.detach().cpu().numpy()
        return {"normalized_actions": pred_np}

    # ------------------------------------------------------------------
    #  Visualization
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def visualize_batch(
        self,
        batch: List[dict],
        n_samples: int = 1,
        distributed_all_ranks: bool = False,
    ) -> dict:
        """Future-branch visualization logged to wandb every N training steps.

        Layout per sample (horizontal strip):
          [curr_v0][curr_v1]  |  [pred_v0][pred_v1]  |  [gt_v0][gt_v1]

        Enabled via YAML:
          trainer:
            visualization:
              enabled: true
              train_every_n_steps: 1000
              num_samples: 1
        """
        if not isinstance(batch, list):
            batch = [batch]
        limit = min(max(int(n_samples), 0), len(batch))
        if limit == 0:
            return {}

        try:
            import wandb as _wandb
        except ImportError:
            _wandb = None

        was_training = self.training
        self.eval()
        outputs: dict = {}

        try:
            examples = self._prepare_examples(batch[:limit])
            _dev = next(self.parameters()).device
            state_pre = self._state_batch_or_none(examples, _dev, torch.float32)
            _, qwen_inputs, _, future_hidden = self._encode_qwen_hidden(examples, state=state_pre)
            device, dtype = future_hidden.device, future_hidden.dtype
            num_views = self._num_views_from_config()

            # Predicted future frames: FutureDiTBranch.sample per view
            pred_futures: list = []
            for v, branch in enumerate(self.future_dit_branches):
                obs_v = future_hidden[:, v * self._fut_per_view : (v + 1) * self._fut_per_view]
                pred_futures.append(branch.sample(obs_v))  # (B, 3, H, W)

            # GT future frames (None if future_rgb not in batch)
            gt_futures = self._collect_future_rgb(examples, device, torch.float32)

            for idx in range(limit):
                # Current frame: last num_views images after history prepend
                curr_pils = [self._to_rgb_pil(im) for im in examples[idx]["image"][-num_views:]]
                pred_pils = [self._to_rgb_pil(pred_futures[v][idx]) for v in range(num_views)]
                gt_pils = [self._to_rgb_pil(gt_futures[v][idx]) for v in range(num_views)] if gt_futures else []

                groups = [curr_pils, pred_pils]
                if gt_pils:
                    groups.append(gt_pils)

                strip = _make_future_strip(groups)
                caption = str(examples[idx].get("lang", ""))[:120]
                key = f"viz/future/sample_{idx}"
                outputs[key] = _wandb.Image(strip, caption=caption) if _wandb else strip

        except Exception as exc:
            logger.warning("[UamGR00T_DT] visualize_batch failed: %s", exc, exc_info=True)
        finally:
            if was_training:
                self.train()

        return outputs

    # ------------------------------------------------------------------
    #  Data helpers
    # ------------------------------------------------------------------

    def _prepare_examples(self, examples: List[dict]) -> List[dict]:
        unpacked = super()._prepare_examples(examples)
        for orig, out in zip(examples, unpacked):
            for key in self._DT_PASSTHROUGH_KEYS:
                if key in orig:
                    out[key] = orig[key]
            raw_state = orig.get("state")
            if raw_state is not None:
                arr = np.asarray(raw_state, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] > 1:
                    indices = self._configured_gr00t_state_indices()
                    if indices is None:
                        indices = list(range(7))
                    if max(indices) < arr.shape[-1]:
                        sa = arr[:, indices].copy()
                        out["state"] = torch.as_tensor(sa)
            history = orig.get("image_history")
            if history:
                imgs: List = []
                for fv in history:
                    imgs.extend(fv)
                imgs.extend(out["image"])
                out["image"] = imgs
        return unpacked
