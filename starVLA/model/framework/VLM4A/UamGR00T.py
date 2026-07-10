"""UamVLAGR00T framework: QwenGR00T action path + UAMVLA aux heads.

The action path intentionally mirrors ``QwenGR00T.py``:
  image/lang -> Qwen-VL hidden_states[-1] -> GR00T DiT-B flow-matching head.

UAM-specific code is restricted to sample unpacking, sidecar loading, optional
aux-head construction, and aux loss/visualization.  The OFT action-token prompt
and OFT L1 regression head are not used.
"""
from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image

from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.UamVLAOFT import UamVLAOFT
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)


@FRAMEWORK_REGISTRY.register("UamGR00T")
class UamVLAGR00T(UamVLAOFT):
    """UAMVLA framework using the GR00T flow-matching action head.

    This class inherits ``UamVLAOFT`` only for UAM helper methods.  Its
    constructor calls ``baseframework.__init__`` directly and then mirrors
    ``Qwen_GR00T.__init__`` so no OFT modules are constructed.
    """

    def __init__(self, config) -> None:
        self._init_gr00t_components(config)

        self._init_uamvla_sidecars()
        self.aux_heads = nn.ModuleDict()
        self._maybe_build_aux_heads()
        if hasattr(self, "_maybe_build_aux_loss_control"):
            self._maybe_build_aux_loss_control()

    def _init_gr00t_components(self, config) -> None:
        """Initialize Qwen-VL and the GR00T flow-matching action head.

        Keep this aligned with ``Qwen_GR00T.__init__``:
          1. merge ``QwenGR00TDefaultConfig`` with YAML,
          2. build Qwen-VL,
          3. set DiT cross-attention dim from loaded VLM hidden size,
          4. build ``FlowmatchingActionHead``,
          5. cache ``action_horizon``.
        """
        baseframework.__init__(self)
        self.config = merge_framework_config(QwenGR00TDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _init_uamvla_sidecars(self) -> None:
        """Initialize the UAMVLA sidecar state used by inherited helpers."""
        self._init_uamvla_sidecar_roots()
        self._init_lerobot_video_path_config()

        self._image_target_cache: "OrderedDict[tuple[str, int, int], torch.Tensor]" = OrderedDict()
        self._image_target_cache_maxsize = int(
            self.config.datasets.vla_data.get("image_target_cache_maxsize", 256)
        )

        self.aux_state_slice = self.config.datasets.vla_data.get(
            "aux_state_slice",
            {
                "target_pose_rot6d": [15, 21],
                "target_pose_trans": [21, 24],
                "static_cam_rot6d": [24, 30],
                "static_cam_trans": [30, 33],
            },
        )

    def _prepare_examples(self, examples: List[dict]) -> List[dict]:
        """Unpack LeRobot samples when sidecar keys are present."""
        return [
            self._unpack_lerobot_sample(example)
            if "__trajectory_id" in example
            else example
            for example in examples
        ]

    def _unpack_lerobot_sample(self, sample: dict) -> dict:
        """Unpack sidecar sample and expose CALVIN ``robot_obs[:7]`` to GR00T.

        UAMVLA packs CALVIN state as ``robot_obs(15) + aux pose/camera(18)``.
        For GR00T's 7-D state branch we keep the action-aligned proprio state:
        TCP position, TCP Euler orientation, and gripper opening width.
        """
        out = UamVLAOFT._unpack_lerobot_sample(self, sample)
        if "state" in sample:
            out["state"] = self._extract_gr00t_state_from_sample_state(sample["state"])
        return out

    @staticmethod
    def _cfg_get(container, key: str, default=None):
        if container is None:
            return default
        if hasattr(container, "get"):
            try:
                return container.get(key, default)
            except TypeError:
                pass
        return getattr(container, key, default)

    def _configured_gr00t_state_indices(self) -> list[int] | None:
        """Return optional packed-state indices for the GR00T state branch.

        Default UAM-CALVIN 33D state uses ``robot_obs[:7]``.  StarVLA-first
        CALVIN state is ``[x,y,z,roll,pitch,yaw,pad,gripper,...]`` and should
        pass ``[0,1,2,3,4,5,7]`` so gripper is not replaced by the padding dim.
        """
        datasets_cfg = self._cfg_get(getattr(self, "config", None), "datasets", None)
        vla_cfg = self._cfg_get(datasets_cfg, "vla_data", None)
        indices = self._cfg_get(vla_cfg, "gr00t_state_indices", None)
        if indices is None:
            framework_cfg = self._cfg_get(getattr(self, "config", None), "framework", None)
            action_cfg = self._cfg_get(framework_cfg, "action_model", None)
            indices = self._cfg_get(action_cfg, "gr00t_state_indices", None)
        if indices is None:
            return None
        return [int(i) for i in indices]

    def _extract_gr00t_state_from_sample_state(self, state) -> torch.Tensor:
        indices = self._configured_gr00t_state_indices()
        if indices is None:
            return self._extract_gr00t_state_from_packed_calvin_state(state)
        return self._extract_gr00t_state_by_indices(state, indices)

    @staticmethod
    def _extract_gr00t_state_from_packed_calvin_state(state) -> torch.Tensor:
        """Return ``robot_obs[:7]`` as ``(1, 7)`` float tensor for GR00T."""
        return UamVLAGR00T._extract_gr00t_state_by_indices(state, list(range(7)))

    @staticmethod
    def _extract_gr00t_state_by_indices(state, indices: list[int]) -> torch.Tensor:
        """Return selected packed-state dims as ``(1, len(indices))`` tensor."""
        if not torch.is_tensor(state):
            state = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        else:
            state = state.to(dtype=torch.float32)

        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)
        elif state.ndim > 1:
            state = state.reshape(-1, state.shape[-1])[0]

        if not indices:
            raise RuntimeError(
                "gr00t_state_indices must contain at least one index."
            )
        max_index = max(indices)
        if state.shape[-1] <= max_index:
            raise RuntimeError(
                f"Expected packed CALVIN state with at least {max_index + 1} dims, "
                f"got shape {tuple(state.shape)}."
            )
        return state[..., indices].reshape(1, len(indices))

    def _state_batch_or_none(self, examples: List[dict], device, dtype):
        """Return raw proprio state only when the GR00T state encoder is enabled."""
        state_dim = int(self.config.framework.action_model.get("state_dim", 0) or 0)
        if state_dim <= 0 or "state" not in examples[0]:
            return None
        state = [example["state"] for example in examples]
        return torch.as_tensor(np.asarray(state), device=device, dtype=dtype)

    def _assert_image_token_count(self, input_ids: torch.Tensor, examples: List[dict]) -> None:
        """Keep UAMVLA's configured Qwen3-VL image-token invariant."""
        if hasattr(self.qwen_vl_interface, "image_token_id"):
            image_token_id = self.qwen_vl_interface.image_token_id
        else:
            image_token_id = self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        patches_per_view = getattr(self, "_qwen_patches_per_view", lambda: 400)()
        num_views = len(examples[0]["image"])
        expected = patches_per_view * num_views
        counts = (input_ids == image_token_id).sum(dim=1)
        if not (counts == expected).all():
            raise RuntimeError(
                f"image_pad token count mismatch: expected {expected} per sample, "
                f"got {counts.tolist()}. Check _force_resize_640 + Qwen3VLProcessor."
            )

    def _encoder_attention_mask(self, qwen_inputs: dict) -> torch.Tensor | None:
        """Bool KV mask for the action DiT's cross-attention over ``vl_embs``.

        The Qwen tokenizer right-pads, and the DiT cross-attends to the whole
        sequence, so without this mask it reads padding hidden states as valid
        context.  ``attn1`` forwards the mask to ``scaled_dot_product_attention``
        unchanged: the raw int64 0/1 mask raises, only bool (or additive -inf)
        is honoured.

        Off by default: enabling it shifts every UamGR00T-family baseline, so a
        run with it on must not be compared against numbers produced without it.
        """
        if not bool(self.config.framework.action_model.get("use_encoder_mask", False)):
            return None
        mask = qwen_inputs.get("attention_mask")
        return None if mask is None else mask.bool()

    def _encode_qwen_hidden(self, examples: List[dict]):
        """Build Qwen inputs without action-token prompt and return last hidden."""
        batch_images = [self._force_resize_640(example["image"]) for example in examples]
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
        self._assert_image_token_count(qwen_inputs["input_ids"], examples)
        return examples, qwen_inputs, hidden

    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Training forward: GR00T flow-matching action loss plus aux losses."""
        examples = self._prepare_examples(examples)
        examples, qwen_inputs, hidden = self._encode_qwen_hidden(examples)
        gt_actions = [example["action"] for example in examples]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.as_tensor(
                np.asarray([
                    action.detach().float().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
                    for action in gt_actions
                ]),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            actions_target = actions[:, -self.action_horizon:, :]
            if actions_target.shape[1] != self.action_horizon:
                raise RuntimeError(
                    f"Expected at least {self.action_horizon} action steps, "
                    f"got {actions.shape[1]}. Check data_config action_horizon."
                )

            repeated_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            actions_repeated = actions_target.repeat(repeated_steps, 1, 1)
            hidden_repeated = hidden.repeat(repeated_steps, 1, 1)

            state = self._state_batch_or_none(examples, hidden.device, hidden.dtype)
            state_repeated = state.repeat(repeated_steps, 1, 1) if state is not None else None

            encoder_mask = self._encoder_attention_mask(qwen_inputs)
            total = self.action_model(
                hidden_repeated, actions_repeated, state_repeated,
                encoder_attention_mask=(
                    encoder_mask.repeat(repeated_steps, 1) if encoder_mask is not None else None
                ),
            )

        log_metrics = {"action_loss_fm": total.detach()}

        batch_dict = self._collate_aux(examples, qwen_inputs)
        assert "input_ids" in batch_dict, "future/recon heads require input_ids"
        if hasattr(self, "_compute_aux_training_losses"):
            total, aux_metrics = self._compute_aux_training_losses(
                total,
                hidden,
                batch_dict,
                global_step=UamVLAOFT._global_step_from_kwargs(kwargs),
            )
            log_metrics.update(aux_metrics)
        elif hasattr(self, "aux_suite"):
            masks = {
                name: self._resolve_head_mask(name, batch_dict, hidden.shape[0], hidden.device)
                for name in self.aux_heads
            }
            aux_loss, aux_metrics = self.aux_suite(
                action_loss=total,
                hidden_states=hidden,
                batch=batch_dict,
                masks=masks,
                global_step=int(kwargs.get("global_step", 0) or 0),
            )
            total = total + aux_loss
            log_metrics.update(aux_metrics)
        else:
            for name, head in self.aux_heads.items():
                mask = self._resolve_head_mask(name, batch_dict, hidden.shape[0], hidden.device)
                out = head.compute_loss(hidden, batch_dict, mask=mask)
                if out.loss is not None:
                    total = total + out.loss
                    log_metrics[f"{name}_loss_weighted"] = out.loss.detach()
                for metric_name, metric_value in out.metrics.items():
                    log_metrics[self._aux_metric_log_key(name, metric_name)] = metric_value

        return {"action_loss": total, **log_metrics}

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        """Inference path using GR00T flow-matching sampling."""
        if not isinstance(examples, list):
            examples = [examples]

        examples = self._prepare_examples(examples)
        examples, qwen_inputs, hidden = self._encode_qwen_hidden(examples)
        state = self._state_batch_or_none(examples, hidden.device, hidden.dtype)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                hidden, state,
                encoder_attention_mask=self._encoder_attention_mask(qwen_inputs),
            )

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def _visualization_forward_context(
        self,
        selected: list[dict],
    ) -> tuple[list[dict], np.ndarray, torch.Tensor, dict]:
        """Run one GR00T inference pass and build shared aux visualization inputs."""
        examples = self._prepare_examples(selected)
        examples, qwen_inputs, hidden = self._encode_qwen_hidden(examples)
        state = self._state_batch_or_none(examples, hidden.device, hidden.dtype)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(hidden, state)

        batch_dict = self._collate_aux(examples, qwen_inputs)
        batch_dict["image"] = self._make_visualization_image_batch(
            examples,
            device=qwen_inputs["input_ids"].device,
        )
        batch_dict["instruction"] = [example["lang"] for example in examples]

        return examples, pred_actions.detach().cpu().numpy(), hidden, batch_dict

    @torch.inference_mode()
    def visualize_batch(
        self,
        batch: List[dict],
        n_samples: int = 1,
        distributed_all_ranks: bool = False,
    ) -> dict:
        """Visualize action predictions plus enabled aux heads under GR00T keys."""
        if not isinstance(batch, list):
            batch = [batch]
        limit = min(max(int(n_samples), 0), len(batch))
        if limit == 0:
            return {}

        selected = batch[:limit]
        was_training = bool(getattr(self, "training", False))
        self.eval()
        try:
            examples, pred_actions, hidden_states, batch_dict = self._visualization_forward_context(selected)
            if pred_actions.ndim == 2:
                pred_actions = pred_actions[None, ...]

            try:
                import wandb
            except ImportError:
                wandb = None

            outputs = {}
            for idx, sample in enumerate(examples):
                pred = pred_actions[idx]
                horizon = int(getattr(self, "action_horizon", pred.shape[0]))

                gt_action = None
                if "action" in sample:
                    gt_action = self._to_action_numpy(sample["action"])
                    if gt_action.ndim >= 2:
                        gt_action = gt_action[-horizon:, : pred.shape[-1]]

                images = sample.get("image", [])
                if isinstance(images, Image.Image) or not isinstance(images, (list, tuple)):
                    images = [images]

                canvas = self._make_visualization_canvas(
                    images=list(images),
                    pred_action=pred,
                    gt_action=gt_action,
                )
                caption = str(sample.get("lang", ""))
                outputs[f"viz/uamvla_gr00t/action_{idx}"] = (
                    wandb.Image(canvas, caption=caption) if wandb is not None else canvas
                )

            return self._collect_aux_head_visualizations_gr00t(
                hidden_states,
                batch_dict,
                num_samples=limit,
                outputs=outputs,
                distributed_all_ranks=distributed_all_ranks,
            )
        finally:
            if was_training:
                self.train()

    def _collect_aux_head_visualizations_gr00t(
        self,
        hidden_states: torch.Tensor,
        batch_dict: dict,
        num_samples: int,
        outputs: dict,
        distributed_all_ranks: bool = False,
    ) -> dict:
        """Append enabled aux-head visualizations under ``viz/uamvla_gr00t``."""
        if not self.aux_heads:
            return outputs

        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        profile = os.environ.get("UAMVLA_AUX_PROFILE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0") or 0)
        for name, head in self.aux_heads.items():
            if not hasattr(head, "visualize"):
                continue
            try:
                mask = self._resolve_head_mask(name, batch_dict, batch_size, device)
                valid_count = int(mask.to(dtype=torch.bool).sum().item())
                if distributed_all_ranks and dist.is_initialized():
                    valid_any = torch.tensor([int(valid_count > 0)], device=device, dtype=torch.int32)
                    dist.all_reduce(valid_any, op=dist.ReduceOp.MIN)
                    if int(valid_any.item()) == 0:
                        if profile:
                            print(
                                f"[uamvla-aux-profile][rank{rank}] "
                                f"step=viz head={name} skip valid={valid_count}/{batch_size}",
                                flush=True,
                            )
                        continue
                if profile:
                    print(
                        f"[uamvla-aux-profile][rank{rank}] "
                        f"step=viz head={name} start valid={valid_count}/{batch_size}",
                        flush=True,
                    )
                    t_start = time.perf_counter()
                else:
                    t_start = None
                kwargs = {}
                if name == "pose":
                    kwargs["camera_params"] = getattr(head, "camera_params", None)
                images = head.visualize(
                    hidden_states,
                    batch_dict,
                    mask=mask,
                    num_samples=num_samples,
                    **kwargs,
                )
                for idx, image in enumerate(images or []):
                    outputs[f"viz/uamvla_gr00t/{name}_{idx}"] = image
                if profile:
                    print(
                        f"[uamvla-aux-profile][rank{rank}] "
                        f"step=viz head={name} done elapsed={time.perf_counter() - t_start:.3f}s",
                        flush=True,
                    )
            except Exception as exc:
                logger.warning("UamVLAGR00T %s visualization failed: %s", name, exc)
        return outputs
