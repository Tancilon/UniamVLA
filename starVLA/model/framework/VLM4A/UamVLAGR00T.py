"""UamVLAGR00T framework: QwenGR00T + UAMVLA perception aux heads.

This variant keeps the UAMVLA sidecar loading and optional pose/future/recon
auxiliary heads from ``UamVLAOFT``, but replaces OFT's action-token MLP path
with the GR00T flow-matching action head.  No action placeholder tokens are
added to the language prompt; the action head conditions directly on the last
Qwen-VL hidden state.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.framework.VLM4A.UamVLAOFT import UamVLAOFT
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)


@FRAMEWORK_REGISTRY.register("UamVLAGR00T")
class UamVLAGR00T(Qwen_GR00T, UamVLAOFT):
    """UAMVLA framework using the GR00T flow-matching action head.

    The multiple inheritance is intentionally shallow: ``Qwen_GR00T`` owns the
    VLM and action head construction, while ``UamVLAOFT`` contributes the
    dataset sidecar helpers, aux-head builders, and visualization utilities.
    This class overrides the action forward/inference paths so the OFT
    action-token prompt is never used.
    """

    def __init__(self, config) -> None:
        self._init_gr00t_components(config)

        self._init_uamvla_sidecars()
        self.aux_heads = nn.ModuleDict()
        self._maybe_build_aux_heads()

    def _init_gr00t_components(self, config) -> None:
        """Initialize Qwen-VL and the GR00T flow-matching action head.

        This intentionally mirrors ``Qwen_GR00T.__init__`` instead of calling
        it directly.  ``Qwen_GR00T.__init__`` uses ``super().__init__()``, and
        under this class' multiple-inheritance MRO that would enter
        ``UamVLAOFT.__init__`` without the required config argument.
        """
        from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
        from starVLA.model.framework.base_framework import baseframework
        from starVLA.model.framework.share_tools import merge_framework_config
        from starVLA.model.modules.action_model.GR00T_ActionHeader import (
            get_action_model as get_gr00t_action_model,
        )
        from starVLA.model.modules.vlm import get_vlm_model

        baseframework.__init__(self)
        self.config = merge_framework_config(QwenGR00TDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        self.action_model = get_gr00t_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _init_uamvla_sidecars(self) -> None:
        """Initialize the UAMVLA sidecar state used by inherited helpers."""
        from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

        mixture = DATASET_NAMED_MIXTURES[self.config.datasets.vla_data.data_mix]
        dataset_name = mixture[0][0]
        self.sidecar_root = Path(self.config.datasets.vla_data.data_root_dir) / dataset_name
        self._init_lerobot_video_path_config()

        self._image_target_cache: "OrderedDict[tuple[int, int], torch.Tensor]" = OrderedDict()
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
            out["state"] = self._extract_gr00t_state_from_packed_calvin_state(sample["state"])
        return out

    @staticmethod
    def _extract_gr00t_state_from_packed_calvin_state(state) -> torch.Tensor:
        """Return ``robot_obs[:7]`` as ``(1, 7)`` float tensor for GR00T."""
        if not torch.is_tensor(state):
            state = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        else:
            state = state.to(dtype=torch.float32)

        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)
        elif state.ndim > 1:
            state = state.reshape(-1, state.shape[-1])[0]

        if state.shape[-1] < 7:
            raise RuntimeError(
                f"Expected packed CALVIN state with at least 7 dims, got shape {tuple(state.shape)}."
            )
        return state[..., :7].reshape(1, 7)

    def _state_batch_or_none(self, examples: List[dict], device, dtype):
        """Return raw proprio state only when the GR00T state encoder is enabled."""
        state_dim = int(self.config.framework.action_model.get("state_dim", 0) or 0)
        if state_dim <= 0 or "state" not in examples[0]:
            return None
        state = [example["state"] for example in examples]
        return torch.as_tensor(np.asarray(state), device=device, dtype=dtype)

    def _assert_image_token_count(self, input_ids: torch.Tensor, examples: List[dict]) -> None:
        """Keep UAMVLA's fixed 640x640 Qwen3-VL image-token invariant."""
        if hasattr(self.qwen_vl_interface, "image_token_id"):
            image_token_id = self.qwen_vl_interface.image_token_id
        else:
            image_token_id = self.qwen_vl_interface.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        patches_per_view = 400
        num_views = len(examples[0]["image"])
        expected = patches_per_view * num_views
        counts = (input_ids == image_token_id).sum(dim=1)
        if not (counts == expected).all():
            raise RuntimeError(
                f"image_pad token count mismatch: expected {expected} per sample, "
                f"got {counts.tolist()}. Check _force_resize_640 + Qwen3VLProcessor."
            )

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

            total = self.action_model(hidden_repeated, actions_repeated, state_repeated)

        log_metrics = {"action_loss_fm": total.detach()}

        batch_dict = self._collate_aux(examples, qwen_inputs)
        assert "input_ids" in batch_dict, "future/recon heads require input_ids"
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
        examples, _qwen_inputs, hidden = self._encode_qwen_hidden(examples)
        state = self._state_batch_or_none(examples, hidden.device, hidden.dtype)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(hidden, state)

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
    def visualize_batch(self, batch: List[dict], n_samples: int = 1) -> dict:
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
    ) -> dict:
        """Append enabled aux-head visualizations under ``viz/uamvla_gr00t``."""
        if not self.aux_heads:
            return outputs

        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        for name, head in self.aux_heads.items():
            if not hasattr(head, "visualize"):
                continue
            try:
                mask = self._resolve_head_mask(name, batch_dict, batch_size, device)
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
            except Exception as exc:
                logger.warning("UamVLAGR00T %s visualization failed: %s", name, exc)
        return outputs
