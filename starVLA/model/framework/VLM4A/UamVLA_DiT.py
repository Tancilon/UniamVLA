import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.DiTActionHeader import get_action_model
from starVLA.model.modules.uamvla.components.deepstack_history_fusion import DeepStackHistoryFusion
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY

DIT_CONDITION_DIMS = {"DiT-S": 384, "DiT-B": 768, "DiT-L": 1024}


@FRAMEWORK_REGISTRY.register("UamVLA_DiT")
class UamVLA_DiT(baseframework):
    def __init__(self, config):
        super().__init__()
        self.config = config
        data = config.datasets.vla_data
        self.num_history_frames = data.num_history_frames
        self.history_interval = data.history_interval
        if any(type(value) is not int or value <= 0 for value in (self.num_history_frames, self.history_interval)):
            raise ValueError("num_history_frames and history_interval must be positive integers")
        action = config.framework.action_model
        action.action_hidden_dim = DIT_CONDITION_DIMS[action.action_model_type]
        self.action_horizon = int(action.action_horizon)
        action.n_condition_token = self.action_horizon
        self.repeated_diffusion_steps = int(action.get("repeated_diffusion_steps", 4))
        self.cfg_scale = float(action.get("cfg_scale", 1.5))
        self.num_inference_timesteps = int(action.num_inference_timesteps)
        obs_size = list(config.framework.obs_image_size)
        if len(obs_size) != 2 or int(obs_size[0]) <= 0 or int(obs_size[0]) != int(obs_size[1]):
            raise ValueError("obs_image_size must specify a positive square image size")
        self.image_size = int(obs_size[0])
        self.qwen_image_size = int(config.framework.qwen_image_size)

        self.qwen_vl_interface = get_vlm_model(config=config)
        backbone = self.qwen_vl_interface.model.model
        vision_stride = int(backbone.config.vision_config.patch_size) * int(backbone.visual.spatial_merge_size)
        self.patches_per_view = (self.qwen_image_size // vision_stride) ** 2
        dim = backbone.config.text_config.hidden_size
        self.history_fusion = DeepStackHistoryFusion(dim)
        self.action_condition_projector = torch.nn.Linear(dim, action.action_hidden_dim)
        self.action_model = get_action_model(config=config)
        self.action_token = "🔍"
        ids = self.qwen_vl_interface.processor.tokenizer(
            self.action_token, add_special_tokens=False,
        )["input_ids"]
        if len(ids) != 1:
            raise ValueError("The action placeholder must tokenize to exactly one token")
        self.action_token_id = ids[0]
        self.action_prompt_suffix = (
            f" Please predict the next {self.action_horizon} robot actions: <action>"
            + self.action_token * self.action_horizon + "<action>."
        )

    def _prepare_examples(self, examples):
        if not isinstance(examples, list):
            examples = [examples]
        prepared = []
        for sample in examples:
            history = sample["image_history"]
            if len(history) != self.num_history_frames or len(sample["image"]) != 2 or any(len(frame) != 2 for frame in history):
                raise ValueError(f"Expected two current views and {self.num_history_frames} pairs of historical views")
            step = sample["__base_index"] if "__trajectory_id" in sample else sample["step"]
            prepared.append({
                **sample,
                "image": [self._prepare_image(im) for im in sample["image"]],
                "image_history": [[self._prepare_image(im) for im in frame] for frame in history],
                "step": int(step),
            })
        return prepared

    def _prepare_image(self, image):
        # Prepare observations at their input size. The VLM processor alone
        # resizes them to qwen_image_size before patch extraction.
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        image = image.convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        return image

    def _encode_qwen_hidden(self, examples):
        interface = self.qwen_vl_interface
        backbone = interface.model.model
        inputs = interface.build_qwenvl_inputs(
            images=[sample["image"] for sample in examples],
            instructions=[sample["lang"] + self.action_prompt_suffix for sample in examples],
        )
        device = inputs["input_ids"].device
        history_images = [im for sample in examples for frame in sample["image_history"] for im in frame]
        history_inputs = interface.processor.image_processor(
            images=history_images,
            return_tensors="pt",
        ).to(device)
        b = len(examples)
        n = self.patches_per_view
        for grid, count in ((inputs["image_grid_thw"], b * 2), (history_inputs["image_grid_thw"], b * self.num_history_frames * 2)):
            tokens = grid.prod(-1) // backbone.visual.spatial_merge_size**2
            if len(grid) != count or not torch.all(tokens == n):
                raise ValueError("Current/history image grids must match the configured image size")

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            image_features, current_levels = backbone.get_image_features(
                inputs["pixel_values"],
                inputs["image_grid_thw"],
            )
            _, history_levels = backbone.get_image_features(
                history_inputs["pixel_values"],
                history_inputs["image_grid_thw"],
            )
            current = torch.stack([x.reshape(b, 2, n, -1) for x in current_levels], dim=1)
            history = torch.stack([x.reshape(b, self.num_history_frames, 2, n, -1) for x in history_levels], dim=1)
            fused = self.history_fusion(history, current, [sample["step"] for sample in examples])

            embeds = backbone.get_input_embeddings()(inputs["input_ids"])
            image_features = torch.cat(image_features).to(embeds.dtype)
            image_mask, _ = backbone.get_placeholder_mask(
                inputs["input_ids"],
                inputs_embeds=embeds,
                image_features=image_features,
            )
            embeds = embeds.masked_scatter(image_mask, image_features)
            visual_mask = image_mask[..., 0]
            positions, _ = backbone.get_rope_index(
                inputs["input_ids"],
                inputs["image_grid_thw"],
                attention_mask=inputs["attention_mask"],
            )
            out = backbone.language_model(
                inputs_embeds=embeds,
                attention_mask=inputs["attention_mask"],
                position_ids=positions,
                visual_pos_masks=visual_mask,
                deepstack_visual_embeds=[fused[:, i].reshape(-1, fused.shape[-1]) for i in range(3)],
                use_cache=False,
                return_dict=True,
            )
        return examples, inputs, out.last_hidden_state

    def _action_condition(self, inputs, hidden):
        # select the last eight placeholders in sequence order.
        mask = (inputs["input_ids"] == self.action_token_id) & inputs["attention_mask"].bool()
        if (mask.sum(dim=1) < self.action_horizon).any():
            raise ValueError("Each sample must contain eight action placeholders")
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        indices = positions.masked_fill(~mask, -1).topk(self.action_horizon, dim=1).values.sort(dim=1).values
        queries = hidden.gather(1, indices[..., None].expand(-1, -1, hidden.shape[-1]))
        return self.action_condition_projector(queries.to(self.action_condition_projector.weight.dtype))

    def forward(self, examples, **kwargs):
        examples, inputs, hidden = self._encode_qwen_hidden(self._prepare_examples(examples))
        with torch.autocast(hidden.device.type, enabled=False):
            condition = self._action_condition(inputs, hidden)
            actions = torch.stack([
                torch.as_tensor(sample["action"], device=condition.device, dtype=condition.dtype)
                for sample in examples
            ])
            if actions.shape[1:] != (8, 7) or not torch.isfinite(actions).all():
                raise ValueError("Expected finite action targets [B,8,7]")
            repeated = self.repeated_diffusion_steps
            noise_pred, noise, _ = self.action_model(
                actions.repeat(repeated, 1, 1), condition.repeat(repeated, 1, 1),
            )
            loss = self.action_model.loss(noise_pred.float(), noise.float())
        return {"action_loss": loss, "action_loss_diffusion": loss.detach()}

    @torch.inference_mode()
    def predict_action(self, examples, *, cfg_scale=None, use_ddim=True, num_ddim_steps=None, **kwargs):
        _, inputs, hidden = self._encode_qwen_hidden(self._prepare_examples(examples))
        with torch.autocast(hidden.device.type, enabled=False):
            condition = self._action_condition(inputs, hidden)
            actions = self._sample_dit_actions(condition, cfg_scale, use_ddim, num_ddim_steps)
        return {"normalized_actions": actions.float().cpu().numpy()}

    def _sample_dit_actions(self, condition, cfg_scale=None, use_ddim=True, num_ddim_steps=None):
        cfg_scale = float(self.cfg_scale if cfg_scale is None else cfg_scale)
        num_ddim_steps = int(self.num_inference_timesteps if num_ddim_steps is None else num_ddim_steps)
        net = self.action_model.net
        dtype = next(net.parameters()).dtype
        condition = condition.to(dtype)
        batch = condition.shape[0]
        noise = torch.randn(batch, self.action_horizon, 7, device=condition.device, dtype=dtype)
        if cfg_scale > 1:
            noise = torch.cat([noise, noise])
            uncondition = net.z_embedder.uncondition.unsqueeze(0).expand(batch, -1, -1)
            condition = torch.cat([condition, uncondition])

        # Diffusion arithmetic promotes x to FP32; cast at the network boundary
        # so both CFG and unguided inference also work with BF16 serving weights.
        def denoise(x, t, **model_kwargs):
            x = x.to(dtype)
            if cfg_scale > 1:
                return net.forward_with_cfg(x, t, **model_kwargs)
            return net(x, t, **model_kwargs)

        model_kwargs = {"z": condition}
        if cfg_scale > 1:
            model_kwargs["cfg_scale"] = cfg_scale
        if use_ddim:
            if (self.action_model.ddim_diffusion is None
                    or self.action_model.ddim_diffusion.num_timesteps != num_ddim_steps):
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                denoise, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=condition.device, eta=0.0,
            )
        else:
            samples = self.action_model.diffusion.p_sample_loop(
                denoise, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=condition.device,
            )
        return samples[:batch]
