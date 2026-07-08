"""QWen3-VL interface with dedicated h_R token support (ReconVLA §3.2).

Subclasses _QWen3_VL_Interface and overrides build_qwenvl_inputs to accept
an optional text_suffix that is placed AFTER all image tokens in the content
list.  This allows dedicated reconstructive tokens (h_R) to attend to the
full instruction + image context via causal attention, matching the ReconVLA
sequence design:

    [instruction] → [image×N_views] → [recon×N + action×H]
"""
from __future__ import annotations

import torch

from starVLA.model.modules.vlm.QWen3 import (
    IGNORE_INDEX,
    _ACTION_TOKEN_MAX,
    _ACTION_TOKEN_MIN,
    _QWen3_VL_Interface,
)


class _QWen3_VL_hR_Interface(_QWen3_VL_Interface):
    """QWen3-VL with text_suffix support for dedicated h_R / action tokens."""

    def build_qwenvl_inputs(
        self,
        images,
        instructions,
        text_suffix: str | None = None,
        solutions=None,
        **kwargs,
    ):
        """Build inputs with optional text_suffix placed AFTER image tokens.

        Args:
            text_suffix: Text appended after all image content blocks.
                Typically ``recon_token×N + action_suffix``.  When None,
                behaviour is identical to the parent class.
        """
        assert len(images) == len(instructions)
        messages = []
        for imgs, instruction in zip(images, instructions):
            if "CoT_prompt" in self.config.datasets.vla_data:
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content = [{"type": "text", "text": prompt}]
            for img in imgs:
                image_content = {"type": "image", "image": img}
                if self.fixed_image_pixels is not None:
                    image_content["min_pixels"] = self.fixed_image_pixels
                    image_content["max_pixels"] = self.fixed_image_pixels
                content.append(image_content)
            if text_suffix is not None:
                content.append({"type": "text", "text": text_suffix})

            msg = [{"role": "user", "content": content}]
            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        if solutions is not None:
            labels = batch_inputs["input_ids"].clone()
            for i in range(labels.size(0)):
                seq = labels[i]
                mask_seq = (seq >= _ACTION_TOKEN_MIN) & (seq <= _ACTION_TOKEN_MAX)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    seq[: nonzero_indices[0].item()] = IGNORE_INDEX
                else:
                    seq[:] = IGNORE_INDEX
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)
