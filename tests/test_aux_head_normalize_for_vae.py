from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from starVLA.model.modules.uamvla.aux_heads.future_head import FutureHead
from starVLA.model.modules.uamvla.aux_heads.recon_head import ReconHead


def _head_with_cpu_image_buffers(head_cls):
    head = head_cls.__new__(head_cls)
    nn.Module.__init__(head)
    head.register_buffer(
        "image_mean",
        torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1),
        persistent=False,
    )
    head.register_buffer(
        "image_std",
        torch.tensor([0.25, 0.25, 0.25]).view(1, -1, 1, 1),
        persistent=False,
    )
    return head


@pytest.mark.parametrize("head_cls", [ReconHead, FutureHead])
def test_normalize_for_vae_uses_image_tensor_device(head_cls):
    head = _head_with_cpu_image_buffers(head_cls)
    images = torch.ones((1, 3, 2, 2), device="meta")

    normalized = head._normalize_for_vae(images)

    assert normalized.device == images.device
