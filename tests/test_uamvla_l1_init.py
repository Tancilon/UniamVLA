"""L1 smoke (Mac local): UamVLA framework init succeeds with mocked backbone.

Full backbone load happens only on remote with checkpoint available.
"""
import pytest
from unittest.mock import MagicMock


def test_uamvla_init_with_full_aux_heads_mocked(monkeypatch):
    pytest.importorskip("torch")
    pytest.skip("L1 init with all aux heads requires VAE ckpt — run on remote in WP8 L1 task")


def test_uamvla_init_action_head_only_local(monkeypatch):
    """Local-runnable subset: instantiate UamVLA with action_head only, mocked backbone.

    Full init (with pose/future/recon + VAE) deferred to L1 remote test.
    """
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf
    import starVLA.model.framework.VLM4A.UamVLA as uamvla_mod
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA

    fake_backbone = MagicMock()
    fake_backbone.config.hidden_size = 4096
    fake_backbone.image_token_id = 0
    backbone_params = [nn.Parameter(torch.randn(2))]
    state_encoder_params = [nn.Parameter(torch.randn(2))]
    fake_backbone.parameters.side_effect = lambda: iter(backbone_params + state_encoder_params)
    fake_backbone.state_encoder.parameters.side_effect = lambda: iter(state_encoder_params)
    monkeypatch.setattr(uamvla_mod, "build_uamvla_backbone", lambda cfg: fake_backbone)

    cfg = OmegaConf.create({
        "framework": {
            "aux_heads": {
                "action": {"enabled": True, "lr": 1e-4, "loss_weight": 1.0},
                "pose":   {"enabled": False},
                "future": {"enabled": False},
                "recon":  {"enabled": False},
            }
        }
    })
    model = UamVLA(config=cfg)
    assert hasattr(model, "qwen_vl_interface")
    assert hasattr(model, "aux_heads")
    assert "action" in model.aux_heads
    assert "pose" not in model.aux_heads
    assert "future" not in model.aux_heads
    assert "recon" not in model.aux_heads
    assert model.vae is None  # no future/recon → no VAE construction
    assert model.action_horizon == 8  # future_action_window_size=7 + 1
