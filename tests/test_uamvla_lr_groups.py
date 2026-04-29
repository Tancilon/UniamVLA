import torch
import torch.nn as nn
from unittest.mock import MagicMock
from omegaconf import OmegaConf


def test_get_lr_groups_returns_expected_names(monkeypatch):
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA
    import starVLA.model.framework.VLM4A.UamVLA as uamvla_mod

    backbone_params = [nn.Parameter(torch.randn(2))]
    state_encoder_params = [nn.Parameter(torch.randn(2))]

    fake_backbone = MagicMock()
    fake_backbone.config.hidden_size = 4096
    # Return fresh iter on each call so multiple .parameters() calls work
    fake_backbone.parameters.side_effect = lambda: iter(backbone_params + state_encoder_params)
    fake_backbone.state_encoder.parameters.side_effect = lambda: iter(state_encoder_params)
    fake_backbone.image_token_id = 0
    # Patch at the UamVLA module level — that's where the local name is bound
    monkeypatch.setattr(uamvla_mod, "build_uamvla_backbone", lambda cfg: fake_backbone)

    # Use OmegaConf so merge_framework_config can set cfg.framework properly
    cfg = OmegaConf.create({"framework": {"aux_heads": {
        "action": {"enabled": True, "lr": 1e-4},
        "pose":   {"enabled": False},
        "future": {"enabled": False},
        "recon":  {"enabled": False},
    }}})
    model = UamVLA(config=cfg)
    lr_cfg = MagicMock()
    lr_cfg.qwen_vl_interface = 2e-5
    lr_cfg.state_encoder = 1e-4
    lr_cfg.base = 1e-4
    groups = model.get_lr_groups(lr_cfg)
    names = {g["name"] for g in groups}
    assert "qwen_vl_interface" in names
    assert "state_encoder" in names
    assert "aux_head_action" in names


def test_supports_training_tag_vla_only():
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA
    assert hasattr(UamVLA, "supports_training_tag")
