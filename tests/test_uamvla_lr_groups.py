import torch
import torch.nn as nn
from unittest.mock import MagicMock
from omegaconf import OmegaConf


class _FakeTokenizer:
    unk_token_id = 0

    def add_tokens(self, tokens):
        return len(tokens)

    def __len__(self):
        return 1024

    def convert_tokens_to_ids(self, token):
        if token == "<|action_start|>":
            return 100
        if isinstance(token, str) and token.startswith("<ACT_"):
            return 200 + int(token[len("<ACT_"):-1])
        return 1


class _BackboneWithSharedLmHead(nn.Module):
    hidden_size = 4
    image_token_id = 1

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.state_encoder = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 1024)
        self.config = MagicMock()
        self.config.hidden_size = self.hidden_size
        self._tokenizer = _FakeTokenizer()

    def get_tokenizer(self):
        return self._tokenizer

    def resize_token_embeddings(self, new_size):
        return None

    def get_lm_head(self):
        return self.lm_head


def test_get_lr_groups_returns_expected_names(monkeypatch):
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA
    import starVLA.model.framework.VLM4A.UamVLA as uamvla_mod

    fake_backbone = _BackboneWithSharedLmHead()
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


def test_get_lr_groups_deduplicates_shared_lm_head(monkeypatch):
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA
    import starVLA.model.framework.VLM4A.UamVLA as uamvla_mod

    fake_backbone = _BackboneWithSharedLmHead()
    monkeypatch.setattr(uamvla_mod, "build_uamvla_backbone", lambda cfg: fake_backbone)

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
    param_ids = [id(p) for group in groups for p in group["params"]]

    assert len(param_ids) == len(set(param_ids))
    torch.optim.AdamW(groups, lr=lr_cfg.base)


def test_supports_training_tag_vla_only():
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA
    assert hasattr(UamVLA, "supports_training_tag")
