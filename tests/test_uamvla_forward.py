"""Tests for Task 18: UamVLA.forward (multi-aux-head loss summing).

The full forward path requires a real Qwen3-VL backbone (large checkpoint
download) and a working dataloader fixture (Task 24+). Both prerequisites
are out of scope for this test file — the L2 1-step training smoke
(Task 31) exercises the full pipeline.

Pure-tensor collator helpers used by ``forward`` are covered separately
in ``tests/test_collator_helpers.py``.
"""
import pytest


def test_forward_returns_action_loss_key():
    """Full forward test requires backbone + dataloader fixtures — defer to L2 1-step training smoke (Task 31)."""
    pytest.skip("Full forward test requires backbone + dataloader fixtures — defer to L2 1-step training smoke")


def test_forward_moves_processor_tensors_to_model_device(monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    import torch.nn as nn
    from omegaconf import OmegaConf

    import starVLA.model.framework.VLM4A.UamVLA as uamvla_mod
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA

    class _FakeTokenizer:
        pad_token_id = 0
        unk_token_id = None

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

    class _MetaBackbone(nn.Module):
        hidden_size = 4
        image_token_id = 1

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.empty((), device="meta"))
            self.lm_head = nn.Linear(self.hidden_size, 1024, device="meta")
            self.config = SimpleNamespace(hidden_size=self.hidden_size)
            self.tokenizer = _FakeTokenizer()
            self.seen_devices = None

        def get_tokenizer(self):
            return self.tokenizer

        def resize_token_embeddings(self, new_size):
            return None

        def get_lm_head(self):
            return self.lm_head

        def build_inputs(self, images, instructions, canonical_state=None):
            return {
                "input_ids": torch.tensor([[5, 6, 7]], dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "pixel_values": torch.zeros(1, 3, 2, 2),
                "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
                "canonical_state": canonical_state,
            }

        def forward(self, **kwargs):
            self.seen_devices = {
                "input_ids": kwargs["input_ids"].device.type,
                "attention_mask": kwargs["attention_mask"].device.type,
                "canonical_state": kwargs["canonical_state"]["arm_0"]["ee_pose"].device.type,
            }
            assert self.seen_devices == {
                "input_ids": "meta",
                "attention_mask": "meta",
                "canonical_state": "meta",
            }
            B, L = kwargs["input_ids"].shape
            hidden = torch.empty(B, L, self.hidden_size, device="meta")
            return SimpleNamespace(hidden_states=[hidden])

    fake_backbone = _MetaBackbone()
    monkeypatch.setattr(uamvla_mod, "build_uamvla_backbone", lambda cfg: fake_backbone)

    cfg = OmegaConf.create({
        "framework": {
            "action_model": {"future_action_window_size": 0, "num_bins": 256, "action_dim": 7},
            "embodiment": {"name": "franka_calvin", "action_dim": 7},
            "aux_heads": {
                "action": {"enabled": False},
                "pose": {"enabled": False},
                "future": {"enabled": False},
                "recon": {"enabled": False},
            },
        }
    })
    model = UamVLA(config=cfg)

    examples = [{
        "image": [object()],
        "lang": "pick up the block",
        "canonical_state": {
            "arm_0": {
                "ee_pose": torch.zeros(6),
                "joint_pos": torch.zeros(7),
            },
            "gripper_0": torch.zeros(1),
        },
        "action": torch.zeros(1, 7),
        "action_mask": torch.ones(1, 7, dtype=torch.long),
    }]

    out = model.forward(examples)

    assert "action_loss" in out
    assert fake_backbone.seen_devices["input_ids"] == "meta"
