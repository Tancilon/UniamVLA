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


def test_uamvla_exposes_action_start_id_with_mock_tokenizer(monkeypatch):
    """With a non-int-returning tokenizer, action_start_id falls back to -1 (mock sentinel).

    This verifies the attribute is exposed and is an int, and that the
    ``except (TypeError, ValueError)`` mock-fallback path in ``UamVLA.__init__``
    yields ``-1``. The positive-id case is covered by integration tests where
    a real tokenizer is used.
    """
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf
    import starVLA.model.framework.VLM4A.UamVLA as uamvla_mod
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA

    # A tokenizer whose convert_tokens_to_ids returns a non-int → triggers
    # the TypeError branch in the action_start_id resolution.
    class _NonIntTokenizer:
        unk_token_id = None
        def convert_tokens_to_ids(self, token):
            return object()  # int(object()) raises TypeError
        def add_tokens(self, tokens):
            return 0
        def __len__(self):
            return 1000

    fake_backbone = MagicMock()
    fake_backbone.config.hidden_size = 4096
    fake_backbone.image_token_id = 0
    backbone_params = [nn.Parameter(torch.randn(2))]
    state_encoder_params = [nn.Parameter(torch.randn(2))]
    fake_backbone.parameters.side_effect = lambda: iter(backbone_params + state_encoder_params)
    fake_backbone.state_encoder.parameters.side_effect = lambda: iter(state_encoder_params)
    fake_backbone.get_tokenizer.return_value = _NonIntTokenizer()
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

    assert hasattr(model, "action_start_id")
    assert isinstance(model.action_start_id, int)
    # Non-int tokenizer return ⇒ int() raises TypeError ⇒ fallback to -1
    assert model.action_start_id == -1


def test_uamvla_raises_when_action_start_token_collides_with_unk(monkeypatch):
    """If <|action_start|> isn't registered, tokenizer.convert_tokens_to_ids returns
    unk_token_id. UamVLA.__init__ MUST raise RuntimeError to surface the misconfig
    immediately rather than silently mis-tokenizing at training time.
    """
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf
    import starVLA.model.framework.VLM4A.UamVLA as uamvla_mod
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA

    # A tokenizer that returns the same id for <|action_start|> as its unk_token_id.
    class _FakeTokenizerUnregistered:
        unk_token_id = 99
        def add_tokens(self, tokens):
            return len(tokens)
        def __len__(self):
            return 1000
        def convert_tokens_to_ids(self, token):
            if token == "<|action_start|>":
                return 99  # collision with unk_token_id ⇒ should raise
            if isinstance(token, str) and token.startswith("<ACT_"):
                # Plausible action-token id, distinct per bin.
                idx = int(token[len("<ACT_"):-1])
                return 200 + idx
            return 99

    class _FakeBackbone:
        def __init__(self):
            self.config = MagicMock()
            self.config.hidden_size = 4096
            self.image_token_id = 0
            self.state_encoder = MagicMock()
            self.state_encoder.parameters.side_effect = lambda: iter([nn.Parameter(torch.randn(2))])
        def get_tokenizer(self):
            return _FakeTokenizerUnregistered()
        def resize_token_embeddings(self, n):
            pass
        def parameters(self):
            return iter([nn.Parameter(torch.randn(2))])
        def get_lm_head(self):
            return nn.Linear(4, 4)

    monkeypatch.setattr(uamvla_mod, "build_uamvla_backbone", lambda cfg: _FakeBackbone())

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

    with pytest.raises(RuntimeError, match="register_structural_tokens"):
        UamVLA(config=cfg)
