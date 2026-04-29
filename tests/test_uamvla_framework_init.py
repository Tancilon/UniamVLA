"""Tests for Task 17: UamVLA framework class skeleton.

test_uamvla_class_registered — verifies auto-discovery picks up the new file
    and the @FRAMEWORK_REGISTRY.register decorator fires.

test_uamvla_class_minimal_construct_no_backbone — verifies the import path and
    exercises UamVLADefaultConfig field defaults (cheap, no HF download needed).
"""
import pytest


def test_uamvla_class_registered():
    # Import UamVLA directly — the @FRAMEWORK_REGISTRY.register decorator fires on import.
    # _auto_import_framework_modules() would also work but requires all other VLM4A framework
    # dependencies (qwen_vl_utils, etc.) to be installed; direct import is more targeted.
    import starVLA.model.framework.VLM4A.UamVLA  # noqa: F401 — side-effect: registers class
    from starVLA.model.tools import FRAMEWORK_REGISTRY
    assert "UamVLA" in FRAMEWORK_REGISTRY._registry


def test_uamvla_class_minimal_construct_no_backbone(monkeypatch):
    """Import UamVLA class and verify UamVLADefaultConfig fields — no HF download."""
    from starVLA.model.framework.VLM4A.UamVLA import UamVLA, UamVLADefaultConfig
    pytest.importorskip("torch")

    # Mock build_uamvla_backbone to avoid loading Qwen3-VL
    import starVLA.model.modules.uamvla.backbone_wrapper as bw

    class _FakeBackbone:
        class _FakeConfig:
            hidden_size = 4096
        config = _FakeConfig()
        tokenizer = None
        image_token_id = 151655  # Qwen3-VL typical value

        def __call__(self, *args, **kwargs):
            raise RuntimeError("not used in init test")

    monkeypatch.setattr(bw, "build_uamvla_backbone", lambda cfg: _FakeBackbone())

    # Class identity check
    assert UamVLA.__name__ == "UamVLA"

    # UamVLADefaultConfig dataclass smoke
    cfg = UamVLADefaultConfig()
    assert cfg.name == "UamVLA"
    assert cfg.embodiment["name"] == "franka_libero"
    assert cfg.embodiment["action_dim"] == 7
    assert cfg.action_model["future_action_window_size"] == 7
    assert cfg.action_model["num_bins"] == 256
    assert cfg.state_encoder["type"] == "modular"
    assert cfg.aux_heads["action"]["enabled"] is True
    assert cfg.aux_heads["pose"]["enabled"] is True
    assert cfg.aux_heads["future"]["enabled"] is True
    assert cfg.aux_heads["recon"]["enabled"] is True
