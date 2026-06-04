from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def test_auxvla_reconvla_ar_recon_config_matches_official_recipe():
    cfg = OmegaConf.load("starVLA/config/training/auxvla_reconvla_ar_recon.yaml")

    assert Path(cfg.framework.reconvla.model_path).name == "pretrain-checkpoint-10388"
    assert cfg.framework.reconvla.training_mode == "reconvla_ar_recon"
    assert cfg.framework.reconvla.inference_mode == "reconvla_ar_normalized"
    assert cfg.framework.reconvla.ar_input_mode == "official_compose"
    assert cfg.framework.reconvla.action_token_source == "raw_with_reconvla_statistics"
    assert cfg.framework.reconvla.disable_internal_recon_loss is False
    assert cfg.framework.reconvla.reconstruct_image is False
    assert cfg.framework.reconvla.lora.enabled is True
    assert cfg.framework.reconvla.lora.r == 32
    assert cfg.framework.reconvla.lora.init_lora_weights == "gaussian"
    assert cfg.framework.reconvla.lora.train_mm_projector is True
    assert cfg.framework.reconvla.lora.train_mm_inv_projector is True
    assert cfg.framework.reconvla.lora.target_modules == ["all-linear-except-frozen"]
    assert cfg.framework.reconvla.lora.exclude_modules == ["vision_tower", "pixel_decoder"]
    assert cfg.framework.action_model.action_horizon == 5
    assert cfg.framework.action_model.future_action_window_size == 4
    assert cfg.datasets.vla_data.data_mix == "uamvla_calvin_abc"
    assert cfg.datasets.vla_data.action_type == "calvin_rel_action"
    assert cfg.datasets.vla_data.image_resize == 384
    assert cfg.datasets.vla_data.obs_image_size == 384
    assert cfg.trainer.learning_rate.qwen_vl_interface == 2.0e-5
    assert cfg.trainer.freeze_modules is None
