from pathlib import Path

import yaml


def test_uamvla_oft_training_config_defines_required_trainer_intervals():
    cfg_path = Path("starVLA/config/training/uamvla_oft_calvin_abcd.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    trainer = cfg["trainer"]
    assert "eval_interval" in trainer
    assert isinstance(trainer["eval_interval"], int)
    assert trainer["eval_interval"] > 0


def test_uamvla_oft_training_config_runs_action_plus_recon_only():
    cfg_path = Path("starVLA/config/training/uamvla_oft_calvin_abcd.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    aux_heads = cfg["framework"]["aux_heads"]
    assert aux_heads["pose"]["enabled"] is False
    assert aux_heads["future"]["enabled"] is False
    assert aux_heads["recon"]["enabled"] is True
