from __future__ import annotations

import json
import types

from omegaconf import OmegaConf


def test_load_policy_from_config_builds_model_and_attaches_stats(monkeypatch, tmp_path):
    from deployment.model_server import server_policy

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = run_dir / "config.yaml"
    stats_path = run_dir / "dataset_statistics.json"
    OmegaConf.save(
        OmegaConf.create(
            {
                "framework": {"name": "AuxVLAGR00T"},
                "trainer": {},
                "datasets": {"vla_data": {}},
            }
        ),
        config_path,
    )
    stats = {"franka": {"action": {"mean": [0.0], "std": [1.0]}}}
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    captured = {}

    def fake_apply_config_compat(cfg):
        captured["framework"] = cfg.framework.name

    def fake_build_framework(cfg):
        return types.SimpleNamespace(config=cfg)

    monkeypatch.setattr(server_policy, "apply_config_compat", fake_apply_config_compat)
    monkeypatch.setattr(server_policy, "build_framework", fake_build_framework)

    model = server_policy.load_policy_from_config(config_path)

    assert captured["framework"] == "AuxVLAGR00T"
    assert model.config.framework.name == "AuxVLAGR00T"
    assert model.norm_stats == stats


def test_load_policy_prefers_config_yaml_over_checkpoint(monkeypatch, tmp_path):
    from deployment.model_server import server_policy

    config_path = tmp_path / "config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "framework": {"name": "AuxVLAGR00T"},
                "trainer": {},
                "datasets": {"vla_data": {}},
            }
        ),
        config_path,
    )
    monkeypatch.setattr(server_policy, "apply_config_compat", lambda cfg: None)
    monkeypatch.setattr(server_policy, "build_framework", lambda cfg: types.SimpleNamespace(config=cfg))
    monkeypatch.setattr(
        server_policy.baseframework,
        "from_pretrained",
        lambda ckpt_path: (_ for _ in ()).throw(AssertionError("checkpoint loader should not run")),
    )

    args = types.SimpleNamespace(config_yaml=str(config_path), ckpt_path="ignored.pt")
    model = server_policy.load_policy(args)

    assert model.config.framework.name == "AuxVLAGR00T"
