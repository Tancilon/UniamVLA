from __future__ import annotations

import json
import subprocess
import sys

from omegaconf import OmegaConf


def test_prepare_reconvla_ar_diagnostic_run_disables_lora_by_default(tmp_path):
    source_run_dir = tmp_path / "source_run"
    source_run_dir.mkdir()
    (source_run_dir / "dataset_statistics.json").write_text(
        json.dumps({"franka": {"action": {"mean": [0.0], "std": [1.0]}}}),
        encoding="utf-8",
    )

    base_config = tmp_path / "base.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "framework": {
                    "name": "AuxVLAGR00T",
                    "reconvla": {
                        "lora": {
                            "enabled": True,
                            "r": 32,
                        }
                    },
                    "action_model": {
                        "action_horizon": 8,
                        "future_action_window_size": 7,
                    },
                }
            }
        ),
        base_config,
    )

    output_run_dir = tmp_path / "diagnostic_run"
    result = subprocess.run(
        [
            sys.executable,
            "tools/probes/prepare_reconvla_ar_diagnostic_run.py",
            "--source-run-dir",
            str(source_run_dir),
            "--base-config",
            str(base_config),
            "--output-run-dir",
            str(output_run_dir),
            "--model-path",
            "ckpt/checkpoint-5554",
            "--vision-tower-path",
            "ckpt/siglip-so400m-patch14-384",
            "--action-stat-path",
            "third_party/ReconVLA/reconvla/statistics.yaml",
            "--input-mode",
            "official_compose",
        ],
        check=True,
        cwd=".",
        text=True,
        capture_output=True,
    )

    cfg = OmegaConf.load(output_run_dir / "config.yaml")
    assert "ckpt_path=" in result.stdout
    assert cfg.framework.reconvla.inference_mode == "reconvla_ar_normalized"
    assert cfg.framework.reconvla.lora.enabled is False
    assert cfg.framework.action_model.action_horizon == 5
    assert (output_run_dir / "dataset_statistics.json").exists()
    assert (output_run_dir / "checkpoints" / "reconvla_ar_diagnostic.pt").exists()
