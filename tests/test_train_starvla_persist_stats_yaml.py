"""Spec §6.1 — verify trainer copies the source statistics.yaml verbatim
into output_dir alongside dataset_statistics.json."""
from pathlib import Path

import pytest


def test_copy_stats_yaml_to_run_dir_copies_file(tmp_path: Path):
    pytest.importorskip("wandb", reason="wandb not installed; runs on remote")
    pytest.importorskip("accelerate", reason="accelerate not installed; runs on remote")
    from starVLA.training.train_starvla import VLATrainer

    src = tmp_path / "statistics.yaml"
    src.write_text("view_names:\n- static\n- wrist\n")
    out = tmp_path / "out"
    out.mkdir()

    VLATrainer._copy_stats_yaml_to_run_dir(src, out)

    dst = out / "statistics.yaml"
    assert dst.exists()
    assert dst.read_text() == src.read_text()


def test_copy_stats_yaml_to_run_dir_noop_when_source_missing(tmp_path: Path):
    pytest.importorskip("wandb", reason="wandb not installed; runs on remote")
    pytest.importorskip("accelerate", reason="accelerate not installed; runs on remote")
    from starVLA.training.train_starvla import VLATrainer

    src = tmp_path / "absent.yaml"
    out = tmp_path / "out"
    out.mkdir()

    # Should not raise.
    VLATrainer._copy_stats_yaml_to_run_dir(src, out)
    assert not (out / "statistics.yaml").exists()
