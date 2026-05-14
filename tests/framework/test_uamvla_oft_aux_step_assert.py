"""Step-level assertion: on a deterministic mini-batch, each aux head loss must
be strictly GREATER than its dummy_loss (which is `0.0 * params.sum()` = 0).

This catches the failure mode where a head silently returns dummy_loss most
of the time (e.g. mask all-False because pose_gt / image_target / image_future
isn't populated by _unpack_lerobot_sample / _collate_aux).

Tests grow incrementally — PR 5 adds pose, PR 6 adds future, PR 7 adds recon.

NOTE: fixture is module-scoped to avoid OOM — UamVLAOFT loads a Qwen3-VL-8B
backbone (~16 GB bf16); 3 function-scoped fixtures would blow past 80 GB on
H100. CUDA_VISIBLE_DEVICES must be set EXTERNALLY by the caller (e.g.
`CUDA_VISIBLE_DEVICES=4 pytest ...`); module-scoped fixtures can't take
function-scoped `monkeypatch`.
"""
import subprocess
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf


def _free_gpu_id() -> int | None:
    """Return index of first GPU with <100 MiB used and <5% utilization."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"]
    ).decode()
    for line in out.strip().splitlines():
        idx, mem, util = [x.strip() for x in line.split(",")]
        if int(mem) < 100 and int(util) < 5:
            return int(idx)
    return None


@pytest.fixture(scope="module")
def configured_model_and_batch():
    """Build UamVLAOFT once and reuse across the 3 step-assert tests in this
    module. Function-scope would OOM (Qwen3-VL-8B is ~16 GB; 3 instances > 80 GB).
    Caller must set CUDA_VISIBLE_DEVICES externally.
    """
    if _free_gpu_id() is None:
        pytest.skip("No free GPU available — per CLAUDE.md, do not run on CPU "
                    "and do not kill other processes. Free a GPU and retry.")

    from starVLA.model.framework.VLM4A.UamVLAOFT import UamVLAOFT
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    cfg = OmegaConf.load("starVLA/config/training/uamvla_oft_calvin_abcd.yaml")
    # Mirror train_starvla.py's pre-construction config normalization.
    from starVLA.model.framework.share_tools import apply_config_compat  # noqa: E402
    cfg = apply_config_compat(cfg)
    # Smoke fixtures override the data mix + root from yaml (which points at
    # the full preprocessed dataset for training) to the smoke variant.
    cfg.datasets.vla_data.data_mix = "uamvla_calvin_abcd_smoke"
    cfg.datasets.vla_data.data_root_dir = "playground/Datasets"
    if not Path("playground/Datasets/UAMVLA_LEROBOT_CALVIN_ABCD_SMOKE").exists():
        pytest.skip("CALVIN preprocessed dataset not found")

    torch.manual_seed(42)
    model = UamVLAOFT(cfg).cuda()
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))
    return model, batch


def test_pose_head_loss_below_dummy(configured_model_and_batch):
    """pose head's real loss must be > dummy (= 0); proves head was invoked
    rather than short-circuited via mask all-False."""
    model, batch = configured_model_and_batch
    out = model.forward(batch)
    assert "pose_loss" in out, "pose head did not contribute a loss entry"
    real_loss = out["pose_loss"].item()
    dummy_loss = model.aux_heads["pose"].get_dummy_loss().item()
    assert real_loss > dummy_loss, (
        f"pose head returned dummy_loss path: real={real_loss}, "
        f"dummy={dummy_loss}. Check that pose_gt + point_cloud are populated "
        f"in batch_dict and pose_mask has any True entries."
    )


def test_future_head_loss_below_dummy(configured_model_and_batch):
    """future head's real loss must be > dummy (= 0)."""
    model, batch = configured_model_and_batch
    out = model.forward(batch)
    assert "future_loss" in out, "future head did not contribute a loss entry"
    real_loss = out["future_loss"].item()
    dummy_loss = model.aux_heads["future"].get_dummy_loss().item()
    assert real_loss > dummy_loss, (
        f"future head returned dummy_loss path: real={real_loss}, "
        f"dummy={dummy_loss}. Check that image_future is populated in batch_dict "
        f"and future_mask has any True entries."
    )


def test_recon_head_loss_below_dummy(configured_model_and_batch):
    """recon head's real loss must be > dummy (= 0)."""
    model, batch = configured_model_and_batch
    out = model.forward(batch)
    assert "recon_loss" in out, "recon head did not contribute a loss entry"
    real_loss = out["recon_loss"].item()
    dummy_loss = model.aux_heads["recon"].get_dummy_loss().item()
    assert real_loss > dummy_loss, (
        f"recon head returned dummy_loss path: real={real_loss}, "
        f"dummy={dummy_loss}. Check that image_target is populated in batch_dict "
        f"and recon_mask has any True entries."
    )
