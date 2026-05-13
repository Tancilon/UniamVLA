"""Single-batch forward + backward smoke test for ``UamVLAOFT``
(aux heads OFF, PR 4 baseline B).

Verifies that:
  * the framework class is constructible from
    ``starVLA/config/training/uamvla_oft_calvin_abcd.yaml``,
  * a single-batch forward through Qwen3-VL runs without OOM,
  * the returned ``action_loss`` is a finite scalar tensor,
  * ``loss.backward()`` propagates non-zero grads to at least one
    trainable parameter.

Per CLAUDE.md GPU rule: this test requires a free GPU
(``memory.used < 100 MiB`` and ``utilization.gpu < 5%``). When no idle
GPU is found we ``pytest.skip`` rather than fall back to CPU or kill any
running process.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def _free_gpu_id() -> int | None:
    """Return the index of the first idle GPU, or ``None`` if all busy.

    Idle is defined by CLAUDE.md as ``memory.used < 100 MiB`` AND
    ``utilization.gpu < 5%``.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            timeout=10,
        ).decode()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx, mem, util = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if mem < 100 and util < 5:
            return idx
    return None


@pytest.fixture
def gpu_id():
    gid = _free_gpu_id()
    if gid is None:
        pytest.skip(
            "No free GPU available — per CLAUDE.md, do not run on CPU "
            "and do not kill other processes. Free a GPU and retry."
        )
    return gid


def test_uamvla_oft_single_batch(gpu_id):
    """End-to-end single-batch forward + backward on the smoke mixture."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch  # noqa: E402  — import after CUDA_VISIBLE_DEVICES is set
    from omegaconf import OmegaConf  # noqa: E402

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "starVLA" / "config" / "training" / "uamvla_oft_calvin_abcd.yaml"
    cfg = OmegaConf.load(cfg_path)
    # train_starvla.py applies this normalization before model construction;
    # mirror it here so the YAML's `future_action_window_size` is exposed as
    # `action_horizon` (which Qwenvl_OFT.__init__ reads).
    from starVLA.model.framework.share_tools import apply_config_compat  # noqa: E402
    cfg = apply_config_compat(cfg)

    # The smoke mixture is whatever DATASET_NAMED_MIXTURES["uamvla_calvin_abcd"]
    # currently points at. Skip cleanly if the dataset directory isn't
    # available on this machine.
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES  # noqa: E402

    mixture = DATASET_NAMED_MIXTURES[cfg.datasets.vla_data.data_mix]
    dataset_dir = repo_root / cfg.datasets.vla_data.data_root_dir / mixture[0][0]
    if not dataset_dir.exists():
        pytest.skip(f"Preprocessed dataset not found at {dataset_dir}")

    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset  # noqa: E402
    from starVLA.model.framework.VLM4A.UamVLAOFT import UamVLAOFT  # noqa: E402

    model = UamVLAOFT(cfg).cuda()

    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    from torch.utils.data import DataLoader  # noqa: E402

    loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))

    out = model.forward(batch)
    assert "action_loss" in out
    loss = out["action_loss"]
    assert torch.is_tensor(loss), f"loss is not a tensor: {type(loss)}"
    assert torch.isfinite(loss), f"loss is not finite: {loss}"

    loss.backward()
    grad_count = sum(
        1
        for p in model.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert grad_count > 0, "no params received non-zero grads"
