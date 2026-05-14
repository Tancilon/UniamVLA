"""Regression guard for the LeRobotMixtureDataset.__getitem__ sidecar
passthrough (commit c332573).

PR 1 added the __trajectory_id / __base_index injection to
LeRobotSingleDataset.__getitem__. The mixture path
(LeRobotMixtureDataset.__getitem__) reimplements step loading and bypasses
the single-dataset method — so it needed its own patch. This was caught
by PR 3 end-to-end testing and fixed in commit c332573.

If a future upstream merge of starVLA loses the mixture-path patch, the
end-to-end tests (test_uamvla_calvin_mixture.py, GPU smokes) will still
catch it but with a distant failure signature (UamVLAOFT.aux_heads.pose
falls into dummy_loss → SR drop, many layers from root cause). This test
files the regression-detect right at the mixture passthrough boundary so
the root cause is named in the failure message.

CPU-only; ~10 ms when CALVIN smoke is on disk; skip cleanly otherwise.
"""
from pathlib import Path

import pytest


CALVIN_SMOKE = Path("playground/Datasets/UAMVLA_LEROBOT_CALVIN_ABCD_SMOKE")


@pytest.fixture
def mixture_dataset():
    if not CALVIN_SMOKE.exists():
        pytest.skip(
            f"CALVIN smoke dataset missing: {CALVIN_SMOKE}. "
            f"Run `python runners/preprocess_calvin.py --max_episodes 3 "
            f"--input_dir datasets/calvin/task_ABCD_D/training "
            f"--output_dir {CALVIN_SMOKE}` to regenerate."
        )
    from omegaconf import OmegaConf

    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    cfg = OmegaConf.create({
        "datasets": {
            "vla_data": {
                "data_root_dir": str(CALVIN_SMOKE.parent),
                "data_mix": "uamvla_calvin_abcd_smoke",
                "include_state": True,
                "per_device_batch_size": 1,
            }
        }
    })
    return get_vla_dataset(data_cfg=cfg.datasets.vla_data)


def test_mixture_passes_through_trajectory_id_and_base_index(mixture_dataset):
    """LeRobotMixtureDataset.__getitem__ must inject __trajectory_id and
    __base_index, mirroring the single-dataset patch from PR 1."""
    sample = mixture_dataset[0]
    assert "__trajectory_id" in sample, (
        "LeRobotMixtureDataset.__getitem__ did not inject __trajectory_id. "
        "Check that the mixture-path patch (commit c332573) is still in place."
    )
    assert "__base_index" in sample, (
        "LeRobotMixtureDataset.__getitem__ did not inject __base_index. "
        "Check that the mixture-path patch (commit c332573) is still in place."
    )
    assert isinstance(sample["__trajectory_id"], int)
    assert isinstance(sample["__base_index"], int)


def test_mixture_sample_ids_in_valid_range(mixture_dataset):
    """The injected ids must be in valid range for the smoke dataset
    (3 episodes × {65, 42, 45} frames). Don't assume sample[0] maps to
    (traj=0, base=0) — LeRobotMixtureDataset weights/samples differently.

    Smoke any random sample to confirm ids are sane integers, not garbage."""
    sample = mixture_dataset[0]
    traj = sample["__trajectory_id"]
    base = sample["__base_index"]
    assert 0 <= traj < 3, f"trajectory_id {traj} out of smoke-dataset range [0, 3)"
    assert base >= 0, f"base_index {base} is negative"
    # Each episode has < 100 frames in the smoke set.
    assert base < 100, f"base_index {base} unexpectedly large for smoke dataset"


def test_mixture_standard_keys_preserved(mixture_dataset):
    """The mixture-path passthrough must not drop the existing LeRobot keys
    (action / image / lang). Regression guard for the c332573 edit not having
    accidentally shifted unrelated logic."""
    sample = mixture_dataset[0]
    for k in ("action", "image", "lang"):
        assert k in sample, f"sample missing existing key: {k}"
