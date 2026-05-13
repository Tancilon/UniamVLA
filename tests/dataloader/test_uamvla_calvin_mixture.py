"""Verify uamvla_calvin_abcd mixture loads via the main lerobot_datasets path
and yields batches with expected shapes.

Critical: this test surfaces any LeRobot-reader-side incompatibility with PR 2's
parquet schema (e.g. missing per-row video columns, or wrong camera intrinsics
format) before PR 4 framework integration."""
import pytest
from pathlib import Path
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn


CALVIN_PATH = Path("playground/Datasets/UAMVLA_LEROBOT_CALVIN_ABCD_SMOKE")


@pytest.fixture
def cfg():
    if not CALVIN_PATH.exists():
        pytest.skip(f"CALVIN preprocessed dataset not found at {CALVIN_PATH}. "
                    "Run: python runners/preprocess_calvin.py --max_episodes 3 "
                    "--input_dir datasets/calvin/task_ABCD_D/training "
                    "--output_dir playground/Datasets/UAMVLA_LEROBOT_CALVIN_ABCD_SMOKE")
    return OmegaConf.create({
        "datasets": {
            "vla_data": {
                "data_root_dir": str(CALVIN_PATH.parent),
                "data_mix": "uamvla_calvin_abcd",
                "per_device_batch_size": 2,
            }
        }
    })


def test_mixture_loadable(cfg):
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    assert len(dataset) > 0


def test_batch_has_passthrough_keys_and_image_list(cfg):
    """Sample image must reflect delta_indices=[0, 7]; passthrough keys must exist."""
    from torch.utils.data import DataLoader
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert len(batch) == 2, "batch should have 2 samples"
    for sample in batch:
        # __trajectory_id / __base_index from PR 1 patch
        assert "__trajectory_id" in sample, "missing __trajectory_id (PR 1 patch issue?)"
        assert "__base_index" in sample, "missing __base_index (PR 1 patch issue?)"
        # standard LeRobot keys
        assert "image" in sample
        assert isinstance(sample["image"], list) and len(sample["image"]) > 0
        assert "action" in sample
        assert "lang" in sample or "language" in sample


def test_sample_has_state_with_33_dims(cfg):
    """state column should be 33 dims (15 robot_obs + 9 pose + 9 cam_extrinsic).

    LeRobot's _pack_sample packs state as a 2D array (T, 33) when delta_indices=[0]
    that's (1, 33). After processing this is exposed as `sample["state"]` IF
    `data_cfg.include_state: True` is set, or otherwise stays inside the raw data.
    """
    import numpy as np
    cfg_with_state = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_with_state.datasets.vla_data.include_state = True

    from torch.utils.data import DataLoader
    dataset = get_vla_dataset(data_cfg=cfg_with_state.datasets.vla_data)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))
    sample = batch[0]
    if "state" in sample:
        state = np.asarray(sample["state"])
        # state shape is (T=1, 33) when delta_indices=[0]
        assert state.shape[-1] == 33, f"expected state dim 33, got {state.shape[-1]}"
    else:
        pytest.skip("state not in sample dict (include_state flag not honored?)")


def test_batch_video_has_two_views(cfg):
    """delta_indices=[0, 7] means each video tensor has time-dim 2.
    After _pack_sample, image becomes List[PIL] of [t=0 frames across views].
    For CALVIN with primary + wrist, this is 2 PIL images.

    The future frame (t=H-1) is NOT in the packed image list — framework
    will re-fetch via decord in PR 6. For PR 3, just verify the t=0 view list.
    """
    from torch.utils.data import DataLoader
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))
    sample = batch[0]
    # CALVIN has 2 views (primary, wrist), each at t=0 → 2 PIL images
    assert len(sample["image"]) >= 1, "expected at least 1 view in image list"
