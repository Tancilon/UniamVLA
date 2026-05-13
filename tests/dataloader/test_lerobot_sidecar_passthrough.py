"""Verify LeRobotSingleDataset.__getitem__ passes through __trajectory_id / __base_index.

These two int keys are required by UamVLAOFT to look up sidecar files
(point_cloud / image_target) without modifying the LeRobot core data
contract for existing frameworks (QwenFast / QwenPI / QwenGR00T etc).
"""
import pytest
from pathlib import Path

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


@pytest.fixture
def libero_goal_dataset_path():
    """Use the existing LIBERO goal dataset already preprocessed in the repo
    as the test fixture — we are only asserting the pass-through behavior,
    which is dataset-agnostic.
    """
    path = Path("playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot")
    if not path.exists():
        pytest.skip(f"LIBERO goal LeRobot dataset not found at {path}")
    return path


def test_sample_contains_trajectory_id_and_base_index(libero_goal_dataset_path):
    """A sample drawn from LeRobotSingleDataset must include __trajectory_id
    and __base_index (the patch under test). They must be plain ints."""
    from starVLA.dataloader.gr00t_lerobot.data_config import Libero4in1DataConfig

    cfg = Libero4in1DataConfig()
    dataset = LeRobotSingleDataset(
        dataset_path=libero_goal_dataset_path,
        modality_configs=cfg.modality_config(),
        transforms=cfg.transform(),
        embodiment_tag=EmbodimentTag.FRANKA,
        video_backend="decord",
    )
    sample = dataset[0]
    assert "__trajectory_id" in sample, "sample missing __trajectory_id key"
    assert "__base_index" in sample, "sample missing __base_index key"
    assert isinstance(sample["__trajectory_id"], int)
    assert isinstance(sample["__base_index"], int)


def test_passthrough_keys_dont_break_existing_keys(libero_goal_dataset_path):
    """Standard LeRobot keys (action, image, lang) must still be present
    after the patch."""
    from starVLA.dataloader.gr00t_lerobot.data_config import Libero4in1DataConfig

    cfg = Libero4in1DataConfig()
    dataset = LeRobotSingleDataset(
        dataset_path=libero_goal_dataset_path,
        modality_configs=cfg.modality_config(),
        transforms=cfg.transform(),
        embodiment_tag=EmbodimentTag.FRANKA,
        video_backend="decord",
    )
    sample = dataset[0]
    for k in ("action", "image", "lang", "language"):
        assert k in sample, f"sample missing existing key: {k}"
