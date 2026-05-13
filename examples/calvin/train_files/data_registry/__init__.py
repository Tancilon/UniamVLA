# examples/calvin/train_files/data_registry/__init__.py
"""CALVIN data registry for UamVLAOFT — auto-discovered by starVLA
gr00t_lerobot.registry at import time."""
from .data_config import (
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    DATASET_NAMED_MIXTURES,
)

__all__ = ["ROBOT_TYPE_CONFIG_MAP", "ROBOT_TYPE_TO_EMBODIMENT_TAG", "DATASET_NAMED_MIXTURES"]
