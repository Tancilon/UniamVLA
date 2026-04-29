"""Dataset statistics dataclass for preprocessor pipelines.

Ported from UamVLA: uamvla/data/statistics.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DatasetStatistics:
    robot_obs_mean: np.ndarray
    robot_obs_std: np.ndarray
    scene_obs_mean: np.ndarray | None
    scene_obs_std: np.ndarray | None
    action_min_bound: np.ndarray
    action_max_bound: np.ndarray
    action_dim: int
