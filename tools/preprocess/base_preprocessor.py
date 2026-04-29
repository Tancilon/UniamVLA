from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tools.statistics import DatasetStatistics


class BasePreprocessor(ABC):
    @abstractmethod
    def process(self, input_dir: str, output_dir: str):
        """Read raw dataset, output unified format (data.jsonl, statistics.yaml, images/, metadata.yaml)."""
        ...

    def compute_statistics(self, samples: list[dict]) -> DatasetStatistics:
        """Auto-compute normalization params from sample list."""
        actions = np.array([s["action"] for s in samples])
        robot_obs = np.array([s["robot_obs"] for s in samples])
        return DatasetStatistics(
            robot_obs_mean=robot_obs.mean(axis=0),
            robot_obs_std=robot_obs.std(axis=0),
            scene_obs_mean=None,
            scene_obs_std=None,
            action_min_bound=actions.min(axis=0),
            action_max_bound=actions.max(axis=0),
            action_dim=actions.shape[1],
        )
