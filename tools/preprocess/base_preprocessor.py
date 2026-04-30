from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tools.statistics import DatasetStatistics, EmbodimentStats


class BasePreprocessor(ABC):
    @abstractmethod
    def process(self, input_dir: str, output_dir: str):
        """Read raw dataset, output unified format (data.jsonl, statistics.yaml, images/, metadata.yaml)."""
        ...

    def compute_statistics(
        self,
        samples: list[dict],
        embodiment: str = "default",
        view_names: list[str] | None = None,
    ) -> DatasetStatistics:
        """Auto-compute normalization params from sample list.

        Builds a single-embodiment DatasetStatistics from raw samples.
        Each sample must have key 'action' (1-D array).
        """
        actions = np.array([s["action"] for s in samples])
        action_dim = actions.shape[1]
        es = EmbodimentStats(
            action_dim=action_dim,
            action_min_bound=actions.min(axis=0),
            action_max_bound=actions.max(axis=0),
        )
        return DatasetStatistics(
            max_action_dim=action_dim,
            view_names=view_names or [],
            embodiment_stats={embodiment: es},
            scene_obs_mean=None,
            scene_obs_std=None,
        )
