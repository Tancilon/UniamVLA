from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class EmbodimentStats:
    action_dim: int
    action_min_bound: np.ndarray
    action_max_bound: np.ndarray


@dataclass
class DatasetStatistics:
    max_action_dim: int
    view_names: list[str] = field(default_factory=list)
    embodiment_stats: dict[str, EmbodimentStats] = field(default_factory=dict)
    scene_obs_mean: np.ndarray | None = None
    scene_obs_std: np.ndarray | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetStatistics":
        with open(path) as f:
            data = yaml.safe_load(f)

        if "view_names" not in data:
            raise ValueError(
                f"{path}: missing required top-level field 'view_names'. "
                f"Example: view_names: [static, wrist]"
            )
        view_names = list(data["view_names"])
        if not view_names:
            raise ValueError(f"{path}: 'view_names' must be a non-empty list.")

        embodiment_stats = {}
        for name, cfg in data["embodiment_stats"].items():
            embodiment_stats[name] = EmbodimentStats(
                action_dim=cfg["action_dim"],
                action_min_bound=np.array(cfg["action_min_bound"]),
                action_max_bound=np.array(cfg["action_max_bound"]),
            )

        return cls(
            max_action_dim=data["max_action_dim"],
            view_names=view_names,
            embodiment_stats=embodiment_stats,
            scene_obs_mean=np.array(data["scene_obs_mean"]) if "scene_obs_mean" in data else None,
            scene_obs_std=np.array(data["scene_obs_std"]) if "scene_obs_std" in data else None,
        )

    def normalize_action(self, action: np.ndarray, embodiment: str) -> np.ndarray:
        """Normalize 24D action: first action_dim dims use per-embodiment bounds, rest passthrough."""
        es = self.embodiment_stats[embodiment]
        result = action.copy()
        dim = es.action_dim
        range_ = es.action_max_bound - es.action_min_bound
        range_ = np.where(range_ == 0, 1.0, range_)
        result[:dim] = (action[:dim] - es.action_min_bound) / range_ * 2.0 - 1.0
        return result

    def denormalize_action(self, normalized: np.ndarray, embodiment: str) -> np.ndarray:
        """Denormalize 24D action: first action_dim dims use per-embodiment bounds, rest passthrough."""
        es = self.embodiment_stats[embodiment]
        result = normalized.copy()
        dim = es.action_dim
        result[:dim] = (normalized[:dim] + 1.0) / 2.0 * (es.action_max_bound - es.action_min_bound) + es.action_min_bound
        return result
