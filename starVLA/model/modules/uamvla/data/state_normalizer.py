"""Per-field state normalization for canonical_state dicts.

Composes existing `Normalizer` (q99 / mean_std / min_max) one instance per field,
matching OpenVLA-OFT's `normalize_action_and_proprio` recipe for the proprio half.
Spec: docs/superpowers/specs/2026-04-30-state-input-normalization-design.md
"""
from __future__ import annotations

import copy
from typing import Optional

import torch

from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer


_VALID_MODES = ("q99", "mean_std", "min_max", "none")


class StateNormalizer:
    """
    Apply per-field normalization to a canonical_state dict.

    Args:
        stats_dict: Loaded statistics.yaml as a dict. Must contain
            stats_dict["state_stats"][embodiment] when mode != "none".
        embodiment: Embodiment key, e.g. "franka_libero".
        mode: One of "q99" | "mean_std" | "min_max" | "none". Default "q99".
        apply_to: Optional list of dotted field paths (e.g. ["arm_0.ee_pose"]).
            If None, every field present in state_stats[embodiment] is normalized.
            Fields not in this list pass through unchanged.
    """

    def __init__(
        self,
        stats_dict: dict,
        embodiment: str,
        mode: str = "q99",
        apply_to: Optional[list[str]] = None,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Valid: {_VALID_MODES}")
        self.mode = mode
        self.embodiment = embodiment
        self._normalizers: dict[str, Normalizer] = {}

        if mode == "none":
            return

        if "state_stats" not in stats_dict:
            raise KeyError(
                "statistics.yaml is missing 'state_stats' block. Re-run the "
                "preprocessor to regenerate it, or set normalization mode to 'none'."
            )
        if embodiment not in stats_dict["state_stats"]:
            raise KeyError(
                f"state_stats has no entry for embodiment '{embodiment}'. "
                f"Available: {list(stats_dict['state_stats'].keys())}"
            )

        emb_stats = stats_dict["state_stats"][embodiment]
        for field_path, field_stats in emb_stats.items():
            if apply_to is not None and field_path not in apply_to:
                continue
            self._normalizers[field_path] = Normalizer(
                mode=mode, statistics=dict(field_stats)
            )

    def __call__(self, canonical_state: dict) -> dict:
        if self.mode == "none" or not self._normalizers:
            return canonical_state

        out = copy.deepcopy(canonical_state)
        for field_path, normalizer in self._normalizers.items():
            keys = field_path.split(".")
            ref = out
            for k in keys[:-1]:
                ref = ref[k]
            tensor = ref[keys[-1]]
            assert isinstance(tensor, torch.Tensor), (
                f"Field {field_path} must be a torch.Tensor; got {type(tensor)}"
            )
            ref[keys[-1]] = normalizer.forward(tensor)
        return out
