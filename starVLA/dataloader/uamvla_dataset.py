# starVLA/dataloader/uamvla_dataset.py
"""UamVLA dataloader plugin: reads UamVLA preprocessor JSONL → starVLA examples list.

Plugin entry: get_vla_dataset(data_cfg) — invoked by starVLA trainer when
datasets.vla_data.dataset_py == "uamvla_dataset".

Spec: docs/superpowers/specs/2026-04-29-starvla-migration-design.md §5
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from starVLA.model.modules.uamvla.data.embodiment_registry import get_embodiment_config
from starVLA.utils.point_cloud import clean_point_cloud


def _pil_to_chw_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a (C, H, W) float32 tensor in [0, 1].

    Used for aux-head image targets (image_target / image_future) which must
    reach the framework's collator as tensors so stack_optional_tensor_fields
    can build a (B, C, H, W) batch. The dataset's primary image list stays
    PIL (build_inputs feeds PIL to Qwen3VLProcessor), so the conversion here
    is intentionally limited to aux fields.
    """
    arr = np.array(img, dtype=np.uint8, copy=True)  # (H, W, C); PIL buffer can be read-only
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0


DATASET_NAMED_MIXTURES = {
    "libero_uamvla": [
        # (data_subdir, weight, embodiment_tag)
        ("libero_spatial", 1.0, "franka_libero"),
        # Phase 2: + libero_object, libero_goal, libero_10
    ],
    "calvin_uamvla": [
        ("task_D_D", 1.0, "franka_calvin"),         # existing — kept for backward compat
    ],
    "calvin_abc_d_uamvla": [
        ("task_ABC_D", 1.0, "franka_calvin"),
    ],
    "calvin_abcd_d_uamvla": [
        ("task_ABCD_D/training", 1.0, "franka_calvin"),
    ],
}


class UamVLADataset(Dataset):
    """Read UamVLA JSONL → produce starVLA-compatible examples dict per __getitem__."""

    def __init__(
        self,
        data_root: Path | str,
        embodiment: str = "franka_libero",
        action_horizon: int = 8,
        max_samples: Optional[int] = None,
        transforms=None,
        normalization: Optional[dict] = None,
    ):
        self.data_root = Path(data_root)
        self.embodiment = embodiment
        self.action_horizon = action_horizon
        self.transforms = transforms

        # Load samples
        with open(self.data_root / "data.jsonl") as f:
            self.samples = [json.loads(l) for l in f if l.strip()]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Load stats
        with open(self.data_root / "statistics.yaml") as f:
            self.stats = yaml.safe_load(f)
        emb_stats = self.stats["embodiment_stats"][embodiment]
        self.action_min = np.array(emb_stats["action_min_bound"], dtype=np.float32)
        self.action_max = np.array(emb_stats["action_max_bound"], dtype=np.float32)
        self.view_names = list(self.stats.get("view_names", ["static", "wrist"]))

        # Embodiment adapter (stateless singleton)
        self.adapter = get_embodiment_config(embodiment)["adapter"]

        # State normalizer: build from stats already loaded into self.stats.
        from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer
        norm_cfg = normalization or {"mode": "q99"}
        self.state_normalizer = StateNormalizer(
            stats_dict=self.stats,
            embodiment=embodiment,
            mode=norm_cfg.get("mode", "q99"),
            apply_to=norm_cfg.get("apply_to"),
        )

        # Episode index for action chunk slicing
        self._episode_index = {(s["episode_id"], int(s["step_idx"])): i
                               for i, s in enumerate(self.samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        raw = self.samples[idx]

        # Load multi-view images
        images = []
        for img_path in raw["image"]:
            img = Image.open(self.data_root / img_path).convert("RGB")
            if self.transforms:
                img = self.transforms(img)
            images.append(img)

        # Action chunk (H, 7), normalized
        action, action_mask = self._slice_action_chunk(raw)

        # Canonical state via adapter, then apply state normalizer
        canonical_state = self.state_normalizer(self.adapter.to_canonical(raw))

        sample = {
            "image": images,
            "lang": raw["instruction"],
            "action": action,
            "action_mask": action_mask,
            "canonical_state": canonical_state,
            "embodiment": self.embodiment,
            "view_names": self.view_names,
        }

        # Optional aux head targets (loaded if present in JSONL)
        self._load_aux_targets(raw, sample)
        return sample

    def _slice_action_chunk(self, raw):
        H = self.action_horizon
        eid = raw["episode_id"]
        t = int(raw["step_idx"])
        T = int(raw["total_steps"])
        action = np.zeros((H, 7), dtype=np.float32)
        mask = np.zeros((H, 7), dtype=np.int64)

        for k in range(H):
            future_t = t + k
            if future_t >= T:
                break
            row_idx = self._episode_index.get((eid, future_t))
            if row_idx is None:
                break
            future_raw = self.samples[row_idx]
            # Trim 24D → 7D, then min-max normalize to [-1, 1]
            raw_action = np.array(future_raw["action"][:7], dtype=np.float32)
            normalized = self._normalize_action(raw_action)
            action[k] = normalized
            mask[k] = np.array(future_raw["action_mask"][:7], dtype=np.int64)
        return torch.tensor(action, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

    def _normalize_action(self, action):
        """min-max → [-1, 1]."""
        denom = (self.action_max - self.action_min) + 1e-8
        return 2.0 * (action - self.action_min) / denom - 1.0

    def _load_aux_targets(self, raw, sample):
        """Load image_target / image_future / point_cloud / pose_gt / static_cam_extrinsic if present."""
        # image_target / image_future are unconditionally tensor-ified — the
        # framework's stack_optional_tensor_fields requires .shape/.dtype on
        # whatever comes out of __getitem__. We deliberately do NOT route
        # these through self.transforms (which is reserved for the primary
        # image list and is allowed to be None to keep PIL semantics for
        # Qwen3VLProcessor).
        if "image_target" in raw and raw["image_target"]:
            try:
                img = Image.open(self.data_root / raw["image_target"]).convert("RGB")
                sample["image_target"] = _pil_to_chw_tensor(img)
            except FileNotFoundError:
                pass
        if "image_future" in raw and raw["image_future"]:
            try:
                img = Image.open(self.data_root / raw["image_future"]).convert("RGB")
                sample["image_future"] = _pil_to_chw_tensor(img)
            except FileNotFoundError:
                pass
        if "pose_6d" in raw:
            sample["pose_gt"] = {
                "rotation":    torch.tensor(raw["pose_6d"]["rotation"], dtype=torch.float32),
                "translation": torch.tensor(raw["pose_6d"]["translation"], dtype=torch.float32),
            }
        if "static_cam_extrinsic" in raw:
            rot_flat = torch.tensor(raw["static_cam_extrinsic"]["rotation"], dtype=torch.float32)
            sample["static_cam_extrinsic"] = {
                "rotation": rot_flat.view(3, 3),
                "translation": torch.tensor(raw["static_cam_extrinsic"]["translation"], dtype=torch.float32),
            }
        if "point_cloud" in raw and raw["point_cloud"]:
            try:
                pts = np.load(self.data_root / raw["point_cloud"])
                pts = clean_point_cloud(pts)
                sample["point_cloud"] = torch.tensor(pts, dtype=torch.float32)
            except FileNotFoundError:
                pass


def collate_fn(batch):
    """Pass through. The framework's forward() handles stacking via collator_helpers.

    starVLA convention (see lerobot_datasets.py:19-20): trivial list passthrough.
    """
    return batch


def resolve_data_dir(data_cfg) -> Path:
    """Resolve the on-disk data directory for the configured single-subdir mixture.

    Single source of truth for `data_root_dir / subdir` so the dataloader and any
    other component (e.g. PoseHead's stats_path auto-derivation) stay in lockstep.
    Raises NotImplementedError on multi-subdir mixtures, matching get_vla_dataset.
    """
    data_root = Path(data_cfg.data_root_dir)
    mixture = DATASET_NAMED_MIXTURES.get(data_cfg.data_mix)
    if mixture is None:
        raise ValueError(
            f"Unknown data_mix '{data_cfg.data_mix}'. "
            f"Available: {list(DATASET_NAMED_MIXTURES)}"
        )
    if len(mixture) != 1:
        raise NotImplementedError("Multi-subdir mixture deferred to Phase 2")
    subdir, _weight, _embodiment = mixture[0]
    return data_root / subdir


def get_vla_dataset(data_cfg, mode: str = "train", **kwargs) -> Dataset:
    """starVLA plugin entry point. Returns a Dataset that yields starVLA examples dicts."""
    data_root = Path(data_cfg.data_root_dir)
    mixture = DATASET_NAMED_MIXTURES.get(data_cfg.data_mix)
    if mixture is None:
        raise ValueError(f"Unknown data_mix '{data_cfg.data_mix}'. Available: {list(DATASET_NAMED_MIXTURES)}")

    # Pull state encoder normalization config if present.
    framework = getattr(data_cfg, "framework", None) or kwargs.get("framework")
    norm_cfg = None
    if framework is not None:
        # Handle both OmegaConf DictConfig (attribute access) and plain dict (key access).
        if hasattr(framework, "state_encoder"):
            state_enc = framework.state_encoder
        elif isinstance(framework, dict):
            state_enc = framework.get("state_encoder", {})
        else:
            state_enc = None
        if state_enc is not None:
            if hasattr(state_enc, "normalization"):
                raw_norm = state_enc.normalization
            elif isinstance(state_enc, dict):
                raw_norm = state_enc.get("normalization")
            else:
                raw_norm = None
            if raw_norm is not None:
                # Convert OmegaConf DictConfig → plain dict so UamVLADataset
                # can call .get() on it. Lazy-import to avoid hard dep in tests.
                try:
                    from omegaconf import OmegaConf
                    norm_cfg = OmegaConf.to_container(raw_norm, resolve=True)
                except ImportError:
                    norm_cfg = dict(raw_norm) if raw_norm is not None else None

    # Phase 1: single embodiment, single subdir; multi-subdir support deferred
    if len(mixture) == 1:
        subdir, _weight, embodiment = mixture[0]
        return UamVLADataset(
            data_root=data_root / subdir,
            embodiment=embodiment,
            action_horizon=int(data_cfg.get("action_horizon", 8)),
            normalization=norm_cfg,
        )
    # Phase 2: ConcatDataset across multiple subdirs with weights
    raise NotImplementedError("Multi-subdir mixture deferred to Phase 2")
