from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class _ModalityConfig:
    delta_indices: list[int]
    modality_keys: list[str]


class _ComposedModalityTransform:
    def __init__(self, transforms):
        self.transforms = transforms


class _StateActionToTensor:
    def __init__(self, apply_to):
        self.apply_to = apply_to


class _StateActionTransform:
    def __init__(self, apply_to, normalization_modes):
        self.apply_to = apply_to
        self.normalization_modes = normalization_modes


class _StateActionSinCosTransform:
    def __init__(self, apply_to):
        self.apply_to = apply_to


class _GenericTransform:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _EmbodimentTag:
    FRANKA = "franka"
    NEW_EMBODIMENT = "new_embodiment"
    OXE_DROID = "oxe_droid"
    OXE_BRIDGE = "oxe_bridge"
    OXE_RT1 = "oxe_rt1"
    GR1 = "gr1"


def _install_gr00t_registry_stubs(monkeypatch):
    modules = {
        "starVLA.dataloader": types.ModuleType("starVLA.dataloader"),
        "starVLA.dataloader.gr00t_lerobot": types.ModuleType(
            "starVLA.dataloader.gr00t_lerobot"
        ),
        "starVLA.dataloader.gr00t_lerobot.data_config": types.ModuleType(
            "starVLA.dataloader.gr00t_lerobot.data_config"
        ),
        "starVLA.dataloader.gr00t_lerobot.datasets": types.ModuleType(
            "starVLA.dataloader.gr00t_lerobot.datasets"
        ),
        "starVLA.dataloader.gr00t_lerobot.embodiment_tags": types.ModuleType(
            "starVLA.dataloader.gr00t_lerobot.embodiment_tags"
        ),
        "starVLA.dataloader.gr00t_lerobot.mixtures": types.ModuleType(
            "starVLA.dataloader.gr00t_lerobot.mixtures"
        ),
        "starVLA.dataloader.gr00t_lerobot.transform": types.ModuleType(
            "starVLA.dataloader.gr00t_lerobot.transform"
        ),
        "starVLA.dataloader.gr00t_lerobot.transform.base": types.ModuleType(
            "starVLA.dataloader.gr00t_lerobot.transform.base"
        ),
        "starVLA.dataloader.gr00t_lerobot.transform.concat": types.ModuleType(
            "starVLA.dataloader.gr00t_lerobot.transform.concat"
        ),
        "starVLA.dataloader.gr00t_lerobot.transform.state_action": types.ModuleType(
            "starVLA.dataloader.gr00t_lerobot.transform.state_action"
        ),
        "starVLA.dataloader.gr00t_lerobot.transform.video": types.ModuleType(
            "starVLA.dataloader.gr00t_lerobot.transform.video"
        ),
    }
    modules["starVLA.dataloader.gr00t_lerobot.data_config"].ROBOT_TYPE_CONFIG_MAP = {}
    modules[
        "starVLA.dataloader.gr00t_lerobot.embodiment_tags"
    ].ROBOT_TYPE_TO_EMBODIMENT_TAG = {}
    modules["starVLA.dataloader.gr00t_lerobot.embodiment_tags"].EmbodimentTag = (
        _EmbodimentTag
    )
    modules["starVLA.dataloader.gr00t_lerobot.mixtures"].DATASET_NAMED_MIXTURES = {}
    modules["starVLA.dataloader.gr00t_lerobot.datasets"].ModalityConfig = (
        _ModalityConfig
    )
    modules["starVLA.dataloader.gr00t_lerobot.transform.base"].ComposedModalityTransform = (
        _ComposedModalityTransform
    )
    modules["starVLA.dataloader.gr00t_lerobot.transform.concat"].ConcatTransform = (
        _GenericTransform
    )
    modules[
        "starVLA.dataloader.gr00t_lerobot.transform.state_action"
    ].StateActionToTensor = _StateActionToTensor
    modules[
        "starVLA.dataloader.gr00t_lerobot.transform.state_action"
    ].StateActionTransform = _StateActionTransform
    modules[
        "starVLA.dataloader.gr00t_lerobot.transform.state_action"
    ].StateActionSinCosTransform = _StateActionSinCosTransform
    video_mod = modules["starVLA.dataloader.gr00t_lerobot.transform.video"]
    for name in (
        "VideoColorJitter",
        "VideoCrop",
        "VideoResize",
        "VideoToNumpy",
        "VideoToTensor",
    ):
        setattr(video_mod, name, _GenericTransform)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _load_registry(monkeypatch):
    _install_gr00t_registry_stubs(monkeypatch)
    path = Path("starVLA/dataloader/gr00t_lerobot/registry.py").resolve()
    spec = importlib.util.spec_from_file_location("_test_gr00t_registry", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_test_gr00t_registry", module)
    spec.loader.exec_module(module)
    return module


def test_uamvla_libero_h8_registry_entries_exist(monkeypatch):
    registry = _load_registry(monkeypatch)
    assert "uamvla_libero_franka_h8" in registry.ROBOT_TYPE_CONFIG_MAP
    assert "uamvla_libero_all_h8" in registry.DATASET_NAMED_MIXTURES
    assert registry.DATASET_NAMED_MIXTURES["uamvla_libero_all_h8"] == [
        ("lerobot_libero_object", 1.0, "uamvla_libero_franka_h8"),
        ("lerobot_libero_goal", 1.0, "uamvla_libero_franka_h8"),
        ("lerobot_libero_spatial", 1.0, "uamvla_libero_franka_h8"),
        ("lerobot_libero_10", 1.0, "uamvla_libero_franka_h8"),
    ]


def test_uamvla_libero_data_config_shapes(monkeypatch):
    registry = _load_registry(monkeypatch)
    cfg = registry.ROBOT_TYPE_CONFIG_MAP["uamvla_libero_franka_h8"]
    modality = cfg.modality_config()
    assert modality["video"].delta_indices == [0]
    assert modality["action"].delta_indices == list(range(8))
    assert modality["state"].delta_indices == [0]
    assert cfg.state_keys == [
        "state.robot_obs",
        "state.target_pose_rot6d",
        "state.target_pose_trans",
        "state.static_cam_rot6d",
        "state.static_cam_trans",
    ]


def test_uamvla_gr00t_libero_yaml_loads():
    cfg = OmegaConf.load("starVLA/config/training/uamvla_gr00t_libero.yaml")
    assert cfg.framework.name == "UamVLAGR00T"
    assert cfg.datasets.vla_data.data_root_dir == "datasets/libero2uam"
    assert cfg.datasets.vla_data.data_mix == "uamvla_libero_all_h8"
    assert cfg.framework.action_model.state_dim == 7
    assert cfg.framework.action_model.action_horizon == 8
    assert cfg.datasets.vla_data.aux_state_slice.target_pose_rot6d == [15, 21]


def test_auxvla_gr00t_libero_yaml_loads():
    cfg = OmegaConf.load("starVLA/config/training/auxvla_gr00t_libero.yaml")
    assert cfg.framework.name == "AuxVLAGR00T"
    assert cfg.framework.reconvla.model_path == "ckpt/pretrain-checkpoint-10388"
    assert cfg.framework.reconvla.vision_tower_path == "ckpt/siglip-so400m-patch14-384"
    assert cfg.framework.reconvla.single_view_mode == "primary"
    assert cfg.framework.reconvla.synthetic_image_token_id == -200
    assert cfg.framework.reconvla.disable_internal_recon_loss is True
    assert cfg.datasets.vla_data.data_mix == "uamvla_libero_all_h8"
    assert cfg.datasets.vla_data.obs == ["video.primary_image", "video.wrist_image"]
    assert cfg.framework.action_model.action_horizon == 8
    assert cfg.framework.aux_heads.recon.enabled is False


def test_auxvla_gr00t_lora_libero_yaml_loads():
    cfg = OmegaConf.load("starVLA/config/training/auxvla_gr00t_lora_libero.yaml")
    assert cfg.run_id == "auxvla_gr00t_lora_libero_primary_h8"
    assert cfg.framework.name == "AuxVLAGR00T"
    assert cfg.framework.reconvla.lora.enabled is True
    assert cfg.framework.reconvla.lora.r == 16
    assert cfg.framework.reconvla.lora.lora_alpha == 32
    assert "q_proj" in cfg.framework.reconvla.lora.target_modules
    assert cfg.trainer.freeze_modules is None
