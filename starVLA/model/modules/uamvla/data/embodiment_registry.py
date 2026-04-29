"""EmbodimentRegistry — central config for all supported embodiments.

Each entry defines:
  - adapter: EmbodimentAdapter instance for raw obs → canonical state conversion
  - limb_config: list of limb specs for ModularStateEncoder instantiation
  - special_tokens: ordered list of special tokens that occupy state positions
                    in the prompt (must match limb_config order × tokens-per-limb)

Adding a new embodiment:
  1. Implement adapter in embodiment_adapters.py
  2. Add registry entry below
  3. If new special tokens are needed, add them to ALL_STATE_SPECIAL_TOKENS in
     uamvla/models/state_encoder/special_tokens.py
"""
from __future__ import annotations

from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter, CalvinAdapter


# Single-arm Franka with parallel gripper — used by both LIBERO and CALVIN
_SINGLE_ARM_LIMB_CONFIG = [
    {"id": "arm_0",     "type": "arm",     "args": {"n_joints": 7}},
    {"id": "gripper_0", "type": "gripper", "args": {}},
]

_SINGLE_ARM_SPECIAL_TOKENS = [
    "<|state_ee_main|>",
    "<|state_joint_main|>",
    "<|state_gripper_main|>",
]

# NOTE: Both single-arm entries share the same _SINGLE_ARM_LIMB_CONFIG and
# _SINGLE_ARM_SPECIAL_TOKENS list objects (this is intentional — same physical
# embodiment). Treat the dicts returned by get_embodiment_config() as read-only.
# If you need to modify a config, deep-copy it first; in-place mutation will
# silently affect every embodiment that shares the same constants.

EMBODIMENT_REGISTRY: dict = {
    "franka_libero": {
        "adapter":        LiberoAdapter(),
        "limb_config":    _SINGLE_ARM_LIMB_CONFIG,
        "special_tokens": _SINGLE_ARM_SPECIAL_TOKENS,
    },
    "franka_calvin": {
        "adapter":        CalvinAdapter(),
        "limb_config":    _SINGLE_ARM_LIMB_CONFIG,
        "special_tokens": _SINGLE_ARM_SPECIAL_TOKENS,
    },
}

# Phase 2 — bimanual (RoboTwin 2.0). Uncomment when adapter is implemented.
# Note: when uncommenting, also move the entry back inside the EMBODIMENT_REGISTRY dict.
# "robotwin2_dual": {
#     "adapter": RoboTwinAdapter(),
#     "limb_config": [
#         {"id": "arm_0",     "type": "arm",     "args": {"n_joints": 7}},
#         {"id": "gripper_0", "type": "gripper", "args": {}},
#         {"id": "arm_1",     "type": "arm",     "args": {"n_joints": 7}},
#         {"id": "gripper_1", "type": "gripper", "args": {}},
#     ],
#     "special_tokens": [
#         "<|state_ee_L|>", "<|state_joint_L|>", "<|state_gripper_L|>",
#         "<|state_ee_R|>", "<|state_joint_R|>", "<|state_gripper_R|>",
#     ],
# }


def get_embodiment_config(name: str) -> dict:
    """Lookup with informative error if missing."""
    if name not in EMBODIMENT_REGISTRY:
        raise KeyError(
            f"Embodiment '{name}' not registered. Available: {sorted(EMBODIMENT_REGISTRY.keys())}"
        )
    return EMBODIMENT_REGISTRY[name]


def list_embodiments() -> list[str]:
    return list(EMBODIMENT_REGISTRY.keys())
