"""P3 probe (spec §8.1): verify libero_env can import StateNormalizer + LiberoAdapter.

If this test fails inside libero_env, R3 fallback (inline q99 in client) activates.
"""
import importlib

import pytest


@pytest.mark.libero_env
def test_libero_env_can_import_state_normalizer():
    """The two modules client-side conversion depends on must be importable."""
    importlib.import_module(
        "starVLA.model.modules.uamvla.data.embodiment_adapter"
    )
    importlib.import_module(
        "starVLA.model.modules.uamvla.data.state_normalizer"
    )
