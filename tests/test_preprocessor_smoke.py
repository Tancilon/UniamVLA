"""Smoke test: libero_preprocessor module imports and class instantiates without an env."""
import pytest


def test_libero_preprocessor_imports():
    """Import the module and check known constants."""
    pytest.importorskip("libero", reason="LIBERO benchmark not installed; runs on remote")
    from tools.preprocess.libero_preprocessor import LiberoPreprocessor, MAX_ACTION_DIM, FRANKA_ACTION_DIM
    assert MAX_ACTION_DIM == 24
    assert FRANKA_ACTION_DIM == 7


def test_libero_preprocessor_construct_no_env():
    """Construct without target resolver — __init__ should not require LIBERO/MuJoCo."""
    pytest.importorskip("libero", reason="LIBERO benchmark not installed; runs on remote")
    from tools.preprocess.libero_preprocessor import LiberoPreprocessor
    p = LiberoPreprocessor(suite="libero_spatial", target_object_keyword=None, target_resolver=None)
    assert p.suite == "libero_spatial"
