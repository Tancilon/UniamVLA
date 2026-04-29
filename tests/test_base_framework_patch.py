import torch


def test_baseframework_has_visualize_batch_default():
    from starVLA.model.framework.base_framework import baseframework
    assert hasattr(baseframework, "visualize_batch")
    # Default returns empty dict (no behavior change for existing frameworks)
    fake = baseframework()
    out = fake.visualize_batch({}, n_samples=1)
    assert out == {}
