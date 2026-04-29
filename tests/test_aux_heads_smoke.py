def test_aux_heads_base_imports():
    from starVLA.model.modules.uamvla.aux_heads.base import AuxHead, HeadOutput
    assert AuxHead is not None
    assert HeadOutput is not None


def test_action_head_imports():
    from starVLA.model.modules.uamvla.aux_heads.action_head import ActionHead
    assert ActionHead is not None


def test_pose_head_imports():
    import pytest
    try:
        from starVLA.model.modules.uamvla.aux_heads.pose_head import PoseHead
        assert PoseHead is not None
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"PoseHead requires CUDA-built PointNet2: {e}")


def test_future_head_imports():
    from starVLA.model.modules.uamvla.aux_heads.future_head import FutureHead
    assert FutureHead is not None


def test_recon_head_imports():
    from starVLA.model.modules.uamvla.aux_heads.recon_head import ReconHead
    assert ReconHead is not None
