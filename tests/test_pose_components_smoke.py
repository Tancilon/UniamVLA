def test_pose_components_imports():
    from starVLA.model.modules.uamvla.components.pose.pose_utils import get_pose_dim, rotation_6d_to_matrix
    from starVLA.model.modules.uamvla.components.pose.sde import init_sde
    # PointNet2 CUDA extension may not be compiled on Mac — gate that import
    import pytest
    try:
        from starVLA.model.modules.uamvla.components.pose.pts_encoder import PointNet2Wrapper
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"PointNet2 CUDA extension not available locally: {e}")
