def test_pose_components_imports():
    from starVLA.model.modules.uamvla.components.pose.pose_utils import get_pose_dim, rotation_6d_to_matrix
    from starVLA.model.modules.uamvla.components.pose.sde import init_sde
    # PointNet2 CUDA extension may not be compiled on Mac — gate that import
    import pytest
    try:
        from starVLA.model.modules.uamvla.components.pose.pts_encoder import PointNet2Wrapper
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"PointNet2 CUDA extension not available locally: {e}")


def test_pointnet2_wrapper_syncs_buffers_to_input_device():
    import torch
    import torch.nn as nn
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required to verify PointNet2 buffer device sync")

    from starVLA.model.modules.uamvla.components.pose.pts_encoder import PointNet2Wrapper

    wrapper = object.__new__(PointNet2Wrapper)
    nn.Module.__init__(wrapper)
    wrapper.encoder = nn.BatchNorm1d(3)
    wrapper.use_cuda_backend = False

    assert wrapper.encoder.running_mean.device.type == "cpu"

    PointNet2Wrapper._sync_buffers_to_device(wrapper.encoder, torch.device("cuda:0"))

    assert wrapper.encoder.running_mean.device.type == "cuda"
