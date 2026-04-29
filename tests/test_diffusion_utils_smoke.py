def test_create_diffusion_imports():
    from starVLA.utils.diffusion_utils import create_diffusion
    assert callable(create_diffusion)
