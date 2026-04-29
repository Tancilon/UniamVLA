import pytest


def test_denoiser_imports():
    from starVLA.model.modules.uamvla.components.denoiser.scheduler import ReconDenoiser
    assert ReconDenoiser is not None


def test_vae_imports():
    pytest.importorskip("diffusers", reason="diffusers not installed; runs on remote")
    from starVLA.model.modules.uamvla.components.pixel_decoder.vae import VAEPixelDecoder
    assert VAEPixelDecoder is not None
