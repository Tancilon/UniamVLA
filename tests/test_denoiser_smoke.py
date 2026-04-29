import pytest


def test_denoiser_imports():
    from starVLA.model.modules.uamvla.components.denoiser.scheduler import ReconDenoiser
    assert ReconDenoiser is not None


def test_vae_imports():
    pytest.importorskip("diffusers", reason="diffusers not installed; runs on remote")
    from starVLA.model.modules.uamvla.components.pixel_decoder.vae import VAEPixelDecoder
    assert VAEPixelDecoder is not None


def test_readers_import():
    from starVLA.model.modules.uamvla.components.query_reader import TaskQueryReader
    from starVLA.model.modules.uamvla.components.spatial_reader import slice_image_tokens
    assert TaskQueryReader is not None
    assert callable(slice_image_tokens)
