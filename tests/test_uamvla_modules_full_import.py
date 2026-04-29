"""All UamVLA submodules should import together without circular reference."""
import pytest


def test_aggregate_imports_no_external_deps():
    """Aggregate import of every UamVLA submodule that doesn't transitively load diffusers.

    Catches circular-import bugs locally even when diffusers is unavailable.
    """
    import starVLA.model.modules.uamvla.state_encoder.modular_state_encoder
    import starVLA.model.modules.uamvla.aux_heads.base
    import starVLA.model.modules.uamvla.aux_heads.action_head
    import starVLA.model.modules.uamvla.aux_heads.future_head
    import starVLA.model.modules.uamvla.aux_heads.recon_head
    import starVLA.model.modules.uamvla.aux_heads.pose_head
    import starVLA.model.modules.uamvla.components.pose.pose_utils
    import starVLA.model.modules.uamvla.components.denoiser.scheduler
    import starVLA.model.modules.uamvla.components.spatial_reader
    import starVLA.model.modules.uamvla.data.embodiment_adapter
    import starVLA.model.modules.uamvla.data.embodiment_registry
    import starVLA.model.modules.uamvla.data.action_tokenizer
    import starVLA.model.modules.uamvla.data.chat_template


def test_pixel_decoder_vae_import():
    """Cross-import sanity for the diffusers-dependent pixel_decoder.vae module."""
    pytest.importorskip("diffusers", reason="diffusers not installed; runs on remote")
    import starVLA.model.modules.uamvla.components.pixel_decoder.vae
