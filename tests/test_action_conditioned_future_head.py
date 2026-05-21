import torch

from starVLA.model.modules.uamvla.components.action_chunk_encoder import ActionChunkEncoder


def test_action_chunk_encoder_outputs_hidden_size():
    encoder = ActionChunkEncoder(
        action_dim=7,
        hidden_size=16,
        action_embed_dim=8,
        num_layers=1,
        num_heads=2,
        max_horizon=8,
    )
    actions = torch.randn(3, 8, 7)
    out = encoder(actions)
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()
