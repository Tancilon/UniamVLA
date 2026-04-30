def test_modular_state_encoder_imports():
    from starVLA.model.modules.uamvla.state_encoder.modular_state_encoder import ModularStateEncoder
    # Cannot fully construct without embodiment registry yet — just verify class exists
    assert ModularStateEncoder is not None


def test_special_tokens_helpers_import():
    from starVLA.model.modules.uamvla.state_encoder.special_tokens import register_state_tokens
    assert callable(register_state_tokens)


def test_normalized_canonical_state_passes_through_modular_encoder(sample_dataset_dir):
    """Full pipeline: dataset -> StateNormalizer -> ModularStateEncoder -> 3-token output."""
    import torch
    from starVLA.dataloader.uamvla_dataset import UamVLADataset
    from starVLA.model.modules.uamvla.state_encoder.modular_state_encoder import (
        ModularStateEncoder,
    )

    HIDDEN_DIM = 1024  # Smoke-only; production uses Qwen3-VL's 3584.
    encoder = ModularStateEncoder(embodiment="franka_libero", hidden_dim=HIDDEN_DIM)
    encoder.eval()

    ds = UamVLADataset(
        data_root=sample_dataset_dir,
        embodiment="franka_libero",
        action_horizon=8,
        normalization={
            "mode": "q99",
            "apply_to": ["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"],
        },
    )
    sample = ds[0]
    # Add batch dim to each tensor (encoder expects (B, ...)).
    cs = {}
    for k, v in sample["canonical_state"].items():
        if isinstance(v, dict):
            cs[k] = {k2: v2.unsqueeze(0) for k2, v2 in v.items()}
        else:
            cs[k] = v.unsqueeze(0)

    with torch.no_grad():
        out = encoder(cs)

    assert out.shape == (1, 3, HIDDEN_DIM), f"unexpected encoder output shape: {out.shape}"
    assert torch.isfinite(out).all(), "non-finite values in encoder output"
