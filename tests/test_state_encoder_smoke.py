def test_modular_state_encoder_imports():
    from starVLA.model.modules.uamvla.state_encoder.modular_state_encoder import ModularStateEncoder
    # Cannot fully construct without embodiment registry yet — just verify class exists
    assert ModularStateEncoder is not None


def test_special_tokens_helpers_import():
    from starVLA.model.modules.uamvla.state_encoder.special_tokens import register_state_tokens
    assert callable(register_state_tokens)
