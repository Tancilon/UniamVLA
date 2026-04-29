def test_aux_heads_base_imports():
    from starVLA.model.modules.uamvla.aux_heads.base import AuxHead, HeadOutput
    assert AuxHead is not None
    assert HeadOutput is not None


def test_action_head_imports():
    from starVLA.model.modules.uamvla.aux_heads.action_head import ActionHead
    assert ActionHead is not None
