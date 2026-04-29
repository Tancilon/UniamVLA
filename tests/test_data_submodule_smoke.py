def test_embodiment_adapter_imports():
    from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter, EmbodimentAdapter
    adapter = LiberoAdapter()
    assert adapter.embodiment_name == "franka_libero"


def test_embodiment_registry_get_config():
    from starVLA.model.modules.uamvla.data.embodiment_registry import get_embodiment_config
    cfg = get_embodiment_config("franka_libero")
    assert "adapter" in cfg
    assert "special_tokens" in cfg


def test_action_tokenizer_imports():
    from starVLA.model.modules.uamvla.data.action_tokenizer import ActionTokenizer
    assert ActionTokenizer is not None


def test_chat_template_imports():
    from starVLA.model.modules.uamvla.data.chat_template import ChatTemplate, Qwen3VLChatTemplate
    assert Qwen3VLChatTemplate is not None
