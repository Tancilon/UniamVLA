"""Tests for chat templates — verify <|action_start|> is emitted as a single literal."""


def test_qwen3_vl_chat_template_emits_action_start_token():
    from starVLA.model.modules.uamvla.data.chat_template import (
        Qwen3VLChatTemplate, ACTION_START_TOKEN,
    )
    assert ACTION_START_TOKEN == "<|action_start|>"

    tpl = Qwen3VLChatTemplate()
    out = tpl.format(
        instruction="pick up the red block",
        embodiment="franka_libero",
        view_names=["primary"],
    )

    # Exactly one occurrence of the new token, zero of the old literal.
    assert out.count("<|action_start|>") == 1
    assert "<action_start>" not in out  # old literal must be gone


def test_qwen3_chat_template_emits_action_start_token():
    from starVLA.model.modules.uamvla.data.chat_template import Qwen3ChatTemplate

    tpl = Qwen3ChatTemplate()
    out = tpl.format(
        instruction="pick up the red block",
        embodiment="franka_libero",
        view_names=["primary"],
    )

    assert out.count("<|action_start|>") == 1
    assert "<action_start>" not in out


def test_action_start_token_constant_exported():
    from starVLA.model.modules.uamvla.data import chat_template
    assert hasattr(chat_template, "ACTION_START_TOKEN")
    assert chat_template.ACTION_START_TOKEN == "<|action_start|>"
