"""Integration smoke: register_structural_tokens → chat template → tokenizer round-trip."""
import pytest


def test_action_start_token_round_trips_through_real_tokenizer():
    """After register_structural_tokens, the chat template's emitted <|action_start|>
    must tokenize to a single token id (not a multi-subtoken BPE split)."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    from starVLA.model.modules.uamvla.state_encoder.special_tokens import (
        register_structural_tokens,
    )
    from starVLA.model.modules.uamvla.data.chat_template import (
        Qwen3VLChatTemplate, ACTION_START_TOKEN,
    )

    # Use a small public Qwen tokenizer for the smoke test. If unavailable
    # offline, skip — this test is best-effort.
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except Exception as e:
        pytest.skip(f"Qwen tokenizer not available offline: {e}")

    class _StubBackbone:
        def __init__(self, tk):
            self._tk = tk
        def resize_token_embeddings(self, n):
            pass  # not exercising the model path here
        def get_input_embeddings(self):
            class _E:
                class weight:
                    pass
            e = _E()
            e.weight.shape = (len(self._tk),)
            return e

    register_structural_tokens(tok, _StubBackbone(tok))

    # Chat template renders the prompt; tokenize it and verify the action-start
    # token appears as exactly one id in the output.
    prompt = Qwen3VLChatTemplate().format(
        instruction="pick up the red block",
        embodiment="franka_libero",
        view_names=["primary"],
    )
    assert ACTION_START_TOKEN in prompt

    encoded = tok.encode(prompt, add_special_tokens=False)
    action_start_id = tok.convert_tokens_to_ids(ACTION_START_TOKEN)

    assert action_start_id != tok.unk_token_id
    assert encoded.count(action_start_id) == 1, (
        f"<|action_start|> was not tokenized as a single id "
        f"(found {encoded.count(action_start_id)} occurrences of id {action_start_id})"
    )


def test_no_old_action_start_literal_remains_in_codebase():
    """Belt-and-suspenders: assert no Python file under starVLA/ or tests/
    still references the old literal '<action_start>'."""
    import pathlib
    import subprocess

    # Anchor paths off __file__ so the test is robust to pytest's cwd. Without
    # this, running from a sibling dir produces grep returncode=2 (path not
    # found) which the original `if returncode == 0:` guard silently swallowed.
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    starvla_dir = repo_root / "starVLA"
    tests_dir = repo_root / "tests"

    result = subprocess.run(
        ["grep", "-rn", "<action_start>",
         str(starvla_dir), str(tests_dir), "--include=*.py"],
        capture_output=True, text=True,
    )
    # grep: 0 = matches found, 1 = no matches, 2 = error. Anything else is bug.
    assert result.returncode in (0, 1), (
        f"grep failed (rc={result.returncode}): {result.stderr}"
    )

    # Note: tests/test_chat_template.py has the old literal inside negative
    # assertions ('not in out' checks), and this very test file also names the
    # literal in its docstrings and grep argument. Both are legitimate; we
    # allowlist them via substring match (path-prefix-independent) and require
    # ZERO matches anywhere else.
    if result.returncode == 0:
        allowed_substrings = (
            "tests/test_chat_template.py:",
            "tests/test_action_start_integration.py:",
        )
        lines = result.stdout.strip().split("\n")
        non_test_matches = [
            line for line in lines
            if not any(s in line for s in allowed_substrings)
        ]
        assert not non_test_matches, (
            "Old literal still present in production code:\n"
            + "\n".join(non_test_matches)
        )
