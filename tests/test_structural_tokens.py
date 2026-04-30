"""Tests for register_structural_tokens — adds <|action_start|> as a single special token."""


class _FakeTokenizer:
    """Minimal HF-tokenizer-like stub: tracks added special tokens and assigns ids."""
    unk_token_id = 1

    def __init__(self):
        self._vocab = {}
        self._special_tokens = []
        self._next_id = 100

    def add_special_tokens(self, payload):
        added = 0
        for tok in payload.get("additional_special_tokens", []):
            if tok not in self._vocab:
                self._vocab[tok] = self._next_id
                self._next_id += 1
                self._special_tokens.append(tok)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)

    def get_input_embeddings(self):
        return self  # stand-in; only .weight.shape[0] is checked

    @property
    def weight(self):
        class _W:
            shape = (self._next_id,)
        w = _W()
        w.shape = (self._next_id,)
        return w

    def __len__(self):
        return self._next_id


class _FakeBackbone:
    """Stub exposing resize_token_embeddings + get_embed_tokens."""
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.resized_to = None

    def resize_token_embeddings(self, new_size):
        self.resized_to = new_size

    def get_embed_tokens(self):
        # Return an object with .weight.shape[0] == tokenizer length
        class _E:
            class weight:
                shape = (len(self._tokenizer),)
        e = _E()
        e.weight.shape = (len(self._tokenizer),)
        return e


def test_register_structural_tokens_adds_action_start():
    from starVLA.model.modules.uamvla.state_encoder.special_tokens import (
        register_structural_tokens, STRUCTURAL_SPECIAL_TOKENS,
    )
    assert "<|action_start|>" in STRUCTURAL_SPECIAL_TOKENS

    tok = _FakeTokenizer()
    bb = _FakeBackbone(tok)
    ids = register_structural_tokens(tok, bb)

    assert "<|action_start|>" in ids
    assert ids["<|action_start|>"] == tok.convert_tokens_to_ids("<|action_start|>")
    assert ids["<|action_start|>"] != tok.unk_token_id
    assert bb.resized_to == len(tok)


def test_register_structural_tokens_is_idempotent():
    from starVLA.model.modules.uamvla.state_encoder.special_tokens import (
        register_structural_tokens,
    )
    tok = _FakeTokenizer()
    bb = _FakeBackbone(tok)
    ids_first = register_structural_tokens(tok, bb)
    bb.resized_to = None  # reset to detect a second resize
    ids_second = register_structural_tokens(tok, bb)

    assert ids_first == ids_second
    # Second call must NOT trigger another resize (no new tokens added).
    assert bb.resized_to is None
