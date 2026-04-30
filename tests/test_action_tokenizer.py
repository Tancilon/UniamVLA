import numpy as np
import pytest

from starVLA.model.modules.uamvla.data.action_tokenizer import ActionTokenizer


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self._vocab = {}

    def add_tokens(self, tokens):
        added = 0
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = 100 + len(self._vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self._vocab[token]


def test_action_tokenizer_uses_all_configured_bins():
    tokenizer = _FakeTokenizer()
    action_tokenizer = ActionTokenizer(tokenizer, n_bins=256)

    token_ids = action_tokenizer.encode(np.array([-1.0, 1.0], dtype=np.float32))

    assert token_ids == [
        tokenizer.convert_tokens_to_ids("<ACT_0>"),
        tokenizer.convert_tokens_to_ids("<ACT_255>"),
    ]


def test_action_tokenizer_decodes_last_bin_token():
    tokenizer = _FakeTokenizer()
    action_tokenizer = ActionTokenizer(tokenizer, n_bins=256)
    last_token_id = tokenizer.convert_tokens_to_ids("<ACT_255>")

    decoded = action_tokenizer.decode([last_token_id])

    assert decoded.shape == (1,)
    assert decoded[0] == pytest.approx(1.0 - (1.0 / 256.0))
