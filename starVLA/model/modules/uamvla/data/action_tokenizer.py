import numpy as np
from transformers import PreTrainedTokenizer


class ActionTokenizer:
    """Discretizes continuous actions into explicit special tokens.

    Unlike the old design that overwrites vocab tail tokens, this uses
    explicitly added <ACT_0> ~ <ACT_{n-1}> tokens.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        n_bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        self.n_bins = n_bins
        self.min_action = min_action
        self.max_action = max_action

        # Register action tokens
        action_tokens = [f"<ACT_{i}>" for i in range(n_bins)]
        tokenizer.add_tokens(action_tokens)

        self.bin_edges = np.linspace(min_action, max_action, n_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        self.bin_to_token_id = {
            i: tokenizer.convert_tokens_to_ids(f"<ACT_{i}>")
            for i in range(n_bins)
        }
        self.token_id_to_bin = {v: k for k, v in self.bin_to_token_id.items()}

    def encode(self, actions: np.ndarray) -> list[int]:
        """Continuous actions -> discrete bin indices -> token IDs."""
        clipped = np.clip(actions, self.bin_edges[0], self.bin_edges[-1])
        bins = np.digitize(clipped, self.bin_edges) - 1
        bins = np.clip(bins, 0, self.n_bins - 1)
        return [self.bin_to_token_id[int(b)] for b in bins]

    def decode(self, token_ids: list[int]) -> np.ndarray:
        """Token IDs -> bin centers -> continuous actions.

        Non-action tokens (e.g. from an under-trained model) are mapped to the
        middle bin (≈ 0.0 action) so evaluation can proceed without crashing.
        """
        mid_bin = self.n_bins // 2
        bins = [self.token_id_to_bin.get(tid, mid_bin) for tid in token_ids]
        return self.bin_centers[bins]
