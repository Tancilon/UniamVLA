"""ActionLogitsProcessor — generate-time logits constraint for action chunks.

Activates on detecting <|action_start|> as the most recent generated token, then
masks all logits outside [action_begin_id, action_begin_id + n_bins) for exactly
action_chunk_len subsequent steps. State is per-batch-row.

Caller must construct a fresh instance per generate() call.
"""
from __future__ import annotations

import torch

try:
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover — transformers is a hard dep at runtime
    LogitsProcessor = object  # type: ignore[assignment]


class ActionLogitsProcessor(LogitsProcessor):
    """Constrains generation to action-bin tokens after <|action_start|>.

    Args:
        action_start_id: Token id of <|action_start|> (sentinel marking action segment start).
        action_begin_id: First action-bin token id (= <ACT_0> id).
        n_bins: Number of action bins (typically 256). Allowed range is
                [action_begin_id, action_begin_id + n_bins).
        action_chunk_len: Number of generation steps to keep masking active.
                          Typically action_horizon * action_dim.
    """

    def __init__(
        self,
        action_start_id: int,
        action_begin_id: int,
        n_bins: int,
        action_chunk_len: int,
    ):
        self.action_start_id = int(action_start_id)
        self.action_begin_id = int(action_begin_id)
        self.action_end_id = int(action_begin_id) + int(n_bins)  # exclusive
        self.action_chunk_len = int(action_chunk_len)
        # Lazily allocated on first __call__ once we know batch size + device.
        self._remaining: torch.Tensor | None = None
        self._last_seq_len: int = -1
        self._bounds_checked: bool = False

    def __call__(
        self,
        input_ids: torch.LongTensor,    # (B, S_so_far)
        scores: torch.FloatTensor,      # (B, vocab)
    ) -> torch.FloatTensor:
        B = scores.shape[0]
        device = scores.device

        # One-time bounds check: action range must lie within vocab.
        # Catches misconfig (e.g. action_end_id past vocab tail) at the first call
        # rather than silently mis-masking via Python slice truncation.
        if not self._bounds_checked:
            vocab = scores.shape[1]
            if self.action_end_id > vocab:
                raise ValueError(
                    f"ActionLogitsProcessor: action range "
                    f"[{self.action_begin_id}, {self.action_end_id}) exceeds vocab size {vocab}. "
                    f"Check action_begin_id + n_bins against tokenizer vocab."
                )
            if self.action_begin_id < 0:
                raise ValueError(
                    f"ActionLogitsProcessor: action_begin_id={self.action_begin_id} is negative."
                )
            self._bounds_checked = True

        # Lazy-init / re-init if batch size changed across calls.
        if self._remaining is None or self._remaining.shape[0] != B:
            self._remaining = torch.zeros(B, dtype=torch.long, device=device)
        elif self._remaining.device != device:
            self._remaining = self._remaining.to(device)

        # Detect new generate() call: sequence length shrank vs. last call.
        # Within a single generate(), seq_len only grows; a shorter input_ids
        # means a fresh prompt arrived. Reset all state to avoid carrying over
        # mid-action-segment counters into the new generation.
        cur_seq_len = int(input_ids.shape[1])
        if cur_seq_len < self._last_seq_len:
            self._remaining.zero_()
        self._last_seq_len = cur_seq_len

        # Detect entry: did the last-generated token equal action_start_id?
        last_tokens = input_ids[:, -1]
        just_entered = (last_tokens == self.action_start_id)
        # Set/reset remaining for rows that just entered (overwrites prior state
        # on re-entry, matching the documented "counter reset" semantics).
        self._remaining = torch.where(
            just_entered,
            torch.full_like(self._remaining, self.action_chunk_len),
            self._remaining,
        )

        active = self._remaining > 0
        if active.any():
            # Build a (vocab,) mask: 0 inside action range, -inf elsewhere.
            mask = torch.full(
                (scores.shape[1],), float("-inf"), dtype=scores.dtype, device=device
            )
            mask[self.action_begin_id : self.action_end_id] = 0.0
            # Apply per-row: scores + mask for active rows, scores unchanged for inactive.
            scores = torch.where(
                active.unsqueeze(1),
                scores + mask.unsqueeze(0),
                scores,
            )
            # Decrement only the active rows.
            self._remaining = torch.where(
                active,
                self._remaining - 1,
                self._remaining,
            )

        return scores
