"""Tests for ActionLogitsProcessor — generate-time logits constraint for action chunks."""
import pytest
import torch


def _make_processor(action_start_id=42, action_begin_id=100, n_bins=4, action_chunk_len=3):
    """Helper: small, debuggable parameter set."""
    from starVLA.model.modules.uamvla.inference.action_logits_processor import (
        ActionLogitsProcessor,
    )
    return ActionLogitsProcessor(
        action_start_id=action_start_id,
        action_begin_id=action_begin_id,
        n_bins=n_bins,
        action_chunk_len=action_chunk_len,
    )


def _scores(B, vocab=200):
    """Helper: deterministic scores for inspection."""
    return torch.arange(B * vocab, dtype=torch.float32).view(B, vocab)


def test_processor_inactive_before_action_start():
    """Without seeing action_start_id, scores must pass through unchanged."""
    p = _make_processor()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    scores = _scores(B=1)
    out = p(input_ids, scores.clone())
    assert torch.equal(out, scores), "scores were modified before action_start was seen"


def test_processor_activates_immediately_after_action_start():
    """The step right after action_start: only [action_begin_id, action_begin_id+n_bins) survives."""
    p = _make_processor(action_begin_id=100, n_bins=4, action_chunk_len=3)
    # Last generated token IS action_start_id (42) → mask the next prediction.
    input_ids = torch.tensor([[1, 2, 42]])
    scores = _scores(B=1)
    out = p(input_ids, scores.clone())

    # Action positions [100..104) untouched
    assert torch.equal(out[0, 100:104], scores[0, 100:104])
    # Everything else is -inf
    mask = torch.ones(scores.shape[1], dtype=torch.bool)
    mask[100:104] = False
    assert torch.isinf(out[0, mask]).all()
    assert (out[0, mask] < 0).all()  # specifically -inf, not +inf


def test_processor_masks_for_exactly_action_chunk_len_steps():
    """Mask is active for action_chunk_len total steps, then automatically deactivates."""
    p = _make_processor(action_begin_id=100, n_bins=4, action_chunk_len=3)
    # Step 0: action_start in input_ids → mask THIS step
    input_ids = torch.tensor([[42]])
    s0 = _scores(B=1)
    out0 = p(input_ids, s0.clone())
    assert torch.isinf(out0[0, 0]), "step 0 should be masked"

    # Step 1: a regular action token in input_ids (not action_start) → should still mask
    input_ids = torch.tensor([[42, 101]])
    s1 = _scores(B=1)
    out1 = p(input_ids, s1.clone())
    assert torch.isinf(out1[0, 0]), "step 1 should still be masked"

    # Step 2: another action token → still mask (3rd and final masked step)
    input_ids = torch.tensor([[42, 101, 102]])
    s2 = _scores(B=1)
    out2 = p(input_ids, s2.clone())
    assert torch.isinf(out2[0, 0]), "step 2 should still be masked (final masked step)"

    # Step 3: action_chunk_len exhausted → mask must be off
    input_ids = torch.tensor([[42, 101, 102, 103]])
    s3 = _scores(B=1)
    out3 = p(input_ids, s3.clone())
    assert torch.equal(out3, s3), "step 3 should be unmasked (action_chunk_len exhausted)"


def test_processor_per_row_independence():
    """Two batch rows enter action mode at different times — masks must be independent."""
    p = _make_processor(action_begin_id=100, n_bins=4, action_chunk_len=2)

    # Row 0 just emitted action_start; row 1 emitted a normal token.
    input_ids = torch.tensor([[1, 2, 42], [1, 2, 9]])
    scores = _scores(B=2)
    out = p(input_ids, scores.clone())

    # Row 0 masked: only [100..104) survives
    assert torch.isinf(out[0, 0])
    assert torch.equal(out[0, 100:104], scores[0, 100:104])
    # Row 1 unchanged
    assert torch.equal(out[1], scores[1])


def test_processor_resets_counter_on_re_entry():
    """A second action_start within the same generate() call resets the counter."""
    p = _make_processor(action_begin_id=100, n_bins=4, action_chunk_len=2)

    # Enter action mode
    p(torch.tensor([[42]]), _scores(1).clone())  # step 0: mask, _remaining→1
    p(torch.tensor([[42, 101]]), _scores(1).clone())  # step 1: mask, _remaining→0

    # Re-enter — last token is action_start again
    out = p(torch.tensor([[42, 101, 42]]), _scores(1).clone())
    # Should be masked again (counter reset to action_chunk_len=2)
    assert torch.isinf(out[0, 0])


def test_processor_resets_state_on_new_generate_call():
    """Reusing an instance across generate() calls: state must auto-reset
    when a shorter input_ids (new prompt) arrives."""
    p = _make_processor(action_begin_id=100, n_bins=4, action_chunk_len=3)

    # Simulate generate() call 1: enter action mode at step 0, advance to step 1.
    p(torch.tensor([[1, 2, 3, 42]]), _scores(B=1).clone())  # seq_len=4, just entered, _remaining→2 after decrement
    p(torch.tensor([[1, 2, 3, 42, 101]]), _scores(B=1).clone())  # seq_len=5, still masking, _remaining→1

    # Simulate generate() call 2 with a fresh, shorter prompt — should reset state.
    # The first token of the new prompt is NOT action_start, so masking should be OFF.
    out = p(torch.tensor([[5, 6]]), _scores(B=1).clone())  # seq_len=2 < 5
    expected = _scores(B=1)
    assert torch.equal(out, expected), (
        "state did not reset on new generate(): mask still active in fresh sequence"
    )


def test_processor_raises_on_out_of_bounds_action_range():
    """If action range exceeds vocab, raise ValueError on first call."""
    p = _make_processor(action_begin_id=180, n_bins=50, action_chunk_len=2)
    # vocab=200 by default in _scores; action_end_id = 180+50 = 230 > 200 → must raise
    with pytest.raises(ValueError, match="exceeds vocab size"):
        p(torch.tensor([[1, 2, 3]]), _scores(B=1))
