import pytest


def test_predict_action_returns_normalized_actions_key():
    """Output dict must have 'normalized_actions' as np.ndarray with shape (B, T, 7) for Franka.
    Full test deferred to L5 eval dry-run (Task 33)."""
    pytest.skip(
        "Full predict_action test requires backbone weights + action_token_begin_id wiring"
        " — defer to L5 eval dry-run"
    )
