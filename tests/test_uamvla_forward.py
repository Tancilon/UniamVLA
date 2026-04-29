"""Tests for Task 18: UamVLA.forward (multi-aux-head loss summing).

The full forward path requires a real Qwen3-VL backbone (large checkpoint
download) and a working dataloader fixture (Task 24+). Both prerequisites
are out of scope for this test file — the L2 1-step training smoke
(Task 31) exercises the full pipeline.

Pure-tensor collator helpers used by ``forward`` are covered separately
in ``tests/test_collator_helpers.py``.
"""
import pytest


def test_forward_returns_action_loss_key():
    """Full forward test requires backbone + dataloader fixtures — defer to L2 1-step training smoke (Task 31)."""
    pytest.skip("Full forward test requires backbone + dataloader fixtures — defer to L2 1-step training smoke")
