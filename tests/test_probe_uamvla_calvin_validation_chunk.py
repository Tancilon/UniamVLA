from __future__ import annotations

import numpy as np

from tools.probes.probe_uamvla_calvin_validation_chunk import (
    ValidationWindow,
    _build_target_chunk,
    _normalize_action,
    _select_probe_frames,
)


def test_normalize_action_maps_bounds_to_unit_interval():
    action = np.array([-2.0, 0.0, 4.0], dtype=np.float32)
    action_min = np.array([-2.0, -1.0, 2.0], dtype=np.float32)
    action_max = np.array([2.0, 1.0, 6.0], dtype=np.float32)

    normalized = _normalize_action(action, action_min, action_max)

    np.testing.assert_allclose(normalized, [-1.0, 0.0, 0.0], atol=1e-6)


def test_build_target_chunk_skips_short_terminal_chunks():
    actions_by_frame = {
        10: np.arange(7, dtype=np.float32),
        11: np.arange(7, dtype=np.float32) + 1.0,
    }
    action_min = np.full(7, -10.0, dtype=np.float32)
    action_max = np.full(7, 10.0, dtype=np.float32)

    assert _build_target_chunk(
        actions_by_frame=actions_by_frame,
        start_frame=10,
        ep_end=11,
        horizon=8,
        action_min=action_min,
        action_max=action_max,
        min_valid_steps=3,
    ) is None


def test_build_target_chunk_returns_mask_for_valid_future_frames():
    actions_by_frame = {
        frame: np.full(7, frame, dtype=np.float32)
        for frame in range(20, 24)
    }
    action_min = np.zeros(7, dtype=np.float32)
    action_max = np.full(7, 40.0, dtype=np.float32)

    target, mask = _build_target_chunk(
        actions_by_frame=actions_by_frame,
        start_frame=20,
        ep_end=23,
        horizon=8,
        action_min=action_min,
        action_max=action_max,
        min_valid_steps=4,
    )

    assert target.shape == (8, 7)
    assert mask.shape == (8, 7)
    assert mask[:4].all()
    assert not mask[4:].any()
    np.testing.assert_allclose(target[0], np.zeros(7), atol=1e-6)
    np.testing.assert_allclose(target[3], np.full(7, 0.15), atol=1e-6)


def test_select_probe_frames_filters_task_and_respects_offset_and_start_index():
    windows = [
        ValidationWindow(0, 100, 163, "turn the blue block right", "rotate_blue_block_right"),
        ValidationWindow(1, 200, 263, "turn the red block right", "rotate_red_block_right"),
        ValidationWindow(2, 300, 363, "take the blue block and rotate it right", "rotate_blue_block_right"),
    ]

    selected = _select_probe_frames(
        windows=windows,
        instruction=None,
        contains=False,
        task="rotate_blue_block_right",
        task_contains=False,
        start_index=1,
        num_samples=1,
        frame_offset=40,
        horizon=8,
        min_valid_steps=8,
    )

    assert len(selected) == 1
    assert selected[0].window.window_idx == 2
    assert selected[0].frame == 340
    assert selected[0].step_idx == 40
