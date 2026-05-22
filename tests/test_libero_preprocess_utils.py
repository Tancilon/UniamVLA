from __future__ import annotations

import numpy as np

from tools.preprocess.libero_preprocess_utils import (
    EpisodePlan,
    compute_episode_plan,
    gaussian_heatmap_from_pixel,
    mask_to_token_grid,
    pack_robot_obs,
    pointcloud_to_tcp_distance,
    select_render_gpus,
    select_segment_aware_future_tcp,
    smooth_active_targets,
)


def test_smooth_active_targets_removes_short_island():
    raw = ["A", "A", "A", "B", "A", "A", "C", "C", "C"]
    assert smooth_active_targets(raw, min_segment_len=2) == [
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "C",
        "C",
        "C",
    ]


def test_smooth_active_targets_keeps_real_switch():
    raw = ["A", "A", "A", "B", "B", "B"]
    assert smooth_active_targets(raw, min_segment_len=2) == raw


def test_mask_to_token_grid_shape_and_range():
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 1
    grid = mask_to_token_grid(mask, target_size=20)
    assert grid.shape == (1, 20, 20)
    assert grid.dtype == np.float32
    assert 0.0 <= float(grid.min()) <= float(grid.max()) <= 1.0
    assert float(grid[:, 8:12, 8:12].mean()) > 0.8


def test_gaussian_heatmap_from_pixel_peaks_near_expected_cell():
    heat = gaussian_heatmap_from_pixel(
        np.array([128.0, 128.0]), 256, 256, 20, sigma=1.5
    )
    assert heat.shape == (1, 20, 20)
    y, x = np.unravel_index(int(heat[0].argmax()), heat[0].shape)
    assert abs(x - 10) <= 1
    assert abs(y - 10) <= 1
    assert np.isclose(float(heat.max()), 1.0)


def test_pointcloud_to_tcp_distance():
    points = np.array([[0, 0, 0], [2, 0, 0], [5, 0, 0]], dtype=np.float32)
    tcp = np.array([[3, 0, 0]], dtype=np.float32)
    assert pointcloud_to_tcp_distance(points, tcp) == 1.0
    assert np.isinf(
        pointcloud_to_tcp_distance(np.zeros((0, 3), dtype=np.float32), tcp)
    )


def test_select_segment_aware_future_tcp_bounds_segment_by_action_horizon():
    tcp = np.arange(30, dtype=np.float32).reshape(10, 3)
    active = ["drawer"] * 8 + ["bowl"] * 2

    out = select_segment_aware_future_tcp(
        tcp,
        active_targets=active,
        frame_idx=1,
        local_window_size=4,
        action_chunk_horizon=3,
    )

    assert np.array_equal(out, tcp[1:4])


def test_select_segment_aware_future_tcp_fills_short_segment_with_local_window():
    tcp = np.arange(21, dtype=np.float32).reshape(7, 3)
    active = ["drawer", "drawer", "bowl", "bowl", "bowl", "bowl", "bowl"]

    out = select_segment_aware_future_tcp(
        tcp,
        active_targets=active,
        frame_idx=0,
        local_window_size=4,
        action_chunk_horizon=8,
    )

    assert np.array_equal(out, tcp[0:4])


def test_pack_robot_obs_is_15d_float32():
    obs = pack_robot_obs(
        ee_pos=np.array([1, 2, 3]),
        ee_ori=np.array([4, 5, 6]),
        joint_states=np.arange(7),
        gripper_states=np.array([0.1, 0.2]),
    )
    assert obs.shape == (15,)
    assert obs.dtype == np.float32
    assert np.allclose(
        obs[:7], np.array([1, 2, 3, 4, 5, 6, 0.1], dtype=np.float32)
    )
    assert np.allclose(obs[7:14], np.arange(7, dtype=np.float32))
    assert np.isclose(obs[14], np.float32(0.2))


def test_compute_episode_plan_assigns_contiguous_indices():
    frame_counts = {
        "task_a.hdf5": [3, 2],
        "task_b.hdf5": [4],
    }
    plans = compute_episode_plan(frame_counts)
    assert plans == {
        ("task_a.hdf5", 0): EpisodePlan(
            episode_index=0, row_start=0, length=3
        ),
        ("task_a.hdf5", 1): EpisodePlan(
            episode_index=1, row_start=3, length=2
        ),
        ("task_b.hdf5", 0): EpisodePlan(
            episode_index=2, row_start=5, length=4
        ),
    }


def test_select_render_gpus_defaults_to_visible_list():
    assert select_render_gpus("0,1,3", None) == ["0", "1", "3"]
    assert select_render_gpus("", None) == ["0"]
    assert select_render_gpus("0,1,2", "2,0") == ["2", "0"]
