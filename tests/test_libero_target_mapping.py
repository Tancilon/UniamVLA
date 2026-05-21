from __future__ import annotations

from pathlib import Path

import pytest

from tools.preprocess.libero_target_mapping import (
    LIBERO_STANDARD_SUITES,
    TASK_TARGET_POLICIES,
    get_task_policy,
    normalize_task_name,
    validate_policy_table,
)


EXPECTED_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
}


def test_standard_suite_list_is_four_suite_benchmark():
    assert LIBERO_STANDARD_SUITES == (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    )


def test_policy_table_has_40_entries_with_expected_suite_counts():
    counts = {suite: 0 for suite in EXPECTED_COUNTS}
    for key in TASK_TARGET_POLICIES:
        suite, task = key.split("/", 1)
        assert task
        counts[suite] += 1
    assert counts == EXPECTED_COUNTS
    assert len(TASK_TARGET_POLICIES) == 40


def test_normalize_task_name_strips_demo_suffix_and_extension():
    assert (
        normalize_task_name(
            "libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5"
        )
        == "libero_goal/open_the_middle_drawer_of_the_cabinet"
    )


def test_representative_single_and_multi_target_policies():
    bowl = get_task_policy(
        "libero_spatial",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5",
    )
    assert [t.body_pattern for t in bowl.candidate_targets] == ["black_bowl"]
    assert bowl.fallback_target_pattern == "black_bowl"
    assert bowl.part_grounding.enabled is False

    both = get_task_policy(
        "libero_10",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5",
    )
    assert [t.body_pattern for t in both.candidate_targets] == [
        "alphabet_soup",
        "cream_cheese",
    ]
    assert both.active_target_rule == "closest_pointcloud_to_future_tcp"

    stove = get_task_policy(
        "libero_goal",
        "turn_on_the_stove_demo.hdf5",
    )
    assert stove.part_grounding.enabled is True
    assert stove.part_grounding.grounding_level == "part"
    assert "button" in stove.part_grounding.geom_patterns


def test_policy_table_validates_against_local_hdf5_names_when_available():
    root = Path("datasets/libero")
    if not root.exists():
        pytest.skip("official LIBERO HDF5 root is not present")
    missing = validate_policy_table(root)
    assert missing == []
