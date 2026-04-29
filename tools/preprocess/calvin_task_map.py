"""Hardcoded mapping from CALVIN's 34 task labels to the PyBullet body/link
name used as the `pose_6d.object_id` in starVLA samples.

The object_ids on the right-hand side correspond to entries that
`PlayTableSimEnv.scene.objects` exposes at simulation time.

NOTE: The orchestrator will assert every value exists in `env.scene.objects`
at worker startup (introduced in Task 4).  A task label encountered at
runtime that is not in this map raises KeyError, which the orchestrator
converts to a dropped window under `--on_resolve_failure skip`.

Kept importable under Python 3.8 (calvin_env conda env) via
`from __future__ import annotations`.
"""
from __future__ import annotations

CALVIN_TASK_TO_OBJECT: dict[str, str] = {
    # Slider
    # PyBullet body__link convention: the fixed-scene body name is "table"
    # (set by the hydra scene config), and its child links come from the
    # URDF with a "_link" suffix, e.g. "slide_link", "drawer_link".
    "move_slider_left":        "table__slide_link",
    "move_slider_right":       "table__slide_link",
    # Drawer
    "open_drawer":             "table__drawer_link",
    "close_drawer":            "table__drawer_link",
    # Switch + button (both mounted on the shelf)
    "turn_on_lightbulb":       "table__switch_link",
    "turn_off_lightbulb":      "table__switch_link",
    "turn_on_led":             "table__button_link",
    "turn_off_led":            "table__button_link",
    # Per-color rotate / push (3 colors × 4 ops = 12 entries)
    "rotate_red_block_right":  "block_red",
    "rotate_red_block_left":   "block_red",
    "rotate_blue_block_right": "block_blue",
    "rotate_blue_block_left":  "block_blue",
    "rotate_pink_block_right": "block_pink",
    "rotate_pink_block_left":  "block_pink",
    "push_red_block_right":    "block_red",
    "push_red_block_left":     "block_red",
    "push_blue_block_right":   "block_blue",
    "push_blue_block_left":    "block_blue",
    "push_pink_block_right":   "block_pink",
    "push_pink_block_left":    "block_pink",
    # Lift: 3 colors × 3 locations = 9 entries
    "lift_red_block_table":    "block_red",
    "lift_red_block_slider":   "block_red",
    "lift_red_block_drawer":   "block_red",
    "lift_blue_block_table":   "block_blue",
    "lift_blue_block_slider":  "block_blue",
    "lift_blue_block_drawer":  "block_blue",
    "lift_pink_block_table":   "block_pink",
    "lift_pink_block_slider":  "block_pink",
    "lift_pink_block_drawer":  "block_pink",
    # Place (gripper-held block → receptacle; target = receptacle)
    "place_in_slider":         "table__slide_link",
    "place_in_drawer":         "table__drawer_link",
    # Stack / unstack — convention: target = block being picked up
    "stack_block":             "block_red",
    "unstack_block":           "block_red",
    # Push into drawer — target = destination receptacle (the drawer)
    "push_into_drawer":        "table__drawer_link",
}


def resolve_target_object(task_label: str) -> str:
    """Return the PyBullet object_id for `task_label`.

    Raises KeyError if the label is unknown.
    """
    return CALVIN_TASK_TO_OBJECT[task_label]
