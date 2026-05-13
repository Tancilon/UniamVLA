"""Mapping from CALVIN's 34 task labels to the PyBullet body/link name
used as the `pose_6d.object_id` in starVLA samples.

The 32 color-/fixture-specific tasks resolve deterministically via
`resolve_target_object`. The remaining two — `stack_block` and
`unstack_block` — are color-agnostic in CALVIN: the manipulated block
varies per trajectory, so the target MUST be inferred from the window's
scene_obs at runtime. Callers route those two tasks through
`infer_stack_block`, which also needs the scene letter (A/B/C/D) because
each CALVIN scene yaml lists `movable_objects` in a different order, and
scene_obs follows that order.

The object_ids on the right-hand side correspond to entries that
`PlayTableSimEnv.scene.objects` exposes at simulation time. Resolver
failures show up as `KeyError` to the caller; the preprocessor converts
those to dropped windows under `--on_resolve_failure skip`.

Kept importable under Python 3.8 (calvin_env conda env) via
`from __future__ import annotations`.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

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
    # Push into drawer — target = destination receptacle (the drawer)
    "push_into_drawer":        "table__drawer_link",
}

# Tasks whose target cannot be resolved from the label alone — call
# `infer_stack_block` with the window's start/end scene_obs instead.
STACK_TASKS: frozenset[str] = frozenset({"stack_block", "unstack_block"})


# Per-scene movable_objects iteration order. Read directly from
# third_party/calvin/calvin_env/conf/scene/calvin_scene_<X>.yaml's
# `objects.movable_objects` dict — Python 3.7+ preserves insertion order,
# `PlayTableScene.load()` iterates that dict to populate
# `self.movable_objects`, and `PlayTableScene.get_obs()` appends each
# block's `(xyz, euler)` in that same order. The order differs per scene:
#
#   scene A: pink, blue, red
#   scene B: red,  blue, pink
#   scene C: blue, red,  pink
#   scene D: red,  blue, pink   (D_eval shares D's order)
#
# Treating every scene as scene A's order — the bug previously here —
# silently corrupts the inferred target for non-A windows.
SCENE_BLOCK_ORDER: dict[str, tuple[str, str, str]] = {
    "A": ("block_pink", "block_blue", "block_red"),
    "B": ("block_red",  "block_blue", "block_pink"),
    "C": ("block_blue", "block_red",  "block_pink"),
    "D": ("block_red",  "block_blue", "block_pink"),
}

# CALVIN scene_obs layout (24-dim, euler_obs=True — the only mode CALVIN
# ABCD_D / ABC_D dumps ship). `PlayTableScene.get_obs()` concatenates
#   [door states (slider, drawer), button, switch, lights (lightbulb, led),
#    movable_objects.get_state() in YAML order, each (xyz, euler)].
# That gives a 6-dim prefix of fixtures/lights followed by 3 × 6-dim
# blocks; per-block xyz lives at the first 3 entries of its slot.
_SCENE_OBS_DIM = 24
_PREFIX_DIMS = 6
_PER_BLOCK_DIMS = 6  # xyz (3) + euler (3)


def resolve_target_object(task_label: str) -> str:
    """Return the PyBullet object_id for `task_label`.

    Raises KeyError if the label is unknown. Raises ValueError for the
    color-agnostic stack/unstack tasks — those require runtime inference
    via `infer_stack_block`, and silently falling back to a hardcoded
    color here mislabels ~2/3 of stack/unstack windows.
    """
    if task_label in STACK_TASKS:
        raise ValueError(
            f"Task {task_label!r} is color-agnostic in CALVIN; resolve "
            f"its target via infer_stack_block(task_label, scene_letter, "
            f"scene_obs_start, scene_obs_end) instead."
        )
    return CALVIN_TASK_TO_OBJECT[task_label]


def infer_stack_block(
    task_label: str,
    scene_letter: str,
    scene_obs_start: np.ndarray,
    scene_obs_end: np.ndarray,
) -> str:
    """Pick the colored block manipulated in a stack_block/unstack_block window.

    The robot moves exactly one block during a stack or unstack action:
    for `unstack_block` it lifts the top block off the stack; for
    `stack_block` it places the held block onto another. Either way, the
    manipulated block's xyz position changes by ~tens of cm between the
    window's start and end frames, while the other two blocks stay
    essentially still (sub-mm physics jitter at most). The block with
    the largest start→end xyz displacement is therefore the intended
    target — the moved/manipulated block (the one the gripper is on),
    not the support block underneath. This matches the convention the
    previous hardcoded mapping documented in `CALVIN_TASK_TO_OBJECT`.

    Why this exists: CALVIN's `stack_block` / `unstack_block` task labels
    do NOT encode color. A hardcoded "always block_red" fallback (the
    previous behavior) mislabelled the target in roughly 2/3 of
    stack/unstack windows.

    `scene_letter` must be one of "A"/"B"/"C"/"D" (case-insensitive).
    Each scene lists `movable_objects` in a different order, so the
    slot→block-name mapping is scene-dependent. Pass it through from the
    `SceneResolver` that already drives the rest of the preprocessor.

    Both `scene_obs_start` and `scene_obs_end` must be 24-dim arrays in
    CALVIN's euler_obs layout, free of NaN/Inf.
    """
    if task_label not in STACK_TASKS:
        raise ValueError(
            f"infer_stack_block called with non-stack task {task_label!r}; "
            f"valid tasks are {sorted(STACK_TASKS)}."
        )

    letter = scene_letter.upper() if isinstance(scene_letter, str) else scene_letter
    try:
        order = SCENE_BLOCK_ORDER[letter]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CALVIN scene letter {scene_letter!r}; expected one "
            f"of {sorted(SCENE_BLOCK_ORDER)}. Block-slot order in "
            f"scene_obs is scene-dependent (each scene yaml lists "
            f"movable_objects in a different order)."
        ) from exc

    for name, arr in (("start", scene_obs_start), ("end", scene_obs_end)):
        if arr.shape != (_SCENE_OBS_DIM,):
            raise ValueError(
                f"scene_obs_{name} must have shape ({_SCENE_OBS_DIM},), "
                f"got {arr.shape}. The CALVIN ABCD_D / ABC_D dumps used "
                f"by this repo are 24-dim (euler_obs=True). A 27-dim "
                f"(quaternion) array, a multi-frame stack, or any other "
                f"shape will index the wrong block slots."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"scene_obs_{name} contains non-finite values (NaN or Inf); "
                f"cannot compute block displacement reliably."
            )

    best_name: str | None = None
    best_disp = -1.0
    per_block: list[tuple[str, float]] = []
    for i, block_name in enumerate(order):
        xyz_lo = _PREFIX_DIMS + i * _PER_BLOCK_DIMS
        xyz_slot = slice(xyz_lo, xyz_lo + 3)
        disp = float(np.linalg.norm(
            scene_obs_end[xyz_slot] - scene_obs_start[xyz_slot]
        ))
        per_block.append((block_name, disp))
        if disp > best_disp:
            best_disp = disp
            best_name = block_name
    assert best_name is not None  # SCENE_BLOCK_ORDER values are non-empty

    # Sanity guard. A real stack/unstack annotation involves the gripper
    # moving one block by tens of cm; the other two should drift by <1mm.
    # If `best_disp` is below ~1cm or another block came within 50% of
    # the winner, the heuristic is on thin ice — surface a WARNING so
    # dataset audits can investigate, but still return the best guess
    # (CALVIN's annotation says SOME stack/unstack motion happened in
    # this window, so falling back to KeyError would lose the sample
    # without giving the operator a chance to vet it).
    sorted_disps = sorted((d for _, d in per_block), reverse=True)
    if best_disp < _MIN_PICK_DISP:
        logger.warning(
            "infer_stack_block(%s, scene=%s) -> %s but max displacement "
            "is only %.4f m (< %.3f m sanity floor). Per-block: %s. "
            "Check the annotation and the window's scene_obs.",
            task_label, letter, best_name, best_disp, _MIN_PICK_DISP,
            ", ".join(f"{n}={d:.4f}" for n, d in per_block),
        )
    elif len(sorted_disps) >= 2 and sorted_disps[1] > _TIE_FRACTION * sorted_disps[0]:
        logger.warning(
            "infer_stack_block(%s, scene=%s) -> %s but the runner-up's "
            "displacement is >%.0f%% of the winner's (per-block: %s). "
            "Two blocks moved comparably — the picked target may be "
            "wrong.",
            task_label, letter, best_name, _TIE_FRACTION * 100,
            ", ".join(f"{n}={d:.4f}" for n, d in per_block),
        )
    else:
        logger.debug(
            "infer_stack_block(%s, scene=%s) -> %s (displacements: %s)",
            task_label, letter, best_name,
            ", ".join(f"{n}={d:.3f}" for n, d in per_block),
        )
    return best_name


# Tuning constants for the sanity warnings above. `_MIN_PICK_DISP` is
# the floor below which we don't trust the picked block (CALVIN block
# manipulations are tens of cm; sub-1cm is jitter territory).
# `_TIE_FRACTION` flags the case where two blocks moved by comparable
# amounts (e.g. >70% of the winner's displacement) — suggests the
# gripper bumped a second block hard enough that the heuristic is
# unreliable.
_MIN_PICK_DISP = 0.01     # 1 cm
_TIE_FRACTION = 0.70
