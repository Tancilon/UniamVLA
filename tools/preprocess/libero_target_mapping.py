from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


LIBERO_STANDARD_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


@dataclass(frozen=True)
class TargetSpec:
    body_pattern: str
    kind: str = "object"


@dataclass(frozen=True)
class PartGroundingSpec:
    enabled: bool = False
    body_patterns: tuple[str, ...] = ()
    geom_patterns: tuple[str, ...] = ()
    grounding_level: str = "object"


@dataclass(frozen=True)
class TaskTargetPolicy:
    suite: str
    task_name: str
    candidate_targets: tuple[TargetSpec, ...]
    active_target_rule: str = "closest_pointcloud_to_future_tcp"
    fallback_target_pattern: str = ""
    part_grounding: PartGroundingSpec = field(default_factory=PartGroundingSpec)

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.task_name}"


def _policy(
    suite: str,
    task_name: str,
    candidates: tuple[str, ...],
    fallback: str,
    part: PartGroundingSpec | None = None,
) -> TaskTargetPolicy:
    return TaskTargetPolicy(
        suite=suite,
        task_name=task_name,
        candidate_targets=tuple(TargetSpec(c) for c in candidates),
        fallback_target_pattern=fallback,
        part_grounding=part or PartGroundingSpec(),
    )


DRAWER_PART = PartGroundingSpec(
    enabled=True,
    body_patterns=("wooden_cabinet",),
    geom_patterns=("handle", "drawer", "cabinet"),
    grounding_level="part",
)
STOVE_PART = PartGroundingSpec(
    enabled=True,
    body_patterns=("stove",),
    geom_patterns=("button", "knob"),
    grounding_level="part",
)
MICROWAVE_PART = PartGroundingSpec(
    enabled=True,
    body_patterns=("microwave",),
    geom_patterns=("handle", "door"),
    grounding_level="part",
)


_POLICIES = [
    _policy("libero_spatial", "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate", ("black_bowl", "wooden_cabinet"), "black_bowl", DRAWER_PART),
    _policy("libero_spatial", "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_spatial", "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_object", "pick_up_the_alphabet_soup_and_place_it_in_the_basket", ("alphabet_soup",), "alphabet_soup"),
    _policy("libero_object", "pick_up_the_bbq_sauce_and_place_it_in_the_basket", ("bbq_sauce",), "bbq_sauce"),
    _policy("libero_object", "pick_up_the_butter_and_place_it_in_the_basket", ("butter",), "butter"),
    _policy("libero_object", "pick_up_the_chocolate_pudding_and_place_it_in_the_basket", ("chocolate_pudding",), "chocolate_pudding"),
    _policy("libero_object", "pick_up_the_cream_cheese_and_place_it_in_the_basket", ("cream_cheese",), "cream_cheese"),
    _policy("libero_object", "pick_up_the_ketchup_and_place_it_in_the_basket", ("ketchup",), "ketchup"),
    _policy("libero_object", "pick_up_the_milk_and_place_it_in_the_basket", ("milk",), "milk"),
    _policy("libero_object", "pick_up_the_orange_juice_and_place_it_in_the_basket", ("orange_juice",), "orange_juice"),
    _policy("libero_object", "pick_up_the_salad_dressing_and_place_it_in_the_basket", ("salad_dressing",), "salad_dressing"),
    _policy("libero_object", "pick_up_the_tomato_sauce_and_place_it_in_the_basket", ("tomato_sauce",), "tomato_sauce"),
    _policy("libero_goal", "open_the_middle_drawer_of_the_cabinet", ("wooden_cabinet",), "wooden_cabinet", DRAWER_PART),
    _policy("libero_goal", "open_the_top_drawer_and_put_the_bowl_inside", ("wooden_cabinet", "black_bowl"), "black_bowl", DRAWER_PART),
    _policy("libero_goal", "push_the_plate_to_the_front_of_the_stove", ("plate",), "plate"),
    _policy("libero_goal", "put_the_bowl_on_the_plate", ("black_bowl",), "black_bowl"),
    _policy("libero_goal", "put_the_bowl_on_the_stove", ("black_bowl",), "black_bowl"),
    _policy("libero_goal", "put_the_bowl_on_top_of_the_cabinet", ("black_bowl",), "black_bowl"),
    _policy("libero_goal", "put_the_cream_cheese_in_the_bowl", ("cream_cheese",), "cream_cheese"),
    _policy("libero_goal", "put_the_wine_bottle_on_the_rack", ("wine_bottle",), "wine_bottle"),
    _policy("libero_goal", "put_the_wine_bottle_on_top_of_the_cabinet", ("wine_bottle",), "wine_bottle"),
    _policy("libero_goal", "turn_on_the_stove", ("stove",), "stove", STOVE_PART),
    _policy("libero_10", "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", ("stove", "moka_pot"), "moka_pot", STOVE_PART),
    _policy("libero_10", "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", ("black_bowl", "wooden_cabinet"), "black_bowl", DRAWER_PART),
    _policy("libero_10", "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it", ("yellow_and_white_mug", "microwave"), "yellow_and_white_mug", MICROWAVE_PART),
    _policy("libero_10", "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove", ("moka_pot",), "moka_pot"),
    _policy("libero_10", "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", ("alphabet_soup", "cream_cheese"), "alphabet_soup"),
    _policy("libero_10", "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket", ("alphabet_soup", "tomato_sauce"), "alphabet_soup"),
    _policy("libero_10", "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket", ("cream_cheese", "butter"), "cream_cheese"),
    _policy("libero_10", "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate", ("white_mug", "yellow_and_white_mug"), "white_mug"),
    _policy("libero_10", "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate", ("white_mug", "chocolate_pudding"), "white_mug"),
    _policy("libero_10", "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy", ("book",), "book"),
]

TASK_TARGET_POLICIES = {p.key: p for p in _POLICIES}


def normalize_task_name(path_or_name: str | Path) -> str:
    path = Path(path_or_name)
    suite = path.parent.name if path.parent.name in LIBERO_STANDARD_SUITES else ""
    stem = path.name
    if stem.endswith(".hdf5"):
        stem = stem[:-5]
    if stem.endswith("_demo"):
        stem = stem[:-5]
    return f"{suite}/{stem}" if suite else stem


def get_task_policy(
    suite: str,
    hdf5_name_or_task_name: str | Path,
) -> TaskTargetPolicy:
    task = normalize_task_name(Path(suite) / Path(hdf5_name_or_task_name).name)
    if task not in TASK_TARGET_POLICIES:
        raise KeyError(f"No curated LIBERO target policy for {task}")
    return TASK_TARGET_POLICIES[task]


def validate_policy_table(libero_root: str | Path) -> list[str]:
    root = Path(libero_root)
    missing: list[str] = []
    for suite in LIBERO_STANDARD_SUITES:
        for h5_path in sorted((root / suite).glob("*.hdf5")):
            key = normalize_task_name(h5_path)
            if key not in TASK_TARGET_POLICIES:
                missing.append(key)
    return missing
