"""Unit tests for tools.preprocess.calvin_scene_splitter and calvin_split_merger.

Exercise the pure-Python paths only — no calvin_env / PyBullet needed.
The splitter's hydra rewrite step is exercised against a synthetic
`merged_config.yaml` plus dummy `calvin_scene_<X>.yaml` files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


# ---------- helpers --------------------------------------------------------

def _write_scene_info(split_dir: Path, ranges: dict[str, tuple[int, int]]) -> None:
    info = {f"scene_{k}": np.array([s, e], dtype=np.int64)
            for k, (s, e) in ranges.items()}
    np.save(split_dir / "scene_info.npy",
            np.array(info, dtype=object), allow_pickle=True)


def _write_lang_ann(split_dir: Path, windows: list[tuple[str, str, tuple[int, int]]]) -> None:
    """windows: list of (annotation, task_label, (ep_start, ep_end))."""
    (split_dir / "lang_annotations").mkdir(exist_ok=True)
    lang = {
        "language": {
            "ann":  [w[0] for w in windows],
            "task": [w[1] for w in windows],
            "emb":  np.zeros((len(windows), 4), dtype=np.float32),
        },
        "info": {
            "indx": [tuple(w[2]) for w in windows],
        },
    }
    np.save(split_dir / "lang_annotations" / "auto_lang_ann.npy",
            np.array(lang, dtype=object), allow_pickle=True)


def _write_synthetic_input(tmp_path: Path) -> Path:
    """Build a synthetic multi-scene CALVIN split."""
    src = tmp_path / "src"
    src.mkdir()
    _write_scene_info(src, {
        "A": (0, 99),
        "B": (100, 199),
        "C": (200, 299),
        "D": (300, 399),
    })
    _write_lang_ann(src, [
        ("rotate red right", "rotate_red_block_right", (10, 73)),    # A
        ("open drawer",      "open_drawer",            (110, 173)),  # B
        ("turn on led",      "turn_on_led",            (210, 273)),  # C
        ("push pink left",   "push_pink_block_left",   (310, 373)),  # D
    ])
    # Episode npz files — one per frame across all 4 scene ranges.
    # Touch a sparse subset to validate the symlink loop's "missing-file
    # tolerance" path.
    for t in (10, 73, 110, 200, 250, 350, 399):
        (src / f"episode_{t:07d}.npz").write_bytes(b"\0")
    # Hydra config with cfg.env.scene shape (matches CALVIN's merged_config).
    (src / ".hydra").mkdir()
    merged = {
        "env": {
            "_target_": "calvin_env.envs.play_table_env.PlayTableSimEnv",
            "scene": {"_target_": "old.scene.D"},
            "cameras": {"static": "x", "wrist": "y"},
        },
        "seed": 0,
    }
    with open(src / ".hydra" / "merged_config.yaml", "w") as f:
        yaml.safe_dump(merged, f)
    # Dummy scene config dir.
    cfg_dir = tmp_path / "scene_cfg"
    cfg_dir.mkdir()
    for letter in "ABCD":
        with open(cfg_dir / f"calvin_scene_{letter}.yaml", "w") as f:
            yaml.safe_dump({"_target_": f"calvin_env.scene.scene_{letter}",
                            "objects": {f"block_{letter}": "..."}}, f)
    return src


# ---------- splitter -------------------------------------------------------

def test_splitter_partitions_lang_windows_per_scene(tmp_path):
    from tools.preprocess.calvin_scene_splitter import split_calvin_by_scene

    src = _write_synthetic_input(tmp_path)
    out_root = tmp_path / "virtual"
    cfg_dir = tmp_path / "scene_cfg"

    splits = split_calvin_by_scene(
        input_dir=src, output_dir=out_root,
        scenes=("A", "B", "C", "D"),
        scene_config_dir=cfg_dir,
    )
    assert sorted(splits) == ["A", "B", "C", "D"]
    for letter, vs in splits.items():
        assert vs.is_dir()
        lang = np.load(vs / "lang_annotations" / "auto_lang_ann.npy",
                       allow_pickle=True).item()
        assert len(lang["language"]["ann"]) == 1, letter
        # The kept window's frame range must be the one matching this scene.
        idx = lang["info"]["indx"][0]
        scene_lo = {"A": 0, "B": 100, "C": 200, "D": 300}[letter]
        scene_hi = scene_lo + 99
        assert scene_lo <= int(idx[0]) <= scene_hi


def test_splitter_rewrites_hydra_scene_subtree(tmp_path):
    from tools.preprocess.calvin_scene_splitter import split_calvin_by_scene

    src = _write_synthetic_input(tmp_path)
    cfg_dir = tmp_path / "scene_cfg"
    splits = split_calvin_by_scene(
        input_dir=src, output_dir=tmp_path / "virtual",
        scenes=("A", "C"), scene_config_dir=cfg_dir,
    )

    for letter, vs in splits.items():
        cfg = yaml.safe_load((vs / ".hydra" / "merged_config.yaml").read_text())
        assert cfg["env"]["scene"]["_target_"].endswith(f"scene_{letter}"), letter
        # Other branches preserved verbatim.
        assert cfg["env"]["cameras"]["static"] == "x"
        assert cfg["seed"] == 0


def test_splitter_only_links_episodes_in_range(tmp_path):
    from tools.preprocess.calvin_scene_splitter import split_calvin_by_scene

    src = _write_synthetic_input(tmp_path)
    cfg_dir = tmp_path / "scene_cfg"
    splits = split_calvin_by_scene(
        input_dir=src, output_dir=tmp_path / "virtual",
        scenes=("A", "B"), scene_config_dir=cfg_dir,
    )
    a_eps = sorted(p.name for p in splits["A"].glob("episode_*.npz"))
    b_eps = sorted(p.name for p in splits["B"].glob("episode_*.npz"))
    # Frames touched in scene A's [0,99] range: 10, 73.
    assert a_eps == ["episode_0000010.npz", "episode_0000073.npz"]
    # Scene B's [100,199]: only 110.
    assert b_eps == ["episode_0000110.npz"]


def test_filter_lang_annotations_drops_episodes_field(tmp_path):
    """Real CALVIN dumps include `info.episodes` whose length is NOT
    parallel to `info.indx` (semantics differ by version). Earlier
    splitter naively indexed it with window indices and crashed.

    The filter must run cleanly regardless of the `episodes` shape and
    must NOT propagate it (no downstream reader needs it).
    """
    from tools.preprocess.calvin_scene_splitter import filter_lang_annotations

    src = tmp_path / "auto_lang_ann.npy"
    payload = {
        "language": {
            "ann":  ["a", "b", "c", "d"],
            "task": ["t0", "t1", "t2", "t3"],
            "emb":  np.zeros((4, 2), dtype=np.float32),
        },
        "info": {
            "indx": [(0, 10), (20, 30), (200, 210), (220, 230)],
            # Per-EPISODE list (length 2), NOT per-window (length 4).
            "episodes": [(0, 50), (200, 250)],
        },
    }
    np.save(src, np.array(payload, dtype=object), allow_pickle=True)

    out = filter_lang_annotations(src, frame_range=(0, 100))
    assert out["language"]["ann"] == ["a", "b"]
    assert out["language"]["task"] == ["t0", "t1"]
    assert "episodes" not in out["info"]
    assert out["language"]["emb"].shape == (2, 2)


def test_filter_lang_annotations_drops_mismatched_emb(tmp_path):
    """If `language.emb` length doesn't match `info.indx`, the filter
    should drop it with a warning rather than corrupt the alignment.
    """
    from tools.preprocess.calvin_scene_splitter import filter_lang_annotations

    src = tmp_path / "auto_lang_ann.npy"
    payload = {
        "language": {
            "ann":  ["a", "b"],
            "task": ["t0", "t1"],
            "emb":  np.zeros((5, 2), dtype=np.float32),  # wrong length on purpose
        },
        "info": {
            "indx": [(0, 10), (20, 30)],
        },
    }
    np.save(src, np.array(payload, dtype=object), allow_pickle=True)

    out = filter_lang_annotations(src, frame_range=(0, 100))
    assert out["language"]["ann"] == ["a", "b"]
    assert "emb" not in out["language"]


def test_splitter_accepts_calvin_scene_prefix_keys(tmp_path):
    """Real CALVIN dumps (e.g. task_ABCD_D shipped to users) use
    `calvin_scene_A/B/C/D` as the scene_info.npy key prefix, not the
    older `scene_A/B/C/D`. The splitter must accept both.
    """
    from tools.preprocess.calvin_scene_splitter import load_scene_ranges

    src = tmp_path / "src"
    src.mkdir()
    info = {
        "calvin_scene_A": np.array([0, 99], dtype=np.int64),
        "calvin_scene_D": np.array([300, 399], dtype=np.int64),
    }
    np.save(src / "scene_info.npy", np.array(info, dtype=object), allow_pickle=True)

    ranges = load_scene_ranges(src)
    assert ranges == {"A": (0, 99), "D": (300, 399)}


def test_splitter_skips_scenes_with_no_windows(tmp_path):
    from tools.preprocess.calvin_scene_splitter import split_calvin_by_scene

    src = tmp_path / "src"
    src.mkdir()
    # Scene A has range [0,99] but no windows fall inside.
    _write_scene_info(src, {"A": (0, 99), "B": (100, 199)})
    _write_lang_ann(src, [
        ("turn on led", "turn_on_led", (110, 173)),  # B only
    ])
    (src / ".hydra").mkdir()
    yaml.safe_dump(
        {"env": {"scene": {"_target_": "old"}}, "seed": 0},
        open(src / ".hydra" / "merged_config.yaml", "w"),
    )
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    for letter in "AB":
        yaml.safe_dump(
            {"_target_": f"new_{letter}"},
            open(cfg_dir / f"calvin_scene_{letter}.yaml", "w"),
        )

    from tools.preprocess.calvin_scene_splitter import split_calvin_by_scene
    out = split_calvin_by_scene(
        input_dir=src, output_dir=tmp_path / "virtual",
        scenes=("A", "B"), scene_config_dir=cfg_dir,
    )
    assert "A" not in out and "B" in out


# ---------- merger ---------------------------------------------------------

def _write_per_scene_pp_output(
    out_dir: Path,
    dataset_source: str,
    scene_letter: str,
    n_samples: int = 3,
) -> None:
    """Synthesize a fake preprocess output dir matching CalvinPreprocessor's layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in (
        "images/obs/static", "images/obs/wrist",
        "images/target", "images/future",
        "depth/static", "depth/wrist",
        "point_clouds",
    ):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    rng = np.random.default_rng({"A": 0, "B": 1, "C": 2, "D": 3}[scene_letter])
    # One image_future per episode (preprocessor writes the last frame's
    # static rgb here, keyed by episode_id — not sample id).
    eid = f"{dataset_source}_ep00000"
    (out_dir / "images/future" / f"{eid}.jpg").write_bytes(b"\0")
    for i in range(n_samples):
        sid = f"{dataset_source}_ep00000_step{i:04d}"
        # Touch one file per per-frame subdir using sid-prefixed names so
        # cross-scene merges don't collide.
        for sub in ("images/obs/static", "images/obs/wrist",
                    "depth/static", "depth/wrist"):
            (out_dir / sub / f"{sid}.jpg").write_bytes(b"\0")
        rows.append({
            "id": sid,
            "episode_id": f"{dataset_source}_ep00000",
            "step_idx": i,
            "total_steps": n_samples,
            "image": [f"images/obs/static/{sid}.jpg",
                      f"images/obs/wrist/{sid}.jpg"],
            "instruction": "do something",
            "embodiment": "franka_calvin",
            "action_dim": 7,
            "action": (rng.standard_normal(7) * 0.1).tolist() + [0.0] * 17,
            "action_mask": [1] * 7 + [0] * 17,
            "robot_obs": [0.10, -0.09, 0.57, 3.10, 0.02, 1.50, 0.08,
                          -0.59, 0.90, 1.74, -1.84, -0.92, 1.60, 0.59, 1.0],
            "dataset_source": dataset_source,
            "image_future": f"images/future/{dataset_source}_ep00000.jpg",
            "depth_static": f"depth/static/{sid}.npy",
            "depth_wrist":  f"depth/wrist/{sid}.npy",
            "static_cam_extrinsic": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                     "translation": [0, 0, 1]},
            "wrist_cam_extrinsic":  {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                     "translation": [0, 0, 0.5]},
        })
    with open(out_dir / "data.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # Minimal statistics.yaml carrying camera intrinsics — the merger
    # only reads the `cameras.{static,wrist}.intrinsic` paths.
    stats = {
        "view_names": ["static", "wrist"],
        "max_action_dim": 24,
        "embodiment_stats": {"franka_calvin": {
            "action_dim": 7,
            "action_min_bound": [-1] * 7,
            "action_max_bound": [1] * 7,
        }},
        "cameras": {
            "static": {"intrinsic": [[200, 0, 128], [0, 200, 128], [0, 0, 1]]},
            "wrist":  {"intrinsic": [[200, 0, 128], [0, 200, 128], [0, 0, 1]]},
        },
    }
    with open(out_dir / "statistics.yaml", "w") as f:
        yaml.safe_dump(stats, f, sort_keys=False)


def test_merger_concatenates_jsonl_with_unified_source(tmp_path):
    from tools.preprocess.calvin_split_merger import merge_scene_outputs

    pp_root = tmp_path / "pp"
    inputs = []
    for letter in "AB":
        d = pp_root / f"scene_{letter}"
        _write_per_scene_pp_output(
            d, dataset_source=f"task_ABCD_D_scene{letter}",
            scene_letter=letter, n_samples=3,
        )
        inputs.append(d)

    out = tmp_path / "merged"
    merge_scene_outputs(
        inputs=inputs, output_dir=out,
        dataset_source="task_ABCD_D",
    )

    rows = [json.loads(l) for l in (out / "data.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    # Every row's dataset_source rewritten.
    assert {r["dataset_source"] for r in rows} == {"task_ABCD_D"}
    # Episode ids preserve scene-letter disambiguation.
    eids = {r["episode_id"] for r in rows}
    assert eids == {"task_ABCD_D_sceneA_ep00000",
                    "task_ABCD_D_sceneB_ep00000"}
    # Sample ids are globally unique.
    assert len({r["id"] for r in rows}) == 6


def test_merger_links_per_frame_artifacts(tmp_path):
    from tools.preprocess.calvin_split_merger import merge_scene_outputs

    pp_root = tmp_path / "pp"
    inputs = []
    for letter in "AB":
        d = pp_root / f"scene_{letter}"
        _write_per_scene_pp_output(
            d, dataset_source=f"task_ABCD_D_scene{letter}",
            scene_letter=letter, n_samples=2,
        )
        inputs.append(d)

    out = tmp_path / "merged"
    merge_scene_outputs(
        inputs=inputs, output_dir=out,
        dataset_source="task_ABCD_D",
    )
    # Each scene contributed 2 static frames -> 4 total in merged dir.
    static_files = sorted((out / "images/obs/static").iterdir())
    assert len(static_files) == 4
    # image_future is one-per-episode — both scenes contributed their own.
    future_files = sorted((out / "images/future").iterdir())
    assert len(future_files) == 2


def test_merger_rerun_clears_stale_artifacts(tmp_path):
    """Rerunning the merger with a smaller/different input set must NOT leak
    files from the prior run into the merged dir.
    """
    from tools.preprocess.calvin_split_merger import merge_scene_outputs

    out = tmp_path / "merged"

    # First run: scenes A + B.
    pp_root = tmp_path / "pp"
    inputs = []
    for letter in "AB":
        d = pp_root / f"scene_{letter}"
        _write_per_scene_pp_output(
            d, dataset_source=f"task_ABCD_D_scene{letter}",
            scene_letter=letter, n_samples=2,
        )
        inputs.append(d)
    merge_scene_outputs(
        inputs=inputs, output_dir=out, dataset_source="task_ABCD_D",
    )
    assert len(list((out / "images/obs/static").iterdir())) == 4

    # Second run: only scene A. Stale B-frame files must be gone.
    merge_scene_outputs(
        inputs=[pp_root / "scene_A"], output_dir=out,
        dataset_source="task_ABCD_D",
    )
    static_files = list((out / "images/obs/static").iterdir())
    assert len(static_files) == 2
    assert all("sceneA" in p.name for p in static_files)


def test_merger_fallback_scene_letter_from_dirname(tmp_path):
    """If the per-scene preprocess used a dataset_source NOT ending in
    `_sceneX`, the merger falls back to the input dir's name suffix.
    """
    from tools.preprocess.calvin_split_merger import merge_scene_outputs

    pp_root = tmp_path / "pp"
    # Note: dataset_source has no `_scene<X>` suffix, but the dir is named scene_C.
    d = pp_root / "scene_C"
    _write_per_scene_pp_output(
        d, dataset_source="task_ABCD_D_run42",
        scene_letter="C", n_samples=2,
    )
    out = tmp_path / "merged"
    merge_scene_outputs(
        inputs=[d], output_dir=out, dataset_source="task_ABCD_D",
    )
    rows = [json.loads(l) for l in (out / "data.jsonl").read_text().splitlines()]
    # Scene letter inferred from dir name "scene_C" -> last char "C".
    assert all(r["episode_id"] == "task_ABCD_D_sceneC_ep00000" for r in rows)


def test_splitter_rewrites_env_scene_when_both_keys_present(tmp_path):
    """If merged_config.yaml has BOTH cfg.scene and cfg.env.scene (as some
    hydra dumps emit), the splitter MUST rewrite cfg.env.scene because that
    is the subtree CALVIN's eval/preproc instantiate.
    """
    from tools.preprocess.calvin_scene_splitter import rewrite_merged_config

    src_hydra = tmp_path / "src_hydra"
    src_hydra.mkdir()
    yaml.safe_dump(
        {
            "scene": {"_target_": "old.scene.D"},
            "env": {
                "_target_": "calvin_env.envs.play_table_env.PlayTableSimEnv",
                "scene": {"_target_": "old.scene.D"},
                "cameras": {"static": "x"},
            },
        },
        open(src_hydra / "merged_config.yaml", "w"),
    )
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    yaml.safe_dump(
        {"_target_": "calvin_env.scene.scene_A"},
        open(cfg_dir / "calvin_scene_A.yaml", "w"),
    )

    dst_hydra = tmp_path / "dst_hydra"
    rewrite_merged_config(
        src_hydra_dir=src_hydra, dst_hydra_dir=dst_hydra,
        scene_letter="A", scene_config_dir=cfg_dir,
    )
    cfg = yaml.safe_load((dst_hydra / "merged_config.yaml").read_text())
    # Both keys should now point at scene A.
    assert cfg["env"]["scene"]["_target_"] == "calvin_env.scene.scene_A"
    assert cfg["scene"]["_target_"] == "calvin_env.scene.scene_A"


def test_merger_regenerates_statistics(tmp_path):
    from tools.preprocess.calvin_split_merger import merge_scene_outputs

    pp_root = tmp_path / "pp"
    inputs = []
    for letter in "AB":
        d = pp_root / f"scene_{letter}"
        _write_per_scene_pp_output(
            d, dataset_source=f"task_ABCD_D_scene{letter}",
            scene_letter=letter, n_samples=4,
        )
        inputs.append(d)

    out = tmp_path / "merged"
    merge_scene_outputs(
        inputs=inputs, output_dir=out,
        dataset_source="task_ABCD_D",
    )

    stats = yaml.safe_load((out / "statistics.yaml").read_text())
    # Schema sanity — same shape as a normal preprocess run.
    assert stats["view_names"] == ["static", "wrist"]
    assert "franka_calvin" in stats["embodiment_stats"]
    assert "cameras" in stats and "static" in stats["cameras"]


def test_merger_detects_duplicate_ids(tmp_path):
    """If two scenes used the SAME dataset_source, the merger must fail loudly."""
    from tools.preprocess.calvin_split_merger import merge_scene_outputs

    pp_root = tmp_path / "pp"
    inputs = []
    for letter in "AB":
        d = pp_root / f"scene_{letter}"
        # Same source -> identical sample ids after rewrite.
        _write_per_scene_pp_output(
            d, dataset_source="task_ABCD_D_sceneA",   # <-- intentional dup
            scene_letter=letter, n_samples=2,
        )
        inputs.append(d)

    with pytest.raises(RuntimeError, match="Duplicate sample id"):
        merge_scene_outputs(
            inputs=inputs, output_dir=tmp_path / "merged",
            dataset_source="task_ABCD_D",
        )
