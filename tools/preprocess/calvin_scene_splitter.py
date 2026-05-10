"""Partition a CALVIN multi-scene split into per-scene virtual splits.

Why this exists
---------------
The current preprocessor builds one PyBullet env per split run, and that env's
static fixtures (switch / button / drawer / slider) come from the URDF picked
by the dataset's `.hydra/merged_config.yaml`. CALVIN's per-frame `scene_obs`
only carries dynamic state (joint positions of fixtures + 3-block 6D poses);
the fixture **positions** are baked into the URDF at construction time.

For multi-scene splits (`task_ABCD_D`, `task_ABC_D`), this means: when the
preprocessor renders a frame whose `scene_obs` was originally captured under
scene A, but the env's URDF was loaded for scene D, the block xyz lands at
A's table coords while the switch/button/drawer are at D's positions —
two bodies overlap visually (PyBullet's `env.reset` does not resolve
collisions). Result: garbage rgb_static for ~3/4 of training frames.

This module builds N virtual single-scene split directories under a work
dir. Each virtual split:
  • symlinks **only** the `episode_*.npz` files inside its scene's frame range
  • ships a filtered `lang_annotations/auto_lang_ann.npy` (only its windows)
  • copies the original `.hydra/` and rewrites `merged_config.yaml` so
    `cfg.env.scene` points at the target scene's calvin_env scene config

The caller then runs `runners/preprocess_calvin.py` once per virtual split,
each with the right URDF baked in. A separate merger (calvin_split_merger.py)
fuses the per-scene outputs into one UAM-format dataset.

CLI
---
    python -m tools.preprocess.calvin_scene_splitter \\
        --input_dir  /path/to/task_ABCD_D/training \\
        --output_dir /tmp/abcd_scene_split \\
        [--scenes A,B,C,D] \\
        [--scene_config_dir /path/to/calvin_env/conf/scene]

If `--scene_config_dir` is omitted, the splitter tries to import
`calvin_env` and resolve the conf dir from the package layout. If both
fail, it errors out with a copy-pasteable hint to find the right path.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

VALID_SCENES = ("A", "B", "C", "D")

# Both legacy CALVIN dumps (`scene_A`) and newer dumps (`calvin_scene_A`) are
# in the wild. The preprocessor's older SceneResolver did a naive
# `.replace("scene_", "")` which left `calvin_A` for the latter — that
# happens to be harmless there because the resolved letter string is only
# used for an info log. The splitter actually USES the letter, so parse
# strictly here.
_SCENE_KEY_RE = re.compile(r"^(?:calvin_)?scene_([A-D])$", re.IGNORECASE)


# ---------- scene-config discovery -----------------------------------------

def _candidate_scene_config_dirs() -> list[Path]:
    """Return likely on-disk locations of calvin_env's scene YAMLs.

    Tried in order. First hit that contains all needed `calvin_scene_X.yaml`
    files wins. We do NOT eagerly probe the filesystem — we just enumerate
    candidates; the caller filters.
    """
    out: list[Path] = []
    try:
        import calvin_env  # noqa: WPS433 — runtime only on the CALVIN env
    except Exception as exc:
        logger.info(
            "calvin_env not importable here (%s); skipping auto-discovery.",
            exc,
        )
        return out

    pkg_root = Path(calvin_env.__file__).resolve().parent
    # Common layouts in the calvin_env source tree:
    #   <pkg>/conf/scene/calvin_scene_A.yaml          # vendored conf
    #   <pkg>/../conf/scene/calvin_scene_A.yaml       # editable install
    #   <pkg>/../../calvin_env/conf/scene/...         # src-layout install
    out.extend([
        pkg_root / "conf" / "scene",
        pkg_root.parent / "conf" / "scene",
        pkg_root.parent.parent / "calvin_env" / "conf" / "scene",
    ])
    return out


def resolve_scene_config_dir(
    explicit: Path | None,
    scenes: Iterable[str],
) -> Path:
    """Pick a directory that contains `calvin_scene_<X>.yaml` for every X in `scenes`.

    Order:
      1. `explicit` (the --scene_config_dir CLI arg) if provided
      2. auto-discovered candidates from calvin_env package layout
    """
    needed = [f"calvin_scene_{s.upper()}.yaml" for s in scenes]
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.extend(_candidate_scene_config_dirs())

    for cand in candidates:
        if not cand.is_dir():
            continue
        if all((cand / n).exists() for n in needed):
            logger.info("Using scene config dir: %s", cand)
            return cand

    tried = "\n  ".join(str(c) for c in candidates) or "(none — calvin_env not importable)"
    raise FileNotFoundError(
        f"Could not locate calvin_env scene configs for {sorted(scenes)}.\n"
        f"Tried:\n  {tried}\n\n"
        f"Pass --scene_config_dir <path> pointing at the directory that "
        f"contains {needed}. Find it on the server with:\n"
        f"  find / -name 'calvin_scene_A.yaml' 2>/dev/null | head"
    )


# ---------- scene_info parsing ---------------------------------------------

def load_scene_ranges(split_dir: Path) -> dict[str, tuple[int, int]]:
    """Return {scene_letter: (start, end)} from <split>/scene_info.npy.

    Both bounds inclusive — matching the convention used elsewhere in this
    file (CALVIN's npy ships them as inclusive ranges).
    """
    info_path = Path(split_dir) / "scene_info.npy"
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} missing. The splitter only handles multi-scene "
            f"splits that ship scene_info.npy (task_ABC_D, task_ABCD_D)."
        )
    info = np.load(info_path, allow_pickle=True).item()
    ranges: dict[str, tuple[int, int]] = {}
    for key, rng in info.items():
        m = _SCENE_KEY_RE.match(str(key))
        if not m:
            raise ValueError(
                f"Unexpected scene key in {info_path}: {key!r}. "
                f"Expected one of: scene_A..D or calvin_scene_A..D."
            )
        letter = m.group(1).upper()
        start, end = int(rng[0]), int(rng[1])
        if end < start:
            raise ValueError(
                f"Empty/inverted range for scene {letter}: [{start}, {end}]"
            )
        ranges[letter] = (start, end)
    return ranges


# ---------- lang annotation filtering --------------------------------------

def filter_lang_annotations(
    src_lang_path: Path,
    frame_range: tuple[int, int],
) -> dict:
    """Return a copy of auto_lang_ann.npy keeping only windows fully inside frame_range.

    A window is kept iff its [ep_start, ep_end] is fully contained in
    [frame_range[0], frame_range[1]] (matching SceneResolver.resolve_for_window
    in calvin_preprocessor.py).
    """
    src = np.load(src_lang_path, allow_pickle=True).item()
    anns = src["language"]["ann"]
    tasks = src["language"]["task"]
    embs = src["language"].get("emb")
    indx = src["info"]["indx"]
    ep_start_end = src["info"].get("episodes")  # CALVIN v2: list[(s, e)] per window

    lo, hi = frame_range
    keep: list[int] = []
    for i, rng in enumerate(indx):
        s, e = int(rng[0]), int(rng[1])
        if lo <= s <= hi and lo <= e <= hi:
            keep.append(i)

    new_anns = [anns[i] for i in keep]
    new_tasks = [tasks[i] for i in keep]
    new_indx = [indx[i] for i in keep]

    out = {
        "language": {
            "ann": new_anns,
            "task": new_tasks,
        },
        "info": {
            "indx": new_indx,
        },
    }
    if embs is not None:
        # `emb` is typically a (N, D) ndarray; index along axis 0.
        try:
            out["language"]["emb"] = np.asarray(embs)[keep]
        except Exception:
            out["language"]["emb"] = [embs[i] for i in keep]
    if ep_start_end is not None:
        out["info"]["episodes"] = [ep_start_end[i] for i in keep]
    return out


# ---------- hydra config rewrite -------------------------------------------

def rewrite_merged_config(
    src_hydra_dir: Path,
    dst_hydra_dir: Path,
    scene_letter: str,
    scene_config_dir: Path,
) -> None:
    """Copy `src_hydra_dir/` to `dst_hydra_dir/`, rewriting merged_config.yaml's scene.

    The realistic layout (per `examples/calvin/eval_files/eval_calvin.py:198-210`)
    is `cfg.env.scene` — that's the subtree CALVIN's eval path
    `hydra.utils.instantiate(cfg.env, ...)` actually consumes. We rewrite
    `cfg.env.scene` whenever present. We also rewrite a top-level
    `cfg.scene` if it exists, defensively, so the file stays internally
    consistent (some hydra dumps duplicate the subtree). At least one of
    the two MUST be present, otherwise we error.
    """
    from omegaconf import OmegaConf

    src_hydra_dir = Path(src_hydra_dir)
    dst_hydra_dir = Path(dst_hydra_dir)
    if dst_hydra_dir.exists():
        shutil.rmtree(dst_hydra_dir)
    shutil.copytree(src_hydra_dir, dst_hydra_dir)

    merged_path = dst_hydra_dir / "merged_config.yaml"
    if not merged_path.exists():
        raise FileNotFoundError(
            f"{merged_path} not found. The splitter expects CALVIN's "
            f"hydra to ship merged_config.yaml (the fully-resolved cfg)."
        )
    cfg = OmegaConf.load(merged_path)

    scene_yaml = scene_config_dir / f"calvin_scene_{scene_letter.upper()}.yaml"
    new_scene = OmegaConf.load(scene_yaml)

    rewritten: list[str] = []
    if "env" in cfg and "scene" in cfg.env:
        cfg.env.scene = new_scene
        rewritten.append("env.scene")
    if "scene" in cfg and OmegaConf.is_config(cfg.scene):
        cfg.scene = new_scene
        rewritten.append("scene")

    if not rewritten:
        raise RuntimeError(
            f"Could not find a `scene` subtree to override in {merged_path}. "
            f"Top-level keys: {list(cfg.keys())}. "
            f"Inspect manually and either edit merged_config.yaml directly, "
            f"or extend rewrite_merged_config() with the new layout."
        )

    OmegaConf.save(cfg, merged_path)
    logger.info(
        "Rewrote %s -> scene=%s (cfg paths: %s)",
        merged_path, scene_letter.upper(), ", ".join(rewritten),
    )


# ---------- episode symlinking ---------------------------------------------

def symlink_episodes_in_range(
    src_dir: Path,
    dst_dir: Path,
    frame_range: tuple[int, int],
) -> int:
    """Symlink `src_dir/episode_<t>.npz` → `dst_dir/...` for t in [lo, hi].

    Returns the number of symlinks actually created (skipping frames whose
    npz files are missing — robust to small CALVIN gaps near episode
    boundaries).
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = frame_range
    n = 0
    for t in range(lo, hi + 1):
        name = f"episode_{t:07d}.npz"
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)
        n += 1
    return n


# ---------- main splitter --------------------------------------------------

def split_calvin_by_scene(
    input_dir: Path,
    output_dir: Path,
    scenes: Iterable[str] = VALID_SCENES,
    scene_config_dir: Path | None = None,
) -> dict[str, Path]:
    """Build per-scene virtual splits under output_dir.

    Returns a dict mapping scene_letter -> path to the virtual split dir
    that the preprocessor can consume directly (`runners/preprocess_calvin.py
    --input_dir <returned path> --default_scene <letter>`).
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_letters = [s.upper() for s in scenes]
    for s in scene_letters:
        if s not in VALID_SCENES:
            raise ValueError(f"Invalid scene letter: {s!r}; valid={VALID_SCENES}")

    cfg_dir = resolve_scene_config_dir(scene_config_dir, scene_letters)

    # 1. Read scene ranges.
    scene_ranges = load_scene_ranges(input_dir)
    missing = [s for s in scene_letters if s not in scene_ranges]
    if missing:
        logger.warning(
            "Requested scenes not present in scene_info.npy: %s "
            "(available: %s) — skipping.",
            missing, sorted(scene_ranges),
        )
        scene_letters = [s for s in scene_letters if s in scene_ranges]

    if not scene_letters:
        raise RuntimeError(
            f"No requested scenes are present in {input_dir}/scene_info.npy."
        )

    # 2. Pre-load lang annotations once.
    src_lang = input_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not src_lang.exists():
        raise FileNotFoundError(f"{src_lang} not found.")

    src_hydra = input_dir / ".hydra"
    if not src_hydra.is_dir():
        raise FileNotFoundError(
            f"{src_hydra} not found. The splitter needs the original "
            f"hydra config to clone for each virtual split."
        )

    out: dict[str, Path] = {}
    for letter in scene_letters:
        rng = scene_ranges[letter]
        vs_dir = output_dir / f"scene_{letter}"
        vs_dir.mkdir(parents=True, exist_ok=True)

        # Filtered lang annotations.
        new_lang = filter_lang_annotations(src_lang, rng)
        n_windows = len(new_lang["info"]["indx"])
        if n_windows == 0:
            logger.warning(
                "Scene %s has frame range %s but no language windows fall "
                "inside it; skipping.", letter, rng,
            )
            continue
        (vs_dir / "lang_annotations").mkdir(exist_ok=True)
        np.save(
            vs_dir / "lang_annotations" / "auto_lang_ann.npy",
            np.array(new_lang, dtype=object),
            allow_pickle=True,
        )

        # Hydra config (with overridden scene).
        rewrite_merged_config(
            src_hydra_dir=src_hydra,
            dst_hydra_dir=vs_dir / ".hydra",
            scene_letter=letter,
            scene_config_dir=cfg_dir,
        )

        # Episode symlinks (only frames in this scene's range).
        n_links = symlink_episodes_in_range(
            src_dir=input_dir, dst_dir=vs_dir, frame_range=rng,
        )

        # Optional: ep_start_end_ids.npy — symlink as-is (preprocessor
        # never reads it but downstream tooling may). Same for any other
        # top-level metadata files we don't explicitly handle.
        for extra in ("ep_start_end_ids.npy", "statistics.yaml"):
            src_extra = input_dir / extra
            if src_extra.exists():
                dst_extra = vs_dir / extra
                if dst_extra.exists() or dst_extra.is_symlink():
                    dst_extra.unlink()
                os.symlink(src_extra.resolve(), dst_extra)

        logger.info(
            "scene %s: range=%s, windows=%d, episode_links=%d -> %s",
            letter, rng, n_windows, n_links, vs_dir,
        )
        out[letter] = vs_dir

    if not out:
        raise RuntimeError(
            "Splitter produced no virtual splits — check scene_info.npy "
            "and lang annotations."
        )
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Split a multi-scene CALVIN training/validation dir "
                    "into per-scene virtual splits.",
    )
    p.add_argument("--input_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument(
        "--scenes", default="A,B,C,D",
        help="Comma-separated scene letters to extract. Default: A,B,C,D.",
    )
    p.add_argument(
        "--scene_config_dir", default=None, type=Path,
        help="Directory containing calvin_scene_A.yaml ... calvin_scene_D.yaml. "
             "If omitted, auto-discovered from an importable calvin_env package.",
    )
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    out = split_calvin_by_scene(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scenes=scenes,
        scene_config_dir=args.scene_config_dir,
    )
    print("Created virtual splits:")
    for letter, path in out.items():
        print(f"  scene_{letter}: {path}")
    print(
        "\nNext step (one per scene):\n"
        "  python runners/preprocess_calvin.py \\\n"
        "      --input_dir <virtual_split>  --output_dir <preproc_out>/scene_X \\\n"
        "      --dataset_source <name>_sceneX  --default_scene X \\\n"
        "      --on_missing_target skip"
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    main()
