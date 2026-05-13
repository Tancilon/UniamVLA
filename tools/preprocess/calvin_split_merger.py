"""Merge per-scene UAM-format CALVIN preprocess outputs into one dataset dir.

Companion to `calvin_scene_splitter.py`. After running
`runners/preprocess_calvin.py` once per virtual scene split (each with a
distinct `--dataset_source`), use this module to fuse the per-scene output
dirs into a single unified UAM dataset that matches the layout produced by
a single-scene preprocess run.

What it does
------------
1. Concatenates all per-scene `data.jsonl` rows; rewrites every row's
   `dataset_source` field to a single user-supplied value (so downstream
   `mixtures.py` keys can stay simple). Optionally rewrites `episode_id`
   and `id` prefixes to drop the scene tag.
2. Hard-links (or copies, if cross-fs) every per-frame artifact —
   `images/obs/{static,wrist}`, `images/target`, `images/future`,
   `depth/{static,wrist}`, `point_clouds` — into the merged dir. Sample IDs
   are globally unique across scenes by construction (each preprocess run
   used its own `--dataset_source`), so no rename is needed.
3. Rewrites `statistics.yaml` over the merged sample list using the same
   helper the in-process orchestrator uses (via private import — kept in
   sync with `calvin_preprocessor.py`).

CLI
---
    python -m tools.preprocess.calvin_split_merger \\
        --inputs  preproc/scene_A preproc/scene_B preproc/scene_C preproc/scene_D \\
        --output  datasets/uamvla_calvin/task_ABCD_D_merged \\
        --dataset_source task_ABCD_D
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Subdirs that the per-frame writer in calvin_preprocessor populates.
PER_FRAME_SUBDIRS = (
    "images/obs/static",
    "images/obs/wrist",
    "images/target",
    "images/future",
    "depth/static",
    "depth/wrist",
    "point_clouds",
)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link src → dst if same filesystem; fall back to copy.

    Symlinks would make the merged dir non-relocatable; hard links keep it
    self-contained at zero space cost when files share an fs.

    If `dst` already exists we unlink and replace — the caller has
    typically already wiped the destination subtree, but staying robust
    here means a single repeat call won't leave stale content.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        # Cross-filesystem or fs without hardlink support.
        import shutil
        shutil.copy2(src, dst)


def _warn_unknown_artifact_dirs(input_dir: Path) -> None:
    """Log a warning for any per-frame data dir not in PER_FRAME_SUBDIRS.

    Catches the case where the upstream preprocessor adds a new artifact
    (e.g., `images/wrist_target`) but the merger's hardcoded list isn't
    updated — without this guard the new artifact would silently never
    reach the merged dataset.
    """
    expected_top = {"images", "depth", "point_clouds", "shards"}
    expected_full = set(PER_FRAME_SUBDIRS)
    expected_sentinels = {"data.jsonl", "statistics.yaml"}
    for p in input_dir.iterdir():
        if not p.is_dir():
            if p.name not in expected_sentinels:
                logger.debug("Ignoring unrecognised file %s", p)
            continue
        if p.name not in expected_top:
            logger.warning(
                "Unrecognised top-level dir %s under %s — not merged. "
                "If a new per-frame artifact was added to the preprocessor, "
                "extend PER_FRAME_SUBDIRS.", p.name, input_dir,
            )
            continue
        # Recurse one level for `images/` and `depth/` containers.
        if p.name in {"images", "depth"}:
            for child in p.iterdir():
                if not child.is_dir():
                    continue
                rel = f"{p.name}/{child.name}"
                if rel not in expected_full and not any(
                    full.startswith(rel + "/") for full in expected_full
                ):
                    logger.warning(
                        "Unrecognised subdir %s under %s — not merged. "
                        "Extend PER_FRAME_SUBDIRS to include it.",
                        rel, input_dir,
                    )


def _harvest_dir(src_dir: Path, dst_dir: Path) -> int:
    """Move all files from src_dir/* into dst_dir/* via hard links.

    Returns the number of files linked. Skips if src_dir doesn't exist
    (some scenes may legitimately produce no per-frame artifacts of a
    given kind, e.g. no point_clouds when on_missing_target=skip dropped
    them).
    """
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        _link_or_copy(p, dst_dir / p.name)
        n += 1
    return n


def merge_scene_outputs(
    inputs: list[Path],
    output_dir: Path,
    dataset_source: str,
    rewrite_episode_id: bool = True,
) -> Path:
    """Fuse per-scene UAM dataset dirs at `inputs` into `output_dir`.

    Each input must be a directory produced by a successful run of
    `runners/preprocess_calvin.py` (i.e., contain `data.jsonl`,
    `statistics.yaml`, and the per-frame subdirs above).

    `dataset_source` overwrites every row's `dataset_source` field.

    `rewrite_episode_id`: if True, replace the leading per-scene
    dataset_source prefix in `episode_id` and `id` with the new
    `dataset_source`. e.g. `task_ABCD_D_sceneA_ep00012_step0003` ->
    `task_ABCD_D_ep00012_step0003`. Sample-IDs across scenes were globally
    unique via the per-scene prefix, so after rewriting we must NOT
    introduce collisions. We append the scene letter as a safe disambig
    suffix to `episode_id` to keep uniqueness.
    """
    if not inputs:
        raise ValueError("merge_scene_outputs requires at least one input dir.")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Wipe per-frame subdirs so a rerun doesn't leak stale artifacts
    # from a previous merge whose inputs differed. data.jsonl and
    # statistics.yaml are overwritten unconditionally below.
    import shutil as _shutil
    for sub in PER_FRAME_SUBDIRS:
        sub_path = output_dir / sub
        if sub_path.exists():
            _shutil.rmtree(sub_path)
        sub_path.mkdir(parents=True)

    all_rows: list[dict] = []
    seen_ids: set[str] = set()
    camera_intrinsics: dict | None = None

    for inp in inputs:
        inp = Path(inp).resolve()
        jsonl = inp / "data.jsonl"
        if not jsonl.exists():
            raise FileNotFoundError(f"{jsonl} missing — was preprocess for this scene successful?")

        # Pull camera intrinsics from any one input (they're identical
        # across scenes — same camera, same RENDER_W/H).
        if camera_intrinsics is None:
            stats_path = inp / "statistics.yaml"
            if stats_path.exists():
                import yaml
                with open(stats_path) as f:
                    stats = yaml.safe_load(f)
                cams = stats.get("cameras") or {}
                if "static" in cams and "wrist" in cams:
                    camera_intrinsics = {
                        "static": cams["static"]["intrinsic"],
                        "wrist":  cams["wrist"]["intrinsic"],
                    }

        # Hard-link per-frame files into the merged dir.
        for sub in PER_FRAME_SUBDIRS:
            n = _harvest_dir(inp / sub, output_dir / sub)
            logger.debug("[%s] linked %d files from %s", inp.name, n, sub)
        # Warn if the input dir contains per-frame subdirs we don't
        # recognise — most likely a new artifact kind was added to the
        # preprocessor and PER_FRAME_SUBDIRS forgot to track it.
        _warn_unknown_artifact_dirs(inp)

        # Read & rewrite rows.
        scene_letter: str | None = None
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                # Detect per-scene source prefix on first row.
                if scene_letter is None:
                    src = str(row.get("dataset_source", ""))
                    if src.endswith(("_sceneA", "_sceneB", "_sceneC", "_sceneD")):
                        scene_letter = src[-1]
                    else:
                        scene_letter = inp.name[-1] if inp.name.startswith("scene_") else "X"

                row["dataset_source"] = dataset_source
                if rewrite_episode_id:
                    row["episode_id"] = _rewrite_id(
                        row["episode_id"], dataset_source, scene_letter,
                    )
                    row["id"] = _rewrite_id(
                        row["id"], dataset_source, scene_letter,
                    )

                if row["id"] in seen_ids:
                    raise RuntimeError(
                        f"Duplicate sample id after rewrite: {row['id']!r}. "
                        f"Did two scenes share a dataset_source?"
                    )
                seen_ids.add(row["id"])
                all_rows.append(row)

    if camera_intrinsics is None:
        raise RuntimeError(
            "Could not extract camera intrinsics from any input's "
            "statistics.yaml — re-run preprocess to regenerate stats."
        )

    all_rows.sort(key=lambda r: r["id"])
    out_jsonl = output_dir / "data.jsonl"
    with open(out_jsonl, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    logger.info("Merged %d samples -> %s", len(all_rows), out_jsonl)

    _write_merged_statistics(all_rows, output_dir, camera_intrinsics)
    return output_dir


def _rewrite_id(old: str, dataset_source: str, scene_letter: str) -> str:
    """Rebuild a sample/episode id under the unified dataset_source.

    Per-scene preprocessing produces ids like
        f"{src}_sceneX_ep{idx:05d}_step{k:04d}"  (sample id)
        f"{src}_sceneX_ep{idx:05d}"              (episode id)
    where {src} can also be the full dataset_source if user passed e.g.
    `--dataset_source task_ABCD_D_sceneA`.

    We rewrite by replacing the prefix-up-to-`_ep` (or `_step` for
    safety). We keep `_sceneX` baked into the rewritten episode id
    (suffixed onto the unified source) so episode ids remain globally
    unique without parsing structure further.
    """
    # Find the marker that splits the per-scene prefix from the rest.
    for marker in ("_ep", "_step"):
        pos = old.find(marker)
        if pos >= 0:
            tail = old[pos:]   # "_ep00012" or "_ep00012_step0003"
            return f"{dataset_source}_scene{scene_letter}{tail}"
    # Fallback: leave as-is — will surface as a duplicate-id error if it
    # collides, which is the desired loud failure.
    logger.warning("Could not parse id %r — leaving unchanged.", old)
    return old


def _write_merged_statistics(
    rows: list[dict],
    output_dir: Path,
    camera_intrinsics: dict,
) -> None:
    """Re-emit statistics.yaml from the merged row list.

    DEPRECATED post 2026-05-13 (spec §2.2): the JSONL-era statistics.yaml
    pipeline was removed when uamvla migrated to UamVLAOFT (LeRobot computes
    stats automatically into `meta/stats_gr00t.json`). This function previously
    delegated to `tools.preprocess.calvin_preprocessor._write_statistics`,
    which was deleted in the cleanup commit. Raises NotImplementedError until
    rewritten against LeRobot's stats pipeline.
    """
    raise NotImplementedError(
        "_write_merged_statistics depended on the deleted "
        "tools.preprocess.calvin_preprocessor._write_statistics helper. "
        "The statistics.yaml schema was removed in 2026-05-13 cleanup "
        "(spec §2.2); LeRobot now auto-computes stats into "
        "meta/stats_gr00t.json on first dataset load. "
        "Rewrite this against LeRobot's pipeline if multi-scene preprocessing "
        "is needed."
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge per-scene UAM CALVIN preprocess outputs.",
    )
    p.add_argument(
        "--inputs", nargs="+", required=True, type=Path,
        help="Per-scene preprocess output dirs (each must contain data.jsonl).",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--dataset_source", required=True,
        help="The unified dataset_source field to write into every merged row.",
    )
    p.add_argument(
        "--keep_per_scene_id_prefix", action="store_true",
        help="Don't rewrite per-scene id prefixes. Use only if you intentionally "
             "want sample ids like `task_ABCD_D_sceneA_ep0042_step0003`.",
    )
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    out = merge_scene_outputs(
        inputs=args.inputs,
        output_dir=args.output,
        dataset_source=args.dataset_source,
        rewrite_episode_id=not args.keep_per_scene_id_prefix,
    )
    print(f"Merged dataset at: {out}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    main()
