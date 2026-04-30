"""Minimal entry point to regenerate statistics.yaml without re-running MuJoCo.

Usage (recompute stats from existing data.jsonl):
    python tools/preprocess/run_libero_preprocess.py --stats-only \
        --output-dir datasets/uamvla_test/libero_spatial

Usage (full preprocessing from source HDF5):
    python tools/preprocess/run_libero_preprocess.py \
        --input-dir dataset/libero2uam/raw/libero_spatial_test \
        --output-dir datasets/uamvla_test/libero_spatial

Environment variables:
    LIBERO_INPUT_DIR   source HDF5 directory (overridden by --input-dir)
    LIBERO_OUTPUT_DIR  output directory      (overridden by --output-dir)
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Make project root importable when invoked as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.preprocess.libero_preprocessor import (  # noqa: E402
    LiberoPreprocessor,
    STATIC_CAM,
    WRIST_CAM,
)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess LIBERO data to UamVLA format")
    p.add_argument(
        "--input-dir",
        default=os.environ.get("LIBERO_INPUT_DIR", ""),
        help="Directory containing source .hdf5 files",
    )
    p.add_argument(
        "--output-dir",
        default=os.environ.get(
            "LIBERO_OUTPUT_DIR",
            "datasets/uamvla_test/libero_spatial",
        ),
        help="Output directory for UamVLA dataset",
    )
    p.add_argument(
        "--suite",
        default="libero_spatial",
        help="Suite name embedded in sample IDs (default: libero_spatial)",
    )
    p.add_argument(
        "--stats-only",
        action="store_true",
        help=(
            "Skip MuJoCo replay; recompute statistics.yaml only from the "
            "existing data.jsonl in --output-dir.  Requires the output-dir "
            "to already contain data.jsonl and a statistics.yaml with camera "
            "intrinsics (used as the canonical camera parameters)."
        ),
    )
    return p.parse_args()


def recompute_stats_only(output_dir: Path, suite: str):
    """Recompute statistics.yaml from existing data.jsonl without MuJoCo."""
    jsonl_path = output_dir / "data.jsonl"
    stats_path = output_dir / "statistics.yaml"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"data.jsonl not found in {output_dir}")
    if not stats_path.exists():
        raise FileNotFoundError(
            f"statistics.yaml not found in {output_dir} — "
            f"need it to read camera intrinsics"
        )

    # Load samples
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logging.info(f"Loaded {len(samples)} samples from {jsonl_path}")

    # Reconstruct camera_intrinsics from existing yaml
    with open(stats_path) as f:
        old_stats = yaml.safe_load(f)
    camera_intrinsics = {
        STATIC_CAM: old_stats["cameras"]["static"]["intrinsic"],
        WRIST_CAM: old_stats["cameras"]["wrist"]["intrinsic"],
    }
    logging.info("Reconstructed camera intrinsics from existing statistics.yaml")

    preprocessor = LiberoPreprocessor(suite=suite)
    preprocessor._write_statistics(samples, output_dir, camera_intrinsics)
    logging.info("statistics.yaml rewritten with new schema")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.stats_only:
        recompute_stats_only(output_dir, args.suite)
        return

    if not args.input_dir:
        print(
            "ERROR: --input-dir (or LIBERO_INPUT_DIR) is required for full preprocessing.",
            file=sys.stderr,
        )
        sys.exit(1)

    preprocessor = LiberoPreprocessor(suite=args.suite)
    preprocessor.process(args.input_dir, str(output_dir))


if __name__ == "__main__":
    main()
