#!/usr/bin/env bash
# Thin launcher for VDA depth preprocessing over the CALVIN LeRobot dataset.
# Log capture (tee) is the caller's responsibility, per repo convention.
# Bind a GPU explicitly, e.g.:
#   CUDA_VISIBLE_DEVICES=0 bash runners/run_preprocess_depth.sh --limit 3
set -e
cd "$(dirname "$0")/.."

python runners/preprocess_depth_vda.py \
  --dataset_root datasets/task_ABC_D_scene_D_lerobot \
  "$@"
