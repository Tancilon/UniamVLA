#!/usr/bin/env bash
# Thin launcher for RGB frozen-VAE latent preprocessing (FutureLatentHead targets).
# Log capture (tee) is the caller's responsibility, per repo convention.
# Bind a GPU explicitly, e.g.:
#   CUDA_VISIBLE_DEVICES=0 bash runners/run_preprocess_rgb_latent.sh --limit 3 --report_psnr
set -e
cd "$(dirname "$0")/.."

python runners/preprocess_rgb_latent.py \
  --dataset_root datasets/task_ABC_D_scene_D_lerobot \
  "$@"
