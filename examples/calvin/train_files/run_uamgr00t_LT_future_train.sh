#!/bin/bash
# UamGR00T_LT — Seer-style future-latent (foresight) supervision on CALVIN ABC_D scene D.
#
# Requires the precomputed sidecar:
#   python runners/preprocess_rgb_latent.py --dataset_root datasets/task_ABC_D_scene_D_lerobot
#
# No logging to disk here; the caller pipes through tee. CLI overrides pass
# through "$@", e.g. --trainer.max_train_steps 20 or --trainer.is_resume true.
set -euo pipefail

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-8}" \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/run_uamgr00t_LT_future_train.yaml \
  "$@"
