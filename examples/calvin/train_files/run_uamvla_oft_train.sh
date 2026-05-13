#!/bin/bash
# examples/calvin/train_files/run_uamvla_oft_train.sh
#
# UamVLAOFT (baseline B) — CALVIN ABCD_D, all aux heads OFF.
#
# Per CLAUDE.md training-launcher rules:
#   - script does NOT redirect logs (the runner command adds `tee logs/...`)
#   - script supports CLI override via "$@" pass-through, so callers can
#     do e.g. `bash run_uamvla_oft_train.sh --trainer.is_resume true`
#     without editing this file.

set -euo pipefail

# NCCL networking: auto-detect for single-node 8-GPU runs.
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

config_yaml=./starVLA/config/training/uamvla_oft_calvin_abcd.yaml

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  "$@"
