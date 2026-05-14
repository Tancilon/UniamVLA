#!/bin/bash
# examples/calvin/train_files/run_uamvla_oft_train.sh
#
# UamVLAOFT — CALVIN ABCD_D training launcher.
# Aux heads (pose / future / recon) all ENABLED per yaml (PR 5/6/7).
#
# Per CLAUDE.md training-launcher rules:
#   - script does NOT redirect logs (the runner command adds `tee logs/...`)
#   - script supports CLI override via "$@" pass-through, e.g.
#       bash run_uamvla_oft_train.sh --trainer.is_resume true
#   - GPU count via NUM_GPUS env var (default 8 for full-node runs)

set -euo pipefail

# NCCL networking: auto-detect for single-node multi-GPU runs.
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

config_yaml=./starVLA/config/training/uamvla_oft_calvin_abcd.yaml
NUM_GPUS=${NUM_GPUS:-8}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  "$@"
