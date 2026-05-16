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
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"

config_yaml=${CONFIG_YAML:-./starVLA/config/training/uamvla_oft_calvin_abcd.yaml}
# Caller can override:
#   NUM_GPUS=4   bash run_uamvla_oft_train.sh   (4-GPU run, default 8)
#   GRAD_ACCUM=4 bash run_uamvla_oft_train.sh   (effective_batch = per_device_batch × NUM_GPUS × GRAD_ACCUM)
#   CONFIG_YAML=./starVLA/config/training/uamvla_oft_calvin_d.yaml bash run_uamvla_oft_train.sh
NUM_GPUS=${NUM_GPUS:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
# DEEPSPEED_CONFIG: accelerate config file path; switch to deepspeed_zero3.yaml
# when ZeRO-2 OOMs on the full-finetune (e.g. Qwen3-VL-8B trainable param set).
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}

accelerate launch \
  --config_file ${DEEPSPEED_CONFIG} \
  --num_processes ${NUM_GPUS} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  "$@"
