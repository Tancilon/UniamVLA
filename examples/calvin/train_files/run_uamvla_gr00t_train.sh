#!/bin/bash
# examples/calvin/train_files/run_uamvla_gr00t_train.sh
#
# Reusable launcher for UamVLAGR00T CALVIN training.
# Run from the repository root:
#   bash examples/calvin/train_files/run_uamvla_gr00t_train.sh
#
# Useful overrides:
#   GPU_ID=4 bash examples/calvin/train_files/run_uamvla_gr00t_train.sh
#   CONFIG_YAML=./starVLA/config/training/uamvla_gr00t_calvin_abcd.yaml \
#     bash examples/calvin/train_files/run_uamvla_gr00t_train.sh
#   NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/calvin/train_files/run_uamvla_gr00t_train.sh
#
# Smoke overrides are passed through as OmegaConf dotlist args:
#   bash examples/calvin/train_files/run_uamvla_gr00t_train.sh \
#     --trainer.max_train_steps=10 \
#     --trainer.save_interval=999999 \
#     --trainer.eval_interval=999999 \
#     --trainer.visualization.enabled=false

set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# NCCL networking defaults for single-node smoke runs.
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"

if [[ -n "${GPU_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

CONFIG_YAML="${CONFIG_YAML:-./starVLA/config/training/uamvla_gr00t_calvin_d.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"

accelerate launch \
  --config_file "${DEEPSPEED_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  "$@"
