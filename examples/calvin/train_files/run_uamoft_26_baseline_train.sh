#!/usr/bin/env bash
set -euo pipefail


export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# NCCL defaults for single-node multi-GPU.
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"

# GPU selection.
if [[ -n "${GPU_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

# Training config.
CONFIG_YAML="${CONFIG_YAML:-./starVLA/config/training/uamvla_gr00t_calvin.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"





RUN_ROOT_DIR="${RUN_ROOT_DIR:-playground/Checkpoints}"
RUN_ID="${RUN_ID:-uamvla_gr00t_calvin_d_h8}"


mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}"
cp "$0" "${RUN_ROOT_DIR}/${RUN_ID}/"

accelerate launch \
  --config_file "${DEEPSPEED_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  "$@"
