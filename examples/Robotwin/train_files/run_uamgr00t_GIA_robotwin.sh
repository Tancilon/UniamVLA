#!/usr/bin/env bash
# UamGR00T_GIA — Geometry·Interaction·Action three-head training on RoboTwin.
#
# Sidecar prerequisites:
#   python runners/preprocess_robotwin2_depth_affordance.py --dataset_root datasets/robotwin2
#   python runners/preprocess_rgb_latent.py                 --dataset_root datasets/robotwin2
#
# No logging to disk here; the caller pipes through tee. CLI overrides pass
# through "$@", e.g. --trainer.max_train_steps 20 or --trainer.is_resume true.
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONWARNINGS="ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"

if [[ -n "${GPU_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

CONFIG_YAML="${CONFIG_YAML:-./examples/Robotwin/train_files/run_uamgr00t_GIA_robotwin.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-./wandb}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/Checkpoints}"
RUN_ID="${RUN_ID:-uamvla_gr00t_gia_robotwin}"

mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}"
cp "$0" "${RUN_ROOT_DIR}/${RUN_ID}/"

accelerate launch \
  --config_file "${DEEPSPEED_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  "$@"
