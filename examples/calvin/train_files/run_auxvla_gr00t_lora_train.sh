#!/bin/bash
# examples/calvin/train_files/run_auxvla_gr00t_lora_train.sh
#
# Reusable launcher for AuxVLAGR00T LoRA CALVIN training.
# Run from the repository root:
#   bash examples/calvin/train_files/run_auxvla_gr00t_lora_train.sh
#
# Useful overrides:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
#     bash examples/calvin/train_files/run_auxvla_gr00t_lora_train.sh
#   CUDA_VISIBLE_DEVICES=2,3,4,5 NUM_GPUS=4 MASTER_PORT=29662 \
#     bash examples/calvin/train_files/run_auxvla_gr00t_lora_train.sh
#   MAX_TRAIN_STEPS=1000 SAVE_INTERVAL=500 \
#     bash examples/calvin/train_files/run_auxvla_gr00t_lora_train.sh
#
# Additional OmegaConf dotlist overrides are passed through:
#   bash examples/calvin/train_files/run_auxvla_gr00t_lora_train.sh \
#     --trainer.max_train_steps=100 \
#     --trainer.visualization.enabled=false

set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export UAMVLA_TRAIN_STEP_PROFILE="${UAMVLA_TRAIN_STEP_PROFILE:-0}"

# NCCL networking defaults for single-node multi-GPU runs.
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"

if [[ -n "${GPU_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

CONFIG_YAML="${CONFIG_YAML:-starVLA/config/training/auxvla_gr00t_lora.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29661}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-playground/Checkpoints}"
RUN_ID="${RUN_ID:-auxvla_gr00t_lora_calvin_abc_8gpu_full}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-datasets/calvin2uam}"
DATA_MIX="${DATA_MIX:-uamvla_calvin_abc_h8}"
ACTION_TYPE="${ACTION_TYPE:-delta_qpos}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-20}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"

IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_LIST[@]}" -ne "${NUM_GPUS}" ]]; then
  echo "CUDA_VISIBLE_DEVICES exposes ${#GPU_LIST[@]} GPUs, but NUM_GPUS=${NUM_GPUS}."
  echo "Set them consistently, e.g. CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4."
  exit 1
fi

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

accelerate launch \
  --config_file "${DEEPSPEED_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MASTER_PORT}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --run_id "${RUN_ID}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT_DIR}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.action_type "${ACTION_TYPE}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.eval_interval 999999 \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --trainer.visualization.enabled false \
  "$@"
