#!/bin/bash
# examples/LIBERO/train_files/run_uamvla_gr00t_libero_train.sh
#
# Reusable launcher for UamVLA-GR00T LIBERO training.
# Run from the repository root:
#   bash examples/LIBERO/train_files/run_uamvla_gr00t_libero_train.sh
#
# Local 2-GPU smoke example:
#   CUDA_VISIBLE_DEVICES=6,7 NUM_GPUS=2 DATA_MIX=uamvla_libero_goal_h8 \
#     WANDB_MODE=offline bash examples/LIBERO/train_files/run_uamvla_gr00t_libero_train.sh \
#       --trainer.max_train_steps 2 \
#       --trainer.num_warmup_steps 0 \
#       --trainer.save_interval 999999 \
#       --trainer.eval_interval 999999 \
#       --trainer.visualization.enabled false
#
# Full 8-GPU example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
#     bash examples/LIBERO/train_files/run_uamvla_gr00t_libero_train.sh

set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# NCCL defaults for single-node smoke/full runs. Override these from the
# environment on clusters that require a specific interface or IB HCA.
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"

if [[ -n "${GPU_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

CONFIG_YAML="${CONFIG_YAML:-./starVLA/config/training/uamvla_gr00t_libero.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-starVLA/config/deepseeds/uamvla_gr00t_zero3.yaml}"

BASE_VLM="${BASE_VLM:-ckpt/Qwen3-VL-8B-Instruct}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-datasets/libero2uam}"
DATA_MIX="${DATA_MIX:-uamvla_libero_all_h8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./playground/Checkpoints}"
RUN_ID="${RUN_ID:-uamvla_gr00t_libero_all_8b_h8}"

output_dir="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

echo "[uamvla-gr00t-libero] config=${CONFIG_YAML}"
echo "[uamvla-gr00t-libero] base_vlm=${BASE_VLM}"
echo "[uamvla-gr00t-libero] data_root=${DATA_ROOT_DIR}"
echo "[uamvla-gr00t-libero] data_mix=${DATA_MIX}"
echo "[uamvla-gr00t-libero] num_gpus=${NUM_GPUS} grad_accum=${GRAD_ACCUM}"
echo "[uamvla-gr00t-libero] deepspeed_config=${DEEPSPEED_CONFIG}"
echo "[uamvla-gr00t-libero] run_id=${RUN_ID}"

accelerate launch \
  --config_file "${DEEPSPEED_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT_DIR}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  "$@"
