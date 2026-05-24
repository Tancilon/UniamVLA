#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${STARVLA_DIR}"

DEFAULT_CKPT="${STARVLA_DIR}/playground/Checkpoints/uamvla_gr00t_libero_8b_h8_no_pose_bs128/checkpoints/steps_5000_pytorch_model.pt"

CKPT="${CKPT:-${DEFAULT_CKPT}}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6694}"
STARVLA_PYTHON="${STARVLA_PYTHON:-conda run -n uamvla python}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

echo "=========================================="
echo " UamVLA LIBERO policy server"
echo "=========================================="
echo " STARVLA_DIR    : ${STARVLA_DIR}"
echo " CKPT           : ${CKPT}"
echo " GPU_ID         : ${GPU_ID}"
echo " PORT           : ${PORT}"
echo " STARVLA_PYTHON : ${STARVLA_PYTHON}"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPU_ID}" ${STARVLA_PYTHON} deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" \
  --port "${PORT}" \
  --use_bf16
