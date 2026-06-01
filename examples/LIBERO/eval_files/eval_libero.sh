#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${STARVLA_DIR}"

DEFAULT_CKPT="${STARVLA_DIR}/playground/Checkpoints/uamvla_gr00t_libero_8b_h8_no_pose_bs128/checkpoints/steps_5000_pytorch_model.pt"

CKPT="${CKPT:-${DEFAULT_CKPT}}"
LIBERO_HOME="${LIBERO_HOME:-${STARVLA_DIR}/../LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-conda run -n libero_env python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-2}"
LIBERO_IMAGE_TRANSFORM="${LIBERO_IMAGE_TRANSFORM:-rotate180}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

if [[ ! -d "${LIBERO_HOME}" ]]; then
  echo "[ERROR] LIBERO_HOME does not exist: ${LIBERO_HOME}" >&2
  echo "Set LIBERO_HOME=/path/to/LIBERO and rerun." >&2
  exit 1
fi

export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

folder_name="$(basename "$(dirname "$(dirname "${CKPT}")")")_$(basename "$(dirname "${CKPT}")")_$(basename "${CKPT}")"
model_root="${CKPT%%/checkpoints/*}"
video_out_path="${model_root}/results/${TASK_SUITE}/${folder_name}"

mkdir -p "${video_out_path}"

echo "=========================================="
echo " LIBERO evaluator"
echo "=========================================="
echo " STARVLA_DIR          : ${STARVLA_DIR}"
echo " CKPT                 : ${CKPT}"
echo " LIBERO_HOME          : ${LIBERO_HOME}"
echo " LIBERO_PYTHON        : ${LIBERO_PYTHON}"
echo " HOST                 : ${HOST}"
echo " PORT                 : ${PORT}"
echo " TASK_SUITE           : ${TASK_SUITE}"
echo " NUM_TRIALS_PER_TASK  : ${NUM_TRIALS_PER_TASK}"
echo " LIBERO_IMAGE_TRANSFORM: ${LIBERO_IMAGE_TRANSFORM}"
echo " VIDEO_OUT_PATH       : ${video_out_path}"
echo "=========================================="

${LIBERO_PYTHON} ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.image-transform "${LIBERO_IMAGE_TRANSFORM}" \
  --args.video-out-path "${video_out_path}"
