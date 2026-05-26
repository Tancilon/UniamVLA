#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${STARVLA_DIR}"

SCRIPT_PATH="${SCRIPT_DIR}/eval_libero_parall.sh"

###############################################################################
# ============ USER CONFIG: override with environment variables ============
###############################################################################

# If CKPT_LIST_TEXT is empty, all .pt files in CKPT_DIR will be evaluated.
# CKPT_LIST_TEXT accepts whitespace-separated checkpoint paths.
DEFAULT_CKPT="${STARVLA_DIR}/playground/Checkpoints/uamvla_gr00t_libero_8b_h8_no_pose_bs128/checkpoints/steps_5000_pytorch_model.pt"
CKPT_DIR="${CKPT_DIR:-}"
CKPT_LIST_TEXT="${CKPT_LIST_TEXT:-${DEFAULT_CKPT}}"

# Whitespace-separated task suite names.
TASK_SUITES_TEXT="${TASK_SUITES_TEXT:-libero_spatial libero_object libero_goal libero_10}"

# Whitespace-separated GPU ids. Keep this to one manually selected idle GPU by default.
GPU_IDS="${GPU_IDS:-0}"

# Each job uses BASE_PORT + job_index.
BASE_PORT="${BASE_PORT:-6694}"

# Small local default. Override with NUM_TRIALS_PER_TASK=50 for full evaluation.
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-2}"

# Current launcher is serial. Sleep still gives the previous server time to release resources.
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"

###############################################################################
# ============ END USER CONFIG ============
###############################################################################

read -r -a CKPT_LIST <<< "${CKPT_LIST_TEXT}"
read -r -a TASK_SUITES <<< "${TASK_SUITES_TEXT}"
read -r -a GPU_LIST <<< "${GPU_IDS}"

if [[ ${#CKPT_LIST[@]} -eq 0 || -z "${CKPT_LIST[0]:-}" ]]; then
  if [[ -z "${CKPT_DIR}" ]]; then
    echo "[ERROR] CKPT_LIST_TEXT is empty and CKPT_DIR is not set." >&2
    exit 1
  fi

  mapfile -t CKPT_LIST < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name "*.pt" | sort)
  if [[ ${#CKPT_LIST[@]} -eq 0 ]]; then
    echo "[ERROR] No .pt files found in ${CKPT_DIR}" >&2
    exit 1
  fi
fi

if [[ ${#TASK_SUITES[@]} -eq 0 || -z "${TASK_SUITES[0]:-}" ]]; then
  echo "[ERROR] TASK_SUITES_TEXT is empty." >&2
  exit 1
fi

if [[ ${#GPU_LIST[@]} -eq 0 || -z "${GPU_LIST[0]:-}" ]]; then
  echo "[ERROR] GPU_IDS is empty." >&2
  exit 1
fi

num_gpus=${#GPU_LIST[@]}
job_index=0
gpu_job_count=()

for ((i = 0; i < num_gpus; i++)); do
  gpu_job_count[$i]=0
done

echo "=========================================="
echo " Auto Eval LIBERO"
echo "=========================================="
echo " STARVLA_DIR          : ${STARVLA_DIR}"
echo " Checkpoints          : ${CKPT_LIST[*]}"
echo " Task suites          : ${TASK_SUITES[*]}"
echo " GPU ids              : ${GPU_LIST[*]}"
echo " BASE_PORT            : ${BASE_PORT}"
echo " NUM_TRIALS_PER_TASK  : ${NUM_TRIALS_PER_TASK}"
echo " Launcher             : serial"
echo "=========================================="

for ckpt in "${CKPT_LIST[@]}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERROR] Checkpoint not found: ${ckpt}" >&2
    exit 1
  fi

  for task in "${TASK_SUITES[@]}"; do
    gpu_idx=$((job_index % num_gpus))
    gpu_id=${GPU_LIST[$gpu_idx]}
    port=$((BASE_PORT + job_index))
    ckpt_name="$(basename "${ckpt}" .pt)"

    echo "[Job ${job_index}] GPU=${gpu_id}  port=${port}  ckpt=${ckpt_name}  task=${task}"

    bash "${SCRIPT_PATH}" "${ckpt}" "${task}" "${gpu_id}" "${port}" "${NUM_TRIALS_PER_TASK}"

    gpu_job_count[$gpu_idx]=$((gpu_job_count[$gpu_idx] + 1))
    job_index=$((job_index + 1))

    sleep "${SLEEP_BETWEEN}"
  done
done

echo "=========================================="
echo " All evaluations completed!"
echo " GPU job distribution:"
for ((i = 0; i < num_gpus; i++)); do
  echo "   GPU ${GPU_LIST[$i]}: ${gpu_job_count[$i]} jobs"
done
echo "=========================================="
