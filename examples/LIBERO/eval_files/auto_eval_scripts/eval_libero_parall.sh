#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${STARVLA_DIR}"

DEFAULT_LIBERO_HOME="${STARVLA_DIR}/third_party/LIBERO"

export LIBERO_HOME="${LIBERO_HOME:-${DEFAULT_LIBERO_HOME}}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/libero_numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/libero_mpl_config}"

LIBERO_PYTHON="${LIBERO_PYTHON:-conda run -n libero_env python}"
STARVLA_PYTHON="${STARVLA_PYTHON:-conda run -n uamvla python}"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <ckpt_path> <task_suite_name> <gpu_id> <port> [num_trials_per_task]" >&2
  exit 1
fi

your_ckpt="$1"
task_suite_name="$2"
gpu_id="$3"
base_port="$4"
num_trials_per_task="${5:-${NUM_TRIALS_PER_TASK:-2}}"
host="${HOST:-127.0.0.1}"

if [[ ! -f "${your_ckpt}" ]]; then
  echo "[ERROR] Checkpoint not found: ${your_ckpt}" >&2
  exit 1
fi

if [[ ! -d "${LIBERO_HOME}" ]]; then
  echo "[ERROR] LIBERO_HOME does not exist: ${LIBERO_HOME}" >&2
  echo "Set LIBERO_HOME=/path/to/LIBERO and rerun." >&2
  exit 1
fi

LIBERO_NCCL_SO="${LIBERO_NCCL_SO:-$(${LIBERO_PYTHON} -c 'import site; from pathlib import Path; matches=[]; [matches.append(Path(base) / "nvidia" / "nccl" / "lib" / "libnccl.so.2") for base in site.getsitepackages() if (Path(base) / "nvidia" / "nccl" / "lib" / "libnccl.so.2").exists()]; print(matches[0] if matches else "")')}"
LIBERO_LD_PRELOAD="${LD_PRELOAD:-}"
if [[ -n "${LIBERO_NCCL_SO}" && -f "${LIBERO_NCCL_SO}" ]]; then
  LIBERO_LD_PRELOAD="${LIBERO_NCCL_SO}${LIBERO_LD_PRELOAD:+:${LIBERO_LD_PRELOAD}}"
fi

LIBERO_BENCHMARK_ROOT="${LIBERO_HOME}/libero/libero"
LIBERO_CONFIG_FILE="${LIBERO_CONFIG_PATH}/config.yaml"
if [[ ! -f "${LIBERO_CONFIG_FILE}" ]]; then
  mkdir -p "${LIBERO_CONFIG_PATH}"
  {
    echo "assets: ${LIBERO_BENCHMARK_ROOT}/assets"
    echo "bddl_files: ${LIBERO_BENCHMARK_ROOT}/bddl_files"
    echo "benchmark_root: ${LIBERO_BENCHMARK_ROOT}"
    echo "datasets: ${LIBERO_BENCHMARK_ROOT}/../datasets"
    echo "init_states: ${LIBERO_BENCHMARK_ROOT}/init_files"
  } > "${LIBERO_CONFIG_FILE}"
  echo "Initialized LIBERO config: ${LIBERO_CONFIG_FILE}"
fi

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    echo "Killing policy server process with PID: ${server_pid}"
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

model_root="${your_ckpt%%/checkpoints/*}"
folder_name="$(basename "$(dirname "$(dirname "${your_ckpt}")")")_$(basename "$(dirname "${your_ckpt}")")_$(basename "${your_ckpt}")"

video_out_path="${model_root}/videos/${task_suite_name}/${folder_name}"
log_path="${model_root}/logs/${task_suite_name}"
mkdir -p "${video_out_path}" "${log_path}"

echo "=========================================="
echo " LIBERO batch eval job"
echo "=========================================="
echo " STARVLA_DIR          : ${STARVLA_DIR}"
echo " CKPT                 : ${your_ckpt}"
echo " TASK_SUITE           : ${task_suite_name}"
echo " NUM_TRIALS_PER_TASK  : ${num_trials_per_task}"
echo " GPU_ID               : ${gpu_id}"
echo " HOST                 : ${host}"
echo " PORT                 : ${base_port}"
echo " LIBERO_HOME          : ${LIBERO_HOME}"
echo " LIBERO_PYTHON        : ${LIBERO_PYTHON}"
echo " LIBERO_NCCL_SO       : ${LIBERO_NCCL_SO:-<not found>}"
echo " STARVLA_PYTHON       : ${STARVLA_PYTHON}"
echo " VIDEO_OUT_PATH       : ${video_out_path}"
echo " LOG_PATH             : ${log_path}/${folder_name}.log"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${gpu_id}" ${STARVLA_PYTHON} deployment/model_server/server_policy.py \
  --ckpt_path "${your_ckpt}" \
  --port "${base_port}" \
  --use_bf16 &

server_pid=$!

DEBUG="${LIBERO_DEBUG:-}" LD_PRELOAD="${LIBERO_LD_PRELOAD}" ${LIBERO_PYTHON} ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${your_ckpt}" \
  --args.host "${host}" \
  --args.port "${base_port}" \
  --args.task-suite-name "${task_suite_name}" \
  --args.num-trials-per-task "${num_trials_per_task}" \
  --args.video-out-path "${video_out_path}" \
  2>&1 | tee "${log_path}/${folder_name}.log"

echo "Evaluation completed. Videos saved to ${video_out_path}, logs saved to ${log_path}/${folder_name}.log"
