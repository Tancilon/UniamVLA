#!/usr/bin/env bash
# UamGR00T_LT — joint depth, affordance, and future supervision on CALVIN ABC.
#
# Requires depth, affordance, and future sidecars in every A/B/C dataset. For each scene:
#   python runners/preprocess_depth_vda.py       --dataset_root datasets/task_ABC_D_scene_<SCENE>_lerobot
#   python runners/preprocess_depth_latent.py    --dataset_root datasets/task_ABC_D_scene_<SCENE>_lerobot
#   python runners/preprocess_affordance_vrb.py  --dataset_root datasets/task_ABC_D_scene_<SCENE>_lerobot
#   python runners/preprocess_affordance_px.py   --dataset_root datasets/task_ABC_D_scene_<SCENE>_lerobot
#   python runners/preprocess_rgb_latent.py      --dataset_root datasets/task_ABC_D_scene_<SCENE>_lerobot
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

CONFIG_YAML="${CONFIG_YAML:-./examples/calvin/train_files/run_uamgr00t_LT_depth_afford_future_train.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-./wandb}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-playground/Checkpoints}"
RUN_ID="${RUN_ID:-uamvla_gr00t_lt_calvin_abc_depth_afford_future}"

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
