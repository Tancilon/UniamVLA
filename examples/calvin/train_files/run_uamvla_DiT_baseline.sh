#!/bin/bash
# Foreground training; all output goes to logs/. CLI overrides follow the YAML.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
exec > "logs/uamvla_dit_ds_calvin_abc_${RUN_STAMP}.log" 2>&1

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate uamvla

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export UAMVLA_TRAIN_STEP_PROFILE=0
export PYTHONWARNINGS="ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning"

exec accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${SCRIPT_DIR}/run_uamvla_DiT_calvin.yaml" \
  --run_id "uamvla_dit_ds_calvin_abc" \
  "$@"
