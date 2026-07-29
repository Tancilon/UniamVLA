#!/bin/bash
# Seer-aligned finetune: UamGR00T_DT on RoboTwin all tasks (K=10, action_chunk=16, exec=1)
# Usage:
#   bash run_uamgr00t_DT_robotwin_finetune.sh [extra args]
# Examples:
#   # Resume from checkpoint:
#   bash run_uamgr00t_DT_robotwin_finetune.sh --trainer.is_resume true
#
#   # Load pretrained weights (Seer two-stage):
#   bash run_uamgr00t_DT_robotwin_finetune.sh \
#     --trainer.resume_from_checkpoint results/Checkpoints/uamvla_gr00t_dt_robotwin_pretrain/checkpoint_100000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/run_uamgr00t_DT_robotwin_finetune.yaml"

# Explicitly load ds_config.yaml so DeepSpeedPlugin() (module-level in
# train_starvla.py) picks up bf16: true instead of falling back to fp32.
ACCELERATE_DEEPSPEED_CONFIG_FILE="${REPO_ROOT}/starVLA/config/deepseeds/ds_config.yaml" \
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  "$@"
