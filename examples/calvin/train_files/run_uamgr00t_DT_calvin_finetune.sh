#!/bin/bash
# Seer-aligned finetune: UamGR00T_DT on CALVIN ABC (K=10, action_chunk=3, exec=1)
# Usage:
#   bash run_uamgr00t_DT_calvin_finetune.sh [extra args]
# Examples:
#   # Resume from checkpoint:
#   bash run_uamgr00t_DT_calvin_finetune.sh --trainer.is_resume true
#
#   # Load pretrained weights (Seer two-stage):
#   bash run_uamgr00t_DT_calvin_finetune.sh \
#     --trainer.resume_from_checkpoint results/Checkpoints/uamvla_gr00t_dt_calvin_pretrain/checkpoint_50000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/run_uamgr00t_DT_calvin_finetune.yaml"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  "$@"
