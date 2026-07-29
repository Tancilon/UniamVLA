#!/bin/bash
# Seer-aligned pretrain: UamGR00T_DT on RoboTwin all tasks (K=14, future_offset=3)
#
# Designed for 8x GPU — uses ZeRO-2.
# On tighter cards (<=48 GB), override with ZeRO-3:
#   bash run_uamgr00t_DT_robotwin_pretrain.sh \
#     --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml
#
# Usage:
#   bash run_uamgr00t_DT_robotwin_pretrain.sh [extra args]
# Examples:
#   # Resume:
#   bash run_uamgr00t_DT_robotwin_pretrain.sh --trainer.is_resume true
#
# After pretrain finishes, use run_uamgr00t_DT_robotwin_finetune.sh and pass
#   --trainer.resume_from_checkpoint <pretrain_ckpt_dir>
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/run_uamgr00t_DT_robotwin_pretrain.yaml"

# Explicitly load ds_config.yaml so DeepSpeedPlugin() (module-level in
# train_starvla.py) picks up bf16: true instead of falling back to fp32.
ACCELERATE_DEEPSPEED_CONFIG_FILE="${REPO_ROOT}/starVLA/config/deepseeds/ds_config.yaml" \
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  "$@"
