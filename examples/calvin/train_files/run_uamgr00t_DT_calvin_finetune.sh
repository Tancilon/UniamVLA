#!/bin/bash
# UamGR00T_DT finetune on CALVIN ABC
# Architecture: K=10 history frames + future_query_tokens in Qwen + GR00T action model
# action_chunk=8, no exec_horizon truncation
# Usage:
#   bash run_uamgr00t_DT_calvin_finetune.sh [extra args]
# Examples:
#   # Resume from checkpoint:
#   bash run_uamgr00t_DT_calvin_finetune.sh --trainer.is_resume true
#
#   # Override pretrained weights:
#   bash run_uamgr00t_DT_calvin_finetune.sh \
#     --trainer.pretrained_checkpoint /path/to/pytorch_model.pt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/run_uamgr00t_DT_calvin_finetune.yaml"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  "$@"
