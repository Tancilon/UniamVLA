#!/bin/bash

###########################################################################################
# Policy server for UamVLAOFT CALVIN ABCD->D checkpoints.
# Run from the repository root in the starVLA (uamvla) conda environment.
#
# server_policy.py routes through FRAMEWORK_REGISTRY: baseframework.from_pretrained()
# auto-detects the framework class (UamVLAOFT was registered in PR 4 via
# @FRAMEWORK_REGISTRY.register("UamVLAOFT") and is auto-imported through the
# framework package's _auto_import_framework_modules() scan).
#
# === Please modify the following paths according to your environment ===
export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/miniconda3/envs/uamvla/bin/python

your_ckpt=${CKPT_PATH:-playground/Checkpoints/uamvla_oft_calvin_abcd_baseline/checkpoints/latest_pytorch_model.pt}
gpu_id=${GPU_ID:-0}
port=${PORT:-5694}
# === End of environment variable configuration ===
###########################################################################################

set -euo pipefail

CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    "$@"
# Extra CLI args ("$@") are forwarded to server_policy.py so callers can override
# anything without editing the script. Common examples:
#   CKPT_PATH=playground/.../steps_120000_pytorch_model.pt bash run_policy_server_uamvla_oft.sh
#   bash run_policy_server_uamvla_oft.sh --port 5710 --idle_timeout -1
