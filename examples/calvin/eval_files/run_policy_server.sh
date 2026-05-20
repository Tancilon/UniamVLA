#!/bin/bash
set -euo pipefail

export PYTHONPATH=$(pwd):${PYTHONPATH:-} # let LIBERO find the websocket tools from main repo
star_vla_python=${STAR_VLA_PYTHON:-/home/user01/miniconda3/envs/uamvla/bin/python}
your_ckpt=${CKPT_PATH:-playground/Checkpoints/uamvla_gr00t_calvin_d_4b_h8_state7_recon_bs128_200k/checkpoints/steps_5000_pytorch_model.pt}
gpu_id=${GPU_ID:-0}
port=${PORT:-5694}
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    --use_bf16 \
    --idle_timeout -1 \
    "$@"

# #################################
