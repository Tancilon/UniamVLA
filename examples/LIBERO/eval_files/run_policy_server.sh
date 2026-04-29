#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UniamVLA
LIBERO_HOME=/home/jye624/Projcets/LIBERO
STARVLA_PYTHON=$(conda run -n uamvla which python)
LIBERO_PYTHON=/home/jye624/.conda/envs/libero/bin/python

# === Checkpoint ===
CKPT=${STARVLA_DIR}/playground/Checkpoints/uamvla_libero_phase1/checkpoints/steps_50000_pytorch_model.pt

export star_vla_python=${STARVLA_PYTHON}
your_ckpt=${CKPT}   
gpu_id=0
port=6694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
