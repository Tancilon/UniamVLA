#!/bin/bash

###########################################################################################
# Policy server for UamVLA CALVIN ABC->D checkpoints.
# Run from the repository root in the starVLA conda environment.
#
# === Please modify the following paths according to your environment ===
export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/miniconda3/envs/uamvla/bin/python

your_ckpt=playground/Checkpoints/uamvla_calvin_abcd_phase1/checkpoints/steps_150000_pytorch_model.pt
gpu_id=0
port=5694
# === End of environment variable configuration ===
###########################################################################################

CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    "$@"
# Extra CLI args ("$@") are forwarded to server_policy.py so callers can override
# anything without editing the script. Common examples:
#   bash run_policy_server_uamvla_abcd.sh --ckpt_path .../steps_120000_pytorch_model.pt
#   bash run_policy_server_uamvla_abcd.sh --port 5710
