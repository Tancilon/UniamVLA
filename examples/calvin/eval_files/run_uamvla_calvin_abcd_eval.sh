#!/bin/bash

###########################################################################################
# CALVIN ABC->D eval client for UamVLA checkpoints.
# Run from the repository root in the calvin conda environment, AFTER the
# matching policy server (run_policy_server_uamvla_abcd.sh) is up.
#
# unnorm_key uses the embodiment tag because train_starvla.py writes
# dataset_statistics.json keyed by embodiment (see _save_dataset_statistics_json).
# For calvin_abc_d_uamvla the embodiment is `franka_calvin`.
#
# === Please modify the following paths according to your environment ===
export PYTHONPATH=$(pwd):${PYTHONPATH}
export calvin_python=/inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/miniconda3/envs/calvin_env/bin/python

host="127.0.0.1"
base_port=5694
unnorm_key="franka_calvin"

your_ckpt=playground/Checkpoints/uamvla_calvin_abcd_phase1/checkpoints/steps_150000_pytorch_model.pt

# Original CALVIN dataset (must contain validation/). ABC->D evaluates on D env.
dataset_path=/inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UamVLA/datasets/task_ABC_D
calvin_config_path=/inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UamVLA/third_party/calvin/calvin_models/conf
eval_sequences_path=examples/calvin/eval_files/eval_sequences.json
num_sequences=1000
# === End of environment variable configuration ===
###########################################################################################

# Anchor eval_log_dir to <project_root>/logs/... regardless of CWD.
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
eval_log_dir="${project_root}/logs/calvin_eval/${folder_name}"
mkdir -p "${eval_log_dir}"

${calvin_python} ./examples/calvin/eval_files/eval_calvin.py \
    --args.host "${host}" \
    --args.port ${base_port} \
    --args.resize_size 256 \
    --args.use_train_renderer True \
    --args.pretrained-path ${your_ckpt} \
    --args.unnorm-key ${unnorm_key} \
    --args.dataset_path ${dataset_path} \
    --args.calvin_config_path ${calvin_config_path} \
    --args.eval_sequences_path ${eval_sequences_path} \
    --args.num_sequences ${num_sequences} \
    --args.eval_log_dir "${eval_log_dir}" \
    "$@"
# Extra CLI args ("$@") are forwarded to eval_calvin.py so callers can override
# anything without editing the script. Common examples:
#   bash run_uamvla_calvin_abcd_eval.sh --args.num_sequences 10 --args.debug True
#   bash run_uamvla_calvin_abcd_eval.sh --args.port 5710
