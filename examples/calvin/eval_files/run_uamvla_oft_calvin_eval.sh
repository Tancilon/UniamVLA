#!/bin/bash

###########################################################################################
# CALVIN ABCD->D eval client for UamVLAOFT checkpoints.
# Run from the repository root in the calvin conda environment, AFTER the
# matching policy server (run_policy_server_uamvla_oft.sh) is up.
#
# eval_calvin.py uses tyro CLI; arguments follow the --args.<name> convention.
# unnorm_key uses the embodiment tag because train_starvla.py writes
# dataset_statistics.json keyed by embodiment (see _save_dataset_statistics_json).
# For uamvla_calvin_abcd the embodiment is `franka_calvin`.
#
# === Please modify the following paths according to your environment ===
export PYTHONPATH=$(pwd):${PYTHONPATH}
export calvin_python=/inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/miniconda3/envs/calvin_env/bin/python

host=${HOST:-"127.0.0.1"}
base_port=${PORT:-5694}
unnorm_key=${UNNORM_KEY:-"franka_calvin"}

your_ckpt=${CKPT_PATH:-playground/Checkpoints/uamvla_oft_calvin_abcd_baseline/checkpoints/latest_pytorch_model.pt}

# Original CALVIN dataset (must contain validation/). ABCD->D evaluates on D env.
dataset_path=${CALVIN_DATASET_PATH:-/inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UamVLA/datasets/task_ABCD_D}
calvin_config_path=${CALVIN_CONFIG_PATH:-/inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UamVLA/third_party/calvin/calvin_models/conf}
eval_sequences_path=${EVAL_SEQUENCES_PATH:-examples/calvin/eval_files/eval_sequences.json}
num_sequences=${NUM_SEQUENCES:-1000}
# === End of environment variable configuration ===
###########################################################################################

set -euo pipefail

# Anchor eval_log_dir to <project_root>/logs/... regardless of CWD.
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
eval_log_dir="${project_root}/logs/calvin_eval/uamvla_oft_${folder_name}"
mkdir -p "${eval_log_dir}"

${calvin_python} ./examples/calvin/eval_files/eval_calvin.py \
    --args.host "${host}" \
    --args.port ${base_port} \
    --args.resize_size 256 \
    --args.use_train_renderer \
    --args.gripper_binarize_threshold 0.0 \
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
#   bash run_uamvla_oft_calvin_eval.sh --args.num_sequences 10 --args.debug
#   bash run_uamvla_oft_calvin_eval.sh --args.port 5710
