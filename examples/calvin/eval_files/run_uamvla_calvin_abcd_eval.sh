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
export calvin_python=/path/to/your/conda/envs/calvin/bin/python

host="127.0.0.1"
base_port=5694
unnorm_key="franka_calvin"

your_ckpt=playground/Checkpoints/uamvla_calvin_abcd_phase1/checkpoints/steps_150000_pytorch_model.pt

# Original CALVIN dataset (must contain validation/). ABC->D evaluates on D env.
dataset_path=/path/to/calvin/task_D_D
calvin_config_path=/path/to/calvin/calvin_models/conf
eval_sequences_path=examples/calvin/eval_files/eval_sequences.json
num_sequences=1000
# === End of environment variable configuration ===
###########################################################################################

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

${calvin_python} ./examples/calvin/eval_files/eval_calvin.py \
    --args.host "${host}" \
    --args.port ${base_port} \
    --args.pretrained-path ${your_ckpt} \
    --args.unnorm-key ${unnorm_key} \
    --args.dataset_path ${dataset_path} \
    --args.calvin_config_path ${calvin_config_path} \
    --args.eval_sequences_path ${eval_sequences_path} \
    --args.num_sequences ${num_sequences} \
    --args.eval_log_dir tmp/calvin/eval_logs/${folder_name} \
    "$@"
# Extra CLI args ("$@") are forwarded to eval_calvin.py so callers can override
# anything without editing the script. Common examples:
#   bash run_uamvla_calvin_abcd_eval.sh --args.num_sequences 10 --args.debug True
#   bash run_uamvla_calvin_abcd_eval.sh --args.port 5710
