#!/bin/bash
set -euo pipefail

###########################################################################################
# === Please modify the following paths according to your environment ===
export PYTHONPATH=$(pwd):${PYTHONPATH:-} # let Calvin client find websocket tools from main repo
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-calvin}
calvin_python=${CALVIN_PYTHON:-/home/user01/miniconda3/envs/calvin_venv/bin/python}

host=${HOST:-127.0.0.1}
base_port=${PORT:-5694}
unnorm_key=${UNNORM_KEY:-franka}
your_ckpt=${CKPT_PATH:-playground/Checkpoints/uamvla_gr00t_calvin_d_4b_h8_state7_recon_bs128_200k/checkpoints/steps_5000_pytorch_model.pt}
dataset_path=${DATASET_PATH:-datasets/calvin/task_D_D}
calvin_config_path=${CALVIN_CONFIG_PATH:-third_party/calvin/calvin_models/conf}
eval_sequences_path=${EVAL_SEQUENCES_PATH:-examples/calvin/eval_files/eval_sequences.json}
num_sequences=${NUM_SEQUENCES:-1000}

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR=${LOG_DIR:-logs/calvin_eval_${folder_name}_$(date +"%Y%m%d_%H%M%S")}
mkdir -p ${LOG_DIR}

${calvin_python} ./examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path "${your_ckpt}" \
    --args.unnorm-key "${unnorm_key}" \
    --args.host "$host" \
    --args.port "$base_port" \
    --args.dataset_path "${dataset_path}" \
    --args.calvin_config_path "${calvin_config_path}" \
    --args.eval_sequences_path "${eval_sequences_path}" \
    --args.num_sequences "$num_sequences" \
    --args.eval_log_dir "${LOG_DIR}" \
    "$@"
