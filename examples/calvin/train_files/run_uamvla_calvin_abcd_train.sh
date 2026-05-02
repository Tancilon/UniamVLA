

# NCCL networking: auto-detect for single-node 8-GPU runs.
# - Bootstrap: NCCL picks the first non-`lo` UP interface (verified: this
#   container exposes only `eth0` + `lo` under /sys/class/net/).
# - Data: NVLink + SHM within one node — IB is not used (`ibv_devices`
#   returns an empty list inside this container).
# For multi-node, set these explicitly to your cluster's actual interface
# names (typical: eth0 / ens3 / bond0 for socket; mlx5_X for IB HCA).
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=UamVLA
freeze_module_list=''
base_vlm=ckpt/Qwen3-VL-8B-Instruct
config_yaml=./starVLA/config/training/uamvla_calvin_abcd.yaml
calvin_data_root=datasets/uamvla_calvin
data_mix=calvin_abc_d_uamvla
run_root_dir=./playground/Checkpoints
run_id=uamvla_calvin_abcd_phase1
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${calvin_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 5000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project uamvla_calvin \
  --wandb_entity tancilon1-fudan-university-school-of-management \
  "$@"
  # Extra CLI args ("$@") are forwarded to train_starvla.py so callers can
  # override anything without editing the script. Common examples:
  #   bash run_uamvla_calvin_abcd_train.sh --trainer.is_resume true
  #   bash run_uamvla_calvin_abcd_train.sh --trainer.max_train_steps 150000
  #   bash run_uamvla_calvin_abcd_train.sh --is_debug true



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
