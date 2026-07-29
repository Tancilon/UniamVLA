#!/usr/bin/env bash
# Batch sidecar generation for RoboTwin2 LeRobot tasks.
#
# This launcher mirrors the UamGR00T_LT CALVIN sidecar pipeline for every task
# under datasets/robotwin2/RoboTwin-Clean. It only orchestrates existing
# preprocessors; each preprocessor remains resumable unless --overwrite is set.
#
# Smoke test without writing:
#   bash runners/preprocess_robotwin2_sidecars.sh --dry-run --task adjust_bottle
#
# Process one task on the default high camera:
#   CUDA_VISIBLE_DEVICES=0 bash runners/preprocess_robotwin2_sidecars.sh --task adjust_bottle
#
# Process all tasks with a small limit:
#   CUDA_VISIBLE_DEVICES=0 bash runners/preprocess_robotwin2_sidecars.sh --limit 2
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET_ROOT="${DATASET_ROOT:-datasets/robotwin2/RoboTwin-Clean}"
CAMERA="${CAMERA:-observation.images.cam_high}"
TASKS=()
STAGES="${STAGES:-depth_vda,depth_latent,affordance_vrb,affordance_px,rgb_latent}"

DRY_RUN=0
VERIFY_ONLY=0
OVERWRITE=0
LIMIT="${LIMIT:--1}"

DEVICE="${DEVICE:-cuda}"
QWEN_IMAGE_SIZE="${QWEN_IMAGE_SIZE:-224}"
VAE_PATH="${VAE_PATH:-ckpt/pretrained_vae}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-32}"
REPORT_PSNR=0

VDA_ENCODER="${VDA_ENCODER:-vitl}"
VDA_INPUT_SIZE="${VDA_INPUT_SIZE:-518}"
VDA_CHECKPOINT="${VDA_CHECKPOINT:-}"
VDA_VIS_FIRST_N="${VDA_VIS_FIRST_N:-0}"

AFFORDANCE_OBJECTS="${AFFORDANCE_OBJECTS:-}"
GDINO_PATH="${GDINO_PATH:-}"
VRB_CKPT="${VRB_CKPT:-}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.3}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
K_RATIO="${K_RATIO:-6.0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
GDINO_BATCH="${GDINO_BATCH:-16}"
MAX_BOXES_PER_FRAME="${MAX_BOXES_PER_FRAME:-8}"
AFFORDANCE_SEED="${AFFORDANCE_SEED:-0}"
AFFORDANCE_SAVE_VIS="${AFFORDANCE_SAVE_VIS:-0}"

usage() {
  cat <<'USAGE'
Usage:
  bash runners/preprocess_robotwin2_sidecars.sh [options]

Options:
  --dataset-root PATH       Root containing per-task LeRobot datasets.
                            Default: datasets/robotwin2/RoboTwin-Clean
  --camera NAME             Video/sidecar camera key.
                            Default: observation.images.cam_high
  --task NAME               Add one task. May be repeated. Default: all tasks.
  --stages LIST             Comma list from:
                            depth_vda,depth_latent,affordance_vrb,affordance_px,rgb_latent
  --limit N                 Pass --limit N to each preprocessor.
  --overwrite               Recompute existing artifacts.
  --verify-only             Run each selected preprocessor in --verify_only mode.
  --dry-run                 Print commands without executing.
  --device DEVICE           Device for torch stages. Default: cuda
  --vae-path PATH           VAE path for latent stages. Default: ckpt/pretrained_vae
  --qwen-image-size N       Must match training YAML. Default: 224
  --vae-batch-size N        Batch size for VAE latent stages. Default: 32
  --report-psnr             Add --report_psnr to latent stages.
  --vda-encoder NAME        vits, vitb, or vitl. Default: vitl
  --vda-input-size N        Default: 518
  --vda-checkpoint PATH     Optional VDA checkpoint override.
  --vda-vis-first-n N       Save depth visualizations for first N processed videos.
  --objects TEXT            GroundingDINO object prompt for affordance_vrb.
  --gdino-path PATH         GroundingDINO local model path override.
  --vrb-ckpt PATH           VRB checkpoint override.
  --frame-stride N          Affordance inference stride. Default: 1
  --help                    Show this help.

Environment variables with matching uppercase names can also be used, e.g.
DATASET_ROOT, CAMERA, STAGES, LIMIT, VAE_PATH, AFFORDANCE_OBJECTS.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --camera) CAMERA="$2"; shift 2 ;;
    --task) TASKS+=("$2"); shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --vae-path) VAE_PATH="$2"; shift 2 ;;
    --qwen-image-size) QWEN_IMAGE_SIZE="$2"; shift 2 ;;
    --vae-batch-size) VAE_BATCH_SIZE="$2"; shift 2 ;;
    --report-psnr) REPORT_PSNR=1; shift ;;
    --vda-encoder) VDA_ENCODER="$2"; shift 2 ;;
    --vda-input-size) VDA_INPUT_SIZE="$2"; shift 2 ;;
    --vda-checkpoint) VDA_CHECKPOINT="$2"; shift 2 ;;
    --vda-vis-first-n) VDA_VIS_FIRST_N="$2"; shift 2 ;;
    --objects) AFFORDANCE_OBJECTS="$2"; shift 2 ;;
    --gdino-path) GDINO_PATH="$2"; shift 2 ;;
    --vrb-ckpt) VRB_CKPT="$2"; shift 2 ;;
    --box-threshold) BOX_THRESHOLD="$2"; shift 2 ;;
    --text-threshold) TEXT_THRESHOLD="$2"; shift 2 ;;
    --k-ratio) K_RATIO="$2"; shift 2 ;;
    --frame-stride) FRAME_STRIDE="$2"; shift 2 ;;
    --gdino-batch) GDINO_BATCH="$2"; shift 2 ;;
    --max-boxes-per-frame) MAX_BOXES_PER_FRAME="$2"; shift 2 ;;
    --affordance-seed) AFFORDANCE_SEED="$2"; shift 2 ;;
    --affordance-save-vis) AFFORDANCE_SAVE_VIS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Dataset root not found: $DATASET_ROOT" >&2
  exit 1
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
  mapfile -t TASKS < <(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d \
    ! -name '.cache' -printf '%f\n' | sort)
fi

IFS=',' read -r -a STAGE_LIST <<< "$STAGES"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

append_common_flags() {
  local -n out=$1
  if [[ "$OVERWRITE" -eq 1 ]]; then
    out+=(--overwrite)
  fi
  if [[ "$VERIFY_ONLY" -eq 1 ]]; then
    out+=(--verify_only)
  fi
  if [[ "$LIMIT" != "-1" ]]; then
    out+=(--limit "$LIMIT")
  fi
}

echo "dataset_root=$DATASET_ROOT"
echo "camera=$CAMERA"
echo "stages=$STAGES"
echo "tasks=${#TASKS[@]}"

for task in "${TASKS[@]}"; do
  task_root="$DATASET_ROOT/$task"
  if [[ ! -d "$task_root/meta" || ! -d "$task_root/videos" ]]; then
    echo "Skipping non-LeRobot task directory: $task_root" >&2
    continue
  fi

  echo
  echo "==> $task"
  for stage in "${STAGE_LIST[@]}"; do
    case "$stage" in
      depth_vda)
        cmd=(python runners/preprocess_depth_vda.py
          --dataset_root "$task_root"
          --camera "$CAMERA"
          --encoder "$VDA_ENCODER"
          --input_size "$VDA_INPUT_SIZE"
          --vis_first_n "$VDA_VIS_FIRST_N")
        [[ -n "$VDA_CHECKPOINT" ]] && cmd+=(--checkpoint "$VDA_CHECKPOINT")
        append_common_flags cmd
        run_cmd "${cmd[@]}"
        ;;
      depth_latent)
        cmd=(python runners/preprocess_depth_latent.py
          --dataset_root "$task_root"
          --camera "$CAMERA"
          --vae_path "$VAE_PATH"
          --qwen_image_size "$QWEN_IMAGE_SIZE"
          --batch_size "$VAE_BATCH_SIZE"
          --device "$DEVICE")
        [[ "$REPORT_PSNR" -eq 1 ]] && cmd+=(--report_psnr)
        append_common_flags cmd
        run_cmd "${cmd[@]}"
        ;;
      affordance_vrb)
        cmd=(python runners/preprocess_affordance_vrb.py
          --dataset_root "$task_root"
          --camera "$CAMERA"
          --box_threshold "$BOX_THRESHOLD"
          --text_threshold "$TEXT_THRESHOLD"
          --k_ratio "$K_RATIO"
          --frame_stride "$FRAME_STRIDE"
          --gdino_batch "$GDINO_BATCH"
          --max_boxes_per_frame "$MAX_BOXES_PER_FRAME"
          --seed "$AFFORDANCE_SEED"
          --save_vis "$AFFORDANCE_SAVE_VIS"
          --device "$DEVICE")
        [[ -n "$AFFORDANCE_OBJECTS" ]] && cmd+=(--objects "$AFFORDANCE_OBJECTS")
        [[ -n "$GDINO_PATH" ]] && cmd+=(--gdino_path "$GDINO_PATH")
        [[ -n "$VRB_CKPT" ]] && cmd+=(--vrb_ckpt "$VRB_CKPT")
        append_common_flags cmd
        run_cmd "${cmd[@]}"
        ;;
      affordance_px)
        cmd=(python runners/preprocess_affordance_px.py
          --dataset_root "$task_root"
          --camera "$CAMERA"
          --qwen_image_size "$QWEN_IMAGE_SIZE")
        append_common_flags cmd
        run_cmd "${cmd[@]}"
        ;;
      rgb_latent)
        cmd=(python runners/preprocess_rgb_latent.py
          --dataset_root "$task_root"
          --camera "$CAMERA"
          --vae_path "$VAE_PATH"
          --qwen_image_size "$QWEN_IMAGE_SIZE"
          --batch_size "$VAE_BATCH_SIZE"
          --device "$DEVICE")
        [[ "$REPORT_PSNR" -eq 1 ]] && cmd+=(--report_psnr)
        append_common_flags cmd
        run_cmd "${cmd[@]}"
        ;;
      "")
        ;;
      *)
        echo "Unknown stage: $stage" >&2
        exit 2
        ;;
    esac
  done
done
