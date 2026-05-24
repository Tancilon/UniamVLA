# LIBERO Local Small Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing LIBERO two-terminal evaluation scripts run locally against the downloaded `uamvla_gr00t_libero_8b_h8_no_pose_bs128` checkpoint with manual GPU selection and small four-suite evaluation.

**Architecture:** Keep the existing server/client split. `run_policy_server.sh` starts only the UamVLA policy server in the `uamvla` environment; `eval_libero.sh` starts only the LIBERO evaluator in the `libero_env` environment and reads action statistics from the same checkpoint run directory.

**Tech Stack:** Bash, conda, Python, StarVLA/UamVLA policy server, LIBERO evaluator, websocket client/server.

---

## Files

- Modify: `examples/LIBERO/eval_files/run_policy_server.sh`
  - Responsibility: start the policy server with a local default checkpoint, configurable GPU, port, and Python command.
- Modify: `examples/LIBERO/eval_files/eval_libero.sh`
  - Responsibility: run the LIBERO evaluator with local defaults, configurable suite/trial count/port/Python/LIBERO_HOME, and result paths under the checkpoint directory.
- No new runtime scripts.
- Do not modify `examples/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall.sh`.
- Do not modify `examples/LIBERO/eval_files/auto_eval_scripts/auto_eval_libero.sh`.

## Task 1: Localize Policy Server Script

**Files:**
- Modify: `examples/LIBERO/eval_files/run_policy_server.sh`

- [ ] **Step 1: Replace the script body**

Replace the current file with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${STARVLA_DIR}"

DEFAULT_CKPT="${STARVLA_DIR}/playground/Checkpoints/uamvla_gr00t_libero_8b_h8_no_pose_bs128/checkpoints/steps_5000_pytorch_model.pt"

CKPT="${CKPT:-${DEFAULT_CKPT}}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6694}"
STARVLA_PYTHON="${STARVLA_PYTHON:-conda run -n uamvla python}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

echo "=========================================="
echo " UamVLA LIBERO policy server"
echo "=========================================="
echo " STARVLA_DIR    : ${STARVLA_DIR}"
echo " CKPT           : ${CKPT}"
echo " GPU_ID         : ${GPU_ID}"
echo " PORT           : ${PORT}"
echo " STARVLA_PYTHON : ${STARVLA_PYTHON}"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPU_ID}" ${STARVLA_PYTHON} deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" \
  --port "${PORT}" \
  --use_bf16
```

- [ ] **Step 2: Run Bash syntax check**

Run:

```bash
bash -n examples/LIBERO/eval_files/run_policy_server.sh
```

Expected: exit code `0` and no output.

- [ ] **Step 3: Commit**

Run:

```bash
git add examples/LIBERO/eval_files/run_policy_server.sh
git commit -m "eval: localize LIBERO policy server script"
```

## Task 2: Localize LIBERO Evaluator Script

**Files:**
- Modify: `examples/LIBERO/eval_files/eval_libero.sh`

- [ ] **Step 1: Replace the script body**

Replace the current file with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${STARVLA_DIR}"

DEFAULT_CKPT="${STARVLA_DIR}/playground/Checkpoints/uamvla_gr00t_libero_8b_h8_no_pose_bs128/checkpoints/steps_5000_pytorch_model.pt"

CKPT="${CKPT:-${DEFAULT_CKPT}}"
LIBERO_HOME="${LIBERO_HOME:-${STARVLA_DIR}/../LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-conda run -n libero_env python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-2}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

if [[ ! -d "${LIBERO_HOME}" ]]; then
  echo "[ERROR] LIBERO_HOME does not exist: ${LIBERO_HOME}" >&2
  echo "Set LIBERO_HOME=/path/to/LIBERO and rerun." >&2
  exit 1
fi

export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

folder_name="$(basename "$(dirname "$(dirname "${CKPT}")")")_$(basename "$(dirname "${CKPT}")")_$(basename "${CKPT}")"
model_root="${CKPT%%/checkpoints/*}"
video_out_path="${model_root}/results/${TASK_SUITE}/${folder_name}"

mkdir -p "${video_out_path}"

echo "=========================================="
echo " LIBERO evaluator"
echo "=========================================="
echo " STARVLA_DIR          : ${STARVLA_DIR}"
echo " CKPT                 : ${CKPT}"
echo " LIBERO_HOME          : ${LIBERO_HOME}"
echo " LIBERO_PYTHON        : ${LIBERO_PYTHON}"
echo " HOST                 : ${HOST}"
echo " PORT                 : ${PORT}"
echo " TASK_SUITE           : ${TASK_SUITE}"
echo " NUM_TRIALS_PER_TASK  : ${NUM_TRIALS_PER_TASK}"
echo " VIDEO_OUT_PATH       : ${video_out_path}"
echo "=========================================="

${LIBERO_PYTHON} ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.video-out-path "${video_out_path}"
```

- [ ] **Step 2: Run Bash syntax check**

Run:

```bash
bash -n examples/LIBERO/eval_files/eval_libero.sh
```

Expected: exit code `0` and no output.

- [ ] **Step 3: Verify early failure for invalid checkpoint**

Run:

```bash
CKPT=/tmp/does-not-exist.pt bash examples/LIBERO/eval_files/eval_libero.sh
```

Expected: exit code nonzero and stderr contains:

```text
[ERROR] Checkpoint not found: /tmp/does-not-exist.pt
```

- [ ] **Step 4: Commit**

Run:

```bash
git add examples/LIBERO/eval_files/eval_libero.sh
git commit -m "eval: localize LIBERO evaluator script"
```

## Task 3: Lightweight End-to-End Readiness Checks

**Files:**
- Verify: `examples/LIBERO/eval_files/run_policy_server.sh`
- Verify: `examples/LIBERO/eval_files/eval_libero.sh`

- [ ] **Step 1: Check both scripts parse**

Run:

```bash
bash -n examples/LIBERO/eval_files/run_policy_server.sh
bash -n examples/LIBERO/eval_files/eval_libero.sh
```

Expected: both commands exit `0` and print no syntax errors.

- [ ] **Step 2: Check script configuration output without starting a long run**

Run:

```bash
CKPT=/tmp/does-not-exist.pt GPU_ID=0 PORT=6694 bash examples/LIBERO/eval_files/run_policy_server.sh
```

Expected: exit code nonzero and stderr contains:

```text
[ERROR] Checkpoint not found: /tmp/does-not-exist.pt
```

- [ ] **Step 3: Inspect final diff**

Run:

```bash
git diff HEAD~2..HEAD -- examples/LIBERO/eval_files/run_policy_server.sh examples/LIBERO/eval_files/eval_libero.sh
```

Expected: only the two LIBERO eval scripts changed; no changes to parallel scripts.

- [ ] **Step 4: Provide user commands for manual smoke test**

Report these commands to the user:

```bash
nvidia-smi
GPU_ID=<idle_gpu> PORT=6694 bash examples/LIBERO/eval_files/run_policy_server.sh
```

In a second terminal:

```bash
TASK_SUITE=libero_spatial NUM_TRIALS_PER_TASK=1 PORT=6694 bash examples/LIBERO/eval_files/eval_libero.sh
```

For four-suite small evaluation:

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  TASK_SUITE="$suite" NUM_TRIALS_PER_TASK=2 PORT=6694 \
    bash examples/LIBERO/eval_files/eval_libero.sh
done
```

## Self-Review

- Spec coverage: The plan localizes only `run_policy_server.sh` and `eval_libero.sh`, uses the real LIBERO 8B checkpoint, keeps two-terminal workflow, supports manual GPU/port/suite/trial overrides, and avoids touching unrelated processes.
- Placeholder scan: No `TBD`, `TODO`, or unspecified implementation steps.
- Type consistency: Environment variable names are consistent across the plan and design: `CKPT`, `GPU_ID`, `PORT`, `STARVLA_PYTHON`, `LIBERO_HOME`, `LIBERO_PYTHON`, `HOST`, `TASK_SUITE`, and `NUM_TRIALS_PER_TASK`.
