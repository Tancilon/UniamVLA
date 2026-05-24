# LIBERO Local Small Evaluation Design

Date: 2026-05-24

## Goal

Run a small local LIBERO evaluation for the real LIBERO intermediate checkpoint:

`playground/Checkpoints/uamvla_gr00t_libero_8b_h8_no_pose_bs128/checkpoints/steps_5000_pytorch_model.pt`

The evaluation should cover the four standard LIBERO suites at small scale:

- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_10`

The local machine already has two conda environments:

- `uamvla` for the StarVLA/UamVLA policy server
- `libero_env` for LIBERO simulation and evaluation

## Chosen Approach

Use the existing two-terminal workflow with minimal script changes:

1. Start the policy server from `examples/LIBERO/eval_files/run_policy_server.sh`.
2. Start the LIBERO evaluator from `examples/LIBERO/eval_files/eval_libero.sh`.
3. Manually run each suite in sequence.

This keeps the original evaluation structure intact and avoids introducing a new wrapper or changing the parallel evaluation scripts.

## Script Changes

### `run_policy_server.sh`

Make the script local-friendly and configurable:

- Resolve the repository root from the script location instead of using a remote hard-coded path.
- Default `CKPT` to the LIBERO 8B intermediate checkpoint.
- Support environment variable overrides:
  - `CKPT`
  - `GPU_ID`
  - `PORT`
  - `STARVLA_PYTHON`
- Default `STARVLA_PYTHON` to `conda run -n uamvla python` if no explicit Python path is provided.
- Start only the policy server.
- Do not kill or inspect unrelated processes.

Expected usage:

```bash
GPU_ID=0 PORT=6694 bash examples/LIBERO/eval_files/run_policy_server.sh
```

### `eval_libero.sh`

Make the evaluator local-friendly and configurable:

- Resolve the repository root from the script location instead of using a remote hard-coded path.
- Default `CKPT` to the same LIBERO 8B intermediate checkpoint so action unnormalization statistics are read from the matching run directory.
- Support environment variable overrides:
  - `CKPT`
  - `LIBERO_HOME`
  - `LIBERO_PYTHON`
  - `HOST`
  - `PORT`
  - `TASK_SUITE`
  - `NUM_TRIALS_PER_TASK`
- Default `LIBERO_PYTHON` to `conda run -n libero_env python` if no explicit Python path is provided.
- Save results under the checkpoint run directory:
  - `results/<suite>/...`
- Start only the evaluator.
- Do not start or kill the policy server.

Expected usage:

```bash
TASK_SUITE=libero_spatial NUM_TRIALS_PER_TASK=2 PORT=6694 bash examples/LIBERO/eval_files/eval_libero.sh
```

## Evaluation Flow

1. Use `nvidia-smi` to choose an idle GPU manually.
2. Start the policy server in the `uamvla` environment:

```bash
GPU_ID=<idle_gpu> PORT=6694 bash examples/LIBERO/eval_files/run_policy_server.sh
```

3. In another terminal, run a one-trial smoke test in the `libero_env` environment:

```bash
TASK_SUITE=libero_spatial NUM_TRIALS_PER_TASK=1 PORT=6694 bash examples/LIBERO/eval_files/eval_libero.sh
```

4. If the smoke test succeeds, run the four small evaluations sequentially:

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  TASK_SUITE="$suite" NUM_TRIALS_PER_TASK=2 PORT=6694 \
    bash examples/LIBERO/eval_files/eval_libero.sh
done
```

## Safety Requirements

- The scripts must not kill processes that they did not start.
- GPU selection is manual through `GPU_ID`.
- The scripts must not automatically select a GPU or clear GPU memory.
- If a selected GPU is busy, the user chooses another GPU and reruns the command.
- If a selected port is busy, the user chooses another `PORT` and starts both scripts with the same value.

## Error Handling

- If the checkpoint path does not exist, the scripts should fail early with a clear message.
- If `LIBERO_HOME` is missing or invalid, `eval_libero.sh` should fail early with a clear message.
- If the policy server is unavailable, `eval_libero.py` will fail during client connection or inference; the log path should make this easy to diagnose.
- If conda environments are unavailable, the shell command should fail visibly.

## Validation

The implementation is considered ready when:

- `run_policy_server.sh` prints the resolved checkpoint path, GPU id, port, and Python command.
- `eval_libero.sh` prints the resolved checkpoint path, suite, trial count, port, `LIBERO_HOME`, and Python command.
- A `libero_spatial` one-trial smoke test can connect to the policy server and write logs/videos under the checkpoint run directory.
- The same evaluator command can be reused for the other three suites by changing `TASK_SUITE`.
