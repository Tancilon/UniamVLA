# CALVIN ABC->D UamVLA Evaluation Debug Notes

## 中文速记

这份文件用于后续恢复 CALVIN ABC->D 评测 debug 上下文。当前核心结论是：

- 已修过 P0 图像预处理、gripper 二值阈值、`num_sequences` 生效、`replan_steps` 生效。
- 这些修复只能小幅改善，20 sequences 仍约为 `1/5 60%`、`2/5 25%`、`3/5 0%`，说明主因大概率仍在 policy-server 推理链路。
- 最新未充分验证的关键修复是 `b5c479c Force UamVLA action token generation`，需要服务器 `git pull` 后重启 policy server 再评测。
- 下一步最该看的是 server 生成的 ACT token 数量、`raw_actions` 分布、以及离线 train sample action MSE。
- 2026-05-07 后续观察：开启 `b5c479c` 后前 `3/20` 仍为 `0%`，但前 3 条 sequence 首任务分别是 `rotate_blue_block_right`、`turn_off_led`、`lift_pink_block_slider`，样本本身偏难，不能只凭 `3/20` 判定更坏。当前更应该抓 action token / action distribution，而不是继续按 SR 盲改。

Date: 2026-05-07
Branch: `starVLA_dev`
Primary checkpoint: `playground/Checkpoints/uamvla_calvin_abcd_phase1/checkpoints/steps_150000_pytorch_model.pt`

## Symptom

UamVLA CALVIN ABC->D evaluation is far below the CALVIN D_D baselines in
`examples/calvin/README.md`. The stable pattern is not just low 1-task success;
the sequence success decays too fast.

Observed before fixes, after 116/1000 sequences:

```text
1/5 = 50.0%
2/5 = 16.4%
3/5 = 6.0%
4/5 = 2.6%
5/5 = 0.9%
```

Reference D_D baselines from `examples/calvin/README.md`:

```text
qwenpi    90.9 / 79.5 / 69.6 / 62.2 / 55.4
qwengr00t 91.7 / 81.9 / 72.7 / 65.3 / 58.1
```

ABC->D is harder than D_D, but a `50 -> 16 -> 6` chain decay points to a
pipeline/protocol issue, not just capacity.

State passthrough is active on eval:

```text
*** UamVLA state passthrough enabled (... statistics.yaml, adapter: CalvinAdapter) ***
```

## Current Status

After the fixes through `3b126d0`, a 20-sequence eval still looked poor:

```text
1/5 : 60.0%
2/5 : 25.0%
3/5 : 0.0%
4/5 : 0.0%
5/5 : 0.0%
Average successful sequence length: 0.85
```

This means image preprocessing, gripper threshold, and replan interval were not
the dominant root cause. The latest untested server-side fix is:

```text
b5c479c Force UamVLA action token generation
```

Important: this commit changes policy-server inference code. The policy server
must be restarted after pulling it.

After testing `b5c479c`, the early progress still looked bad:

```text
1/5 : 0.0%
2/5 : 0.0%
3/5 : 0.0%
4/5 : 0.0%
5/5 : 0.0%
progress: 3/20
```

The first three sequence starts in `eval_sequences.json` are all historically
hard/fragile tasks:

```text
0: rotate_blue_block_right
1: turn_off_led
2: lift_pink_block_slider
```

So `0/3` is not enough by itself. The next step is to run with
`UAMVLA_DEBUG_ACTIONS=1` and inspect action-token counts plus normalized/raw
action ranges.

## Fixes Already Landed

### `77a396f Align CALVIN eval rendering with training`

Purpose: remove the P0 image preprocessing mismatch.

Changes:

- CALVIN eval now uses `CalvinEnvAdapter(env).render_cameras(width=256, height=256)`.
- Eval feeds training-style `rgb_static` and `rgb_wrist`.
- `ModelClient._resize_image()` no-ops when the image is already target-sized.
- `run_uamvla_calvin_abcd_eval.sh` sets `--args.resize_size 256` and uses train renderer.

Verdict: useful alignment, but not the main cause. Performance stayed poor.

### `300ee87 Fix CALVIN eval bool flag`

Purpose: fix tyro bool CLI syntax.

Problem:

```bash
--args.use_train_renderer True
```

caused:

```text
Unrecognized options: True
```

Fix:

```bash
--args.use_train_renderer
```

### `bbdc6de Fix CALVIN gripper threshold`

Purpose: fix P1 gripper convention mismatch.

Evidence:

- Training uses CALVIN `rel_actions` directly.
- CALVIN gripper convention is signed, normally `+1=open`, `-1=closed`.
- The generic `ModelClient` thresholded gripper at `0.5`, which is suitable for
  `[0, 1]` style outputs but not signed CALVIN outputs.

Fix:

- `ModelClient` keeps default `gripper_binarize_threshold=0.5`.
- CALVIN eval passes `gripper_binarize_threshold=0.0`.

Verdict: correct protocol fix, but not dominant. A 20-sequence result after this
was approximately:

```text
1/5 : 50.0%
2/5 : 20.0%
3/5 : 5.0%
4/5 : 0.0%
5/5 : 0.0%
```

### `8fe713f Respect CALVIN eval sequence limit`

Purpose: make small A/B evals real.

Problem:

Running:

```bash
bash examples/calvin/eval_files/run_uamvla_calvin_abcd_eval.sh --args.num_sequences 20
```

still showed a tqdm denominator of `1000`, because eval loaded all sequences and
never sliced by `num_sequences`. The script also passed a fixed
`--args.num_sequences 1000`, which made overrides fragile.

Fix:

- Added `_limit_eval_sequences(eval_sequences, num_sequences)`.
- Removed fixed `--args.num_sequences 1000` from the wrapper script.

### `3b126d0 Honor CALVIN eval replan steps`

Purpose: make `replan_steps=5` actually affect action queries.

Problem:

`eval_calvin.py` had `replan_steps=5`, but `ModelClient` queried the policy
server only every checkpoint chunk, i.e. `future_action_window_size + 1 = 8`.
CALVIN was therefore running one closed-loop action plus seven open-loop steps.

Fix:

- Added `action_query_interval` to `ModelClient`.
- Default stays chunk size, so other benchmarks retain old behavior.
- CALVIN passes `action_query_interval=replan_steps`.

Verdict: small improvement only. A 20-sequence result became:

```text
1/5 : 60.0%
2/5 : 25.0%
3/5 : 0.0%
4/5 : 0.0%
5/5 : 0.0%
```

### `b5c479c Force UamVLA action token generation`

Purpose: fix a likely server-side action generation failure mode.

Hypothesis:

UamVLA inference previously relied on `ActionLogitsProcessor` detecting
`<|action_start|>` from the `input_ids` passed through Hugging Face
`generate(input_ids=..., inputs_embeds=...)`. If that detection does not fire
on the first decode step, generation is not constrained to `<ACT_*>` tokens.
Then `_decode_generated_actions()` filters out non-action tokens and pads missing
tokens with the middle bin, producing near-zero conservative actions.

This failure mode matches the observed behavior: some single tasks succeed, but
long-horizon chains quickly collapse.

Fix:

- `ActionLogitsProcessor(force_active=True)` can force action-token masking from
  the first generated token.
- `UamVLA.predict_action()` now sets `force_active=True` by default.

Status: needs server-side retest. Restart the policy server after pulling this
commit.

## Evidence That Is Still Missing

The next useful evidence should come from the server/action path, not more
client-side guessing.

Need to inspect:

- Number of generated ACT tokens per server call. Expected: `8 * 7 = 56`.
- `generated_ids` shape and whether returned sequence is prompt+new or new-only.
- Normalized action stats per chunk: min, max, mean, std, per-dim values.
- Unnormalized `raw_actions` stats client-side, especially whether xyz/rot are
  near zero and whether gripper is saturated.
- Offline action MSE on training samples from `datasets/uamvla_calvin/task_ABC_D`.

If `b5c479c` does not improve eval, add temporary trace logging rather than more
behavioral fixes.

## Suggested Next Diagnostics

### 1. Server action-token trace

Add temporary logging in `starVLA/model/framework/VLM4A/UamVLA.py` around
`predict_action()`:

- `generated_ids.shape`
- `prompt_len`
- `chunk_len`
- count of ids in `[self._act0_id, self._act0_id + n_bins)`
- first 10 decoded action tokens
- normalized action `min/max/mean/std`

Expected healthy signal:

```text
act_count = 56
normalized_actions not all near 0
rotation and position dims have nontrivial variance
```

### 2. Client raw-action trace

Add temporary logging in `examples/LIBERO/eval_files/model2libero_interface.py`
after unnormalization:

- normalized action stats before unnorm
- raw action stats after unnorm
- first action of each chunk
- query step number

Expected healthy CALVIN raw ranges:

```text
xyz deltas roughly around [-0.02, 0.02]
rot deltas roughly around [-0.1, 0.1]
gripper signed/binary path should end as +/-1 in eval_calvin.py
```

### 3. Offline train-sample action MSE

Run the checkpoint on held-out or training JSONL samples without CALVIN env
rollout. Compare:

- model `normalized_actions[:, :H, :]`
- dataset `example["action"]`

If offline MSE is already poor, the issue is checkpoint/training quality or
model action-token generation, not simulator eval.

### 4. Dataset/checkpoint sanity

Check these files on the server:

```bash
playground/Checkpoints/uamvla_calvin_abcd_phase1/config.yaml
playground/Checkpoints/uamvla_calvin_abcd_phase1/dataset_statistics.json
playground/Checkpoints/uamvla_calvin_abcd_phase1/statistics.yaml
datasets/uamvla_calvin/task_ABC_D/statistics.yaml
datasets/uamvla_calvin/task_ABC_D/data.jsonl
```

Confirm:

- `framework.name == UamVLA`
- `future_action_window_size == 7`
- `dataset_statistics.json["franka_calvin"]["action"]["mask"] == [true]*6 + [false]`
- action min/max look like CALVIN relative deltas, not absolute poses
- `statistics.yaml` in run dir matches the dataset used for training

## Server Commands

Terminal 1: restart policy server after pulling latest code.

```bash
cd /inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UamVLA
git pull --ff-only origin starVLA_dev
git rev-parse --short HEAD

mkdir -p logs && set -o pipefail
WANDB_MODE=offline bash examples/calvin/eval_files/run_policy_server_uamvla_abcd.sh \
  2>&1 | tee "logs/run_policy_server_uamvla_abcd_$(date +%Y%m%d_%H%M%S).log"
```

Terminal 2: run a 20-sequence smoke eval.

```bash
cd /inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UamVLA

mkdir -p logs && set -o pipefail
WANDB_MODE=offline bash examples/calvin/eval_files/run_uamvla_calvin_abcd_eval.sh --args.num_sequences 20 \
  2>&1 | tee "logs/run_uamvla_calvin_abcd_eval_$(date +%Y%m%d_%H%M%S).log"
```

## Local Dirty Files Not Related To This Debug

Local working tree had unrelated pre-existing changes throughout the debug:

```text
 M .gitignore
 D datasets/uamvla_test/libero_spatial/statistics.yaml
?? uv.lock
```

Do not revert these unless explicitly requested.
