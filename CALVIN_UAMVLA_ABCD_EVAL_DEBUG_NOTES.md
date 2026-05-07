# CALVIN ABC->D UamVLA Evaluation Debug Notes

## 中文速记

这份文件用于后续恢复 CALVIN ABC->D 评测 debug 上下文。当前核心结论是：

- 已修过 P0 图像预处理、gripper 二值阈值、`num_sequences` 生效、`replan_steps` 生效。
- 这些修复只能小幅改善，20 sequences 仍约为 `1/5 50-60%`、`2/5 20-25%`、`3/5 0-5%`，说明主因不在单纯 eval wrapper 小错。
- `UAMVLA_DEBUG_ACTIONS=1` 已确认 policy server 每次生成 `act_counts=[56]`，即 `8 * 7` 个 ACT token 数量正确；action token generation 不是当前最强嫌疑。
- checkpoint `dataset_statistics.json`、checkpoint `statistics.yaml`、训练数据 `task_ABC_D/statistics.yaml` 都显示 CALVIN action 7 维 min/max 为 `[-1, 1]`，所以 `raw_actions ~= normalized_actions` 对这个 UAM CALVIN pipeline 是预期现象，不再视作 P2 反归一化错误。
- 训练 action summary 覆盖 `1,071,743` 条，整体分布与 eval 输出大体同尺度；但第一个 eval task `rotate_blue_block_right` 中，模型输出的 yaw 维接近 0，而训练同义指令的 yaw 均值约 `-0.30`。
- 官方 eval language 与训练 JSONL instruction 是 `0/34` 精确匹配；但把 eval annotation 替换为训练中真实出现过的等价句子后，前 `4/20` 仍 `0%`，因此 language paraphrase shift 不是唯一/主导解释。
- `20f9bb9 Add CALVIN train-chunk probe` 已跑过：ABC train-frame 上模型通常能把 expert chunk 预测到接近 action-token 量化误差，说明 action token/generation/状态 passthrough 主路径不是坏的。
- 例外：`turn the blue block right` 这类 phrasing/trajectory mode 的 train-frame MAE 明显更高，但 yaw 方向仍大多正确；它像“checkpoint 对某些 mode 较弱”，不像整体 convention flip。
- 新增本地 probe：`tools/probes/probe_uamvla_calvin_validation_chunk.py`。下一步应在 D validation expert frame 上比较预测 chunk 与 expert `rel_actions`，判断是 ABC->D 泛化/ckpt 问题，还是只在 live rollout 闭环控制中崩。

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
the dominant root cause.

After `b5c479c` and `802ee0f`, action-token diagnostics showed the server is
loading the correct UamVLA policy and producing the expected number of action
tokens:

```text
Loaded policy class=UamVLA framework=UamVLA action_horizon=8 action_dim=7 action_start_id=151678 act0_id=151679
expected_act_tokens=56, act_counts=[56]
```

However, eval performance remains poor. After seen-language annotation
replacement, early trend still looked bad:

```text
1/5 : 0.0%
2/5 : 0.0%
3/5 : 0.0%
4/5 : 0.0%
5/5 : 0.0%
progress: 4/20
```

The first few sequence starts in `eval_sequences.json` are historically
hard/fragile, but the lack of improvement under seen-language replacement
means we should stop guessing around eval annotation and run offline expert-frame
action probes.

```text
0: rotate_blue_block_right
1: turn_off_led
2: lift_pink_block_slider
```

Current highest-yield next step:

```text
Run tools/probes/probe_uamvla_calvin_validation_chunk.py against the running policy server.
```

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

Status after server-side retest: action token count is healthy (`56/56` per
query), but eval success did not recover. This commit remains a useful guardrail,
but the evidence no longer points to missing ACT-token masking as the main root
cause.

### `802ee0f Add UamVLA CALVIN action diagnostics`

Purpose: expose server/client action evidence without changing rollout
semantics.

Debug env var:

```bash
UAMVLA_DEBUG_ACTIONS=1
```

Signals collected:

- Server loaded `class=UamVLA`, `framework=UamVLA`, `action_horizon=8`,
  `action_dim=7`.
- Every parsed server query had `expected_act_tokens=56`, `act_counts=[56]`.
- Client normalized/raw action traces showed nontrivial action ranges and gripper
  binarization.

Verdict: ACT token generation and client chunk consumption are not obviously
broken.

### `20f9bb9 Add CALVIN train-chunk probe`

Purpose: add a read-only probe that compares model predictions against
preprocessed training label chunks on the exact training frames.

Script:

```text
tools/probes/probe_uamvla_calvin_train_chunk.py
```

This was the diagnostic that separated:

- checkpoint/action-head/generation failure: train frames already have high MAE
  or wrong signs;
- rollout/eval-environment mismatch: train frames match labels, but CALVIN env
  rollout still fails.

### Local change: add CALVIN validation expert-frame probe

Purpose: compare the policy server against original CALVIN validation expert
frames, not preprocessed ABC train JSONL frames.

Script:

```text
tools/probes/probe_uamvla_calvin_validation_chunk.py
```

It loads `datasets/task_ABC_D/validation/episode_*.npz`, resets the real CALVIN
env to expert `(robot_obs, scene_obs)`, renders through `CalvinEnvAdapter` at
`256x256`, builds normalized canonical state with the checkpoint statistics,
queries the running policy server, then compares predicted normalized action
chunk to future expert `rel_actions`.

Default behavior only uses full 8-step chunks (`--min-valid-steps 8`) so
terminal one-step windows do not create misleading large MAE spikes.

## Evidence Collected After `802ee0f`

### Action stats are intentionally `[-1, 1]`

Downloaded files:

```text
/Users/tancilon/Downloads/dataset_statistics.json
/Users/tancilon/Downloads/uamvla_calvin_abcd_phase1/config.yaml
/Users/tancilon/Downloads/uamvla_calvin_abcd_phase1/statistics.yaml
/Users/tancilon/Downloads/task_ABC_D/statistics.yaml
```

Findings:

- `dataset_statistics.json["franka_calvin"]["action"]["min/max"]` are all
  `[-1, 1]`.
- `mask == [true, true, true, true, true, true, false]`.
- checkpoint `statistics.yaml` and dataset `task_ABC_D/statistics.yaml` are
  identical.
- `config.yaml` points to `calvin_abc_d_uamvla`, `datasets/uamvla_calvin`,
  `uamvla_dataset`, `future_action_window_size=7`, `action_dim=7`.

Updated verdict: P2 action denormalization is verified consistent for this
pipeline. The original “xyz should be around `[-0.02, 0.02]`” assumption does
not match the preprocessed UAM CALVIN action convention.

### Training action summary

Downloaded file:

```text
/Users/tancilon/Downloads/uamvla_calvin_action_summary.json
```

Overall:

```text
n = 1,071,743
min = [-1, -1, -1, -1, -1, -1, -1]
max = [ 1,  1,  1,  1,  1,  1,  1]
mean ~= [0.001, 0.010, -0.006, -0.002, -0.000, -0.005, -0.084]
std  ~= [0.250, 0.205,  0.213,  0.159,  0.175,  0.355,  0.996]
```

Eval debug log aggregate was broadly same scale, not a catastrophic scaling
bug:

```text
eval first-action std / train std ~= [0.887, 0.571, 0.796, 0.846, 0.826, 0.471, 0.977]
```

But sequence 0 starts with `rotate_blue_block_right`, and for training
instruction `"take the blue block and rotate it right"` the yaw-like sixth
action dimension has mean about `-0.306`. During the failed eval rollout, the
model's first-sequence yaw mean was near zero:

```text
seq0 eval chunk-mean yaw ~= -0.0096
train rotate-blue-right yaw ~= -0.306
```

This is currently the sharpest behavioral discrepancy.

### Eval language mismatch is real but not sufficient

Downloaded files:

```text
/Users/tancilon/Downloads/new_playtable_validation.yaml
/Users/tancilon/Downloads/new_playtable_tasks.yaml
```

Findings:

- Official eval `new_playtable_validation.yaml` has 34 task annotations.
- Exact matches in training summary: `0/34`.
- Example:

```text
eval:  "take the blue block and rotate it to the right"
train: "take the blue block and rotate it right"
```

Seen-language A/B:

- Replaced eval annotation with training-seen equivalent strings.
- Early trend still poor: `0%` through `4/20`.

Updated verdict: language paraphrase shift may hurt, but it is unlikely to be
the single dominant root cause.

### Train-frame probe results

User ran the train-frame probe against the server.

For training instruction:

```text
take the blue block and rotate it right
```

Results:

- With the same training wording and with eval paraphrase
  `"take the blue block and rotate it to the right"`, predictions were identical
  for the tested samples.
- At `start-index 0/200/800`, aggregate MAE was near action-token quantization
  error, roughly `0.002`.
- At `start-index 1600/2400`, the apparent spikes were caused by terminal or
  near-terminal samples with only one valid future action. This motivated
  `--min-valid-steps 8` in the new validation probe.
- The later rotate phase does contain large yaw labels (e.g. around `-0.78`),
  and the model matched them on several ABC train frames. Earlier “yaw near zero
  at the first frame” is therefore not by itself a bug; approach/grasp frames can
  naturally have small yaw.

For training instruction:

```text
turn the blue block right
```

The first four train rows were much weaker:

```text
overall_mae ~= 0.0548
yaw MAE    ~= 0.1643
```

Sign agreement was still mostly correct, especially on yaw. Updated verdict:
ABC train-frame imitation is mostly healthy, but some rotate-language or
trajectory modes are weaker. This does not explain the full closed-loop collapse
by itself.

## Evidence That Is Still Missing

The next useful evidence is no longer generic server token counts or ABC
train-frame imitation; those are mostly known good. We need D validation
expert-frame prediction quality.

Need to inspect:

- Offline action MAE/RMSE on D validation expert frames from original
  `datasets/task_ABC_D/validation`.
- Compare window start (`--frame-offset 0`) vs later manipulation phase
  (`--frame-offset 40`) for rotate tasks.
- If D validation expert-frame MAE is high: likely ABC->D generalization,
  checkpoint quality, or D-domain visual/state distribution issue.
- If D validation expert-frame MAE is low but live rollout still fails: inspect
  closed-loop rollout, action application/control frequency, and state/image
  progression after model actions.

Avoid further behavioral fixes until this probe result is known.

## Suggested Next Diagnostics

### 1. D validation expert-frame probe: rotate-blue start

Run with policy server already up:

```bash
cd /inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UniamVLA

python tools/probes/probe_uamvla_calvin_validation_chunk.py \
  --calvin-root datasets/task_ABC_D \
  --stats-root datasets/uamvla_calvin/task_ABC_D \
  --task rotate_blue_block_right \
  --frame-offset 0 \
  --num-samples 4 \
  --host 127.0.0.1 \
  --port 5694
```

### 2. D validation expert-frame probe: later rotate phase

```bash
python tools/probes/probe_uamvla_calvin_validation_chunk.py \
  --calvin-root datasets/task_ABC_D \
  --stats-root datasets/uamvla_calvin/task_ABC_D \
  --task rotate_blue_block_right \
  --frame-offset 40 \
  --num-samples 4 \
  --host 127.0.0.1 \
  --port 5694
```

Interpretation:

- High MAE on both offsets: checkpoint/D-domain generalization branch.
- Low MAE here but bad live eval: closed-loop rollout/control branch.
- Start good but offset 40 bad: manipulation-phase weakness, especially rotate
  dynamics/language modes.

### 3. ABC train-frame chunk probe: historical baseline

This has already been run, but keep the command for regression checks:

```bash
python tools/probes/probe_uamvla_calvin_train_chunk.py \
  --data-root datasets/uamvla_calvin/task_ABC_D \
  --instruction "take the blue block and rotate it right" \
  --num-samples 4 \
  --host 127.0.0.1 \
  --port 5694
```

Interpretation:

- Low MAE and correct signs are expected for many ABC train-frame samples.
- Large errors here would indicate regression in action-head/generation path.

### 4. ABC train-frame chunk probe: eval paraphrase

Same image/state/label sample, but send official eval wording:

```bash
python tools/probes/probe_uamvla_calvin_train_chunk.py \
  --data-root datasets/uamvla_calvin/task_ABC_D \
  --instruction "take the blue block and rotate it right" \
  --query-lang "take the blue block and rotate it to the right" \
  --num-samples 4 \
  --host 127.0.0.1 \
  --port 5694
```

Historical result: for the tested rotate-blue samples, training wording and eval
paraphrase produced identical predictions, so this is no longer the main branch.

### 5. Optional: repeat for easier successful task

Use an easier task that often succeeds in rollout, e.g. drawer/slider:

```bash
python tools/probes/probe_uamvla_calvin_train_chunk.py \
  --data-root datasets/uamvla_calvin/task_ABC_D \
  --instruction "pull the handle of the drawer" \
  --num-samples 4 \
  --host 127.0.0.1 \
  --port 5694
```

This gives a useful contrast against rotate failures.

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
