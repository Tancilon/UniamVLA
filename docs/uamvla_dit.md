# UamVLA_DiT

The registry name is retained, but the action model is now the CogACT-derived
`DiTActionHeader.ActionModel`, matching hybridVLA's RynnBrainOFT diffusion path.
It is not the GR00T flow-matching head. Old GR00T checkpoints cannot be loaded
into this architecture; start a new run from `ckpt/RynnBrain-CoP-8B`.

## Model and data

RynnBrain encodes two current camera images and H historical pairs. Set
`datasets.vla_data.num_history_frames` (H) and `history_interval` (S) to positive
integers for historical fusion (H may also be zero for the baseline). History offsets are `[-H*S, ..., -S]`, oldest to newest, excluding
the current frame. Defaults H=5, S=5 give `[-25,-20,-15,-10,-5]`; H=3, S=4
gives `[-12,-8,-4]`. Training sampling and evaluation read the same settings. Each camera's history is fused independently into the
three native Qwen3-VL DeepStack maps. The latest historical frame queries all
H historical frames; a learned sigmoid gate blends the result with current
features plus episode-step encoding. History does not extend the text sequence.
Observations remain 224×224 (`obs_image_size` and dataset `image_resize`). The
processor alone resizes them to 256×256 (`qwen_image_size`), producing 64 merged
tokens per view: 16×16 patches with 2×2 spatial merging. Current and historical
images use the same processor. For 224×224 input this matches the local original
RynnBrain processor's grid and pixel tensor; the original hybridVLA experiment's
checkpoint processor has not been independently verified.

The task prompt receives the same suffix as RynnBrainOFT:
` Please predict the next 8 robot actions: <action>🔍🔍🔍🔍🔍🔍🔍🔍<action>.`
The last eight action placeholders are gathered in sequence order from the last
language hidden layer. A learned `Linear(4096,768)` produces `[B,8,768]` conditions.
These text positions do not receive DeepStack image residuals.

The DiT-B action expert has 12 layers, width 768 and 12 heads. Its sequence has
8 condition tokens and 8 action tokens, with a learned positional embedding.
The output is an 8×7 action chunk. There is no robot state encoder or state input.
The future RGB denoiser module remains in the repository for later work but is
not imported or constructed by this framework. There are no future queries,
future labels, auxiliary losses, or future-image visualizations.

The `calvin_abc_dit` mixture uses scenes A/B/C. Video offsets are generated from H and S with a final current-frame offset 0
(default `[-25,-20,-15,-10,-5,0]`); the recipe sets `include_state: false` and
`future_offset: 0`. Early history clamps to frame zero. Action offsets remain
0 through 7, with the existing dataset normalization and boundary padding.
No action validity mask is introduced.

## Training and inference

Training uses 100 Gaussian diffusion steps with `squaredcos_cap_v2`, epsilon
prediction, fixed small variance, and 0.1 condition dropout. Conditions and action
labels are repeated four times with independent noise and sampled timesteps.
The objective is FP32 noise-prediction MSE. `action_loss` is the trainer's total
loss; `action_loss_diffusion` is its detached logging value. There is no separate
language loss.

The recipe uses code defaults for diffusion repetition (4), diffusion steps
(100), noise schedule (`squaredcos_cap_v2`) and CFG scale (1.5). The framework
sets `n_condition_token` from `action_horizon` during initialization; it need
not be listed in the input YAML and may appear in the saved runtime config.

Online inference defaults to 10 DDIM steps, CFG scale 1.5, eta 0 and no clipping
of predicted clean actions, matching hybridVLA's CALVIN client. `predict_action`
accepts `use_ddim`, `num_ddim_steps` and `cfg_scale`, and returns NumPy
`normalized_actions` of shape `[B,8,7]`. Explicit timestep requests rebuild the
DDIM sampler when needed. The trainer's existing action check requests 20 steps.
`use_ddim=False` selects the full ancestral diffusion sampler.

The VLM and history fusion learning rates are 5e-6. The action head uses 5e-5,
and the condition projector uses the base learning rate of 1e-5, matching
hybridVLA's corresponding modules. Training starts fresh, uses BF16 and
ZeRO-2, and saves these modules in the normal strict framework checkpoint.

```bash
conda activate uamvla
python -m pytest tests/framework/test_uamvla_dit.py tests/test_calvin_eval_train_renderer.py -q
python -m tools.smoke_uamvla_dit --backward --offload-activations
NUM_PROCESSES=8 bash examples/calvin/train_files/run_uamvla_DiT_calvin.sh
```

New run names begin with `uamvla_dit_diffusion_calvin_abc_`. Unit tests use a
small randomly initialized Qwen3-VL and the production DiT-B, with the local
RynnBrain tokenizer/processor. They check gradient flow after the zero-initialized
output starts learning, conditioning positions, CFG, sampler switching, BF16,
learning-rate groups, strict state-dict loading and the dataset contract.

## CALVIN ABC→D evaluation

Run from the repository root. Start the policy server in `uamvla` with a checkpoint
from the new architecture:

```bash
CKPT_PATH=/path/to/new-run/checkpoints/steps_10000_pytorch_model.pt \
  GPU_ID=0 bash examples/calvin/eval_files/run_policy_server.sh
```

In a second terminal, activate `calvin_env` and use the same checkpoint:

```bash
CKPT_PATH=/path/to/new-run/checkpoints/steps_10000_pytorch_model.pt \
  bash examples/calvin/eval_files/eval_calvin.sh
```

The evaluation script defaults to scene D under `datasets/calvin/task_ABC_D`,
1000 sequences, `franka` statistics, localhost port 5694 and EGL. It enables the
training renderer: both cameras render at 256×256, then PIL BICUBIC resizes to
224×224 for both current images and history, before the VLM processor resizes
to 256×256. This evaluation rendering path is unchanged by the token-count
change. The actual `_lerobot_lt` training videos contain 200×200 primary and
84×84 wrist images, so re-rendering at 256 is not identical to training input
preparation; online rendering also does not reproduce video compression.
Replanning defaults to the checkpoint horizon of
8 actions; `--args.replan-steps` can explicitly override the interval.
History updates on every environment step and clears at subtask boundaries.
No robot state or future images are sent for this framework.

Evaluation reads H and S from the checkpoint config and keeps H*S prior environment
frames. It samples history before appending the current frame, including on steps
that reuse cached actions, and repeats frame zero when history is insufficient.
Both fields are required; older configs missing history_interval need an explicit
value. History length does not change learned parameter shapes.

## Current-frame baseline

Run `bash examples/calvin/train_files/run_uamvla_DiT_baseline.sh` to use the
separate baseline YAML. It sets `num_history_frames: 0` and omits the
`history_fusion` learning-rate group. `history_interval: 5` is retained but has
no sampling effect when H=0. Logs and checkpoints use the distinct
`uamvla_dit_baseline_calvin_abc_` prefix with a timestamp.

The baseline samples only video offset `[0]`, constructs no history fusion
parameters, and needs no history images or episode step. Current-image native
Qwen3-VL DeepStack features are injected unchanged; historical attention,
gates and episode-time encoding are absent. VLM weights, processor, action
prompt, projector, DiT and training hyperparameters stay the same. Evaluation
reads H=0 from the checkpoint config and does not cache or send history.
Train a separate baseline checkpoint; historical-fusion checkpoints have extra
parameters and cannot be strictly loaded as baseline checkpoints.
