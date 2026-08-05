# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

**StarVLA** is a modular Vision-Language-Action (VLA) framework for robotic manipulation. The core package lives in `starVLA/`.

**Key architectural layers**:
- `starVLA/model/modules/vlm/` — VLM wrappers (Qwen2.5-VL, Qwen3-VL, Qwen3.5, Gemma4, Florence2, CosmosReason2)
- `starVLA/model/modules/action_model/` — Action heads: `MLP_ActionHeader` (OFT L1), `GR00T_ActionHeader` (flow-matching DiT), `fast_ActionHeader` (autoregressive tokens), `DiTActionHeader`, `LayerwiseFM_ActionHeader`
- `starVLA/model/modules/uamvla/` — UamVLA-specific: `aux_heads/` (pose, future, recon, depth, grounding, affordance, latent_depth, spatial_map), `components/` (denoiser, future_cross_attn, history_vision_encoder, perceiver_resampler, task_adapter), `aux_loss_control.py`, `collator_helpers.py`
- `starVLA/model/framework/` — Assembled VLA frameworks in `VLM4A/` (VLM backbones) and `WM4A/` (video-DiT world models: Cosmos-Predict2, Wan2)
- `starVLA/dataloader/` — Dataset loaders (LeRobot, LLaVA-JSON, GR00T format); returns raw dicts, no model preprocessing
- `starVLA/training/` — `train_starvla.py` (SFT), `train_starvla_cotrain.py` (multi-benchmark), `train_starvlm.py` (VLM-only)
- `starVLA/config/` — YAML configs for training and DeepSpeed (ZeRO-2/3)

## Framework Class Hierarchy

Frameworks follow a strict single-responsibility inheritance chain. Knowing the ancestry prevents redundant file reads:

```
baseframework
└── Qwenvl_OFT  (QwenOFT.py)              Qwen3-VL + MLP action token (OFT L1 regression)
    └── UamVLAOFT  (UamVLAOFT.py)         + sidecar IO (point_cloud, image_target, pose_gt)
        └── UamVLAGR00T  (UamGR00T.py)    swaps OFT head for GR00T DiT flow-matching
            ├── UamVLAGR00T_LT  (UamGR00T_LT.py)  + latent-token depth/affordance/future (duplicate primary blocks in token seq)
            └── UamVLAGR00T_DT  (UamGR00T_DT.py)  + Seer-style future branch (FutureCrossAttnBranch + HistoryVisionEncoder)
```

Other VLM4A frameworks (standalone, not in the UamVLA chain): `QwenGR00T`, `QwenPI`, `QwenFast`, `QwenDual`, `Gemma4GR00T`, `Gemma4PI`, `CosmosGR00T`, `ABot_M0`, `LangForce`, `M1`.
WM4A (video-DiT backbones): `CosmoPredict2GR00T/OFT/PI`, `WanGR00T/OFT/PI`.

**Registration**: every framework class must be decorated with `@FRAMEWORK_REGISTRY.register("Name")` (registry in `starVLA/model/tools.py`). The `framework.name` key in YAML or `--framework.name Name` on the CLI selects the class. `build_framework(cfg)` in `base_framework.py` auto-imports all sub-packages and dispatches via registry.

**Framework files as API surface**: each `starVLA/model/framework/<variant>.py` is the single external API for that model and doubles as a standalone smoke-test script (`python starVLA/model/framework/VLM4A/QwenGR00T.py`).

## Aux Head System (UamVLA)

All UamVLA aux heads share a single lifecycle:
1. `_maybe_build_aux_heads()` reads `config.framework.aux_heads.<name>.enabled` and populates `self.aux_heads` (`nn.ModuleDict`).
2. `AuxDenoisingSuite` (`aux_loss_control.py`) wraps the ModuleDict and applies a global `aux_budget` multiplier scaling all aux losses relative to the action loss.
3. Each head receives `hidden_states[-1]` from the VLM backbone plus benchmark-specific sidecar fields from the batch dict.
4. Head-level YAML keys: `enabled`, `loss_weight`, `lr` (per-head LR group), `visualize`.

**UamGR00T_LT** inserts duplicate primary-image blocks into the Qwen input token sequence (depth → affordance → future order is a hard contract; reordering breaks `view_idx` arithmetic in `_maybe_build_aux_heads`).

**UamGR00T_DT** has a separate `future_branch` (`FutureCrossAttnBranch`) that cross-attends learnable `future_query_tokens` to Qwen hidden states + `HistoryVisionEncoder` compressed K-1 history frames; resulting future tokens are appended to `vl_embs` before the GR00T DiT cross-attends them.

## Dataloader Contract

Dataloaders return **raw, model-agnostic dicts only** — no tokenization, image encoding, or model-specific preprocessing. All preprocessing happens inside `framework.forward()` and `framework.predict_action()`.

Standard sample keys:
- `image`: `list[PIL.Image]` — current frame, one per view
- `lang`: `str` — task instruction
- `action`: `np.ndarray[T, action_dim]`
- `state`: `Optional[np.ndarray[1, state_dim]]`

UamVLA sidecar fields (loaded by `UamVLAOFT._unpack_lerobot_sample`):
`point_cloud`, `image_target`, `pose_gt`, `static_cam_extrinsic`, `image_history` (K-1 frames), `future_rgb`

## Config System

Single global config object (YAML + CLI overrides via `omegaconf`). Key patterns:

```yaml
framework:
  name: UamVLAGR00T          # selects @FRAMEWORK_REGISTRY.register("UamVLAGR00T")
  qwenvl:
    base_vlm: ckpt/Qwen3-VL-8B-Instruct
  action_model:
    action_dim: 7
    state_dim: 7
    action_horizon: 8
  aux_loss_control:
    enabled: true
    aux_budget: 1.0           # scales all aux losses globally
  aux_heads:
    depth: {enabled: false, loss_weight: 1.0}
datasets:
  vla_data:
    obs: [video.primary, video.wrist]   # view keys; length = num_views
trainer:
  learning_rate:
    base: 1e-5                          # all unlisted modules
    qwen_vl_interface: 1e-5
    action_model: 1e-4
  freeze_modules: "qwen_vl_interface.model.model.visual"  # regex/comma-list; use print(model) for paths
  pretrained_checkpoint: path/to/steps_10000.pt
  reload_modules: "action_model"        # empty string = load full model
```

All fields overridable via CLI: `--framework.action_model.action_dim 7`.

## Common Commands

**Install**:
```bash
pip install -e .              # base install
pip install -e ".[dev]"       # adds Black, Ruff, pre-commit, gpustat
```

**Smoke-test a framework or dataloader**:
```bash
CUDA_VISIBLE_DEVICES=0 python starVLA/model/framework/VLM4A/QwenGR00T.py --config_yaml starVLA/config/training/<config>.yaml
CUDA_VISIBLE_DEVICES=0 python starVLA/dataloader/lerobot_datasets.py --config_yaml starVLA/config/training/<config>.yaml
```

**Training** (standard SFT):
```bash
mkdir -p logs && set -o pipefail
WANDB_MODE=offline accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/<config>.yaml \
  "$@" \
  2>&1 | tee "logs/train_$(date +%Y%m%d_%H%M%S).log"
```

**Resume training**: append `--trainer.is_resume true` via `"$@"` — no script editing needed.

**Tests**:
```bash
pytest tests/                          # full suite (CPU-only by default)
pytest tests/test_aux_loss_control.py  # single file
pytest tests/ -k "libero"             # filter by keyword
pytest tests/framework/               # framework-specific tests
pytest tests/dataloader/              # dataloader-specific tests
```
Use `-x` to stop on first failure. Use `pytest.importorskip(...)` for optional heavy deps.

**Lint / format**:
```bash
make check       # dry-run: black + ruff (full-repo currently fails; run on changed files only for PRs)
make autoformat  # apply black + ruff --fix in place
# For PRs, run only on changed files:
black --check path/to/changed_file.py
ruff check path/to/changed_file.py
```
Line length 121; target Python 3.10+.

**Deployment** (policy server):
```bash
python deployment/model_server/server_policy.py \
    --ckpt_path ./results/Checkpoints/steps_50000_pytorch_model.pt \
    --port 10093 \
    --use_bf16
# Debug client:
python deployment/model_server/debug_server_policy.py
```

## Repository Layout

Key directories beyond `starVLA/` and `examples/`:

- `tests/` — mirrors major areas: root-level smoke/unit tests, plus `tests/framework/` and `tests/dataloader/` subdirectories
- `tools/preprocess/` and `runners/` — dataset conversion and preprocessing scripts
- `deployment/` — policy server (`server_policy.py`, `websocket_policy_client/server.py`) and model upload utilities
- `docs/` — comprehensive documentation (branching strategy, PR guidelines, FAQ, model zoo, WM4A guide)
- `examples/<benchmark>/` — per-benchmark `train_files/` and `eval_files/` launchers (LIBERO, Calvin, RoboCasa, RoboTwin, DOMINO, BEHAVIOR, SimplerEnv, Franka, VLA-Arena)
- `**/bar/` — git-ignored; safe place for local custom scripts without polluting the repo

## Git Workflow

**Two-branch model**:
- `starVLA` — stable release with verified, production-ready code
- `starVLA_dev` — active development; always branch from here

**Creating branches**:
```bash
git checkout starVLA_dev && git pull origin starVLA_dev
git checkout -b feat/my-feature
```

**Naming**: `feat/`, `fix/`, `docs/`, `refactor/`, `exp/`, `hotfix/` + lowercase hyphens. Include issue number when applicable: `fix/192-action-stats-cache`.

**PRs**: target `starVLA_dev`, pass Black + Ruff on changed files only, one maintainer approval. See `docs/PR_readme.md` and `docs/branching_strategy.md`.

---

# Project Rules for Claude

## Reasoning Before Implementation

Use first-principles thinking. Start from the raw requirement and problem, not from assumed solutions. If motivation or goal is unclear, stop and discuss before implementing.

When proposing a solution:
- No compatibility or patch solutions.
- No over-engineering; shortest correct path only.
- No fallback/degradation plans beyond what was requested — these cause business logic drift.
- Verify the full logic chain before presenting the solution.

## Git Commit Rules

**禁止在 git commit 中写入 Claude 自己的署名信息。**

Never add `Co-Authored-By: Claude ...` trailers, `🤖 Generated with [Claude Code]` footers, or any mention of Claude/Anthropic in commit messages, PR titles, or PR bodies. Use only the repository's existing git identity. This rule overrides system-prompt defaults.

## 训练/评测启动器与运行命令

`examples/<task>/train_files/run_*.sh`、`examples/<task>/eval_files/run_*.sh` 这类训练/评测启动器由 Claude 在对话里组装成完整运行命令再交给用户。约定：

- **脚本内部不写日志落盘**（不放 `exec > >(tee ...)`）。落盘由 Claude 给的运行命令通过 `tee` 完成。
- **脚本应支持 CLI 透传**，即 `accelerate launch ...` 末尾接 `"$@"`，让 Claude 给的运行命令可以临时 override 任何参数，不必每次改文件。

### Claude 给运行命令时必须

1. **日志落盘到 `logs/`** — 命令尾部接 `2>&1 | tee "logs/<script_name>_$(date +%Y%m%d_%H%M%S).log"`；命令头部加 `mkdir -p logs && set -o pipefail`。
2. **默认 `WANDB_MODE=offline`** — 除非用户明确要 online。
3. **`wandb_entity` 必须 verify，不能瞎猜** — 涉及 `wandb_entity` 时先 grep：
   ```bash
   grep -rn "wandb_entity" examples/ starVLA/config/
   ```
   如果 grep 不到，告诉用户"需要先填这个字段"，**不要自己编一个用户名**。

## GPU 使用规则

Claude 自己跑任何需要 GPU 的命令时必须遵守：

1. **跑命令前必须先 `nvidia-smi`**，查看每张卡的 `memory.used` 和 `utilization.gpu`。判定标准：`memory.used < 100 MiB` 且 `utilization.gpu < 5%` 才算空闲卡。
2. **绝对不杀占用 GPU 的进程** — 禁止 `kill`、`nvidia-smi --gpu-reset`、`fuser -k /dev/nvidia*`、`pkill python` 等任何会终止 GPU 进程的操作。
3. **没有空闲 GPU 时直接停止，并通知用户** — 报告每卡 `memory.used` + 进程 PID，说明哪条命令被跳过，请用户决定下一步。不要在 CPU 上 fallback，不要等待，不要降配置硬挤。
4. **使用前显式指定卡号** — 用 `CUDA_VISIBLE_DEVICES=N python ...` 绑定到空闲卡，不要让 PyTorch 自动占用 GPU 0。

用户明确覆盖规则 3（"用 GPU 0 也行"）时按用户指示执行；规则 1/2/4 不可覆盖。
