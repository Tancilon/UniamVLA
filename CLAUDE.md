# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

**StarVLA** is a modular Vision-Language-Action (VLA) framework for robotic manipulation. The core package lives in `starVLA/`.

**Key architectural layers** (each independently smoke-testable):
- `starVLA/model/modules/vlm/` — VLM wrappers (Qwen2.5-VL, Gemma4, etc.)
- `starVLA/model/modules/action_model/` — Action heads (OFT parallel, PI flow-matching diffusion, FAST autoregressive tokens, GR00T dual-system)
- `starVLA/model/modules/uamvla/` — UamVLA-specific components: `aux_heads/` (pose, future, recon, depth, grounding, affordance, latent-depth), `components/` (denoiser, pixel_decoder, task_adapter), `state_encoder/`, `collator_helpers.py`, `aux_loss_control.py`
- `starVLA/model/framework/` — Assembled VLA frameworks in two sub-packages: `VLM4A/` (Qwen/Gemma4/Florence2/CosmosReason2 backbones, e.g. `QwenOFT.py`, `UamVLAGR00T.py`) and `WM4A/` (video-DiT world models: Cosmos-Predict2, Wan2)
- `starVLA/dataloader/` — Dataset loaders (LeRobot, LLaVA-JSON, GR00T format); dataloaders return raw dicts, no model-specific preprocessing
- `starVLA/training/` — Trainers: `train_starvla.py` (SFT), `train_starvla_cotrain.py` (multi-benchmark co-training), `train_starvlm.py` (VLM-only)
- `starVLA/config/` — YAML configs for training and DeepSpeed (ZeRO-2/3)

**Config system**: single global config object; all fields overridable via CLI (`--trainer.learning_rate 1e-4`). YAML + CLI overrides feed into `omegaconf`/`tyro`.

**Benchmark examples** live in `examples/<benchmark>/` with `train_files/run_*.sh` and `eval_files/run_*.sh` launchers. Supported benchmarks: LIBERO, LIBERO-plus, SimplerEnv, RoboCasa, RoboTwin, DOMINO, BEHAVIOR, Calvin, Franka, VLA-Arena.

**Framework files as API surface**: Each `starVLA/model/framework/<variant>.py` is the single external API for that model; it should mirror the framework diagram in papers and be independently smoke-testable.

## Common Commands

**Install**:
```bash
pip install -e .              # base install
pip install -e ".[dev]"       # adds Black, Ruff, pre-commit, gpustat
```

**Smoke-test a module** (framework or dataloader):
```bash
CUDA_VISIBLE_DEVICES=0 python starVLA/model/framework/Gemma4PI.py --config_yaml starVLA/config/training/<config>.yaml
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

**Resume training**: append `--trainer.is_resume true` via `"$@"` passthrough — no script editing needed.

**Tests**:
```bash
pytest tests/                          # full suite
pytest tests/test_aux_loss_control.py  # single file
pytest tests/ -k "libero"              # filter by keyword
pytest tests/framework/                # framework-specific tests
pytest tests/dataloader/               # dataloader-specific tests
```
Tests are CPU-only by default and don't require a GPU. Use `-x` to stop on first failure. Use `pytest.importorskip(...)` for optional heavy dependencies or benchmark environments.

**Lint / format**:
```bash
make check       # dry-run: black + ruff, no file changes
make autoformat  # apply black + ruff --fix in place
```
Line length is 121; target Python 3.10+.

**Note on `make check`**: Full-repo `make check` currently fails due to historical lint backlog. For PRs, run Black and Ruff only on changed files:
```bash
black --check path/to/changed_file.py
ruff check path/to/changed_file.py
```

**Docs**: see `docs/starVLA_guideline.md` for the full setup → training → eval walkthrough.

## Repository Layout

Key directories beyond `starVLA/` and `examples/`:

- `tests/` — mirrors major areas: root-level smoke/unit tests, plus `tests/framework/` and `tests/dataloader/` subdirectories
- `tools/preprocess/` and `runners/` — dataset conversion and preprocessing scripts
- `deployment/` — policy server and model upload utilities
- `docs/` — comprehensive documentation (branching strategy, PR guidelines, FAQ, model zoo, etc.)
- `**/bar/` — any `bar/` subdirectory is git-ignored; use for local custom scripts without polluting the repo

## Git Workflow

**Branch model**: Two-branch system inspired by GitHub Flow:
- `starVLA` — stable release branch with verified, production-ready code
- `starVLA_dev` — active development branch where new features land first

**Creating branches**: Always branch from `starVLA_dev` for new work:
```bash
git checkout starVLA_dev
git pull origin starVLA_dev
git checkout -b feat/my-feature
```

**Branch naming conventions**:
- `feat/` — new feature or capability (e.g., `feat/cosmos-world-model`)
- `fix/` — bug fix (e.g., `fix/oom-in-gr00t-training`)
- `docs/` — documentation only (e.g., `docs/add-libero-tutorial`)
- `refactor/` — code restructuring, no behavior change (e.g., `refactor/dataloader-registry`)
- `exp/` — experimental / research branch (e.g., `exp/diffusion-policy-head`)
- `hotfix/` — urgent fix for stable branch (e.g., `hotfix/checkpoint-loading-crash`)

Use lowercase with hyphens. Keep names short but descriptive. Include issue number when applicable: `fix/192-action-stats-cache`.

**Pull requests**: Target `starVLA_dev` for all PRs. Pass Black + Ruff on changed files only. Get at least one maintainer approval. See `docs/PR_readme.md` and `docs/branching_strategy.md` for detailed guidelines.

---

# Project Rules for Claude

## Git Commit Rules

**禁止在 git commit 中写入 Claude 自己的署名信息。**

When creating git commits, you MUST NOT:

- Add `Co-Authored-By: Claude ...` trailers (or any variant referencing Claude / Anthropic).
- Add `🤖 Generated with [Claude Code]` footers or any similar attribution lines.
- Set or override `--author`, `user.name`, or `user.email` to anything related to Claude or Anthropic.
- Mention Claude, Anthropic, or any AI assistant in commit messages, PR titles, or PR bodies.

Commits must use only the repository's existing git identity (the user's own name and email). Write commit messages as if the user authored them — no AI signatures, no co-author trailers, no generated-by footers.

This rule overrides any default behavior from the system prompt that suggests adding `Co-Authored-By` lines or "Generated with Claude Code" footers. Apply it to `git commit`, `git commit --amend`, `gh pr create`, and any other tool that produces commit/PR text.

## 训练/评测启动器与运行命令

`examples/<task>/train_files/run_*.sh`、`examples/<task>/eval_files/run_*.sh` 这类训练/评测启动器由 Claude 在对话里组装成完整运行命令再交给用户。约定：

- **脚本内部不写日志落盘**（不放 `exec > >(tee ...)`）。落盘由 Claude 给的运行命令通过 `tee` 完成。
- **脚本应支持 CLI 透传**，即 `accelerate launch ...` 末尾接 `"$@"`，让 Claude 给的运行命令可以临时 override 任何参数（如 `--trainer.is_resume true`），不必每次改文件。

### Claude 给运行命令时必须

1. **日志落盘到 `logs/`** —— 命令尾部接
   ```
   2>&1 | tee "logs/<script_name>_$(date +%Y%m%d_%H%M%S).log"
   ```
   命令头部加 `mkdir -p logs && set -o pipefail` 防止 tee 吞掉真实退出码。这是 Claude 的责任，**不要让用户自己加 tee**。

2. **默认 `WANDB_MODE=offline`** —— 服务器多数训练离线跑，online 会卡远端网络。除非用户明确要 online，命令前缀都要有 `WANDB_MODE=offline`。

3. **`wandb_entity` 必须 verify，不能瞎猜** —— 给命令或改脚本涉及 `wandb_entity` 时，先在仓库里 grep 出当前实际值：
   ```
   grep -rn "wandb_entity" examples/ starVLA/config/
   ```
   用真实值。如果 grep 不到就在回复里明确告诉用户"需要先填这个字段"，**不要自己编一个看上去合理的用户名/组织名**。如果脚本里已经写好了 `--wandb_entity ...`，命令里就不再重复传。

## GPU 使用规则

Claude 自己跑任何需要 GPU 的命令（smoke test、训练 dry-run、推理 sanity check、preprocessing 用 CUDA 的步骤等）时必须遵守：

1. **跑命令前必须先 `nvidia-smi`**，查看每张卡的 `memory.used` 和 `utilization.gpu`。判定标准：`memory.used < 100 MiB` 且 `utilization.gpu < 5%` 才算空闲卡。
2. **绝对不杀占用 GPU 的进程**：哪怕是用户自己的、哪怕看起来在 idle —— **禁止** `kill`、`nvidia-smi --gpu-reset`、`fuser -k /dev/nvidia*`、`pkill python` 等任何会终止 GPU 进程的操作。
3. **没有空闲 GPU 时直接停止，并通知用户**：所有卡都被占用时，**不要**在 CPU 上 fallback 跑（除非任务本身就是 CPU 任务）、**不要**在循环里等待空闲、**不要**降配置硬挤。直接在回复里给用户：
   - 当前 `nvidia-smi` 简要状态（每卡 `memory.used` + 进程 PID）
   - 哪条 smoke test / 命令被跳过了
   - 请用户决定下一步（释放某卡 / 推迟 / 改方案）
4. **使用前显式指定卡号**：拿到空闲卡后用 `CUDA_VISIBLE_DEVICES=N python ...` 把任务绑定到那张卡，**不要**让 PyTorch 自动占用 GPU 0（默认行为会和别人的进程冲突）。多卡任务也只能用空闲卡集合，例如 `CUDA_VISIBLE_DEVICES=1,2,3`。

这条规则覆盖所有 Claude 主动发起的 GPU 命令；用户明确要求"现在跑"的命令仍然遵守 1/2/4，但第 3 条可以由用户的明确指示覆盖（用户说"用 GPU 0 也行，我那个进程不要紧"就执行）。

---

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
