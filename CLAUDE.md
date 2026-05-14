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
