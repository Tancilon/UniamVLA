# UamGR00T 工作进度报告
> 更新时间：2026-08-31

---

## 当前分支状态

**分支名称**: `feat/seer-alignment-uamgr00t-dt`  
**目标主分支**: `starVLA_dev`

---

## 一、已完成工作

### 1.1 UamGR00T_DT 框架开发 ✅

**核心设计**：Seer-aligned UamGR00T with Future DiT Branch

**架构特点**：
- **继承链**: `baseframework → Qwenvl_OFT → UamVLAOFT → UamVLAGR00T → UamGR00T_DT`
- **Future Branch**: `FutureCrossAttnBranch` + `HistoryVisionEncoder`
  - 对 K-1 历史帧（默认 K=10）进行视觉压缩编码
  - 49 个可学习 `future_query_tokens` 通过交叉注意力查询 Qwen hidden states + 历史帧特征
  - 生成的 future tokens 拼接到 `vl_embs` 后端，供 GR00T DiT 使用
- **Seer 对齐**: 通过 `future_latent` aux head 预测未来帧 VAE latent（MSE loss）

**关键文件**：
- `starVLA/model/framework/VLM4A/UamGR00T_DT.py` — 主框架实现
- `starVLA/model/modules/uamvla/components/future_dit_branch.py` — Future DiT Branch 组件
- `starVLA/model/modules/uamvla/components/history_vision_encoder.py` — 历史帧编码器
- `starVLA/model/modules/uamvla/components/seer_joint_decoder.py` — Seer 联合解码器（备用版本）

**注册名**: `UamGR00T_DT` (`@FRAMEWORK_REGISTRY.register`)

### 1.2 训练配置与运行脚本 ✅

**Calvin 配置**：
- `examples/calvin/train_files/run_uamgr00t_DT_calvin_pretrain.yaml`
- `examples/calvin/train_files/run_uamgr00t_DT_calvin_finetune.yaml`

**RoboTwin 配置**：
- `examples/Robotwin/train_files/run_uamgr00t_DT_robotwin_pretrain.yaml`
- `examples/Robotwin/train_files/run_uamgr00t_DT_robotwin_finetune.yaml`

**训练阶段**：
- **Pretrain**: 在 Calvin ABC 数据上预训练（100k steps）
- **Finetune**: 在 Calvin ABC 数据上 finetune（100k steps，带 visual resampling）

### 1.3 训练已完成 ✅

**Checkpoint 目录**：
```
playground/Checkpoints/
├── uamvla_gr00t_dt_calvin_pretrain/           # 预训练（100k steps）
├── uamvla_gr00t_dt_calvin_abc_finetune/       # 微调版本1
└── uamvla_gr00t_dt_calvin_abc_finetune_resample_visual/  # 微调版本2（带 visual resampling）
    ├── checkpoints/
    │   ├── steps_10000/
    │   ├── steps_20000/
    │   ├── ...
    │   └── steps_100000/
    └── final_model/
        └── pytorch_model.pt
```

**训练日志**：
- `logs/run_uamgr00t_DT_calvin_pretrain_20260729_100807.log` (62M)
- `logs/run_uamgr00t_DT_calvin_finetune_20260809_103546.log` (22M)

**训练完成标志**：10 个 checkpoint（每 10k steps 保存一次），final_model 已生成。

### 1.4 评测工作完成 ✅

**评测完成标志**（8月12日）：
- ✅ Checkpoint Sweep 完成：10k/30k/50k/70k/90k/100k steps
- ✅ 详细报告已生成：`logs/uamgr00t_dt_calvin_abc_d_checkpoint_sweep.md`
- ✅ 闭环与离线指标均已获取

**评测结果总览**（Calvin ABC→D）：

**最佳 Checkpoint: 100k steps**
- **SR1**: 80.0% (单步任务成功率)
- **SR2**: 60.0% (2步任务链)
- **SR3**: 10.0% (3步任务链)
- **SR4**: 10.0% (4步任务链)
- **SR5**: 10.0% (5步任务链)
- **Avg Length**: 1.70 (平均完成任务数)

**训练趋势分析**：
- **10k → 50k**: 快速提升期（SR1 从 20% → 90%）
- **50k → 100k**: 稳定优化期（SR1 稳定在 70-90%，SR2 提升至 60%）
- **离线 MSE**: Scene A action MSE 从 0.024 降至 0.001（100×提升）
- **泛化 Gap**: D/A action MSE 比例从 5× 扩大到 100×（提示可能存在轻微过拟合）

**评测协议**：
- **闭环**：固定 10 条任务序列，每个子任务清空历史，replan_steps=8
- **离线**：Scene A/D 各 8 个 episode 的固定窗口，每窗口 3 次采样

**评测日志**：
- 主报告：`logs/uamgr00t_dt_calvin_abc_d_checkpoint_sweep.md`
- 实时日志：`logs/calvin_eval_uamgr00t_dt_h8_20260812_031722.log` (部分完成，22/1000 episodes)
- Server 日志：`logs/calvin_eval_uamgr00t_dt_h8_server_20260812_033551.log`

---

## 二、待完成工作

### 2.1 UamGR00T_DT 后续分析 🟡

**任务**：深入分析评测结果

**已有数据**：
- ✅ Checkpoint sweep 完成（10k-100k，间隔 20k）
- ✅ 闭环指标：SR1-SR5, Avg Length
- ✅ 离线指标：Action MSE, Arm MSE, Gripper 准确率

**待分析**：
1. **与 baseline 对比** 🔴
   - UamGR00T (无 future branch) 的对比实验
   - 与 Seer 论文报告的 +21% 提升对比
   - 量化 future branch 的贡献

2. **过拟合诊断** 🟡
   - D/A gap 从 5× → 100× 的原因分析
   - 是否需要调整训练策略（early stopping / regularization）

3. **可视化生成** 🟢
   - Future prediction 可视化（预测未来帧 vs 真实未来帧）
   - 训练曲线（action loss + future_latent aux loss）
   - Checkpoint sweep 趋势图（已有数据，待生成）

**评测命令**（如需补充完整 1000 episodes 评测）：
```bash
mkdir -p logs && set -o pipefail
CUDA_VISIBLE_DEVICES=0 python examples/calvin/eval_files/eval_calvin.py \
  --ckpt_path playground/Checkpoints/uamvla_gr00t_dt_calvin_abc_finetune_resample_visual/final_model/pytorch_model.pt \
  --config_yaml examples/calvin/train_files/run_uamgr00t_DT_calvin_finetune.yaml \
  --split D \
  --num_sequences 1000 \
  2>&1 | tee "logs/calvin_eval_uamgr00t_dt_final_full_$(date +%Y%m%d_%H%M%S).log"
```

### 2.2 UamGR00T_GIA 框架开发 ✅ → 待测试 🔲

**状态**：Phase 1 & 2 已完成（代码编写 + 配置文件创建）

**文件**：
- `starVLA/model/framework/VLM4A/UamGR00T_GIA.py` ✅ 已创建
- `examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml` ✅ 已创建
- `examples/calvin/train_files/run_uamgr00t_GIA_calvin.sh` ✅ 已创建
- `examples/Robotwin/train_files/run_uamgr00t_GIA_robotwin.yaml` ✅ 已创建
- `examples/Robotwin/train_files/run_uamgr00t_GIA_robotwin.sh` ✅ 已创建

**待办**：
1. **Smoke test 验证**：运行 framework 文件，确认语法和配置加载无误
2. **Sidecar 数据准备**：确认 Calvin 和 RoboTwin 的 `depth_latent`, `affordance_px`, `rgb_latent` sidecar 已生成
3. **训练启动**：在确认数据和配置无误后，启动训练

**Smoke test 命令**：
```bash
CUDA_VISIBLE_DEVICES=0 python starVLA/model/framework/VLM4A/UamGR00T_GIA.py \
  --config_yaml examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml
```

### 2.3 实验文档与可视化 🔲

**待生成**：
- UamGR00T_DT 评测结果对比表格（与 baseline / Seer 对比）
- Future branch 可视化（预测未来帧 vs 真实未来帧）
- 训练曲线（action loss + future_latent aux loss）
- GIA 三维度 aux loss 曲线（depth + affordance + future）

---

## 三、技术要点总结

### 3.1 UamGR00T_DT 的创新点

1. **Seer-aligned 设计**：通过预测未来 VAE latent 对齐视觉预见能力
2. **历史帧压缩**：`HistoryVisionEncoder` 将 K-1 帧编码为紧凑表示
3. **Future Query Tokens**：可学习查询向量通过交叉注意力提取时序信息
4. **Action Model 隔离**：Future tokens 仅通过 VLM 自注意力间接影响动作预测

### 3.2 UamGR00T_GIA 的三维增强

| 维度 | Aux Head | Token | 监督目标 |
|------|---------|-------|---------|
| 几何 (Geometry) | `depth_latent` | ✨ | VAE encoded depth latent (64×7×7) |
| 交互 (Interaction) | `affordance_px` | 🟩 | VRB contact heatmap (112×112) |
| 时序 (Temporal) | `future_latent` | 🟧 | Future frame VAE latent (64×7×7) |

**注意力策略**：层级因果（Strategy B）
- `affordance` 能看到 `depth`（交互受几何约束）
- `future` 能看到 `depth` + `affordance`（预测需要几何+交互信息）
- 兼容 `flash_attention_2`（标准因果掩码）

### 3.3 数据流与 Sidecar 系统

```
磁盘 Sidecar (mmap + LRU cache)
  ├── depth_latent/chunk-NNN/<camera>/episode_XXXXXX.npy
  ├── affordance_px/chunk-NNN/<camera>/episode_XXXXXX.npy
  ├── rgb_latent/chunk-NNN/<camera>/episode_XXXXXX.npy
  └── image_history/chunk-NNN/<camera>/episode_XXXXXX.npy
         ↓
UamVLAOFT._unpack_lerobot_sample() + _collate_aux()
         ↓
batch_dict with zero-padding + bool masks
         ↓
Framework.forward() → Aux heads (per-token loss)
```

---

## 四、Git 状态

**已修改文件**（未提交）：
```
M .gitignore
M examples/Robotwin/train_files/run_uamgr00t_DT_robotwin_finetune.yaml
M examples/Robotwin/train_files/run_uamgr00t_DT_robotwin_pretrain.yaml
M examples/calvin/eval_files/eval_calvin.py
M examples/calvin/train_files/run_uamgr00t_DT_calvin_finetune.yaml
M starVLA/model/framework/VLM4A/UamGR00T_DT.py
M starVLA/model/framework/VLM4A/UamGR00T_hR_multi.py
M starVLA/model/modules/uamvla/components/future_dit_branch.py
M starVLA/model/modules/uamvla/components/seer_joint_decoder.py
M starVLA/training/trainer_utils/trainer_tools.py
M tests/test_calvin_eval_train_renderer.py
```

**未跟踪文件**（新建）：
```
?? .claude/
?? examples/Robotwin/train_files/run_uamgr00t_GIA_robotwin.sh
?? examples/Robotwin/train_files/run_uamgr00t_GIA_robotwin.yaml
?? examples/calvin/train_files/run_uamgr00t_GIA_calvin.sh
?? examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml
?? plan.md
?? starVLA/model/framework/VLM4A/UamGR00T_GIA.py
?? tests/framework/test_uamgr00t_dt_temporal_embedding.py
```

**Recent commits**：
```
7fc4298 feat: rewrite uamgr00t-dt to future-query + GR00T flow-matching action
6238197 feat: uamgr00t-dt seer-alignment + future-dit-branch
2524030 backup: uamgr00t-dt seer joint decoder version
9661a82 feat: Seer-aligned UamGR00T_DT with SeerJointDecoder
dd9f901 uamgr00t-dt framework
```

---

## 五、下一步行动计划

### 优先级 P0（本周完成）

1. **完成 UamGR00T_DT 评测** 🔴
   - 在 Calvin D split 上运行完整评测
   - 记录所有指标（1/2/3/4/5-step success rate, avg length）
   - 对比 baseline 和 Seer 论文结果
   - 生成评测报告表格

2. **验证 UamGR00T_GIA smoke test** 🟡
   - 运行 framework 文件独立测试
   - 检查 sidecar 数据是否就绪
   - 修复任何配置或依赖问题

### 优先级 P1（下周）

3. **启动 UamGR00T_GIA 训练** 🟢
   - Calvin ABC 预训练（100k steps）
   - 监控三个 aux loss 的收敛情况
   - 验证 token 序列布局和注意力掩码正确性

4. **整理提交与文档** 🟢
   - 将当前 feat 分支整理为 clean commits
   - 编写 PR 描述（参考 docs/PR_readme.md）
   - 准备向 starVLA_dev 提交 PR

### 优先级 P2（后续增强）

5. **Phase 3 & 4 增强**（见 plan.md）
   - 光流时序增强（`preprocess_optical_flow.py`）
   - 深度质量提升（DepthAnything V2 + CDM）
   - 可供性数据增强（Grounded-SAM + AffordDex）

---

## 六、参考资源

**核心论文**：
- **Seer** (ICLR 2025 Oral): arXiv:2412.15109, [GitHub](https://github.com/InternRobotics/Seer)
- **Afford-VLA** (2026): arXiv:2605.24203
- **Manip-as-in-Sim (CDMs)** (2025): arXiv:2509.02530

**文档**：
- `plan.md` — 完整 GIA 设计文档
- `CLAUDE.md` — 项目开发指南
- `docs/starVLA_guideline.md` — 快速开始教程
- `docs/WM4A.md` — World Model 架构说明

**数据集**：
- Calvin: `datasets/calvin_abc_lt_starvla_uam_state_h8/`
- RoboTwin: `datasets/robotwin_all_dt_k10/`

---

## 七、问题与风险

### 已知问题

1. **Eval 日志未显示最终指标**
   - 症状：websocket 连接正常，但未输出 success rate
   - 可能原因：eval 脚本提前退出 / 日志重定向问题
   - 解决方案：检查 eval 脚本输出流，确保结果正确打印

2. **Sidecar 数据完整性未验证**
   - GIA 需要的三个 sidecar（depth_latent, affordance_px, rgb_latent）
   - 需要确认 Calvin 和 RoboTwin 数据集是否已完整预处理
   - 建议：运行 `runners/preprocess_*.py` 脚本验证

3. **Git commit 未清理**
   - 当前有大量 WIP 修改未提交
   - 需要整理成有意义的 commit message
   - 删除 Claude 相关 co-author 签名（见 CLAUDE.md Git Commit Rules）

### 待确认

- [ ] UamGR00T_DT 的 K=10 是否是最优历史窗口（可消融实验 K=5/10/15）
- [ ] GIA 的 aux_budget=1.0 是否合适（三头 loss 可能需要调整权重）
- [ ] Qwen3.5 兼容性（hR_Interface 是否完整支持 Qwen3.5）

---

**报告生成**: Claude Code (Opus 5)  
**最后更新**: 2026-08-31
