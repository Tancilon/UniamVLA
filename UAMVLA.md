# UAMVLA

> **Universal Auxiliary-enhanced Multimodal Vision-Language-Action**
> 
> 更新时间：2026-08-31

借用 ReconVLA 思想，在 action 预测时，加入 2D/3D 重建、future/depth 预测等改善 VLM 中向量空间状态，借此对齐仿真/真实世界中的任务含义和当前观察到的状态。

为何当前在仿真数据集（如 Calvin、LIBERO）上，小参数量模型和大参数量模型无法拉开差距？可能的原因是，当前的任务在语义指令上无法对齐每一个视觉，需要对齐一个语言指令 token 与不同的视觉 token（即插即用？适配不同的 VLM？）

**Overleaf 论文**: https://www.overleaf.com/project/6a2e2f49b8429b4d280e4be4

---

## 框架演进路线

```
UamVLAOFT (基础)
    ↓
UamVLAGR00T (GR00T DiT action head)
    ↓
    ├── UamGR00T_LT (Latent Token) — 独立 token 块插入主序列
    │   └── depth/affordance/future tokens 作为独立视觉块（duplicate primary blocks）
    │
    ├── UamGR00T_DT (Dual Temporal) — Seer 对齐 + 历史帧编码 ✅ 当前重点
    │   └── FutureDiTBranch + HistoryVisionEncoder
    │       - K=10 历史帧压缩
    │       - 49 个可学习 future_query_tokens
    │       - 交叉注意力融合 Qwen hidden states + 历史信息
    │
    └── UamGR00T_GIA (Geometry·Interaction·Action) — 三维增强 🚧 开发中
        └── hR_multi (hierarchical multi-head) 基础设施
            - ✨ depth_latent (几何)
            - 🟩 affordance_px (交互)
            - 🟧 future_latent (时序)
```

## Plan and idea

* [x] 在 StarVLA 上完成 ReconVLA 的复现（Calvin）
* [x] **UamGR00T_LT**: 独立 token 作为 aux head（depth/affordance/future）
* [x] **UamGR00T_DT**: Seer-aligned 时序预测框架（历史帧编码 + Future DiT Branch）
* [x] 完成 UamGR00T_DT Calvin ABC 预训练（100k steps）
* [x] 完成 UamGR00T_DT Calvin ABC 微调（100k steps with visual resampling）
* [ ] **UamGR00T_DT 评测**：Calvin ABC→D 完整评测（获取 1/2/3/4/5-step success rate）
* [ ] **UamGR00T_GIA**: 三维增强框架（Geometry·Interaction·Action）
  * [x] 框架代码实现
  * [x] 训练配置文件（Calvin + RoboTwin）
  * [ ] Smoke test 验证
  * [ ] Sidecar 数据准备验证
  * [ ] 启动训练
* [ ] 探索不同 aux head 的消融结果（利用可视化）
* [ ] 选择不同的 benchmark 进行结果验证（LIBERO, RoboTwin）
* [ ] 考虑真机验证（数据集需要 RGB + language 外的其他模态信息）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MjFiMGFmNGU2ZTgxMjk3NzYyZDFjNzY3ZDk1MThiNjFfOTYzNmU0YTNiMzVjMDA1ZDc5MDZhZWIxZmQ3NTVhY2JfSUQ6NzY1NDk5NzI0MzE0NTAzMDYxMF8xNzg4MTc5ODcyOjE3ODgyNjYyNzJfVjM)

## Eval Results

### UamGR00T_LT (Latent Token 系列)

> 早期版本，使用独立 token 块作为 aux head

**Calvin D→D** (7/22 实验):
- `uamvla_gr00t_lt_calvin_abc_depth_steps100k_4090.log` (275K)
- `uamvla_gr00t_lt_calvin_abc_depth_future_steps100k_4090.log` (245K)
- `uamvla_gr00t_lt_calvin_abc_depth_afford_future_steps100k_4090.log` (264K)

### UamGR00T_DT (Dual Temporal - Seer-aligned)

> **当前重点**：Seer 风格时序预测，历史帧编码 + Future DiT Branch

**训练状态**：
- ✅ 预训练完成：Calvin ABC, 100k steps (7/29-8/1)
- ✅ 微调完成：Calvin ABC, 100k steps with visual resampling (8/9-8/11)
- ✅ 评测完成：Calvin ABC→D Checkpoint Sweep (8/12)

**模型配置**：
- **Base VLM**: Qwen3-VL-4B-Instruct
- **Action Model**: DiT-B (157M parameters)
- **历史窗口**: K=10 frames
- **Views**: 2 (primary + wrist)
- **Future tokens per view**: 49
- **Action horizon**: 8
- **State dim**: 7

**Checkpoints**：
```
playground/Checkpoints/uamvla_gr00t_dt_calvin_abc_finetune_resample_visual/
├── checkpoints/
│   ├── steps_10000_pytorch_model.pt  (9.7GB)
│   ├── steps_20000_pytorch_model.pt  (9.7GB)
│   ├── ...
│   └── steps_100000_pytorch_model.pt (9.7GB)
└── final_model/
    └── pytorch_model.pt
```

**评测结果**（Calvin ABC→D，2026-08-12）：

**闭环任务链 Success Rate**（固定 10 条序列，每个子任务清空历史）：

| Checkpoint | SR1 | SR2 | SR3 | SR4 | SR5 | Avg Length |
|------------|-----|-----|-----|-----|-----|------------|
| 10k steps  | 20.0% | 10.0% | 0.0% | 0.0% | 0.0% | 0.30 |
| 30k steps  | 40.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.40 |
| 50k steps  | 90.0% | 50.0% | 20.0% | 0.0% | 0.0% | 1.60 |
| 70k steps  | 70.0% | 50.0% | 20.0% | 10.0% | 0.0% | 1.50 |
| 90k steps  | 70.0% | 50.0% | 10.0% | 10.0% | 10.0% | 1.50 |
| **100k steps** | **80.0%** | **60.0%** | **10.0%** | **10.0%** | **10.0%** | **1.70** ✅ |

**离线 Action MSE**（Scene A/D 固定窗口，每窗口 3 次采样）：

| Checkpoint | Scene A Action MSE | Scene D Action MSE | D/A Ratio |
|------------|-------------------|-------------------|-----------|
| 10k steps  | 0.024 ± 0.046 | 0.120 ± 0.159 | 5.0× |
| 30k steps  | 0.008 ± 0.011 | 0.066 ± 0.107 | 8.8× |
| 50k steps  | 0.004 ± 0.001 | 0.064 ± 0.097 | 15.6× |
| 70k steps  | 0.002 ± 0.001 | 0.051 ± 0.059 | 31.7× |
| 90k steps  | 0.001 ± 0.000 | 0.065 ± 0.104 | 64.1× |
| **100k steps** | **0.001 ± 0.000** | **0.089 ± 0.139** | 100.1× |

**关键观察**：
- ✅ **最佳闭环指标**：100k steps，Avg Length = 1.70
- ✅ **SR1 达到 80%**：单步任务成功率显著提升
- ⚠️ **D/A gap 扩大**：训练后期 Scene D (未见环境) action MSE 相对 Scene A 的比例从 5× 扩大到 100×，提示可能存在轻微过拟合
- ✅ **Gripper 准确率**：Scene A 在所有 checkpoint 均达到 99%+，Scene D 在 85-92% 范围

**评测协议**：
- **离线**：Scene A/D 各取 episode 0,1,2,3,4,10,20,50 的中间合法窗口；每个窗口 10 帧历史 + 8-step GT；每窗口采样 3 次
- **闭环**：CALVIN ABC→D，固定 `eval_sequences.json` 前 10 条，`replan_steps=8`，每个子任务清空历史

**详细报告**：`logs/uamgr00t_dt_calvin_abc_d_checkpoint_sweep.md`

**Baseline 对比目标**（参考 Seer 论文 ICLR 2025）：
- Seer 在 Calvin ABC→D: +21% over baseline
- Seer 在真实任务: +43%
- **待补充**：与 UamGR00T (无 future branch) 的直接对比

### UamGR00T_GIA (Geometry·Interaction·Action)

> 🚧 **开发中**：三维感知增强框架

**设计要点**：
- **三个维度**：
  - ✨ Geometry (depth_latent): 几何结构感知
  - 🟩 Interaction (affordance_px): 基于几何的交互区域定位
  - 🟧 Temporal (future_latent): 基于几何+交互的未来状态预测
- **注意力策略**: Strategy B（层级因果）
  - affordance 能看到 depth
  - future 能看到 depth + affordance
  - 兼容 flash_attention_2
- **Token 序列**: `[instruction] → [primary×49] → [wrist×49] → [✨×49] → [🟩×49] → [🟧×49]`

**当前状态**：
- [x] 框架代码完成（`UamGR00T_GIA.py`）
- [x] 配置文件完成（Calvin + RoboTwin）
- [ ] Smoke test 待运行
- [ ] 训练待启动







## aux head 实验计划

* [x] 0624版本：image token 先于 instruction tokens（motivation work）
* [x] 保证 image tokens attend to instruction tokens
* [x] **LT 系列**：使用 special token 来提升 aux head 的区分度（串行/并行）
  - 深度 token (✨)、可供性 token (🟩)、未来 token (🟧)
  - Token 作为独立视觉块插入主序列（duplicate primary blocks）
* [x] **DT 系列**：Seer-aligned Future Branch
  - 独立的 Future DiT Branch 架构
  - 历史帧编码器（HistoryVisionEncoder）
  - 可学习 future_query_tokens 通过交叉注意力融合信息
* [x] **hR_multi 基础设施**：多头 aux head 管理
  - `_MultiHeadBase` 抽象基类
  - `_head_order()` 定义 token 注册顺序
  - `_CFG_TO_HEAD_KEY` 映射配置到 head key
  - Action Model 隔离（剥离 aux head tokens）
* [x] 调整每个 aux head 的 weight 计算方式，防止在训练后期仍然抢占大量 action loss
  - `AuxDenoisingSuite` 全局 aux_budget 控制
  - 每个 head 独立 loss_weight 配置
  - EMA 相对均衡乘子（见 7/8 研究记录）
* [ ] **GIA 系列**：层级因果注意力（Strategy B）
  - depth → affordance → future 的因果依赖链
  - 兼容 flash_attention_2（标准因果掩码）
  - 三维感知协同优化

## 研究记录

| 日期 | log |
|------|-----|
| **8/31** | **UamGR00T_DT 微调完成 + GIA 框架开发**<br>1. UamGR00T_DT 微调训练完成（100k steps），10个checkpoint已保存<br>2. 评测进行中，policy server可正常加载，待获取最终success rate<br>3. UamGR00T_GIA框架代码完成，三维增强（Geometry·Interaction·Action）<br>4. GIA配置文件完成（Calvin + RoboTwin），待smoke test验证<br>5. 整理项目进度报告（PROGRESS_REPORT.md）<br>6. 更新CLAUDE.md开发文档（新增调试指南和benchmark章节） |
| **8/9-8/11** | **UamGR00T_DT 微调训练**<br>在Calvin ABC数据上进行100k steps微调（带visual resampling）<br>训练日志：`logs/run_uamgr00t_DT_calvin_finetune_20260809_103546.log` (22M)<br>Checkpoint每10k步保存一次，共10个 (每个9.7GB) |
| **7/29-8/1** | **UamGR00T_DT 预训练完成**<br>在Calvin ABC数据上完成100k steps预训练<br>训练日志：`logs/run_uamgr00t_DT_calvin_pretrain_20260729_100807.log` (62M)<br>模型参数：DiT-B 157M, K=10历史帧, 2 views, future/view=49 tokens |
| **7/22** | **UamGR00T_LT 系列实验**<br>在Calvin D→D上测试三种配置：<br>- depth only<br>- depth + future<br>- depth + affordance + future<br>验证独立token块作为aux head的有效性 |
| **7/8** | **Aux Loss 控制机制优化**<br>更新了loss配置：<br>`total_loss = action_loss_fm + aux_total`<br>`aux_total = Σ_i eff_w[i] × raw_loss[i]`<br>`eff_w[i] = global_mult(step) × rel_mult[i](step) × base_w[i]`<br><br>实现训练阶段动态权重调整：<br>- 初期warmup：逐步提升aux loss权重<br>- 中期稳定：保持恒定权重<br>- 后期衰减：余弦降低，最后只有action loss<br>- 引入EMA相对均衡乘子，平衡每个aux head的数值范围 |
| **7/4** | **Special Token + Loss平衡探索**<br>1. 将aux head配置成独立token，性能得到提升<br>2. 在D→D数据集上验证有效组合，同步进行ABC→D验证<br>3. 探索special token在并行和串行上的性能区别（论文所需）<br>4. 发现loss损失不平衡问题（action:recon=2:1, action:recon:depth=49:49:2）<br>5. 设计新的weight计算方式，控制损失占比权重<br>6. 与dq探索适配aux head的benchmark，提前准备数据处理 |

## Benchmark

### Calvin 相关

**当前重点**：ABC→D（泛化评测）

**UamGR00T_DT 状态**：
- ✅ 训练完成：ABC 预训练 (100k) + ABC 微调 (100k)
- 🔄 评测进行中：ABC→D split
- ⏳ 待获取指标：1/2/3/4/5-step success rate, average length

**UamGR00T_GIA 计划**：
- 数据集：`calvin_abc_lt_starvla_uam_state_h8`
- Sidecar 需求：`depth_latent/`, `affordance_px/`, `rgb_latent/`
- 训练配置：见 `examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml`

**历史结果**：
- D→D（7/22）：UamGR00T_LT 系列实验
- ABC→D（待补充）

---

## 技术架构详解

### 1. UamGR00T_DT (Dual Temporal - Seer-aligned)

**核心创新**：
- **历史帧编码**：`HistoryVisionEncoder` 压缩 K-1 帧历史信息
- **Future DiT Branch**：独立分支生成 future tokens
- **交叉注意力融合**：49个可学习 `future_query_tokens` 查询 Qwen hidden states + 历史特征
- **Seer 对齐**：通过 `future_latent` aux head 预测未来帧 VAE latent

**数据流**：
```
Current frame (t) ──────────────────────────┐
                                             ↓
History frames (t-K+1 to t-1)          Qwen3-VL
         ↓                                   ↓
  HistoryVisionEncoder              hidden_states[-1]
         ↓                                   ↓
  compressed_history ──────────────→ Future DiT Branch
                                    (cross attention)
                                             ↓
                                    future_tokens (49×2)
                                             ↓
                    vl_embs = concat(qwen_hidden, future_tokens)
                                             ↓
                                        GR00T DiT
                                             ↓
                                    predicted actions
```

**关键文件**：
- `starVLA/model/framework/VLM4A/UamGR00T_DT.py`
- `starVLA/model/modules/uamvla/components/future_dit_branch.py`
- `starVLA/model/modules/uamvla/components/history_vision_encoder.py`

**配置示例**：
```yaml
framework:
  name: UamGR00T_DT
  history_frames: 10
  future_branch:
    dit_hidden: 512
    dit_depth: 2
    num_future_queries: 49
    pixel_loss_weight: 0.5
```

### 2. UamGR00T_GIA (Geometry·Interaction·Action)

**三维感知架构**：

| 维度 | Aux Head | Token | 输入 | 输出 | Loss |
|------|---------|-------|------|------|------|
| 几何 Geometry | `LatentDepthHead` | ✨×49 | hidden_states | VAE depth latent (64×7×7) | MSE |
| 交互 Interaction | `AffordanceTokenHead` | 🟩×49 | hidden_states | VRB heatmap (112×112) | MSE + fg_MSE |
| 时序 Temporal | `FutureLatentHead` | 🟧×49 | hidden_states | Future VAE latent (64×7×7) | MSE |

**Token 序列布局**（以 qwen_image_size=224 为例）：
```
[instruction] → [primary×49] → [wrist×49] → [✨×49] → [🟩×49] → [🟧×49]
 语言指令        真实视觉                    深度      可供性     未来
```

**注意力策略（Strategy B - 层级因果）**：
```
depth (✨)    → affordance (🟩)  → future (🟧)
    ↓                ↓                  ↓
几何结构感知   基于几何的交互区域   基于几何+交互的
                  定位              未来状态预测
```

**因果链设计理由**：
- `affordance` tokens 能看到 `depth` tokens（交互受3D几何约束）
- `future` tokens 能看到 `depth` + `affordance`（规划需要几何和交互信息）
- 使用标准 Qwen 因果掩码，**兼容 `flash_attention_2`**

**Action Model 隔离**：
- GR00T DiT 仅接收 `[instruction] + [primary×49] + [wrist×49]` 的 hidden states
- Aux head tokens (✨🟩🟧) 被剥离，仅通过 Qwen 自注意力间接影响图像 token 表示

**关键文件**：
- `starVLA/model/framework/VLM4A/UamGR00T_GIA.py`
- `starVLA/model/framework/VLM4A/UamGR00T_hR_multi.py` (_MultiHeadBase)
- `starVLA/model/modules/uamvla/aux_heads/latent_depth_head.py`
- `starVLA/model/modules/uamvla/aux_heads/affordance_token_head.py`
- `starVLA/model/modules/uamvla/aux_heads/future_latent_head.py`

**配置示例**：
```yaml
framework:
  name: UamGR00T_GIA
  aux_loss_control:
    enabled: true
    aux_budget: 1.0
  aux_heads:
    depth_latent:
      enabled: true
      loss_weight: 0.5
      probe_enabled: true
    affordance_px:
      enabled: true
      loss_weight: 0.5
    future_latent:
      enabled: true
      loss_weight: 0.5
      future_offset: 8
```

### 3. Sidecar 数据系统

**预处理脚本**：
```bash
# 深度 latent (几何)
runners/preprocess_depth_latent.py
# 深度 pixel probe
runners/preprocess_depth_vda.py
# 可供性 pixel
runners/preprocess_affordance_px.py
# 可供性 VRB heatmap
runners/preprocess_affordance_vrb.py
# 未来 latent (时序)
runners/preprocess_rgb_latent.py
# RoboTwin 专用
runners/preprocess_robotwin2_depth_affordance.py
```

**数据流**：
```
磁盘 Sidecar (mmap + 64-entry LRU cache)
  depth_latent/chunk-NNN/<camera>/episode_XXXXXX.npy   [F, 64, 7, 7]
  affordance_px/chunk-NNN/<camera>/episode_XXXXXX.npy  [F, 112, 112]
  rgb_latent/chunk-NNN/<camera>/episode_XXXXXX.npy     [F, 64, 7, 7]
         ↓
UamVLAOFT._unpack_lerobot_sample()
         ↓
_collate_aux() with zero-padding + bool masks
         ↓
batch_dict:
  batch["depth_latent"]     (B, 64, 7, 7)  + depth_latent_mask
  batch["affordance_px"]    (B, 112, 112)  + affordance_px_mask
  batch["future_latent"]    (B, 64, 7, 7)  + future_latent_mask
```

---

## 文献支撑

### 时序维度（UamGR00T_DT 核心参考）

**[T1] Seer — ICLR 2025 Oral** ⭐
- arXiv: [2412.15109](https://arxiv.org/abs/2412.15109)
- 代码: [InternRobotics/Seer](https://github.com/InternRobotics/Seer)
- 方法: 预测未来视觉状态，通过逆动力学从预测未来推导动作
- 结果: Calvin ABC-D +21%, 真实任务 +43%
- **启发**: `FutureLatentHead` 和 Future DiT Branch 的直接设计来源

**[T2] FlowVLA — 2025**
- arXiv: [2508.18269](https://arxiv.org/abs/2508.18269)
- 方法: 视觉思维链 vt → ft（光流）→ vt+1
- **启发**: Phase 3 增强方向（光流条件化）

**[T4] VideoVLA — NeurIPS 2025**
- arXiv: [2512.06963](https://arxiv.org/abs/2512.06963)
- 方法: 多模态 DiT 联合建模视频、语言、动作
- **启发**: 未来帧重建与动作预测联合训练范式

### 几何维度（UamGR00T_GIA depth 分支）

**[G1] Manip-as-in-Sim (CDMs) — 2025**
- arXiv: [2509.02530](https://arxiv.org/abs/2509.02530)
- 代码: [ByteDance-Seed/manip-as-in-sim-suite](https://github.com/ByteDance-Seed/manip-as-in-sim-suite)
- 方法: 神经数据引擎对真实深度传感器噪声建模
- **启发**: Phase 4 深度质量提升方向

**[G3] 3D Geometric Priors + Dual-head Aux — 2026**
- arXiv: [2603.07624](https://arxiv.org/abs/2603.07624)
- 方法: 双头辅助学习，一头预测动作，另一头预测物理几何
- **启发**: `LatentDepthHead` 是该思路的实例化

### 交互维度（UamGR00T_GIA affordance 分支）

**[A1] Afford-VLA — 2026**
- arXiv: [2605.24203](https://arxiv.org/abs/2605.24203)
- HuggingFace: [papers/2605.24203](https://huggingface.co/papers/2605.24203)
- 方法: 可学习 `<AFF>` token 查询任务相关交互区域
- **启发**: emoji token (🟩×49) 是 `<AFF>` token 的轻量实现

**[A2] CoA-VLA (Chain-of-Affordance) — 2024/2025**
- arXiv: [2412.20451](https://arxiv.org/abs/2412.20451)
- 方法: 动作预测前插入可供性链式推理步骤
- **启发**: 层级注意力中 affordance tokens 先行，为 future tokens 提供交互先验

**[A3] AffordDex (AAAI 2026)**
- 代码: [Maxwell-Zhao/AffordDex](https://github.com/Maxwell-Zhao/AffordDex)
- 方法: 人类先验可供性融入灵巧抓取
- **启发**: Phase 3 可供性热图预处理增强方向

---

## 下一步工作

### 优先级 P0（本周）

1. **完成 UamGR00T_DT 评测** 🔴
   - 在 Calvin D split 上运行完整评测
   - 记录所有指标并与 Seer baseline 对比
   - 生成评测报告表格

2. **验证 UamGR00T_GIA smoke test** 🟡
   ```bash
   CUDA_VISIBLE_DEVICES=0 python starVLA/model/framework/VLM4A/UamGR00T_GIA.py \
     --config_yaml examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml
   ```

### 优先级 P1（下周）

3. **启动 UamGR00T_GIA 训练** 🟢
   ```bash
   mkdir -p logs && set -o pipefail
   WANDB_MODE=offline bash examples/calvin/train_files/run_uamgr00t_GIA_calvin.sh \
     2>&1 | tee "logs/run_uamgr00t_GIA_calvin_$(date +%Y%m%d_%H%M%S).log"
   ```

4. **整理提交与文档** 🟢
   - 清理 git commits（移除 WIP 修改）
   - 编写 PR 描述
   - 准备向 `starVLA_dev` 提交 PR

### 优先级 P2（后续增强）

5. **Phase 3 & 4 增强**（参考 `plan.md`）
   - 光流时序增强（`preprocess_optical_flow.py`）
   - 深度质量提升（DepthAnything V2 + CDM）
   - 可供性数据增强（Grounded-SAM + AffordDex）

---

## 附录

### 相关文档

- `plan.md` — UamGR00T_GIA 完整设计文档
- `PROGRESS_REPORT.md` — 当前工作进度报告
- `CLAUDE.md` — 项目开发指南
- `docs/starVLA_guideline.md` — 快速开始教程
- `docs/research/uamvla_gr00t_aux_branches_2026-05-20.md` — Aux branches 研究文档

### Image token 先于 instruction tokens 记录

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZGI4OTE1NDE5OGU2NWNjMTA5MzE2YWYyODUwYWZkYThfODhiNzM1ZjljMzI4YTQ1OGE3MjRhN2FhZTk0OWRjNDJfSUQ6NzY1NDk5MzMwODAwMjk5NTM5NF8xNzg4MTc5ODcyOjE3ODgyNjYyNzJfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZTY5OWViZWY4NzZmNTYxY2RkMjBjNTNmZmI2OWMzMDZfMWFlOTUyNzYzZTAyYzIyMDE5MTc3NTE3ODkyZjRhOTZfSUQ6NzY1NDk5MzQ5MjM5NzMxMzIzMV8xNzg4MTc5ODcyOjE3ODgyNjYyNzJfVjM)



