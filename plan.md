# plan.md：UamGR00T-GIA 三维增强框架设计

## 1. 背景与目标

在 `UamGR00T_hR_multi.py` 的多头（multi-head）基础设施之上，构建专注于三个感知维度的新框架 **UamGR00T_GIA**（Geometry · Interaction · Action）。

三个维度与对应 aux head：

| 维度 | Aux Head Key | 实现类 | Token |
|------|-------------|--------|-------|
| 几何 (Geometry) | `depth_latent` | `LatentDepthHead` | ✨ |
| 交互 (Interaction) | `affordance_px` | `AffordanceTokenHead` | 🟩 |
| 时序 (Temporal) | `future_latent` | `FutureLatentHead` | 🟧 |

---

## 2. 当前 Aux Head 确认

### 2.1 LatentDepthHead（几何）
- **任务**：per-token 回归冻结 VAE 编码的深度 latent
- **输入**：对应 `view_idx` 处的 hidden states，shape `(B, 49, D)`
- **监督目标**：`batch["depth_latent"]`，shape `(B, 64, 7, 7)`
- **额外 probe**：线性 probe 从 detach 特征解码深度像素（`depth_px`），报告 `probe_r2_pixel`，不反传
- **Sidecar 路径**：`depth_latent/chunk-NNN/<camera>/episode_XXXXXX.npy`

### 2.2 AffordanceTokenHead（交互）
- **任务**：per-token 回归 VRB 接触热图
- **输入**：对应 `view_idx` 处的 hidden states，shape `(B, 49, D)`
- **监督目标**：`batch["affordance_px"]`，shape `(B, 112, 112)`（rearrange 为 `B×49×256`）
- **Loss**：MSE；额外 `fg_mse`（仅对像素值 > 0.1 的前景区域）防止全零塌缩
- **Sidecar 路径**：`affordance_px/chunk-NNN/<camera>/episode_XXXXXX.npy`

### 2.3 FutureLatentHead（时序）
- **任务**：Seer 风格预见——per-token 回归未来帧的冻结 VAE latent
- **输入**：对应 `view_idx` 处的 hidden states，shape `(B, 49, D)`
- **监督目标**：`batch["future_latent"]`，shape `(B, 64, 7, 7)`，时间偏移 `future_offset`（默认=`action_horizon`）
- **Sidecar 路径**：`rgb_latent/chunk-NNN/<camera>/episode_XXXXXX.npy[base + future_offset]`

---

## 3. 文献调研

### 3.1 几何维度 (Depth / Geometry)

**[G1] Manip-as-in-Sim (CDMs) — 2025** ✅ 可复现
- arXiv: [2509.02530](https://arxiv.org/abs/2509.02530)
- 代码: https://github.com/ByteDance-Seed/manip-as-in-sim-suite
- 方法: 神经数据引擎对真实深度传感器噪声建模，从仿真生成高质量 CDM 深度数据，实现策略零样本 sim-to-real 迁移。
- 启发: 可用 CDM 流程替代仿真直接渲染深度，提升 `depth_latent` 监督质量。

**[G2] Depth Diffusion for Robot Manipulation — 2025**
- arXiv: [2511.22505](https://arxiv.org/abs/2511.22505)
- 方法: 扩散模型将含噪传感器深度转换为仿真级别干净深度。
- 启发: 深度 latent 的监督目标可来自扩散去噪后的深度图，增强几何鲁棒性。

**[G3] 3D Geometric Priors + Dual-head Aux — 2026**
- arXiv: [2603.07624](https://arxiv.org/abs/2603.07624)
- 方法: 双头辅助学习，一头预测动作，另一头预测物理几何，强制 latent space 与几何结构对齐。
- 启发: `LatentDepthHead` 正是该思路的实例化；probe 分支提供可解释性。

### 3.2 交互维度 (Affordance / Interaction)

**[A1] Afford-VLA — 2026**
- arXiv: [2605.24203](https://arxiv.org/abs/2605.24203)
- HuggingFace: https://huggingface.co/papers/2605.24203
- 方法: 可学习 `<AFF>` token 从多模态特征查询任务相关交互区域，affordance 掩码直接条件化动作生成。
- 启发: 与本设计的 `🟩×49` token 方案高度一致——emoji token 是 `<AFF>` token 的轻量实现，无需改动 tokenizer。

**[A2] CoA-VLA (Chain-of-Affordance) — 2024/2025**
- arXiv: [2412.20451](https://arxiv.org/abs/2412.20451)
- 方法: 动作预测前插入可供性链式推理步骤，将视觉和文本线索基础化为结构性交互点。
- 启发: 层级注意力中 affordance tokens 先行，为 future tokens 提供交互先验，正是本方案的设计。

**[A3] AffordDex (AAAI 2026)** ✅ 可复现
- 代码: https://github.com/Maxwell-Zhao/AffordDex
- 方法: 人类先验可供性融入灵巧抓取，提供接触点功能性约束。
- 启发: 可供性热图预处理中引入人类先验接触模式（尤其用于真实数据）。

**[A4] AffordanceVLA — 2026**
- arXiv: [2606.06155](https://arxiv.org/abs/2606.06155)
- 方法: 结构化可供性预测作为感知-动作中间表示，含自动数据增强流程应对标注稀缺。
- 启发: 数据增强策略可补充 affordance_px sidecar 的样本多样性。

### 3.3 时序维度 (Future / Temporal)

**[T1] Seer — ICLR 2025 Oral** ⭐ 核心参考 ✅ 可复现
- arXiv: [2412.15109](https://arxiv.org/abs/2412.15109)
- 代码: https://github.com/InternRobotics/Seer
- 方法: 预测未来视觉状态，通过逆动力学模型从预测未来推导动作。CALVIN ABC-D +21%，真实任务 +43%。
- 启发: `FutureLatentHead` 的直接设计来源。当前实现已是 Seer 风格（latent 空间回归）。

**[T2] FlowVLA — 2025**
- arXiv: [2508.18269](https://arxiv.org/abs/2508.18269)
- 方法: 视觉思维链 vt → ft（光流）→ vt+1，强制显式运动推理后再预测未来帧。
- 启发: Phase 3 增强——在 `FutureLatentHead` 中加入光流条件，提升物理合理性。

**[T3] AHEAD — 2026**
- arXiv: [2606.02486](https://arxiv.org/abs/2606.02486)
- 方法: 轻量世界模型在 VLA 特征空间预测未来 patch token，条件于光流的逐 token 速度/加速度。
- 启发: 未来 token 的运动速度场条件化可作为增强方向（Phase 3）。

**[T4] VideoVLA — NeurIPS 2025**
- arXiv: [2512.06963](https://arxiv.org/abs/2512.06963)
- 方法: 多模态 DiT 联合建模视频、语言、动作，视觉想象力与动作预测高度相关。
- 启发: 未来帧重建与动作预测的联合训练范式验证了本设计的有效性。

---

## 4. 新框架设计：UamGR00T_GIA

### 4.1 继承链

```
baseframework
└── Qwenvl_OFT  (QwenOFT.py)
    └── UamVLAOFT  (UamVLAOFT.py)
        └── UamVLAGR00T  (UamGR00T.py)
            └── _MultiHeadBase  (UamGR00T_hR_multi.py)
                └── UamGR00T_GIA  ← 新建 (UamGR00T_GIA.py)
```

### 4.2 Token 序列布局

以 `qwen_image_size=224`（patches_per_view=49）为例：

```
[instruction] → [primary×49] → [wrist×49] → [✨×49] → [🟩×49] → [🟧×49]
                  real views                  depth    afford    future
```

总 token 数 = instruction_len + 2×49（图像）+ 3×49（aux heads）。

### 4.3 注意力策略：层级 B (Hierarchical Causal)

**选择 Strategy B 的理由：**

```
depth(✨) → affordance(🟩) → future(🟧)
   ↓               ↓               ↓
几何结构感知   基于几何的         基于几何+交互的
              交互区域定位       未来状态预测
```

因果链：理解 3D 几何 → 识别可操作区域 → 预测操作后未来状态

- `affordance` tokens 通过因果注意力能看到 `depth` tokens（affordance 受 3D 约束）
- `future` tokens 能看到 `depth` + `affordance`（规划需要几何和交互信息）
- 使用标准 Qwen 因果掩码，**兼容 `flash_attention_2`**（Strategy A 需要 4D mask，不兼容）

`_head_order()` 实现：
```python
def _head_order(self) -> list[str]:
    return ["depth", "affordance", "future"]

_CFG_TO_HEAD_KEY = {
    "depth":      "depth_latent",
    "affordance": "affordance_px",
    "future":     "future_latent",
}
```

### 4.4 Action Model 隔离

继承自 `_MultiHeadBase._hidden_for_action()`：
- 剥离所有 ✨/🟩/🟧 位置的 hidden states
- GR00T DiT 仅接收 `[instruction] + [primary×49] + [wrist×49]` 的 hidden
- Aux head token 信息仅通过 Qwen 自注意力间接影响图像 token 表示

### 4.5 Aux Loss 控制

```python
# _MultiHeadBase._compute_aux_training_losses
# → DynamicAuxSuite（若配置）→ AuxDenoisingSuite（兜底）
total_loss = action_loss + aux_budget × Σ(loss_weight_i × head_loss_i)
```

### 4.6 关键实现点

1. **`_CFG_TO_HEAD_KEY` 必须与 YAML `aux_heads` 字段名精确对应**
   `depth_latent` / `affordance_px` / `future_latent` 是 sidecar 系统的固定 key，不能改名。

2. **token 注册顺序 = `_head_order()` 顺序**
   `_register_all_head_tokens()` 按 `_head_order()` 遍历，emoji 注册先后决定 suffix 拼接顺序。

3. **sidecar 验证**
   三个 head 在 `__init__` 时各自调用 `meta.json` 验证（与 `UamGR00T_hR_LT` 相同逻辑），
   Calvin 和 RoboTwin sidecar 需提前用 `runners/` 脚本生成（见第 5 节）。

4. **`view_idx=0` 固定**
   hR 系列所有 head 的 `view_idx=0`——每个 emoji token 全局唯一，`slice_image_tokens` 按 token_id 定位，无位置脆弱性（对比 LT 系列需 `view_idx=num_views+n`）。

5. **Qwen3.5 兼容**
   `_QWen3_5_VL_Interface` 已修复（本次对话），但 hR 系列需要 `_QWen3_VL_hR_Interface`（支持 `text_suffix=`）。
   Qwen3.5 对应接口需确认 `use_hR_interface: true` 路径（`__init__.py` 的 `get_vlm_model` 判断）。

---

## 5. 数据预处理方案

### 5.1 数据流总览

```
磁盘 Sidecar 目录
  depth_latent/chunk-NNN/<camera>/episode_XXXXXX.npy   [F, 64, 7, 7]
  depth_px/chunk-NNN/<camera>/episode_XXXXXX.npy       [F, 112, 112]
  rgb_latent/chunk-NNN/<camera>/episode_XXXXXX.npy     [F, 64, 7, 7]
  affordance_px/chunk-NNN/<camera>/episode_XXXXXX.npy  [F, 112, 112]
         ↓ mmap O(1) 读取，64-entry LRU cache
UamVLAOFT._unpack_lerobot_sample()
         ↓ stack_optional_tensor_fields (zero-pad + bool mask)
_collate_aux() → batch_dict
  batch["depth_latent"]     (B, 64, 7, 7)  + depth_latent_mask
  batch["affordance_px"]    (B, 112, 112)  + affordance_px_mask
  batch["future_latent"]    (B, 64, 7, 7)  + future_latent_mask
```

### 5.2 预处理脚本（现有）

| 维度 | 脚本 | 输出目录 |
|------|------|---------|
| 几何 - depth latent | `runners/preprocess_depth_latent.py` | `depth_latent/` |
| 几何 - depth pixel probe | `runners/preprocess_depth_vda.py` | `depth_px/` |
| 交互 - affordance px | `runners/preprocess_affordance_px.py` | `affordance_px/` |
| 交互 - VRB heatmap | `runners/preprocess_affordance_vrb.py` | `affordance_heatmaps/` |
| 时序 - future latent | `runners/preprocess_rgb_latent.py` | `rgb_latent/` |
| RoboTwin 专用 | `runners/preprocess_robotwin2_depth_affordance.py` | depth + affordance |

### 5.3 文献驱动的预处理增强方向

#### 5.3.1 几何增强（受 [G1] CDMs 启发）
- **当前**：仿真渲染深度 → frozen VAE encode → `depth_latent`
- **增强**：
  1. 用 DepthAnything V2（metric depth）重新处理 RGB 帧，得到高质量度量深度
  2. （可选）用 CDM 扩散去噪做 sim-to-real 深度精修
  3. 再 frozen VAE encode 得到更准确的 `depth_latent`
- **实现**：修改 `runners/preprocess_depth_latent.py` 的深度来源（替换 depth sensor → DAV2）

#### 5.3.2 交互增强（受 [A1][A3] 启发）
- **当前**：VRB 仿真接触热图（Calvin/RoboTwin 仿真直接导出）
- **增强（用于真实机器人数据）**：
  1. 用 Grounded-SAM + LLM 生成零样本接触区域 heatmap
  2. 或用 AffordDex 的人类先验接触点模型生成功能性 affordance
- **实现**：在 `runners/preprocess_affordance_px.py` 中扩展标注来源

#### 5.3.3 时序增强（受 [T2] FlowVLA + [T3] AHEAD 启发，Phase 3）
- **当前**：frozen VAE 将未来 RGB 帧编码为 `future_latent[t + future_offset]`
- **增强**：
  1. 计算当前帧 → 未来帧光流 ft（用 RAFT 或 UniFlow）
  2. 将光流 resize 到 7×7 patch 粒度，作为额外条件
  3. `FutureLatentHead` 中：`condition = concat(image_tokens, flow_tokens)` → 运动先验条件化
- **新增脚本**：`runners/preprocess_optical_flow.py` → `flow_px/` sidecar
- **新增字段**：`batch["flow_px"]`，在 `_collate_aux()` 注册

---

## 6. 训练配置模板

### 6.1 Calvin（基线）

```yaml
# examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml
run_id: uamvla_gr00t_gia_calvin_abc
run_root_dir: playground/Checkpoints
seed: 42
wandb_entity: tancilon1-fudan-university-school-of-management
wandb_project: uamvla

framework:
  name: UamGR00T_GIA
  obs_image_size: [224, 224]
  qwen_image_size: 224
  qwenvl:
    base_vlm: ./ckpt/Qwen3-VL-4B-Instruct
    attn_implementation: flash_attention_2   # Strategy B 兼容
    vl_hidden_dim: 2048
    enable_gradient_checkpointing: true
  action_model:
    action_model_type: DiT-B
    action_hidden_dim: 1024
    hidden_size: 1024
    add_pos_embed: true
    action_dim: 7
    state_dim: 7
    action_horizon: 8
    future_action_window_size: 7
    past_action_window_size: 0
    repeated_diffusion_steps: 8
    num_inference_timesteps: 4
    num_target_vision_tokens: 32
    diffusion_model_cfg:
      dropout: 0.2
      final_dropout: true
      interleave_self_attention: true
      norm_type: ada_norm
      num_layers: 16
      positional_embeddings: null
  aux_loss_control:
    enabled: true
    aux_budget: 1.0
  aux_heads:
    depth_latent:
      enabled: true
      loss_weight: 0.5
      latent_channels: 64
      camera: image
      probe_enabled: true
    affordance_px:
      enabled: true
      loss_weight: 0.5
      camera: image
    future_latent:
      enabled: true
      loss_weight: 0.5
      latent_channels: 64
      camera: image
      future_offset: 8

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: datasets
    data_mix: calvin_abc_lt_starvla_uam_state_h8
    enable_sidecar_io: true
    strict_sidecar_io: false
    image_target_cache_maxsize: 256
    include_state: true
    gr00t_state_indices: [0, 1, 2, 3, 4, 5, 7]
    obs:
      - video.primary_image
      - video.wrist_image
    state_key: state
    action_key: action
    state_dim: 8
    full_state_dim: 8
    action_dim: 7
    action_horizon: 8
    action_type: delta_qpos
    per_device_batch_size: 8
    load_all_data_for_training: true

trainer:
  max_train_steps: 100000
  num_warmup_steps: 5000
  save_interval: 10000
  eval_interval: 100
  learning_rate:
    base: 2.5e-5
    qwen_vl_interface: 1.0e-5
    action_model: 1.0e-4
  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 1.0e-6
  freeze_modules: ""
  max_grad_norm: 1.0
  weight_decay: 0.0
  gradient_accumulation_steps: 2
  gradient_checkpointing: true
  optimizer:
    name: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-8
    weight_decay: 1.0e-8
```

### 6.2 RoboTwin（action_dim=14，3 views）

与 Calvin 配置相同结构，调整：
- `action_dim: 14`，`state_dim: 14`，`action_horizon: 16`
- `obs: [video.cam_high, video.cam_left_wrist, video.cam_right_wrist]`
- `data_mix: robotwin_all`（或 `robotwin_all_dt_k10`）

---

## 7. 实现计划

### ✅ Phase 1 — 新 Framework 文件（已完成）

**文件**：`starVLA/model/framework/VLM4A/UamGR00T_GIA.py`

实现要点：
- 继承 `_MultiHeadBase`（`UamGR00T_hR_multi.py`），注册名 `"UamGR00T_GIA"`
- `_head_order()` → `["depth", "affordance", "future"]`（Strategy B 层级因果，兼容 flash_attention_2）
- `_maybe_build_aux_heads()` 构建三头（`aux_heads` 注册键遵循 `_MultiHeadBase._CFG_TO_HEAD_KEY` 约定）：
  - `"depth"` → `LatentDepthHead`（✨，`view_idx=0`）
  - `"affordance"` → `AffordanceTokenHead`（🟩，`view_idx=0`）
  - `"future"` → `FutureLatentHead`（🟧，`view_idx=0`）
- 三个 sidecar fail-fast 验证：`_assert_depth_latent_meta` / `_assert_affordance_px_meta` / `_assert_future_latent_meta`
- `_GIA_NON_HEAD_KEYS` 过滤框架级 key（`future_offset` 由 dataloader 消费，不传入 head 构造器）
- 语法验证通过：`python3 -m py_compile` ✅

**验收命令**：
```bash
CUDA_VISIBLE_DEVICES=N python starVLA/model/framework/VLM4A/UamGR00T_GIA.py \
  --config_yaml examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml
```

> **与原计划的一处修正**：`_CFG_TO_HEAD_KEY` 不需要在 GIA 中重新定义——直接沿用 `_MultiHeadBase` 中已有的
> `"depth"→"depth"`, `"affordance"→"affordance"`, `"future"→"future"` 映射，YAML 中 aux_heads 字段相应使用
> `depth` / `affordance` / `future`（而非原计划中的 `depth_latent` / `affordance_px` / `future_latent`）。

---

### ✅ Phase 2 — 训练配置文件（已完成）

**文件（4 个，均已创建并 chmod +x）**：

| 文件 | 关键参数 |
|------|---------|
| `examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml` | action_dim=7, 2 views, future_offset=8, batch=8 |
| `examples/calvin/train_files/run_uamgr00t_GIA_calvin.sh` | `$@` CLI透传，WANDB_MODE=offline |
| `examples/Robotwin/train_files/run_uamgr00t_GIA_robotwin.yaml` | action_dim=14, 3 views, future_offset=16, batch=4 |
| `examples/Robotwin/train_files/run_uamgr00t_GIA_robotwin.sh` | `$@` CLI透传，WANDB_MODE=offline |

**启动命令**：
```bash
# Calvin
mkdir -p logs && set -o pipefail
WANDB_MODE=offline bash examples/calvin/train_files/run_uamgr00t_GIA_calvin.sh \
  2>&1 | tee "logs/run_uamgr00t_GIA_calvin_$(date +%Y%m%d_%H%M%S).log"

# RoboTwin
mkdir -p logs && set -o pipefail
WANDB_MODE=offline bash examples/Robotwin/train_files/run_uamgr00t_GIA_robotwin.sh \
  2>&1 | tee "logs/run_uamgr00t_GIA_robotwin_$(date +%Y%m%d_%H%M%S).log"
```

---

### Phase 3 — 光流时序增强（可选，基线验证后）

**新增**：
- `runners/preprocess_optical_flow.py` — 生成 `flow_px/` sidecar（用 RAFT）
- 修改 `starVLA/model/modules/uamvla/aux_heads/future_latent_head.py`：接受可选 `flow_condition`
- 修改 `starVLA/model/modules/uamvla/collator_helpers.py`：注册 `flow_px` 字段

### Phase 4 — 深度质量提升（可选）

- 修改 `runners/preprocess_depth_latent.py`：接入 DepthAnything V2 metric depth
- 依赖：DepthAnything V2 checkpoint

---

## 8. 参考文献汇总

| 编号 | 论文 | 年份 | 维度 | 代码 |
|------|------|------|------|------|
| G1 | Manip-as-in-Sim (CDMs) | 2025 | 几何 | [ByteDance-Seed/manip-as-in-sim-suite](https://github.com/ByteDance-Seed/manip-as-in-sim-suite) ✅ |
| G2 | Depth Diffusion for Manipulation | 2025 | 几何 | arXiv:2511.22505 |
| G3 | 3D Geometric Priors Dual-head | 2026 | 几何 | arXiv:2603.07624 |
| A1 | Afford-VLA | 2026 | 交互 | arXiv:2605.24203 |
| A2 | CoA-VLA | 2024/2025 | 交互 | arXiv:2412.20451 |
| A3 | AffordDex (AAAI 2026) | 2026 | 交互 | [Maxwell-Zhao/AffordDex](https://github.com/Maxwell-Zhao/AffordDex) ✅ |
| A4 | AffordanceVLA | 2026 | 交互 | arXiv:2606.06155 |
| T1 | Seer (ICLR 2025 Oral) | 2025 | 时序 | [InternRobotics/Seer](https://github.com/InternRobotics/Seer) ✅ |
| T2 | FlowVLA | 2025 | 时序 | arXiv:2508.18269 |
| T3 | AHEAD | 2026 | 时序 | arXiv:2606.02486 |
| T4 | VideoVLA (NeurIPS 2025) | 2025 | 时序 | arXiv:2512.06963 |
