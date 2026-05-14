# UamVLA → QwenOFT 集成迁移设计 (UamVLAOFT) — CALVIN ABCD_D 切片

**Date**: 2026-05-13
**Owner**: tancilon
**Status**: Draft v2, awaiting user review (codex-reviewed)
**Replaces (in spirit, not deletes)**: [2026-04-29-starvla-migration-design.md](2026-04-29-starvla-migration-design.md)

---

## 1. 背景

[2026-04-29-starvla-migration-design.md](2026-04-29-starvla-migration-design.md) 把 uamvla 从独立仓库搬进了 starVLA，但搬完后 uamvla 跟 starVLA 主线仍然走三条不共享的旁路：

| 关注点 | starVLA 主线 | uamvla 现在 |
|---|---|---|
| **数据格式** | LeRobot v2/v3 parquet + `meta/modality.json` + `examples/<task>/train_files/data_registry/` | `tools/preprocess/{libero,calvin}_preprocessor.py` 产 JSONL + `statistics.yaml` |
| **Dataloader** | [`lerobot_datasets.py`](../../starVLA/dataloader/lerobot_datasets.py) → `LeRobotMixtureDataset` + `ComposedModalityTransform` | [`uamvla_dataset.py`](../../starVLA/dataloader/uamvla_dataset.py) → `UamVLADataset` |
| **VLM 封装** | [`_QWen3_VL_Interface`](../../starVLA/model/modules/vlm/QWen3.py)（QwenFast / QwenOFT / QwenPI / QwenGR00T 都用） | [`backbone_wrapper.py`](../../starVLA/model/modules/uamvla/backbone_wrapper.py) 直接 `Qwen3VLForConditionalGeneration.from_pretrained` + 自定义 chat template + state token splicing |
| **Action 编码** | QwenFast：FAST BPE；QwenOFT：连续回归（🔍 占位符 + MLP head + L1） | uamvla：每维 256 档分箱 `<ACT_i>` + `<\|action_start\|>` + `ActionLogitsProcessor` |
| **Loss** | HF `forward.loss`（QwenFast）或自算 L1（QwenOFT） | 4 个 aux head loss 求和后塞进 `action_loss` |

集成度低的实际成本：维护两套 Qwen3-VL 调用代码、两套数据预处理、两个 dataloader、两个 framework 类。任何 Qwen3-VL processor 升级或者 transformers 版本变化，uamvla 旁路都要单独适配。

## 2. 目标与非目标

### 2.1 目标

- **单一数据路径**：CALVIN ABCD_D 数据走 LeRobot parquet，下线 JSONL 预处理和 `uamvla_dataset.py`
- **单一 VLM 入口**：uamvla 复用 `_QWen3_VL_Interface` 和 QwenOFT 的 `build_qwenvl_inputs` / `processor.apply_chat_template` 路径
- **保留三个感知 aux head**：pose（6D 物体位姿）、future（未来帧重建）、recon（目标物体重建），它们是 uamvla 的核心研究 contribution
- **action 改由 L1 regression on 🔍 query tokens 负责**：复用 [QwenOFT](../../starVLA/model/framework/VLM4A/QwenOFT.py) 的 prompt suffix + `_gather_action_token_embeddings` + [`L1RegressionActionHead`](../../starVLA/model/modules/action_model/MLP_ActionHeader.py) 三件套（OpenVLA-OFT 风格）
- **SR 硬目标：达 OpenVLA-OFT 级 SR**：CALVIN ABCD_D 端到端 average sequence length 与 SR 在 OpenVLA-OFT 报告的可复现范围内（具体数值在 implementation plan 阶段从 OpenVLA-OFT repo 取最新值固化）；次门槛 SR ≥ 80%，<80% 视为方案失败需重新设计

### 2.2 非目标

- **本 spec 不涵盖 LIBERO**：本地仅有 CALVIN ABCD_D，LIBERO 数据集后续单独再开 spec。Phase 1 预处理器、Phase 1 验收、Phase 2 训练/验收**全部以 CALVIN ABCD_D 为唯一目标**。`tools/preprocess/libero_preprocessor.py` 在本轮不动；当 LIBERO 上线时按本 spec 同样的模式扩展即可
- **不保留 state encoder / ModularStateEncoder / canonical state / EmbodimentAdapter / 9 个 `<\|state_*\|>` token / `_replace_state_tokens`**。Robot state 信息完全靠图像感知（agentview + wrist cam）
- **不保留 per-dim 分箱 action 离散化路径**（`<ACT_0>..<ACT_255>` + `<\|action_start\|>` + `ActionTokenizer` + `ActionLogitsProcessor`）。Action 改用 L1 连续回归
- **不动 QwenOFT 自身**：所有 uamvla 专属改动通过子类化 / 新建 framework 完成，不污染 `Qwenvl_OFT` 类的代码路径（QwenFast / QwenPI / QwenGR00T 等共用 `_QWen3_VL_Interface` 的 framework 也不受影响）
- **不做 VLM co-training**：`supports_training_tag("vlm")` 返回 False
- **不保留旧 `UamVLA.py` ckpt 兼容**：旧 UamVLA framework 训出的 ckpt 在 Phase 2 完成后**不可用**（删 framework 同时删旧 eval server 入口）。用户已确认接受这点

### 2.3 研究身份转变

`UamVLAOFT` 不是原 uamvla 算法的等价物，但**比 FAST 替代版改动小得多**：

| Design feature | 原 uamvla | UamVLAOFT | 变化幅度 |
|---|---|---|---|
| Action 表示 | 每维 256 档分箱（离散） | 连续 L1 回归 | **小**：连续是离散化的极限版本，仍是 per-dim 独立预测；OpenVLA-OFT 论文在 LIBERO 上证明连续回归优于离散 token |
| State 输入 | `canonical_state` → `ModularStateEncoder` → splice 进 `inputs_embeds` | 无 state，靠图像感知 | **大**：损失一条研究 contribution |
| Aux heads | action / pose / future / recon 四头 | pose / future / recon 三头（action 由 MLP head 独立负责） | **小**：感知 aux head 全部保留 |
| Backbone | 自定义 chat template + state token + 256 个 ACT_i + action_start | QwenOFT 原生：instruction 末尾追加 prompt suffix + 🔍 占位符 | **小**：仍然是 Qwen3-VL，仍然是 prefix-decoder |

总体上这是"在 QwenOFT 上加 3 个感知 aux head + 抛弃 state encoder"，论文层面应按"QwenOFT + perception aux heads"重新表述定位。

## 3. 总体架构

### 3.1 两阶段交付

```
                          ┌──────────────────────────────────────────────────────────┐
                          │ Phase 1: CALVIN ABCD_D → LeRobot 格式                       │
                          │   - 改 calvin_preprocessor.py：JSONL → parquet              │
                          │   - 新 CALVIN DataConfig（aux state 走标准 state modality）│
                          │   - 改 LeRobotSingleDataset._pack_sample 透传 trajectory_id │
                          │   - Phase 1 验收 = LeRobot smoke test，不要求老 UamVLA 跑通 │
                          └──────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
                          ┌──────────────────────────────────────────────────────────┐
                          │ Phase 2: UamVLAOFT framework                               │
                          │   - 加载原版 Qwen3-VL-8B-Instruct（无须烧 ckpt）             │
                          │   - UamVLAOFT(Qwenvl_OFT)                                  │
                          │   - 三个 aux head 挂在 forward.hidden_states[-1] 上          │
                          │   - sidecar IO 在 framework 内部按 trajectory_id 反查        │
                          │   - 训练 + 推理路径都 force-resize 640×640                  │
                          │   - 删除大量旧 uamvla 代码（含 UamVLA.py）                  │
                          └──────────────────────────────────────────────────────────┘
```

Phase 1 可独立验证（LeRobot dataloader smoke test）；Phase 2 在 Phase 1 稳定后启动。

### 3.2 训练时数据流（Phase 1 + Phase 2 全部完成后）

```
CALVIN ABCD_D 原始数据 (datasets/calvin/task_ABCD_D/{training,validation}/episode_*.npz)
        │
        ▼ calvin_preprocessor.py
LeRobot parquet (CALVIN side)
  ├─ data/chunk-000/episode_*.parquet
  │    ├─ video.primary_image / video.wrist_image (delta=[0])
  │    ├─ video.primary_image (delta=[H-1])         ← image_future 用 delta_indices 取
  │    ├─ action.{x,y,z,roll,pitch,yaw,gripper} (delta=[0..H-1])
  │    ├─ state.target_pose_rot6d / state.target_pose_trans  ← pose_gt（合并进 state 列）
  │    ├─ state.static_cam_rot6d / state.static_cam_trans   ← static_cam_extrinsic（合并进 state 列）
  │    └─ annotation.human.action.task_description  ← lang
  ├─ videos/chunk-000/...mp4
  ├─ meta/{modality.json,episodes.jsonl,tasks.jsonl,info.json,stats_gr00t.json}
  ├─ point_clouds/<trajectory_id>/<base_index>.npy   ← sidecar，命名规约
  ├─ image_targets/<trajectory_id>.png               ← sidecar，按 episode 存
  └─ camera_params.json                              ← PoseHead viz 用的相机内参
        │
        ▼ LeRobotSingleDataset.__getitem__ → _pack_sample (patched per §4.4)
batch dict (LeRobot-style):
  {"image": List[PIL], "lang": str, "action": (H,7), "state": (S,),
   "__trajectory_id": int, "__base_index": int}    ← patched 多两个 int key
        │
        ▼
  UamVLAOFT.forward(examples)
    ├─ _unpack_lerobot_sample(e)            ← LeRobot key → framework key + sidecar IO（§4.5）
    │    ├─ split state tensor → pose_gt, static_cam_extrinsic
    │    ├─ load point_clouds/<__trajectory_id>/<__base_index>.npy → point_cloud
    │    └─ load image_targets/<__trajectory_id>.png (cached) → image_target
    ├─ _force_resize_640(image)             ← 训练 + 推理同函数（§5.2）
    ├─ instructions += " Please predict the next K robot actions: <action>🔍🔍...🔍<action>."
    ├─ qwen_inputs = build_qwenvl_inputs(images, instructions)        ← user-turn only
    ├─ qwenvl_outputs = self.qwen_vl_interface(**qwen_inputs,
    │                                          output_hidden_states=True)
    ├─ hidden = qwenvl_outputs.hidden_states[-1]   # (B, L, H)
    ├─ assert (input_ids == image_token_id).sum(1) == ppv * V       ← spatial_reader 不变量校验
    ├─ action_queries = _gather_action_token_embeddings(hidden, input_ids, 🔍_id)
    ├─ pred_actions = L1RegressionActionHead(action_queries)          # (B, H, 7)
    ├─ total = L1(pred_actions, gt_actions)                            ← action loss
    ├─ batch_dict = _collate_aux(examples, qwen_inputs)               ← 包含 input_ids！
    ├─ for head in {pose, future, recon}:
    │     total += head.compute_loss(hidden, batch_dict, mask)         ← aux losses
    └─ return {"action_loss": total, ...metrics}
```

## 4. Phase 1：CALVIN ABCD_D → LeRobot

### 4.1 数据 schema

CALVIN DataConfig 落在 `examples/calvin/train_files/data_registry/data_config.py`。

> **重要变更（v2，应对 codex Showstopper #1）**：`aux_state_keys` 这类自定义 modality 类别**不被 LeRobot 支持** —— [datasets.py:1802-1806](../../starVLA/dataloader/gr00t_lerobot/datasets.py#L1783) 的 `get_data_by_modality` 硬编码只识别 `video / state / action / language` 四类。修复方案：把 pose target 和 static cam extrinsic **并入标准 `state` modality**，在 framework 端按 index slice 出来。

```python
class UamVLACalvinDataConfig:
    # ── 标准 LeRobot modality 字段 ──────────────────────────────────
    video_keys = ["video.primary_image", "video.wrist_image"]

    # robot proprio + aux state 全部合并进同一个 state modality（LeRobot 只识别这一类）
    # PoseHead 不消费 robot proprio；只消费末尾的 pose + cam 字段（按 index 切）
    state_keys = [
        # 不参与 backbone 输入（state encoder 抛弃），但 LeRobot stats / normalization
        # 仍可以正常算；列在这里是为了走通 LeRobot pipeline
        "state.robot_obs",                # 15 维（CALVIN scene_obs 标准布局，详见 4.3）
        # uamvla aux 字段（按 index 切给 PoseHead）
        "state.target_pose_rot6d",        # 6 维 (rotation 6D)
        "state.target_pose_trans",        # 3 维 (translation)
        "state.static_cam_rot6d",         # 6 维 (静态相机外参 viz 用)
        "state.static_cam_trans",         # 3 维
    ]
    aux_state_slice = {                    # framework 用这个 slice 表
        "target_pose_rot6d": (15, 21),
        "target_pose_trans": (21, 24),
        "static_cam_rot6d":  (24, 30),
        "static_cam_trans":  (30, 33),
    }

    action_keys = ["action.x", "action.y", "action.z",
                   "action.roll", "action.pitch", "action.yaw",
                   "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(8))         # action_horizon=8

    # ── sidecar 配置（framework 端反查，不走 transform）─────────────────
    sidecar_dataset_root = None             # 运行时由 framework __init__ 注入
    pointcloud_subdir = "point_clouds"
    image_target_subdir = "image_targets"
    camera_params_filename = "camera_params.json"   # 替代旧 statistics.yaml 的相机内参源
```

### 4.2 aux modality 存法（v2 拍板）

| 字段 | 存法 | 修订理由（相对 v1） |
|---|---|---|
| `image_target` | **按 episode (trajectory_id) 存 sidecar**：`<dataset_root>/image_targets/<trajectory_id>.png` | v1 是按 task 存，但 CALVIN 的 stack/unstack 任务同 task 有多目标 —— v2 直接按 episode 存，规避降级分支，CALVIN/LIBERO 统一处理 |
| `image_future` | **不存**，dataloader 端用 `ModalityConfig(delta_indices=[H-1], modality_keys=["video.primary_image"])` 取未来帧 | 不变 |
| `pose_6d` + `static_cam_extrinsic` | **合并进标准 `state` modality**，按 index slice | v1 是新 `aux_state` modality —— LeRobot 不支持。改为合并进 `state` |
| `point_cloud` | **sidecar `.npy` per-frame**：`<dataset_root>/point_clouds/<trajectory_id>/<base_index>.npy`，**framework 端 IO**，不在 transform 里加载 | v1 是 transform 加载 —— `_pack_sample` 不透传 transform 写入的非标准字段。改为 framework 端用 `__trajectory_id` + `__base_index` 直接 IO |
| `camera_params` (相机内参) | **`<dataset_root>/camera_params.json`**：dataset-level 单文件（CALVIN agentview 内参在整个数据集恒定） | v1 沿用 `statistics.yaml` —— 已退役。改为独立 json，PoseHead viz 从这里读 |
| `canonical_state` | **不存**（state encoder 抛弃） | 不变 |

### 4.3 预处理器改写

**仅改 `tools/preprocess/calvin_preprocessor.py`**（不动 libero_preprocessor.py，本 spec 不涵盖 LIBERO）。

输入：`/mnt/data/dengqi/code/UniamVLA/datasets/calvin/task_ABCD_D/{training,validation}/episode_*.npz` —— CALVIN 原始 per-step npz 文件（每帧一个 npz，含 `actions`、`scene_obs`、`rgb_static`、`rgb_gripper`、`depth_static` 等）。

输出布局：

```
playground/Datasets/UAMVLA_LEROBOT_CALVIN_ABCD/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet     ← 每个 episode 一个 parquet，行=帧
│       └── ...
├── videos/                            ← 视频流（LeRobot v2 video backend）
│   └── chunk-000/
│       ├── video.primary_image/
│       │   ├── episode_000000.mp4
│       │   └── ...
│       └── video.wrist_image/
│           └── ...
├── meta/
│   ├── modality.json                  ← 声明 state/action/video/language 字段范围
│   ├── episodes.jsonl                 ← episode 列表 + task_id 映射
│   ├── tasks.jsonl
│   ├── info.json
│   └── stats_gr00t.json               ← LeRobot 标准统计（自动从 parquet 算）
├── image_targets/
│   └── <trajectory_id>.png            ← 目标物体裁图 sidecar（按 episode 存）
├── point_clouds/
│   └── <trajectory_id>/
│       └── <base_index>.npy           ← (1024, 3) float32
└── camera_params.json                 ← 相机内参 / 视场角（PoseHead viz 用）
```

预处理器对每个 CALVIN episode（连续的一段 episode_NNNNNNN.npz）：
- 视频帧（`rgb_static` / `rgb_gripper`）→ 按 LeRobot 约定写两路 mp4
- 7D action → parquet `action.*` 列
- 33 维 state（15 维 `state.robot_obs` + 9 维 pose + 9 维 cam extrinsic）→ parquet `state.*` 列；其中 robot_obs 取自 `scene_obs` 前 15 维（详见 [calvin_preprocessor.py](../../tools/preprocess/calvin_preprocessor.py) 现状），pose/cam 由现有 target object resolver + camera 内外参生成
- task instruction → parquet `annotation.human.action.task_description` 列（从 CALVIN 的 `lang_annotations/*.npy` 读）
- 目标物体裁图（同 task 一次性裁好）→ `image_targets/<trajectory_id>.png`
- 点云（每帧由 depth_static + cam 内参反投影）→ `point_clouds/<trajectory_id>/<base_index>.npy`
- 相机内参 → 整个 dataset 一份 `camera_params.json`

退役 `statistics.yaml`，统计由 LeRobot 在 dataloader 首次加载时自动算 + 缓存到 `meta/stats_gr00t.json`。相机内参从 `statistics.yaml` 分离到独立 `camera_params.json`（应对 codex Minor #3）。

### 4.4 Dataloader 改造（最小侵入 patch）

> **应对 codex Showstopper #2、#3**：`_pack_sample` 不透传非标准字段 → `_unpack_lerobot_sample` 拿不到 trajectory_id 反查 sidecar、aux head 拿不到 `input_ids`。修复必须在 LeRobot core patch。

**Patch 1: `LeRobotSingleDataset.__getitem__` + `_pack_sample`**（最小侵入，~10 行）

[datasets.py:1357](../../starVLA/dataloader/gr00t_lerobot/datasets.py#L1357) 改为：

```python
def __getitem__(self, index: int) -> dict:
    trajectory_id, base_index = self.all_steps[index]
    raw_data = self.get_step_data(trajectory_id, base_index)
    raw_data["__trajectory_id"] = int(trajectory_id)
    raw_data["__base_index"] = int(base_index)
    data = self.transforms(raw_data)
    return self._pack_sample(data)

def _pack_sample(self, data: dict) -> dict:
    # ... (existing logic for action / image / lang / state) ...
    sample = {
        "action": action,
        "image": all_images,
        "lang": language,
        "language": language,
    }
    # ...existing state branch...
    # NEW: pass through sidecar lookup keys (codex Showstopper #2 fix)
    if "__trajectory_id" in data:
        sample["__trajectory_id"] = int(data["__trajectory_id"])
    if "__base_index" in data:
        sample["__base_index"] = int(data["__base_index"])
    return sample
```

理由：
- 用 `__` 前缀避免和 LeRobot modality 名字冲突
- 两个 int 字段语义固定、对其他 framework 零兼容性破坏（QwenFast/QwenPI/QwenGR00T 不读就忽略）
- 不需要在 modality_config 里加 "passthrough_keys" 概念，比方案 (a) 改动小一个数量级

**Patch 2：不需要写 PointCloudLoadTransform / ImageTargetByTaskTransform**

v1 spec 里写的两个 transform 删除。sidecar IO 全部移到 framework 的 `_unpack_lerobot_sample`（§4.5）。`DataConfig.transform()` 用标准 `StateActionToTensor` / `StateActionTransform`，但 **normalization 范围要显式控制**（详见下方代码），不能简单照搬 [Libero4in1DataConfig](../../examples/LIBERO/train_files/data_registry/data_config.py)。

**DataConfig.modality_config()** —— *codex v2 拍板修正*：[get_data_by_modality (datasets.py:1808)](../../starVLA/dataloader/gr00t_lerobot/datasets.py#L1808) 是 exact string match (`else: raise ValueError(f"Invalid modality: {modality}")`)，**`"video_future"` 会立即 raise**。修复：把 future 帧合并进单个 `"video"` ModalityConfig，用 `delta_indices=[0, action_horizon-1]`（双偏移）一次取 t=0 和 t=H-1 两帧；framework 端 `_unpack_lerobot_sample` 按时间索引拆出 `image` 和 `image_future`：

```python
def modality_config(self):
    return {
        # video 单 ModalityConfig，delta_indices 双帧：t=0 (观测) + t=H-1 (future)
        "video": ModalityConfig(
            delta_indices=[0, self.action_horizon - 1],
            modality_keys=self.video_keys,            # ["video.primary_image", "video.wrist_image"]
        ),
        "state": ModalityConfig(delta_indices=[0], modality_keys=self.state_keys),
        "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
        "language": ModalityConfig(delta_indices=[0], modality_keys=self.language_keys),
    }
```

只有 4 种标准 modality 类别，零 LeRobot core patch。**注**：双偏移意味着 `video.*` tensor 第 0 维变成时间维（shape `[2, H, W, C]`），`_unpack_lerobot_sample` 用 `[0]` 取观测帧、`[1]` 取 future 帧；wrist 视图的 t=H-1 帧不用（framework 只取 primary 的 future）。

**DataConfig.transform()** —— *codex v2 拍板修正*：[StateActionTransform (state_action.py:297)](../../starVLA/dataloader/gr00t_lerobot/transform/state_action.py#L297) 的契约是"`apply_to` 列入的 key 才走 transform；其中出现在 `normalization_modes` 里的 key 才被归一化"。把 pose 6D（构造保证 [-1,1]）和 cam_extrinsic（rigid，metric 单位）和 robot_obs 一起归一化是**数值污染**。`transform()` 必须显式分两段：

```python
def transform(self):
    return ComposedModalityTransform(transforms=[
        # ── action 路径 ──────────────────────────────────────────────
        StateActionToTensor(apply_to=self.action_keys),
        StateActionTransform(
            apply_to=self.action_keys,
            normalization_modes={k: "min_max" for k in self.action_keys[:-1]},  # 除 gripper 外都 min_max
            # NOTE: gripper 在 apply_to 里但不在 normalization_modes —— 不归一化
        ),
        # ── state 路径：robot_obs 走归一化，pose / cam 字段仅 ToTensor ─────────
        StateActionToTensor(apply_to=self.state_keys),                  # 全字段 ToTensor
        StateActionTransform(
            apply_to=["state.robot_obs"],                               # ⚠️ 只列 robot_obs
            normalization_modes={"state.robot_obs": "mean_std"},
            # ⚠️ pose / cam 字段已被 ToTensor，但不出现在第二个 Transform 的 apply_to，
            # 所以跳过归一化逻辑，保留原数值
        ),
    ])
```

PoseHead / `static_cam_extrinsic` 消费方拿到的就是未经污染的原始值，与原 uamvla JSONL 路径行为一致。

### 4.5 framework 端的 LeRobot key → framework key 映射（含 sidecar IO）

> **应对 codex Showstopper #3**：`_collate_aux` 必须透传 `input_ids`。framework 端的 `_unpack_lerobot_sample` 负责把 LeRobot key 转 framework key **且** 加载 sidecar。

`UamVLAOFT.__init__` 时从 `config.datasets.vla_data.data_root_dir` 拿 `dataset_root` 字符串，存为 `self.sidecar_root`。`_unpack_lerobot_sample` 是 instance method（codex Minor #2 提示 `_resolve_head_mask` 在旧代码是 module-level 函数，新代码统一用 instance method）。

映射表：

| LeRobot key | framework key | 处理 |
|---|---|---|
| `video.primary_image` (delta=[0, H-1]) → `[0]` 切片 | `image[0]: PIL` (primary) | tensor → PIL |
| `video.wrist_image` (delta=[0, H-1]) → `[0]` 切片 | `image[1]: PIL` (wrist) | tensor → PIL |
| `video.primary_image` (delta=[0, H-1]) → `[1]` 切片 | `image_future: Tensor` (C,H,W) | tensor → CHW；wrist 的 `[1]` 切片忽略 |
| `state` (33 维) | 按 `aux_state_slice` 切片 → `pose_gt`、`static_cam_extrinsic` | 6D rotation → 3×3 matrix（用 [`rotation_6d_to_matrix`](../../starVLA/model/modules/uamvla/components/pose/pose_utils.py)，**不是** v1 spec 里写错的 `rot6d_to_matrix`） |
| `action` (H, 7) | `action: Tensor` (H, 7) | 直接透传 |
| `lang` | `lang: str` | 直接透传 |
| `__trajectory_id` (int) | 用来反查 sidecar | `np.load(sidecar_root/point_clouds/<id>/<base>.npy)` → `point_cloud`；`Image.open(sidecar_root/image_targets/<id>.png)` → `image_target`（结果 LRU cached by `__trajectory_id`） |
| `__base_index` (int) | 同上 | 用于 point_cloud 文件名 |

framework 的 `forward(examples)` 在最前面调 `_unpack_lerobot_sample`，输出含 `image / lang / action / image_future / pose_gt / static_cam_extrinsic / point_cloud / image_target` 的标准 framework sample。

### 4.6 Phase 1 删除清单

- [`starVLA/dataloader/uamvla_dataset.py`](../../starVLA/dataloader/uamvla_dataset.py)（277 行）
- `tools/preprocess/base_preprocessor.py` 的 `compute_statistics` 路径（统计交给 LeRobot）
- `tools/statistics.py`（自定义 `DatasetStatistics` / `EmbodimentStats` 类）
- 数据目录 `datasets/uamvla_calvin/` 旧 JSONL 产物（首次重 preprocess 后）

### 4.7 Phase 1 保留清单

- `tools/preprocess/calvin_preprocessor.py` 的**核心提取逻辑**（npz 读取、scene_obs 解析、target 物体识别、点云生成）—— 只换"写出格式"那一段
- `tools/preprocess/target_object_resolver.py`（LLM 辅助目标物体识别）
- `starVLA/utils/geometry.py`、`rotation.py`、`point_cloud.py`（被 framework 复用）
- `tools/preprocess/libero_preprocessor.py`（本 spec 不动；未来 LIBERO 上线时按本设计扩展）

### 4.8 Phase 1 验收（弱化，应对 codex Significant gap #1）

> **变更**：v1 要求"老 UamVLA framework 通过 adapter shim 跑 50 步"。这个 shim 需要重建 canonical_state / action_mask / view_names 等旧契约（codex 指出 shim 比想象的复杂）。**v2 取消 adapter shim 验收**。

新验收项：

- [ ] 重 preprocess 完一份 CALVIN ABCD_D 训练集（写出 parquet + sidecar + meta + camera_params.json）
- [ ] 新 LeRobot dataset 通过 [`lerobot_datasets.py` __main__](../../starVLA/dataloader/lerobot_datasets.py#L102-L139) 的 batch iteration smoke test
- [ ] sample dict 含 `__trajectory_id`、`__base_index` 两个 int（patch 1 生效）
- [ ] `state` modality 33 维（15 robot_obs + 9 pose + 9 cam_extrinsic）
- [ ] sidecar 文件存在：`point_clouds/<traj>/<base>.npy` 和 `image_targets/<traj>.png` 全部可加载
- [ ] **smoke test 命令在 Claude 跑之前先 `nvidia-smi` 检查空闲卡，按 [CLAUDE.md GPU 使用规则](../../CLAUDE.md#GPU-使用规则) 执行**

## 5. Phase 2：UamVLAOFT framework

### 5.1 VLM ckpt 加载（无须预处理）

QwenOFT 用 emoji 🔍 当 action query placeholder，**不需要往 vocab 注入新 token**，所以可以直接加载 HF Hub 原版 `Qwen/Qwen3-VL-8B-Instruct`，不需要烧 `-Action` 后缀 ckpt。

**唯一需要 verify 的不变量**：Qwen3-VL tokenizer 把 🔍 切成单 token。在 `UamVLAOFT.__init__` 末尾加 sanity check：

```python
sanity_ids = self.qwen_vl_interface.processor.tokenizer(
    "🔍" * self.chunk_len, add_special_tokens=False,
)["input_ids"]
assert len(sanity_ids) == self.chunk_len, (
    f"🔍 must tokenize to 1 token each, got {len(sanity_ids)} for chunk_len={self.chunk_len}. "
    f"Check Qwen3-VL tokenizer version or replace 🔍 with a registered special token."
)
```

### 5.2 UamVLAOFT framework

**新文件**：`starVLA/model/framework/VLM4A/UamVLAOFT.py`

```python
@FRAMEWORK_REGISTRY.register("UamVLAOFT")
class UamVLAOFT(Qwenvl_OFT):
    """QwenOFT + pose/future/recon perception aux heads.

    Inherits L1 action regression + Qwen3-VL backbone from QwenOFT;
    adds three aux heads consuming hidden_states[-1] alongside the L1 action loss.
    No state encoder — robot state is inferred from images only.
    Sidecar IO (point_cloud / image_target) happens in _unpack_lerobot_sample.
    """

    def __init__(self, config):
        super().__init__(config)   # 父类构造 _QWen3_VL_Interface + L1RegressionActionHead

        # Sanity check: 🔍 emoji 切成单 token（§5.1）
        sanity_ids = self.qwen_vl_interface.processor.tokenizer(
            "🔍" * self.chunk_len, add_special_tokens=False,
        )["input_ids"]
        assert len(sanity_ids) == self.chunk_len, (
            f"🔍 must tokenize to 1 token each, got {len(sanity_ids)} for chunk_len={self.chunk_len}"
        )

        # Sidecar root，从 config 注入（§4.5）
        self.sidecar_root = Path(self.config.datasets.vla_data.data_root_dir)
        self._image_target_cache: dict[int, torch.Tensor] = {}
        self.aux_state_slice = self.config.datasets.vla_data.get(
            "aux_state_slice", {
                "target_pose_rot6d": [15, 21],
                "target_pose_trans": [21, 24],
                "static_cam_rot6d":  [24, 30],
                "static_cam_trans":  [30, 33],
            }
        )

        # Aux heads（pose/future/recon，无 action_head）
        # ...（同 v1 spec §5.2 的构造逻辑，省略）

    def forward(self, examples, **kwargs):
        # ① LeRobot key → framework key + sidecar IO（§4.5）
        examples = [self._unpack_lerobot_sample(e) for e in examples]

        batch_images = [self._force_resize_640(e["image"]) for e in examples]
        instructions = [e["lang"] for e in examples]
        gt_actions = [e["action"] for e in examples]

        # ② QwenOFT 风格 prompt suffix
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [s + prompt_suffix for s in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs, output_hidden_states=True, return_dict=True,
            )
        hidden = qwenvl_outputs.hidden_states[-1]   # (B, L, H)

        # ③ Spatial reader 不变量校验（应对 codex Significant gap #2）
        input_ids = qwen_inputs["input_ids"]
        ppv = 400
        num_views = len(examples[0]["image"])
        img_token_count = (input_ids == self.qwen_vl_interface.image_token_id).sum(dim=1)
        if not (img_token_count == ppv * num_views).all():
            raise RuntimeError(
                f"image_pad token count mismatch: expected {ppv * num_views} per sample, "
                f"got {img_token_count.tolist()}. Check _force_resize_640 + Qwen3VLProcessor."
            )

        # ④ Action loss：L1 regression on 🔍 queries
        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                hidden, input_ids, action_token_id=self.action_token_id,
            )
            pred_actions = self.action_model.predict_action(action_queries)
            gt_actions_t = torch.tensor(np.array(gt_actions),
                                         device=pred_actions.device, dtype=pred_actions.dtype)
            gt_actions_t = gt_actions_t[:, -self.action_horizon:, :]
            total = self.l1_loss(pred_actions, gt_actions_t)
        log_metrics = {"action_loss_l1": total.detach()}

        # ⑤ Aux head losses（batch_dict 必须含 input_ids，应对 codex Showstopper #3）
        batch_dict = self._collate_aux(examples, qwen_inputs)
        assert "input_ids" in batch_dict, "future_head / recon_head require batch['input_ids']"
        for name, head in self.aux_heads.items():
            mask = self._resolve_head_mask(name, batch_dict, hidden.shape[0], hidden.device)
            out = head.compute_loss(hidden, batch_dict, mask=mask)
            if out.loss is not None:
                total = total + out.loss
                log_metrics[f"{name}_loss"] = out.loss.detach()
            log_metrics.update({f"{name}_{k}": v for k, v in out.metrics.items()})

        return {"action_loss": total, **log_metrics}

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        """Override 父类 predict_action：必须用同一个 _force_resize_640 保持
        训练 / 推理图像尺寸一致（应对 codex Significant gap #3）。
        """
        if not isinstance(examples, list):
            examples = [examples]
        examples = [self._unpack_lerobot_sample(e) if "__trajectory_id" in e else e
                    for e in examples]
        batch_images = [self._force_resize_640(e["image"]) for e in examples]
        instructions = [e["lang"] for e in examples]

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [s + prompt_suffix for s in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs, output_hidden_states=True, return_dict=True,
            )
        hidden = qwenvl_outputs.hidden_states[-1]
        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                hidden, qwen_inputs["input_ids"], action_token_id=self.action_token_id,
            )
            pred_actions = self.action_model.predict_action(action_queries)
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
```

私有方法契约：

- **`_unpack_lerobot_sample(sample)`** → 输入 LeRobot 风格 dict（含 `__trajectory_id`/`__base_index`/`state`/...），输出 framework 风格 dict（含 `image`/`lang`/`action`/`pose_gt`/`static_cam_extrinsic`/`point_cloud`/`image_target`/`image_future`）。sidecar IO 在此处发生
- **`_collate_aux(examples, qwen_inputs)`** → 复用 [`collator_helpers.stack_optional_tensor_fields`](../../starVLA/model/modules/uamvla/collator_helpers.py) / `stack_pose_gt` / `stack_static_cam_extrinsic`，输出 batch_dict (含 `image_target`、`image_future`、`point_cloud`、`pose_gt`、`static_cam_extrinsic` 及对应 mask) **+ `input_ids`**（codex Showstopper #3 修复）
- **`_resolve_head_mask`** → 把旧 [UamVLA.py:144](../../starVLA/model/framework/VLM4A/UamVLA.py#L144) 的 module-level 函数 promote 成 UamVLAOFT 的 instance method（codex Minor #2 修复）
- **`_force_resize_640(image_list)`** → 把 List[PIL] 里每张图 resize 到 640×640。**训练和推理都调它**（codex Significant gap #3 修复）

### 5.3 配置文件

`starVLA/config/training/uamvla_oft_calvin_abcd.yaml`：

```yaml
framework:
  name: UamVLAOFT
  qwenvl:
    base_vlm: Qwen/Qwen3-VL-8B-Instruct       # 原版，无须 -Action 后缀
    attn_implementation: flash_attention_2
  action_model:
    action_model_type: MLP
    action_dim: 7
    action_hidden_dim: 3584                    # 8B Qwen3-VL hidden_size（运行时由 framework 覆盖）
    future_action_window_size: 7               # action_horizon = 8
    past_action_window_size: 0
  obs_image_size: [640, 640]                    # ⚠️ 应对 codex SG #3：父类 predict_action 看这个 key 才会 resize；我们 override 了 predict_action 但保留这条以防万一
  vae:
    path: ckpt/pretrained_vae
  aux_heads:
    pose:
      enabled: true
      loss_weight: 0.5
      lr: 1.0e-4
      pose_mode: rot_matrix
      sde_mode: ve
      num_queries: 4
      semantic_dim: 512
      sampling_steps: 500
    future:
      enabled: true
      loss_weight: 0.1
      lr: 1.0e-4
      view_idx: 0
      target_resize: 320
      denoiser_depth: 3
      denoiser_embed_dim: 1024
      repeat_factor: 4
    recon:
      enabled: true
      loss_weight: 0.1
      lr: 1.0e-4
      view_idx: 0
      target_resize: 320
      denoiser_depth: 3
      denoiser_embed_dim: 1024
      repeat_factor: 4

datasets:
  vla_data:
    dataset_py: lerobot_datasets               # 走主线 dataloader
    data_root_dir: playground/Datasets/UAMVLA_LEROBOT_CALVIN_ABCD
    data_mix: uamvla_calvin_abcd
    action_type: delta_qpos
    per_device_batch_size: 8
    obs: ["video.primary_image", "video.wrist_image"]
    aux_state_slice:
      target_pose_rot6d: [15, 21]
      target_pose_trans: [21, 24]
      static_cam_rot6d:  [24, 30]
      static_cam_trans:  [30, 33]
```

### 5.4 Phase 2 删除清单

**Framework 层**：
- [`starVLA/model/framework/VLM4A/UamVLA.py`](../../starVLA/model/framework/VLM4A/UamVLA.py)（838 行）—— 被 `UamVLAOFT` 取代

**Backbone 层**：
- [`starVLA/model/modules/uamvla/backbone_wrapper.py`](../../starVLA/model/modules/uamvla/backbone_wrapper.py) —— 被 `_QWen3_VL_Interface` 取代

**State encoder 层**：
- [`starVLA/model/modules/uamvla/state_encoder/`](../../starVLA/model/modules/uamvla/state_encoder/) 整个目录（含 `modular_state_encoder.py`、`limb_encoders.py`、`special_tokens.py`）

**Data / encoding 层**：
- [`starVLA/model/modules/uamvla/data/embodiment_adapter.py`](../../starVLA/model/modules/uamvla/data/embodiment_adapter.py)
- [`starVLA/model/modules/uamvla/data/embodiment_registry.py`](../../starVLA/model/modules/uamvla/data/embodiment_registry.py)
- [`starVLA/model/modules/uamvla/data/state_normalizer.py`](../../starVLA/model/modules/uamvla/data/state_normalizer.py)
- [`starVLA/model/modules/uamvla/data/_normalizer_lite.py`](../../starVLA/model/modules/uamvla/data/_normalizer_lite.py)
- [`starVLA/model/modules/uamvla/data/action_tokenizer.py`](../../starVLA/model/modules/uamvla/data/action_tokenizer.py)
- [`starVLA/model/modules/uamvla/data/chat_template.py`](../../starVLA/model/modules/uamvla/data/chat_template.py)

**Inference 层**：
- [`starVLA/model/modules/uamvla/inference/action_logits_processor.py`](../../starVLA/model/modules/uamvla/inference/action_logits_processor.py)

**Aux head**：
- [`starVLA/model/modules/uamvla/aux_heads/action_head.py`](../../starVLA/model/modules/uamvla/aux_heads/action_head.py) —— action 改由 QwenOFT 的 `L1RegressionActionHead` 负责

**配置**：
- [`starVLA/config/training/uamvla_libero.yaml`](../../starVLA/config/training/uamvla_libero.yaml)、[`uamvla_calvin.yaml`](../../starVLA/config/training/uamvla_calvin.yaml)、[`uamvla_calvin_abcd.yaml`](../../starVLA/config/training/uamvla_calvin_abcd.yaml) —— 被 `uamvla_oft_calvin_abcd.yaml` 取代

**Eval / deployment 旁路**（应对 codex Significant gap #4）：
- [`examples/calvin/eval_files/run_policy_server_uamvla_abcd.sh`](../../examples/calvin/eval_files/run_policy_server_uamvla_abcd.sh)
- [`examples/calvin/eval_files/run_uamvla_calvin_abcd_eval.sh`](../../examples/calvin/eval_files/run_uamvla_calvin_abcd_eval.sh)

替换为 `run_policy_server_uamvla_oft.sh` / `run_uamvla_oft_calvin_eval.sh`。**旧 UamVLA ckpt 在 Phase 2 完成后不可加载**（用户已确认接受）。

### 5.5 Phase 2 保留清单

- [`starVLA/model/modules/uamvla/aux_heads/base.py`](../../starVLA/model/modules/uamvla/aux_heads/base.py) `HeadOutput` + `AuxHead` ABC + `get_dummy_loss`
- [`pose_head.py`](../../starVLA/model/modules/uamvla/aux_heads/pose_head.py) / [`future_head.py`](../../starVLA/model/modules/uamvla/aux_heads/future_head.py) / [`recon_head.py`](../../starVLA/model/modules/uamvla/aux_heads/recon_head.py)
- [`components/pose/`](../../starVLA/model/modules/uamvla/components/pose/) PointNet2 + GenPose2 + SDE + score_net + sampler
- [`components/denoiser/`](../../starVLA/model/modules/uamvla/components/denoiser/) DiT scheduler / common
- [`components/pixel_decoder/vae.py`](../../starVLA/model/modules/uamvla/components/pixel_decoder/vae.py)
- [`components/{query_reader,spatial_reader,task_adapter}.py`](../../starVLA/model/modules/uamvla/components/)
- [`collator_helpers.py`](../../starVLA/model/modules/uamvla/collator_helpers.py) 的 `stack_optional_tensor_fields` / `stack_pose_gt` / `stack_static_cam_extrinsic`
- [`starVLA/utils/{geometry,rotation,point_cloud}.py`](../../starVLA/utils/)

### 5.6 不再需要的工程清理

QwenFast 路线本来要做的两件顺手事在 QwenOFT 路线下**直接消失**：

- ~~烧 `Qwen3-VL-8B-Instruct-Action` ckpt~~（OFT 不需要 vocab 注入）
- ~~把 `_QWen3_VL_Interface` 的 `_ACTION_TOKEN_MIN/MAX` 改为动态查询~~（OFT 不走那段 if 分支）

### 5.7 Phase 2 验收（SR 硬目标）

> **变更（用户决策 #2）**：v1 写 "本 spec 不优化 SR / 只打通管线"，与 SR ≥ 90% 自相矛盾（codex Significant gap #5）。**v2 修正 §2.1 scope 为"达 OpenVLA-OFT 级 SR"**，SR 是硬目标，loss 权重 / LR / epoch 数都可以为它服务。

- [ ] **CALVIN ABCD_D 训练**：UamVLAOFT + Qwen3-VL-8B-Instruct + 三个 aux head，达 OpenVLA-OFT 论文报告的 CALVIN ABCD_D 性能区间（average sequence length / SR）。
- [ ] **次门槛**：单任务 SR ≥ 80%；< 80% 视为方案失败需要重新设计
- [ ] **中间 baseline 验收**（codex v2 建议，应对 OpenVLA-OFT 用 OpenVLA-7B 而我们用 Qwen3-VL-8B-Instruct 的预训练差异）：先跑一份**纯 QwenOFT（不挂 aux head，UamVLAOFT 把 `aux_heads` 全 disable）** 训练，记录其 CALVIN ABCD_D SR 作为 baseline B；UamVLAOFT 全配置版本 SR 必须 ≥ B（aux head 不能让 action SR 退化）。**baseline 跑通本身就是方案的可行性证据**，与达 OpenVLA-OFT SR 是独立验收
- [ ] **aux head 非 dummy-loss 步级断言**（codex v2 建议）：在一个固定 deterministic mini-batch 上跑 forward，分别 assert `pose_head_loss < pose_head.get_dummy_loss()`、`future_head_loss < future_head.get_dummy_loss()`、`recon_head_loss < recon_head.get_dummy_loss()` —— 防止某个 head 大部分 step 走 dummy_loss 路径但被 epoch-level "loss 下降" 验收漏过
- [ ] 单次 forward + backward 无 OOM（80GB H100 + batch_size=8 + bf16 + DeepSpeed ZeRO-2）—— **smoke test 严格遵守 [CLAUDE.md GPU 使用规则](../../CLAUDE.md#GPU-使用规则)**
- [ ] 🔍 emoji sanity check 通过（`len(tokenizer("🔍" * 8))` == 8）
- [ ] `(input_ids == image_token_id).sum(1) == 800` (= ppv 400 × 2 views) 校验通过

## 6. 项目布局变更

```
UniamVLA/
├── starVLA/
│   ├── model/framework/VLM4A/
│   │   ├── UamVLA.py                 [DELETE]
│   │   └── UamVLAOFT.py              [NEW] 继承 Qwenvl_OFT
│   ├── model/modules/uamvla/
│   │   ├── backbone_wrapper.py       [DELETE]
│   │   ├── state_encoder/            [DELETE 整个目录]
│   │   ├── inference/                [DELETE 整个目录]
│   │   ├── data/                     [DELETE 大部分文件，保留 __init__.py 即可]
│   │   ├── aux_heads/
│   │   │   ├── action_head.py        [DELETE]
│   │   │   ├── pose_head.py          [KEEP]
│   │   │   ├── future_head.py        [KEEP]
│   │   │   ├── recon_head.py         [KEEP]
│   │   │   └── base.py               [KEEP]
│   │   ├── components/               [KEEP 整个目录]
│   │   └── collator_helpers.py       [KEEP]
│   ├── dataloader/
│   │   ├── uamvla_dataset.py         [DELETE]
│   │   ├── lerobot_datasets.py       [UNCHANGED]
│   │   └── gr00t_lerobot/
│   │       ├── datasets.py           [PATCH] _pack_sample 透传 __trajectory_id / __base_index
│   │       └── transform/            [UNCHANGED]（不新增 PointCloudLoadTransform / ImageTargetByTaskTransform）
│   └── config/training/
│       ├── uamvla_libero.yaml        [DELETE]
│       ├── uamvla_calvin.yaml        [DELETE]
│       ├── uamvla_calvin_abcd.yaml   [DELETE]
│       └── uamvla_oft_calvin_abcd.yaml  [NEW]
├── examples/calvin/
│   ├── train_files/
│   │   ├── data_registry/
│   │   │   ├── data_config.py        [NEW] UamVLACalvinDataConfig
│   │   │   └── modality.json         [NEW]
│   │   ├── run_uamvla_calvin_train.sh      [DELETE]
│   │   ├── run_uamvla_calvin_abcd_train.sh [DELETE]
│   │   └── run_uamvla_oft_train.sh         [NEW]
│   └── eval_files/
│       ├── run_policy_server_uamvla_abcd.sh   [DELETE]
│       ├── run_uamvla_calvin_abcd_eval.sh     [DELETE]
│       ├── run_policy_server_uamvla_oft.sh    [NEW]
│       └── run_uamvla_oft_calvin_eval.sh      [NEW]
├── tools/preprocess/
│   ├── libero_preprocessor.py        [UNCHANGED]（本 spec 不涵盖 LIBERO）
│   ├── calvin_preprocessor.py        [REWRITE 输出格式：JSONL → LeRobot parquet + sidecar + camera_params.json]
│   ├── base_preprocessor.py          [DELETE compute_statistics 路径]
│   └── target_object_resolver.py     [UNCHANGED]
└── docs/superpowers/specs/
    └── 2026-05-13-uamvla-on-qwenoft-design.md  [本文件]
```

## 7. 风险与未决项

| # | 风险 | 缓解 |
|---|---|---|
| 1 | **🔍 emoji tokenization 稳定性**：Qwen3-VL tokenizer 必须把 🔍 切成单 token，否则 `_gather_action_token_embeddings` 的 `topk(k=chunk_len)` 假设错位，aux head 也会拿错 query 位置 | `__init__` 末尾加 sanity assert（§5.1）；transformers 版本升级时这个 assert 会立刻报警，不会静默错算 |
| 2 | ~~`video_future` modality 类别识别~~ | **已 codex v2 拍板修复**：合并到单个 `"video"` ModalityConfig + `delta_indices=[0, H-1]` 双偏移；不再有 `video_future` 类别，不动 LeRobot core（详见 §4.4） |
| 2b | **State normalization 污染 pose / cam_extrinsic 字段** | 已 codex v2 拍板修复：DataConfig `transform()` 把 `state.robot_obs` 单独走第二个 `StateActionTransform`，pose / cam 字段不进 `apply_to` 第二段 → 跳过归一化（详见 §4.4） |
| 2c | **`delta_indices=[0, H-1]` 在 episode 末尾 silent padding**：[datasets.py:1611-1612](../../starVLA/dataloader/gr00t_lerobot/datasets.py#L1611) `np.minimum(step_indices, trajectory_lengths-1)` 把超界 index clamp 到最后一帧。episode 末尾 H-1 步内 `image_future` 是最后一帧重复，不是真正 H 步后未来 | 接受：action 的 delta_indices 也是这套机制，uamvla 旧实现已容忍；如发现影响 future head 收敛，按 [calvin_preprocessor 现有 `delete_pause_frame` 机制](../../starVLA/dataloader/gr00t_lerobot/datasets.py) 在 episode 末尾过滤掉 (episode_length - action_horizon + 1, episode_length) 这几步 base_index |
| 3 | **`_pack_sample` patch 影响其他 framework**：透传 `__trajectory_id` / `__base_index` 两个新 int key | 用 `__` 前缀避免冲突；其他 framework 不读就忽略；patch 是加字段不删字段，零兼容性破坏 |
| 4 | **L1 regression 精度 vs per-dim 分箱**：OFT 是连续回归（OpenVLA-OFT 论文证明 LIBERO 上优于离散 token） | 接受，预期比 FAST 路线精度更好 |
| 5 | **抛弃 state encoder → SR 损失**：robot state 信息只能靠图像感知 | 接受，是研究身份转变的代价（§2.3） |
| 6 | **强制 image resize 640×640**：训练 / 推理路径都过 `_force_resize_640`，padded 图传给 Qwen3VLProcessor 必须产生 `image_grid_thw=(1,40,40)` | 在 forward 里加 image_pad token 计数校验（§5.2 ③）；首次跑挂掉立刻能定位 |
| 7 | **aux head loss 权重需要重调**：L1 action loss 量级（典型 ~0.1）与旧 ActionHead CE（典型 ~1.0）不同，与 pose/future/recon loss 的相对量级要重新校准 | Phase 2 验收阶段 ablation；§5.7 把 SR 当硬目标后，loss 权重调参是允许的范围内的优化 |
| 8 | **CALVIN preprocessor 改写成本**：[`calvin_preprocessor.py`](../../tools/preprocess/calvin_preprocessor.py) 当前已处理 scene/task/target object resolution；改写时不能丢这些识别逻辑 | 改写策略：保留 npz 读取 + scene 解析 + target 识别 + 点云生成；只换"写出格式"那一段（JSONL → parquet + sidecar） |
| 9 | **smoke test 触发 GPU 占用冲突**：任何 GPU 命令必须遵守 [CLAUDE.md GPU 使用规则](../../CLAUDE.md#GPU-使用规则) | Phase 1/2 实施时 Claude 跑命令前必须 `nvidia-smi`，没空闲卡就停下通知用户 |
| 10 | **旧 UamVLA ckpt 不可加载**：删 `UamVLA.py` 同时删 eval server 入口（§5.4） | 用户已确认接受；如果未来需要回到旧路径再单独从 git 历史恢复 |

## 8. 验收标准

### 8.1 Phase 1（CALVIN ABCD_D 数据迁移）

- [ ] CALVIN ABCD_D 重 preprocess 完整无报错，产出符合 §4.3 布局
- [ ] 新 LeRobot dataset 通过 [`lerobot_datasets.py` __main__](../../starVLA/dataloader/lerobot_datasets.py#L102-L139) 的 batch iteration smoke test
- [ ] sample dict 含 `__trajectory_id`、`__base_index`、`image`、`lang`、`action`、`state`(33 维) 全部字段
- [ ] `meta/stats_gr00t.json` 自动生成成功
- [ ] sidecar 文件全部可加载：`point_clouds/<traj>/<base>.npy` shape == (1024, 3)；`image_targets/<traj>.png` 可 PIL 打开
- [ ] `camera_params.json` 存在且 PoseHead viz 能从中读出 fx/fy/cx/cy

### 8.2 Phase 2（UamVLAOFT）

- [ ] 🔍 emoji sanity check 通过（`len(tokenizer("🔍" * 8))` == 8）
- [ ] `(input_ids == image_token_id).sum(1) == 800` 校验通过（2 views × ppv 400）
- [ ] `_collate_aux` 输出 batch_dict 含 `input_ids`（future/recon head 不会因为缺 input_ids 崩）
- [ ] `UamVLAOFT.forward` 在 CALVIN ABCD_D 上单 batch forward + backward 无 OOM、loss 数值有限
- [ ] `UamVLAOFT.predict_action` 输出 shape (B, H, 7)，图像与训练同尺寸 640×640
- [ ] CALVIN ABCD_D 训练完整 epoch 后，pose/future/recon 三个 head loss 都在下降
- [ ] **CALVIN ABCD_D SR ≥ OpenVLA-OFT 报告区间**（硬目标；具体数值在 implementation plan 阶段从 OpenVLA-OFT repo 取最新值固化）；次门槛 80%
- [ ] **中间 baseline**：纯 QwenOFT（不挂 aux head）训练的 CALVIN ABCD_D SR 作为 baseline B；UamVLAOFT 全配置版本 SR ≥ B
- [ ] **aux head 步级断言**：固定 deterministic mini-batch 上 pose/future/recon 三个 head 的 loss 都严格小于其 `get_dummy_loss()` 返回值

### 8.3 集成度指标

- [ ] `starVLA/dataloader/uamvla_dataset.py` 删除
- [ ] `starVLA/model/framework/VLM4A/UamVLA.py` 删除
- [ ] `starVLA/model/modules/uamvla/backbone_wrapper.py` 删除
- [ ] `starVLA/model/modules/uamvla/state_encoder/` 删除
- [ ] `starVLA/model/modules/uamvla/data/{chat_template,action_tokenizer,embodiment_adapter,...}.py` 删除
- [ ] `starVLA/model/modules/uamvla/inference/` 删除
- [ ] `tools/preprocess/calvin_preprocessor.py` 输出 LeRobot parquet（不再产 JSONL）
- [ ] `UamVLAOFT` framework 通过 `from starVLA.model.framework.VLM4A.UamVLAOFT import UamVLAOFT` 可导入
- [ ] CI / 测试套件全绿（旧 uamvla 单测随删除文件一起移除）

## 9. 范围外

- **LIBERO 数据集预处理 / 训练 / 评测**：本地无 LIBERO 数据，本 spec 不涵盖 [`tools/preprocess/libero_preprocessor.py`](../../tools/preprocess/libero_preprocessor.py) 的改写、`examples/LIBERO/` 下任何文件的修改、`uamvla_libero.yaml` 之外的 LIBERO 配置
- VLM co-training（`supports_training_tag("vlm")` 实现）
- 异构 embodiment co-training（同 batch 混 CALVIN + 其他 embodiment）
- Cross-embodiment 24D 统一 action 空间（本 spec 只用 7D Franka action）
- 把 🔍 placeholder 换成正式的 special token（注册到 vocab 而非用 emoji），这属于工程改进，不影响算法

## 10. 后续工作

Phase 2 验收通过后的潜在方向（**非本 spec 承诺**）：

1. **LIBERO 上线**：下载 LIBERO 数据后开新 spec，按本设计扩展 `tools/preprocess/libero_preprocessor.py` 和 `examples/LIBERO/train_files/data_registry/`
2. 拿到稳定 baseline 后，回头做 ablation：UamVLAOFT vs 原 uamvla（带 state encoder + per-dim 分箱），量化 state encoder 抛弃对 CALVIN / LIBERO SR 的影响
3. 把 🔍 emoji placeholder 替换为正式注册的 `<|action_query|>` special token（用 `add_qwen_special_tokens` 工具一次性烧），更稳健、更显式
4. 把 `_QWen3_VL_Interface` 提取出更标准化的 `output_hidden_states` 返回路径接口，让其他 framework（QwenPI / QwenGR00T）也能像 UamVLAOFT 一样挂 aux head
