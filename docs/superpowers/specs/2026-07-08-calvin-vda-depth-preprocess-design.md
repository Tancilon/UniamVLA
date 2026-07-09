# CALVIN LeRobot 数据集 Video-Depth-Anything 深度预处理设计

日期：2026-07-08
状态：已确认（设计评审通过，待实现）

## 1. 目标与范围

用 Video-Depth-Anything（VDA，relative 模式，ViT-L）对
`datasets/task_ABC_D_scene_D_lerobot` 的静态相机 `image` 全部 5124 个视频做
离线深度推理，结果保存到该数据集的 `depth/` 目录，供后续 `DepthHead`
辅助监督（auxiliary supervision）使用。

**范围内**：批量推理脚本、启动器、输出格式与元数据、对齐校验。
**范围外**：dataloader / 训练侧接线（后续独立任务）；腕部相机 `wrist_image`
（脚本通过 `--camera` 参数保留扩展能力，但本次不跑）。

### 背景决策记录

- 原始 CALVIN 数据（`datasets/calvin/task_ABC_D/`）自带仿真器 GT 度量深度
  （`depth_static`/`depth_gripper`），但经讨论**明确选择 VDA 估计深度**而非 GT。
- 现有消费方 `starVLA/model/modules/uamvla/aux_heads/depth_head.py` 的
  `relative_inverse_depth_per_frame` 对深度取逆后**逐帧归一化到 [0,1]**，
  即 scale-invariant 监督——度量尺度会被丢弃。

## 2. 核心技术决策

| 决策点 | 结论 | 理由 |
|---|---|---|
| relative vs metric | **relative** | 监督目标逐帧归一化，metric 尺度信息被丢弃；Metric-VDA 在真实数据上标定，对仿真场景绝对尺度不可信；relative 相对结构精度与时序一致性更好；relative 权重已下载 |
| 权重 | **`video_depth_anything_vitl.pth`**（relative ViT-L, 381.8M） | 一次性离线预处理，质量优先；fp16 下 30.9 万帧单卡数小时可完成 |
| 相机 | **仅静态 `image`**（200×200, 5124 个视频） | 腕部 84×84 分辨率极低、近距快速运动，VDA 质量存疑，作监督信号噪声风险大 |
| 保存格式 | **逐 episode `np.savez_compressed`，float16** | 视频编码（8-bit 有损）会量化/污染监督目标；npz 直接 `np.load`→tensor；体量约 24GB 可接受 |
| 输出语义 | **relative inverse depth（disparity），模型原始输出，未归一化** | 见 §4 语义警告 |

## 3. 推理参数定案

| 参数 | 取值 | 理由 |
|---|---|---|
| `input_size` | **518** | 模型原生训练分辨率（DINOv2 ViT patch=14，518=14×37）；200×200 源被上采样至 518 推理再缩回，是官方对低分辨率输入的标准处理；调低会偏离训练分辨率导致质量下降 |
| 帧采样 | **全帧**（`target_fps=-1`、`max_len=-1` 语义，脚本内硬编码不暴露） | 深度必须与 RGB 逐帧 1:1 对齐，任何抽帧都会破坏对齐 |
| 精度 | **fp16**（autocast，默认） | 监督目标进 head 前会归一化，fp16 精度足够；速度约 2 倍、显存减半 |
| 视频解码 | **pyav（自行解码，不用 VDA 的 `read_video_frames`）** | ① 数据集为 AV1 编码，decord 对 AV1 支持差；② LeRobot loader 训练时用 pyav/torchcodec，预处理与训练解码链一致可保证帧级对齐 |
| 时序窗口 | 内部常量 `INFER_LEN=32, OVERLAP=10`，不改 | episode 34~65 帧（均值 60.3），2~3 个滑窗，内部复制末帧补齐；脚本保存前 `depths[:T]` 截断防御 |
| `max_res` | 不暴露 | 200×200 永不触发缩放 |

## 4. 脚本设计

### 位置与文件

- `examples/calvin/preprocess_files/preprocess_depth_vda.py` —— 批量推理主脚本
- `examples/calvin/preprocess_files/run_preprocess_depth.sh` —— 薄启动器，
  遵守仓库约定：内部不写 tee 落盘，末尾 `"$@"` 透传

通过 `sys.path` 引入 `third_party/Video-Depth-Anything`，**不改动 third_party
任何文件**。

### CLI 参数

```
--dataset_root   datasets/task_ABC_D_scene_D_lerobot
--camera         image        # 默认 image；将来可跑 wrist_image
--encoder        vitl         # vits 仅作快速调试
--input_size     518
--checkpoint     third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth
--shard_id 0 --num_shards 1   # 多卡分片：按 episode 排序后 idx % num_shards
--overwrite                   # 默认关：跳过已存在 npz → 断点续跑
--verify_only                 # 只做全量对齐校验，不推理
```

### 数据流

对每个视频（模型仅加载一次）：

1. pyav 解码全部帧 → `(T, 200, 200, 3)` uint8 RGB
2. `infer_video_depth(frames, fps=10, input_size=518, fp32=False)`
   → `(T', 200, 200)` float32
3. `depths[:T]` 截断 → 转 float16
4. `np.savez_compressed(out_path, depths=...)`

### 输出布局（镜像 videos/ 结构）

```
depth/
  chunk-{XXX}/image/episode_{XXXXXX}.npz   # key "depths": float16, (T, 200, 200)
  meta.json                                # 生产者元数据
  failures_{shard_id}.txt                  # 失败清单（如有）
```

`meta.json` 字段：模型名与权重文件、encoder、input_size、精度、
**语义 = relative inverse depth（disparity，未归一化模型原始输出）**、
dtype、生成日期、生产脚本路径。

### ⚠️ 语义警告（接线时必读）

VDA relative 模式输出**已经是逆深度**。现有
`DepthHead.relative_inverse_depth_per_frame` 假设输入是深度并自行取 `1/d`。
后续 dataloader/target 接线时必须据 `meta.json` 标记**跳过取逆**，
否则监督目标被倒两次，方向完全错误。

## 5. 校验与错误处理

- **逐视频**：推理后 assert 深度帧数 == 解码 RGB 帧数，不匹配记为失败。
- **失败不中断**：单视频异常写入 `depth/failures_{shard_id}.txt` 并继续，
  结束打印汇总。
- **全量校验**（`--verify_only`）：读 `meta/episodes.jsonl` 每 episode 的
  `length`，与对应 npz 的帧数逐一比对，报告缺失/不匹配清单。

## 6. 运行计划

1. **Smoke test 先行**：单卡跑 2~3 个视频，确认 ① pyav 能完整解出 AV1 帧且
   帧数与 `episodes.jsonl` 一致；② npz 内容、数值范围、单文件体量符合预期；
   ③ 对 1 个 episode 额外存可视化 mp4（复用 VDA `save_video(is_depths=True)`）
   人工目检。
2. **全量运行**：按当时 `nvidia-smi` 空闲卡数决定 `--num_shards`，每卡一个
   分片进程，`CUDA_VISIBLE_DEVICES=N` 显式绑卡（遵守 CLAUDE.md GPU 规则）。
3. 日志由运行命令 `tee` 到 `logs/`。
4. 预估：约 24GB 产物（压缩前），vitl fp16 单卡数小时；磁盘余量 19T，无压力。

## 7. 测试策略

- Smoke test（见上）覆盖解码、对齐、格式、目检。
- 全量结束跑 `--verify_only` 出对齐报告，缺失/不匹配为零才算完成。
- 断点续跑验证：中断后重跑，确认已完成 episode 被跳过、总数最终对上。
