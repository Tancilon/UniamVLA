# Qwen3.5 支持程度分析报告

> 更新时间：2026-08-31

---

## 一、当前支持状态总览

### ✅ 已实现的组件

1. **VLM Interface 实现完成**
   - 文件：`starVLA/model/modules/vlm/QWen3_5.py`
   - 类名：`_QWen3_5_VL_Interface`
   - 状态：✅ 完整实现
   - 创建者：Shijie LIAN (HUST) & Jinhui YE (HKUST), 2026

2. **模型自动路由**
   - 文件：`starVLA/model/modules/vlm/__init__.py`
   - 逻辑：`if "Qwen3.5" in vlm_name` → 加载 `_QWen3_5_VL_Interface`
   - 状态：✅ 已集成

3. **配置文件支持**
   - GIA Calvin: `examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml`
   - GIA RoboTwin: `examples/Robotwin/train_files/run_uamgr00t_GIA_robotwin.yaml`
   - 状态：✅ 已配置 `base_vlm: ./ckpt/Qwen3.5-4B`

---

## 二、核心实现细节

### 2.1 _QWen3_5_VL_Interface 关键特性

**基础架构**：
```python
class _QWen3_5_VL_Interface(nn.Module):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        # 默认模型: Qwen/Qwen3.5-VL-4B-Instruct
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-VL-4B-Instruct")
        
        # 从 HuggingFace transformers 加载
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
```

**关键功能**：
1. **Forward Pass**: 委托给底层 Qwen3.5-VL backbone
2. **Generate**: 自回归解码接口
3. **build_qwenvl_inputs**: 构建模型输入（images + instructions + solutions）

**与 Qwen2.5/Qwen3 对齐**：
```python
# Line 81-82: 对齐配置
self.model.config.hidden_size = self.model.config.text_config.hidden_size
```

**Action Token 支持**（Fast 模型）：
```python
# Line 84-87: 仅当模型包含 "-Action" 时启用
if "-Action" in model_id:
    self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN  # 248077
    self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX  # 248077 + 2047
```

### 2.2 数据处理流程

**输入格式**（遵循官方 Qwen3.5-VL Instruct 格式）：
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img1},
            {"type": "image", "image": img2},
            {"type": "text", "text": instruction}
        ]
    }
]
```

**Processor 处理**：
```python
batch_inputs = self.processor.apply_chat_template(
    messages,
    tokenize=True,
    padding=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
```

**Action Token Masking**（训练时）：
- 仅保留 action token 范围内的 labels
- 其他 token 设为 `IGNORE_INDEX = -100`
- Padding token 也 mask 为 -100

---

## 三、⚠️ 当前限制与待解决问题

### 3.1 缺少 hR_Interface 变体

**问题描述**：
- Qwen3 有 `_QWen3_VL_hR_Interface` 支持 `text_suffix` 参数
- Qwen3.5 **没有对应的 hR 变体**
- UamGR00T_GIA 的 `_MultiHeadBase` 需要 `text_suffix` 支持

**影响范围**：
- ❌ UamGR00T_GIA 无法使用 Qwen3.5（需要 `text_suffix` 传递 emoji tokens）
- ✅ UamGR00T_DT 可以使用 Qwen3.5（不需要 `text_suffix`）

**代码证据**：
```python
# UamGR00T_hR_multi.py:217
qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
    images,
    instructions,
    text_suffix=self._build_all_head_suffix(),  # ← 需要 text_suffix 参数
)
```

**对比 Qwen3 hR_Interface**：
```python
# QWen3_hR.py:26-40
class _QWen3_VL_hR_Interface(_QWen3_VL_Interface):
    def build_qwenvl_inputs(
        self,
        images,
        instructions,
        text_suffix: str | None = None,  # ← 支持 text_suffix
        solutions=None,
        **kwargs,
    ):
        # ...
        if text_suffix is not None:
            content.append({"type": "text", "text": text_suffix})
```

### 3.2 Transformers 版本依赖

**要求**：
```python
# QWen3_5.py:13-18
try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 model class is unavailable. "
        "Please install transformers >= 5.2.0 or check your transformers version."
    )
```

**待验证**：
- 当前环境的 transformers 版本是否 >= 5.2.0
- `Qwen3_5ForConditionalGeneration` 类是否可用

### 3.3 Image Size 处理差异

**Qwen3.5 特性**（注释中提到）：
```yaml
# run_uamgr00t_GIA_calvin.yaml:24
qwen_image_size: 256   # Qwen3.5 processor always outputs 8×8=64 tokens/view 
                       # regardless of pixel budget
```

**与 Qwen3 的区别**：
- Qwen3: 支持 `min_pixels` / `max_pixels` 动态控制 token 数量
- Qwen3.5: **固定输出 8×8=64 tokens/view**（根据配置注释）

**影响**：
- 如果代码期望动态 token 数量（如 49 tokens），可能需要适配
- 当前 GIA 配置注释中提到 64 tokens/view，与 49 tokens 设计不一致

---

## 四、兼容性矩阵

| 框架 | Qwen2.5-VL | Qwen3-VL | Qwen3.5-VL | 说明 |
|------|-----------|----------|------------|------|
| **UamVLAOFT** | ✅ | ✅ | ✅ | 基础框架，无特殊要求 |
| **UamVLAGR00T** | ✅ | ✅ | ✅ | 标准 GR00T，无特殊要求 |
| **UamGR00T_LT** | ✅ | ✅ | ❓ | 需要 token 插入，可能需要适配 |
| **UamGR00T_DT** | ✅ | ✅ | ✅ | 不需要 text_suffix，完全兼容 |
| **UamGR00T_GIA** | ✅ | ✅ (with hR) | ❌ | **需要 hR_Interface，当前不可用** |

**图例**：
- ✅ 完全支持
- ✅ (with hR) 需要 hR_Interface 变体
- ❓ 可能需要适配
- ❌ 当前不支持

---

## 五、解决方案建议

### 5.1 创建 QWen3_5_hR.py（推荐）

**实现步骤**：
1. 复制 `QWen3_hR.py` 为 `QWen3_5_hR.py`
2. 将父类从 `_QWen3_VL_Interface` 改为 `_QWen3_5_VL_Interface`
3. 保留 `text_suffix` 参数支持逻辑
4. 更新 `__init__.py` 中的路由逻辑

**代码示例**：
```python
# starVLA/model/modules/vlm/QWen3_5_hR.py
from starVLA.model.modules.vlm.QWen3_5 import (
    IGNORE_INDEX,
    _ACTION_TOKEN_MAX,
    _ACTION_TOKEN_MIN,
    _QWen3_5_VL_Interface,
)

class _QWen3_5_VL_hR_Interface(_QWen3_5_VL_Interface):
    """Qwen3.5-VL with text_suffix support for dedicated h_R / action tokens."""
    
    def build_qwenvl_inputs(
        self,
        images,
        instructions,
        text_suffix: str | None = None,
        solutions=None,
        **kwargs,
    ):
        # ... (与 QWen3_hR 相同的逻辑)
```

**更新路由逻辑**：
```python
# starVLA/model/modules/vlm/__init__.py
elif "Qwen3.5" in vlm_name:
    qwenvl_cfg = getattr(config.framework, "qwenvl", {})
    use_hr = getattr(qwenvl_cfg, "use_hR_interface", False) or (
        hasattr(qwenvl_cfg, "get") and qwenvl_cfg.get("use_hR_interface", False)
    )
    if use_hr:
        from .QWen3_5_hR import _QWen3_5_VL_hR_Interface
        return _QWen3_5_VL_hR_Interface(config)
    from .QWen3_5 import _QWen3_5_VL_Interface
    return _QWen3_5_VL_Interface(config)
```

### 5.2 更新 GIA 配置文件

**在 YAML 中启用 hR_Interface**：
```yaml
# examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml
framework:
  qwenvl:
    base_vlm: ./ckpt/Qwen3.5-4B
    use_hR_interface: true  # ← 新增此行
    attn_implementation: flash_attention_2
```

### 5.3 验证 Image Token 数量

**问题**：
- 配置注释说 Qwen3.5 固定输出 64 tokens/view
- GIA 设计期望 49 tokens/view（8×8 vs 7×7）

**验证方法**：
```python
# 在 smoke test 中验证
CUDA_VISIBLE_DEVICES=0 python starVLA/model/framework/VLM4A/UamGR00T_GIA.py \
  --config_yaml examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml

# 检查输出日志中的 token 数量
# 预期：每个 view 的 image token 数量
```

**如果确实是 64 tokens**：
- 更新 GIA 中的 `num_img_tokens=49` → `num_img_tokens=64`
- 更新 emoji token 重复次数（64 个而非 49 个）

---

## 六、行动计划

### P0 - 立即执行（使能 GIA + Qwen3.5）

1. **创建 QWen3_5_hR.py** 🔴
   - 复制 QWen3_hR.py 的逻辑
   - 继承 `_QWen3_5_VL_Interface`
   - 测试 `text_suffix` 参数传递

2. **更新路由逻辑** 🔴
   - 修改 `__init__.py` 支持 `use_hR_interface` 判断
   - 与 Qwen3 保持一致的接口选择逻辑

3. **验证 Image Token 数量** 🟡
   - Smoke test GIA framework
   - 确认每个 view 的 token 数量（49 vs 64）
   - 如有差异，更新相关配置

### P1 - 后续优化

4. **验证 Transformers 版本** 🟢
   - 检查环境中 transformers 版本 >= 5.2.0
   - 确认 `Qwen3_5ForConditionalGeneration` 可用

5. **测试完整训练流程** 🟢
   - 使用 Qwen3.5 运行 GIA smoke test
   - 验证数据处理和模型前向传播

6. **文档更新** 🟢
   - 在 CLAUDE.md 中添加 Qwen3.5 使用说明
   - 更新 UAMVLA.md 中的模型支持列表

---

## 七、技术参考

### 文件路径索引

**VLM Interfaces**：
```
starVLA/model/modules/vlm/
├── __init__.py                  # VLM 路由逻辑
├── QWen2_5.py                  # Qwen2.5-VL 接口
├── QWen3.py                    # Qwen3-VL 接口
├── QWen3_hR.py                 # Qwen3-VL hR 变体（支持 text_suffix）
├── QWen3_5.py                  # Qwen3.5-VL 接口 ✅
└── QWen3_5_hR.py               # Qwen3.5-VL hR 变体 ❌ 待创建
```

**Framework 使用方**：
```
starVLA/model/framework/VLM4A/
├── UamGR00T_hR_multi.py        # 多头基类，需要 text_suffix
├── UamGR00T_GIA.py             # GIA 框架，继承 hR_multi
└── UamGR00T_DT.py              # DT 框架，不需要 text_suffix
```

**配置文件**：
```
examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml
examples/Robotwin/train_files/run_uamgr00t_GIA_robotwin.yaml
```

### 关键常量

```python
# QWen3_5.py
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
_ACTION_TOKEN_MIN = 248077
_ACTION_TOKEN_MAX = 248077 + 2047
```

### Token ID 参考（UamGR00T_hR_multi.py:56）

```python
# Qwen3.5-4B tokenizer verified single-token emoji:
◇ = 158871
● = 42493
○ = 153089
✨ = 169379
▲ = 169006
▶ = 169199
```

---

## 八、总结

### 当前状态
- ✅ **基础 Qwen3.5 支持已实现**（标准 VLM interface）
- ❌ **缺少 hR 变体**（GIA 所需的 text_suffix 支持）
- ❓ **Image token 数量待验证**（49 vs 64 tokens/view）

### 最关键的阻塞
**UamGR00T_GIA 无法使用 Qwen3.5**，因为：
1. 需要 `_QWen3_5_VL_hR_Interface` 类（当前不存在）
2. 需要 `text_suffix` 参数支持（用于传递 emoji tokens）

### 解决方案
创建 `QWen3_5_hR.py`（工作量：~30分钟）：
- 复制 `QWen3_hR.py` 逻辑
- 继承 `_QWen3_5_VL_Interface`
- 更新路由逻辑

### 验证方法
```bash
# 创建 hR interface 后运行
CUDA_VISIBLE_DEVICES=0 python starVLA/model/framework/VLM4A/UamGR00T_GIA.py \
  --config_yaml examples/calvin/train_files/run_uamgr00t_GIA_calvin.yaml
```

---

**报告生成**: Claude Code (Opus 5)  
**最后更新**: 2026-08-31
