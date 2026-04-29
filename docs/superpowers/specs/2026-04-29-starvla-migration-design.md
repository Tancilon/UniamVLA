# starVLA Migration Design (UamVLA → UniamVLA fork)

**Date**: 2026-04-29
**Owner**: tancilon (`Tancilon/UniamVLA`)
**Status**: Approved (pending implementation kickoff)
**Source repo (read-only reference)**: `/Users/tancilon/develop/localgit/UamVLA`
**Target repo (work happens here)**: `/Users/tancilon/develop/localgit/UniamVLA`

---

## 1. Context & Motivation

UamVLA is a vision-language-action research project comprising:

- **Backbone**: Qwen3-VL-8B-Instruct, with custom state-token injection in the input embedding path
- **State encoder**: `ModularStateEncoder` over per-limb canonical state (`ee_pose`, `joint_pos`, `gripper`)
- **Embodiment adapters**: `LiberoAdapter`, `CalvinAdapter` — convert raw observations to a shared canonical schema
- **Four auxiliary heads**:
  - `ActionHead` — token-based action prediction via backbone `lm_head`
  - `PoseHead` — SE(3) score-matching diffusion (GenPose2-style) for 6D object pose
  - `FutureHead` — DiT-based future-frame reconstruction conditioned on image-token hidden states
  - `ReconHead` — DiT-based seg-crop reconstruction (target object) conditioned on image-token hidden states
- **Training stack (current)**: PyTorch Lightning + Hydra
- **Eval stack (current)**: ~2,143 LOC of bespoke runners (LIBERO + CALVIN HTTP server + JSON report aggregation)

This spec captures the migration of UamVLA's research components into a private fork of starVLA at `/Users/tancilon/develop/localgit/UniamVLA` (GitHub `Tancilon/UniamVLA`). Goals:

1. Inherit starVLA's mature multi-benchmark eval infrastructure (LIBERO, CALVIN, SimplerEnv, RoboTwin, BEHAVIOR, VLA-Arena, ...)
2. Replace UamVLA's bespoke training stack with starVLA's accelerate + DeepSpeed + OmegaConf-based stack
3. Standardize on starVLA conventions (framework registry, dataloader plugin mechanism, WebSocket eval) so future benchmarks plug in for free

The migration follows the **A1+B1** decision: preserve all UamVLA research components ("保架构换框架"), execute as a vertical slice (Phase 1 = LIBERO end-to-end), retrain from-scratch from Qwen3-VL-8B-Instruct (no checkpoint reuse).

---

## 2. Scope

### 2.1 Phase 1 (this spec)

- LIBERO Spatial end-to-end on UniamVLA: data preprocessing → training → eval → SR report
- All 4 aux heads enabled (action, pose, future, recon)
- From-scratch retrain from Qwen3-VL-8B-Instruct base
- Homogeneous per-batch embodiment (LIBERO only)
- No VLM co-training (`supports_training_tag("vlm")` returns `False`)
- Acceptance: LIBERO Spatial 10 task × 50 trial **SR ≥ 60%** (absolute threshold)

### 2.2 Phase 2 (out of scope; future work)

- CALVIN integration (preprocessor port, EmbodimentAdapter wiring, CALVIN env eval)
- Selective benchmark expansion: SimplerEnv / RoboTwin / VLA-Arena
- Cross-embodiment heterogeneous-batch co-training
- True 24D unified-action-space runtime (Phase 1 keeps JSONL=24D, runtime=7D — see §6.2)
- Action ensemble enabling (Phase 1.5 once baseline established)
- VLM co-training (`forward_vlm` implementation)

---

## 3. Project Layout

### 3.1 New / modified files in UniamVLA

```
UniamVLA/
├── starVLA/
│   ├── model/
│   │   ├── framework/VLM4A/
│   │   │   └── UamVLA.py                       [NEW] @FRAMEWORK_REGISTRY.register("UamVLA")
│   │   ├── framework/
│   │   │   └── base_framework.py               [PATCHED] +visualize_batch default no-op
│   │   └── modules/uamvla/                     [NEW] UamVLA-specific submodules (mirrors UamVLA layout for 1:1 portability)
│   │       ├── __init__.py
│   │       ├── state_encoder/                  ← from UamVLA/uamvla/models/state_encoder/
│   │       │   ├── modular_state_encoder.py
│   │       │   ├── limb_encoders.py
│   │       │   └── special_tokens.py
│   │       ├── aux_heads/                      ← from UamVLA/uamvla/models/aux_heads/
│   │       │   ├── base.py                     (HeadOutput dataclass + AuxHead ABC)
│   │       │   ├── action_head.py
│   │       │   ├── pose_head.py
│   │       │   ├── future_head.py
│   │       │   └── recon_head.py
│   │       ├── components/                     ← from UamVLA/uamvla/models/components/
│   │       │   ├── pose/                       ← PointNet2Wrapper, PoseScoreNet, SDE, sampler, pose_utils
│   │       │   ├── denoiser/                   ← DiT scheduler / dit / common
│   │       │   ├── pixel_decoder/              ← VAEPixelDecoder
│   │       │   ├── query_reader.py
│   │       │   ├── spatial_reader.py
│   │       │   └── task_adapter.py
│   │       ├── data/                           ← selected files from UamVLA/uamvla/data/
│   │       │   ├── embodiment_adapter.py       ← from embodiment_adapters.py
│   │       │   ├── embodiment_registry.py
│   │       │   ├── action_tokenizer.py
│   │       │   └── chat_template.py
│   │       └── backbone_wrapper.py             ← NEW: Qwen3-VL + state-token replacement (assembles state_encoder + chat_template + token replacement)
│   ├── dataloader/
│   │   └── uamvla_dataset.py                   [NEW] dataloader plugin (reads UamVLA JSONL)
│   ├── utils/                                  [NEW]
│   │   ├── geometry.py                         ← from UamVLA/uamvla/utils/geometry_utils.py
│   │   ├── rotation.py                         ← from UamVLA/uamvla/utils/rotation_utils.py
│   │   └── point_cloud.py                      ← from UamVLA/uamvla/utils/point_cloud_utils.py
│   ├── config/training/
│   │   └── uamvla_libero.yaml                  [NEW] training config (single-file OmegaConf)
│   └── training/
│       └── train_starvla.py                    [PATCHED] +viz hook, +get_lr_groups dispatch
├── tools/preprocess/                           [NEW]
│   ├── __init__.py
│   ├── base_preprocessor.py
│   ├── libero_preprocessor.py                  ← (image flip changed to [::-1, ::-1] — §6.5)
│   ├── target_object_resolver.py
│   ├── calvin_preprocessor.py                  ← Phase 2 (ported but not used in Phase 1)
│   ├── calvin_env_adapter.py                   ← Phase 2
│   └── calvin_task_map.py                      ← Phase 2
├── runners/
│   └── preprocess_libero.py                    [NEW] ported entry point; imports rewritten
├── examples/LIBERO/
│   ├── train_files/run_libero_train.sh         [MODIFIED IN-PLACE] paths/framework name
│   ├── eval_files/run_policy_server.sh         [MODIFIED IN-PLACE] paths/ckpt
│   └── eval_files/eval_libero.sh               [MODIFIED IN-PLACE] paths/ckpt/task_suite
└── docs/superpowers/specs/
    └── 2026-04-29-starvla-migration-design.md  (this file)
```

### 3.2 Discarded UamVLA components (not ported)

- `runners/eval_libero.py`, `runners/eval_calvin.py`, `runners/serve_calvin.py`, `runners/merge_eval_reports.py` — replaced by starVLA's WebSocket eval pattern
- `uamvla/eval_runners/` (entire directory, ~1,109 LOC) — replaced by `model2libero_interface.py` + WebSocket
- `uamvla/training/lightning_module.py`, `uamvla/training/callbacks.py` — replaced by starVLA's accelerate trainer
- `configs/` (entire Hydra tree) — replaced by single-file OmegaConf YAML
- `runners/train.py` — replaced by `starVLA/training/train_starvla.py`
- UamVLA's Lightning checkpoint loading — replaced by starVLA's safetensors `from_pretrained`

### 3.3 starVLA mainline files unchanged (verified)

- `deployment/model_server/server_policy.py`
- `examples/LIBERO/eval_files/eval_libero.py`
- `examples/LIBERO/eval_files/model2libero_interface.py`

These are interface-aligned (UamVLA YAML uses `framework.action_model.future_action_window_size`; norm_stats follows `dataset_statistics.json` schema). Zero modifications required.

---

## 4. Framework Class Interface

### 4.1 Class skeleton

```python
@FRAMEWORK_REGISTRY.register("UamVLA")
class UamVLA(baseframework):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = merge_framework_config(UamVLADefaultConfig, config)
        self.qwen_vl_interface = build_uamvla_backbone(self.config)
        hidden_size = self.qwen_vl_interface.config.hidden_size

        embodiment = self.config.framework.embodiment.name  # "franka_libero"
        self.embodiment_adapter = build_embodiment_adapter(embodiment)
        self.state_encoder = ModularStateEncoder(embodiment=embodiment, hidden_dim=hidden_size)
        register_state_tokens(self.qwen_vl_interface.tokenizer, self.qwen_vl_interface)

        self.aux_heads = nn.ModuleDict(build_aux_heads(self.config, hidden_size=hidden_size))
        self.action_horizon = int(self.config.framework.action_model.future_action_window_size) + 1
```

### 4.2 forward — multi-head loss summed into single `action_loss`

starVLA's `train_starvla.py:367-370` only backprops `output_dict["action_loss"]`. UamVLA sums per-head losses into that scalar; per-head metrics are returned as detached scalars for logging only.

```python
def forward(self, examples, **kwargs) -> dict:
    qwen_inputs = self.qwen_vl_interface.build_inputs(
        images=[e["image"] for e in examples],
        instructions=[e["lang"] for e in examples],
        canonical_state=[e["canonical_state"] for e in examples],
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        backbone_out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
        hidden = backbone_out.hidden_states[-1]

    batch_dict = collate_for_heads(examples)
    total = torch.tensor(0.0, device=hidden.device, requires_grad=True)
    log_metrics = {}
    for name, head in self.aux_heads.items():
        mask = batch_dict.get(f"{name}_mask")
        out = head.compute_loss(hidden, batch_dict, mask=mask)
        if out.loss is not None:
            total = total + out.loss
            log_metrics[f"{name}_loss"] = out.loss.detach()
        for mk, mv in out.metrics.items():
            log_metrics[f"{name}_{mk}"] = mv

    return {"action_loss": total, **log_metrics}
```

Mathematically identical to UamVLA's current `lightning_module.training_step` (which sums `head.loss` externally).

### 4.3 predict_action — eval-time, action head only

```python
@torch.inference_mode()
def predict_action(self, examples, **kwargs) -> dict:
    if not isinstance(examples, list):
        examples = [examples]
    qwen_inputs = self.qwen_vl_interface.build_inputs(
        images=[to_pil_preserve(e["image"]) for e in examples],
        instructions=[e["lang"] for e in examples],
        canonical_state=[e["canonical_state"] for e in examples],
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        backbone_out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
        hidden = backbone_out.hidden_states[-1]
    pred_actions = self.aux_heads["action"].predict(hidden)  # (B, action_horizon, 7)
    return {"normalized_actions": pred_actions.detach().cpu().numpy()}
```

Pose / Future / Recon heads do not run at eval — saving compute and matching their semantic role (training-time auxiliary supervision).

### 4.4 visualize_batch — wandb logging hook

```python
def visualize_batch(self, batch, n_samples=1) -> dict[str, "wandb.Image"]:
    out = {}
    hidden = self._backbone_forward(batch).hidden_states
    for name, head in self.aux_heads.items():
        if not hasattr(head, "visualize"):
            continue
        mask = batch.get(f"{name}_mask")
        if mask is None or mask.any():
            imgs = head.visualize(hidden, batch, mask, num_samples=n_samples)
            for i, img in enumerate(imgs):
                out[f"viz/{name}/{i}"] = img
    return out
```

Triggered by trainer hook (see §7.2).

### 4.5 get_lr_groups — per-head LR routing

```python
def get_lr_groups(self, lr_cfg) -> list[dict]:
    groups = [
        {"name": "qwen_vl_interface", "params": list(self.qwen_vl_interface.parameters()), "lr": lr_cfg.qwen_vl_interface},
        {"name": "state_encoder",     "params": list(self.state_encoder.parameters()),     "lr": lr_cfg.state_encoder},
    ]
    for name, head in self.aux_heads.items():
        head_cfg = self.config.framework.aux_heads[name]
        groups.append({
            "name": f"aux_head_{name}",
            "params": list(head.parameters()),
            "lr": float(head_cfg.get("lr", lr_cfg.base)),
        })
    return groups
```

starVLA trainer dispatches to this method when present (§7.1). Per-head LR is read from `framework.aux_heads.{name}.lr` in the YAML.

### 4.6 examples dict schema (extends starVLA standard)

```python
example = {
    # === starVLA standard ===
    "image": [PIL.Image, PIL.Image],   # [static, wrist]
    "lang": str,
    "action": np.ndarray,               # (H=8, 7) min-max normalized to [-1, 1]

    # === UamVLA extension ===
    "canonical_state": {                # nested dict per-limb (semantically distinct from starVLA's flat "state")
        "arm_0": {
            "ee_pose":   torch.Tensor,  # (9,) [xyz + 6D rotation]
            "joint_pos": torch.Tensor,  # (7,)
        },
        "gripper_0": torch.Tensor,      # (1,) normalized to [0, 1]
    },
    "action_mask": np.ndarray,          # (H=8, 7) per-step per-dim mask (collator builds token labels with this)
    "view_names": ["static", "wrist"],

    # === Aux head targets (per-sample optional; collator builds {name}_mask) ===
    "pose_gt": {"rotation": Tensor(6,), "translation": Tensor(3,)},
    "image_future": PIL.Image,
    "image_target": PIL.Image,
    "point_cloud": Tensor(1024, 3),
    "static_cam_extrinsic": {"rotation": Tensor(3,3), "translation": Tensor(3,)},  # PoseHead.visualize only
}
```

The dataloader's `__getitem__` produces this; `EmbodimentAdapter.to_canonical(raw_obs)` precomputes `canonical_state`. Collator stacks samples and emits `{recon,future,pose}_mask` (B,) bool tensors based on field presence.

### 4.7 Aux head field requirements (verified from source)

| Head | `batch[]` reads (compute_loss) | Sample-level JSONL fields | Source |
|------|-------------------------------|---------------------------|--------|
| ActionHead | `labels` | `action` (24D), `action_mask` (24D), `instruction`, `embodiment`, `episode_id`, `step_idx`, `total_steps`, `ee_pos`, `ee_axis_angle`, `joint_pos`, `gripper_qpos`, `image[]`, `view_names` | `action_head.py:35` |
| PoseHead | `point_cloud` (B,N,3); `pose_gt.rotation` (B,6); `pose_gt.translation` (B,3) | `pose_6d` (renamed `pose_gt` in dataset); `point_cloud` (.npy); `static_cam_extrinsic` (viz only) | `pose_head.py:167,187,188` |
| FutureHead | `input_ids`; `image_future` (B,3,H,W) | `image_future` (.jpg) | `future_head.py:111,114` |
| ReconHead | `input_ids`; `image_target` (B,3,H,W) | `image_target` (.jpg, seg crop) | `recon_head.py:119,123` |

Notes:
- ActionHead reads `labels`, NOT raw `action` field. The collator builds `labels` from `action`+`action_mask` via `ActionTokenizer.encode()` + chat-template scaffolding (`collator.py:135-159`). The model never sees raw 7D action arrays.
- `f"{name}_mask"` masks (B,) are collator-built from sample field presence: `pose_mask` ← `pose_gt` present; `recon_mask` ← `image_target` present; `future_mask` ← `image_future` present. ActionHead has no sample-level mask (uses token-level `labels != -100` internally).

---

## 5. Data Pipeline

### 5.1 Decision: skip LeRobot HDF5 conversion; write a custom dataloader plugin

**Reason**: UamVLA data is much richer than LeRobot's schema accepts. UamVLA JSONL stores point clouds, depth maps, future frames, 6D object pose, segmentation crops, and per-step canonical state fields (`ee_pos`, `ee_axis_angle`, `joint_pos`, `gripper_qpos` as separate keys, not concatenated). LeRobot's schema (`schema.py`) only supports state/action/video/annotation modalities with concatenated state vectors.

starVLA's dataloader is plugin-able via `datasets.vla_data.dataset_py` config. New file `starVLA/dataloader/uamvla_dataset.py` exposes `get_vla_dataset(data_cfg)` — reads UamVLA preprocessor JSONL + per-sample auxiliary files directly.

### 5.2 Action dimension: 24D in JSONL, 7D runtime (Phase 1)

UamVLA preprocessor pads action to 24D (`MAX_ACTION_DIM` = cross-embodiment future-proofing). Current UamVLA runtime trims to 7D unconditionally (`base_dataset.py:126`). This spec preserves status quo: JSONL stays 24D, runtime is 7D for Franka. Phase 2 may revisit if multi-embodiment co-training requires true unified-dim action heads.

### 5.3 Action normalization round-trip

| Stage | Operation | Where |
|-------|-----------|-------|
| Training dataloader | min-max → `[-1, 1]` | `uamvla_dataset.py.__getitem__` |
| Norm-stats persistence | `dataset_statistics.json` written to ckpt dir at trainer startup | trainer side |
| Eval ckpt load | `from_pretrained` attaches `model.norm_stats` | `base_framework.py:241` |
| Eval client | `unnormalize_actions(normalized, stats)` | `model2libero_interface.py:125` |

`dataset_statistics.json` schema (matches starVLA's expectation):

```json
{
  "franka_libero": {
    "action": {
      "min":  [...7 values...],
      "max":  [...7 values...],
      "mask": [true, true, true, true, true, true, false]
    }
  }
}
```

`mask[6] = false` indicates the gripper dimension does NOT use min-max (the eval client binarizes via `np.where(<0.5, 0, 1)` at `model2libero_interface.py:128-129`). UamVLA's training-time gripper is continuous in `[0, 1]`, compatible with this client behavior.

### 5.4 Image flip alignment (RETRAIN-DEPENDENT)

starVLA eval client flips images `obs[::-1, ::-1]` (180° rotation, `eval_libero.py:139-140`). UamVLA preprocessor currently flips only vertically: `rgb[::-1].copy()` (`libero_preprocessor.py:653`). Train/eval mismatch was the root cause of an earlier UamVLA SR=0 incident.

**Resolution**: change ported preprocessor's `_render_rgb` from `rgb[::-1].copy()` → `rgb[::-1, ::-1].copy()` to match starVLA eval. From-scratch retrain absorbs this cleanly (no historical checkpoints to re-align).

### 5.5 EmbodimentAdapter location: dataloader `__getitem__`

`uamvla_dataset.UamVLADataset.__getitem__` calls `self.adapter.to_canonical(raw_obs)` per sample. The model receives ready-made `canonical_state` and never executes raw-obs → canonical conversion. State token replacement happens in `backbone_wrapper.py` inside the framework's forward path.

### 5.6 Mixture registration

```python
# starVLA/dataloader/uamvla_dataset.py (or sibling registry file)
DATASET_NAMED_MIXTURES = {
    "libero_uamvla": [
        # (data_subdir, weight, embodiment_tag)
        ("libero_spatial", 1.0, "franka_libero"),
        # Phase 2: + libero_object, libero_goal, libero_10
    ],
}
```

`data_root_dir` (from YAML) + `data_subdir` resolves to the UamVLA preprocessor output directory containing `data.jsonl` + asset files.

### 5.7 Preprocessor migration

UamVLA preprocessing code physically ports to UniamVLA per directive:

```
UniamVLA/tools/preprocess/                ← from UamVLA/uamvla/data/preprocessing/
UniamVLA/runners/preprocess_libero.py     ← from UamVLA/runners/
UniamVLA/starVLA/utils/                   ← supporting utilities (geometry, rotation, point_cloud)
```

Import rewrites required: `uamvla.data.preprocessing.*` → `tools.preprocess.*`; `uamvla.utils.*` → `starVLA.utils.*`. Affects ~7-8 files.

---

## 6. Configuration

Single flat YAML at `starVLA/config/training/uamvla_libero.yaml`. Three top-level sections (starVLA convention).

```yaml
run_id: uamvla_libero_phase1
run_root_dir: playground/Checkpoints
seed: 42
trackers: [jsonl, wandb]
wandb_project: uamvla
wandb_entity: tancilon
is_debug: false
version_id: "0.1"

# ───────────────────────────────────────────────────────
# framework: model construction
# ───────────────────────────────────────────────────────
framework:
  name: UamVLA
  embodiment:
    name: franka_libero
    action_dim: 7
  qwenvl:
    base_vlm: ckpt/Qwen3-VL-8B-Instruct
    attn_implementation: flash_attention_2
  action_model:
    future_action_window_size: 7        # = action_horizon - 1; starVLA convention (read by ModelClient)
    num_bins: 256                        # ActionTokenizer bin count
    action_dim: 7
  state_encoder:
    type: modular
    register_special_tokens: true
  vae:
    path: ckpt/pretrained_vae
  aux_heads:
    action:
      enabled: true
      loss_weight: 1.0
      lr: 1.0e-4
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
      target_resize: 320                 # ppv=400 → 20*16
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

# ───────────────────────────────────────────────────────
# datasets: data loader plugin selection
# ───────────────────────────────────────────────────────
datasets:
  vla_data:
    dataset_py: uamvla_dataset           # starVLA/dataloader/uamvla_dataset.py
    data_root_dir: datasets/uamvla_libero
    data_mix: libero_uamvla
    image_size: 640                       # Qwen3-VL ppv=400 ⇒ 640x640
    per_device_batch_size: 2              # 8B backbone is tighter than 4B (starvla_cotrain_libero.yaml uses 16)
    delete_pause_frame: false

# ───────────────────────────────────────────────────────
# trainer: optimization / logging
# ───────────────────────────────────────────────────────
trainer:
  epochs: 50
  max_train_steps: 100000
  num_warmup_steps: 500
  save_interval: 2000
  eval_interval: 5000

  learning_rate:
    base:                  1.0e-4         # fallback for unmatched param groups
    qwen_vl_interface:     2.0e-5         # backbone (UamVLA's backbone_lr equivalent)
    state_encoder:         1.0e-4
    # per-head LR is read from framework.aux_heads.{name}.lr by UamVLA.get_lr_groups()

  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 1.0e-6

  freeze_modules: null                    # Phase 1 from-scratch — no freeze
  loss_scale:
    vla: 1.0                              # vlm not used (supports_training_tag returns False)

  max_grad_norm: 1.0
  weight_decay: 0.01
  logging_frequency: 10
  gradient_clipping: 1.0
  gradient_accumulation_steps: 16
  gradient_checkpointing: true

  optimizer:
    name: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-8
    weight_decay: 1.0e-8

  visualization:
    enabled: true
    train_every_n_steps: 1000
    num_samples: 1

  is_resume: false
  resume_epoch: null
  resume_step: null
  enable_gradient_checkpointing: true
  enable_mixed_precision_training: true
```

### Key config decisions (rationale traces)

| Config field | Decision | Reason |
|--------------|----------|--------|
| Single flat YAML | No Hydra-style composition | starVLA convention; trainer does `OmegaConf.load` once |
| `framework.aux_heads.{name}.enabled` | Per-head toggle | `UamVLA.__init__` skips disabled heads |
| `framework.action_model.future_action_window_size` | Use starVLA's name | `model2libero_interface.py:153` reads this key — alignment enables 0-mod client reuse |
| `framework.aux_heads.{name}.lr` | Per-head LR via `get_lr_groups` | Allows pose/future/recon to take different LR than action |
| `freeze_modules: null` | No freeze | Phase 1 from-scratch retrain; Phase 2 may freeze backbone for fine-tuning |
| `visualization.enabled: true` | Use trainer hook | Gives wandb image logging for pose/future/recon during training |
| `loss_scale.vla: 1.0` only | No `vlm` key | `supports_training_tag("vlm") == False` |

DeepSpeed ZeRO-2 + bf16 are NOT in this YAML — they're configured via `accelerate config` and passed to `accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml ...`.

---

## 7. starVLA Mainline Patches

Two small additive patches in `starVLA/`:

### 7.1 `starVLA/training/train_starvla.py` — get_lr_groups dispatch

```python
def setup_optimizer_and_scheduler(model, cfg):
    if hasattr(model, "get_lr_groups"):
        param_groups = model.get_lr_groups(cfg.trainer.learning_rate)
    else:
        param_groups = build_param_lr_groups(model=model, cfg=cfg)
    # ... rest unchanged
```

Backwards-compatible: frameworks without `get_lr_groups` use the existing `build_param_lr_groups`.

### 7.2 `starVLA/training/train_starvla.py` — visualization hook

Inside the training loop (after step completes):

```python
viz_cfg = cfg.trainer.get("visualization", {})
if viz_cfg.get("enabled", False) and self.completed_steps % viz_cfg.train_every_n_steps == 0:
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(self.model)
        viz_imgs = unwrapped.visualize_batch(batch, n_samples=viz_cfg.num_samples)
        if viz_imgs:
            wandb.log(viz_imgs, step=self.completed_steps)
```

### 7.3 `starVLA/model/framework/base_framework.py` — default no-op visualize_batch

```python
def visualize_batch(self, batch: dict, n_samples: int = 1) -> dict:
    """Default: no visualizations. Subclasses may override."""
    return {}
```

Other framework classes already in this fork gain a no-op method. No behavior change for any existing framework.

---

## 8. Eval Integration

### 8.1 Architecture (verified from code)

starVLA's WebSocket pattern is two-process:

- **Server** (`uamvla` conda env): `deployment/model_server/server_policy.py` loads ckpt via `from_pretrained`, listens on WebSocket, dispatches `model.predict_action(examples)`.
- **Client** (`libero_env` conda env, Py 3.8): `examples/LIBERO/eval_files/eval_libero.py` runs LIBERO env, uses `ModelClient` to call server's `predict_action`. Client un-normalizes actions via stats loaded from ckpt directory; reformats to LIBERO's `{world_vector, rotation_delta, open_gripper}` action format.

### 8.2 Code reuse: ZERO modifications to starVLA Python

| File | Status |
|------|--------|
| `deployment/model_server/server_policy.py` | unchanged |
| `examples/LIBERO/eval_files/eval_libero.py` | unchanged |
| `examples/LIBERO/eval_files/model2libero_interface.py` | unchanged |

Achieved by:
- UamVLA YAML uses `framework.action_model.future_action_window_size` (matches `model2libero_interface.py:153`)
- `dataset_statistics.json` matches the schema `model2libero_interface.py:144` reads
- `predict_action` returns `{"normalized_actions": ndarray}` with shape (B, T, 7) for Franka

### 8.3 In-place wrapper edits

Three bash wrappers, each with explicit `# === Please modify... ===` blocks (starVLA's editable-template convention):

**`examples/LIBERO/train_files/run_libero_train.sh`**:
```diff
-Framework_name=QwenOFT
+Framework_name=UamVLA
-base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
+base_vlm=ckpt/Qwen3-VL-8B-Instruct
-config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
+config_yaml=./starVLA/config/training/uamvla_libero.yaml
-libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
+libero_data_root=datasets/uamvla_libero
-data_mix=libero_all
+data_mix=libero_uamvla
-run_id=1229_libero4in1_qwen3oft
+run_id=uamvla_libero_phase1
-  --datasets.vla_data.per_device_batch_size 16
+  --datasets.vla_data.per_device_batch_size 2
-  --wandb_project starVLA_Libero
+  --wandb_project uamvla
-  --wandb_entity jinhuiye
+  --wandb_entity tancilon
```

**`examples/LIBERO/eval_files/run_policy_server.sh`**:
```diff
-STARVLA_DIR=/home/jye624/Projcets/starVLA
+STARVLA_DIR=/inspire/.../UniamVLA
-CKPT=${STARVLA_DIR}/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
+CKPT=${STARVLA_DIR}/playground/Checkpoints/uamvla_libero_phase1/checkpoints/steps_50000_pytorch_model.pt
-STARVLA_PYTHON=/home/jye624/.conda/envs/starVLA/bin/python
+STARVLA_PYTHON=$(conda run -n uamvla which python)
```

**`examples/LIBERO/eval_files/eval_libero.sh`**:
```diff
-STARVLA_DIR=/home/jye624/Projcets/starVLA
+STARVLA_DIR=/inspire/.../UniamVLA
-CKPT=...0405_libero4in1_CosmoPredict2GR00T...
+CKPT=...uamvla_libero_phase1...
-task_suite_name=libero_goal
+task_suite_name=libero_spatial
-export LIBERO_Python=/home/jye624/.conda/envs/libero/bin/python
+export LIBERO_Python=$(conda run -n libero_env which python)
```

The `cp $0 ${output_dir}/` line in `run_libero_train.sh:30` is preserved — gives every run a self-copy of its launcher for reproducibility.

### 8.4 Action ensemble

`ModelClient` defaults to `action_ensemble=True`, `adaptive_ensemble_horizon=3`. **Phase 1 keeps the default ON** to avoid any client-side modifications (zero-mod constraint, §8.2). Phase 1.5 may run an ablation by adding a `--action_ensemble` CLI flag to `eval_libero.py` and comparing SR with/without ensemble.

### 8.5 Conda env split

| Process | conda env | Key dependencies |
|---------|-----------|------------------|
| Training / Server | `uamvla` | torch, transformers ≥ 4.57, Qwen3-VL, deepspeed, accelerate, websockets |
| LIBERO Eval Client | `libero_env` (Py 3.8) | libero, robosuite, mujoco, websocket-client |

---

## 9. Workpackage Breakdown

| WP | Title | Output | Risk |
|----|-------|--------|------|
| WP1 | Dataloader plugin | `starVLA/dataloader/uamvla_dataset.py` (UamVLA JSONL → `examples` list) | Field shape mismatch with collator expectations; new code path |
| WP2 | UamVLAFramework class + UamVLA submodules | `starVLA/model/framework/VLM4A/UamVLA.py` + `starVLA/model/modules/uamvla/*.py` | Multi-aux-head loss summing semantics; state token replacement timing; M-RoPE under inputs_embeds path |
| WP3 | predict_action implementation | `predict_action()` method on UamVLAFramework | Output shape contract `(B, T, 7)`; `.cpu().numpy()` placement |
| WP4 | YAML + 3 wrapper edits | `starVLA/config/training/uamvla_libero.yaml`; modified `run_libero_train.sh`, `run_policy_server.sh`, `eval_libero.sh` | Per-head LR routing; `accelerate launch` arg passthrough |
| WP5 | Mixture registration + utils porting + preprocessor port | `DATASET_NAMED_MIXTURES["libero_uamvla"]`; `starVLA/utils/{geometry,rotation,point_cloud}.py`; `tools/preprocess/`; `runners/preprocess_libero.py` | Import path rewrites across ~10 files |
| WP6 | (DROPPED — checkpoint conversion) | — | from-scratch retrain |
| WP7 | starVLA mainline patches + eval integration | `base_framework.py` `+visualize_batch`; `train_starvla.py` `+viz hook`, `+get_lr_groups dispatch` | Hook ordering; main-process-only wandb logging |
| WP8 | Smoke test ladder L1-L7 + Phase 1 acceptance run | pytest fixtures, smoke configs, L7 baseline run | GPU time budget; remote env reproducibility |

---

## 10. Smoke Tests & Acceptance

### 10.1 Smoke ladder

| Level | Test | Pass criterion | Where | ~ELT |
|-------|------|----------------|-------|------|
| L1 | Unit smoke | `UamVLAFramework.__init__` succeeds; 4 aux heads register; `register_state_tokens` resizes vocab | Mac local pytest | < 1 min |
| L2 | 1-step training | All loss values finite at step 1; no NaN/Inf | 8-GPU remote, `--max_train_steps 1` | < 5 min |
| L3 | Dataloader smoke | `uamvla_dataset.UamVLADataset[0]` → collator → batch dict has all expected fields with correct shapes | Mac local pytest with mock JSONL | < 2 min |
| L4 | 50-step training | `action_loss` decreases ≥ 5%; per-head losses bounded | Remote | ~ 15 min |
| L5 | 1-task eval dry-run | 1 LIBERO Spatial task × 1 episode completes (any SR including 0) | Remote (server + libero_env) | ~ 10 min |
| L6 | Partial eval | LIBERO Spatial 10 task × 5 trial; full pipeline runs | Remote | ~ 1-2 h |
| **L7** | **Full Phase 1 acceptance run** | **LIBERO Spatial 10 task × 50 trial, SR ≥ 60%** | Remote | ~ 6-12 h |

### 10.2 Acceptance criteria (Phase 1 done = all of)

| Dimension | Standard |
|-----------|----------|
| Quantitative primary | LIBERO Spatial 10 task × 50 trial **SR ≥ 60%** (absolute threshold; reference: OpenVLA ~85%, π0 ~95%) |
| Training stability | 100k step wall-clock ≤ 1.2× UamVLA's current finetune time on equivalent hardware (8×H100 / 8×H200) |
| Per-head health | `action_loss`, `pose_loss`, `future_loss`, `recon_loss` all monotonically decreasing trend; none NaN/Inf |
| Eval qualitative | At least 1 success video per task; agent behavior visually plausible (not random thrashing) |
| Reproducibility | Run dir contains `config.full.yaml`, `config.yaml` (accessed-snapshot), `dataset_statistics.json`, copied launcher script; `from_pretrained(<ckpt>)` reload succeeds end-to-end |
| Code quality gate | starVLA mainline patches (`get_lr_groups` dispatch, viz hook, `visualize_batch` default) + new framework class + dataloader plugin all have pytest coverage |

### 10.3 Failure-mode checklist (high-risk pitfalls)

| Symptom | Likely cause | First check |
|---------|--------------|-------------|
| NaN at step 1 | dtype mismatch (fp32 / bf16 boundary); register_state_tokens didn't resize embeddings | UamVLAFramework.__init__ ordering; backbone_wrapper cast points |
| `action_loss` stuck high | min-max normalization off; pad tokens not set to -100 | collator labels build (`collator.py:135-159`); statistics → JSON conversion |
| `pose_loss` flat | point_cloud shape wrong; score_dtype / pts_center dtype mismatch | `pose_head._prepare_score_inputs` |
| `future_loss` / `recon_loss` flat | `image_token_id` misidentified; `patches_per_view` ≠ chat-template image expansion count | `slice_image_tokens` output shape vs `n_patches` |
| Eval SR=0 | image flip mismatch; un-normalize stats wrong; chunk size wrong | preprocessor `[::-1, ::-1]`? stats has `mask`? `future_action_window_size: 7`? |
| Eval slow / unstable | bf16 not enabled; ensemble chunk artifacts | `--use_bf16` server flag; client `use_ddim=True num_ddim_steps=10` |

Several echo prior UamVLA debugging incidents:
- Image flip → `feedback_libero_image_flip.md`
- Dtype handling → `feedback_dtype_handling.md`
- GPU multiprocess → `feedback_gpu_multiprocess_spawn.md`
- LiberoEnv reset → `feedback_libero_env_reset.md`

---

## 11. Out of Scope (Phase 2 hooks)

Captured here so they aren't re-discovered:

- CALVIN integration (CalvinAdapter, calvin_preprocessor port already in tree but not used; CALVIN env eval client)
- SimplerEnv / RoboTwin / VLA-Arena selective integration
- True 24D unified-action runtime (currently 7D internal; revisit when multi-embodiment co-training requires it)
- Cross-embodiment heterogeneous-batch co-training
- VLM co-training (`forward_vlm` implementation)
- Action ensemble re-enable (Phase 1.5)
- Visualization dashboards beyond wandb image logging
- Upstream PR back to starVLA mainline (assumes private fork remains the home)

---

## 12. References

### Memory entries (build context)

- `project_starvla_migration.md` — parent decision record (paths, locked decisions)
- `feedback_no_claude_attribution.md` — git commit attribution rule
- `feedback_libero_image_flip.md` — past SR=0 root cause
- `feedback_dtype_handling.md` — DeepSpeed BF16 + frozen backbone patterns
- `feedback_gpu_multiprocess_spawn.md` — fork vs spawn after GPU init
- `feedback_libero_env_reset.md` — robosuite reset() requirement
- `project_base_model_choice.md` — Qwen3-VL-8B-Instruct selection (640x640, ppv=400)

### Source files cited

- `starVLA/model/framework/base_framework.py:79-181` — baseframework contract; compute_loss tag dispatch
- `starVLA/training/train_starvla.py:78-101` — setup_optimizer_and_scheduler
- `starVLA/training/train_starvla.py:367-370` — single `action_loss` backprop
- `examples/LIBERO/eval_files/model2libero_interface.py:125-153` — un-normalize + chunk-size read
- `examples/LIBERO/eval_files/eval_libero.py:139-140` — image flip
- `UamVLA/uamvla/data/preprocessing/libero_preprocessor.py:653` — preprocessor render flip
- `UamVLA/uamvla/data/collator.py:135-159` — action token labels build
- `UamVLA/uamvla/data/base_dataset.py:126` — 24D → 7D action trim
- `UamVLA/uamvla/models/aux_heads/{action,pose,future,recon}_head.py` — aux head field reads
- `UamVLA/uamvla/models/uamvla_model.py:347-352` — per-head mask lookup pattern

### External

- starVLA upstream: `https://github.com/starVLA/starVLA`
- UniamVLA fork: `https://github.com/Tancilon/UniamVLA`
- LIBERO benchmark: published SR references (OpenVLA ~85%, π0 ~95%)
