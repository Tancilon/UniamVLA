# AuxVLAGR00T ReconVLA AR Recon Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `AuxVLAGR00T` training mode that uses official ReconVLA autoregressive action-token CE plus official internal `vm_loss` reconstruction, with LoRA, 15-D robot state prompt injection, and CALVIN 5-step chunks.

**Architecture:** Keep the existing GR00T action-head path unchanged. Add a separate `framework.reconvla.training_mode=reconvla_ar_recon` branch that builds official-style ReconVLA input IDs, labels, target images, and calls `ReconQwen2ForCausalLM` directly. Preserve raw action and raw 15-D `robot_obs` in the dataloader before StarVLA normalization so AR tokenization does not double-normalize.

**Tech Stack:** PyTorch, OmegaConf, PEFT LoRA, ReconVLA `ActionTokenizer`, StarVLA LeRobot dataloader transforms, pytest, CUDA no-save smoke probes.

---

## File Structure

- Modify `starVLA/dataloader/gr00t_lerobot/transform/base.py`
  - Add a tiny transform that copies raw modality keys before normalization.
- Modify `starVLA/dataloader/gr00t_lerobot/transform/__init__.py`
  - Export the raw-preserve transform.
- Modify `starVLA/dataloader/gr00t_lerobot/datasets.py`
  - Pack preserved raw robot state/action into `sample["uamvla_raw_state"]` and `sample["uamvla_raw_action"]`.
- Modify `examples/calvin/train_files/data_registry/data_config.py`
  - Add the raw-preserve transform to CALVIN configs before `StateActionTransform`.
- Modify `starVLA/model/framework/VLM4A/UamVLAOFT.py`
  - Pass raw state/action fields through `_unpack_lerobot_sample`.
- Modify `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`
  - Add ReconVLA AR training builders, official target preprocessing, recon module checks, and `forward()` branch.
- Create `tools/probes/smoke_auxvla_reconvla_ar_recon_no_save.py`
  - CUDA no-save probe for the official AR+recon training path.
- Modify `starVLA/config/training/auxvla_gr00t_lora.yaml`
  - No default behavior changes; only add commented reference if needed.
- Create `starVLA/config/training/auxvla_reconvla_ar_recon.yaml`
  - Dedicated opt-in config for the official recipe experiment.
- Modify or create tests:
  - `tests/dataloader/test_preserve_raw_transform.py`
  - `tests/framework/test_auxvla_gr00t_static.py`
  - `tests/framework/test_auxvla_gr00t_interface.py`
  - `tests/tools/test_smoke_auxvla_reconvla_ar_recon_no_save.py`

Do not commit unrelated local changes. At the time this plan was written, `examples/calvin/train_files/data_registry/data_config.py` already had an unstaged user change; inspect it before editing and keep changes additive.

---

### Task 1: Preserve Raw CALVIN State And Action Before Normalization

**Files:**
- Modify: `starVLA/dataloader/gr00t_lerobot/transform/base.py`
- Modify: `starVLA/dataloader/gr00t_lerobot/transform/__init__.py`
- Modify: `starVLA/dataloader/gr00t_lerobot/datasets.py`
- Modify: `examples/calvin/train_files/data_registry/data_config.py`
- Modify: `starVLA/model/framework/VLM4A/UamVLAOFT.py`
- Test: `tests/dataloader/test_preserve_raw_transform.py`

- [ ] **Step 1: Write the failing raw-preserve transform test**

Create `tests/dataloader/test_preserve_raw_transform.py`:

```python
from __future__ import annotations

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.transform.base import PreserveRawModalityTransform


def test_preserve_raw_modality_transform_copies_numpy_and_tensor_values():
    transform = PreserveRawModalityTransform(
        apply_to=["state.robot_obs", "action.x"],
        output_key="uamvla_raw_reconvla",
    )
    robot_obs = np.arange(15, dtype=np.float32).reshape(1, 15)
    action_x = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    data = {
        "state.robot_obs": robot_obs,
        "action.x": action_x,
    }

    out = transform(data)

    assert "uamvla_raw_reconvla" in out
    assert np.allclose(out["uamvla_raw_reconvla"]["state.robot_obs"], robot_obs)
    assert torch.allclose(out["uamvla_raw_reconvla"]["action.x"], action_x)

    robot_obs[...] = -1
    action_x.fill_(-2)
    assert not np.allclose(out["uamvla_raw_reconvla"]["state.robot_obs"], robot_obs)
    assert not torch.allclose(out["uamvla_raw_reconvla"]["action.x"], action_x)
```

- [ ] **Step 2: Run the failing transform test**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/dataloader/test_preserve_raw_transform.py -q
```

Expected: FAIL with `ImportError` or `AttributeError` because `PreserveRawModalityTransform` does not exist.

- [ ] **Step 3: Implement `PreserveRawModalityTransform`**

Add this class near `ComposedModalityTransform` in `starVLA/dataloader/gr00t_lerobot/transform/base.py`:

```python
class PreserveRawModalityTransform(ModalityTransform):
    """Copy raw modality values before later transforms normalize them."""

    output_key: str = Field(
        default="uamvla_raw_reconvla",
        description="Destination key containing raw values keyed by original modality key.",
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        raw: dict[str, Any] = dict(data.get(self.output_key, {}))
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            if hasattr(value, "detach"):
                raw[key] = value.detach().clone()
            else:
                raw[key] = np.asarray(value).copy()
        data[self.output_key] = raw
        return data
```

Also add `import numpy as np` at the top of `base.py`.

- [ ] **Step 4: Export the transform**

Modify `starVLA/dataloader/gr00t_lerobot/transform/__init__.py` to import it:

```python
from .base import ComposedModalityTransform, ModalityTransform, PreserveRawModalityTransform
```

If the file already imports `ComposedModalityTransform`, add only `PreserveRawModalityTransform`.

- [ ] **Step 5: Run the transform test**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/dataloader/test_preserve_raw_transform.py -q
```

Expected: PASS.

- [ ] **Step 6: Pack raw values in LeRobot samples**

In `starVLA/dataloader/gr00t_lerobot/datasets.py`, inside `_pack_sample()` after the normalized `action` and optional normalized `state` are assembled, add:

```python
        raw_reconvla = data.get("uamvla_raw_reconvla")
        if isinstance(raw_reconvla, dict):
            if "state.robot_obs" in raw_reconvla:
                sample["uamvla_raw_state"] = {
                    "robot_obs": np.asarray(raw_reconvla["state.robot_obs"], dtype=np.float32)
                }

            raw_action_parts = []
            for action_key in self.modality_keys["action"]:
                if action_key in raw_reconvla:
                    raw_action_parts.append(np.asarray(raw_reconvla[action_key], dtype=np.float32))
            if raw_action_parts:
                sample["uamvla_raw_action"] = np.concatenate(raw_action_parts, axis=1).astype(np.float32)
```

Place this block before returning `sample`.

- [ ] **Step 7: Add raw preserve to CALVIN data config**

In `examples/calvin/train_files/data_registry/data_config.py`, import:

```python
from starVLA.dataloader.gr00t_lerobot.transform.base import (
    ComposedModalityTransform,
    PreserveRawModalityTransform,
)
```

Then add the preserve transform as the first transform in `UamVLACalvinDataConfig.transform()`:

```python
            PreserveRawModalityTransform(
                apply_to=["state.robot_obs", *self.action_keys],
                output_key="uamvla_raw_reconvla",
            ),
```

Keep the existing `StateActionToTensor` and `StateActionTransform` order unchanged after this new first step.

- [ ] **Step 8: Pass raw fields through framework unpacking**

In `starVLA/model/framework/VLM4A/UamVLAOFT.py`, inside `_unpack_lerobot_sample()` before `return out`, add:

```python
        if "uamvla_raw_state" in sample:
            out["uamvla_raw_state"] = sample["uamvla_raw_state"]
            robot_obs = sample["uamvla_raw_state"].get("robot_obs")
            if robot_obs is not None:
                out["robot_obs"] = np.asarray(robot_obs, dtype=np.float32).reshape(-1)

        if "uamvla_raw_action" in sample:
            raw_action = sample["uamvla_raw_action"]
            if not torch.is_tensor(raw_action):
                raw_action = torch.as_tensor(np.asarray(raw_action), dtype=torch.float32)
            else:
                raw_action = raw_action.to(dtype=torch.float32)
            out["uamvla_raw_action"] = raw_action
```

- [ ] **Step 9: Add a dataloader packing regression test**

Add this test to `tests/dataloader/test_preserve_raw_transform.py`:

```python
def test_pack_sample_preserves_reconvla_raw_robot_obs_and_action(monkeypatch):
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

    dataset = object.__new__(LeRobotSingleDataset)
    dataset.data_cfg = {"image_resize": 8, "include_state": True}
    dataset.modality_keys = {
        "video": ["video.primary_image", "video.wrist_image"],
        "language": ["annotation.human.action.task_description"],
        "action": ["action.x", "action.y"],
        "state": ["state.robot_obs"],
    }

    image = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    data = {
        "video.primary_image": image,
        "video.wrist_image": image,
        "annotation.human.action.task_description": ["move"],
        "action.x": np.ones((5, 1), dtype=np.float32),
        "action.y": np.ones((5, 1), dtype=np.float32) * 2,
        "state.robot_obs": np.ones((1, 15), dtype=np.float32) * 3,
        "uamvla_raw_reconvla": {
            "action.x": np.ones((5, 1), dtype=np.float32) * 4,
            "action.y": np.ones((5, 1), dtype=np.float32) * 5,
            "state.robot_obs": np.arange(15, dtype=np.float32).reshape(1, 15),
        },
    }

    sample = LeRobotSingleDataset._pack_sample(dataset, data)

    assert sample["uamvla_raw_action"].shape == (5, 2)
    assert np.allclose(sample["uamvla_raw_action"][0], [4.0, 5.0])
    assert sample["uamvla_raw_state"]["robot_obs"].shape == (1, 15)
```

- [ ] **Step 10: Run dataloader tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/dataloader/test_preserve_raw_transform.py tests/dataloader/test_uamvla_calvin_mixture.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit Task 1**

Commit only Task 1 files:

```bash
git add \
  starVLA/dataloader/gr00t_lerobot/transform/base.py \
  starVLA/dataloader/gr00t_lerobot/transform/__init__.py \
  starVLA/dataloader/gr00t_lerobot/datasets.py \
  examples/calvin/train_files/data_registry/data_config.py \
  starVLA/model/framework/VLM4A/UamVLAOFT.py \
  tests/dataloader/test_preserve_raw_transform.py
git commit -m "feat: preserve raw CALVIN state and actions"
```

---

### Task 2: Build ReconVLA AR Training Inputs And Labels

**Files:**
- Modify: `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`
- Test: `tests/framework/test_auxvla_gr00t_static.py`
- Test: `tests/framework/test_auxvla_gr00t_interface.py`

- [ ] **Step 1: Write static tests for AR action labels**

Add to `tests/framework/test_auxvla_gr00t_static.py`:

```python
def test_reconvla_ar_training_requires_35_action_labels(monkeypatch):
    module = _load_auxvla_module(monkeypatch)

    model = object.__new__(module.AuxVLAGR00T)
    model.action_horizon = 5
    model.config = _AttrDict(
        framework=_AttrDict(action_model=_AttrDict(action_dim=7, action_horizon=5))
    )

    actions = torch.zeros(5, 7)
    flat = module.AuxVLAGR00T._ar_training_action_array(model, {"uamvla_raw_action": actions})
    assert flat.shape == (35,)

    with pytest.raises(RuntimeError, match="Expected 35 action values"):
        module.AuxVLAGR00T._ar_training_action_array(model, {"uamvla_raw_action": torch.zeros(4, 7)})
```

Add another test for raw state:

```python
def test_reconvla_ar_training_requires_raw_robot_obs(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)

    with pytest.raises(RuntimeError, match="15-D robot_obs"):
        module.AuxVLAGR00T._ar_training_robot_obs(model, {"state": np.zeros((1, 7), dtype=np.float32)})

    out = module.AuxVLAGR00T._ar_training_robot_obs(
        model,
        {"uamvla_raw_state": {"robot_obs": np.arange(15, dtype=np.float32).reshape(1, 15)}},
    )
    assert out.shape == (15,)
```

- [ ] **Step 2: Run static tests to verify failure**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py::test_reconvla_ar_training_requires_35_action_labels \
  tests/framework/test_auxvla_gr00t_static.py::test_reconvla_ar_training_requires_raw_robot_obs -q
```

Expected: FAIL because helper methods do not exist.

- [ ] **Step 3: Add action/state extraction helpers**

In `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`, add methods on `AuxVLAGR00T` near `_extract_reconvla_ar_robot_obs`:

```python
    def _ar_training_robot_obs(self, example: dict) -> np.ndarray:
        return self._extract_reconvla_ar_robot_obs(example)

    def _ar_training_action_array(self, example: dict) -> np.ndarray:
        action_dim = int(self.config.framework.action_model.get("action_dim", 7))
        action_horizon = int(self.config.framework.action_model.get("action_horizon", self.action_horizon))
        expected = action_horizon * action_dim
        source = example.get("uamvla_raw_action", example.get("action"))
        if source is None:
            raise RuntimeError("ReconVLA AR training requires `uamvla_raw_action` or `action`.")
        if torch.is_tensor(source):
            values = source.detach().float().cpu().numpy()
        else:
            values = np.asarray(source, dtype=np.float32)
        values = values.reshape(-1, action_dim)
        values = values[:action_horizon].reshape(-1).astype(np.float32)
        if values.shape[0] != expected:
            raise RuntimeError(
                f"Expected {expected} action values for ReconVLA AR labels, got {values.shape[0]}."
            )
        return values
```

- [ ] **Step 4: Add `encode_ar_action_tokens` to `ReconVLAInterface`**

In `ReconVLAInterface.__init__`, change the import to:

```python
from recon.action_tokenizer import ActionTokenizer, encode_actions, encode_robot_obs
```

and save:

```python
self._encode_actions = encode_actions
```

Then add:

```python
    def encode_ar_action_tokens(
        self,
        actions: np.ndarray,
        source: str = "raw_with_reconvla_statistics",
    ) -> torch.Tensor:
        actions = np.asarray(actions, dtype=np.float32).reshape(-1)
        if source == "raw_with_reconvla_statistics":
            action_text = " ".join(str(float(value)) for value in actions)
            token_ids, _decoded = self._encode_actions(
                action_text,
                self.action_tokenizer,
                str(self._action_stat_path()),
            )
        elif source == "starvla_normalized_direct":
            token_ids, _decoded = self.action_tokenizer(actions)
        else:
            raise ValueError(
                "Unsupported framework.reconvla.action_token_source="
                f"{source!r}. Expected `raw_with_reconvla_statistics` or "
                "`starvla_normalized_direct`."
            )
        return torch.as_tensor(token_ids, dtype=torch.long)
```

- [ ] **Step 5: Add AR training input builder**

Add to `ReconVLAInterface`:

```python
    def build_reconvla_ar_training_inputs(
        self,
        images,
        instructions,
        robot_obs,
        actions,
        target_images,
        input_mode: str,
        action_token_source: str,
        action_horizon: int,
        action_dim: int,
    ) -> dict:
        ar_inputs = self.build_reconvla_ar_inputs(
            images=images,
            instructions=instructions,
            robot_obs=robot_obs,
            input_mode=input_mode,
        )
        input_rows = []
        label_rows = []
        action_token_counts = []
        for row, action in zip(ar_inputs["input_ids"], actions):
            valid_len = int(row.ne(self.tokenizer.pad_token_id).sum().item())
            prompt = row[:valid_len]
            action_tokens = self.encode_ar_action_tokens(action, source=action_token_source)
            expected = int(action_horizon) * int(action_dim)
            if action_tokens.numel() != expected:
                raise RuntimeError(
                    f"ReconVLA AR action tokenizer produced {action_tokens.numel()} tokens, expected {expected}."
                )
            eos = torch.as_tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            full = torch.cat([prompt.detach().cpu(), action_tokens, eos], dim=0)
            labels = torch.full_like(full, -100)
            labels[prompt.numel(): prompt.numel() + action_tokens.numel()] = action_tokens
            input_rows.append(full)
            label_rows.append(labels)
            action_token_counts.append(int(action_tokens.numel()))

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_rows,
            batch_first=True,
            padding_value=pad_token_id,
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            label_rows,
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = padded_input_ids.ne(pad_token_id)
        target_tensor = torch.stack(
            [self.preprocess_target_image(image) for image in target_images],
            dim=0,
        )
        device = self.device
        return {
            "input_ids": padded_input_ids.to(device),
            "labels": padded_labels.to(device),
            "attention_mask": attention_mask.to(device),
            "images": ar_inputs["images"].to(device),
            "target_images": target_tensor.to(device),
            "action_token_counts": action_token_counts,
        }
```

- [ ] **Step 6: Add target image preprocessing helper**

Add to `ReconVLAInterface`:

```python
    def preprocess_target_image(self, image) -> torch.Tensor:
        if torch.is_tensor(image):
            image = _to_rgb_pil(image)
        else:
            image = _to_rgb_pil(image)
        return self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
```

- [ ] **Step 7: Run static/interface tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py \
  tests/framework/test_auxvla_gr00t_interface.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

```bash
git add starVLA/model/framework/VLM4A/AuxVLAGR00T.py tests/framework/test_auxvla_gr00t_static.py tests/framework/test_auxvla_gr00t_interface.py
git commit -m "feat: build ReconVLA AR training labels"
```

---

### Task 3: Add `reconvla_ar_recon` Forward Branch

**Files:**
- Modify: `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`
- Test: `tests/framework/test_auxvla_gr00t_static.py`

- [ ] **Step 1: Write failing forward branch test**

Add to `tests/framework/test_auxvla_gr00t_static.py`:

```python
def test_forward_reconvla_ar_recon_uses_qwen_loss_not_gr00t(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    model.action_horizon = 5
    model.config = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(
                training_mode="reconvla_ar_recon",
                ar_input_mode="official_compose",
                action_token_source="raw_with_reconvla_statistics",
            ),
            action_model=_AttrDict(action_dim=7, action_horizon=5),
        )
    )
    model._prepare_examples = types.MethodType(lambda self, examples, require_reconvla_target=False: examples, model)
    model._ar_training_robot_obs = types.MethodType(lambda self, example: np.arange(15, dtype=np.float32), model)
    model._ar_training_action_array = types.MethodType(lambda self, example: np.zeros(35, dtype=np.float32), model)
    model._reconvla_ar_images = types.MethodType(lambda self, examples, input_mode: [e["image"] for e in examples], model)
    model._require_reconvla_internal_recon = types.MethodType(lambda self: None, model)

    class _FakeInterface:
        def __init__(self):
            self.training_inputs = None

        def build_reconvla_ar_training_inputs(self, **kwargs):
            self.training_inputs = kwargs
            return {
                "input_ids": torch.ones(1, 40, dtype=torch.long),
                "attention_mask": torch.ones(1, 40, dtype=torch.bool),
                "labels": torch.cat([torch.full((1, 5), -100), torch.ones(1, 35, dtype=torch.long)], dim=1),
                "images": torch.zeros(1, 3, 8, 8),
                "target_images": torch.zeros(1, 3, 8, 8),
                "action_token_counts": [35],
            }

        def __call__(self, **kwargs):
            return types.SimpleNamespace(
                loss=torch.tensor(3.0, requires_grad=True),
                lm_loss=torch.tensor(2.0),
                vm_loss=torch.tensor(1.0),
            )

    model.qwen_vl_interface = _FakeInterface()

    class _ForbiddenActionModel:
        def __call__(self, *args, **kwargs):
            raise AssertionError("GR00T action head must not be called in reconvla_ar_recon mode")

    model.action_model = _ForbiddenActionModel()
    example = {
        "image": [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)],
        "lang": "move",
        "image_target": torch.zeros(3, 8, 8),
    }

    out = module.AuxVLAGR00T.forward(model, [example])

    assert out["action_loss"].item() == 3.0
    assert out["action_loss_ar"].item() == 3.0
    assert out["reconvla_lm_loss"].item() == 2.0
    assert out["reconvla_vm_loss"].item() == 1.0
    assert model.qwen_vl_interface.training_inputs["action_horizon"] == 5
```

- [ ] **Step 2: Run failing test**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py::test_forward_reconvla_ar_recon_uses_qwen_loss_not_gr00t -q
```

Expected: FAIL because the forward branch does not exist.

- [ ] **Step 3: Add training mode helpers**

In `AuxVLAGR00T.py`, add:

```python
    def _reconvla_training_mode(self) -> str:
        recon_cfg = self.config.framework.get("reconvla", {})
        return str(recon_cfg.get("training_mode", "gr00t"))

    def _require_reconvla_internal_recon(self) -> None:
        model = self.qwen_vl_interface.model
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        inner = base.get_model() if hasattr(base, "get_model") else getattr(base, "model", base)
        if not hasattr(inner, "mm_inv_projector"):
            raise RuntimeError("ReconVLA AR recon training requires `mm_inv_projector`.")
        if not hasattr(inner, "pixel_decoder"):
            raise RuntimeError("ReconVLA AR recon training requires `pixel_decoder`.")
        model.config.recon_enable = True
        model.config.reconstruct_image = False
```

In `ReconVLAInterface.__init__`, replace the recon-disable block with:

```python
        if bool(recon_cfg.get("disable_internal_recon_loss", True)):
            self.model.config.recon_enable = False
            self.model.config.reconstruct_image = False
        elif getattr(self.model.config, "mm_pixel_decoder", None):
            self.model.config.recon_enable = True
            self.model.config.reconstruct_image = False
```

- [ ] **Step 4: Add `_forward_reconvla_ar_recon`**

In `AuxVLAGR00T.py`, add before `forward()`:

```python
    def _forward_reconvla_ar_recon(self, examples: List[dict]) -> dict:
        examples = self._prepare_examples(examples, require_reconvla_target=True)
        self._require_reconvla_internal_recon()

        recon_cfg = _cfg_to_plain_dict(self.config.framework.get("reconvla", {}))
        input_mode = str(recon_cfg.get("ar_input_mode", "official_compose"))
        action_token_source = str(
            recon_cfg.get("action_token_source", "raw_with_reconvla_statistics")
        )
        action_dim = int(self.config.framework.action_model.get("action_dim", 7))
        action_horizon = int(
            self.config.framework.action_model.get("action_horizon", self.action_horizon)
        )
        robot_obs = np.stack([self._ar_training_robot_obs(example) for example in examples], axis=0)
        actions = np.stack([self._ar_training_action_array(example) for example in examples], axis=0)
        target_images = [example["image_target"] for example in examples]
        ar_inputs = self.qwen_vl_interface.build_reconvla_ar_training_inputs(
            images=self._reconvla_ar_images(examples, input_mode),
            instructions=[example["lang"] for example in examples],
            robot_obs=robot_obs,
            actions=actions,
            target_images=target_images,
            input_mode=input_mode,
            action_token_source=action_token_source,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        outputs = self.qwen_vl_interface(
            input_ids=ar_inputs["input_ids"],
            attention_mask=ar_inputs["attention_mask"],
            labels=ar_inputs["labels"],
            images=ar_inputs["images"].to(dtype=torch.bfloat16),
            target_images=ar_inputs["target_images"].to(dtype=torch.bfloat16),
            output_hidden_states=True,
            return_dict=True,
        )
        total = outputs.loss
        if total is None:
            raise RuntimeError("ReconVLA AR recon training returned no loss.")
        metrics = {
            "action_loss": total,
            "action_loss_ar": total.detach(),
            "reconvla_action_token_count": torch.as_tensor(
                float(sum(ar_inputs["action_token_counts"]) / len(ar_inputs["action_token_counts"])),
                device=total.device,
            ),
        }
        lm_loss = getattr(outputs, "lm_loss", None)
        vm_loss = getattr(outputs, "vm_loss", None)
        if lm_loss is not None:
            metrics["reconvla_lm_loss"] = lm_loss.detach()
        if vm_loss is not None:
            metrics["reconvla_vm_loss"] = vm_loss.detach()
        return metrics
```

- [ ] **Step 5: Branch in `forward()`**

At the start of `AuxVLAGR00T.forward()` add:

```python
        if self._reconvla_training_mode() == "reconvla_ar_recon":
            return self._forward_reconvla_ar_recon(examples)
```

If the mode is unknown and not `gr00t`, raise:

```python
        training_mode = self._reconvla_training_mode()
        if training_mode not in {"gr00t", "reconvla_ar_recon"}:
            raise ValueError(
                f"Unsupported framework.reconvla.training_mode={training_mode!r}. "
                "Expected `gr00t` or `reconvla_ar_recon`."
            )
```

- [ ] **Step 6: Run forward tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/framework/test_auxvla_gr00t_static.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add starVLA/model/framework/VLM4A/AuxVLAGR00T.py tests/framework/test_auxvla_gr00t_static.py
git commit -m "feat: train AuxVLAGR00T with ReconVLA AR recon loss"
```

---

### Task 4: Add No-Save CUDA Smoke Probe

**Files:**
- Create: `tools/probes/smoke_auxvla_reconvla_ar_recon_no_save.py`
- Test: `tests/tools/test_smoke_auxvla_reconvla_ar_recon_no_save.py`

- [ ] **Step 1: Write CLI/static test**

Create `tests/tools/test_smoke_auxvla_reconvla_ar_recon_no_save.py`:

```python
from pathlib import Path


def test_ar_recon_smoke_script_exists_and_mentions_required_checks():
    path = Path("tools/probes/smoke_auxvla_reconvla_ar_recon_no_save.py")
    assert path.exists()
    text = path.read_text()
    for needle in [
        "reconvla_ar_recon",
        "action_token_count",
        "reconvla_lm_loss",
        "reconvla_vm_loss",
        "mm_inv_projector_lora",
        "SMOKE_OK",
    ]:
        assert needle in text
```

- [ ] **Step 2: Run failing static test**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/tools/test_smoke_auxvla_reconvla_ar_recon_no_save.py -q
```

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Create smoke script**

Create `tools/probes/smoke_auxvla_reconvla_ar_recon_no_save.py` with this structure:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from tools.probes.smoke_auxvla_gr00t_no_save import (
    count_named_parameters,
    ensure_single_process_group,
    grad_norm_named,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", default="starVLA/config/training/auxvla_reconvla_ar_recon.yaml")
    parser.add_argument("--data_root_dir", default="datasets/calvin2uam")
    parser.add_argument("--data_mix", default="uamvla_calvin_abc")
    parser.add_argument("--action_type", default="calvin_rel_action")
    parser.add_argument("--model_path", default="ckpt/pretrain-checkpoint-10388")
    parser.add_argument("--vision_tower_path", default="ckpt/siglip-so400m-patch14-384")
    parser.add_argument("--output_dir", default="/tmp/auxvla_reconvla_ar_recon_no_save_smoke")
    parser.add_argument("--master_port", type=int, default=29639)
    return parser.parse_args()
```

The script should:

```python
cfg.framework.reconvla.training_mode = "reconvla_ar_recon"
cfg.framework.reconvla.inference_mode = "reconvla_ar_normalized"
cfg.framework.reconvla.ar_input_mode = "official_compose"
cfg.framework.reconvla.disable_internal_recon_loss = False
cfg.framework.reconvla.lora.enabled = True
cfg.framework.reconvla.lora.train_mm_projector = True
cfg.framework.reconvla.lora.train_mm_inv_projector = True
cfg.framework.action_model.action_horizon = 5
cfg.framework.action_model.future_action_window_size = 4
cfg.framework.aux_heads.recon.enabled = False
cfg.datasets.vla_data.data_mix = args.data_mix
cfg.datasets.vla_data.per_device_batch_size = 1
cfg.trainer.freeze_modules = None
```

Then build model, fetch one sample from `LeRobotMixtureDataset`, run:

```python
out = model([sample], global_step=0)
loss = out["action_loss"]
loss.backward()
```

Print and assert:

```text
training_mode=reconvla_ar_recon
data_mix=uamvla_calvin_abc
action_horizon=5
action_token_count=35
reconvla_lm_loss=<finite>
reconvla_vm_loss=<finite>
lora_grad_norm>0
mm_inv_projector_lora_grad_norm>0
action_head_grad_count=0
SMOKE_OK
```

- [ ] **Step 4: Run static test**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/tools/test_smoke_auxvla_reconvla_ar_recon_no_save.py -q
```

Expected: PASS.

- [ ] **Step 5: Run existing smoke tests that do not require GPU**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/tools/test_smoke_auxvla_reconvla_ar_recon_no_save.py \
  tests/tools/test_prepare_reconvla_ar_diagnostic_run.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add tools/probes/smoke_auxvla_reconvla_ar_recon_no_save.py tests/tools/test_smoke_auxvla_reconvla_ar_recon_no_save.py
git commit -m "test: add ReconVLA AR recon smoke probe"
```

---

### Task 5: Add Dedicated Training Config And Final Verification

**Files:**
- Create: `starVLA/config/training/auxvla_reconvla_ar_recon.yaml`
- Modify: `examples/calvin/train_files/run_auxvla_gr00t_lora_train.sh` only if needed for comments; avoid behavioral changes.
- Test: `tests/dataloader/test_uamvla_libero_registry.py` or new `tests/config/test_auxvla_reconvla_ar_recon_config.py`

- [ ] **Step 1: Add config test**

Create `tests/config/test_auxvla_reconvla_ar_recon_config.py`:

```python
from omegaconf import OmegaConf


def test_auxvla_reconvla_ar_recon_yaml_loads():
    cfg = OmegaConf.load("starVLA/config/training/auxvla_reconvla_ar_recon.yaml")
    assert cfg.framework.name == "AuxVLAGR00T"
    assert cfg.framework.reconvla.training_mode == "reconvla_ar_recon"
    assert cfg.framework.reconvla.inference_mode == "reconvla_ar_normalized"
    assert cfg.framework.reconvla.disable_internal_recon_loss is False
    assert cfg.framework.reconvla.lora.enabled is True
    assert cfg.framework.reconvla.lora.train_mm_projector is True
    assert cfg.framework.reconvla.lora.train_mm_inv_projector is True
    assert cfg.framework.action_model.action_horizon == 5
    assert cfg.framework.action_model.future_action_window_size == 4
    assert cfg.datasets.vla_data.data_mix == "uamvla_calvin_abc"
    assert cfg.framework.aux_heads.recon.enabled is False
```

- [ ] **Step 2: Run failing config test**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/config/test_auxvla_reconvla_ar_recon_config.py -q
```

Expected: FAIL because the config does not exist.

- [ ] **Step 3: Create config**

Create `starVLA/config/training/auxvla_reconvla_ar_recon.yaml` by copying `auxvla_gr00t_lora.yaml` and changing only these values:

```yaml
run_id: auxvla_reconvla_ar_recon_calvin_abc_h5

framework:
  name: AuxVLAGR00T
  reconvla:
    model_path: ckpt/pretrain-checkpoint-10388
    vision_tower_path: ckpt/siglip-so400m-patch14-384
    training_mode: reconvla_ar_recon
    inference_mode: reconvla_ar_normalized
    ar_input_mode: official_compose
    action_stat_path: third_party/ReconVLA/reconvla/statistics.yaml
    disable_internal_recon_loss: false
    lora:
      enabled: true
      r: 32
      lora_alpha: 16
      lora_dropout: 0.0
      init_lora_weights: gaussian
      bias: none
      task_type: CAUSAL_LM
      train_mm_projector: true
      train_mm_inv_projector: true
      target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
        - gate_proj
        - up_proj
        - down_proj
  action_model:
    action_horizon: 5
    future_action_window_size: 4

datasets:
  vla_data:
    data_root_dir: datasets/calvin2uam
    data_mix: uamvla_calvin_abc
    action_type: calvin_rel_action
    include_state: true
    image_resize: 384

trainer:
  freeze_modules: null
  learning_rate:
    base: 1.0e-4
    qwen_vl_interface: 2.0e-5
    action_model: 0.0
```

Keep `aux_heads.recon.enabled: false`; official recon comes from internal `vm_loss`.

- [ ] **Step 4: Run config test**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/config/test_auxvla_reconvla_ar_recon_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Run full static verification**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/dataloader/test_preserve_raw_transform.py \
  tests/framework/test_auxvla_gr00t_static.py \
  tests/framework/test_auxvla_gr00t_interface.py \
  tests/tools/test_smoke_auxvla_reconvla_ar_recon_no_save.py \
  tests/config/test_auxvla_reconvla_ar_recon_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add starVLA/config/training/auxvla_reconvla_ar_recon.yaml tests/config/test_auxvla_reconvla_ar_recon_config.py
git commit -m "config: add ReconVLA AR recon training config"
```

---

### Task 6: Remote Smoke And Training Commands

**Files:**
- No code files unless the remote smoke reveals a bug.

- [ ] **Step 1: Run no-save smoke on remote single GPU**

Command:

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/probes/smoke_auxvla_reconvla_ar_recon_no_save.py \
  --config_yaml starVLA/config/training/auxvla_reconvla_ar_recon.yaml \
  --data_root_dir datasets/calvin2uam \
  --data_mix uamvla_calvin_abc \
  --action_type calvin_rel_action \
  --model_path ckpt/pretrain-checkpoint-10388 \
  --vision_tower_path ckpt/siglip-so400m-patch14-384 \
  --output_dir /tmp/auxvla_reconvla_ar_recon_no_save_smoke
```

Expected output contains:

```text
training_mode=reconvla_ar_recon
action_horizon=5
action_token_count=35
reconvla_lm_loss=<finite>
reconvla_vm_loss=<finite>
mm_inv_projector_lora_grad_norm=<positive>
SMOKE_OK
```

- [ ] **Step 2: Run 8-card short trainer smoke**

Command:

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_GPUS=8 \
MASTER_PORT=29691 \
CONFIG_YAML=starVLA/config/training/auxvla_reconvla_ar_recon.yaml \
RUN_ID=auxvla_reconvla_ar_recon_calvin_abc_h5_bs512_pbs8_smoke \
RUN_ROOT_DIR=playground/Checkpoints \
DATA_ROOT_DIR=datasets/calvin2uam \
DATA_MIX=uamvla_calvin_abc \
ACTION_TYPE=calvin_rel_action \
PER_DEVICE_BATCH_SIZE=8 \
GRAD_ACCUM=8 \
MAX_TRAIN_STEPS=500 \
NUM_WARMUP_STEPS=50 \
SAVE_INTERVAL=500 \
LOGGING_FREQUENCY=25 \
WANDB_MODE=offline \
UAMVLA_TRAIN_STEP_PROFILE=0 \
bash examples/calvin/train_files/run_auxvla_gr00t_lora_train.sh \
  > logs/auxvla_reconvla_ar_recon_calvin_abc_h5_bs512_pbs8_smoke.log 2>&1
```

Expected:

```text
Total batch size = 512
loss/reconvla_lm_loss finite
loss/reconvla_vm_loss finite
no NaN/Inf
checkpoint saved at step 500
```

- [ ] **Step 3: Full training command if short trainer smoke is healthy**

Command:

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_GPUS=8 \
MASTER_PORT=29692 \
CONFIG_YAML=starVLA/config/training/auxvla_reconvla_ar_recon.yaml \
RUN_ID=auxvla_reconvla_ar_recon_calvin_abc_h5_bs512_pbs8 \
RUN_ROOT_DIR=playground/Checkpoints \
DATA_ROOT_DIR=datasets/calvin2uam \
DATA_MIX=uamvla_calvin_abc \
ACTION_TYPE=calvin_rel_action \
PER_DEVICE_BATCH_SIZE=8 \
GRAD_ACCUM=8 \
MAX_TRAIN_STEPS=10000 \
NUM_WARMUP_STEPS=500 \
SAVE_INTERVAL=2500 \
LOGGING_FREQUENCY=50 \
WANDB_MODE=offline \
UAMVLA_TRAIN_STEP_PROFILE=0 \
bash examples/calvin/train_files/run_auxvla_gr00t_lora_train.sh \
  > logs/auxvla_reconvla_ar_recon_calvin_abc_h5_bs512_pbs8.log 2>&1
```

For `uamvla_calvin_abc` length about `1,046,099`, total batch size 512 gives about 2043 optimizer steps per epoch. `MAX_TRAIN_STEPS=10000` is about 4.9 epochs and is enough for the first official-recipe baseline.

- [ ] **Step 4: Eval command after training**

Use the final model with the already validated AR diagnostic eval path:

```bash
mkdir -p logs

EGL_VISIBLE_DEVICE=0 \
PORT=5696 \
CKPT_PATH=playground/Checkpoints/auxvla_reconvla_ar_recon_calvin_abc_h5_bs512_pbs8/final_model/pytorch_model.pt \
DATASET_PATH=datasets/uam_dataset/calvin/task_ABC_D \
CALVIN_CONFIG_PATH=/inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UamVLA/third_party/calvin/calvin_models/conf \
NUM_SEQUENCES=1000 \
LOG_DIR=logs/calvin_auxvla_reconvla_ar_recon_h5 \
bash examples/calvin/eval_files/eval_calvin.sh \
  > logs/eval_calvin_auxvla_reconvla_ar_recon_h5.log 2>&1
```

- [ ] **Step 5: Final commit if remote smoke reveals no code changes**

No commit needed if Tasks 1-5 are already committed and remote smoke passes. If remote smoke reveals a bug, fix with a focused test and commit:

```bash
git add <changed-files>
git commit -m "fix: stabilize ReconVLA AR recon training smoke"
```

---

## Self-Review Checklist

- Spec coverage:
  - Official AR action-token training is covered by Tasks 2 and 3.
  - Official internal recon `vm_loss` is covered by Tasks 3 and 4.
  - Raw 15-D robot state and raw action preservation are covered by Task 1.
  - LoRA on language/mm_projector/mm_inv_projector is covered by Tasks 4 and 5.
  - `uamvla_calvin_abc` with horizon 5 is covered by Tasks 5 and 6.
- No global behavior changes:
  - Default `training_mode` remains `gr00t`.
  - Existing GR00T forward path remains the fallback branch.
  - StarVLA `aux_heads.recon` remains disabled in official-recipe config.
- Known risk:
  - If pretrain checkpoint lacks usable `pixel_decoder` or `mm_inv_projector`, the setup must fail early rather than silently training AR CE only.
