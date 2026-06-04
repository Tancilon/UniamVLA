# AuxVLAGR00T ReconVLA Online Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose ReconVLA-style crop-plus-wrist reconstruction targets online in AuxVLAGR00T while preserving the GR00T action head and policy eval path.

**Architecture:** Add a small AuxVLAGR00T-local helper that converts the existing crop sidecar plus wrist image into a `(3, 384, 384)` tensor using ReconVLA's CALVIN vertical ratio. Call it only when the recon target is consumed by training or visualization, so missing sidecars fail early for recon training but do not break action-only inference/evaluation.

**Tech Stack:** Python, PyTorch, PIL, pytest, existing AuxVLAGR00T static tests, existing no-save smoke probe.

---

## File Structure

- Modify `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`
  - Add local constants for ReconVLA CALVIN target layout.
  - Add `_compose_reconvla_style_image_target`.
  - Extend `_prepare_examples` with a `require_reconvla_target` flag.
  - Use the flag from `forward` and visualization context, not from `predict_action`.
- Modify `tests/framework/test_auxvla_gr00t_static.py`
  - Add fast unit tests for crop-plus-wrist composition and hard failure behavior.
  - Add a regression test that `predict_action` can still use prepared examples without requiring `image_target`.
- Modify `tools/probes/smoke_auxvla_gr00t_no_save.py`
  - Print prepared `image_target` shape when present.
  - If recon is enabled in config, request the composed target and assert `(1, 3, 384, 384)`.

## Task 1: Add Failing Static Tests

**Files:**
- Modify: `tests/framework/test_auxvla_gr00t_static.py`

- [ ] **Step 1: Add tests near the existing `_select_single_view` tests**

Insert this block after `test_select_single_view_rejects_missing_wrist`:

```python
def test_reconvla_style_image_target_composes_crop_and_wrist(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)

    crop = torch.zeros(3, 2, 2)
    crop[0].fill_(1.0)
    primary = Image.new("RGB", (2, 2), (0, 255, 0))
    wrist = Image.new("RGB", (2, 2), (0, 0, 255))
    example = {
        "image": [primary, wrist],
        "image_target": crop,
    }

    out = module.AuxVLAGR00T._compose_reconvla_style_image_target(model, example)

    assert out is not example
    assert out["image_target"].shape == (3, 384, 384)
    assert out["image_target"].dtype == torch.float32
    assert out["image_target"].min().item() >= 0.0
    assert out["image_target"].max().item() <= 1.0

    crop_height = 384 * 14 // 27
    top = out["image_target"][:, :crop_height]
    bottom = out["image_target"][:, crop_height:]
    assert top[0].mean().item() > 0.99
    assert top[1].mean().item() < 0.01
    assert top[2].mean().item() < 0.01
    assert bottom[0].mean().item() < 0.01
    assert bottom[1].mean().item() < 0.01
    assert bottom[2].mean().item() > 0.99
    assert torch.equal(example["image_target"], crop)


def test_reconvla_style_image_target_requires_crop(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    example = {
        "image": [
            Image.new("RGB", (2, 2), "red"),
            Image.new("RGB", (2, 2), "blue"),
        ],
    }

    with pytest.raises(RuntimeError, match="image_target"):
        module.AuxVLAGR00T._compose_reconvla_style_image_target(model, example)


def test_reconvla_style_image_target_requires_wrist(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    example = {
        "image": [Image.new("RGB", (2, 2), "red")],
        "image_target": torch.zeros(3, 2, 2),
    }

    with pytest.raises(RuntimeError, match="wrist"):
        module.AuxVLAGR00T._compose_reconvla_style_image_target(model, example)


def test_prepare_examples_can_require_reconvla_target(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    crop = torch.ones(3, 2, 2)
    sample = {
        "image": [
            Image.new("RGB", (2, 2), "black"),
            Image.new("RGB", (2, 2), "blue"),
        ],
        "image_target": crop,
        "lang": "open drawer",
        "action": torch.zeros(8, 7),
    }

    out = module.AuxVLAGR00T._prepare_examples(
        model,
        [sample],
        require_reconvla_target=True,
    )

    assert out[0]["image_target"].shape == (3, 384, 384)
    assert sample["image_target"].shape == (3, 2, 2)
```

- [ ] **Step 2: Run tests and verify they fail for missing method/signature**

Run:

```bash
pytest tests/framework/test_auxvla_gr00t_static.py \
  -k "reconvla_style_image_target or prepare_examples_can_require_reconvla_target" \
  -q
```

Expected: FAIL because `_compose_reconvla_style_image_target` does not exist and `_prepare_examples` does not accept `require_reconvla_target`.

## Task 2: Implement Online Target Composition

**Files:**
- Modify: `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`

- [ ] **Step 1: Add constants after `logger = logging.getLogger(__name__)`**

```python
_RECONVLA_CALVIN_TARGET_SIZE = 384
_RECONVLA_CALVIN_CROP_NUMERATOR = 14
_RECONVLA_CALVIN_CROP_DENOMINATOR = 27
```

- [ ] **Step 2: Add helper methods before `_prepare_examples`**

Insert this code immediately before the existing `_prepare_examples` method:

```python
    def _compose_reconvla_style_image_target(self, example: dict) -> dict:
        """Build ReconVLA CALVIN target_image: top crop + bottom wrist image."""
        if "image_target" not in example:
            raise RuntimeError(
                "AuxVLAGR00T recon training requires `image_target` sidecar crop "
                "to compose ReconVLA-style crop-plus-wrist target."
            )
        image_list = example.get("image")
        if not isinstance(image_list, (list, tuple)) or len(image_list) < 2:
            raise RuntimeError(
                "AuxVLAGR00T recon training requires a wrist image at "
                "`example['image'][1]` to compose ReconVLA-style target."
            )

        crop = example["image_target"]
        if not torch.is_tensor(crop):
            crop = torch.as_tensor(np.asarray(crop), dtype=torch.float32)
        if crop.ndim != 3 or crop.shape[0] not in {1, 3, 4}:
            raise RuntimeError(
                "AuxVLAGR00T recon target crop must be CHW with 1, 3, or 4 "
                f"channels, got shape {tuple(crop.shape)}."
            )

        target_size = _RECONVLA_CALVIN_TARGET_SIZE
        crop_height = target_size * _RECONVLA_CALVIN_CROP_NUMERATOR // _RECONVLA_CALVIN_CROP_DENOMINATOR
        wrist_height = target_size - crop_height
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

        crop_img = _to_rgb_pil(crop).resize((target_size, crop_height), resample)
        wrist_img = _to_rgb_pil(image_list[1]).resize((target_size, wrist_height), resample)
        combined = Image.new("RGB", (target_size, target_size))
        combined.paste(crop_img, (0, 0))
        combined.paste(wrist_img, (0, crop_height))

        arr = np.array(combined, dtype=np.uint8)
        out = dict(example)
        out["image_target"] = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return out
```

- [ ] **Step 3: Extend `_prepare_examples`**

Replace the existing method:

```python
    def _prepare_examples(self, examples: List[dict]) -> List[dict]:
        return [
            self._unpack_lerobot_sample(example)
            if "__trajectory_id" in example
            else example
            for example in examples
        ]
```

with:

```python
    def _prepare_examples(
        self,
        examples: List[dict],
        require_reconvla_target: bool = False,
    ) -> List[dict]:
        prepared = [
            self._unpack_lerobot_sample(example)
            if "__trajectory_id" in example
            else example
            for example in examples
        ]
        if require_reconvla_target:
            prepared = [
                self._compose_reconvla_style_image_target(example)
                for example in prepared
            ]
        return prepared
```

- [ ] **Step 4: Request composed target from training forward**

Replace in `forward`:

```python
        examples = self._prepare_examples(examples)
```

with:

```python
        examples = self._prepare_examples(
            examples,
            require_reconvla_target="recon" in getattr(self, "aux_heads", {}),
        )
```

This preserves action-only training and enforces the hard requirement when the recon head is enabled.

- [ ] **Step 5: Request composed target from visualization context only when recon visualization may consume it**

In `_visualization_forward_context`, replace:

```python
        examples = self._prepare_examples(selected)
```

with:

```python
        examples = self._prepare_examples(
            selected,
            require_reconvla_target="recon" in getattr(self, "aux_heads", {}),
        )
```

Do not change `predict_action`; CALVIN online eval does not have `image_target` sidecars and should remain action-only.

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/framework/test_auxvla_gr00t_static.py \
  -k "reconvla_style_image_target or prepare_examples_can_require_reconvla_target" \
  -q
```

Expected: PASS.

- [ ] **Step 7: Run all AuxVLAGR00T static tests**

Run:

```bash
pytest tests/framework/test_auxvla_gr00t_static.py tests/framework/test_auxvla_gr00t_registry.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit implementation and tests**

```bash
git add starVLA/model/framework/VLM4A/AuxVLAGR00T.py tests/framework/test_auxvla_gr00t_static.py
git commit -m "feat: compose ReconVLA-style AuxVLAGR00T targets"
```

## Task 3: Update No-Save Smoke Diagnostics

**Files:**
- Modify: `tools/probes/smoke_auxvla_gr00t_no_save.py`

- [ ] **Step 1: Add a helper near `grad_examples_named`**

```python
def recon_head_enabled(cfg) -> bool:
    aux_heads = getattr(cfg.framework, "aux_heads", None)
    if aux_heads is None or "recon" not in aux_heads:
        return False
    return bool(aux_heads.recon.get("enabled", False))
```

- [ ] **Step 2: Request composed target when recon is enabled**

Replace:

```python
        examples = model._prepare_examples(batch)
```

with:

```python
        examples = model._prepare_examples(
            batch,
            require_reconvla_target=recon_head_enabled(cfg),
        )
        if "image_target" in examples[0]:
            image_targets = torch.stack([example["image_target"] for example in examples])
            print(f"prepared_image_target_shape={tuple(image_targets.shape)}")
            if recon_head_enabled(cfg):
                assert tuple(image_targets.shape) == (1, 3, 384, 384), (
                    "Recon-enabled AuxVLAGR00T smoke expects online "
                    f"crop-plus-wrist image_target, got {tuple(image_targets.shape)}"
                )
```

- [ ] **Step 3: Run smoke script help locally**

Run:

```bash
python tools/probes/smoke_auxvla_gr00t_no_save.py --help >/tmp/auxvla_smoke_help.txt
```

Expected: exits 0 and writes usage text. This does not touch GPUs.

- [ ] **Step 4: Run static tests again**

Run:

```bash
pytest tests/framework/test_auxvla_gr00t_static.py tests/framework/test_auxvla_gr00t_registry.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit smoke diagnostics**

```bash
git add tools/probes/smoke_auxvla_gr00t_no_save.py
git commit -m "test: report AuxVLAGR00T online recon target in smoke"
```

## Task 4: Remote Verification Commands

**Files:**
- No local file changes.

- [ ] **Step 1: Push the branch**

```bash
git push
```

Expected: remote branch receives the two implementation commits.

- [ ] **Step 2: Run remote no-save smoke with recon enabled**

On the remote server, run a single-GPU smoke using the existing environment and dataset paths:

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/probes/smoke_auxvla_gr00t_no_save.py \
  --config_yaml starVLA/config/training/auxvla_gr00t_lora.yaml \
  --data_root_dir datasets/calvin2uam \
  --data_mix uamvla_calvin_abc_h8 \
  --action_type calvin_rel_action \
  --model_path ckpt/pretrain-checkpoint-10388 \
  --vision_tower_path ckpt/siglip-so400m-patch14-384 \
  --enable_lora
```

Expected output includes:

```text
prepared_image_target_shape=(1, 3, 384, 384)
hidden_shape=(1, 737, 3584)
actions_target_shape=(1, 8, 7)
SMOKE_OK
```

If the config keeps `framework.aux_heads.recon.enabled=false`, temporarily add this CLI override only if the smoke runner supports OmegaConf passthrough. If it does not, set recon enabled in a copied remote config for smoke only and do not commit that copied config.

- [ ] **Step 3: Run a trainer smoke before long training**

Use the same 8-card short-step trainer command style as prior runs, with recon enabled and logs redirected to file. Expected behavior:

- no missing `image_target` or missing wrist errors
- recon loss finite
- visualization GT shows top crop and bottom wrist
- action loss remains finite

## Self-Review

- Spec coverage: The plan implements online crop-plus-wrist target composition, keeps GR00T action head unchanged, avoids dataset rewrites, keeps the behavior AuxVLAGR00T-local, and preserves hard failure when recon consumes missing target inputs.
- Scope refinement: The plan intentionally does not require `image_target` in `predict_action`, because CALVIN online eval is action-only and does not provide sidecar targets. This keeps the "mandatory" requirement scoped to recon training/visualization where `image_target` is actually consumed.
- Placeholder scan: No task contains unfinished markers or unspecified error handling.
- Type consistency: The composed target remains a `torch.float32` tensor with shape `(3, 384, 384)` and value range `[0, 1]`, matching existing `ReconHead` input expectations.
