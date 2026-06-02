# AuxVLAGR00T LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a PEFT LoRA baseline for `AuxVLAGR00T` that trains GR00T action head parameters plus Qwen2/Ross language-backbone LoRA adapters while keeping ReconVLA vision/projector components frozen.

**Architecture:** LoRA is implemented inside `AuxVLAGR00T` and `ReconVLAInterface`, not in the generic trainer. `framework.reconvla.lora` controls PEFT wrapping, `AuxVLAGR00T.get_lr_groups()` routes only trainable action/LoRA/aux parameters into optimizer groups, and the no-save smoke probe gains LoRA trainable/gradient assertions.

**Tech Stack:** PyTorch, PEFT (`LoraConfig`, `get_peft_model`, `TaskType`), OmegaConf, Accelerate/DeepSpeed trainer, pytest.

---

## File Structure

- Modify `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`
  - Add LoRA defaults under `AuxVLAGR00TDefaultConfig.reconvla`.
  - Add small config helpers.
  - Add `ReconVLAInterface.apply_language_lora()`.
  - Freeze vision/projector modules when LoRA is enabled.
  - Add `AuxVLAGR00T.get_lr_groups()`.

- Modify `tests/framework/test_auxvla_gr00t_static.py`
  - Test default LoRA config.
  - Test `get_lr_groups()` routing and freeze-module safety.
  - Test LoRA trainable validation helpers without importing real PEFT.

- Modify `tests/framework/test_auxvla_gr00t_interface.py`
  - Add fake PEFT modules.
  - Test `ReconVLAInterface.apply_language_lora()`.
  - Test vision/projector freezing.
  - Test failure when PEFT produces no trainable LoRA params.

- Modify `tools/probes/smoke_auxvla_gr00t_no_save.py`
  - Add `--enable_lora`.
  - Print LoRA/base/projector trainable counts.
  - Assert LoRA gradients when LoRA is enabled.

- Create `starVLA/config/training/auxvla_gr00t_lora_libero.yaml`
  - Separate LoRA baseline config derived from the existing AuxVLAGR00T config.

- Modify `tests/dataloader/test_uamvla_libero_registry.py`
  - Add a YAML load test for the new LoRA config.

---

### Task 1: Add LoRA Defaults

**Files:**
- Modify: `tests/framework/test_auxvla_gr00t_static.py`
- Modify: `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`

- [ ] **Step 1: Write the failing test**

Add this test after `test_auxvla_gr00t_registers_framework` in
`tests/framework/test_auxvla_gr00t_static.py`:

```python
def test_auxvla_lora_defaults_are_disabled(monkeypatch):
    module = _load_auxvla_module(monkeypatch)

    default_cfg = module.AuxVLAGR00TDefaultConfig()
    lora_cfg = default_cfg.reconvla["lora"]

    assert lora_cfg["enabled"] is False
    assert lora_cfg["r"] == 16
    assert lora_cfg["lora_alpha"] == 32
    assert lora_cfg["lora_dropout"] == 0.05
    assert lora_cfg["bias"] == "none"
    assert lora_cfg["task_type"] == "CAUSAL_LM"
    assert lora_cfg["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py::test_auxvla_lora_defaults_are_disabled -q
```

Expected: FAIL with `KeyError: 'lora'`.

- [ ] **Step 3: Add the default LoRA config**

In `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`, update the
`AuxVLAGR00TDefaultConfig.reconvla` default dict to include:

```python
"lora": {
    "enabled": False,
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
},
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py::test_auxvla_lora_defaults_are_disabled -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add starVLA/model/framework/VLM4A/AuxVLAGR00T.py tests/framework/test_auxvla_gr00t_static.py
git commit -m "config: add AuxVLAGR00T LoRA defaults"
```

---

### Task 2: Implement ReconVLAInterface LoRA Wrapping

**Files:**
- Modify: `tests/framework/test_auxvla_gr00t_interface.py`
- Modify: `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`

- [ ] **Step 1: Extend the fake ReconVLA model**

In `tests/framework/test_auxvla_gr00t_interface.py`, replace `_FakeVisionTower`
and `_FakeReconModel` with versions that expose trainable modules:

```python
class _FakeVisionTower(torch.nn.Module):
    is_loaded = True
    image_processor = _FakeImageProcessor()

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)

    def load_model(self, device_map=None):
        self.loaded_device_map = device_map


class _FakeInnerReconModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_projector = torch.nn.Linear(2, 2)
        self.mm_inv_projector = torch.nn.Linear(2, 2)
        self.q_proj = torch.nn.Linear(2, 2)
        self.k_proj = torch.nn.Linear(2, 2)
        self.v_proj = torch.nn.Linear(2, 2)
        self.o_proj = torch.nn.Linear(2, 2)
        self.gate_proj = torch.nn.Linear(2, 2)
        self.up_proj = torch.nn.Linear(2, 2)
        self.down_proj = torch.nn.Linear(2, 2)


class _FakeReconModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(
            hidden_size=3584,
            image_embed_len=729,
            recon_enable=True,
            reconstruct_image=True,
        )
        self.vision_tower = _FakeVisionTower()
        self.model = _FakeInnerReconModel()
        self.forward_kwargs = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        model = cls()
        model.path = path
        model.from_pretrained_kwargs = kwargs
        if "config" in kwargs:
            model.config = kwargs["config"]
        return model

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.vision_tower

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        hidden = torch.ones(1, 5, 3584)
        return types.SimpleNamespace(hidden_states=[hidden], boi_ids=[1], eoi_ids=[3])
```

- [ ] **Step 2: Add fake PEFT installer**

Add this helper below `_install_reconvla_fakes()`:

```python
def _install_fake_peft(monkeypatch, add_lora_param=True):
    fake_peft = types.ModuleType("peft")

    class _FakeTaskType:
        CAUSAL_LM = "CAUSAL_LM"

    class _FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.target_modules = kwargs["target_modules"]

    def _fake_get_peft_model(model, lora_config):
        for param in model.parameters():
            param.requires_grad_(False)
        model.peft_config_seen = lora_config
        if add_lora_param:
            model.lora_A = torch.nn.Parameter(torch.ones(1))
        return model

    fake_peft.TaskType = _FakeTaskType
    fake_peft.LoraConfig = _FakeLoraConfig
    fake_peft.get_peft_model = _fake_get_peft_model
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    return fake_peft
```

- [ ] **Step 3: Write the failing LoRA application test**

Add this test after `test_reconvla_interface_overrides_local_vision_tower_path`:

```python
def test_reconvla_interface_applies_lora_and_freezes_multimodal(monkeypatch):
    module = _load_module(monkeypatch)
    _install_reconvla_fakes(monkeypatch)
    _install_fake_peft(monkeypatch)

    cfg = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(
                model_path="ckpt/pretrain-checkpoint-10388",
                vision_tower_path="ckpt/siglip-so400m-patch14-384",
                lora=_AttrDict(
                    enabled=True,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                ),
            )
        )
    )

    interface = module.ReconVLAInterface(cfg)
    interface.apply_language_lora(cfg.framework.reconvla.lora)

    assert interface.lora_enabled is True
    assert interface.model.peft_config_seen.kwargs["r"] == 16
    assert interface.model.peft_config_seen.kwargs["lora_alpha"] == 32
    assert interface.model.peft_config_seen.kwargs["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]

    trainable = [name for name, param in interface.model.named_parameters() if param.requires_grad]
    assert trainable == ["lora_A"]
    assert all(not p.requires_grad for p in interface.model.get_vision_tower().parameters())
    assert all(not p.requires_grad for p in interface.model.get_model().mm_projector.parameters())
    assert all(not p.requires_grad for p in interface.model.get_model().mm_inv_projector.parameters())
```

- [ ] **Step 4: Write the failing no-trainable-LoRA test**

Add this test after the previous one:

```python
def test_reconvla_interface_rejects_lora_without_trainable_adapters(monkeypatch):
    module = _load_module(monkeypatch)
    _install_reconvla_fakes(monkeypatch)
    _install_fake_peft(monkeypatch, add_lora_param=False)

    cfg = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(
                model_path="ckpt/pretrain-checkpoint-10388",
                lora=_AttrDict(
                    enabled=True,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj"],
                ),
            )
        )
    )

    interface = module.ReconVLAInterface(cfg)
    with pytest.raises(RuntimeError, match="No trainable LoRA parameters"):
        interface.apply_language_lora(cfg.framework.reconvla.lora)
```

Also add `import pytest` near the top of
`tests/framework/test_auxvla_gr00t_interface.py`.

- [ ] **Step 5: Run the tests to verify they fail**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_interface.py::test_reconvla_interface_applies_lora_and_freezes_multimodal \
  tests/framework/test_auxvla_gr00t_interface.py::test_reconvla_interface_rejects_lora_without_trainable_adapters -q
```

Expected: FAIL because `ReconVLAInterface.apply_language_lora` does not exist.

- [ ] **Step 6: Add config/helper functions**

In `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`, add these helpers below
`_ensure_reconvla_pythonpath()`:

```python
def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_to_plain_dict(cfg) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return {key: _cfg_to_plain_dict(value) for key, value in cfg.items()}
    if hasattr(cfg, "items"):
        return {key: _cfg_to_plain_dict(value) for key, value in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_cfg_to_plain_dict(value) for value in cfg]
    return cfg


def _freeze_module(module) -> None:
    if module is None or not hasattr(module, "parameters"):
        return
    for param in module.parameters():
        param.requires_grad_(False)
```

- [ ] **Step 7: Implement `ReconVLAInterface.apply_language_lora`**

In `ReconVLAInterface.__init__`, set:

```python
self.lora_enabled = False
```

Add these methods to `ReconVLAInterface`:

```python
def _inner_reconvla_model(self):
    if hasattr(self.model, "get_model"):
        return self.model.get_model()
    base_model = getattr(self.model, "base_model", None)
    if base_model is not None:
        wrapped = getattr(base_model, "model", None)
        if wrapped is not None and hasattr(wrapped, "get_model"):
            return wrapped.get_model()
        if wrapped is not None and hasattr(wrapped, "model"):
            return wrapped.model
    wrapped_model = getattr(self.model, "model", None)
    if wrapped_model is not None and hasattr(wrapped_model, "model"):
        return wrapped_model.model
    return wrapped_model

def _matched_lora_targets(self, target_modules: list[str]) -> set[str]:
    targets = {str(target) for target in target_modules}
    matched: set[str] = set()
    for name, module in self.model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in targets:
            matched.add(leaf)
    return matched

def _freeze_non_language_multimodal_modules(self) -> None:
    model_for_components = self._inner_reconvla_model()
    vision_tower = None
    if hasattr(self.model, "get_vision_tower"):
        vision_tower = self.model.get_vision_tower()
    elif hasattr(model_for_components, "get_vision_tower"):
        vision_tower = model_for_components.get_vision_tower()
    _freeze_module(vision_tower)
    for attr in ("mm_projector", "mm_inv_projector"):
        _freeze_module(getattr(model_for_components, attr, None))

def lora_trainable_parameter_names(self) -> list[str]:
    return [
        name
        for name, param in self.model.named_parameters()
        if param.requires_grad and "lora_" in name.lower()
    ]

def apply_language_lora(self, lora_cfg) -> None:
    lora_cfg = _cfg_to_plain_dict(lora_cfg)
    if not bool(lora_cfg.get("enabled", False)):
        return
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "AuxVLAGR00T LoRA requires peft. Install peft or set "
            "framework.reconvla.lora.enabled=false."
        ) from exc

    target_modules = list(lora_cfg.get("target_modules") or [])
    if not target_modules:
        raise ValueError("framework.reconvla.lora.target_modules must be non-empty")
    matched_targets = self._matched_lora_targets(target_modules)
    if not matched_targets:
        raise ValueError(
            "No ReconVLA modules matched LoRA target_modules="
            f"{target_modules}. Expected Qwen2/Ross names like q_proj, k_proj, "
            "v_proj, o_proj, gate_proj, up_proj, down_proj."
        )

    task_type_name = str(lora_cfg.get("task_type", "CAUSAL_LM"))
    task_type = getattr(TaskType, task_type_name, task_type_name)
    peft_cfg = LoraConfig(
        task_type=task_type,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=target_modules,
        bias=str(lora_cfg.get("bias", "none")),
    )
    self.model = get_peft_model(self.model, peft_cfg)
    self._freeze_non_language_multimodal_modules()
    trainable_lora = self.lora_trainable_parameter_names()
    if not trainable_lora:
        raise RuntimeError(
            "No trainable LoRA parameters were created for AuxVLAGR00T. "
            "Check framework.reconvla.lora.target_modules."
        )
    self.lora_enabled = True
    logger.info("AuxVLAGR00T LoRA enabled with %d trainable adapter tensors", len(trainable_lora))
```

- [ ] **Step 8: Run the interface tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_interface.py::test_reconvla_interface_applies_lora_and_freezes_multimodal \
  tests/framework/test_auxvla_gr00t_interface.py::test_reconvla_interface_rejects_lora_without_trainable_adapters -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add starVLA/model/framework/VLM4A/AuxVLAGR00T.py tests/framework/test_auxvla_gr00t_interface.py
git commit -m "feat: add ReconVLA language LoRA wrapper"
```

---

### Task 3: Wire LoRA Into AuxVLAGR00T Initialization

**Files:**
- Modify: `tests/framework/test_auxvla_gr00t_static.py`
- Modify: `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`

- [ ] **Step 1: Add a fake ReconVLA interface with LoRA call tracking**

In `tests/framework/test_auxvla_gr00t_static.py`, add this class near
`_FakeActionModel`:

```python
class _FakeReconInterfaceForInit(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=3584))
        self.applied_lora_cfg = None

    def apply_language_lora(self, lora_cfg):
        self.applied_lora_cfg = lora_cfg
```

- [ ] **Step 2: Write the failing init wiring test**

Add this test after `test_auxvla_lora_defaults_are_disabled`:

```python
def test_auxvla_init_applies_reconvla_lora_when_enabled(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    monkeypatch.setattr(module, "ReconVLAInterface", _FakeReconInterfaceForInit)

    cfg = _AttrDict(
        framework=_AttrDict(
            reconvla=_AttrDict(lora=_AttrDict(enabled=True, r=8)),
            action_model=_AttrDict(
                action_horizon=8,
                diffusion_model_cfg=_AttrDict(cross_attention_dim=1),
            ),
            aux_loss_control=_AttrDict(enabled=False),
        ),
        datasets=_AttrDict(vla_data=_AttrDict()),
    )

    model = module.AuxVLAGR00T(cfg)

    assert model.qwen_vl_interface.applied_lora_cfg["enabled"] is True
    assert model.qwen_vl_interface.applied_lora_cfg["r"] == 8
    assert cfg.framework.action_model.diffusion_model_cfg.cross_attention_dim == 3584
```

- [ ] **Step 3: Run the test to verify it fails**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py::test_auxvla_init_applies_reconvla_lora_when_enabled -q
```

Expected: FAIL because `_init_reconvla_components()` does not call
`apply_language_lora()`.

- [ ] **Step 4: Wire LoRA into initialization**

In `AuxVLAGR00T._init_reconvla_components()`, immediately after:

```python
self.qwen_vl_interface = ReconVLAInterface(self.config)
```

add:

```python
lora_cfg = self.config.framework.get("reconvla", {}).get("lora", {})
if bool(_cfg_get(lora_cfg, "enabled", False)):
    self.qwen_vl_interface.apply_language_lora(lora_cfg)
```

- [ ] **Step 5: Run the init wiring test**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py::test_auxvla_init_applies_reconvla_lora_when_enabled -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add starVLA/model/framework/VLM4A/AuxVLAGR00T.py tests/framework/test_auxvla_gr00t_static.py
git commit -m "feat: enable AuxVLAGR00T ReconVLA LoRA"
```

---

### Task 4: Add LR Group Routing And Freeze Safety

**Files:**
- Modify: `tests/framework/test_auxvla_gr00t_static.py`
- Modify: `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`

- [ ] **Step 1: Add fake modules for LR group tests**

Add these classes near `_FakeActionModel` in
`tests/framework/test_auxvla_gr00t_static.py`:

```python
class _NamedParamModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))


class _FakeLoRAInterface(torch.nn.Module):
    lora_enabled = True

    def __init__(self):
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.ones(()))
        self.lora_B = torch.nn.Parameter(torch.ones(()))

    def lora_trainable_parameter_names(self):
        return [
            name
            for name, param in self.named_parameters()
            if param.requires_grad and "lora_" in name.lower()
        ]
```

- [ ] **Step 2: Write the failing LR group test**

Add this test after `test_auxvla_init_applies_reconvla_lora_when_enabled`:

```python
def test_auxvla_get_lr_groups_routes_lora_and_action_params(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    torch.nn.Module.__init__(model)
    model.qwen_vl_interface = _FakeLoRAInterface()
    model.action_model = torch.nn.Linear(2, 2)
    model.aux_heads = torch.nn.ModuleDict({"recon": _NamedParamModule()})
    model.config = _AttrDict(
        trainer=_AttrDict(freeze_modules=None),
        framework=_AttrDict(reconvla=_AttrDict(lora=_AttrDict(enabled=True))),
    )
    lr_cfg = _AttrDict(base=1.0e-4, qwen_vl_interface=1.0e-5, action_model=2.0e-4)

    groups = module.AuxVLAGR00T.get_lr_groups(model, lr_cfg)
    by_name = {group["name"]: group for group in groups}

    assert set(by_name) == {"action_model", "qwen_vl_interface", "base"}
    assert by_name["action_model"]["lr"] == 2.0e-4
    assert by_name["qwen_vl_interface"]["lr"] == 1.0e-5
    assert by_name["base"]["lr"] == 1.0e-4
    assert len(by_name["qwen_vl_interface"]["params"]) == 2
    assert all(param.requires_grad for param in by_name["qwen_vl_interface"]["params"])
    qwen_param_ids = {id(param) for param in by_name["qwen_vl_interface"]["params"]}
    assert id(model.qwen_vl_interface.base_weight) not in qwen_param_ids
```

- [ ] **Step 3: Write the failing freeze safety test**

Add this test after the LR group test:

```python
def test_auxvla_get_lr_groups_rejects_freezing_qwen_interface_with_lora(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = object.__new__(module.AuxVLAGR00T)
    torch.nn.Module.__init__(model)
    model.qwen_vl_interface = _FakeLoRAInterface()
    model.action_model = torch.nn.Linear(2, 2)
    model.aux_heads = torch.nn.ModuleDict()
    model.config = _AttrDict(
        trainer=_AttrDict(freeze_modules="qwen_vl_interface"),
        framework=_AttrDict(reconvla=_AttrDict(lora=_AttrDict(enabled=True))),
    )

    with pytest.raises(RuntimeError, match="Do not freeze qwen_vl_interface"):
        module.AuxVLAGR00T.get_lr_groups(
            model,
            _AttrDict(base=1.0e-4, qwen_vl_interface=1.0e-5, action_model=2.0e-4),
        )
```

- [ ] **Step 4: Run the tests to verify they fail**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py::test_auxvla_get_lr_groups_routes_lora_and_action_params \
  tests/framework/test_auxvla_gr00t_static.py::test_auxvla_get_lr_groups_rejects_freezing_qwen_interface_with_lora -q
```

Expected: FAIL because `AuxVLAGR00T.get_lr_groups()` does not exist.

- [ ] **Step 5: Implement LR group helpers**

Add these methods to `AuxVLAGR00T` before `forward()`:

```python
def _reconvla_lora_enabled(self) -> bool:
    recon_cfg = self.config.framework.get("reconvla", {})
    return bool(_cfg_get(_cfg_get(recon_cfg, "lora", {}), "enabled", False))

def _trainer_freeze_patterns(self) -> list[str]:
    trainer_cfg = getattr(self.config, "trainer", {}) or {}
    freeze_modules = _cfg_get(trainer_cfg, "freeze_modules", "")
    if not isinstance(freeze_modules, str):
        return []
    return [item.strip() for item in freeze_modules.split(",") if item.strip()]

def _frozen_param_ids_from_config(self) -> set[int]:
    frozen: set[int] = set()
    for path in self._trainer_freeze_patterns():
        module = self
        try:
            for attr in path.split("."):
                module = getattr(module, attr)
        except AttributeError:
            continue
        if hasattr(module, "parameters"):
            frozen.update(id(param) for param in module.parameters())
    return frozen

def get_lr_groups(self, lr_cfg):
    base_lr = float(_cfg_get(lr_cfg, "base", 1.0e-4))
    used: set[int] = set()
    groups: list[dict] = []
    frozen_ids = self._frozen_param_ids_from_config()

    freeze_patterns = self._trainer_freeze_patterns()
    freezes_qwen = any(
        pattern == "qwen_vl_interface" or pattern.startswith("qwen_vl_interface.")
        for pattern in freeze_patterns
    )
    if self._reconvla_lora_enabled() and freezes_qwen:
        raise RuntimeError(
            "Do not freeze qwen_vl_interface when framework.reconvla.lora.enabled=true; "
            "it would freeze the LoRA adapter parameters."
        )

    def add_group(name: str, module, lr: float, only_lora: bool = False) -> None:
        if module is None or not hasattr(module, "named_parameters"):
            return
        params = []
        for param_name, param in module.named_parameters():
            if id(param) in used or id(param) in frozen_ids or not param.requires_grad:
                continue
            if only_lora and "lora_" not in param_name.lower():
                continue
            params.append(param)
            used.add(id(param))
        if params:
            groups.append({"params": params, "lr": lr, "name": name})

    add_group(
        "action_model",
        self.action_model,
        float(_cfg_get(lr_cfg, "action_model", base_lr)),
    )

    if self._reconvla_lora_enabled():
        add_group(
            "qwen_vl_interface",
            self.qwen_vl_interface,
            float(_cfg_get(lr_cfg, "qwen_vl_interface", base_lr)),
            only_lora=True,
        )
        if not any(group["name"] == "qwen_vl_interface" for group in groups):
            raise RuntimeError(
                "AuxVLAGR00T LoRA is enabled but no trainable LoRA parameters reached "
                "the optimizer. Remove --trainer.freeze_modules qwen_vl_interface and "
                "check framework.reconvla.lora.target_modules."
            )
    else:
        add_group(
            "qwen_vl_interface",
            self.qwen_vl_interface,
            float(_cfg_get(lr_cfg, "qwen_vl_interface", base_lr)),
        )

    add_group("base", getattr(self, "aux_heads", None), base_lr)

    remaining = [
        param
        for param in self.parameters()
        if param.requires_grad and id(param) not in used and id(param) not in frozen_ids
    ]
    if remaining:
        groups.append({"params": remaining, "lr": base_lr, "name": "base"})

    return groups
```

- [ ] **Step 6: Run the LR group tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py::test_auxvla_get_lr_groups_routes_lora_and_action_params \
  tests/framework/test_auxvla_gr00t_static.py::test_auxvla_get_lr_groups_rejects_freezing_qwen_interface_with_lora -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add starVLA/model/framework/VLM4A/AuxVLAGR00T.py tests/framework/test_auxvla_gr00t_static.py
git commit -m "feat: route AuxVLAGR00T LoRA optimizer groups"
```

---

### Task 5: Add LoRA Training Config

**Files:**
- Create: `starVLA/config/training/auxvla_gr00t_lora_libero.yaml`
- Modify: `tests/dataloader/test_uamvla_libero_registry.py`

- [ ] **Step 1: Write the failing YAML load test**

Open `tests/dataloader/test_uamvla_libero_registry.py` and add this test near
the existing AuxVLAGR00T YAML load test:

```python
def test_auxvla_gr00t_lora_libero_yaml_loads():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("starVLA/config/training/auxvla_gr00t_lora_libero.yaml")

    assert cfg.framework.name == "AuxVLAGR00T"
    assert cfg.framework.reconvla.lora.enabled is True
    assert cfg.framework.reconvla.lora.r == 16
    assert "q_proj" in cfg.framework.reconvla.lora.target_modules
    assert cfg.trainer.freeze_modules is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/dataloader/test_uamvla_libero_registry.py::test_auxvla_gr00t_lora_libero_yaml_loads -q
```

Expected: FAIL because the YAML file does not exist.

- [ ] **Step 3: Create the LoRA config**

Copy `starVLA/config/training/auxvla_gr00t_libero.yaml` to
`starVLA/config/training/auxvla_gr00t_lora_libero.yaml`, then make these exact
edits:

```yaml
run_id: auxvla_gr00t_lora_libero_primary_h8
```

Under `framework.reconvla`, add:

```yaml
    lora:
      enabled: true
      r: 16
      lora_alpha: 32
      lora_dropout: 0.05
      bias: none
      task_type: CAUSAL_LM
      target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
        - gate_proj
        - up_proj
        - down_proj
```

Keep:

```yaml
trainer:
  freeze_modules: null
```

- [ ] **Step 4: Run the YAML load test**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/dataloader/test_uamvla_libero_registry.py::test_auxvla_gr00t_lora_libero_yaml_loads -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add starVLA/config/training/auxvla_gr00t_lora_libero.yaml tests/dataloader/test_uamvla_libero_registry.py
git commit -m "config: add AuxVLAGR00T LoRA config"
```

---

### Task 6: Extend No-Save Smoke For LoRA

**Files:**
- Modify: `tools/probes/smoke_auxvla_gr00t_no_save.py`

- [ ] **Step 1: Add helper functions**

In `tools/probes/smoke_auxvla_gr00t_no_save.py`, add these functions after
`grad_norm()`:

```python
def count_named_parameters(module: torch.nn.Module, predicate) -> int:
    return sum(param.numel() for name, param in module.named_parameters() if predicate(name, param))


def grad_norm_named(module: torch.nn.Module, device: torch.device, predicate) -> torch.Tensor:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for name, param in module.named_parameters():
        if predicate(name, param) and param.grad is not None:
            total = total + param.grad.detach().float().norm().pow(2)
    return total.sqrt()
```

- [ ] **Step 2: Add CLI flag and config override**

In `parse_args()`, add:

```python
parser.add_argument("--enable_lora", action="store_true")
```

After the existing `--repeated_diffusion_steps` config override in `main()`,
add:

```python
if args.enable_lora:
    cfg.framework.reconvla.lora.enabled = True
```

- [ ] **Step 3: Print and assert LoRA trainable counts**

After `action_trainable` is computed, add:

```python
lora_enabled = bool(cfg.framework.reconvla.get("lora", {}).get("enabled", False))
lora_trainable = count_named_parameters(
    model.qwen_vl_interface,
    lambda name, param: param.requires_grad and "lora_" in name.lower(),
)
backbone_base_trainable = count_named_parameters(
    model.qwen_vl_interface,
    lambda name, param: param.requires_grad and "lora_" not in name.lower(),
)
vision_projector_trainable = count_named_parameters(
    model.qwen_vl_interface,
    lambda name, param: param.requires_grad
    and any(key in name for key in ("vision_tower", "mm_projector", "mm_inv_projector")),
)
print(f"lora_enabled={lora_enabled}")
print(f"lora_trainable_params={lora_trainable}")
print(f"backbone_base_trainable_params={backbone_base_trainable}")
print(f"vision_projector_trainable_params={vision_projector_trainable}")
if lora_enabled:
    assert lora_trainable > 0, "LoRA is enabled but has no trainable parameters"
    assert backbone_base_trainable == 0, "base ReconVLA backbone parameters are trainable"
    assert vision_projector_trainable == 0, "vision/projector parameters are trainable"
```

- [ ] **Step 4: Compute LoRA gradients**

After `action_grad_norm = grad_norm(...)`, add:

```python
lora_grad_norm = grad_norm_named(
    model.qwen_vl_interface,
    hidden.device,
    lambda name, param: "lora_" in name.lower(),
)
```

After `print(f"action_grad_norm=...")`, add:

```python
print(f"lora_grad_norm={lora_grad_norm.item():.6f}")
if lora_enabled:
    assert lora_grad_norm.item() > 0, "LoRA adapter did not receive gradients"
```

- [ ] **Step 5: Run syntax check**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m py_compile \
  tools/probes/smoke_auxvla_gr00t_no_save.py
```

Expected: exit 0.

- [ ] **Step 6: Run help check**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python tools/probes/smoke_auxvla_gr00t_no_save.py --help
```

Expected: output includes `--enable_lora`.

- [ ] **Step 7: Commit**

```bash
git add tools/probes/smoke_auxvla_gr00t_no_save.py
git commit -m "test: extend AuxVLAGR00T smoke for LoRA"
```

---

### Task 7: Full Local Verification

**Files:**
- No new files.

- [ ] **Step 1: Run all AuxVLAGR00T tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py \
  tests/framework/test_auxvla_gr00t_interface.py \
  tests/framework/test_auxvla_gr00t_registry.py \
  tests/dataloader/test_uamvla_libero_registry.py::test_auxvla_gr00t_libero_yaml_loads \
  tests/dataloader/test_uamvla_libero_registry.py::test_auxvla_gr00t_lora_libero_yaml_loads \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run syntax checks**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m py_compile \
  starVLA/model/framework/VLM4A/AuxVLAGR00T.py \
  tools/probes/smoke_auxvla_gr00t_no_save.py
```

Expected: exit 0.

- [ ] **Step 3: Review git status**

Run:

```bash
git status --short
```

Expected: clean, because previous tasks committed each change.

---

### Task 8: Remote Validation Commands

**Files:**
- No code changes.

- [ ] **Step 1: Run CALVIN LoRA no-save smoke remotely**

Run on the remote H100 server after pulling the implementation branch:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python tools/probes/smoke_auxvla_gr00t_no_save.py \
  --config_yaml starVLA/config/training/auxvla_gr00t_lora_libero.yaml \
  --data_root_dir datasets/calvin2uam \
  --data_mix uamvla_calvin_abc_h8 \
  --output_dir /tmp/auxvla_gr00t_lora_calvin_abc_no_save_smoke \
  --repeated_diffusion_steps 1 \
  --enable_lora
```

Expected output includes:

```text
lora_enabled=True
lora_trainable_params=<positive integer>
backbone_base_trainable_params=0
vision_projector_trainable_params=0
action_grad_norm=<positive float>
lora_grad_norm=<positive float>
SMOKE_OK
```

- [ ] **Step 2: Run 8-GPU trainer smoke remotely**

Run:

```bash
unset UAMVLA_TRAIN_STEP_PROFILE

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WANDB_MODE=offline \
TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --config_file starVLA/config/deepseeds/uamvla_gr00t_zero3.yaml \
  --num_processes 8 \
  --gradient_accumulation_steps 1 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/auxvla_gr00t_lora_libero.yaml \
  --run_id auxvla_gr00t_lora_calvin_abc_8gpu_s2 \
  --datasets.vla_data.data_root_dir datasets/calvin2uam \
  --datasets.vla_data.data_mix uamvla_calvin_abc_h8 \
  --datasets.vla_data.action_type delta_qpos \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.freeze_modules null \
  --trainer.max_train_steps 2 \
  --trainer.num_warmup_steps 0 \
  --trainer.save_interval 999999 \
  --trainer.eval_interval 999999 \
  --trainer.visualization.enabled false \
  --trainer.logging_frequency 1 \
  --trainer.gradient_accumulation_steps 1 \
  --framework.aux_loss_control.enabled false \
  --framework.action_model.repeated_diffusion_steps 1
```

Expected: no DeepSpeed hang, finite `loss/total`, optimizer step completes, and
final save completes.

- [ ] **Step 3: Start the 500-step baseline if smoke passes**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WANDB_MODE=offline \
TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --config_file starVLA/config/deepseeds/uamvla_gr00t_zero3.yaml \
  --num_processes 8 \
  --gradient_accumulation_steps 1 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/auxvla_gr00t_lora_libero.yaml \
  --run_id auxvla_gr00t_lora_calvin_abc_8gpu_s500 \
  --datasets.vla_data.data_root_dir datasets/calvin2uam \
  --datasets.vla_data.data_mix uamvla_calvin_abc_h8 \
  --datasets.vla_data.action_type delta_qpos \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.freeze_modules null \
  --trainer.max_train_steps 500 \
  --trainer.num_warmup_steps 20 \
  --trainer.save_interval 500 \
  --trainer.eval_interval 999999 \
  --trainer.visualization.enabled false \
  --trainer.logging_frequency 10 \
  --trainer.gradient_accumulation_steps 1 \
  --framework.aux_loss_control.enabled false \
  --framework.action_model.repeated_diffusion_steps 8
```

Expected: finite logged loss, saved checkpoint at step 500, and final model
saved.
