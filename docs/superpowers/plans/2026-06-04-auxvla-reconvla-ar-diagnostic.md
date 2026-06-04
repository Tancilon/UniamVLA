# AuxVLAGR00T ReconVLA AR Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an eval-only AuxVLAGR00T mode that runs official ReconVLA autoregressive action-token inference, returns normalized action chunks, and exercises StarVLA websocket plus `dataset_statistics.json` unnormalization.

**Architecture:** Keep the default AuxVLAGR00T GR00T action-head inference path unchanged. Add an opt-in `framework.reconvla.inference_mode=reconvla_ar_normalized` branch that uses `ReconVLAInterface.generate_normalized_actions(...)` with either official ReconVLA image composition or AuxVLAGR00T single-view composition. Add CALVIN eval and websocket-server support so the official HF-style checkpoint directory can be loaded from config without converting it to a StarVLA `.pt` checkpoint.

**Tech Stack:** PyTorch, Transformers generation, ReconVLA `ActionTokenizer`, OmegaConf, StarVLA websocket server/client, pytest.

---

## File Structure

- Modify `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`
  - Add ReconVLA AR config defaults.
  - Add raw 15-D `robot_obs` extraction for AR diagnostic mode.
  - Add official-compose and auxvla-compose AR input construction in `ReconVLAInterface`.
  - Add AR token generation and normalized action decoding.
  - Branch `AuxVLAGR00T.predict_action` only when `inference_mode=reconvla_ar_normalized`.

- Modify `examples/calvin/eval_files/eval_calvin.py`
  - Detect AuxVLAGR00T ReconVLA AR diagnostic mode from the existing ModelClient config.
  - Send raw 15-D CALVIN `robot_obs` in the websocket example when the diagnostic mode is active.

- Modify `deployment/model_server/server_policy.py`
  - Add `--config_yaml` eval-only loading path.
  - When `--config_yaml` is provided, build the model from config and skip StarVLA checkpoint state-dict loading.
  - Preserve the existing `--ckpt_path` path for all normal checkpoints.

- Modify `examples/calvin/eval_files/run_policy_server.sh`
  - Forward optional `CONFIG_YAML` to `server_policy.py`.

- Create `tools/probes/prepare_reconvla_ar_diagnostic_run.py`
  - Build a lightweight StarVLA-style eval run directory containing `config.yaml`, `dataset_statistics.json`, and a sentinel `.pt` file for ModelClient metadata lookup.

- Create `tools/probes/smoke_auxvla_reconvla_ar_diagnostic.py`
  - Remote-only smoke probe that loads the official ReconVLA checkpoint directory and prints normalized action shape/min/max.

- Modify `tests/framework/test_auxvla_gr00t_static.py`
  - Add static tests for AR routing, missing `robot_obs`, `uamvla_raw_state` compatibility, action-token decode trim/pad, and input-mode image selection.

- Create `tests/deployment/test_server_policy_config.py`
  - Test config-driven model loading without touching GPUs or official checkpoint files.

## Preconditions

- Use `/home/user01/miniconda3/envs/uamvla/bin/python` for local unit tests because the base `python` does not reliably include repo dependencies.
- Do not run GPU training or CALVIN simulation locally.
- The official ReconVLA fine-tuned checkpoint remains a Hugging Face-style directory on the remote server.
- The StarVLA eval client still receives a `CKPT_PATH` pointing inside a run directory so it can read `config.yaml` and `dataset_statistics.json`.

---

### Task 1: AuxVLAGR00T AR Routing Tests

**Files:**
- Modify: `tests/framework/test_auxvla_gr00t_static.py`

- [ ] **Step 1: Add static fakes for AR prediction**

Append these helpers after `_FakeLoRAInterface`:

```python
class _FakeARReconInterface(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generate_normalized_actions(
        self,
        images,
        instructions,
        robot_obs,
        input_mode,
        action_horizon,
        action_dim,
    ):
        self.calls.append(
            {
                "images": images,
                "instructions": instructions,
                "robot_obs": robot_obs,
                "input_mode": input_mode,
                "action_horizon": action_horizon,
                "action_dim": action_dim,
            }
        )
        batch_size = len(instructions)
        return np.full(
            (batch_size, int(action_horizon), int(action_dim)),
            0.25,
            dtype=np.float32,
        )


def _make_predict_model(module, inference_mode="gr00t", ar_input_mode="official_compose"):
    model = object.__new__(module.AuxVLAGR00T)
    torch.nn.Module.__init__(model)
    model.config = _AttrDict(
        framework=_AttrDict(
            name="AuxVLAGR00T",
            reconvla=_AttrDict(
                inference_mode=inference_mode,
                ar_input_mode=ar_input_mode,
                single_view_mode="concat_vertical",
            ),
            action_model=_AttrDict(
                action_horizon=5,
                action_dim=7,
                state_dim=7,
            ),
        )
    )
    model.action_horizon = 5
    model.action_model = _FakeActionModel()
    model.qwen_vl_interface = _FakeARReconInterface()
    model._prepare_examples = lambda examples, require_reconvla_target=False: examples
    model._encode_reconvla_hidden = lambda examples: (
        {"input_ids": torch.ones(len(examples), 4, dtype=torch.long)},
        torch.ones(len(examples), 4, 3584, dtype=torch.bfloat16),
    )
    model._action_model_compute_dtype = lambda hidden_dtype: torch.float32
    model._state_batch_or_none = (
        lambda examples, device, dtype: torch.zeros(len(examples), 1, 7, device=device, dtype=dtype)
    )
    return model
```

- [ ] **Step 2: Add tests for default GR00T routing and AR routing**

Append these tests near the existing `predict_action`/visualization tests:

```python
def test_predict_action_defaults_to_gr00t_path(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = _make_predict_model(module, inference_mode="gr00t")
    example = {
        "image": [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)],
        "lang": "open the drawer",
        "state": np.zeros((1, 7), dtype=np.float32),
        "robot_obs": np.zeros(15, dtype=np.float32),
    }

    out = module.AuxVLAGR00T.predict_action(model, [example])

    assert out["normalized_actions"].shape == (1, 8, 7)
    assert model.qwen_vl_interface.calls == []
    assert len(model.action_model.calls) == 1


def test_predict_action_reconvla_ar_returns_normalized_chunk(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = _make_predict_model(
        module,
        inference_mode="reconvla_ar_normalized",
        ar_input_mode="official_compose",
    )
    example = {
        "image": [np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)],
        "lang": "move the slider left",
        "robot_obs": np.arange(15, dtype=np.float32),
    }

    out = module.AuxVLAGR00T.predict_action(model, [example])

    assert out["normalized_actions"].shape == (1, 5, 7)
    assert out["normalized_actions"].dtype == np.float32
    assert np.allclose(out["normalized_actions"], 0.25)
    assert model.qwen_vl_interface.calls[0]["input_mode"] == "official_compose"
    assert model.qwen_vl_interface.calls[0]["action_horizon"] == 5
    assert model.qwen_vl_interface.calls[0]["action_dim"] == 7
    assert model.qwen_vl_interface.calls[0]["robot_obs"].shape == (1, 15)
    assert len(model.action_model.calls) == 0


def test_predict_action_reconvla_ar_requires_raw_robot_obs(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = _make_predict_model(module, inference_mode="reconvla_ar_normalized")
    example = {
        "image": [np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)],
        "lang": "turn on the lightbulb",
    }

    with pytest.raises(RuntimeError, match="15-D robot_obs"):
        module.AuxVLAGR00T.predict_action(model, [example])


def test_predict_action_reconvla_ar_accepts_uamvla_raw_state(monkeypatch):
    module = _load_auxvla_module(monkeypatch)
    model = _make_predict_model(module, inference_mode="reconvla_ar_normalized")
    example = {
        "image": [np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)],
        "lang": "push the block right",
        "uamvla_raw_state": {"robot_obs": np.arange(15, dtype=np.float32)},
    }

    out = module.AuxVLAGR00T.predict_action(model, [example])

    assert out["normalized_actions"].shape == (1, 5, 7)
    assert np.allclose(model.qwen_vl_interface.calls[0]["robot_obs"][0], np.arange(15, dtype=np.float32))
```

- [ ] **Step 3: Run tests and verify they fail for missing AR methods**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py::test_predict_action_defaults_to_gr00t_path \
  tests/framework/test_auxvla_gr00t_static.py::test_predict_action_reconvla_ar_returns_normalized_chunk \
  tests/framework/test_auxvla_gr00t_static.py::test_predict_action_reconvla_ar_requires_raw_robot_obs \
  tests/framework/test_auxvla_gr00t_static.py::test_predict_action_reconvla_ar_accepts_uamvla_raw_state \
  -q
```

Expected: default GR00T test passes or remains close to passing; AR tests fail because `predict_action` has no AR branch yet.

- [ ] **Step 4: Commit the failing routing tests**

```bash
git add tests/framework/test_auxvla_gr00t_static.py
git commit -m "test: cover AuxVLAGR00T ReconVLA AR routing"
```

---

### Task 2: AuxVLAGR00T AR Routing Implementation

**Files:**
- Modify: `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`

- [ ] **Step 1: Add config defaults**

In `AuxVLAGR00TDefaultConfig.reconvla`, add these entries beside `single_view_mode`:

```python
            "inference_mode": "gr00t",
            "ar_input_mode": "official_compose",
            "action_stat_path": "third_party/ReconVLA/reconvla/statistics.yaml",
            "double_instruction": True,
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
```

- [ ] **Step 2: Add AuxVLAGR00T AR helper methods**

Insert these methods before `predict_action`:

```python
    def _reconvla_inference_mode(self) -> str:
        recon_cfg = self.config.framework.get("reconvla", {})
        return str(recon_cfg.get("inference_mode", "gr00t"))

    def _reconvla_ar_input_mode(self) -> str:
        recon_cfg = self.config.framework.get("reconvla", {})
        return str(recon_cfg.get("ar_input_mode", "official_compose"))

    @staticmethod
    def _extract_reconvla_ar_robot_obs(example: dict) -> np.ndarray:
        if "robot_obs" in example:
            robot_obs = example["robot_obs"]
        elif isinstance(example.get("uamvla_raw_state"), dict) and "robot_obs" in example["uamvla_raw_state"]:
            robot_obs = example["uamvla_raw_state"]["robot_obs"]
        else:
            raise RuntimeError(
                "AuxVLAGR00T reconvla_ar_normalized inference requires raw 15-D robot_obs "
                "at `example['robot_obs']` or `example['uamvla_raw_state']['robot_obs']`."
            )

        robot_obs = np.asarray(robot_obs, dtype=np.float32).copy().reshape(-1)
        if robot_obs.shape[0] != 15:
            raise RuntimeError(
                f"AuxVLAGR00T reconvla_ar_normalized inference requires raw 15-D robot_obs, "
                f"got shape {robot_obs.shape}."
            )
        return robot_obs

    def _reconvla_ar_robot_obs_batch(self, examples: List[dict]) -> np.ndarray:
        return np.stack(
            [self._extract_reconvla_ar_robot_obs(example) for example in examples],
            axis=0,
        ).astype(np.float32)

    def _reconvla_ar_images(self, examples: List[dict], input_mode: str) -> list:
        if input_mode == "official_compose":
            return [example["image"] for example in examples]
        if input_mode == "auxvla_compose":
            return self._single_view_images(examples)
        raise ValueError(
            f"Unsupported framework.reconvla.ar_input_mode={input_mode!r}. "
            "Expected `official_compose` or `auxvla_compose`."
        )
```

- [ ] **Step 3: Modify `predict_action`**

Replace the current `predict_action` method with:

```python
    @torch.inference_mode()
    def predict_action(self, examples, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        examples = self._prepare_examples(examples)

        inference_mode = self._reconvla_inference_mode()
        if inference_mode == "reconvla_ar_normalized":
            input_mode = self._reconvla_ar_input_mode()
            robot_obs = self._reconvla_ar_robot_obs_batch(examples)
            pred_actions = self.qwen_vl_interface.generate_normalized_actions(
                images=self._reconvla_ar_images(examples, input_mode),
                instructions=[example["lang"] for example in examples],
                robot_obs=robot_obs,
                input_mode=input_mode,
                action_horizon=int(self.config.framework.action_model.get("action_horizon", self.action_horizon)),
                action_dim=int(self.config.framework.action_model.get("action_dim", 7)),
            )
            return {"normalized_actions": np.asarray(pred_actions, dtype=np.float32)}

        if inference_mode != "gr00t":
            raise ValueError(
                f"Unsupported framework.reconvla.inference_mode={inference_mode!r}. "
                "Expected `gr00t` or `reconvla_ar_normalized`."
            )

        _recon_inputs, hidden = self._encode_reconvla_hidden(examples)
        action_dtype = self._action_model_compute_dtype(hidden.dtype)
        hidden_for_action = hidden.to(dtype=action_dtype)
        state = self._state_batch_or_none(examples, hidden.device, action_dtype)
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(hidden_for_action, state)
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
```

- [ ] **Step 4: Run routing tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py::test_predict_action_defaults_to_gr00t_path \
  tests/framework/test_auxvla_gr00t_static.py::test_predict_action_reconvla_ar_returns_normalized_chunk \
  tests/framework/test_auxvla_gr00t_static.py::test_predict_action_reconvla_ar_requires_raw_robot_obs \
  tests/framework/test_auxvla_gr00t_static.py::test_predict_action_reconvla_ar_accepts_uamvla_raw_state \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Aux routing implementation**

```bash
git add starVLA/model/framework/VLM4A/AuxVLAGR00T.py tests/framework/test_auxvla_gr00t_static.py
git commit -m "feat: route AuxVLAGR00T ReconVLA AR inference"
```

---

### Task 3: ReconVLAInterface AR Input and Decode

**Files:**
- Modify: `starVLA/model/framework/VLM4A/AuxVLAGR00T.py`
- Modify: `tests/framework/test_auxvla_gr00t_static.py`

- [ ] **Step 1: Add static tests for decode and image-mode selection**

Append these tests to `tests/framework/test_auxvla_gr00t_static.py`:

```python
def test_reconvla_ar_decode_trims_and_pads_action_tokens(monkeypatch):
    module = _load_auxvla_module(monkeypatch)

    class _FakeTokenizer:
        vocab_size = 1000
        pad_token_id = 0
        eos_token_id = 2

    class _FakeActionTokenizer:
        action_token_begin_idx = 744

        def decode_token_ids_to_actions(self, ids):
            ids = np.asarray(ids, dtype=np.int64)
            return (ids.astype(np.float32) - 900.0) / 100.0

    interface = object.__new__(module.ReconVLAInterface)
    torch.nn.Module.__init__(interface)
    interface.tokenizer = _FakeTokenizer()
    interface.action_tokenizer = _FakeActionTokenizer()

    short = module.ReconVLAInterface._decode_action_ids_to_chunk(
        interface,
        np.asarray([900, 901, 902], dtype=np.int64),
        action_horizon=2,
        action_dim=3,
    )
    long = module.ReconVLAInterface._decode_action_ids_to_chunk(
        interface,
        np.asarray([900, 901, 902, 903, 904, 905, 906, 907], dtype=np.int64),
        action_horizon=2,
        action_dim=3,
    )

    assert short.shape == (2, 3)
    assert np.allclose(short[0], [0.0, 0.01, 0.02])
    assert np.allclose(short[1], [0.0, 0.0, 0.0])
    assert long.shape == (2, 3)
    assert np.allclose(long.reshape(-1), [0.0, 0.01, 0.02, 0.03, 0.04, 0.05])


def test_reconvla_ar_filters_generated_action_tokens(monkeypatch):
    module = _load_auxvla_module(monkeypatch)

    class _FakeTokenizer:
        vocab_size = 1000
        pad_token_id = 0
        eos_token_id = 2

    class _FakeActionTokenizer:
        action_token_begin_idx = 744

    interface = object.__new__(module.ReconVLAInterface)
    torch.nn.Module.__init__(interface)
    interface.tokenizer = _FakeTokenizer()
    interface.action_tokenizer = _FakeActionTokenizer()

    tokens = module.ReconVLAInterface._valid_action_token_ids(
        interface,
        torch.tensor([1, 743, 744, 800, 999, 1000, 2, 0], dtype=torch.long),
    )

    assert tokens.tolist() == [744, 800, 999]
```

- [ ] **Step 2: Run decode tests and verify they fail**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py::test_reconvla_ar_decode_trims_and_pads_action_tokens \
  tests/framework/test_auxvla_gr00t_static.py::test_reconvla_ar_filters_generated_action_tokens \
  -q
```

Expected: fail because `_decode_action_ids_to_chunk` and `_valid_action_token_ids` do not exist yet.

- [ ] **Step 3: Add ReconVLA AR constants and imports**

Near the existing ReconVLA target constants, add:

```python
_RECONVLA_AR_IMAGE_SIZE = 334
_RECONVLA_AR_OBS_ANCHOR_TOKEN_ID = 35560
_RECONVLA_AR_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence robot. "
    "The robot provides actions to follow out the user's instructions."
)
```

Inside `ReconVLAInterface.__init__`, extend the local ReconVLA imports:

```python
        from recon.action_tokenizer import ActionTokenizer, encode_robot_obs
        from recon.constants import DEFAULT_IMAGE_TOKEN
        from recon import conversation as conversation_lib
```

After `self._process_images = ...`, add:

```python
        self._default_image_token = DEFAULT_IMAGE_TOKEN
        self._conversation_lib = conversation_lib
        self._encode_robot_obs = encode_robot_obs
```

After `self.tokenizer = AutoTokenizer.from_pretrained(...)`, add:

```python
        self.action_tokenizer = ActionTokenizer(self.tokenizer)
```

At the end of `__init__`, add:

```python
        self._logged_ar_decode_stats = False
```

- [ ] **Step 4: Add ReconVLA AR helper methods**

Insert these methods in `ReconVLAInterface` after `_process_image`:

```python
    def _reconvla_ar_cfg(self) -> dict:
        return _cfg_to_plain_dict(self.config.framework.get("reconvla", {}))

    def _action_stat_path(self) -> Path:
        recon_cfg = self._reconvla_ar_cfg()
        stat_path = Path(str(recon_cfg.get("action_stat_path", "third_party/ReconVLA/reconvla/statistics.yaml")))
        if not stat_path.is_absolute():
            stat_path = _repo_root() / stat_path
        if not stat_path.exists():
            raise FileNotFoundError(
                f"framework.reconvla.action_stat_path does not exist: {stat_path}"
            )
        return stat_path

    @staticmethod
    def _compose_official_ar_image(image_list) -> Image.Image:
        if not isinstance(image_list, (list, tuple)) or len(image_list) < 2:
            raise RuntimeError(
                "ReconVLA AR official_compose requires primary and wrist images in `example['image']`."
            )
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        top = _to_rgb_pil(image_list[0]).resize(
            (_RECONVLA_AR_IMAGE_SIZE, _RECONVLA_AR_IMAGE_SIZE // 2),
            resample,
        )
        bottom = _to_rgb_pil(image_list[1]).resize(
            (_RECONVLA_AR_IMAGE_SIZE, _RECONVLA_AR_IMAGE_SIZE // 2),
            resample,
        )
        combined = Image.new("RGB", (_RECONVLA_AR_IMAGE_SIZE, _RECONVLA_AR_IMAGE_SIZE))
        combined.paste(top, (0, 0))
        combined.paste(bottom, (0, _RECONVLA_AR_IMAGE_SIZE // 2))
        return combined

    @staticmethod
    def _single_ar_image(image_list) -> Image.Image:
        if isinstance(image_list, (list, tuple)):
            if len(image_list) != 1:
                raise RuntimeError(
                    f"ReconVLA AR auxvla_compose expects one composed image, got {len(image_list)}."
                )
            return _to_rgb_pil(image_list[0])
        return _to_rgb_pil(image_list)

    def _ar_image_for_mode(self, image_list, input_mode: str) -> Image.Image:
        if input_mode == "official_compose":
            return self._compose_official_ar_image(image_list)
        if input_mode == "auxvla_compose":
            return self._single_ar_image(image_list)
        raise ValueError(
            f"Unsupported ReconVLA AR input_mode={input_mode!r}. "
            "Expected `official_compose` or `auxvla_compose`."
        )

    def _encode_ar_robot_obs(self, robot_obs: np.ndarray) -> tuple[torch.Tensor, str]:
        robot_obs = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
        if robot_obs.shape[0] != 15:
            raise RuntimeError(f"ReconVLA AR requires 15-D robot_obs, got shape {robot_obs.shape}.")
        robot_obs_text = " ".join(str(float(value)) for value in robot_obs)
        obs_tokens, obs_text = self._encode_robot_obs(
            robot_obs_text,
            self.action_tokenizer,
            str(self._action_stat_path()),
        )
        return torch.as_tensor(obs_tokens, dtype=torch.long), obs_text

    def _build_ar_prompt_ids(self, instruction: str, robot_obs: np.ndarray) -> torch.Tensor:
        recon_cfg = self._reconvla_ar_cfg()
        obs_tokens, obs_text = self._encode_ar_robot_obs(robot_obs)
        if bool(recon_cfg.get("double_instruction", True)):
            user_text = (
                f"{instruction}\n{self._default_image_token}\n"
                f"{instruction}\n{obs_text}"
            )
        else:
            user_text = f"{self._default_image_token}\n{instruction}\n{obs_text}"

        conv = self._conversation_lib.default_conversation.copy()
        conv.system = _RECONVLA_AR_SYSTEM_PROMPT
        conv.append_message(conv.roles[0], user_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self._tokenizer_image_token(
            prompt,
            self.tokenizer,
            return_tensors="pt",
        )
        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        anchor = (input_ids == _RECONVLA_AR_OBS_ANCHOR_TOKEN_ID).nonzero(as_tuple=True)
        if len(anchor) >= 2 and anchor[1].numel() > 0:
            anchor_idx = int(anchor[1][0].item())
            start_obs = max(anchor_idx - 15, 0)
            input_ids = torch.cat(
                (
                    input_ids[:, :start_obs],
                    obs_tokens.unsqueeze(0),
                    input_ids[:, anchor_idx:],
                ),
                dim=1,
            )
        else:
            logger.warning(
                "ReconVLA AR prompt did not contain obs anchor token id %d; "
                "using text obs without token splice.",
                _RECONVLA_AR_OBS_ANCHOR_TOKEN_ID,
            )
        return input_ids.squeeze(0)

    def build_reconvla_ar_inputs(
        self,
        images,
        instructions,
        robot_obs,
        input_mode: str,
    ) -> dict:
        if len(images) != len(instructions):
            raise AssertionError("Images and instructions must have the same length")
        robot_obs = np.asarray(robot_obs, dtype=np.float32)
        if robot_obs.shape != (len(instructions), 15):
            raise RuntimeError(
                f"ReconVLA AR robot_obs must have shape ({len(instructions)}, 15), got {robot_obs.shape}."
            )

        input_ids = []
        image_tensors = []
        for sample_images, instruction, sample_robot_obs in zip(images, instructions, robot_obs):
            ar_image = self._ar_image_for_mode(sample_images, input_mode)
            pixel_values = self.image_processor.preprocess(
                ar_image,
                return_tensors="pt",
            )["pixel_values"][0]
            input_ids.append(self._build_ar_prompt_ids(str(instruction), sample_robot_obs))
            image_tensors.append(pixel_values)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        )
        images_tensor = torch.stack(image_tensors, dim=0)
        device = self.device
        return {
            "input_ids": padded_input_ids.to(device),
            "images": images_tensor.to(device),
        }

    def _valid_action_token_ids(self, token_ids: torch.Tensor) -> np.ndarray:
        ids = token_ids.detach().cpu().to(torch.long).numpy().reshape(-1)
        begin = int(getattr(self.action_tokenizer, "action_token_begin_idx"))
        vocab_size = int(getattr(self.tokenizer, "vocab_size"))
        valid = (ids >= begin) & (ids < vocab_size)
        return ids[valid].astype(np.int64)

    def _decode_action_ids_to_chunk(
        self,
        action_token_ids: np.ndarray,
        action_horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        expected = int(action_horizon) * int(action_dim)
        decoded = np.asarray(
            self.action_tokenizer.decode_token_ids_to_actions(
                np.asarray(action_token_ids, dtype=np.int64)
            ),
            dtype=np.float32,
        ).reshape(-1)
        if decoded.shape[0] < expected:
            logger.warning(
                "ReconVLA AR generated %d action values, expected %d; padding with zeros.",
                decoded.shape[0],
                expected,
            )
            decoded = np.pad(decoded, (0, expected - decoded.shape[0]), mode="constant")
        elif decoded.shape[0] > expected:
            logger.warning(
                "ReconVLA AR generated %d action values, expected %d; trimming.",
                decoded.shape[0],
                expected,
            )
            decoded = decoded[:expected]
        return decoded.reshape(int(action_horizon), int(action_dim)).astype(np.float32)

    def _generated_token_suffix(self, sequences: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 2:
            raise RuntimeError(f"ReconVLA AR generate returned sequences with shape {tuple(sequences.shape)}.")
        input_len = input_ids.shape[1]
        if sequences.shape[1] > input_len and torch.equal(
            sequences[:, :input_len].detach().cpu(),
            input_ids.detach().cpu(),
        ):
            return sequences[:, input_len:]
        return sequences

    def generate_normalized_actions(
        self,
        images,
        instructions,
        robot_obs,
        input_mode: str,
        action_horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        ar_inputs = self.build_reconvla_ar_inputs(
            images=images,
            instructions=instructions,
            robot_obs=robot_obs,
            input_mode=input_mode,
        )
        recon_cfg = self._reconvla_ar_cfg()
        temperature = float(recon_cfg.get("temperature", 0.0) or 0.0)
        top_p = recon_cfg.get("top_p", None)

        with torch.inference_mode():
            outputs = self.model.generate(
                ar_inputs["input_ids"],
                images=ar_inputs["images"].to(dtype=torch.float16, device=self.device, non_blocking=True),
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                num_beams=int(recon_cfg.get("num_beams", 1)),
                max_new_tokens=int(recon_cfg.get("max_new_tokens", 128)),
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        sequences = outputs["sequences"] if isinstance(outputs, dict) else outputs.sequences
        generated = self._generated_token_suffix(sequences, ar_inputs["input_ids"])
        chunks = []
        for batch_idx in range(generated.shape[0]):
            action_ids = self._valid_action_token_ids(generated[batch_idx])
            chunk = self._decode_action_ids_to_chunk(action_ids, action_horizon, action_dim)
            chunks.append(chunk)
        actions = np.stack(chunks, axis=0).astype(np.float32)
        if not self._logged_ar_decode_stats:
            self._logged_ar_decode_stats = True
            logger.info(
                "ReconVLA AR decoded normalized actions: input_mode=%s action_stat_path=%s "
                "shape=%s min=%.4f max=%.4f",
                input_mode,
                self._action_stat_path(),
                actions.shape,
                float(actions.min()) if actions.size else 0.0,
                float(actions.max()) if actions.size else 0.0,
            )
        return actions
```

- [ ] **Step 5: Run static decode tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py::test_reconvla_ar_decode_trims_and_pads_action_tokens \
  tests/framework/test_auxvla_gr00t_static.py::test_reconvla_ar_filters_generated_action_tokens \
  -q
```

Expected: both tests pass.

- [ ] **Step 6: Run full AuxVLAGR00T static tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/framework/test_auxvla_gr00t_static.py -q
```

Expected: all tests in this file pass.

- [ ] **Step 7: Commit ReconVLA AR interface implementation**

```bash
git add starVLA/model/framework/VLM4A/AuxVLAGR00T.py tests/framework/test_auxvla_gr00t_static.py
git commit -m "feat: decode ReconVLA AR actions in AuxVLAGR00T"
```

---

### Task 4: CALVIN Eval Raw State and Config-Driven Server

**Files:**
- Modify: `examples/calvin/eval_files/eval_calvin.py`
- Modify: `deployment/model_server/server_policy.py`
- Modify: `examples/calvin/eval_files/run_policy_server.sh`
- Create: `tests/deployment/test_server_policy_config.py`

- [ ] **Step 1: Add server config-loading tests**

Create `tests/deployment/test_server_policy_config.py`:

```python
from __future__ import annotations

import json
import types

from omegaconf import OmegaConf


def test_load_policy_from_config_builds_model_and_attaches_stats(monkeypatch, tmp_path):
    from deployment.model_server import server_policy

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = run_dir / "config.yaml"
    stats_path = run_dir / "dataset_statistics.json"
    OmegaConf.save(
        OmegaConf.create(
            {
                "framework": {"name": "AuxVLAGR00T"},
                "trainer": {},
                "datasets": {"vla_data": {}},
            }
        ),
        config_path,
    )
    stats = {"franka": {"action": {"mean": [0.0], "std": [1.0]}}}
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    captured = {}

    def fake_apply_config_compat(cfg):
        captured["framework"] = cfg.framework.name

    def fake_build_framework(cfg):
        return types.SimpleNamespace(config=cfg)

    monkeypatch.setattr(server_policy, "apply_config_compat", fake_apply_config_compat)
    monkeypatch.setattr(server_policy, "build_framework", fake_build_framework)

    model = server_policy.load_policy_from_config(config_path)

    assert captured["framework"] == "AuxVLAGR00T"
    assert model.config.framework.name == "AuxVLAGR00T"
    assert model.norm_stats == stats


def test_load_policy_prefers_config_yaml_over_checkpoint(monkeypatch, tmp_path):
    from deployment.model_server import server_policy

    config_path = tmp_path / "config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "framework": {"name": "AuxVLAGR00T"},
                "trainer": {},
                "datasets": {"vla_data": {}},
            }
        ),
        config_path,
    )
    monkeypatch.setattr(server_policy, "apply_config_compat", lambda cfg: None)
    monkeypatch.setattr(server_policy, "build_framework", lambda cfg: types.SimpleNamespace(config=cfg))
    monkeypatch.setattr(
        server_policy.baseframework,
        "from_pretrained",
        lambda ckpt_path: (_ for _ in ()).throw(AssertionError("checkpoint loader should not run")),
    )

    args = types.SimpleNamespace(config_yaml=str(config_path), ckpt_path="ignored.pt")
    model = server_policy.load_policy(args)

    assert model.config.framework.name == "AuxVLAGR00T"
```

- [ ] **Step 2: Run server tests and verify they fail**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest tests/deployment/test_server_policy_config.py -q
```

Expected: fail because `load_policy_from_config` and `load_policy` do not exist yet.

- [ ] **Step 3: Implement config-driven policy loading**

In `deployment/model_server/server_policy.py`, add imports:

```python
import json
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import baseframework, build_framework
from starVLA.model.framework.share_tools import apply_config_compat
```

Replace the existing base-framework import:

```python
from starVLA.model.framework.base_framework import baseframework
```

with the combined import shown above.

Add these functions above `main(args)`:

```python
def load_policy_from_config(config_yaml: str | os.PathLike):
    config_yaml = Path(config_yaml)
    cfg = OmegaConf.load(config_yaml)
    apply_config_compat(cfg)
    vla = build_framework(cfg)

    stats_path = config_yaml.parent / "dataset_statistics.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            vla.norm_stats = json.load(f)
    else:
        logging.warning("No dataset_statistics.json found beside config_yaml: %s", stats_path)
    return vla


def load_policy(args):
    if getattr(args, "config_yaml", None):
        return load_policy_from_config(args.config_yaml)
    return baseframework.from_pretrained(args.ckpt_path)
```

In `main(args)`, replace:

```python
    vla = baseframework.from_pretrained(
        args.ckpt_path,
    )
```

with:

```python
    vla = load_policy(args)
```

In `build_argparser()`, add:

```python
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help="Eval-only config path. When set, build the policy from config and skip checkpoint state loading.",
    )
```

- [ ] **Step 4: Forward CONFIG_YAML in the policy-server shell script**

In `examples/calvin/eval_files/run_policy_server.sh`, replace the final Python invocation with:

```bash
extra_args=()
if [[ -n "${CONFIG_YAML:-}" ]]; then
    extra_args+=(--config_yaml "${CONFIG_YAML}")
fi

CUDA_VISIBLE_DEVICES=${gpu_id} python deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    --use_bf16 \
    --idle_timeout -1 \
    "${extra_args[@]}" \
    "$@"
```

- [ ] **Step 5: Add CALVIN eval raw robot_obs gate**

In `examples/calvin/eval_files/eval_calvin.py`, add this method to `CalvinPolicyClient` after `_client_uses_gr00t_state`:

```python
    @classmethod
    def _client_uses_reconvla_ar_diagnostic(cls, client) -> bool:
        config = getattr(client, "model_config", None)
        framework_name = cls._nested_config_get(config, "framework", "name")
        inference_mode = cls._nested_config_get(config, "framework", "reconvla", "inference_mode")
        return framework_name == "AuxVLAGR00T" and inference_mode == "reconvla_ar_normalized"
```

In `CalvinPolicyClient.__init__`, after `self.send_gr00t_state = ...`, add:

```python
        self.send_reconvla_ar_robot_obs = self._client_uses_reconvla_ar_diagnostic(self.client)
```

In `CalvinPolicyClient.step`, after the `send_gr00t_state` block and before the `uamvla_raw_state` block, add:

```python
        if self.send_reconvla_ar_robot_obs:
            example["robot_obs"] = np.asarray(obs["robot_obs"], dtype=np.float32).copy()
```

- [ ] **Step 6: Run server and Aux tests**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/deployment/test_server_policy_config.py \
  tests/framework/test_auxvla_gr00t_static.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit eval/server support**

```bash
git add \
  deployment/model_server/server_policy.py \
  examples/calvin/eval_files/eval_calvin.py \
  examples/calvin/eval_files/run_policy_server.sh \
  tests/deployment/test_server_policy_config.py
git commit -m "feat: support config-driven ReconVLA AR eval"
```

---

### Task 5: Diagnostic Run Preparation and Smoke Probe

**Files:**
- Create: `tools/probes/prepare_reconvla_ar_diagnostic_run.py`
- Create: `tools/probes/smoke_auxvla_reconvla_ar_diagnostic.py`

- [ ] **Step 1: Add diagnostic run preparation script**

Create `tools/probes/prepare_reconvla_ar_diagnostic_run.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-run-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower-path", required=True)
    parser.add_argument("--action-stat-path", required=True)
    parser.add_argument(
        "--input-mode",
        choices=["official_compose", "auxvla_compose"],
        default="official_compose",
    )
    parser.add_argument("--single-view-mode", default="concat_vertical")
    parser.add_argument("--action-horizon", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    source_run_dir = Path(args.source_run_dir)
    output_run_dir = Path(args.output_run_dir)
    checkpoint_dir = output_run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    source_stats = source_run_dir / "dataset_statistics.json"
    if not source_stats.exists():
        raise FileNotFoundError(f"Missing source dataset_statistics.json: {source_stats}")

    cfg = OmegaConf.load(args.base_config)
    cfg.framework.name = "AuxVLAGR00T"
    cfg.framework.reconvla.model_path = args.model_path
    cfg.framework.reconvla.vision_tower_path = args.vision_tower_path
    cfg.framework.reconvla.inference_mode = "reconvla_ar_normalized"
    cfg.framework.reconvla.ar_input_mode = args.input_mode
    cfg.framework.reconvla.single_view_mode = args.single_view_mode
    cfg.framework.reconvla.action_stat_path = args.action_stat_path
    cfg.framework.reconvla.double_instruction = True
    cfg.framework.reconvla.max_new_tokens = 128
    cfg.framework.reconvla.temperature = 0.0
    cfg.framework.reconvla.top_p = None
    cfg.framework.reconvla.num_beams = 1
    cfg.framework.action_model.action_horizon = int(args.action_horizon)
    cfg.framework.action_model.future_action_window_size = int(args.action_horizon) - 1

    config_path = output_run_dir / "config.yaml"
    stats_path = output_run_dir / "dataset_statistics.json"
    sentinel_ckpt = checkpoint_dir / "reconvla_ar_diagnostic.pt"
    OmegaConf.save(cfg, config_path)
    shutil.copy2(source_stats, stats_path)
    torch.save({}, sentinel_ckpt)

    print(f"config_yaml={config_path}")
    print(f"ckpt_path={sentinel_ckpt}")
    print(f"dataset_statistics={stats_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add remote smoke probe**

Create `tools/probes/smoke_auxvla_reconvla_ar_diagnostic.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", required=True)
    parser.add_argument("--input-mode", choices=["official_compose", "auxvla_compose"], default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_image(height: int, width: int, channel: int) -> np.ndarray:
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., channel] = (x + y) // 2
    image[..., (channel + 1) % 3] = x
    return image


def main():
    args = parse_args()
    config_yaml = Path(args.config_yaml)
    cfg = OmegaConf.load(config_yaml)
    apply_config_compat(cfg)
    if args.input_mode is not None:
        cfg.framework.reconvla.ar_input_mode = args.input_mode
    cfg.framework.reconvla.inference_mode = "reconvla_ar_normalized"

    model = build_framework(cfg).to(args.device).eval()
    example = {
        "image": [make_image(200, 200, 0), make_image(84, 84, 2)],
        "lang": "move the slider left",
        "robot_obs": np.zeros(15, dtype=np.float32),
    }
    with torch.inference_mode():
        out = model.predict_action([example])

    actions = np.asarray(out["normalized_actions"], dtype=np.float32)
    print(f"config_yaml={config_yaml}")
    print(f"input_mode={cfg.framework.reconvla.ar_input_mode}")
    print(f"normalized_actions_shape={actions.shape}")
    print(f"normalized_actions_dtype={actions.dtype}")
    print(f"normalized_actions_min={float(actions.min()):.6f}")
    print(f"normalized_actions_max={float(actions.max()):.6f}")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run script help locally**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python tools/probes/prepare_reconvla_ar_diagnostic_run.py --help
/home/user01/miniconda3/envs/uamvla/bin/python tools/probes/smoke_auxvla_reconvla_ar_diagnostic.py --help
```

Expected: both commands print argument help and exit with code 0.

- [ ] **Step 4: Commit diagnostic scripts**

```bash
git add \
  tools/probes/prepare_reconvla_ar_diagnostic_run.py \
  tools/probes/smoke_auxvla_reconvla_ar_diagnostic.py
git commit -m "test: add ReconVLA AR diagnostic probes"
```

---

### Task 6: Verification and Remote Commands

**Files:**
- No new source files.
- Use the code from Tasks 1-5.

- [ ] **Step 1: Run local verification**

Run:

```bash
/home/user01/miniconda3/envs/uamvla/bin/python -m pytest \
  tests/framework/test_auxvla_gr00t_static.py \
  tests/framework/test_auxvla_gr00t_registry.py \
  tests/deployment/test_server_policy_config.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Inspect changed files**

Run:

```bash
git status --short
git log --oneline -5
```

Expected: working tree is clean after the task commits; latest commits correspond to the AR diagnostic implementation.

- [ ] **Step 3: Prepare official-compose diagnostic run on the remote server**

Run this on the remote server from the repo root, after pulling the branch:

```bash
python tools/probes/prepare_reconvla_ar_diagnostic_run.py \
  --source-run-dir playground/Checkpoints/auxvla_gr00t_lora_concat_calvin_abc_bs512_pbs8 \
  --base-config starVLA/config/training/auxvla_gr00t_lora.yaml \
  --output-run-dir playground/Checkpoints/auxvla_reconvla_ar_diag_official_compose \
  --model-path ckpt/reconvla-official-finetuned-checkpoint \
  --vision-tower-path ckpt/siglip-so400m-patch14-384 \
  --action-stat-path third_party/ReconVLA/reconvla/statistics.yaml \
  --input-mode official_compose
```

Replace `ckpt/reconvla-official-finetuned-checkpoint` with the real official fine-tuned ReconVLA checkpoint directory.

- [ ] **Step 4: Smoke official-compose AR generation on the remote server**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/probes/smoke_auxvla_reconvla_ar_diagnostic.py \
  --config-yaml playground/Checkpoints/auxvla_reconvla_ar_diag_official_compose/config.yaml \
  --input-mode official_compose
```

Expected output includes:

```text
input_mode=official_compose
normalized_actions_shape=(1, 5, 7)
SMOKE_OK
```

- [ ] **Step 5: Run CALVIN official-compose diagnostic eval**

Use two terminals.

Terminal A, policy server:

```bash
CUDA_VISIBLE_DEVICES=0 \
PORT=5694 \
CKPT_PATH=playground/Checkpoints/auxvla_reconvla_ar_diag_official_compose/checkpoints/reconvla_ar_diagnostic.pt \
CONFIG_YAML=playground/Checkpoints/auxvla_reconvla_ar_diag_official_compose/config.yaml \
bash examples/calvin/eval_files/run_policy_server.sh \
  > logs/auxvla_reconvla_ar_diag_official_compose_server.log 2>&1
```

Terminal B, CALVIN eval:

```bash
EGL_VISIBLE_DEVICE=0 \
PORT=5694 \
CKPT_PATH=playground/Checkpoints/auxvla_reconvla_ar_diag_official_compose/checkpoints/reconvla_ar_diagnostic.pt \
DATASET_PATH=datasets/uam_dataset/calvin/task_ABC_D \
CALVIN_CONFIG_PATH=/inspire/ssd/project/space-intelligence-multimodality/liuzhenyang-240108540154/dengqi/code/UamVLA/third_party/calvin/calvin_models/conf \
NUM_SEQUENCES=100 \
LOG_DIR=logs/calvin_auxvla_reconvla_ar_diag_official_compose \
bash examples/calvin/eval_files/eval_calvin.sh \
  --args.resize-size 384 \
  2>&1 | tee logs/eval_auxvla_reconvla_ar_diag_official_compose_100.log
```

Expected: websocket connection succeeds, no missing `robot_obs` error, and eval progress prints CALVIN chain success rates.

- [ ] **Step 6: Prepare and evaluate auxvla-compose diagnostic run**

Run:

```bash
python tools/probes/prepare_reconvla_ar_diagnostic_run.py \
  --source-run-dir playground/Checkpoints/auxvla_gr00t_lora_concat_calvin_abc_bs512_pbs8 \
  --base-config starVLA/config/training/auxvla_gr00t_lora.yaml \
  --output-run-dir playground/Checkpoints/auxvla_reconvla_ar_diag_auxvla_compose \
  --model-path ckpt/reconvla-official-finetuned-checkpoint \
  --vision-tower-path ckpt/siglip-so400m-patch14-384 \
  --action-stat-path third_party/ReconVLA/reconvla/statistics.yaml \
  --input-mode auxvla_compose \
  --single-view-mode concat_vertical
```

Then repeat Steps 4 and 5 with:

```text
playground/Checkpoints/auxvla_reconvla_ar_diag_auxvla_compose
```

Expected interpretation:

- official-compose works and auxvla-compose drops: image/input organization is a likely bottleneck.
- both work: StarVLA websocket, unnormalization, and action-buffer pipeline are likely healthy.
- both fail while official raw ReconVLA eval works: investigate the mixed normalized-action plus StarVLA unnormalization path and the prompt/token splice.

- [ ] **Step 7: Push if credentials allow**

Run:

```bash
git push
```

Expected: push succeeds. If GitHub returns permission denied for user `wxqnl`, report that the local commits are ready and the remote credential needs to be changed.

## Self-Review

- Spec coverage: Tasks 1-3 cover AuxVLAGR00T AR routing, official/auxvla input modes, 15-D `robot_obs`, normalized action decode, and default GR00T preservation. Task 4 covers CALVIN eval raw state and config-driven server loading for HF-style ReconVLA directories. Task 5 covers remote smoke. Task 6 covers official-compose and auxvla-compose CALVIN diagnostic commands.
- Red-flag scan: The plan avoids deferred implementation markers and provides exact files, commands, and code blocks for each changed component.
- Type consistency: `generate_normalized_actions(...)` accepts `images`, `instructions`, `robot_obs`, `input_mode`, `action_horizon`, and `action_dim`; the AuxVLAGR00T routing tests and implementation call the same signature. `server_policy.load_policy(args)` is called by `main(args)` and tested directly.
