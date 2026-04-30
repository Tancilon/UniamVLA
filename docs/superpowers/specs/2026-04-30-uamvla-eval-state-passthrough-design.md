# UamVLA Eval-Time State Passthrough — Design

**Date**: 2026-04-30
**Status**: Draft
**Owner**: tancilon

## 1. Problem

UamVLA training feeds every batch a `canonical_state` (produced by `LiberoAdapter` + `StateNormalizer` over the preprocessor's per-frame fields). The official LIBERO eval client (`examples/LIBERO/eval_files/eval_libero.py` + `model2libero_interface.py`) **never sends state to the model server** — its `example_dict` contains only `image` and `lang`. As a result, training-time forward passes use the `state_encoder`, while eval-time forward passes hit the state-less branch in [`UamVLA.predict_action`](../../../starVLA/model/framework/VLM4A/UamVLA.py#L405-L412). This is a real distribution gap and is the most likely cause of avoidable SR loss.

The fix in scope here: at eval time, build the same `canonical_state` that the training pipeline produces, and pass it through the WebSocket interface to `predict_action`.

Out of scope: changing what training feeds; changing the action-side normalization contract (already aligned via `dataset_statistics.json`); modifying the model itself.

## 2. Goals / Non-goals

**Goals**
- At LIBERO eval, every `predict_action` call receives a `canonical_state` whose tensors are bit-identical (modulo float precision) to the tensors the training pipeline would have produced for the same raw observation.
- Zero impact on non-UamVLA frameworks that share the same eval client.
- Single source of truth for state stats: `statistics.yaml` shipped with the checkpoint.

**Non-goals**
- Replacing the existing `dataset_statistics.json` (action-side stats) — kept as is for compatibility with the official client's `unnormalize_actions`.
- Generalizing to other benchmarks (CALVIN / SimplerEnv etc.) — LIBERO only in this spec.
- Changing the model server's transport or the `predict_action` API.

## 3. Verified facts

The design relies on the following code-verified facts (re-verified during brainstorming):

1. **`LiberoAdapter` input contract** ([embodiment_adapter.py:70-99](../../../starVLA/model/modules/uamvla/data/embodiment_adapter.py#L70-L99)) — requires four independent dict keys:

   | Key | Shape | Notes |
   |---|---|---|
   | `ee_pos` | (3,) | xyz |
   | `ee_axis_angle` | (3,) | rotvec |
   | `joint_pos` | (7,) | radians |
   | `gripper_qpos` | (2,) | meters; only first finger used (parallel-jaw assumption) |

   Output: `{"arm_0": {"ee_pose": Tensor(9,), "joint_pos": Tensor(7,)}, "gripper_0": Tensor(1,)}` where `ee_pose = [xyz(3), 6D-rotation(6)]` and `gripper_norm = qpos[0] / 0.04`.

2. **Preprocessor JSONL fields** ([libero_preprocessor.py:152-155, 226-230](../../../tools/preprocess/libero_preprocessor.py#L152-L155)) — every row carries the same four keys with the shapes above, sourced from HDF5 `obs/{ee_pos, ee_ori, joint_states, gripper_states}`.

3. **`StateNormalizer`** ([state_normalizer.py](../../../starVLA/model/modules/uamvla/data/state_normalizer.py)) — applies q99 / mean_std / min_max per field over `state_stats[embodiment]` from `statistics.yaml`. Training uses `mode="q99"` ([uamvla_dataset.py:75-82](../../../starVLA/dataloader/uamvla_dataset.py#L75-L82)).

4. **Action-side stats are already aligned** — [`_save_dataset_statistics_json`](../../../starVLA/training/train_starvla.py#L195) writes the action `min/max/mask` into `dataset_statistics.json` at trainer startup; eval client unnormalizes from there ([model2libero_interface.py:124-133](../../../examples/LIBERO/eval_files/model2libero_interface.py#L124-L133)). State-side stats are NOT currently persisted.

5. **Eval client current state assembly** ([eval_libero.py:145-152](../../../examples/LIBERO/eval_files/eval_libero.py#L145-L152)) — concatenates `eef_pos(3) + axisangle(eef_quat)(3) + gripper_qpos(2)` = 8-D; this 8-D blob is placed into `observation["observation.state"]` but **not into `example_dict`**, so it never reaches the model server. `joint_pos` is not currently fetched.

## 4. Open questions / risks (must be resolved before / during implementation)

These are unresolved against repo evidence. Each becomes a probe in §8.1.

- **R1**: Does the LIBERO env's `obs` dict expose `robot0_joint_pos` with shape `(7,)`? No code in this repo reads it; cannot verify offline. **Fallback if absent**: this design is blocked; revisit with IK reconstruction or a state-less training option.
- **R2**: Is `_quat2axisangle(robot0_eef_quat)` numerically equivalent to HDF5 `obs/ee_ori` (within ~1e-5)? Both are robosuite-derived in principle; not unit-tested in this repo.
- **R3**: Can `libero_env` (Py 3.8) import `StateNormalizer` (which transitively imports `starVLA.dataloader.gr00t_lerobot.transform.state_action.Normalizer`)? `model2libero_interface.py` already imports `starVLA.model.tools`, but the `Normalizer` chain may pull heavier deps. **Fallback if it fails**: inline the full `Normalizer.forward(mode="q99")` semantics in the client — see §6.3.1 for the four required steps (NOT a bare affine transform; the `q01==q99` mask and final `clamp(-1, 1)` are load-bearing for train/eval equivalence).

## 5. Architecture

End-to-end data flow (eval-time, single step):

```
LIBERO env obs dict
  ├─ robot0_eef_pos          ─┐
  ├─ robot0_eef_quat ──────── │  _quat2axisangle  ─┐
  ├─ robot0_joint_pos        ─┤                    │
  └─ robot0_gripper_qpos     ─┤                    │
                              ▼                    │
                  example_dict["uamvla_raw_state"]: dict of 4 fields
                              │
                  [WebSocket via ModelClient.step] ─── client-side conversion ──┐
                              │                                                  │
                  ModelClient (in libero_env)                                   │
                  · LiberoAdapter().to_canonical(raw)                            │
                  · StateNormalizer(stats_yaml, "franka_libero", "q99")(...)     │
                              │                                                  │
                  payload["canonical_state"] ◄────────────────────────────────── ┘
                              │
                              ▼
                  [WebSocket payload]
                              │
                              ▼
                  websocket_policy_server.py: framework.predict_action(**payload)
                              │
                              ▼
                  UamVLA.predict_action consumes canonical_state via
                  state_encoder (training-equivalent path)
```

Key architectural choice: **conversion happens client-side** (in `model2libero_interface.py`), not server-side. Rationale:
- Server stays generic (no benchmark-specific glue).
- All deps (`LiberoAdapter`, `StateNormalizer`, `yaml`) already available in libero_env (subject to R3).
- Wire format carries `canonical_state` directly, so server-side `predict_action(**payload)` works unchanged.
- Co-locates with existing LIBERO-specific eval glue (`unnormalize_actions`, `read_mode_config`).

## 6. Detailed design

### 6.1 Stats persistence (trainer side)

**File**: [`starVLA/training/train_starvla.py`](../../../starVLA/training/train_starvla.py)

Extend `_save_dataset_statistics_json()` to also copy the source `statistics.yaml` verbatim into the **`output_dir`** (the trainer's `run_root_dir / run_id` directory at [train_starvla.py:58](../../../starVLA/training/train_starvla.py#L58)). This is the same directory `read_mode_config` resolves as `run_dir = checkpoint_pt.parents[1]` ([share_tools.py:375](../../../starVLA/model/framework/share_tools.py#L375)) and the same directory where `dataset_statistics.json` already lives.

After the change, `<output_dir>/` contains:

- `config.yaml`
- `dataset_statistics.json` — action stats, schema unchanged (consumed by official `unnormalize_actions`).
- `statistics.yaml` — full UamVLA stats (consumed by client-side `StateNormalizer` for state input).
- `checkpoints/steps_*.pt`
- `final_model/`

`statistics.yaml` lives at the run-dir level — **not** inside `checkpoints/` or `final_model/` — to mirror the existing `dataset_statistics.json` placement and stay compatible with `read_mode_config`'s path resolution.

Implementation: one `shutil.copy(stats_yaml, output_dir / "statistics.yaml")` after the existing JSON write, guarded by the same `is_main_process` and existence checks. No new schema introduced.

### 6.2 Eval driver wiring

**File**: [`examples/LIBERO/eval_files/eval_libero.py`](../../../examples/LIBERO/eval_files/eval_libero.py)

Augment the per-step `example_dict` (currently `{"image": [...], "lang": ...}` at lines 161-165) with `uamvla_raw_state`, **gated on `client_model.uamvla_state_enabled`** (a public attribute exposed by `ModelClient`, see §6.3.1). This avoids reading `obs["robot0_joint_pos"]` for non-UamVLA models — which would crash if the key isn't present.

```python
if getattr(client_model, "uamvla_state_enabled", False):
    example_dict["uamvla_raw_state"] = {
        "ee_pos":        np.asarray(obs["robot0_eef_pos"],        dtype=np.float32),
        "ee_axis_angle": _quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
        "joint_pos":     np.asarray(obs["robot0_joint_pos"],      dtype=np.float32),
        "gripper_qpos":  np.asarray(obs["robot0_gripper_qpos"],   dtype=np.float32),
    }
```

The 8-D `state` concatenation at lines 145-152 stays as-is (it lives in `observation["observation.state"]` and is harmless; not consumed by UamVLA).

The new key is **UamVLA-specific by name**. Non-UamVLA `ModelClient` instances never see it (gate skips assembly entirely). The naming makes its scope obvious to any reader.

### 6.3 Client-side conversion

**File**: [`examples/LIBERO/eval_files/model2libero_interface.py`](../../../examples/LIBERO/eval_files/model2libero_interface.py)

#### 6.3.1 `__init__` extension

Two pre-existing bugs in the current `__init__` block ([model2libero_interface.py:34, 57](../../../examples/LIBERO/eval_files/model2libero_interface.py#L34-L57)) bite this design and must be fixed first:

- **B1**: `self.unnorm_key = unnorm_key` (line 34) stores the raw constructor arg. `eval_libero.py` doesn't pass `unnorm_key`, so `self.unnorm_key = None` after `__init__`. The local `_check_unnorm_key` call inside `get_action_stats` resolves it but **does not write back to self**.
- **B2**: `policy_ckpt_path` is a `.pt` file path (e.g. `<run_dir>/checkpoints/steps_50000_pytorch_model.pt`, see `run_policy_server.sh`). `read_mode_config` correctly extracts `run_dir = parents[1]`. Anything that needs the run-dir directly (e.g. `<run_dir>/statistics.yaml`) must replicate this — `Path(policy_ckpt_path) / "statistics.yaml"` is **wrong** because the `.pt` file is not a directory.

The new `__init__` block (replaces lines 34 and 57, adds state-side block):

```python
# Resolve unnorm_key once, write back to self (fixes B1).
_, norm_stats = read_mode_config(policy_ckpt_path)
self.unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
self.action_norm_stats = norm_stats[self.unnorm_key]["action"]

# Resolve run_dir from the .pt file path (fixes B2; mirrors read_mode_config).
run_dir = Path(policy_ckpt_path).parents[1]

# Try to load UamVLA state-side artifacts.
self.uamvla_state_enabled = False  # public attribute — gates §6.2 assembly
stats_yaml_path = run_dir / "statistics.yaml"
if stats_yaml_path.exists():
    with open(stats_yaml_path) as f:
        stats_dict = yaml.safe_load(f)
    if (
        isinstance(stats_dict, dict)
        and "state_stats" in stats_dict
        and self.unnorm_key in stats_dict["state_stats"]
    ):
        from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter
        from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer
        self._adapter = LiberoAdapter()
        self._state_normalizer = StateNormalizer(
            stats_dict=stats_dict,
            embodiment=self.unnorm_key,
            mode="q99",
        )
        self.uamvla_state_enabled = True
```

`uamvla_state_enabled` is intentionally a **public attribute** so eval_libero.py can gate its raw-state assembly on it (§6.2).

Detection rule: UamVLA mode iff `statistics.yaml` exists in the run dir AND it contains `state_stats[<resolved unnorm_key>]`. Opt-in by the trainer; other frameworks remain untouched.

**R3 fallback** (if `StateNormalizer` cannot import in libero_env): replace the `from starVLA....StateNormalizer` import with an inline implementation that **fully reproduces** [`Normalizer.forward(mode="q99")`](../../../starVLA/dataloader/gr00t_lerobot/transform/state_action.py#L114-L135) semantics:
1. Load `q01`, `q99` per field from `state_stats[embodiment][field]`.
2. Per-element mask `q01 != q99`; affine `(x - q01) / (q99 - q01) * 2 - 1` only on masked positions.
3. Pass-through `x` on positions where `q01 == q99`.
4. Final `torch.clamp(normalized, -1, 1)`.

Steps 2–4 are non-negotiable for train/eval equivalence. A pure affine transform without (2)–(4) **breaks** §7 contract #5.

#### 6.3.2 `step()` extension

In `step()` ([model2libero_interface.py:76-98](../../../examples/LIBERO/eval_files/model2libero_interface.py#L76-L98)), the per-frame `example` dict is wrapped into `vla_input["examples"] = [example]` and sent to the server. Insert the conversion **after `example["image"] = images`** (line 91) and **before** building `vla_input`:

```python
# Always pop the raw-state key — even if state-passthrough is disabled — so it
# never reaches the server (cleanliness; protects against future kwarg sniffing).
raw = example.pop("uamvla_raw_state", None)

if self.uamvla_state_enabled and raw is not None:
    canonical = self._adapter.to_canonical(raw)
    canonical = self._state_normalizer(canonical)
    example["canonical_state"] = _to_numpy_leaves(canonical)  # see §6.3.3
```

The `pop` is **unconditional** so the wire format never carries `uamvla_raw_state` regardless of state-passthrough toggle. The conversion itself is gated on `uamvla_state_enabled`. If the gate is False, the popped raw state is simply discarded (matches §6.2 — eval_libero.py also won't assemble it in that case, so this branch is mostly defensive).

#### 6.3.3 Tensor serialization

The msgpack-numpy stack ([deployment/model_server/tools/msgpack_numpy.py](../../../deployment/model_server/tools/msgpack_numpy.py)) handles `numpy.ndarray` but **not** `torch.Tensor`. `LiberoAdapter.to_canonical` returns torch tensors, so the client must convert each leaf to numpy before encoding. Implement a helper:

```python
def _to_numpy_leaves(d):
    """Walk 1-level nested dict; convert torch.Tensor leaves to numpy arrays."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.detach().cpu().numpy() if hasattr(vv, "detach") else vv
                      for kk, vv in v.items()}
        else:
            out[k] = v.detach().cpu().numpy() if hasattr(v, "detach") else v
    return out
```

After deserialization on the server side, the leaves arrive as numpy arrays. The reverse (numpy → torch) must happen before `stack_canonical` runs (`torch.stack` only accepts tensors). See §6.4.

### 6.4 Server-side adjustment

`stack_canonical` ([collator_helpers.py:28-50](../../../starVLA/model/modules/uamvla/collator_helpers.py#L28-L50)) currently calls `torch.stack(...)` directly on leaf values. With WebSocket-delivered samples this fails because leaves are numpy arrays.

**Change** (smallest viable): at the entry of `stack_canonical`, coerce each leaf via `torch.as_tensor`:

```python
def stack_canonical(state_list: list[dict]) -> dict:
    template = state_list[0]
    out: dict = {}
    for limb_id, val in template.items():
        if isinstance(val, dict):
            out[limb_id] = {
                k: torch.stack([torch.as_tensor(s[limb_id][k]) for s in state_list], dim=0)
                for k in val.keys()
            }
        else:
            out[limb_id] = torch.stack([torch.as_tensor(s[limb_id]) for s in state_list], dim=0)
    return out
```

`torch.as_tensor` is a no-op on existing tensors and converts numpy / lists to tensors. Side effects on the training path (where leaves are already tensors): none.

Rationale for adjusting `stack_canonical` (vs. coercing inside `predict_action`):
- It's the only code path that reads the leaves; the change is local.
- Any caller — training or eval — benefits transparently.
- One-line change per branch (per-leaf wrap), low risk.

This is the only server-side modification in scope. The framework `predict_action` itself, the WebSocket router, and the policy-loading code stay unchanged.

## 7. Contracts (the things that must hold across train and eval)

These are the invariants the verification suite (§8) enforces:

1. **Field set & shapes**: at the boundary of `LiberoAdapter.to_canonical`, the dict has exactly `{ee_pos(3), ee_axis_angle(3), joint_pos(7), gripper_qpos(2)}` in float32. Identical on both train and eval sides.
2. **Stats source**: training and eval both read `state_stats[embodiment]` from the same `statistics.yaml` file (training reads from dataset dir; eval reads the copy in the ckpt dir, written by §6.1).
3. **Normalizer mode**: `"q99"` on both sides. Hardcoded in eval client; sourced from training config in trainer (default `"q99"` via `UamVLADataset.normalization`).
4. **Output structure**: `{"arm_0": {"ee_pose": Tensor(9,), "joint_pos": Tensor(7,)}, "gripper_0": Tensor(1,)}` — exact tree shape, dtype `float32` post-normalization.
5. **Bit-equivalence**: for the same raw observation, the train-time and eval-time `canonical_state` tensors are equal under `torch.allclose(atol=1e-6)`. (Bit-exact `torch.equal` is the goal but tolerance allows for round-trip serialization noise.)

## 8. Verification plan

### 8.1 Pre-implementation probes (must pass before §6.x changes land)

- **P1** — `tools/probes/probe_libero_obs_keys.py`: spin up a LIBERO env, dump `obs.keys()` and `obs["robot0_joint_pos"].shape`. Pass criterion: key present, shape `(7,)`.
- **P2** — `tests/test_libero_quat_axisangle_alignment.py`: load any HDF5 demo, replay one step in env, compare `_quat2axisangle(env_obs["robot0_eef_quat"])` against the demo's `obs/ee_ori[step]`. Pass criterion: `np.allclose(..., atol=1e-5)` over ≥10 timesteps.
- **P3** — One-line import smoke in `libero_env`:
  ```bash
  python -c "from starVLA.model.modules.uamvla.data.embodiment_adapter import LiberoAdapter; \
             from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer; print('ok')"
  ```
  Pass criterion: exit 0. Failure ⇒ activate inline-normalizer fallback (§6.3.1).

### 8.2 Unit test — train/eval canonical_state equivalence

`tests/test_train_eval_state_alignment.py` exercises both pipelines on the same hand-written raw obs dict and asserts equivalence at every leaf.

- **Train path**: instantiate the `UamVLADataset`-side normalization objects (`LiberoAdapter` + `StateNormalizer(stats_dict, "franka_libero", mode="q99")`) using the **same** `statistics.yaml` the dataset would load — i.e. import the same constructors `UamVLADataset` uses ([uamvla_dataset.py:71-82](../../../starVLA/dataloader/uamvla_dataset.py#L71-L82)) and apply them to the raw dict directly. This avoids requiring image / action-chunk fixtures while still using the production normalization code paths.
- **Eval path**: instantiate the `ModelClient`'s `_adapter` and `_state_normalizer` (the very objects from §6.3.1) and apply to the same raw dict.

Assert: both branches' nested dicts have identical keys, shapes, dtypes, and `torch.allclose(atol=1e-6)` on each leaf.

The test is informative even though both sides use the same constructors — it locks in the contract: any future divergence (e.g., the trainer adding an extra clamp / quantization that the eval side doesn't mirror) breaks the test loudly. This is the load-bearing test for §7.

### 8.3 Fake-client wiring test (catches the contract bugs §8.2 misses)

§8.2 calls the same constructors twice and is structurally tautological — it cannot detect:
- The default-`unnorm_key=None` bug (B1 in §6.3.1) — §8.2 always passes `"franka_libero"` explicitly.
- The wrong `statistics.yaml` path (B2) — §8.2 reads the YAML directly, not via `policy_ckpt_path` resolution.

`tests/test_eval_state_passthrough_wiring.py` covers these by exercising `ModelClient.__init__` and `step()` with realistic inputs, against a stubbed WebSocket client that captures the outbound payload (no real server).

Fixture:
```
tmpdir/
├── config.yaml                     ← minimal valid stub
├── dataset_statistics.json         ← {"franka_libero": {"action": {"min":..., "max":..., "mask":...}}}
├── statistics.yaml                 ← copy of fixture data with state_stats[franka_libero]
└── checkpoints/
    └── fake.pt                     ← empty file (read_mode_config only checks suffix)
```

Test:
```python
def test_modelclient_state_passthrough_engages_with_default_unnorm_key():
    client = ModelClient(
        policy_ckpt_path=tmpdir / "checkpoints" / "fake.pt",
        unnorm_key=None,             # default — must still resolve correctly
        host="0.0.0.0", port=0,
    )
    # Replace the WebSocket client with a stub that records payloads.
    client.client = StubClient()

    # Assertions on init resolution (catches B1 + B2)
    assert client.unnorm_key == "franka_libero"          # was None pre-fix
    assert client.uamvla_state_enabled is True           # path resolution worked

    # Build a representative per-frame example with a UamVLA raw_state attached.
    example = {
        "image": [np.zeros((224, 224, 3), dtype=np.uint8)] * 2,
        "lang": "task description",
        "uamvla_raw_state": {
            "ee_pos":        np.zeros(3, dtype=np.float32),
            "ee_axis_angle": np.zeros(3, dtype=np.float32),
            "joint_pos":     np.zeros(7, dtype=np.float32),
            "gripper_qpos":  np.zeros(2, dtype=np.float32),
        },
    }
    client.step(example, step=0)

    # Inspect what was sent over the (stubbed) wire.
    sent = client.client.last_payload
    sent_example = sent["examples"][0]
    assert "uamvla_raw_state" not in sent_example                # popped (§6.3.2 unconditional)
    assert "canonical_state" in sent_example                      # injected
    assert sent_example["canonical_state"]["arm_0"]["ee_pose"].shape == (9,)
    assert sent_example["canonical_state"]["arm_0"]["joint_pos"].shape == (7,)
    assert sent_example["canonical_state"]["gripper_0"].shape == (1,)
```

A second test variant flips one bit — passes a checkpoint path **without** a sibling `statistics.yaml` — and asserts `uamvla_state_enabled is False`, no `canonical_state` is sent, and `uamvla_raw_state` (if accidentally present in `example`) is still popped.

### 8.4 Integration test — server boot + single infer

Spin up the model server with a real (or mock) UamVLA ckpt that has `statistics.yaml`. From a separate process, instantiate `ModelClient` and call `step()` once with a hand-built `example_dict` containing `uamvla_raw_state`. Assert:
- The server's `predict_action` receives `canonical_state` with the expected nested shape/dtype.
- Returns a `{"normalized_actions": ndarray}` of shape `(1, H, 7)`.

A `print()` of the received `canonical_state` shapes inside `predict_action` (gated by env var, removed before merge) is sufficient to eyeball the contract during the dry run.

### 8.5 End-to-end SR observation (not a gating test)

Run LIBERO Spatial for ≥10 episodes in two configurations:
- **Baseline**: ckpt without `statistics.yaml` ⇒ client falls back to state-less (current behavior).
- **State-passthrough**: ckpt with `statistics.yaml` ⇒ client sends `canonical_state`.

Record SR for both. The expected result is a clear improvement under state-passthrough; this is the design's ultimate justification but not a CI gate (SR depends on training quality, seed, etc.).

## 9. Files touched

| File | Change | LOC estimate |
|---|---|---|
| `starVLA/training/train_starvla.py` | extend `_save_dataset_statistics_json` to also copy `statistics.yaml` | +5 |
| `starVLA/model/modules/uamvla/collator_helpers.py` | wrap leaves in `torch.as_tensor` inside `stack_canonical` | +2 |
| `examples/LIBERO/eval_files/eval_libero.py` | add `uamvla_raw_state` to `example_dict` | +8 |
| `examples/LIBERO/eval_files/model2libero_interface.py` | `__init__` opt-in load + `step()` conversion + tensor→numpy walk | +35 |
| `tools/probes/probe_libero_obs_keys.py` | new (P1) | +20 |
| `tests/test_libero_quat_axisangle_alignment.py` | new (P2) | +40 |
| `tests/test_train_eval_state_alignment.py` | new (§8.2) | +60 |
| `tests/test_eval_state_passthrough_wiring.py` | new (§8.3, fake-client) | +90 |
| `tests/test_eval_state_passthrough_integration.py` | new (§8.4, may be marked `slow`) | +50 |

No deletions. No edits to `UamVLA.predict_action`, the WebSocket transport, the dataloader, or the preprocessor.

## 10. Rollout

Single PR. Order of merging within the PR is irrelevant (changes are additive), but for review clarity:

1. §6.1 (trainer copy)
2. §6.3 (client `__init__` rewrite — fixes B1/B2 + state-passthrough init)
3. §6.3.2 (`step()` conversion + unconditional pop)
4. §6.2 (eval driver gates raw-state assembly on `client.uamvla_state_enabled`)
5. §6.4 (server-side `stack_canonical` `as_tensor` wrap)
6. §8.1 probes + §8.2 alignment + §8.3 fake-client wiring tests
7. §8.4 integration test

If any of P1/P2/P3 fail, surface to the author before continuing — design assumptions are violated and the spec should be revisited rather than worked around.

## 11. Future work (out of scope)

- Generalize to CALVIN: requires a `CalvinAdapter`-driven eval client wrapper following the same pattern. The `statistics.yaml` mechanism transfers directly.
- Move to server-side conversion if multiple benchmarks need state passthrough — at that point the per-benchmark client wrappers become repetitive and a server-side embodiment registry pays off.
- Add a CI hook that fails if `dataset_statistics.json` and `statistics.yaml` disagree on action min/max (cross-check between the two ckpt artifacts).
