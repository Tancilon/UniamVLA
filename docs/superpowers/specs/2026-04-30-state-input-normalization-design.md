# State Input Normalization for UniamVLA — Design Spec

**Date:** 2026-04-30
**Author:** UniamVLA migration team
**Status:** Draft, awaiting user review
**Reference:** OpenVLA-OFT (`prismatic/vla/datasets/rlds/utils/data_utils.py:52`, `prismatic/vla/constants.py`)

## 1. Problem

UniamVLA's state encoder produces token embeddings that go directly into Qwen3-VL's 3584-dim input space, yet the canonical state fed to the encoder is **completely un-normalized**:

- `ee_pos` is in meters, range roughly `[-1, 1.5]` for LIBERO.
- `joint_pos` is in radians, range varies per joint up to `[-π, π]`.
- `gripper_qpos` is normalized to `[0, 1]` already.
- 6D rotation components are `[-1, 1]` by construction.

Without normalization, the input distribution per dimension differs by 1–2 orders of magnitude, so the first `Linear(in_dim, 3584)` layer must absorb both the scale heterogeneity *and* learn meaningful features. With a frozen LLM backbone whose text token embeddings are roughly `N(0, 0.02²)`, this risks producing state tokens whose magnitudes are wildly out of distribution for the LLM, distorting attention weights and impeding gradient flow.

`statistics.yaml` currently stores only action bounds (`action_min_bound`, `action_max_bound`) plus 3 deprecated dims of `robot_obs_mean/std`. There is no per-field state normalization.

OpenVLA-OFT solves the equivalent problem for an 8-dim flat proprio input by clipping each dimension to its 1st/99th quantile and linearly mapping to `[-1, 1]` (`BOUNDS_Q99`), then feeding the normalized vector through a 2-layer MLP with no LayerNorm. The "magic" is upstream normalization, not encoder complexity.

## 2. Goal

Bring OpenVLA-OFT's `BOUNDS_Q99` normalization recipe into UniamVLA's data pipeline, applied **per canonical_state field**. Keep the encoder architecture unchanged.

After this spec ships:
- `LiberoPreprocessor` computes per-field q01/q99 (and min/max/mean/std for completeness) over the entire dataset.
- `statistics.yaml` carries a `state_stats` block with these values, keyed by canonical_state field path.
- `UamVLADataset` (after the embodiment adapter, before encoder) applies `q99` normalization with clipping to `[-1, 1]`.
- `ModularStateEncoder` and per-limb encoders are **untouched**.

## 3. Non-Goals

- Changing encoder MLP depth / width / norm layers. (Already aligned with OpenVLA-OFT.)
- Adding LayerNorm or bottleneck designs.
- Changing per-limb token structure or special-token splice mechanism.
- Normalizing actions differently — actions already use min_max normalization.
- Other embodiments (Bridge, ALOHA): out of scope; this spec covers only `franka_libero`.
- Backbone changes (Qwen3-VL config, etc.).

## 4. Design

### 4.1 Architecture (high level)

```
HDF5 raw → LiberoPreprocessor
                ├─ writes data.jsonl (unchanged)
                └─ writes statistics.yaml { embodiment_stats, state_stats(NEW), ... }

UamVLADataset.__getitem__:
    raw = load_frame(...)
    canonical = adapter.to_canonical(raw)        # raw canonical state, unchanged
    canonical = state_normalizer(canonical)      # NEW: applies q99 + clip per field
    return {..., canonical_state: canonical, ...}

ModularStateEncoder(canonical_state) → state tokens   # unchanged
```

### 4.2 Statistics schema

`statistics.yaml` gains a top-level `state_stats` block:

```yaml
state_stats:
  franka_libero:
    arm_0.ee_pose:
      q01: [..9 floats..]
      q99: [..9 floats..]
      min: [..9 floats..]
      max: [..9 floats..]
      mean: [..9 floats..]
      std: [..9 floats..]
    arm_0.joint_pos:
      q01: [..7 floats..]
      ...
    gripper_0:
      q01: [..1 float..]
      ...
```

**Keying:** dotted path matching the canonical_state nested-dict structure. `arm_0.ee_pose` means `canonical["arm_0"]["ee_pose"]`. `gripper_0` (no dot) is a leaf tensor.

**All six statistics computed (q01/q99/min/max/mean/std)** even though only q01/q99 are used by default. Future modes (`min_max`, `mean_std`) can be enabled without recomputing.

The deprecated `robot_obs_mean/std` block is **removed** in the same change (per existing TODO at `tools/preprocess/libero_preprocessor.py:677`).

### 4.3 Normalization formula

For each field, applied per-dimension independently:

```
x_clipped = clip(x, q01, q99)
x_norm    = 2 * (x_clipped - q01) / (q99 - q01) - 1     # → [-1, 1]
```

Edge case: if `q01 == q99` for some dimension (constant feature), pass through unchanged (matches existing `Normalizer.mode == "q99"` behavior at `state_action.py:121`).

**Finding (2026-04-30, during plan):** `Normalizer.q99` at `state_action.py:114-135` **already clips** to `[-1, 1]` via `torch.clamp` at line 135. The full sequence is: linear map → clamp. This matches OpenVLA-OFT's `BOUNDS_Q99` recipe end-to-end. No change to `state_action.py` is required.

### 4.4 Where normalization is applied

A new `StateNormalizer` lives at `starVLA/model/modules/uamvla/data/state_normalizer.py`:

```python
class StateNormalizer:
    """
    Loads per-field state statistics from statistics.yaml and applies
    BOUNDS_Q99 normalization to a canonical_state dict. Mirrors OpenVLA-OFT's
    normalize_action_and_proprio for the proprio half.
    """
    def __init__(self, stats_path: Path, embodiment: str, mode: str = "q99"): ...
    def __call__(self, canonical_state: dict) -> dict: ...
```

Construction: by `UamVLADataset.__init__`, given the same `statistics.yaml` path it already reads.

Application point: in `UamVLADataset.__getitem__`, **after** `adapter.to_canonical(...)` and **before** the returned dict is consumed by collator / encoder. The adapter stays a pure raw-to-canonical converter; normalization is a separate, swappable transform.

### 4.5 Configurability

YAML config (`starVLA/config/training/uamvla_libero.yaml`) gains:

```yaml
framework:
  state_encoder:
    type: modular
    register_special_tokens: true
    normalization:                       # NEW
      mode: q99                          # one of: q99 | mean_std | min_max | none
      apply_to:                          # which canonical fields to normalize
        - arm_0.ee_pose
        - arm_0.joint_pos
        - gripper_0
```

`mode: none` skips normalization entirely (useful for ablation / debugging). Defaults match OpenVLA-OFT: `q99` (which already clips per §4.3) applied to all canonical state fields.

### 4.6 Preprocessor changes

`LiberoPreprocessor` (`tools/preprocess/libero_preprocessor.py`):

- During the existing single dataset traversal, accumulate per-field running buffers for: `ee_pos`, `ee_axis_angle` (then convert to 6D in stats space too — see open question), `joint_pos`, `gripper_qpos`.
- After traversal: compute q01/q99/min/max/mean/std per dimension.
- **Stats are computed in the *canonical* representation**, not the raw representation. So `ee_axis_angle` (3D raw) is converted to 6D rotation first, and stats are over the 6D representation. This matches what the encoder will see.
- Concretely: extend the preprocessor to instantiate the embodiment adapter, run `adapter.to_canonical(...)` on each frame, accumulate stats over canonical fields.
- Remove `robot_obs_mean/std` write (deprecated TODO).
- Write `state_stats` block instead.

### 4.7 Backward compatibility

- Existing preprocessed datasets (without `state_stats`): if config requests `mode != none`, **raise a clear error** asking the user to re-preprocess. Silent fallback would change model behavior invisibly. Only `mode: none` runs without `state_stats`. Existing test datasets must be re-preprocessed to enable normalization.
- Action stats schema unchanged.
- All `franka_libero`-keyed code still finds what it expects.

## 5. Test strategy

Three test layers:

**Unit (`tests/test_state_normalizer.py`):**
- Given hand-constructed q01/q99 stats and a canonical_state dict, assert output is `[-1, 1]` and round-trip-able for clipped input.
- Edge case q01 == q99 → identity.
- Out-of-range input → clipped to `[-1, 1]`.
- `mode: none` → identity.

**Integration (`tests/test_libero_preprocessor.py`):**
- Run preprocessor on a tiny LIBERO subset (already exists in `datasets/uamvla_test/`).
- Assert `statistics.yaml` contains `state_stats.franka_libero.arm_0.ee_pose.q01` (and friends), shapes match canonical dims (9, 7, 1).
- Assert `robot_obs_mean/std` removed.

**End-to-end smoke (`tests/test_uamvla_dataset.py`):**
- Load preprocessed test data with new statistics.
- Assert `__getitem__` returns canonical_state values within `[-1, 1]` for all normalized fields.
- Assert `ModularStateEncoder.forward(canonical_state)` runs without error and outputs shape `(B, 3, 3584)`.

Existing tests (action normalization, encoder forward) must continue to pass unchanged.

## 6. Migration

Test datasets need to be re-preprocessed once. The existing `datasets/uamvla_test/libero_spatial/` will be regenerated as part of the implementation plan, after the preprocessor change is in.

No production datasets exist yet — this is pre-Phase-2 — so no production migration concern.

## 7. Open questions / decisions logged

1. **Normalize 6D rotation?** 6D rotation components are bounded near `[-1, 1]` already by construction. q99 normalization will be approximately identity. **Decision: yes, normalize uniformly.** Matches OpenVLA-OFT's "normalize everything" simplicity, near-identity for rotations is harmless, code path stays uniform.

2. **q01/q99 vs q05/q95?** OFT uses q01/q99. **Decision: q01/q99**, match OFT.

3. **Per-joint vs whole-vector q01?** OFT computes per-dimension. **Decision: per-dimension** (each of the 9 ee_pose dims gets its own q01/q99). Already implied by schema.

4. **Should the deprecated `robot_obs_mean/std` removal happen in this PR or split?** **Decision: same PR**, since both touch `statistics.yaml` schema and re-preprocessing is required either way.

## 8. Files touched (estimated)

| File | Change |
|---|---|
| `tools/preprocess/libero_preprocessor.py` | Compute canonical-space stats; remove deprecated robot_obs |
| `starVLA/model/modules/uamvla/data/state_normalizer.py` | **New**: `StateNormalizer` class |
| `starVLA/dataloader/uamvla_dataset.py` | Wire `StateNormalizer` into `__getitem__` |
| `starVLA/config/training/uamvla_libero.yaml` | Add `normalization` block |
| `tests/test_state_normalizer.py` | **New**: unit tests |
| `tests/test_libero_preprocessor.py` | Extend: assert state_stats schema |
| `tests/test_uamvla_dataset.py` | Extend: assert post-norm range |
| `datasets/uamvla_test/libero_spatial/statistics.yaml` | Regenerated with new schema |

## 9. Estimated effort

Small-to-medium. ~1-2 days of focused work. The surface is mostly mechanical: compute per-field stats in the preprocessor, plumb config, write YAML, and compose the existing `Normalizer` (see §4.3) in a thin `StateNormalizer` wrapper. No `state_action.py` changes are needed.
