# CALVIN Aux Sidecar Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete CALVIN single-scene and ABCD merged LeRobot preprocessing so all UamVLA-GR00T aux-head sidecars are preserved, validated, and covered by metadata.

**Architecture:** Use a minimal patch over the existing CALVIN pipeline. Single-scene preprocessing writes per-frame grounding levels and aux coverage; the existing merger validates, copies, episode-remaps, and merges all aux sidecars without changing dataloader lookup semantics.

**Tech Stack:** Python, NumPy, PyArrow parquet, pandas, pytest, existing CALVIN/LeRobot preprocessing utilities.

---

## File Structure

- Modify `tools/preprocess/calvin_preprocessor_lerobot.py`
  - Add CALVIN grounding-level helper.
  - Carry `grounding_levels` through `_EpisodeBuffers`.
  - Write per-frame grounding JSON values from those levels.
  - Accumulate and emit `meta/uamvla_aux_coverage.json`.

- Modify `tools/preprocess/calvin_lerobot_merger.py`
  - Validate required aux sidecars for every retained frame.
  - Copy and remap `depths`, `grounding_masks`, and `affordance_heatmaps`.
  - Merge scene-level aux coverage files into the final dataset.

- Modify `tests/test_aux_sidecar_writers.py`
  - Cover target-id to grounding-level semantics.

- Modify `tests/test_calvin_lerobot_chunking.py`
  - Cover `_emit_episode_sidecars()` with per-frame grounding levels.
  - Cover CALVIN aux coverage writer.

- Modify `tests/test_calvin_lerobot_merger.py`
  - Update the synthetic scene fixture to include full aux sidecars and coverage.
  - Cover aux sidecar episode remapping.
  - Cover strict missing-sidecar validation.
  - Cover merged coverage aggregation.

No model, dataloader, config, or LIBERO code changes are part of this plan.

---

### Task 1: Add CALVIN Unit Tests for Grounding Level and Single-Scene Sidecar Metadata

**Files:**
- Modify: `tests/test_aux_sidecar_writers.py`
- Modify: `tests/test_calvin_lerobot_chunking.py`

- [ ] **Step 1: Add a grounding-level helper test**

Add this test to `tests/test_aux_sidecar_writers.py`:

```python
def test_calvin_grounding_level_is_derived_from_target_id():
    from tools.preprocess.calvin_preprocessor_lerobot import (
        CalvinPreprocessorLeRobot,
    )

    assert CalvinPreprocessorLeRobot._grounding_level_for_target_id(
        "table__drawer_link"
    ) == "part"
    assert CalvinPreprocessorLeRobot._grounding_level_for_target_id(
        "table__slide_link"
    ) == "part"
    assert CalvinPreprocessorLeRobot._grounding_level_for_target_id(
        "table__switch_link"
    ) == "part"
    assert CalvinPreprocessorLeRobot._grounding_level_for_target_id(
        "table__button_link"
    ) == "part"
    assert CalvinPreprocessorLeRobot._grounding_level_for_target_id(
        "block_red"
    ) == "object"
    assert CalvinPreprocessorLeRobot._grounding_level_for_target_id(
        "block_blue"
    ) == "object"
```

- [ ] **Step 2: Extend the sidecar writer test for per-frame grounding levels**

In `tests/test_calvin_lerobot_chunking.py`, update
`test_episode_sidecars_write_image_target_per_frame` by adding `depth_targets`,
`grounding_masks`, `affordance_heatmaps`, and `grounding_levels` before the
call to `_emit_episode_sidecars()`:

```python
    depth_targets = [
        np.full((4, 4), 0.5, dtype=np.float32),
        np.full((4, 4), 1.5, dtype=np.float32),
    ]
    grounding_masks = [
        np.zeros((1, 20, 20), dtype=np.float32),
        np.ones((1, 20, 20), dtype=np.float32),
    ]
    affordance_heatmaps = [
        np.full((1, 20, 20), 0.25, dtype=np.float32),
        np.full((1, 20, 20), 0.75, dtype=np.float32),
    ]
    grounding_levels = ["part", "object"]

    preprocessor._emit_episode_sidecars(
        image_targets,
        point_clouds,
        tmp_path,
        episode_index=7,
        depth_targets=depth_targets,
        grounding_masks=grounding_masks,
        grounding_levels=grounding_levels,
        affordance_heatmaps=affordance_heatmaps,
    )
```

Then append these assertions to the same test:

```python
    assert (tmp_path / "depths" / "static" / "7" / "0.npy").exists()
    assert (tmp_path / "depths" / "static" / "7" / "1.npy").exists()
    assert (tmp_path / "grounding_masks" / "static" / "7" / "0.npy").exists()
    assert (tmp_path / "grounding_masks" / "static" / "7" / "1.npy").exists()
    assert (tmp_path / "grounding_masks" / "static" / "7" / "0.json").exists()
    assert (tmp_path / "grounding_masks" / "static" / "7" / "1.json").exists()
    assert (tmp_path / "affordance_heatmaps" / "static" / "7" / "0.npy").exists()
    assert (tmp_path / "affordance_heatmaps" / "static" / "7" / "1.npy").exists()

    with open(tmp_path / "grounding_masks" / "static" / "7" / "0.json") as f:
        assert json.load(f)["grounding_level"] == "part"
    with open(tmp_path / "grounding_masks" / "static" / "7" / "1.json") as f:
        assert json.load(f)["grounding_level"] == "object"
```

- [ ] **Step 3: Add a CALVIN aux coverage writer test**

Add this test to `tests/test_calvin_lerobot_chunking.py`:

```python
def test_preprocessor_writes_uamvla_aux_coverage(tmp_path, monkeypatch):
    _stub_imageio_if_needed(monkeypatch)

    module = importlib.import_module(
        "tools.preprocess.calvin_preprocessor_lerobot",
    )
    CalvinPreprocessorLeRobot = module.CalvinPreprocessorLeRobot

    preprocessor = CalvinPreprocessorLeRobot(default_scene="D")
    coverage = preprocessor._empty_aux_coverage()
    preprocessor._merge_aux_coverage(
        coverage,
        task_name="open drawer",
        lengths={
            "image_target": 2,
            "point_cloud": 2,
            "depth": 2,
            "grounding": 2,
            "affordance": 2,
        },
    )
    preprocessor._emit_aux_coverage(tmp_path, coverage)

    written = json.loads(
        (tmp_path / "meta" / "uamvla_aux_coverage.json").read_text()
    )
    assert written["tasks"]["open drawer"]["depth"] == {"valid": 2, "total": 2}
    assert written["tasks"]["open drawer"]["grounding"] == {
        "valid": 2,
        "total": 2,
    }
    assert written["totals"]["affordance"] == {"valid": 2, "total": 2}
```

- [ ] **Step 4: Run the new failing tests**

Run:

```bash
pytest tests/test_aux_sidecar_writers.py::test_calvin_grounding_level_is_derived_from_target_id tests/test_calvin_lerobot_chunking.py::test_episode_sidecars_write_image_target_per_frame tests/test_calvin_lerobot_chunking.py::test_preprocessor_writes_uamvla_aux_coverage -q
```

Expected: FAIL because `_grounding_level_for_target_id`, `grounding_levels`,
and coverage helpers are not implemented yet.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/test_aux_sidecar_writers.py tests/test_calvin_lerobot_chunking.py
git commit -m "test(calvin): cover aux sidecar metadata"
```

---

### Task 2: Implement Single-Scene CALVIN Grounding Levels and Coverage

**Files:**
- Modify: `tools/preprocess/calvin_preprocessor_lerobot.py`
- Test: `tests/test_aux_sidecar_writers.py`
- Test: `tests/test_calvin_lerobot_chunking.py`

- [ ] **Step 1: Extend `_EpisodeBuffers`**

In `tools/preprocess/calvin_preprocessor_lerobot.py`, add a `grounding_levels`
field immediately after `grounding_masks`:

```python
    grounding_levels: list[str]  # "part" or "object" per frame
```

- [ ] **Step 2: Add coverage helper constants and methods**

Inside `CalvinPreprocessorLeRobot`, add these methods near the existing
sidecar helpers:

```python
    @staticmethod
    def _grounding_level_for_target_id(target_id: str) -> str:
        return "part" if str(target_id).startswith("table__") else "object"

    @staticmethod
    def _empty_aux_coverage() -> dict[str, dict]:
        metrics = ("image_target", "point_cloud", "depth", "grounding", "affordance")
        return {
            "tasks": {},
            "totals": {
                name: {"valid": 0, "total": 0}
                for name in metrics
            },
        }

    @classmethod
    def _merge_aux_coverage(
        cls,
        coverage: dict[str, dict],
        task_name: str,
        lengths: dict[str, int],
    ) -> None:
        metrics = ("image_target", "point_cloud", "depth", "grounding", "affordance")
        task_bucket = coverage["tasks"].setdefault(
            task_name,
            {
                name: {"valid": 0, "total": 0}
                for name in metrics
            },
        )
        for name in metrics:
            count = int(lengths.get(name, 0))
            task_bucket[name]["valid"] += count
            task_bucket[name]["total"] += count
            coverage["totals"][name]["valid"] += count
            coverage["totals"][name]["total"] += count

    @staticmethod
    def _emit_aux_coverage(output_dir: Path, coverage: dict[str, dict]) -> None:
        meta_dir = Path(output_dir) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "uamvla_aux_coverage.json", "w") as f:
            json.dump(coverage, f, indent=2, sort_keys=True)
```

- [ ] **Step 3: Initialize and update coverage in `process()`**

In `process()`, after `global_index = 0`, add:

```python
            aux_coverage = self._empty_aux_coverage()
```

After `_emit_episode_sidecars(...)` succeeds and before advancing
`global_index`, add:

```python
                self._merge_aux_coverage(
                    aux_coverage,
                    task_name=instr,
                    lengths={
                        "image_target": len(buffers.image_targets),
                        "point_cloud": len(buffers.point_clouds),
                        "depth": len(buffers.depth_targets),
                        "grounding": len(buffers.grounding_masks),
                        "affordance": len(buffers.affordance_heatmaps),
                    },
                )
```

After `_emit_meta(...)`, add:

```python
        self._emit_aux_coverage(output_dir, aux_coverage)
```

- [ ] **Step 4: Carry grounding levels through `_process_window_into_buffers()`**

After resolving `target_object_id`, add:

```python
        grounding_level = self._grounding_level_for_target_id(target_object_id)
```

Initialize the list next to `grounding_masks`:

```python
        grounding_levels: list[str] = []
```

After appending a grounding mask, append the level:

```python
            grounding_levels.append(grounding_level)
```

When returning `_EpisodeBuffers`, include:

```python
            grounding_levels=grounding_levels,
```

- [ ] **Step 5: Accept grounding levels in `_emit_episode_sidecars()`**

Update the signature:

```python
    def _emit_episode_sidecars(
        self, image_targets: list[np.ndarray], point_clouds: list,
        output_dir: Path, episode_index: int,
        depth_targets: list[np.ndarray] | None = None,
        grounding_masks: list[np.ndarray] | None = None,
        grounding_levels: list[str] | None = None,
        affordance_heatmaps: list[np.ndarray] | None = None,
    ) -> None:
```

Add `("grounding_levels", grounding_levels)` to the length validation list.

When calling `_emit_aux_denoising_sidecars`, replace the hard-coded
`grounding_level="object"` with:

```python
                    grounding_level=(
                        grounding_levels[base_index]
                        if grounding_levels is not None
                        else "object"
                    ),
```

Update the call site in `process()` to pass:

```python
                    grounding_levels=buffers.grounding_levels,
```

- [ ] **Step 6: Run the single-scene tests**

Run:

```bash
pytest tests/test_aux_sidecar_writers.py::test_calvin_grounding_level_is_derived_from_target_id tests/test_calvin_lerobot_chunking.py::test_episode_sidecars_write_image_target_per_frame tests/test_calvin_lerobot_chunking.py::test_preprocessor_writes_uamvla_aux_coverage -q
```

Expected: PASS.

- [ ] **Step 7: Run the existing sidecar writer tests**

Run:

```bash
pytest tests/test_aux_sidecar_writers.py tests/test_calvin_lerobot_chunking.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit single-scene implementation**

```bash
git add tools/preprocess/calvin_preprocessor_lerobot.py tests/test_aux_sidecar_writers.py tests/test_calvin_lerobot_chunking.py
git commit -m "feat(calvin): write aux coverage and grounding levels"
```

---

### Task 3: Add Merger Tests for Aux Sidecar Remapping and Strict Validation

**Files:**
- Modify: `tests/test_calvin_lerobot_merger.py`

- [ ] **Step 1: Extend `_write_scene_dataset()` to write aux sidecars**

In `tests/test_calvin_lerobot_merger.py`, inside `_write_scene_dataset()`, after
the point cloud writing loop, add sidecar generation for each episode:

```python
        depth_dir = scene_root / "depths" / "static" / str(local_episode_id)
        grounding_dir = (
            scene_root / "grounding_masks" / "static" / str(local_episode_id)
        )
        affordance_dir = (
            scene_root / "affordance_heatmaps" / "static" / str(local_episode_id)
        )
        depth_dir.mkdir(parents=True, exist_ok=True)
        grounding_dir.mkdir(parents=True, exist_ok=True)
        affordance_dir.mkdir(parents=True, exist_ok=True)
        for offset in range(length):
            np.save(
                depth_dir / f"{offset}.npy",
                np.full((2, 2), float(local_episode_id + offset), dtype=np.float32),
            )
            np.save(
                grounding_dir / f"{offset}.npy",
                np.full((1, 20, 20), float(offset), dtype=np.float32),
            )
            _write_json(
                grounding_dir / f"{offset}.json",
                {"grounding_level": "part" if offset == 0 else "object"},
            )
            np.save(
                affordance_dir / f"{offset}.npy",
                np.full((1, 20, 20), 0.5, dtype=np.float32),
            )
```

Before returning `scene_root`, write coverage:

```python
    metric_counts = {
        "image_target": {"valid": total_frames, "total": total_frames},
        "point_cloud": {"valid": total_frames, "total": total_frames},
        "depth": {"valid": total_frames, "total": total_frames},
        "grounding": {"valid": total_frames, "total": total_frames},
        "affordance": {"valid": total_frames, "total": total_frames},
    }
    coverage_tasks = {
        name: {
            key: {"valid": 0, "total": 0}
            for key in metric_counts
        }
        for name in task_names
    }
    for row in episode_rows:
        task_name = row["tasks"][0]
        for key in metric_counts:
            coverage_tasks[task_name][key]["valid"] += int(row["length"])
            coverage_tasks[task_name][key]["total"] += int(row["length"])
    _write_json(
        scene_root / "meta" / "uamvla_aux_coverage.json",
        {"tasks": coverage_tasks, "totals": metric_counts},
    )
```

- [ ] **Step 2: Extend the main merge test to assert aux remapping**

In `test_merge_renumbers_parquet_rows_meta_and_videos`, after point cloud
assertions, add:

```python
    depth_dirs = sorted((out_dir / "depths" / "static").iterdir())
    assert [path.name for path in depth_dirs] == ["0", "1", "2", "3"]
    grounding_dirs = sorted((out_dir / "grounding_masks" / "static").iterdir())
    assert [path.name for path in grounding_dirs] == ["0", "1", "2", "3"]
    affordance_dirs = sorted(
        (out_dir / "affordance_heatmaps" / "static").iterdir()
    )
    assert [path.name for path in affordance_dirs] == ["0", "1", "2", "3"]

    assert (out_dir / "depths" / "static" / "0" / "0.npy").exists()
    assert (out_dir / "grounding_masks" / "static" / "0" / "0.npy").exists()
    assert (out_dir / "grounding_masks" / "static" / "0" / "0.json").exists()
    assert (
        out_dir / "affordance_heatmaps" / "static" / "0" / "0.npy"
    ).exists()
    with open(out_dir / "grounding_masks" / "static" / "0" / "0.json") as f:
        assert json.load(f)["grounding_level"] == "part"
```

- [ ] **Step 3: Add a merged coverage test**

Add this test to `tests/test_calvin_lerobot_merger.py`:

```python
def test_merge_combines_uamvla_aux_coverage(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=10,
        task_names=["open drawer"],
        episode_lengths=[2],
    )
    scene_b = _write_scene_dataset(
        tmp_path,
        "B",
        episode_start=50,
        task_names=["open drawer", "push block"],
        episode_lengths=[3, 1],
    )
    out_dir = tmp_path / "merged"

    merge_lerobot_scene_outputs(
        [scene_a, scene_b],
        out_dir,
        overwrite=False,
        skip_stats=True,
    )

    coverage = json.loads(
        (out_dir / "meta" / "uamvla_aux_coverage.json").read_text()
    )
    assert coverage["totals"]["depth"] == {"valid": 6, "total": 6}
    assert coverage["totals"]["grounding"] == {"valid": 6, "total": 6}
    assert coverage["tasks"]["open drawer"]["affordance"] == {
        "valid": 5,
        "total": 5,
    }
    assert coverage["tasks"]["push block"]["point_cloud"] == {
        "valid": 1,
        "total": 1,
    }
```

- [ ] **Step 4: Add a strict missing-sidecar test**

Add this test to `tests/test_calvin_lerobot_merger.py`:

```python
def test_merge_rejects_missing_aux_sidecar(tmp_path: Path) -> None:
    scene_a = _write_scene_dataset(
        tmp_path,
        "A",
        episode_start=10,
        task_names=["open drawer"],
        episode_lengths=[2],
    )
    missing = scene_a / "depths" / "static" / "10" / "1.npy"
    missing.unlink()
    out_dir = tmp_path / "merged"

    with pytest.raises(CalvinLeRobotMergeError, match="Missing aux sidecar"):
        merge_lerobot_scene_outputs(
            [scene_a],
            out_dir,
            overwrite=False,
            skip_stats=True,
        )
```

- [ ] **Step 5: Run the new failing merger tests**

Run:

```bash
pytest tests/test_calvin_lerobot_merger.py::test_merge_renumbers_parquet_rows_meta_and_videos tests/test_calvin_lerobot_merger.py::test_merge_combines_uamvla_aux_coverage tests/test_calvin_lerobot_merger.py::test_merge_rejects_missing_aux_sidecar -q
```

Expected: FAIL because the merger does not yet validate/copy aux sidecars or
write merged coverage.

- [ ] **Step 6: Commit the failing merger tests**

```bash
git add tests/test_calvin_lerobot_merger.py
git commit -m "test(calvin): cover merged aux sidecars"
```

---

### Task 4: Implement ABCD Merger Aux Sidecar Copying and Coverage Merge

**Files:**
- Modify: `tools/preprocess/calvin_lerobot_merger.py`
- Test: `tests/test_calvin_lerobot_merger.py`

- [ ] **Step 1: Add aux metric and sidecar constants**

Near the top of `tools/preprocess/calvin_lerobot_merger.py`, after imports and
before dataclasses, add:

```python
AUX_METRICS = ("image_target", "point_cloud", "depth", "grounding", "affordance")
AUX_SIDECAR_SPECS = (
    ("depth", Path("depths") / "static", ".npy"),
    ("grounding", Path("grounding_masks") / "static", ".npy"),
    ("grounding", Path("grounding_masks") / "static", ".json"),
    ("affordance", Path("affordance_heatmaps") / "static", ".npy"),
)
```

- [ ] **Step 2: Add aux sidecar path and validation helpers**

Add these helpers near `_copy_point_clouds()`:

```python
def _required_aux_sidecar_paths(
    root: Path,
    episode_index: int,
    base_index: int,
) -> list[Path]:
    return [
        root / relative_dir / str(episode_index) / f"{base_index}{suffix}"
        for _, relative_dir, suffix in AUX_SIDECAR_SPECS
    ]


def _validate_aux_sidecars(scene: _SceneMeta, local_episode: int, base_index: int) -> None:
    for path in _required_aux_sidecar_paths(scene.root, local_episode, base_index):
        if not path.exists():
            raise CalvinLeRobotMergeError(
                f"Missing aux sidecar for scene {scene.root}, "
                f"episode {local_episode}, frame {base_index}: {path}"
            )
```

Update `_validate_scene_assets()` inside the per-frame loop to call:

```python
                _validate_aux_sidecars(scene, local_episode, base_index)
```

- [ ] **Step 3: Add aux sidecar copying**

Add this function near `_copy_point_clouds()`:

```python
def _copy_aux_sidecars(scene: _SceneMeta, output_dir: Path) -> None:
    for episode_row in scene.episodes:
        local_episode = int(episode_row["episode_index"])
        global_episode = scene.episode_index_to_global[local_episode]
        for base_index in range(_episode_length(episode_row)):
            for _, relative_dir, suffix in AUX_SIDECAR_SPECS:
                src_path = (
                    scene.root
                    / relative_dir
                    / str(local_episode)
                    / f"{base_index}{suffix}"
                )
                if not src_path.exists():
                    raise CalvinLeRobotMergeError(
                        f"Missing aux sidecar for scene {scene.root}, "
                        f"episode {local_episode}, frame {base_index}: {src_path}"
                    )
                dst_path = (
                    output_dir
                    / relative_dir
                    / str(global_episode)
                    / f"{base_index}{suffix}"
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
```

In `merge_lerobot_scene_outputs()`, after `_copy_point_clouds(scene, output_path)`,
add:

```python
        _copy_aux_sidecars(scene, output_path)
```

- [ ] **Step 4: Add coverage merge helpers**

Add these functions near `_write_json()` / `_write_jsonl()` helpers:

```python
def _empty_aux_metric_counts() -> dict[str, dict[str, int]]:
    return {name: {"valid": 0, "total": 0} for name in AUX_METRICS}


def _add_aux_metric_counts(
    target: dict[str, dict[str, int]],
    source: dict[str, dict[str, int]],
) -> None:
    for name in AUX_METRICS:
        source_bucket = source.get(name, {})
        target[name]["valid"] += int(source_bucket.get("valid", 0))
        target[name]["total"] += int(source_bucket.get("total", 0))


def _merge_aux_coverage(scenes: Sequence[_SceneMeta]) -> dict[str, dict]:
    merged = {"tasks": {}, "totals": _empty_aux_metric_counts()}
    for scene in scenes:
        path = scene.root / "meta" / "uamvla_aux_coverage.json"
        if not path.exists():
            raise CalvinLeRobotMergeError(
                f"Missing aux coverage metadata for scene {scene.root}: {path}"
            )
        coverage = _read_json(path)
        _add_aux_metric_counts(merged["totals"], coverage.get("totals", {}))
        for task_name, task_counts in coverage.get("tasks", {}).items():
            task_bucket = merged["tasks"].setdefault(
                str(task_name),
                _empty_aux_metric_counts(),
            )
            _add_aux_metric_counts(task_bucket, task_counts)
    return merged
```

In `merge_lerobot_scene_outputs()`, after writing `meta/info.json`, add:

```python
    _write_json(
        output_path / "meta" / "uamvla_aux_coverage.json",
        _merge_aux_coverage(scenes),
    )
```

- [ ] **Step 5: Run targeted merger tests**

Run:

```bash
pytest tests/test_calvin_lerobot_merger.py::test_merge_renumbers_parquet_rows_meta_and_videos tests/test_calvin_lerobot_merger.py::test_merge_combines_uamvla_aux_coverage tests/test_calvin_lerobot_merger.py::test_merge_rejects_missing_aux_sidecar -q
```

Expected: PASS.

- [ ] **Step 6: Run all CALVIN merger tests**

Run:

```bash
pytest tests/test_calvin_lerobot_merger.py tests/test_preprocess_calvin_multiscene.py -q
```

Expected: PASS. If `test_preprocess_calvin_multiscene.py` uses the merger with
synthetic scene directories, update its fixture to create the same aux sidecars
as `_write_scene_dataset()` and re-run this command.

- [ ] **Step 7: Commit merger implementation**

```bash
git add tools/preprocess/calvin_lerobot_merger.py tests/test_calvin_lerobot_merger.py tests/test_preprocess_calvin_multiscene.py
git commit -m "feat(calvin): preserve aux sidecars during merge"
```

---

### Task 5: Run Integration Verification for CALVIN Aux Sidecars

**Files:**
- Read: `docs/superpowers/specs/2026-05-22-calvin-aux-sidecars-abcd-design.md`
- Verify: tests only

- [ ] **Step 1: Run focused preprocessing tests**

Run:

```bash
pytest tests/test_aux_sidecar_writers.py tests/test_calvin_lerobot_chunking.py tests/test_calvin_lerobot_merger.py tests/test_preprocess_calvin_multiscene.py -q
```

Expected: PASS.

- [ ] **Step 2: Run dataloader sidecar smoke tests**

Run:

```bash
pytest tests/dataloader/test_lerobot_sidecar_passthrough.py tests/dataloader/test_lerobot_mixture_sidecar_passthrough.py tests/framework/test_uamvla_aux_sidecars.py -q
```

Expected: PASS. These tests confirm the training-side loader still understands
`trajectory_id` and `base_index` sidecar lookup.

- [ ] **Step 3: Run LIBERO sidecar regression tests**

Run:

```bash
pytest tests/test_libero_lerobot_writer.py tests/test_libero_preprocessor_planning.py tests/test_libero_target_mapping.py -q
```

Expected: PASS. These tests guard against accidental drift in the layout CALVIN
is now matching.

- [ ] **Step 4: Inspect git status**

Run:

```bash
git status --short
```

Expected: only intentional files are modified. Do not include unrelated
existing user changes such as `starVLA/config/training/uamvla_gr00t_libero.yaml`
unless the user explicitly asks to commit them.

- [ ] **Step 5: Commit verification-only fixes if needed**

If a test required a small fixture update, commit it:

```bash
git add tests/test_preprocess_calvin_multiscene.py tests/test_calvin_lerobot_merger.py
git commit -m "test(calvin): align multiscene aux fixtures"
```

If no files changed after verification, skip this commit.

---

### Task 6: Optional Real-Data Smoke Command

**Files:**
- No required code changes

- [ ] **Step 1: Run a one-window CALVIN smoke preprocess when CALVIN env is available**

Run this only in an environment where `calvin_env` imports and the CALVIN split
exists:

```bash
python runners/preprocess_calvin.py \
  --input_dir datasets/calvin/task_D_D/training \
  --output_dir /tmp/uamvla_calvin_d_aux_smoke \
  --dataset_source calvin_d_aux_smoke \
  --default_scene D \
  --max_episodes 1 \
  --on_missing_target abort
```

Expected: command exits with status 0.

- [ ] **Step 2: Check sidecar output**

Run:

```bash
find /tmp/uamvla_calvin_d_aux_smoke -maxdepth 5 -type f | sort | rg 'uamvla_aux_coverage|depths/static|grounding_masks/static|affordance_heatmaps/static|image_targets|point_clouds'
```

Expected: the output includes:

```text
/tmp/uamvla_calvin_d_aux_smoke/meta/uamvla_aux_coverage.json
/tmp/uamvla_calvin_d_aux_smoke/depths/static/0/0.npy
/tmp/uamvla_calvin_d_aux_smoke/grounding_masks/static/0/0.npy
/tmp/uamvla_calvin_d_aux_smoke/grounding_masks/static/0/0.json
/tmp/uamvla_calvin_d_aux_smoke/affordance_heatmaps/static/0/0.npy
/tmp/uamvla_calvin_d_aux_smoke/image_targets/0/0.png
/tmp/uamvla_calvin_d_aux_smoke/point_clouds/0/0.npy
```

- [ ] **Step 3: Do not commit generated smoke data**

Run:

```bash
git status --short
```

Expected: no generated files under `/tmp` appear in git status.

---

## Self-Review Checklist

- Spec coverage:
  - Single-scene sidecar metadata is covered by Tasks 1-2.
  - CALVIN grounding semantics are covered by Tasks 1-2.
  - CALVIN affordance semantics remain unchanged and are protected by existing
    code paths; no model-side change is needed.
  - ABCD sidecar validation/copy/remap is covered by Tasks 3-4.
  - Coverage merge is covered by Tasks 3-4.
  - Verification is covered by Tasks 5-6.
- All implementation steps name concrete files, functions, commands, and
  expected outcomes.
- Function names are consistent across tests and implementation:
  `_grounding_level_for_target_id`, `_empty_aux_coverage`,
  `_merge_aux_coverage`, `_emit_aux_coverage`, `_copy_aux_sidecars`.
