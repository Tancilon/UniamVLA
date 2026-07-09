# CALVIN VDA 深度预处理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 Video-Depth-Anything（relative ViT-L）对 `datasets/task_ABC_D_scene_D_lerobot` 静态相机全部 5124 个视频做单卡离线深度推理，逐 episode 输出 float16 npz 到 `depth/` 目录。

**Architecture:** 单文件批量脚本 `runners/preprocess_depth_vda.py`：纯逻辑函数（视频枚举、路径映射、npz 保存、pyav 解码、对齐校验）与 GPU 推理解耦（推理通过注入的 `infer_fn` 闭包进入主循环），纯逻辑全部 pytest 覆盖，GPU 路径由 smoke test 验证。配薄启动器 `runners/run_preprocess_depth.sh`。

**Tech Stack:** Python / PyTorch / pyav（解码 AV1）/ numpy / Video-Depth-Anything（third_party，只 import 不修改）/ pytest。

**Spec:** `docs/superpowers/specs/2026-07-08-calvin-vda-depth-preprocess-design.md`

## Global Constraints

- **不修改 `third_party/Video-Depth-Anything` 任何文件**，只通过 `sys.path` 引入。
- **单卡单进程**，无多卡分片逻辑。
- 深度输出语义：**relative inverse depth（disparity），模型原始输出，未归一化**，float16，npz key 固定为 `"depths"`。
- 帧对齐是硬约束：全帧解码（不抽帧）、保存前截断到 RGB 帧数、推理后帧数校验。
- GPU 规则（CLAUDE.md）：任何 GPU 命令前先 `nvidia-smi`，只用空闲卡（`memory.used < 100 MiB` 且 `utilization.gpu < 5%`），`CUDA_VISIBLE_DEVICES=N` 显式绑卡；禁止杀任何 GPU 进程；无空闲卡时停下报告，不 fallback、不等待。
- git 提交只用仓库现有身份，**禁止任何 Claude/AI 署名、Co-Authored-By、Generated-with 尾注**。
- 长时运行命令由对话侧组装：头部 `mkdir -p logs && set -o pipefail`，尾部 `2>&1 | tee "logs/<name>_$(date +%Y%m%d_%H%M%S).log"`。
- 关键数据事实（已实测）：视频为 AV1 编码；静态相机 200×200；episode 长度 34~65 帧（`chunk-000/image/episode_000000.mp4` 为 65 帧）；fps=10；`meta/info.json` 的 `chunks_size=1000`；pyav 15.1.0 已安装。

---

### Task 1: 脚本骨架 + 视频枚举与输出路径映射

**Files:**
- Create: `runners/preprocess_depth_vda.py`
- Create: `tests/test_preprocess_depth_vda.py`

**Interfaces:**
- Produces: `list_camera_videos(dataset_root: str|Path, camera: str) -> list[Path]`（排序后的 mp4 列表）；`depth_output_path(video_path: str|Path, dataset_root: str|Path) -> Path`（`videos/` → `depth/`，后缀 `.npz`）；模块级常量 `PROJECT_ROOT: Path`、`VDA_ROOT: Path`。后续所有 Task 在同一文件追加函数。

- [ ] **Step 1: Write the failing test**

创建 `tests/test_preprocess_depth_vda.py`：

```python
"""Tests for runners/preprocess_depth_vda.py (pure logic; GPU inference excluded)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from runners.preprocess_depth_vda import (
    depth_output_path,
    list_camera_videos,
)


def _make_video_tree(root: Path) -> None:
    """Minimal LeRobot-like videos/ tree with empty mp4 placeholder files."""
    for chunk, eps in [("chunk-000", [0, 1]), ("chunk-001", [1000])]:
        for cam in ["image", "wrist_image"]:
            d = root / "videos" / chunk / cam
            d.mkdir(parents=True)
            for ep in eps:
                (d / f"episode_{ep:06d}.mp4").touch()


def test_list_camera_videos_sorted_and_filtered(tmp_path):
    _make_video_tree(tmp_path)
    videos = list_camera_videos(tmp_path, "image")
    assert [v.name for v in videos] == [
        "episode_000000.mp4",
        "episode_000001.mp4",
        "episode_001000.mp4",
    ]
    assert all("wrist_image" not in str(v) for v in videos)


def test_depth_output_path_mirrors_videos_layout(tmp_path):
    video = tmp_path / "videos" / "chunk-002" / "image" / "episode_002345.mp4"
    out = depth_output_path(video, tmp_path)
    assert out == tmp_path / "depth" / "chunk-002" / "image" / "episode_002345.npz"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/data/dengqi/code/UniamVLA && python -m pytest tests/test_preprocess_depth_vda.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'runners.preprocess_depth_vda'`）

- [ ] **Step 3: Write minimal implementation**

创建 `runners/preprocess_depth_vda.py`：

```python
"""Batch Video-Depth-Anything inference over a LeRobot dataset's camera videos.

Writes per-episode compressed npz depth files mirroring the videos/ layout:

    {dataset_root}/depth/chunk-XXX/{camera}/episode_XXXXXX.npz   # key "depths"

Output semantics: relative inverse depth (disparity), raw model output,
unnormalized, float16. Producer metadata is written to depth/meta.json.

Spec: docs/superpowers/specs/2026-07-08-calvin-vda-depth-preprocess-design.md

Usage:
    CUDA_VISIBLE_DEVICES=0 python runners/preprocess_depth_vda.py \\
        --dataset_root datasets/task_ABC_D_scene_D_lerobot
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VDA_ROOT = PROJECT_ROOT / "third_party" / "Video-Depth-Anything"


def list_camera_videos(dataset_root, camera):
    """Return sorted per-episode mp4 paths for one camera across all chunks."""
    videos_dir = Path(dataset_root) / "videos"
    return sorted(videos_dir.glob(f"chunk-*/{camera}/episode_*.mp4"))


def depth_output_path(video_path, dataset_root):
    """Map videos/chunk-X/{cam}/episode_Y.mp4 -> depth/chunk-X/{cam}/episode_Y.npz."""
    rel = Path(video_path).relative_to(Path(dataset_root) / "videos")
    return (Path(dataset_root) / "depth" / rel).with_suffix(".npz")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/data/dengqi/code/UniamVLA && python -m pytest tests/test_preprocess_depth_vda.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add runners/preprocess_depth_vda.py tests/test_preprocess_depth_vda.py
git commit -m "feat: add VDA depth preprocess script skeleton with video enumeration"
```

---

### Task 2: npz 保存（float16 + 截断）

**Files:**
- Modify: `runners/preprocess_depth_vda.py`（追加函数）
- Modify: `tests/test_preprocess_depth_vda.py`（追加测试）

**Interfaces:**
- Produces: `save_depth_npz(out_path: str|Path, depths: np.ndarray, num_frames: int) -> tuple`（截断到 `num_frames`、转 float16、`np.savez_compressed(key="depths")`、自动建父目录，返回保存后的 shape）
- Consumes: 无（纯函数）

- [ ] **Step 1: Write the failing test**

在 `tests/test_preprocess_depth_vda.py` 追加：

```python
from runners.preprocess_depth_vda import save_depth_npz


def test_save_depth_npz_truncates_and_casts_float16(tmp_path):
    out = tmp_path / "depth" / "chunk-000" / "image" / "episode_000000.npz"
    depths = np.random.default_rng(0).random((70, 200, 200)).astype(np.float32)
    shape = save_depth_npz(out, depths, num_frames=65)
    assert shape == (65, 200, 200)
    loaded = np.load(out)["depths"]
    assert loaded.shape == (65, 200, 200)
    assert loaded.dtype == np.float16
    np.testing.assert_allclose(
        loaded.astype(np.float32), depths[:65], atol=1e-3
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocess_depth_vda.py -v`
Expected: FAIL（`ImportError: cannot import name 'save_depth_npz'`）

- [ ] **Step 3: Write minimal implementation**

在 `runners/preprocess_depth_vda.py` 追加：

```python
def save_depth_npz(out_path, depths, num_frames):
    """Truncate to num_frames (defends against VDA's internal last-frame
    padding), cast to float16, and write compressed npz under key "depths"."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    depths = np.asarray(depths)[:num_frames].astype(np.float16)
    np.savez_compressed(out_path, depths=depths)
    return depths.shape
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_preprocess_depth_vda.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add runners/preprocess_depth_vda.py tests/test_preprocess_depth_vda.py
git commit -m "feat: add float16 truncating npz writer for depth maps"
```

---

### Task 3: pyav 全帧解码（含真实 AV1 视频验证）

**Files:**
- Modify: `runners/preprocess_depth_vda.py`（追加函数）
- Modify: `tests/test_preprocess_depth_vda.py`（追加测试）

**Interfaces:**
- Produces: `decode_video_frames(video_path: str|Path) -> np.ndarray`（shape `(T, H, W, 3)` uint8 RGB，全帧不抽帧；解不出帧抛 `ValueError`）
- Consumes: 无（纯函数，`import av` 放函数内部延迟加载）

- [ ] **Step 1: Write the failing test**

在 `tests/test_preprocess_depth_vda.py` 追加：

```python
from runners.preprocess_depth_vda import PROJECT_ROOT, decode_video_frames

_REAL_AV1_VIDEO = (
    PROJECT_ROOT
    / "datasets/task_ABC_D_scene_D_lerobot/videos/chunk-000/image/episode_000000.mp4"
)


def _write_synthetic_video(path: Path, n_frames: int = 7, size: int = 64) -> None:
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            img = np.full((size, size, 3), i * 30, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_decode_video_frames_returns_all_frames(tmp_path):
    video = tmp_path / "ep.mp4"
    _write_synthetic_video(video, n_frames=7, size=64)
    frames = decode_video_frames(video)
    assert frames.shape == (7, 64, 64, 3)
    assert frames.dtype == np.uint8


@pytest.mark.skipif(not _REAL_AV1_VIDEO.exists(), reason="dataset not present")
def test_decode_real_av1_dataset_video():
    """Validates the spec's AV1-decode risk on this machine (no GPU needed)."""
    frames = decode_video_frames(_REAL_AV1_VIDEO)
    assert frames.shape == (65, 200, 200, 3)
    assert frames.dtype == np.uint8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocess_depth_vda.py -v`
Expected: FAIL（`ImportError: cannot import name 'decode_video_frames'`）

- [ ] **Step 3: Write minimal implementation**

在 `runners/preprocess_depth_vda.py` 追加：

```python
def decode_video_frames(video_path):
    """Decode ALL frames as (T, H, W, 3) uint8 RGB via pyav (matches the
    LeRobot training-time decode chain; decord's AV1 support is unreliable)."""
    import av

    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")
    return np.stack(frames, axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_preprocess_depth_vda.py -v`
Expected: 5 passed（含真实 AV1 视频 65 帧解码——这一步同时消掉 spec §6 smoke test 的解码风险项①）

- [ ] **Step 5: Commit**

```bash
git add runners/preprocess_depth_vda.py tests/test_preprocess_depth_vda.py
git commit -m "feat: add pyav full-frame decoder with real AV1 dataset test"
```

---

### Task 4: episode 元数据读取 + `--verify_only` 全量校验

**Files:**
- Modify: `runners/preprocess_depth_vda.py`（追加函数）
- Modify: `tests/test_preprocess_depth_vda.py`（追加测试）

**Interfaces:**
- Produces: `load_episode_meta(dataset_root) -> tuple[dict[int, int], int, int]`（返回 `{episode_index: length}`、`chunks_size`、`fps`，读自 `meta/episodes.jsonl` 与 `meta/info.json`）；`verify_dataset(dataset_root, camera) -> list[str]`（每条为 `"missing: <path>"` 或 `"frame mismatch: <path> has N expected M"`，全对返回空列表）
- Consumes: 无

- [ ] **Step 1: Write the failing test**

在 `tests/test_preprocess_depth_vda.py` 追加：

```python
import json

from runners.preprocess_depth_vda import load_episode_meta, verify_dataset


def _make_meta(root: Path, lengths: dict[int, int], chunks_size: int = 1000) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"chunks_size": chunks_size, "fps": 10})
    )
    with open(meta / "episodes.jsonl", "w") as f:
        for ep, length in lengths.items():
            f.write(json.dumps({"episode_index": ep, "tasks": [], "length": length}) + "\n")


def _write_depth_npz(root: Path, ep: int, n_frames: int, chunks_size: int = 1000) -> None:
    out = (
        root / "depth" / f"chunk-{ep // chunks_size:03d}" / "image"
        / f"episode_{ep:06d}.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, depths=np.zeros((n_frames, 4, 4), dtype=np.float16))


def test_load_episode_meta(tmp_path):
    _make_meta(tmp_path, {0: 65, 1: 34, 1000: 60})
    lengths, chunks_size, fps = load_episode_meta(tmp_path)
    assert lengths == {0: 65, 1: 34, 1000: 60}
    assert chunks_size == 1000
    assert fps == 10


def test_verify_dataset_reports_missing_and_mismatch(tmp_path):
    _make_meta(tmp_path, {0: 65, 1: 34, 1000: 60})
    _write_depth_npz(tmp_path, 0, 65)      # ok
    _write_depth_npz(tmp_path, 1, 30)      # mismatch (expected 34)
    # episode 1000: missing
    problems = verify_dataset(tmp_path, "image")
    assert len(problems) == 2
    assert any("episode_000001.npz" in p and "mismatch" in p for p in problems)
    assert any("episode_001000.npz" in p and "missing" in p for p in problems)


def test_verify_dataset_all_ok_returns_empty(tmp_path):
    _make_meta(tmp_path, {0: 65, 1: 34})
    _write_depth_npz(tmp_path, 0, 65)
    _write_depth_npz(tmp_path, 1, 34)
    assert verify_dataset(tmp_path, "image") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocess_depth_vda.py -v`
Expected: FAIL（`ImportError: cannot import name 'load_episode_meta'`）

- [ ] **Step 3: Write minimal implementation**

在 `runners/preprocess_depth_vda.py` 顶部 import 区追加 `import json`，然后追加函数：

```python
def load_episode_meta(dataset_root):
    """Read per-episode lengths, chunks_size, and fps from LeRobot meta files."""
    root = Path(dataset_root)
    info = json.loads((root / "meta" / "info.json").read_text())
    lengths = {}
    with open(root / "meta" / "episodes.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            lengths[rec["episode_index"]] = rec["length"]
    return lengths, info["chunks_size"], info["fps"]


def verify_dataset(dataset_root, camera):
    """Compare every episode length in meta against its depth npz frame count."""
    lengths, chunks_size, _ = load_episode_meta(dataset_root)
    problems = []
    for ep_idx in sorted(lengths):
        length = lengths[ep_idx]
        npz_path = (
            Path(dataset_root) / "depth" / f"chunk-{ep_idx // chunks_size:03d}"
            / camera / f"episode_{ep_idx:06d}.npz"
        )
        if not npz_path.exists():
            problems.append(f"missing: {npz_path}")
            continue
        n = np.load(npz_path)["depths"].shape[0]
        if n != length:
            problems.append(f"frame mismatch: {npz_path} has {n} expected {length}")
    return problems
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_preprocess_depth_vda.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add runners/preprocess_depth_vda.py tests/test_preprocess_depth_vda.py
git commit -m "feat: add episode meta loading and depth alignment verification"
```

---

### Task 5: 主处理循环（注入式 infer_fn + 断点续跑 + 失败清单）

**Files:**
- Modify: `runners/preprocess_depth_vda.py`（追加函数）
- Modify: `tests/test_preprocess_depth_vda.py`（追加测试）

**Interfaces:**
- Produces: `process_videos(videos, infer_fn, dataset_root, *, fps, input_size=518, overwrite=False, limit=-1, vis_first_n=0) -> tuple[int, int, list[str]]`（返回 `(n_ok, n_skip, failures)`）。`infer_fn(frames: np.ndarray, fps: int, input_size: int) -> np.ndarray` 是注入的推理闭包——GPU 实现由 Task 6 的 `load_model` 提供，测试注入 fake。失败追加写 `{dataset_root}/depth/failures.txt`。`vis_first_n > 0` 时对前 N 个成功视频调 `save_depth_vis`（Task 6 实现；本 Task 测试全部用 `vis_first_n=0`）。
- Consumes: Task 1 `depth_output_path`、Task 2 `save_depth_npz`、Task 3 `decode_video_frames`。

- [ ] **Step 1: Write the failing test**

在 `tests/test_preprocess_depth_vda.py` 追加：

```python
from runners.preprocess_depth_vda import process_videos


def _make_decodable_dataset(root: Path, n_eps: int = 3, n_frames: int = 7):
    """videos/ tree whose mp4s are real decodable synthetic videos."""
    d = root / "videos" / "chunk-000" / "image"
    d.mkdir(parents=True)
    videos = []
    for ep in range(n_eps):
        p = d / f"episode_{ep:06d}.mp4"
        _write_synthetic_video(p, n_frames=n_frames, size=64)
        videos.append(p)
    return videos


def _fake_infer(frames, fps, input_size):
    # VDA pads short clips internally; emulate returning MORE frames than input.
    t, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    return np.linspace(0, 1, (t + 5) * h * w, dtype=np.float32).reshape(t + 5, h, w)


def _broken_infer(frames, fps, input_size):
    return np.zeros((frames.shape[0] - 2, frames.shape[1], frames.shape[2]), np.float32)


def test_process_videos_writes_truncated_npz(tmp_path):
    videos = _make_decodable_dataset(tmp_path, n_eps=2, n_frames=7)
    n_ok, n_skip, failures = process_videos(
        videos, _fake_infer, tmp_path, fps=10
    )
    assert (n_ok, n_skip, failures) == (2, 0, [])
    for ep in range(2):
        loaded = np.load(
            tmp_path / "depth" / "chunk-000" / "image" / f"episode_{ep:06d}.npz"
        )["depths"]
        assert loaded.shape == (7, 64, 64)  # truncated from fake's 12
        assert loaded.dtype == np.float16


def test_process_videos_skips_existing_unless_overwrite(tmp_path):
    videos = _make_decodable_dataset(tmp_path, n_eps=2, n_frames=7)
    process_videos(videos, _fake_infer, tmp_path, fps=10)
    n_ok, n_skip, _ = process_videos(videos, _fake_infer, tmp_path, fps=10)
    assert (n_ok, n_skip) == (0, 2)
    n_ok, n_skip, _ = process_videos(
        videos, _fake_infer, tmp_path, fps=10, overwrite=True
    )
    assert (n_ok, n_skip) == (2, 0)


def test_process_videos_limit(tmp_path):
    videos = _make_decodable_dataset(tmp_path, n_eps=3, n_frames=7)
    n_ok, n_skip, _ = process_videos(videos, _fake_infer, tmp_path, fps=10, limit=1)
    assert (n_ok, n_skip) == (1, 0)


def test_process_videos_records_failure_and_continues(tmp_path):
    videos = _make_decodable_dataset(tmp_path, n_eps=2, n_frames=7)
    n_ok, n_skip, failures = process_videos(
        videos, _broken_infer, tmp_path, fps=10
    )
    assert n_ok == 0
    assert len(failures) == 2
    failures_txt = (tmp_path / "depth" / "failures.txt").read_text()
    assert "episode_000000.mp4" in failures_txt
    assert "episode_000001.mp4" in failures_txt
    # no partial npz left behind
    assert not list((tmp_path / "depth").rglob("*.npz"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocess_depth_vda.py -v`
Expected: FAIL（`ImportError: cannot import name 'process_videos'`）

- [ ] **Step 3: Write minimal implementation**

在 `runners/preprocess_depth_vda.py` 追加：

```python
def process_videos(videos, infer_fn, dataset_root, *, fps, input_size=518,
                   overwrite=False, limit=-1, vis_first_n=0):
    """Run infer_fn over each video; write truncated float16 npz per episode.

    Returns (n_ok, n_skip, failures). Failures are appended to
    depth/failures.txt and never abort the loop. Alignment contract: the
    depth stack must cover every RGB frame (>= T after decode, saved as [:T]).
    """
    from tqdm import tqdm

    depth_root = Path(dataset_root) / "depth"
    failures_path = depth_root / "failures.txt"
    if limit > 0:
        videos = videos[:limit]
    n_ok = n_skip = 0
    failures = []
    for i, video_path in enumerate(tqdm(videos, desc="depth inference")):
        out_path = depth_output_path(video_path, dataset_root)
        if out_path.exists() and not overwrite:
            n_skip += 1
            continue
        try:
            frames = decode_video_frames(video_path)
            depths = np.asarray(infer_fn(frames, fps, input_size))
            if depths.shape[0] < frames.shape[0]:
                raise ValueError(
                    f"depth frames {depths.shape[0]} < rgb frames {frames.shape[0]}"
                )
            save_depth_npz(out_path, depths, frames.shape[0])
            if n_ok < vis_first_n:
                save_depth_vis(
                    depths[: frames.shape[0]],
                    depth_root / "vis" / f"{Path(video_path).stem}_vis.mp4",
                    fps,
                )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - per-video isolation by design
            msg = f"{video_path}: {exc!r}"
            failures.append(msg)
            depth_root.mkdir(parents=True, exist_ok=True)
            with open(failures_path, "a") as f:
                f.write(msg + "\n")
    return n_ok, n_skip, failures
```

注意：`save_depth_vis` 在 Task 6 实现；本 Task 的测试全部 `vis_first_n=0`，不会触达。若执行本 Task 时想让模块可独立 import 成功，可先加占位：

```python
def save_depth_vis(depths, out_path, fps):  # implemented in Task 6
    raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_preprocess_depth_vda.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add runners/preprocess_depth_vda.py tests/test_preprocess_depth_vda.py
git commit -m "feat: add resumable per-video depth processing loop with failure log"
```

---

### Task 6: GPU 模型加载 + 可视化 + meta.json + CLI 入口

**Files:**
- Modify: `runners/preprocess_depth_vda.py`（追加函数与 `main()`）
- Modify: `tests/test_preprocess_depth_vda.py`（追加 parse_args / meta.json 测试）

**Interfaces:**
- Produces:
  - `load_model(encoder: str, checkpoint: str|Path, device: str) -> Callable`——返回符合 `infer_fn(frames, fps, input_size) -> np.ndarray` 契约的闭包（内部 `sys.path` 引入 VDA、`VideoDepthAnything(**MODEL_CONFIGS[encoder], metric=False)`、fp16 autocast）
  - `save_depth_vis(depths, out_path, fps) -> None`（替换 Task 5 占位；复用 VDA `utils.dc_utils.save_video(is_depths=True)`）
  - `write_meta_json(dataset_root, *, camera, encoder, checkpoint, input_size) -> Path`
  - `parse_args(argv=None)`、`main(argv=None) -> int`
- Consumes: Task 1~5 全部函数。

- [ ] **Step 1: Write the failing test**

在 `tests/test_preprocess_depth_vda.py` 追加：

```python
from runners.preprocess_depth_vda import parse_args, write_meta_json


def test_parse_args_defaults():
    args = parse_args(["--dataset_root", "datasets/task_ABC_D_scene_D_lerobot"])
    assert args.dataset_root == "datasets/task_ABC_D_scene_D_lerobot"
    assert args.camera == "image"
    assert args.encoder == "vitl"
    assert args.input_size == 518
    assert args.checkpoint is None  # resolved from encoder at runtime
    assert args.overwrite is False
    assert args.verify_only is False
    assert args.limit == -1
    assert args.vis_first_n == 0


def test_write_meta_json_records_semantics(tmp_path):
    path = write_meta_json(
        tmp_path,
        camera="image",
        encoder="vitl",
        checkpoint="checkpoints/video_depth_anything_vitl.pth",
        input_size=518,
    )
    meta = json.loads(path.read_text())
    assert meta["npz_key"] == "depths"
    assert meta["dtype"] == "float16"
    assert "inverse depth" in meta["semantics"]
    assert meta["encoder"] == "vitl"
    assert meta["input_size"] == 518
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocess_depth_vda.py -v`
Expected: FAIL（`ImportError: cannot import name 'parse_args'`）

- [ ] **Step 3: Write implementation**

在 `runners/preprocess_depth_vda.py` 顶部 import 区追加 `import argparse` 与 `from datetime import datetime`，删除 Task 5 的 `save_depth_vis` 占位，追加：

```python
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _ensure_vda_on_path():
    if str(VDA_ROOT) not in sys.path:
        sys.path.insert(0, str(VDA_ROOT))


def load_model(encoder, checkpoint, device):
    """Load the relative-depth VDA model; return an infer_fn closure."""
    import torch

    _ensure_vda_on_path()
    from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(**MODEL_CONFIGS[encoder], metric=False)
    model.load_state_dict(
        torch.load(str(checkpoint), map_location="cpu"), strict=True
    )
    model = model.to(device).eval()

    def infer_fn(frames, fps, input_size):
        depths, _ = model.infer_video_depth(
            frames, fps, input_size=input_size, device=device, fp32=False
        )
        return np.asarray(depths)

    return infer_fn


def save_depth_vis(depths, out_path, fps):
    """Colormapped mp4 for eyeballing (smoke test only, not a training input)."""
    _ensure_vda_on_path()
    from utils.dc_utils import save_video

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(np.asarray(depths, dtype=np.float32), str(out_path), fps=fps, is_depths=True)


def write_meta_json(dataset_root, *, camera, encoder, checkpoint, input_size):
    depth_root = Path(dataset_root) / "depth"
    depth_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "producer": "runners/preprocess_depth_vda.py",
        "model": "Video-Depth-Anything (relative)",
        "encoder": encoder,
        "checkpoint": str(checkpoint),
        "input_size": input_size,
        "precision": "fp16 (torch.autocast)",
        "semantics": (
            "relative inverse depth (disparity), raw model output, unnormalized. "
            "NOTE: already inverse — downstream target prep must NOT apply 1/d again."
        ),
        "dtype": "float16",
        "npz_key": "depths",
        "camera": camera,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = depth_root / "meta.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Batch VDA relative-depth inference over LeRobot camera videos."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--camera", type=str, default="image")
    parser.add_argument("--encoder", type=str, default="vitl",
                        choices=["vits", "vitb", "vitl"])
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="defaults to VDA checkpoints/video_depth_anything_{encoder}.pth")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run episodes whose npz already exists")
    parser.add_argument("--verify_only", action="store_true",
                        help="only check npz frame counts against meta/episodes.jsonl")
    parser.add_argument("--limit", type=int, default=-1,
                        help="process only the first N videos (smoke test aid)")
    parser.add_argument("--vis_first_n", type=int, default=0,
                        help="save colormapped mp4 for the first N processed videos")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.verify_only:
        problems = verify_dataset(args.dataset_root, args.camera)
        for p in problems:
            print(p)
        print(f"verify: {len(problems)} problem(s)")
        return 1 if problems else 0

    videos = list_camera_videos(args.dataset_root, args.camera)
    if not videos:
        print(f"no videos found under {args.dataset_root}/videos/*/{args.camera}")
        return 1
    _, _, fps = load_episode_meta(args.dataset_root)
    checkpoint = args.checkpoint or (
        VDA_ROOT / "checkpoints" / f"video_depth_anything_{args.encoder}.pth"
    )
    print(f"{len(videos)} videos | encoder={args.encoder} | fps={fps} | ckpt={checkpoint}")
    infer_fn = load_model(args.encoder, checkpoint, device="cuda")
    write_meta_json(
        args.dataset_root, camera=args.camera, encoder=args.encoder,
        checkpoint=checkpoint, input_size=args.input_size,
    )
    n_ok, n_skip, failures = process_videos(
        videos, infer_fn, args.dataset_root, fps=fps,
        input_size=args.input_size, overwrite=args.overwrite,
        limit=args.limit, vis_first_n=args.vis_first_n,
    )
    print(f"done: ok={n_ok} skipped={n_skip} failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_preprocess_depth_vda.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add runners/preprocess_depth_vda.py tests/test_preprocess_depth_vda.py
git commit -m "feat: add VDA model loading, meta.json writer, and CLI entry"
```

---

### Task 7: 薄启动器 `run_preprocess_depth.sh`

**Files:**
- Create: `runners/run_preprocess_depth.sh`

**Interfaces:**
- Consumes: Task 6 的 CLI（`main`）。
- Produces: 供对话侧组装运行命令的启动器；遵守仓库约定——**内部不写 tee 落盘**，末尾 `"$@"` 透传。

- [ ] **Step 1: Write the launcher**

创建 `runners/run_preprocess_depth.sh`：

```bash
#!/usr/bin/env bash
# Thin launcher for VDA depth preprocessing over the CALVIN LeRobot dataset.
# Log capture (tee) is the caller's responsibility, per repo convention.
# Bind a GPU explicitly, e.g.:
#   CUDA_VISIBLE_DEVICES=0 bash runners/run_preprocess_depth.sh --limit 3
set -e
cd "$(dirname "$0")/.."

python runners/preprocess_depth_vda.py \
  --dataset_root datasets/task_ABC_D_scene_D_lerobot \
  "$@"
```

- [ ] **Step 2: Verify passthrough works（不触 GPU）**

Run: `bash runners/run_preprocess_depth.sh --verify_only; echo "exit=$?"`
Expected: 打印 5124 条 `missing: ...`、`verify: 5124 problem(s)`、`exit=1`（深度尚未生成，missing 是预期结果；验证的是启动器、CLI、verify 链路都通）

- [ ] **Step 3: Commit**

```bash
chmod +x runners/run_preprocess_depth.sh
git add runners/run_preprocess_depth.sh
git commit -m "feat: add launcher for VDA depth preprocessing"
```

---

### Task 8: GPU smoke test（手动，遵守 GPU 规则）

**Files:** 无新文件；产出为 `depth/` 下 3 个 npz + 1 个可视化 mp4，人工检查后可保留（断点续跑会跳过它们）。

**Interfaces:**
- Consumes: Task 7 启动器 + 完整脚本。

- [ ] **Step 1: 找空闲 GPU**

Run: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`
判定：`memory.used < 100 MiB` 且 `utilization.gpu < 5%` 的卡才可用。**若无空闲卡：停止，向用户报告每卡状态与被跳过的步骤，等待用户决定。禁止杀进程、禁止 CPU fallback、禁止循环等待。**

- [ ] **Step 2: 跑 3 个视频 + 1 个可视化**

（将 `N` 换成上一步找到的空闲卡号）

```bash
cd /mnt/data/dengqi/code/UniamVLA && mkdir -p logs && set -o pipefail
CUDA_VISIBLE_DEVICES=N bash runners/run_preprocess_depth.sh \
  --limit 3 --vis_first_n 1 \
  2>&1 | tee "logs/depth_smoke_$(date +%Y%m%d_%H%M%S).log"
```

Expected: `done: ok=3 skipped=0 failed=0`，退出码 0。

- [ ] **Step 3: 检查产物**

```bash
python3 - <<'EOF'
import json
import numpy as np
from pathlib import Path

root = Path("datasets/task_ABC_D_scene_D_lerobot")
lengths = {}
with open(root / "meta/episodes.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        lengths[rec["episode_index"]] = rec["length"]

for ep in range(3):
    p = root / f"depth/chunk-000/image/episode_{ep:06d}.npz"
    d = np.load(p)["depths"]
    assert d.dtype == np.float16, d.dtype
    assert d.shape == (lengths[ep], 200, 200), (d.shape, lengths[ep])
    assert np.isfinite(d.astype(np.float32)).all()
    assert d.astype(np.float32).std() > 0, "depth is constant — inference broken"
    print(ep, d.shape, "range", round(float(d.min()), 3), "-", round(float(d.max()), 3),
          "size", p.stat().st_size // 1024, "KiB")
print("meta.json:", json.loads((root / "depth/meta.json").read_text())["semantics"])
EOF
```

Expected: 3 行 shape/range/size 输出（帧数与 episodes.jsonl 一致、数值有限且非常数）+ meta.json 语义行。

- [ ] **Step 4: 人工目检可视化**

打开 `datasets/task_ABC_D_scene_D_lerobot/depth/vis/episode_000000_vis.mp4`，确认：桌面/背景远、机械臂与操作物体近（inverse depth 下近处亮/热色）、时序无闪烁。**此步需用户确认通过后再进 Task 9。**

---

### Task 9: 全量运行 + 最终校验（操作性任务，长时）

**Files:** 无代码改动；产出为全量 5124 个 npz（约 24GB）。

**Interfaces:**
- Consumes: Task 8 已通过的完整链路。

- [ ] **Step 1: 找空闲 GPU**（同 Task 8 Step 1，同样的停止条件）

- [ ] **Step 2: 全量后台运行**

```bash
cd /mnt/data/dengqi/code/UniamVLA && mkdir -p logs && set -o pipefail
CUDA_VISIBLE_DEVICES=N bash runners/run_preprocess_depth.sh \
  2>&1 | tee "logs/depth_full_$(date +%Y%m%d_%H%M%S).log"
```

预期时长：vitl fp16 单卡数小时（30.9 万帧）。断点续跑：中断后原样重跑即可，已完成 episode 自动跳过（smoke test 的 3 个也会被跳过）。

Expected: `done: ok=5121 skipped=3 failed=0`，退出码 0。

- [ ] **Step 3: 全量对齐校验**

```bash
bash runners/run_preprocess_depth.sh --verify_only; echo "exit=$?"
```

Expected: `verify: 0 problem(s)`、`exit=0`。若有 failures/missing：查 `depth/failures.txt`，对失败 episode 用 `--overwrite` 定点重跑后再次校验。

- [ ] **Step 4: 收尾检查与汇报**

```bash
du -sh datasets/task_ABC_D_scene_D_lerobot/depth
find datasets/task_ABC_D_scene_D_lerobot/depth -name "*.npz" | wc -l   # expect 5124
```

向用户汇报：总数、体量、耗时、失败数（应为 0）、meta.json 位置，并提醒下游接线注意事项（输出已是逆深度，`DepthHead` 接线时跳过 `1/d`）。

---

## Self-Review 记录

- **Spec 覆盖**：§2 决策（relative/vitl/npz float16/仅 image）→ Task 6 默认值 + Task 2；§3 参数（518/全帧/fp16/pyav/截断）→ Task 3、5、6；§4 脚本设计（runners 路径、CLI、数据流、输出布局、meta.json 语义警告）→ Task 1、4、5、6、7；§5 校验（逐视频 assert、failures.txt、verify_only）→ Task 4、5；§6 运行计划（smoke → 全量、GPU 规则、tee 日志）→ Task 8、9；§7 测试策略（smoke、verify、断点续跑）→ Task 5 测试 + Task 8/9。无遗漏。
- **超出 spec 的最小新增**：`--limit` 与 `--vis_first_n` 两个 CLI 参数——它们是 spec §6 smoke test（"跑 2~3 个视频"、"存 1 个可视化 mp4"）的实现机制，非新功能。
- **占位符扫描**：Task 5 的 `save_depth_vis` 占位有明确的 Task 6 替换实现，非悬空 TODO。其余无占位。
- **类型/签名一致性**：`infer_fn(frames, fps, input_size)` 契约在 Task 5（消费）与 Task 6（生产）一致；`process_videos` 关键字参数在 Task 5 定义与 Task 6 `main` 调用一致；npz key `"depths"`、`failures.txt`、`meta.json` 命名全文一致。
