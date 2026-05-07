"""Probe UamVLA CALVIN validation-frame action prediction via policy server.

This diagnostic uses the original CALVIN split instead of the preprocessed
UamVLA JSONL. For each selected validation language window it:

  1. loads `robot_obs`, `scene_obs`, and future `rel_actions` from `.npz` files;
  2. resets a real CALVIN env to the expert frame and renders static/wrist RGB
     with the same `CalvinEnvAdapter` path used by preprocessing;
  3. sends image + language + normalized canonical_state to the running policy
     server;
  4. compares predicted normalized action chunk against the expert chunk.

Interpretation:

  - high MAE on D validation expert frames points to ABC->D generalization or
    checkpoint quality;
  - low MAE here but poor live rollout points to closed-loop rollout/control/env
    state progression mismatch.

Example:

    python tools/probes/probe_uamvla_calvin_validation_chunk.py \
      --calvin-root datasets/task_ABC_D \
      --stats-root datasets/uamvla_calvin/task_ABC_D \
      --task rotate_blue_block_right \
      --frame-offset 0 \
      --num-samples 4 \
      --host 127.0.0.1 \
      --port 5694

To probe the later rotate phase of each 64-frame window:

    python tools/probes/probe_uamvla_calvin_validation_chunk.py \
      --calvin-root datasets/task_ABC_D \
      --stats-root datasets/uamvla_calvin/task_ABC_D \
      --task rotate_blue_block_right \
      --frame-offset 40
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ACTION_DIM = 7


@dataclass(frozen=True)
class ValidationWindow:
    window_idx: int
    ep_start: int
    ep_end: int
    instruction: str
    task_label: str


@dataclass(frozen=True)
class FrameSpec:
    window: ValidationWindow
    frame: int
    step_idx: int


def _to_numpy_leaves(obj: Any) -> Any:
    if hasattr(obj, "detach"):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _to_numpy_leaves(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_numpy_leaves(v) for v in obj)
    return obj


def _format_vec(vec: np.ndarray) -> str:
    return np.array2string(vec, precision=4, suppress_small=False, separator=", ")


def _safe_response_data(resp: dict) -> dict:
    if resp.get("status") == "error":
        raise RuntimeError(resp.get("error", {}).get("message", resp))
    if "data" in resp:
        return resp["data"]
    return resp


def _load_stats(stats_root: Path, embodiment: str) -> tuple[dict, np.ndarray, np.ndarray]:
    with (stats_root / "statistics.yaml").open() as f:
        stats = yaml.safe_load(f)
    emb = stats["embodiment_stats"][embodiment]
    action_min = np.asarray(emb["action_min_bound"], dtype=np.float32)
    action_max = np.asarray(emb["action_max_bound"], dtype=np.float32)
    return stats, action_min, action_max


def _normalize_action(action: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    denom = (action_max - action_min) + 1e-8
    return 2.0 * (action - action_min) / denom - 1.0


def _load_validation_windows(split_dir: Path) -> list[ValidationWindow]:
    lang_path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not lang_path.exists():
        raise FileNotFoundError(f"Missing CALVIN language annotations: {lang_path}")

    lang_ann = np.load(lang_path, allow_pickle=True).item()
    anns = lang_ann["language"]["ann"]
    tasks = lang_ann["language"].get("task", [""] * len(anns))
    indx = lang_ann["info"]["indx"]
    if not (len(anns) == len(tasks) == len(indx)):
        raise ValueError(
            f"Malformed {lang_path}: ann/task/indx lengths "
            f"{len(anns)}/{len(tasks)}/{len(indx)} must match"
        )

    windows: list[ValidationWindow] = []
    for i, (ann, task, rng) in enumerate(zip(anns, tasks, indx)):
        start, end = int(rng[0]), int(rng[1])
        windows.append(
            ValidationWindow(
                window_idx=i,
                ep_start=start,
                ep_end=end,
                instruction=str(ann),
                task_label=str(task),
            )
        )
    return windows


def _matches_text(value: str, query: str | None, contains: bool) -> bool:
    if query is None:
        return True
    return query in value if contains else value == query


def _select_probe_frames(
    windows: list[ValidationWindow],
    instruction: str | None,
    contains: bool,
    task: str | None,
    task_contains: bool,
    start_index: int,
    num_samples: int,
    frame_offset: int,
    horizon: int,
    min_valid_steps: int,
) -> list[FrameSpec]:
    selected: list[FrameSpec] = []
    seen_eligible = 0
    for window in windows:
        if not _matches_text(window.instruction, instruction, contains):
            continue
        if not _matches_text(window.task_label, task, task_contains):
            continue

        frame = window.ep_start + frame_offset
        if frame < window.ep_start or frame > window.ep_end:
            continue
        valid_steps = min(horizon, window.ep_end - frame + 1)
        if valid_steps < min_valid_steps:
            continue

        if seen_eligible >= start_index:
            selected.append(FrameSpec(window=window, frame=frame, step_idx=frame - window.ep_start))
            if len(selected) >= num_samples:
                break
        seen_eligible += 1
    return selected


def _load_rel_actions(split_dir: Path, start_frame: int, end_frame: int) -> dict[int, np.ndarray]:
    actions: dict[int, np.ndarray] = {}
    for frame in range(start_frame, end_frame + 1):
        path = split_dir / f"episode_{frame:07d}.npz"
        with np.load(path) as npz:
            actions[frame] = np.asarray(npz["rel_actions"], dtype=np.float32)[:ACTION_DIM]
    return actions


def _load_frame_obs(split_dir: Path, frame: int) -> tuple[np.ndarray, np.ndarray]:
    path = split_dir / f"episode_{frame:07d}.npz"
    with np.load(path) as npz:
        robot_obs = np.asarray(npz["robot_obs"], dtype=np.float32)
        scene_obs = np.asarray(npz["scene_obs"], dtype=np.float32)
    return robot_obs, scene_obs


def _build_target_chunk(
    actions_by_frame: dict[int, np.ndarray],
    start_frame: int,
    ep_end: int,
    horizon: int,
    action_min: np.ndarray,
    action_max: np.ndarray,
    min_valid_steps: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    valid_steps = min(horizon, ep_end - start_frame + 1)
    if valid_steps < min_valid_steps:
        return None

    target = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((horizon, ACTION_DIM), dtype=bool)
    for k in range(valid_steps):
        frame = start_frame + k
        if frame not in actions_by_frame:
            return None
        target[k] = _normalize_action(actions_by_frame[frame], action_min, action_max)
        mask[k] = True
    return target, mask


def _render_example(
    env,
    robot_obs: np.ndarray,
    scene_obs: np.ndarray,
    width: int,
    height: int,
) -> list[np.ndarray]:
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    rendered = env.render_cameras(width=width, height=height)
    return [
        np.asarray(rendered["rgb_static"], dtype=np.uint8).copy(),
        np.asarray(rendered["rgb_wrist"], dtype=np.uint8).copy(),
    ]


def run_probe(args: argparse.Namespace) -> int:
    split_dir = Path(args.calvin_root) / args.split
    stats_root = Path(args.stats_root)
    if not split_dir.exists():
        print(f"FAIL: missing CALVIN split dir {split_dir}", file=sys.stderr)
        return 2
    if not (stats_root / "statistics.yaml").exists():
        print(f"FAIL: missing stats file {stats_root / 'statistics.yaml'}", file=sys.stderr)
        return 2

    windows = _load_validation_windows(split_dir)
    selected = _select_probe_frames(
        windows=windows,
        instruction=args.instruction,
        contains=args.contains,
        task=args.task,
        task_contains=args.task_contains,
        start_index=args.start_index,
        num_samples=args.num_samples,
        frame_offset=args.frame_offset,
        horizon=args.horizon,
        min_valid_steps=args.min_valid_steps,
    )
    if not selected:
        print("FAIL: no eligible validation frames matched the filters.", file=sys.stderr)
        print(
            "Hint: try --task rotate_blue_block_right, --task-contains block, "
            "or lower --min-valid-steps near the end of a window.",
            file=sys.stderr,
        )
        return 1

    stats, action_min, action_max = _load_stats(stats_root, args.embodiment)

    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    from starVLA.model.modules.uamvla.data.embodiment_adapter import CalvinAdapter
    from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer
    from tools.preprocess.calvin_env_adapter import make_calvin_env_adapter

    adapter = CalvinAdapter()
    normalizer = StateNormalizer(
        stats_dict=stats,
        embodiment=args.embodiment,
        mode=args.state_norm_mode,
        apply_to=["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"],
    )

    env = make_calvin_env_adapter(str(split_dir))
    client = None
    all_abs: list[np.ndarray] = []
    all_sq: list[np.ndarray] = []
    all_sign: list[np.ndarray] = []
    all_first_abs: list[np.ndarray] = []

    print(
        f"matched_frames={len(selected)} split={split_dir} "
        f"stats_root={stats_root} server={args.host}:{args.port}"
    )
    print(
        f"filters: task={args.task!r} instruction={args.instruction!r} "
        f"frame_offset={args.frame_offset} horizon={args.horizon} "
        f"min_valid_steps={args.min_valid_steps}"
    )

    try:
        client = WebsocketClientPolicy(args.host, args.port)
        for i, spec in enumerate(selected):
            window = spec.window
            actions_by_frame = _load_rel_actions(
                split_dir=split_dir,
                start_frame=spec.frame,
                end_frame=min(window.ep_end, spec.frame + args.horizon - 1),
            )
            built = _build_target_chunk(
                actions_by_frame=actions_by_frame,
                start_frame=spec.frame,
                ep_end=window.ep_end,
                horizon=args.horizon,
                action_min=action_min,
                action_max=action_max,
                min_valid_steps=args.min_valid_steps,
            )
            if built is None:
                print(f"[{i}] WARN skipped short/missing chunk at frame={spec.frame}")
                continue
            target, mask = built

            robot_obs, scene_obs = _load_frame_obs(split_dir, spec.frame)
            images = _render_example(
                env=env,
                robot_obs=robot_obs,
                scene_obs=scene_obs,
                width=args.render_width,
                height=args.render_height,
            )
            canonical = normalizer(adapter.to_canonical({"robot_obs": robot_obs}))
            query_lang = args.query_lang or window.instruction
            example = {
                "image": images,
                "lang": query_lang,
                "canonical_state": _to_numpy_leaves(canonical),
            }
            resp = client.predict_action(
                {
                    "examples": [example],
                    "do_sample": False,
                    "use_ddim": True,
                    "num_ddim_steps": 10,
                }
            )
            data = _safe_response_data(resp)
            pred = np.asarray(data["normalized_actions"][0], dtype=np.float32)[: args.horizon, :ACTION_DIM]

            valid = mask.astype(bool)
            diff = pred - target
            abs_err = np.where(valid, np.abs(diff), np.nan)
            sq_err = np.where(valid, diff * diff, np.nan)
            sign_agree = np.where(valid, np.sign(pred) == np.sign(target), np.nan)

            all_abs.append(abs_err)
            all_sq.append(sq_err)
            all_sign.append(sign_agree.astype(np.float32))
            all_first_abs.append(np.abs(pred[0] - target[0]))

            valid_steps = int(valid[:, 0].sum())
            print(
                f"\n[{i}] window={window.window_idx} frame={spec.frame} "
                f"step={spec.step_idx}/{window.ep_end - window.ep_start} "
                f"valid_steps={valid_steps}"
            )
            print(f"    task={window.task_label!r}")
            print(f"    ann={window.instruction!r}")
            print(f"    query_lang={query_lang!r}")
            print(f"    target_first={_format_vec(target[0])}")
            print(f"    pred_first  ={_format_vec(pred[0])}")
            print(f"    target_mean ={_format_vec(np.nanmean(np.where(valid, target, np.nan), axis=0))}")
            print(f"    pred_mean   ={_format_vec(np.nanmean(np.where(valid, pred, np.nan), axis=0))}")
            print(f"    mae_per_dim ={_format_vec(np.nanmean(abs_err, axis=0))}")
            print(f"    sign_agree ={_format_vec(np.nanmean(sign_agree, axis=0))}")
    finally:
        if client is not None:
            client.close()
        env.close()

    if not all_abs:
        print("FAIL: no comparable samples produced.", file=sys.stderr)
        return 1

    abs_arr = np.concatenate(all_abs, axis=0)
    sq_arr = np.concatenate(all_sq, axis=0)
    sign_arr = np.concatenate(all_sign, axis=0)
    first_abs = np.stack(all_first_abs)
    rmse = np.sqrt(np.nanmean(sq_arr, axis=0))

    print("\n=== aggregate ===")
    print(f"samples={len(all_abs)} horizon={args.horizon}")
    print(f"mae_per_dim       ={_format_vec(np.nanmean(abs_arr, axis=0))}")
    print(f"rmse_per_dim      ={_format_vec(rmse)}")
    print(f"first_mae_per_dim ={_format_vec(np.nanmean(first_abs, axis=0))}")
    print(f"sign_agree_per_dim={_format_vec(np.nanmean(sign_arr, axis=0))}")
    print(f"overall_mae       ={float(np.nanmean(abs_arr)):.6f}")
    print(f"overall_rmse      ={float(math.sqrt(float(np.nanmean(sq_arr)))):.6f}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calvin-root", default="datasets/task_ABC_D")
    parser.add_argument("--stats-root", default="datasets/uamvla_calvin/task_ABC_D")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--task", default=None, help="Exact CALVIN task label filter, e.g. rotate_blue_block_right.")
    parser.add_argument("--task-contains", action="store_true", help="Use substring matching for --task.")
    parser.add_argument("--instruction", default=None, help="Language annotation filter.")
    parser.add_argument("--query-lang", default=None, help="Optional language sent to the server instead of annotation.")
    parser.add_argument("--contains", action="store_true", help="Use substring matching for --instruction.")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-offset", type=int, default=0, help="Frame offset inside each language window.")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--min-valid-steps", type=int, default=8)
    parser.add_argument("--render-width", type=int, default=256)
    parser.add_argument("--render-height", type=int, default=256)
    parser.add_argument("--embodiment", default="franka_calvin")
    parser.add_argument("--state-norm-mode", default="q99")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    return parser


def main() -> int:
    return run_probe(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
