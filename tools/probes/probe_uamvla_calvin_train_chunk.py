"""Probe UamVLA CALVIN train-frame action prediction via a running policy server.

This is a read-only diagnostic for train/eval mismatch debugging. It samples
rows from a preprocessed UamVLA CALVIN JSONL dataset, sends the same images,
language, and canonical_state to the websocket policy server, then compares the
predicted normalized action chunk against the training label chunk.

Example:

    python tools/probes/probe_uamvla_calvin_train_chunk.py \
      --data-root datasets/uamvla_calvin/task_ABC_D \
      --instruction "take the blue block and rotate it right" \
      --num-samples 4 \
      --host 127.0.0.1 \
      --port 5694

To test an eval paraphrase while keeping labels from the matched train sample:

    python tools/probes/probe_uamvla_calvin_train_chunk.py \
      --data-root datasets/uamvla_calvin/task_ABC_D \
      --instruction "take the blue block and rotate it right" \
      --query-lang "take the blue block and rotate it to the right"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.modules.uamvla.data.embodiment_adapter import CalvinAdapter
from starVLA.model.modules.uamvla.data.state_normalizer import StateNormalizer


ACTION_DIM = 7


def _to_numpy_leaves(obj: Any) -> Any:
    if hasattr(obj, "detach"):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _to_numpy_leaves(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_numpy_leaves(v) for v in obj)
    return obj


def _load_stats(data_root: Path, embodiment: str) -> tuple[dict, np.ndarray, np.ndarray]:
    with (data_root / "statistics.yaml").open() as f:
        stats = yaml.safe_load(f)
    emb = stats["embodiment_stats"][embodiment]
    action_min = np.asarray(emb["action_min_bound"], dtype=np.float32)
    action_max = np.asarray(emb["action_max_bound"], dtype=np.float32)
    return stats, action_min, action_max


def _normalize_action(action: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    denom = (action_max - action_min) + 1e-8
    return 2.0 * (action - action_min) / denom - 1.0


def _iter_jsonl(path: Path):
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            yield row


def _select_rows(
    data_jsonl: Path,
    instruction: str,
    contains: bool,
    start_index: int,
    num_samples: int,
) -> list[dict]:
    selected: list[dict] = []
    seen_matches = 0
    for row in _iter_jsonl(data_jsonl):
        lang = row.get("instruction", "")
        matched = instruction in lang if contains else lang == instruction
        if not matched:
            continue
        if seen_matches >= start_index:
            selected.append(row)
            if len(selected) >= num_samples:
                break
        seen_matches += 1
    return selected


def _collect_action_chunks(
    data_jsonl: Path,
    selected: list[dict],
    horizon: int,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    wanted: dict[str, set[int]] = {}
    for row in selected:
        eid = row["episode_id"]
        t0 = int(row["step_idx"])
        total_steps = int(row["total_steps"])
        wanted.setdefault(eid, set()).update(range(t0, min(total_steps, t0 + horizon)))

    rows: dict[tuple[str, int], dict] = {}
    for row in _iter_jsonl(data_jsonl):
        eid = row.get("episode_id")
        if eid not in wanted:
            continue
        t = int(row["step_idx"])
        if t in wanted[eid]:
            rows[(eid, t)] = row

    chunks: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for row in selected:
        eid = row["episode_id"]
        t0 = int(row["step_idx"])
        total_steps = int(row["total_steps"])
        action = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
        mask = np.zeros((horizon, ACTION_DIM), dtype=bool)
        for k in range(horizon):
            t = t0 + k
            if t >= total_steps:
                break
            future = rows.get((eid, t))
            if future is None:
                continue
            raw_action = np.asarray(future["action"][:ACTION_DIM], dtype=np.float32)
            raw_mask = np.asarray(future.get("action_mask", [1] * ACTION_DIM)[:ACTION_DIM], dtype=bool)
            action[k] = _normalize_action(raw_action, action_min, action_max)
            mask[k] = raw_mask
        chunks[(eid, t0)] = (action, mask)
    return chunks


def _load_images(data_root: Path, row: dict) -> list[np.ndarray]:
    images = []
    for rel in row["image"]:
        with Image.open(data_root / rel) as img:
            images.append(np.asarray(img.convert("RGB"), dtype=np.uint8).copy())
    return images


def _format_vec(vec: np.ndarray) -> str:
    return np.array2string(vec, precision=4, suppress_small=False, separator=", ")


def _safe_response_data(resp: dict) -> dict:
    if resp.get("status") == "error":
        raise RuntimeError(resp.get("error", {}).get("message", resp))
    if "data" in resp:
        return resp["data"]
    return resp


def run_probe(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    data_jsonl = data_root / "data.jsonl"
    if not data_jsonl.exists():
        print(f"FAIL: missing {data_jsonl}", file=sys.stderr)
        return 2

    stats, action_min, action_max = _load_stats(data_root, args.embodiment)
    selected = _select_rows(
        data_jsonl=data_jsonl,
        instruction=args.instruction,
        contains=args.contains,
        start_index=args.start_index,
        num_samples=args.num_samples,
    )
    if not selected:
        mode = "contains" if args.contains else "exact"
        print(f"FAIL: no rows found by {mode} match: {args.instruction!r}", file=sys.stderr)
        return 1

    chunks = _collect_action_chunks(
        data_jsonl=data_jsonl,
        selected=selected,
        horizon=args.horizon,
        action_min=action_min,
        action_max=action_max,
    )

    adapter = CalvinAdapter()
    normalizer = StateNormalizer(
        stats_dict=stats,
        embodiment=args.embodiment,
        mode=args.state_norm_mode,
        apply_to=["arm_0.ee_pose", "arm_0.joint_pos", "gripper_0"],
    )

    client = WebsocketClientPolicy(args.host, args.port)
    all_abs: list[np.ndarray] = []
    all_sq: list[np.ndarray] = []
    all_sign: list[np.ndarray] = []
    all_first_abs: list[np.ndarray] = []

    print(f"matched_rows={len(selected)} query_lang={args.query_lang or '<same as train row>'!r}")
    print(f"server={args.host}:{args.port} data_root={data_root}")

    try:
        for i, row in enumerate(selected):
            target, mask = chunks[(row["episode_id"], int(row["step_idx"]))]
            query_lang = args.query_lang or row["instruction"]
            canonical = normalizer(adapter.to_canonical(row))
            example = {
                "image": _load_images(data_root, row),
                "lang": query_lang,
                "canonical_state": _to_numpy_leaves(canonical),
            }
            resp = client.predict_action({
                "examples": [example],
                "do_sample": False,
                "use_ddim": True,
                "num_ddim_steps": 10,
            })
            data = _safe_response_data(resp)
            pred = np.asarray(data["normalized_actions"][0], dtype=np.float32)[: args.horizon, :ACTION_DIM]

            valid = mask.astype(bool)
            if not valid.any():
                print(f"[{i}] WARN no valid action labels for {row['id']}")
                continue
            diff = pred - target
            abs_err = np.where(valid, np.abs(diff), np.nan)
            sq_err = np.where(valid, diff * diff, np.nan)
            sign_agree = np.where(valid, np.sign(pred) == np.sign(target), np.nan)

            all_abs.append(abs_err)
            all_sq.append(sq_err)
            all_sign.append(sign_agree.astype(np.float32))
            all_first_abs.append(np.abs(pred[0] - target[0]))

            print(f"\n[{i}] id={row['id']} line={row['_line_no']} episode={row['episode_id']} step={row['step_idx']}")
            print(f"    train_lang={row['instruction']!r}")
            print(f"    query_lang={query_lang!r}")
            print(f"    target_first={_format_vec(target[0])}")
            print(f"    pred_first  ={_format_vec(pred[0])}")
            print(f"    target_mean ={_format_vec(np.nanmean(np.where(valid, target, np.nan), axis=0))}")
            print(f"    pred_mean   ={_format_vec(np.nanmean(np.where(valid, pred, np.nan), axis=0))}")
            print(f"    mae_per_dim ={_format_vec(np.nanmean(abs_err, axis=0))}")
            print(f"    sign_agree ={_format_vec(np.nanmean(sign_agree, axis=0))}")
    finally:
        client.close()

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
    parser.add_argument("--data-root", default="datasets/uamvla_calvin/task_ABC_D")
    parser.add_argument("--instruction", required=True, help="Training instruction to select rows from data.jsonl.")
    parser.add_argument("--query-lang", default=None, help="Optional language sent to the server instead of the row instruction.")
    parser.add_argument("--contains", action="store_true", help="Use substring matching for --instruction.")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--embodiment", default="franka_calvin")
    parser.add_argument("--state-norm-mode", default="q99")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    return parser


def main() -> int:
    return run_probe(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
