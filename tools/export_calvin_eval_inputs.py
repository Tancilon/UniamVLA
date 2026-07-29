from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = Path(
    "playground/Checkpoints/uamvla_gr00t_lt_calvin_abc_depth_afford_future"
    "/final_model/pytorch_model.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "playground/Checkpoints/uamvla_gr00t_lt_calvin_abc_depth_afford_future"
    "/eval_inputs"
)
DEFAULT_DATASET_PATH = Path("datasets/calvin/task_ABC_D")
DEFAULT_EVAL_SEQUENCES = Path("examples/calvin/eval_files/eval_sequences.json")
DEFAULT_CALVIN_CONFIG_PATH = Path("third_party/calvin/calvin_models/conf")


def _add_calvin_paths() -> None:
    for path in (
        REPO_ROOT,
        REPO_ROOT / "third_party/calvin/calvin_models",
        REPO_ROOT / "third_party/calvin/calvin_env",
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _load_gr00t_state_indices(ckpt: Path) -> list[int]:
    from omegaconf import OmegaConf

    run_dir = ckpt.parents[1]
    cfg = OmegaConf.load(run_dir / "config.yaml")
    indices = cfg.get("datasets", {}).get("vla_data", {}).get("gr00t_state_indices")
    if indices is None:
        indices = cfg.get("framework", {}).get("action_model", {}).get("gr00t_state_indices")
    if indices is None:
        indices = list(range(7))
    return [int(i) for i in indices]


def _normalize_eval_state(obs: dict[str, Any], ckpt: Path, unnorm_key: str | None) -> list[list[float]]:
    import numpy as np

    from examples.calvin.eval_files.eval_calvin import CalvinPolicyClient

    state_indices = _load_gr00t_state_indices(ckpt)
    mode, stat_a, stat_b = CalvinPolicyClient._load_uamvla_gr00t_state_stats(
        str(ckpt),
        unnorm_key,
        state_indices,
    )
    raw = CalvinPolicyClient._extract_raw_uamvla_gr00t_state(obs, state_indices).reshape(-1)

    normalized = np.zeros_like(raw, dtype=np.float32)
    if mode == "q99":
        mask = stat_a != stat_b
        normalized[mask] = 2 * (raw[mask] - stat_a[mask]) / (stat_b[mask] - stat_a[mask]) - 1
        normalized[~mask] = raw[~mask]
        normalized = np.clip(normalized, -1, 1)
    elif mode == "mean_std":
        mask = stat_b != 0
        normalized[mask] = (raw[mask] - stat_a[mask]) / stat_b[mask]
        normalized[~mask] = raw[~mask]
    else:
        raise ValueError(f"Unsupported state normalization mode: {mode!r}")
    return normalized.reshape(1, len(state_indices)).tolist()


def _lang_for_subtask(val_annotations: Any, subtask: str, diverse_inst: bool, sequence_i: int, subtask_i: int) -> str:
    if diverse_inst:
        lang = val_annotations[sequence_i][subtask_i]
    else:
        lang = val_annotations[subtask][0]
    return str(lang).split("\n")[0].replace("\u2019", "'")


def _load_val_annotations(calvin_config_path: Path) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.load(calvin_config_path / "annotations/new_playtable_validation.yaml")


def _eval_images_from_obs(obs: dict[str, Any], resize_size: int, train_renderer=None) -> list[Any]:
    from deployment.model_server.tools import image_tools

    if train_renderer is not None:
        rendered = train_renderer.render_cameras(width=resize_size, height=resize_size)
        return [
            image_tools.convert_to_uint8(rendered["rgb_static"]),
            image_tools.convert_to_uint8(rendered["rgb_wrist"]),
        ]

    return [
        image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs["rgb_obs"]["rgb_static"], resize_size, resize_size)
        ),
        image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs["rgb_obs"]["rgb_gripper"], resize_size, resize_size)
        ),
    ]


def _save_rgb(array: Any, path: Path) -> None:
    import numpy as np
    from PIL import Image

    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _add_calvin_paths()

    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
    from examples.calvin.eval_files.eval_calvin import make_env

    ckpt = Path(args.ckpt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(str(args.dataset_path))
    train_renderer = None
    if args.use_train_renderer:
        from tools.preprocess.calvin_env_adapter import CalvinEnvAdapter

        train_renderer = CalvinEnvAdapter(env)

    val_annotations = _load_val_annotations(Path(args.calvin_config_path))
    if args.diverse_inst:
        with open(args.diverse_inst_path) as f:
            val_annotations = json.load(f)

    with open(args.eval_sequences_path) as f:
        eval_sequences = json.load(f)
    eval_sequences = eval_sequences[: max(0, int(args.num_samples))]

    manifest: dict[str, Any] = {
        "dataset_path": str(args.dataset_path),
        "eval_sequences_path": str(args.eval_sequences_path),
        "checkpoint": str(ckpt),
        "resize_size": int(args.resize_size),
        "use_train_renderer": bool(args.use_train_renderer),
        "samples": [],
    }

    for sample_i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        obs = env.get_obs()
        subtask_i = 0
        subtask = eval_sequence[subtask_i]
        lang = _lang_for_subtask(val_annotations, subtask, args.diverse_inst, sample_i, subtask_i)
        static_img, wrist_img = _eval_images_from_obs(obs, args.resize_size, train_renderer=train_renderer)

        prefix = f"{sample_i:04d}_seq{sample_i:03d}_subtask{subtask_i}"
        static_name = f"{prefix}_static.png"
        wrist_name = f"{prefix}_wrist.png"
        _save_rgb(static_img, output_dir / static_name)
        _save_rgb(wrist_img, output_dir / wrist_name)

        manifest["samples"].append(
            {
                "index": sample_i,
                "sequence_index": sample_i,
                "subtask_index": subtask_i,
                "subtask": subtask,
                "lang": lang,
                "static_image": static_name,
                "wrist_image": wrist_name,
                "state": _normalize_eval_state(obs, ckpt, args.unnorm_key),
            }
        )

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported {len(manifest['samples'])} CALVIN eval inputs to {output_dir}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export CALVIN eval inputs exactly as sent to UamVLA inference.")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--calvin-config-path", type=Path, default=DEFAULT_CALVIN_CONFIG_PATH)
    parser.add_argument("--eval-sequences-path", type=Path, default=DEFAULT_EVAL_SEQUENCES)
    parser.add_argument("--unnorm-key", type=str, default=None)
    parser.add_argument("--use-train-renderer", action="store_true")
    parser.add_argument("--diverse-inst", action="store_true")
    parser.add_argument("--diverse-inst-path", type=str, default="/mnt/bn/robotics/lxh/robot-flamingo/lang_annotation_cache.json")
    return parser


def main() -> None:
    raise SystemExit(run(build_argparser().parse_args()))


if __name__ == "__main__":
    main()
