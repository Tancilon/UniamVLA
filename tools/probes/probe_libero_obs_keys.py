"""P1 probe (spec §8.1): verify LIBERO env exposes robot0_joint_pos.

Run inside libero_env with a valid LIBERO_HOME / MUJOCO_GL setup:

    python tools/probes/probe_libero_obs_keys.py

Exit code 0 ⇒ key present with shape (7,). Non-zero ⇒ design must be revisited.
"""
from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:
        print(f"FAIL: cannot import LIBERO ({e}); run inside libero_env.", file=sys.stderr)
        return 2

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    bddl = suite.get_task_bddl_file_path(0)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    try:
        obs = env.reset()

        print(f"obs.keys() = {sorted(obs.keys())}")
        if "robot0_joint_pos" not in obs:
            print("FAIL: 'robot0_joint_pos' missing from obs dict.", file=sys.stderr)
            return 1

        arr = np.asarray(obs["robot0_joint_pos"])
        print(f"robot0_joint_pos.shape = {arr.shape}, dtype = {arr.dtype}")
        if arr.shape != (7,):
            print(f"FAIL: expected shape (7,), got {arr.shape}.", file=sys.stderr)
            return 1

        print("OK: robot0_joint_pos present with shape (7,).")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
