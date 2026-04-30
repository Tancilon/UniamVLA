"""Spec §8.4 — server-side stack_canonical receives the numpy-leaved
canonical_state coming from ModelClient.step() (no real WebSocket; we
exercise the deserialization → stack_canonical path directly)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

pytestmark = pytest.mark.slow


def _stats_yaml(tmp_path):
    stats = {
        "view_names": ["static", "wrist"],
        "max_action_dim": 24,
        "embodiment_stats": {"franka_libero": {
            "action_dim": 7,
            "action_min_bound": [-1.0] * 7,
            "action_max_bound": [1.0] * 7,
        }},
        "state_stats": {"franka_libero": {
            "arm_0.ee_pose":   {"q01": [-0.5] * 9, "q99": [0.5] * 9, "min": [-1.] * 9, "max": [1.] * 9, "mean": [0.] * 9, "std": [0.3] * 9},
            "arm_0.joint_pos": {"q01": [-1.0] * 7, "q99": [1.0] * 7, "min": [-2.] * 7, "max": [2.] * 7, "mean": [0.] * 7, "std": [0.5] * 7},
            "gripper_0":       {"q01": [0.0],      "q99": [1.0],      "min": [0.],     "max": [1.],     "mean": [0.5],    "std": [0.3]},
        }},
    }
    path = tmp_path / "statistics.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(stats, f)
    return path


def test_canonical_state_survives_msgpack_and_stack(tmp_path, monkeypatch):
    """ModelClient.step → msgpack-numpy → unpack → stack_canonical(numpy leaves)."""
    from examples.LIBERO.eval_files import model2libero_interface
    from deployment.model_server.tools import msgpack_numpy
    from starVLA.model.modules.uamvla.collator_helpers import stack_canonical

    # ----- Fake run dir
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "fake.pt").write_bytes(b"")
    _stats_yaml(run_dir)  # writes statistics.yaml under run_dir

    # ----- Stub read_mode_config
    monkeypatch.setattr(
        model2libero_interface, "read_mode_config",
        lambda _p: (
            {"framework": {"action_model": {"future_action_window_size": 7}}},
            {"franka_libero": {"action": {"min": [-1.] * 7, "max": [1.] * 7, "mask": [True] * 6 + [False]}}},
        ),
    )

    # ----- Stub websocket: capture payload, route through msgpack-numpy roundtrip,
    # then exercise stack_canonical on the deserialized canonical_state.
    captured = {}

    class RoundtripStub:
        def predict_action(self, query_info):
            data = msgpack_numpy.packb(query_info)
            payload = msgpack_numpy.unpackb(data)
            captured["payload"] = payload
            example = payload["examples"][0]
            # This is the contract: server-side stack_canonical must accept whatever
            # arrives over the wire. After the §6.4 fix it accepts numpy leaves.
            stacked = stack_canonical([example["canonical_state"]])
            captured["stacked"] = stacked
            return {"data": {"normalized_actions": np.zeros((1, 8, 7), dtype=np.float32)}}

    monkeypatch.setattr(
        model2libero_interface, "WebsocketClientPolicy",
        lambda *a, **kw: RoundtripStub(),
    )

    client = model2libero_interface.ModelClient(
        policy_ckpt_path=run_dir / "checkpoints" / "fake.pt",
        unnorm_key=None,
        action_ensemble=False,
    )
    assert client.uamvla_state_enabled

    example = {
        "image": [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)],
        "lang": "task",
        "uamvla_raw_state": {
            "ee_pos":        np.array([0.1, 0.0, 0.5], dtype=np.float32),
            "ee_axis_angle": np.zeros(3, dtype=np.float32),
            "joint_pos":     np.zeros(7, dtype=np.float32),
            "gripper_qpos":  np.array([0.02, 0.02], dtype=np.float32),
        },
    }
    client.step(example, step=0)

    # Wire-format assertions
    sent_example = captured["payload"]["examples"][0]
    assert isinstance(sent_example["canonical_state"]["arm_0"]["ee_pose"], np.ndarray)
    assert "uamvla_raw_state" not in sent_example

    # Server-side stack assertions
    stacked = captured["stacked"]
    assert isinstance(stacked["arm_0"]["ee_pose"], torch.Tensor)
    assert stacked["arm_0"]["ee_pose"].shape == (1, 9)
    assert stacked["arm_0"]["joint_pos"].shape == (1, 7)
    assert stacked["gripper_0"].shape == (1, 1)
