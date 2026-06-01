"""Spec §8.3 — ModelClient + step() wiring tests against a stubbed
WebSocket client. Catches the contract bugs §8.2 alignment test cannot:

- B1 (unnorm_key=None default) → must resolve via _check_unnorm_key in __init__.
- B2 (policy_ckpt_path is a .pt file) → must read run_dir/statistics.yaml,
  not Path(.pt) / "statistics.yaml".
- step() must always pop uamvla_raw_state and inject canonical_state when
  state passthrough is enabled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def stats_yaml_dict() -> dict:
    """Minimal statistics.yaml content with both action and state stats for franka_libero."""
    return {
        "view_names": ["static", "wrist"],
        "max_action_dim": 24,
        "embodiment_stats": {
            "franka_libero": {
                "action_dim": 7,
                "action_min_bound": [-1.0] * 7,
                "action_max_bound": [1.0] * 7,
            }
        },
        "state_stats": {
            "franka_libero": {
                "arm_0.ee_pose": {
                    "q01": [-0.5] * 9,
                    "q99": [0.5] * 9,
                    "min": [-1.0] * 9, "max": [1.0] * 9,
                    "mean": [0.0] * 9, "std": [0.3] * 9,
                },
                "arm_0.joint_pos": {
                    "q01": [-1.0] * 7,
                    "q99": [1.0] * 7,
                    "min": [-2.0] * 7, "max": [2.0] * 7,
                    "mean": [0.0] * 7, "std": [0.5] * 7,
                },
                "gripper_0": {
                    "q01": [0.0],
                    "q99": [1.0],
                    "min": [0.0], "max": [1.0],
                    "mean": [0.5], "std": [0.3],
                },
            }
        },
    }


@pytest.fixture
def fake_run_dir(tmp_path: Path, stats_yaml_dict: dict) -> Path:
    """Build a fake run dir with the .pt path layout read_mode_config expects."""
    run_dir = tmp_path / "fake_run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "fake.pt").write_bytes(b"")  # empty file is fine
    with open(run_dir / "statistics.yaml", "w") as f:
        yaml.safe_dump(stats_yaml_dict, f)
    return run_dir


@pytest.fixture
def patched_read_mode_config(monkeypatch, stats_yaml_dict):
    """Stub read_mode_config so __init__ can succeed without a real config.yaml."""
    fake_norm_stats = {
        "franka_libero": {
            "action": {
                "min": stats_yaml_dict["embodiment_stats"]["franka_libero"]["action_min_bound"],
                "max": stats_yaml_dict["embodiment_stats"]["franka_libero"]["action_max_bound"],
                "mask": [True] * 6 + [False],
            }
        }
    }
    fake_model_config = {
        "framework": {"action_model": {"future_action_window_size": 7}}
    }

    def fake(_path):
        return fake_model_config, fake_norm_stats

    monkeypatch.setattr(
        "examples.LIBERO.eval_files.model2libero_interface.read_mode_config",
        fake,
    )


class StubWebsocketClient:
    """Records the last predict_action payload; never opens a socket."""
    def __init__(self):
        self.last_payload = None
        self.num_calls = 0
    def predict_action(self, query_info: dict) -> dict:
        self.num_calls += 1
        self.last_payload = query_info
        # Return a syntactically valid response.
        return {
            "data": {
                "normalized_actions": np.zeros((1, 8, 7), dtype=np.float32)
            }
        }


@pytest.fixture
def make_client(fake_run_dir, patched_read_mode_config, monkeypatch):
    """Build a ModelClient connected to a stub websocket."""
    from examples.LIBERO.eval_files import model2libero_interface

    # Prevent the real WebsocketClientPolicy from trying to open a socket.
    monkeypatch.setattr(
        model2libero_interface, "WebsocketClientPolicy",
        lambda *a, **kw: StubWebsocketClient(),
    )

    def _make(unnorm_key=None, with_stats_yaml=True, **kwargs):
        if not with_stats_yaml:
            (fake_run_dir / "statistics.yaml").unlink(missing_ok=True)
        client = model2libero_interface.ModelClient(
            policy_ckpt_path=fake_run_dir / "checkpoints" / "fake.pt",
            unnorm_key=unnorm_key,
            action_ensemble=False,
            **kwargs,
        )
        return client

    return _make


# ---------------------------------------------------------------------------
# B1 + B2: __init__ resolves unnorm_key and locates statistics.yaml correctly.
# ---------------------------------------------------------------------------
def test_unnorm_key_is_resolved_when_none_passed(make_client):
    """B1: default unnorm_key=None must be resolved via _check_unnorm_key."""
    client = make_client(unnorm_key=None)
    assert client.unnorm_key == "franka_libero"


def test_state_passthrough_enabled_when_yaml_present(make_client):
    """B2 + state-passthrough init: statistics.yaml at run_dir/ engages the path."""
    client = make_client(unnorm_key=None)
    assert client.uamvla_state_enabled is True
    assert hasattr(client, "_adapter")
    assert hasattr(client, "_state_normalizer")


def test_state_passthrough_disabled_when_yaml_absent(make_client):
    """statistics.yaml absent ⇒ flag stays False; non-UamVLA models unaffected."""
    client = make_client(unnorm_key=None, with_stats_yaml=False)
    assert client.uamvla_state_enabled is False


# ---------------------------------------------------------------------------
# step() pop + convert behavior.
# ---------------------------------------------------------------------------
def _build_example_with_raw_state():
    return {
        "image": [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)],
        "lang": "pick up the bowl",
        "uamvla_raw_state": {
            "ee_pos":        np.array([0.1, 0.0, 0.5], dtype=np.float32),
            "ee_axis_angle": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "joint_pos":     np.zeros(7, dtype=np.float32),
            "gripper_qpos":  np.array([0.02, 0.02], dtype=np.float32),
        },
    }


def test_step_pops_raw_state_and_injects_canonical_state(make_client):
    client = make_client(unnorm_key=None)
    example = _build_example_with_raw_state()
    client.step(example, step=0)

    sent = client.client.last_payload
    assert "examples" in sent
    sent_example = sent["examples"][0]

    # Raw state must be popped (cleanliness: never on the wire).
    assert "uamvla_raw_state" not in sent_example
    # Canonical state must be injected with the expected nested shape.
    assert "canonical_state" in sent_example
    canonical = sent_example["canonical_state"]
    assert canonical["arm_0"]["ee_pose"].shape == (9,)
    assert canonical["arm_0"]["joint_pos"].shape == (7,)
    assert canonical["gripper_0"].shape == (1,)
    # Leaves must be numpy (msgpack-numpy can serialize them).
    assert isinstance(canonical["arm_0"]["ee_pose"], np.ndarray)


def test_step_pops_raw_state_even_when_passthrough_disabled(make_client):
    """If statistics.yaml is absent the gate is False, but step() must still
    pop uamvla_raw_state so it never reaches the wire."""
    client = make_client(unnorm_key=None, with_stats_yaml=False)
    example = _build_example_with_raw_state()
    client.step(example, step=0)

    sent = client.client.last_payload
    sent_example = sent["examples"][0]
    assert "uamvla_raw_state" not in sent_example
    assert "canonical_state" not in sent_example


def test_step_surfaces_policy_server_error_response(make_client):
    """A policy server inference error has no data field; surface its message."""
    client = make_client(unnorm_key=None)

    def fail_predict_action(_query_info: dict) -> dict:
        return {
            "status": "error",
            "ok": False,
            "type": "inference_result",
            "error": {
                "message": "synthetic server failure",
                "traceback": "Traceback line 1\nValueError: synthetic server failure",
            },
        }

    client.client.predict_action = fail_predict_action

    example = {
        "image": [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)],
        "lang": "move the slider left",
    }
    with pytest.raises(RuntimeError, match="Traceback line 1"):
        client.step(example, step=0)


def test_resize_image_is_noop_when_shape_already_matches():
    """CALVIN train-renderer eval feeds 256x256 images; avoid an extra resize."""
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient

    client = ModelClient.__new__(ModelClient)
    client.image_size = [256, 256]
    image = np.full((256, 256, 3), 123, dtype=np.uint8)

    out = client._resize_image(image)

    assert out is image


def test_unnormalize_actions_allows_gripper_threshold_override():
    """CALVIN trains gripper in the same signed normalized space as rel_actions,
    so its eval path needs a zero threshold; keep the legacy 0.5 default for
    other clients."""
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient

    stats = {
        "min": [-1.0] * 7,
        "max": [1.0] * 7,
        "mask": [True] * 6 + [False],
    }
    normalized = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.25],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
        ],
        dtype=np.float32,
    )

    default_out = ModelClient.unnormalize_actions(normalized.copy(), stats)
    calvin_out = ModelClient.unnormalize_actions(
        normalized.copy(),
        stats,
        gripper_binarize_threshold=0.0,
    )

    assert default_out[:, 6].tolist() == [0.0, 0.0, 1.0]
    assert calvin_out[:, 6].tolist() == [1.0, 0.0, 1.0]


def test_unnormalize_actions_can_interpret_libero_gripper_sign_as_open():
    """LIBERO datasets store gripper as action sign (-1=open, +1=close).

    The client still exposes open_gripper to eval_libero.py, so the sign needs
    to become open probability semantics before eval maps it back to LIBERO's
    env action convention.
    """
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient

    stats = {
        "min": [-1.0] * 7,
        "max": [1.0] * 7,
        "mask": [True] * 6 + [False],
    }
    normalized = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.75],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    out = ModelClient.unnormalize_actions(
        normalized.copy(),
        stats,
        gripper_binarize_threshold=0.0,
        gripper_interpretation="libero_sign",
    )

    assert out[:, 6].tolist() == [1.0, 0.0, 0.0]


def test_action_query_interval_replans_before_chunk_boundary(make_client):
    """CALVIN eval uses replan_steps=5 while checkpoints emit chunk=8; the
    client should requery on the requested interval and consume the fresh
    chunk from index 0."""
    client = make_client(unnorm_key=None, action_query_interval=5)
    example = {
        "image": [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)],
        "lang": "move the slider left",
    }

    client.step(dict(example), step=0)
    client.step(dict(example), step=4)
    assert client.client.num_calls == 1

    client.step(dict(example), step=5)
    assert client.client.num_calls == 2
