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
    def predict_action(self, query_info: dict) -> dict:
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

    def _make(unnorm_key=None, with_stats_yaml=True):
        if not with_stats_yaml:
            (fake_run_dir / "statistics.yaml").unlink(missing_ok=True)
        client = model2libero_interface.ModelClient(
            policy_ckpt_path=fake_run_dir / "checkpoints" / "fake.pt",
            unnorm_key=unnorm_key,
            action_ensemble=False,
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
