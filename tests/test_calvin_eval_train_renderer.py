from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import numpy as np
from PIL import Image


def _install_eval_calvin_import_stubs(monkeypatch):
    """Provide lightweight stubs for CALVIN-only deps absent in unit-test envs."""
    calvin_agent = types.ModuleType("calvin_agent")
    evaluation = types.ModuleType("calvin_agent.evaluation")
    utils = types.ModuleType("calvin_agent.evaluation.utils")
    utils.collect_plan = lambda *a, **kw: None
    utils.count_success = lambda results: [0.0] * 5
    utils.get_env_state_for_initial_condition = lambda initial_state: initial_state
    utils.get_log_dir = lambda path: path
    utils.print_and_save = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "calvin_agent", calvin_agent)
    monkeypatch.setitem(sys.modules, "calvin_agent.evaluation", evaluation)
    monkeypatch.setitem(sys.modules, "calvin_agent.evaluation.utils", utils)

    moviepy = types.ModuleType("moviepy")
    moviepy_editor = types.ModuleType("moviepy.editor")
    moviepy_editor.ImageSequenceClip = object
    monkeypatch.setitem(sys.modules, "moviepy", moviepy)
    monkeypatch.setitem(sys.modules, "moviepy.editor", moviepy_editor)

    hydra = types.ModuleType("hydra")
    hydra.utils = types.SimpleNamespace(instantiate=lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, "hydra", hydra)

    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = types.SimpleNamespace(load=lambda *a, **kw: None, create=lambda x: x)
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)

    termcolor = types.ModuleType("termcolor")
    termcolor.colored = lambda text, *_a, **_kw: text
    monkeypatch.setitem(sys.modules, "termcolor", termcolor)

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda iterable, *a, **kw: iterable
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)

    tyro = types.ModuleType("tyro")
    tyro.cli = lambda fn: fn
    monkeypatch.setitem(sys.modules, "tyro", tyro)

    model2libero_interface = types.ModuleType("examples.LIBERO.eval_files.model2libero_interface")
    model2libero_interface.ModelClient = object
    monkeypatch.setitem(
        sys.modules,
        "examples.LIBERO.eval_files.model2libero_interface",
        model2libero_interface,
    )


def _load_eval_calvin(monkeypatch):
    _install_eval_calvin_import_stubs(monkeypatch)
    sys.modules.pop("examples.calvin.eval_files.eval_calvin", None)
    return importlib.import_module("examples.calvin.eval_files.eval_calvin")


def test_eval_calvin_defers_annotations_for_python38_compat():
    source_path = Path("examples/calvin/eval_files/eval_calvin.py")
    first_lines = source_path.read_text().splitlines()[:25]

    assert "from __future__ import annotations" in first_lines


class _FakeModelClient:
    def __init__(self, *args, **kwargs):
        self.uamvla_state_enabled = False
        self.last_example = None

    def step(self, example: dict, step: int = 0):
        self.last_example = example
        return {
            "raw_action": {
                "world_vector": np.zeros(3, dtype=np.float32),
                "rotation_delta": np.zeros(3, dtype=np.float32),
                "open_gripper": np.ones(1, dtype=np.float32),
            }
        }


class _FakeTrainRenderer:
    def __init__(self):
        self.calls = []
        rng = np.random.default_rng(42)
        self.static = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        self.wrist = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)

    def render_cameras(self, width: int, height: int) -> dict:
        self.calls.append((width, height))
        return {"rgb_static": self.static, "rgb_wrist": self.wrist}


def test_calvin_policy_client_uses_train_renderer_images(monkeypatch):
    eval_calvin = _load_eval_calvin(monkeypatch)
    monkeypatch.setattr(eval_calvin, "ModelClient", _FakeModelClient)

    renderer = _FakeTrainRenderer()
    policy = eval_calvin.CalvinPolicyClient(
        host="127.0.0.1",
        port=8000,
        pretrained_path="fake.pt",
        unnorm_key="franka_calvin",
        train_renderer=renderer,
    )

    obs = {
        "rgb_obs": {
            "rgb_static": np.zeros((200, 200, 3), dtype=np.uint8),
            "rgb_gripper": np.zeros((84, 84, 3), dtype=np.uint8),
        },
        "robot_obs": np.zeros(15, dtype=np.float32),
    }
    action = policy.step(obs, "push the drawer")

    assert renderer.calls == [(256, 256)]
    sent_images = policy.client.last_example["image"]
    for sent, rendered in zip(sent_images, (renderer.static, renderer.wrist)):
        assert sent.shape == (224, 224, 3)
        assert sent.dtype == np.uint8
        # Use the training reader's exact operation as the reference.
        expected = np.array(Image.fromarray(rendered).resize((224, 224)))
        np.testing.assert_array_equal(sent, expected)
    assert action.shape == (7,)


@pytest.mark.parametrize('h,s', [(5, 5), (3, 4), (1, 1)])
def test_dit_history_caches_resized_images_on_every_environment_step(monkeypatch, h, s):
    eval_calvin = _load_eval_calvin(monkeypatch)

    class DiTClient(_FakeModelClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_config = {"framework": {"name": "UamVLA_DiT"}, "datasets": {"vla_data": {"num_history_frames": h, "history_interval": s}}}
            self.query_steps = []

        def step(self, example, step=0):
            if step % 8 == 0:
                self.query_steps.append(step)
            return super().step(example, step)

    monkeypatch.setattr(eval_calvin, "ModelClient", DiTClient)
    renderer = _FakeTrainRenderer()
    policy = eval_calvin.CalvinPolicyClient(
        host="127.0.0.1", port=8000, pretrained_path="fake.pt",
        unnorm_key="franka", train_renderer=renderer,
    )
    obs = {"robot_obs": np.zeros(15, dtype=np.float32)}
    frames = []
    for step in range(31):
        renderer.static = np.roll(renderer.static, 1, axis=0)
        renderer.wrist = np.roll(renderer.wrist, 1, axis=1)
        policy.step(obs, "push the drawer")
        sent = policy.client.last_example
        assert "state" not in sent
        frames.append([image.copy() for image in sent["image"]])
        for actual, expected in zip(policy._dit_image_history[-1], frames[-1]):
            np.testing.assert_array_equal(actual, expected)
            assert not np.shares_memory(actual, expected)
        assert len(sent["image_history"]) == h
        for pair, offset in zip(sent["image_history"], range(-h * s, 0, s)):
            for actual, expected in zip(pair, frames[max(step + offset, 0)]):
                np.testing.assert_array_equal(actual, expected)
                assert actual.shape == (224, 224, 3)

    assert renderer.calls == [(256, 256)] * 31
    assert policy.client.query_steps == [0, 8, 16, 24]
    assert len(policy._dit_image_history) == min(31, h * s)
    policy.reset(clear_history=True)
    assert not policy._dit_image_history
    assert policy.step_count == 0


def test_calvin_policy_client_normalizes_uamvla_gr00t_robot_state(monkeypatch, tmp_path):
    eval_calvin = _load_eval_calvin(monkeypatch)

    class UamVLAGR00TModelClient(_FakeModelClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.unnorm_key = "franka"
            self.model_config = {
                "framework": {
                    "name": "UamVLAGR00T",
                    "action_model": {"state_dim": 7},
                },
            }

    monkeypatch.setattr(eval_calvin, "ModelClient", UamVLAGR00TModelClient)

    run_dir = tmp_path / "run"
    ckpt_path = run_dir / "checkpoints" / "fake.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"")
    mean = np.arange(15, dtype=np.float32)
    std = np.arange(15, dtype=np.float32) + 1.0
    with open(run_dir / "dataset_statistics.json", "w") as f:
        json.dump(
            {
                "franka": {
                    "state": {
                        "mean": mean.tolist(),
                        "std": std.tolist(),
                    },
                    "action": {
                        "min": [-1.0] * 7,
                        "max": [1.0] * 7,
                        "mask": [True] * 6 + [False],
                    },
                }
            },
            f,
        )

    policy = eval_calvin.CalvinPolicyClient(
        host="127.0.0.1",
        port=8000,
        pretrained_path=str(ckpt_path),
        unnorm_key="franka_calvin",
    )

    robot_obs = np.arange(15, dtype=np.float32) + 10.0
    obs = {
        "rgb_obs": {
            "rgb_static": np.zeros((200, 200, 3), dtype=np.uint8),
            "rgb_gripper": np.zeros((84, 84, 3), dtype=np.uint8),
        },
        "robot_obs": robot_obs,
    }
    policy.step(obs, "push the drawer")

    sent_state = policy.client.last_example["state"]
    assert sent_state.shape == (1, 7)
    assert sent_state.dtype == np.float32
    expected_state = ((robot_obs[:7] - mean[:7]) / std[:7]).reshape(1, 7)
    np.testing.assert_allclose(sent_state, expected_state)


def test_calvin_policy_client_normalizes_uamgr00t_starvla_state_with_q99_indices(monkeypatch, tmp_path):
    eval_calvin = _load_eval_calvin(monkeypatch)

    class UamGR00TModelClient(_FakeModelClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.unnorm_key = "franka"
            self.model_config = {
                "framework": {
                    "name": "UamGR00T",
                    "action_model": {"state_dim": 7},
                },
                "datasets": {
                    "vla_data": {
                        "gr00t_state_indices": [0, 1, 2, 3, 4, 5, 7],
                    },
                },
            }

    monkeypatch.setattr(eval_calvin, "ModelClient", UamGR00TModelClient)

    run_dir = tmp_path / "run"
    ckpt_path = run_dir / "final_model" / "pytorch_model.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"")
    with open(run_dir / "dataset_statistics.json", "w") as f:
        json.dump(
            {
                "franka": {
                    "state": {
                        "q01": [0.0] * 26,
                        "q99": [10.0] * 26,
                    },
                    "action": {
                        "q01": [-0.5] * 7,
                        "q99": [0.5] * 7,
                        "min": [-1.0] * 7,
                        "max": [1.0] * 7,
                        "mask": [True] * 6 + [False],
                    },
                }
            },
            f,
        )

    policy = eval_calvin.CalvinPolicyClient(
        host="127.0.0.1",
        port=8000,
        pretrained_path=str(ckpt_path),
        unnorm_key="franka",
    )

    robot_obs = np.array(
        [0.0, 5.0, 10.0, 2.5, 7.5, 10.0, 123.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    obs = {
        "rgb_obs": {
            "rgb_static": np.zeros((200, 200, 3), dtype=np.uint8),
            "rgb_gripper": np.zeros((84, 84, 3), dtype=np.uint8),
        },
        "robot_obs": robot_obs,
    }
    policy.step(obs, "push the drawer")

    sent_state = policy.client.last_example["state"]
    assert sent_state.shape == (1, 7)
    expected_state = np.array(
        [[-1.0, 0.0, 1.0, -0.5, 0.5, 1.0, -0.8]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(sent_state, expected_state)


def test_calvin_policy_client_passes_dt_raw_state_history(monkeypatch):
    eval_calvin = _load_eval_calvin(monkeypatch)

    class UamGR00TDTModelClient(_FakeModelClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_config = {
                "framework": {
                    "name": "UamGR00T_DT",
                    "state_dim": 7,
                    # DT intentionally disables the action-head state branch.
                    "action_model": {"state_dim": 0},
                    "history_frames": 10,
                },
                "datasets": {
                    "vla_data": {
                        "gr00t_state_indices": [0, 1, 2, 3, 4, 5, 7],
                    },
                },
            }

    monkeypatch.setattr(eval_calvin, "ModelClient", UamGR00TDTModelClient)
    policy = eval_calvin.CalvinPolicyClient(
        host="127.0.0.1",
        port=8000,
        pretrained_path="fake.pt",
        unnorm_key="franka",
    )

    def obs_at(step):
        robot_obs = np.arange(15, dtype=np.float32) + step * 100.0
        return {
            "rgb_obs": {
                "rgb_static": np.zeros((200, 200, 3), dtype=np.uint8),
                "rgb_gripper": np.zeros((84, 84, 3), dtype=np.uint8),
            },
            "robot_obs": robot_obs,
        }

    expected_states = []
    for step in range(10):
        policy.step(obs_at(step), "push the drawer")
        expected_states.append(obs_at(step)["robot_obs"][[0, 1, 2, 3, 4, 5, 14]])
        if step == 0:
            np.testing.assert_allclose(
                policy.client.last_example["state"],
                np.repeat(expected_states[0][None], 10, axis=0),
            )
            assert policy.client.last_example["state"][0, -1] == obs_at(0)["robot_obs"][14]

    sent_state = policy.client.last_example["state"]
    assert sent_state.shape == (10, 7)
    assert sent_state.dtype == np.float32
    np.testing.assert_allclose(sent_state, np.stack(expected_states))

    policy.reset(clear_history=True)
    policy.step(obs_at(10), "push the drawer")
    np.testing.assert_allclose(
        policy.client.last_example["state"],
        np.repeat(expected_states[-1][None], 10, axis=0) + 100.0,
    )


def test_calvin_policy_client_does_not_pass_robot_state_to_other_frameworks(monkeypatch):
    eval_calvin = _load_eval_calvin(monkeypatch)

    class QwenOFTModelClient(_FakeModelClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_config = {
                "framework": {
                    "name": "QwenOFT",
                    "action_model": {"state_dim": 0},
                },
            }

    monkeypatch.setattr(eval_calvin, "ModelClient", QwenOFTModelClient)

    policy = eval_calvin.CalvinPolicyClient(
        host="127.0.0.1",
        port=8000,
        pretrained_path="fake.pt",
        unnorm_key="franka_calvin",
    )

    obs = {
        "rgb_obs": {
            "rgb_static": np.zeros((200, 200, 3), dtype=np.uint8),
            "rgb_gripper": np.zeros((84, 84, 3), dtype=np.uint8),
        },
        "robot_obs": np.arange(15, dtype=np.float32),
    }
    policy.step(obs, "push the drawer")

    assert "state" not in policy.client.last_example


def test_calvin_policy_client_passes_zero_gripper_threshold(monkeypatch):
    """CALVIN rel_actions use a signed gripper convention; the generic client
    must not apply its legacy 0.5 binary threshold here."""
    eval_calvin = _load_eval_calvin(monkeypatch)
    captured = {}

    class CapturingModelClient(_FakeModelClient):
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(eval_calvin, "ModelClient", CapturingModelClient)

    eval_calvin.CalvinPolicyClient(
        host="127.0.0.1",
        port=8000,
        pretrained_path="fake.pt",
        unnorm_key="franka_calvin",
    )

    assert captured["kwargs"]["gripper_binarize_threshold"] == 0.0


def test_calvin_policy_client_uses_model_default_action_interval(monkeypatch):
    eval_calvin = _load_eval_calvin(monkeypatch)
    captured = {}

    class CapturingModelClient(_FakeModelClient):
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(eval_calvin, "ModelClient", CapturingModelClient)

    eval_calvin.CalvinPolicyClient(
        host="127.0.0.1",
        port=8000,
        pretrained_path="fake.pt",
        unnorm_key="franka_calvin",
    )

    assert captured["kwargs"]["action_query_interval"] is None


def test_calvin_policy_client_can_override_replan_steps(monkeypatch):
    eval_calvin = _load_eval_calvin(monkeypatch)
    captured = {}

    class CapturingModelClient(_FakeModelClient):
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(eval_calvin, "ModelClient", CapturingModelClient)

    eval_calvin.CalvinPolicyClient(
        host="127.0.0.1",
        port=8000,
        replan_steps=5,
        pretrained_path="fake.pt",
        unnorm_key="franka_calvin",
    )

    assert captured["kwargs"]["action_query_interval"] == 5


def test_limit_eval_sequences_respects_requested_count(monkeypatch):
    eval_calvin = _load_eval_calvin(monkeypatch)
    sequences = [
        ("state_0", ["task_a"]),
        ("state_1", ["task_b"]),
        ("state_2", ["task_c"]),
    ]

    assert eval_calvin._limit_eval_sequences(sequences, 2) == sequences[:2]
    assert eval_calvin._limit_eval_sequences(sequences, 10) == sequences


def test_rollout_clears_temporal_history_at_every_subtask_boundary(monkeypatch):
    eval_calvin = _load_eval_calvin(monkeypatch)

    class FakePolicy:
        def __init__(self):
            self.reset_calls = []

        def reset(self, clear_history=True):
            self.reset_calls.append(clear_history)

        def step(self, obs, lang_annotation):
            return np.zeros(7, dtype=np.float32)

    class FakeEnv:
        def get_obs(self):
            return {"rgb_obs": {"rgb_static": np.zeros((1, 1, 3), dtype=np.uint8)}}

        def get_info(self):
            return {"step": 0}

        def step(self, action):
            return self.get_obs(), 0.0, False, {"step": 1}

    class SuccessfulTaskOracle:
        def get_task_info_for_set(self, start_info, current_info, tasks):
            return tasks

    policy = FakePolicy()
    success = eval_calvin.rollout(
        env=FakeEnv(),
        policy=policy,
        task_oracle=SuccessfulTaskOracle(),
        subtask="open_drawer",
        val_annotations={"open_drawer": ["pull the handle to open the drawer"]},
        plans={},
        debug=False,
    )

    assert success is True
    assert policy.reset_calls == [True]
