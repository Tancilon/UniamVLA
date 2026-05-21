from __future__ import annotations

import importlib
import sys
import types

import numpy as np


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
        self.static = np.full((256, 256, 3), 17, dtype=np.uint8)
        self.wrist = np.full((256, 256, 3), 29, dtype=np.uint8)

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
    assert sent_images[0].shape == (256, 256, 3)
    assert sent_images[1].shape == (256, 256, 3)
    assert np.array_equal(sent_images[0], renderer.static)
    assert np.array_equal(sent_images[1], renderer.wrist)
    assert action.shape == (7,)


def test_calvin_policy_client_passes_uamvla_gr00t_robot_state(monkeypatch):
    eval_calvin = _load_eval_calvin(monkeypatch)

    class UamVLAGR00TModelClient(_FakeModelClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_config = {
                "framework": {
                    "name": "UamVLAGR00T",
                    "action_model": {"state_dim": 7},
                },
            }

    monkeypatch.setattr(eval_calvin, "ModelClient", UamVLAGR00TModelClient)

    policy = eval_calvin.CalvinPolicyClient(
        host="127.0.0.1",
        port=8000,
        pretrained_path="fake.pt",
        unnorm_key="franka_calvin",
    )

    robot_obs = np.arange(15, dtype=np.float32)
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
    np.testing.assert_allclose(sent_state, robot_obs[:7].reshape(1, 7))


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
