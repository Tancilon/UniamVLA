from __future__ import annotations

import os

from tools.preprocess.calvin_env_adapter import CalvinEnvAdapter


class _FakePlayTableEnv:
    def __init__(self, *, use_egl: bool = False):
        self.cid = 7
        self.ownsPhysicsClient = True
        self.use_egl = use_egl
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def test_calvin_env_adapter_close_marks_upstream_env_disconnected():
    adapter = CalvinEnvAdapter.__new__(CalvinEnvAdapter)
    env = _FakePlayTableEnv()
    adapter._env = env
    adapter._closed = False

    adapter.close()
    adapter.close()

    assert env.close_calls == 1
    assert env.cid == -1
    assert env.ownsPhysicsClient is False


def test_calvin_env_adapter_close_skips_native_egl_teardown(monkeypatch):
    monkeypatch.delenv("UAMVLA_CALVIN_NATIVE_SAFE_EXIT", raising=False)
    adapter = CalvinEnvAdapter.__new__(CalvinEnvAdapter)
    env = _FakePlayTableEnv(use_egl=True)
    adapter._env = env
    adapter._closed = False

    adapter.close()

    assert env.close_calls == 0
    assert env.cid == -1
    assert env.ownsPhysicsClient is False
    assert os.environ["UAMVLA_CALVIN_NATIVE_SAFE_EXIT"] == "1"
