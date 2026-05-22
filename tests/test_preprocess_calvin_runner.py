from __future__ import annotations

import importlib
import sys
import types

import pytest


def _load_runner(monkeypatch):
    imageio = types.ModuleType("imageio")
    imageio_v3 = types.ModuleType("imageio.v3")
    monkeypatch.setitem(sys.modules, "imageio", imageio)
    monkeypatch.setitem(sys.modules, "imageio.v3", imageio_v3)
    sys.modules.pop("runners.preprocess_calvin", None)
    return importlib.import_module("runners.preprocess_calvin")


def test_preprocess_calvin_safe_exit_is_opt_in(monkeypatch):
    runner = _load_runner(monkeypatch)
    monkeypatch.delenv("UAMVLA_CALVIN_NATIVE_SAFE_EXIT", raising=False)
    monkeypatch.setattr(
        runner.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(AssertionError(code)),
    )

    runner._exit_without_native_teardown_if_requested()


def test_preprocess_calvin_safe_exit_uses_os_exit(monkeypatch):
    runner = _load_runner(monkeypatch)
    monkeypatch.setenv("UAMVLA_CALVIN_NATIVE_SAFE_EXIT", "1")
    monkeypatch.setattr(
        runner.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(SystemExit(code)),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner._exit_without_native_teardown_if_requested()

    assert exc_info.value.code == 0
