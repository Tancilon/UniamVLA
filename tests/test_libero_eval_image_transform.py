from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


def _import_eval_libero(monkeypatch):
    imageio_mod = types.ModuleType("imageio")
    imageio_mod.mimwrite = lambda *args, **kwargs: None
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda iterable, *args, **kwargs: iterable
    tyro_mod = types.ModuleType("tyro")
    tyro_mod.cli = lambda fn: fn
    libero_pkg = types.ModuleType("libero")
    libero_mod = types.ModuleType("libero.libero")
    envs_mod = types.ModuleType("libero.libero.envs")
    libero_mod.benchmark = object()
    libero_mod.get_libero_path = lambda _name: ""
    envs_mod.OffScreenRenderEnv = object
    monkeypatch.setitem(sys.modules, "imageio", imageio_mod)
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)
    monkeypatch.setitem(sys.modules, "tyro", tyro_mod)
    monkeypatch.setitem(sys.modules, "libero", libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_mod)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs_mod)
    return importlib.import_module("examples.LIBERO.eval_files.eval_libero")


def test_preprocess_libero_image_supports_vertical_and_rotate180(monkeypatch):
    eval_libero = _import_eval_libero(monkeypatch)
    image = np.arange(2 * 3 * 1, dtype=np.uint8).reshape(2, 3, 1)

    vertical = eval_libero._preprocess_libero_image(image, "vertical")
    rotate180 = eval_libero._preprocess_libero_image(image, "rotate180")

    np.testing.assert_array_equal(vertical, image[::-1])
    np.testing.assert_array_equal(rotate180, image[::-1, ::-1])
    assert vertical.flags.c_contiguous
    assert rotate180.flags.c_contiguous


def test_preprocess_libero_image_rejects_unknown_transform(monkeypatch):
    eval_libero = _import_eval_libero(monkeypatch)
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="image_transform"):
        eval_libero._preprocess_libero_image(image, "diagonal")
