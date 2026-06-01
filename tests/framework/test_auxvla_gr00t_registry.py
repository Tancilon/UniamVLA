from starVLA.model.framework.base_framework import _auto_import_framework_modules
from starVLA.model.tools import FRAMEWORK_REGISTRY


def test_auxvla_gr00t_auto_import_registers_framework():
    _auto_import_framework_modules()
    assert "AuxVLAGR00T" in FRAMEWORK_REGISTRY._registry
