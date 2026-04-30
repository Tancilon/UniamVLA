"""pytest config for UniamVLA migration tests."""
import sys
from pathlib import Path
import pytest

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    """Register custom markers used by tests in this repo."""
    config.addinivalue_line("markers", "slow: integration tests that exercise full pipelines")
    config.addinivalue_line(
        "markers",
        "libero_env: tests that require the libero_env conda environment / LIBERO sim deps",
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_REAL_DATASET_DIR = ROOT / "datasets" / "uamvla_test" / "libero_spatial"


@pytest.fixture
def sample_dataset_dir():
    """Return path to the real UamVLA test dataset (libero_spatial)."""
    if not (_REAL_DATASET_DIR / "data.jsonl").exists():
        pytest.skip(
            f"data.jsonl not present at {_REAL_DATASET_DIR}; "
            f"run `python tools/preprocess/run_libero_preprocess.py` to generate."
        )
    return _REAL_DATASET_DIR
