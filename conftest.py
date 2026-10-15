"""
Root conftest.py — adds the project root to sys.path so that
`python -m pytest` resolves internal modules (utils, core, memory, etc.)
without requiring the venv to have the project installed as a package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def pytest_addoption(parser):
    """Benchmark CSV target (2026-10-15, BC-37). Must live in the ROOT conftest:
    pytest only honours pytest_addoption there (or in a plugin)."""
    parser.addoption(
        "--benchmark-csv",
        action="store",
        default=None,
        help="Write the benchmark per-case CSV to PATH (e.g. data/benchmark_per_case.csv "
             "to refresh the tracked file). Default: untracked logs/benchmark_per_case.csv.",
    )
