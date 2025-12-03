"""
Project sitecustomize: diagnostics-only outside tests.

Important: pytest-cov uses a sitecustomize shim to enable subprocess coverage.
If we run our own scan here during tests or coverage runs, it can interfere.
So we skip diagnostics when common test/coverage env vars are detected.
"""
from __future__ import annotations

import os
import sys


def _should_skip() -> bool:
    # Signals that pytest/coverage is active; avoid running diagnostics
    markers = (
        "PYTEST_CURRENT_TEST",
        "PYTEST_ADDOPTS",
        "COVERAGE_PROCESS_START",
        "PYTEST_XDIST_WORKER",
    )
    return any(os.getenv(k) for k in markers) or os.getenv("DISABLE_PROJECT_SITECUSTOMIZE") == "1"


def _run_diagnostics() -> None:
    try:
        from scripts.topic_scan import main as _scan_main
    except Exception as e:
        print(f"[sitecustomize] Import error: {e}")
        return

    try:
        _scan_main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[sitecustomize] Runtime error: {e}")


if not _should_skip():
    _run_diagnostics()
    # Exit after diagnostics only in non-test contexts
    sys.exit(0)
