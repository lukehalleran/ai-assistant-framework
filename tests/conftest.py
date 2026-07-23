"""
Root conftest.py — applies to all test sessions.

Caps torch/numpy thread pools so the full suite doesn't saturate all CPU cores
and freeze the machine. Also lowers process priority (nice) on Linux.
"""
import os

# Cap parallelism BEFORE any torch/numpy imports.
# Default: half the cores, minimum 2, so the system stays responsive.
_max_threads = str(max(2, os.cpu_count() // 2))

os.environ.setdefault("OMP_NUM_THREADS", _max_threads)
os.environ.setdefault("MKL_NUM_THREADS", _max_threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _max_threads)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Apply torch thread cap if already imported (some fixtures import early)
try:
    import torch
    torch.set_num_threads(int(_max_threads))
    torch.set_num_interop_threads(2)
except Exception:
    pass

# Lower process priority on Linux so desktop stays usable
try:
    os.nice(10)
except (OSError, AttributeError):
    pass  # Windows or permission denied


# ---------------------------------------------------------------------------
# Safety-test skip guard (2026-07-21)
#
# The tone-flatline incident survived for weeks because the tests that KNEW
# about it (test_tone_detection.py's short-message semantic cases) were skipped
# silently. This hook fails collection if any test in the crisis-safety scope
# (tone detection, escalation tracker, valence retrieval, and the anti-
# amplification regression tests) carries a skip/skipif/xfail marker whose
# reason does NOT contain an explicit "ALLOW_SKIP:" justification. Environment
# gates (e.g. embedder unavailable) are fine — they just have to say so.
# ---------------------------------------------------------------------------
import re as _re

import pytest as _pytest

_SAFETY_TEST_RE = _re.compile(
    r"(tone_detect|tone_detection|tone_execution|escalation|valence|safety_canary|"
    r"anti_amplification|golden_distress|length_invariance|tonelevel_from_string)"
)
_GUARDED_MARKERS = ("skip", "skipif", "xfail")


def _marker_reason(marker) -> str:
    reason = ""
    if getattr(marker, "kwargs", None):
        reason = marker.kwargs.get("reason", "") or ""
    if not reason and getattr(marker, "args", None):
        for a in marker.args:
            if isinstance(a, str):
                reason = a
                break
    return reason


def pytest_collection_modifyitems(config, items):
    violations = []
    for item in items:
        fspath = str(getattr(item, "fspath", "") or "")
        basename = fspath.replace("\\", "/").split("/")[-1]
        if not _SAFETY_TEST_RE.search(basename):
            continue
        for mname in _GUARDED_MARKERS:
            marker = item.get_closest_marker(mname)
            if marker is None:
                continue
            reason = _marker_reason(marker)
            if "ALLOW_SKIP" not in reason:
                violations.append(
                    f"  {item.nodeid} carries @pytest.mark.{mname} without an "
                    f"'ALLOW_SKIP:' justification (reason={reason!r})"
                )
    if violations:
        raise _pytest.UsageError(
            "Crisis-safety tests may not be skipped without explicit justification.\n"
            "Add 'ALLOW_SKIP: <why>' to the marker reason (env gates are fine), or "
            "remove the skip. The tone-flatline incident survived on a silent skip.\n"
            + "\n".join(sorted(set(violations)))
        )
