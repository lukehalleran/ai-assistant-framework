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


# ---------------------------------------------------------------------------
# Adaptive exemplar store sandbox (2026-08-02). Tone tests exercise messages
# that hit the keyword/arbiter confirmation hooks, which LEARN exemplars via
# utils.adaptive_exemplars — without this fixture every test run would write
# test phrases into the user's real data/adaptive_exemplars.json.
# ---------------------------------------------------------------------------
import pytest as _pytest_ae


@_pytest_ae.fixture(autouse=True)
def _sandbox_adaptive_exemplars(tmp_path, monkeypatch):
    import utils.adaptive_exemplars as _ae
    monkeypatch.setattr(_ae, "_STORE_PATH", str(tmp_path / "adaptive_exemplars.json"))
    monkeypatch.setattr(_ae, "_store", None)  # fresh singleton per test
    # Per-text embedding caches (2026-08-21, encode_texts_cached): vectors
    # from one test's fake embedder must not leak into the next test.
    import sys as _sys
    for _mod, _attr in (
        ("utils.tone_detector", "_exemplar_text_emb_cache"),
        ("utils.need_detector", "_need_text_emb_cache"),
        ("core.intent_classifier", "_intent_text_emb_cache"),
        ("utils.web_search_trigger", "_anchor_text_emb_cache"),
    ):
        _m = _sys.modules.get(_mod)  # only clear if already imported
        if _m is not None:
            try:
                getattr(_m, _attr).clear()
            except Exception:
                pass
    # Embedder singletons + version-keyed prototype caches must also reset:
    # a test passing a MagicMock model_manager permanently captured the mock
    # in tone_detector._embedder_cache, poisoning every later real-path test
    # in the process (order-dependent — full-suite alphabetical order never
    # hit it; 2026-08-21). And with the store sandboxed per test, store
    # versions always restart at 0, so version-keyed prototype caches from
    # one test read as fresh in the next.
    for _mod, _attrs in (
        ("utils.tone_detector", ("_embedder_cache", "_exemplar_embeddings_cache")),
        ("utils.need_detector", ("_embedder_cache", "_need_exemplar_embeddings_cache")),
        ("core.intent_classifier", ("_intent_prototype_cache",)),
        ("utils.web_search_trigger",
         ("_search_anchor_embs", "_no_search_anchor_embs", "_search_anchor_version")),
    ):
        _m = _sys.modules.get(_mod)
        if _m is not None:
            for _a in _attrs:
                try:
                    setattr(_m, _a, None)
                except Exception:
                    pass
    yield
    _ae._store = None


# Same sandbox for the learned-relation store (2026-08-05): extractor tests
# exercise triples with invented relations, which would otherwise be recorded
# into the user's real data/learned_relations.json.
@_pytest_ae.fixture(autouse=True)
def _sandbox_learned_relations(tmp_path, monkeypatch):
    import memory.learned_relations as _lr
    monkeypatch.setattr(_lr, "_STORE_PATH", str(tmp_path / "learned_relations.json"))
    monkeypatch.setattr(_lr, "_store", None)  # fresh singleton per test
    yield
    _lr._store = None
