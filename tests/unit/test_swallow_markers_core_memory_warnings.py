"""Wave 3a lane SWA (2026-10-27, class: BC-69, BC-70): four swallow handlers whose failure
means real lost work now log a greppable WARNING (they were debug), without changing
control flow.  Each test injects the failure into the deployed function and asserts the
WARNING tag and that the caller still gets its old fallback value.
"""
import logging
from types import SimpleNamespace

import pytest


def test_calendar_cache_clear_failure_warns(monkeypatch, caplog):
    from core.actions import google_calendar_modify as mod

    def boom():
        raise RuntimeError("cache backend down")

    monkeypatch.setattr(mod.google_calendar, "clear_cache", boom)
    with caplog.at_level(logging.WARNING, logger="actions_calendar_modify"):
        assert mod._invalidate_read_cache("update") is None
    recs = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert recs and "[CalendarModify]" in recs[0].getMessage()
    assert "RuntimeError" in recs[0].getMessage()


def test_expander_clear_cache_failure_warns(caplog):
    from memory import memory_expander as mod

    class Bad:
        def clear_cache(self):
            raise ValueError("x")

    bad = Bad()
    mod.register_expander(bad) if hasattr(mod, "register_expander") else mod._REGISTERED_EXPANDERS.add(bad)
    try:
        with caplog.at_level(logging.WARNING, logger=mod.logger.name):
            mod.notify_chroma_mutation("doc-1")
    finally:
        mod._REGISTERED_EXPANDERS.discard(bad)
    assert any(r.levelno == logging.WARNING and "clear_cache failed" in r.getMessage()
               for r in caplog.records)


def test_proposal_store_bad_metadata_warns_and_is_skipped(monkeypatch, caplog):
    from memory import proposal_store as mod

    coll = SimpleNamespace(count=lambda: 1)
    store = SimpleNamespace(
        collections={mod.COLLECTION_NAME: coll},
        query_collection=lambda *a, **k: [{"metadata": {"proposal_id": "p1"}}],
    )
    ps = mod.ProposalStore(chroma_store=store)
    monkeypatch.setattr(mod.CodeProposal, "from_metadata",
                        classmethod(lambda cls, meta: (_ for _ in ()).throw(KeyError("title"))))
    with caplog.at_level(logging.WARNING):
        assert ps.query_proposals("anything") == []
    assert any(r.levelno == logging.WARNING and "[ProposalStore] Failed to deserialize" in r.getMessage()
               and "KeyError" in r.getMessage() for r in caplog.records)


def test_schedule_supersession_failure_warns_and_returns_none(caplog):
    from memory.fact_verification import FactVerifier

    class BadStore:
        collections = {"facts": SimpleNamespace(count=lambda: 1)}

        def query_collection(self, *a, **k):
            raise RuntimeError("chroma down")

    v = FactVerifier(BadStore())
    with caplog.at_level(logging.WARNING):
        out = v._check_schedule_supersession("work_schedule", "mon tue 9-5", "work schedule mon tue")
    assert out is None
    assert any(r.levelno == logging.WARNING and "[FactVerifier] Schedule supersession" in r.getMessage()
               and "RuntimeError" in r.getMessage() for r in caplog.records)
