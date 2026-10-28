"""Shutdown fact/graph/thread swallows must be visible (WARNING), not DEBUG/silent.

Incident: 2026-10-27 wave-3b swallow audit. The ``# degrades:`` markers in
memory/shutdown_processor.py named real losses (superseded fact not flagged,
fact stored without verification, fact missing its graph edge, thread cap not
enforced) but each was logged at DEBUG or ``pass``, so nobody saw them.
Classes: BC-69 (silent swallow), BC-70 (diagnostic invisible).
Drives the DEPLOYED ShutdownProcessor methods with fake stores.
"""
import json
import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory.fact_verification import ConflictCandidate, FactVerdict, VerificationResult
from memory.shutdown_processor import ShutdownProcessor

QUERIES = [
    "I really like green tea in the evening.",
    "    I really like green tea\n    in the evening.",  # wrapped / indented form
]


def _flag_result(doc_id="doc-123"):
    return VerificationResult(
        verdict=FactVerdict.STORE_AND_FLAG,
        conflicting_candidates=[ConflictCandidate(doc_id=doc_id)],
    )


def _processor(**kw):
    chroma = kw.pop("chroma", MagicMock())
    storage = kw.pop("storage", MagicMock())
    mm = kw.pop("mm", MagicMock())
    base = dict(
        corpus_manager=MagicMock(), chroma_store=chroma, consolidator=MagicMock(),
        fact_extractor=MagicMock(), model_manager=mm, user_profile=MagicMock(),
        storage=storage, session_start=datetime(2026, 10, 27, 0, 0),
    )
    base.update(kw)
    return ShutdownProcessor(**base)


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("query", QUERIES)
class TestPerTurnFactPath:
    def _proc(self, verifier):
        chroma = MagicMock()
        chroma.add_fact.return_value = "fact-1"
        chroma.update_metadata.side_effect = RuntimeError("chroma down")
        storage = MagicMock()
        storage.fact_verifier = verifier
        fx = MagicMock()
        fx.extract_facts = AsyncMock(
            return_value=[SimpleNamespace(content="user | likes | green tea", metadata={})])
        return _processor(chroma=chroma, storage=storage, fact_extractor=fx), chroma

    @pytest.mark.asyncio
    async def test_failed_supersede_flag_warns_with_doc_id_and_fact_still_stored(self, query, caplog):
        verifier = MagicMock()
        verifier.verify = AsyncMock(return_value=_flag_result("doc-123"))
        proc, chroma = self._proc(verifier)
        with caplog.at_level(logging.DEBUG):
            await proc._extract_session_facts([{"query": query, "response": "", "timestamp": "2026-10-27T09:00:00"}])
        assert chroma.add_fact.call_count == 1
        msgs = [r.getMessage() for r in _warnings(caplog)]
        assert any("doc-123" in m and "RuntimeError" in m for m in msgs), msgs

    @pytest.mark.asyncio
    async def test_verifier_failure_warns_and_fact_still_stored(self, query, caplog):
        verifier = MagicMock()
        verifier.verify = AsyncMock(side_effect=ValueError("verifier exploded"))
        proc, chroma = self._proc(verifier)
        with caplog.at_level(logging.DEBUG):
            await proc._extract_session_facts([{"query": query, "response": "", "timestamp": "2026-10-27T09:00:00"}])
        assert chroma.add_fact.call_count == 1
        msgs = [r.getMessage() for r in _warnings(caplog)]
        assert any("verifier failed" in m.lower() and "ValueError" in m for m in msgs), msgs


@pytest.mark.parametrize("query", [
    "I always go to bed around 11",
    "    I always go to bed\n    around 11",
])
class TestLlmFactPath:
    def _proc(self, verify_batch_result=None, ingest_raises=False, update_raises=False):
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value=json.dumps([
            {"subject": "user", "relation": "bedtime_habit",
             "object": "goes to bed around 11", "confidence": 0.9},
        ]))
        chroma = MagicMock()
        chroma.add_fact.return_value = "fact-77"
        if update_raises:
            chroma.update_metadata.side_effect = RuntimeError("chroma down")
        profile = MagicMock()
        profile.profile = {"categories": {}}
        storage = MagicMock()
        verifier = MagicMock()
        verifier.verify_batch = AsyncMock(return_value=verify_batch_result or [])
        storage.fact_verifier = verifier if verify_batch_result else None
        storage.graph_memory = MagicMock()
        storage.entity_resolver = MagicMock()
        if ingest_raises:
            storage._ingest_fact_to_graph.side_effect = RuntimeError("graph down")
        return _processor(chroma=chroma, storage=storage, mm=mm, user_profile=profile), chroma

    @pytest.mark.asyncio
    async def test_failed_supersede_flag_warns_counts_and_fact_stored(self, query, caplog):
        proc, chroma = self._proc(verify_batch_result=[_flag_result("doc-555")], update_raises=True)
        convs = [{"query": query, "response": "Good.", "timestamp": "2026-10-27T09:00:00"}]
        with caplog.at_level(logging.DEBUG), patch("memory.memory_storage._get_graph_enabled", return_value=False):
            await proc._extract_llm_facts(convs)
        assert chroma.add_fact.call_count == 1
        msgs = [r.getMessage() for r in _warnings(caplog)]
        assert any("doc-555" in m and "RuntimeError" in m for m in msgs), msgs
        summary = [r.getMessage() for r in caplog.records if "[LLM Facts] kept=" in r.getMessage()]
        assert summary and "flag_failures=1" in summary[0], summary

    @pytest.mark.asyncio
    async def test_summary_line_unchanged_when_nothing_failed(self, query, caplog):
        proc, chroma = self._proc()
        convs = [{"query": query, "response": "Good.", "timestamp": "2026-10-27T09:00:00"}]
        with caplog.at_level(logging.DEBUG), patch("memory.memory_storage._get_graph_enabled", return_value=False):
            await proc._extract_llm_facts(convs)
        summary = [r.getMessage() for r in caplog.records if "[LLM Facts] kept=" in r.getMessage()]
        assert summary and "flag_failures" not in summary[0], summary

    @pytest.mark.asyncio
    async def test_graph_ingestion_failure_warns_with_relation_and_fact_id_not_object(self, query, caplog):
        proc, chroma = self._proc(ingest_raises=True)
        convs = [{"query": query, "response": "Good.", "timestamp": "2026-10-27T09:00:00"}]
        with caplog.at_level(logging.DEBUG), patch("memory.memory_storage._get_graph_enabled", return_value=True):
            await proc._extract_llm_facts(convs)
        assert chroma.add_fact.call_count == 1
        msgs = [r.getMessage() for r in _warnings(caplog)]
        hit = [m for m in msgs if "Graph ingestion failed" in m]
        assert hit, msgs
        assert "bedtime_habit" in hit[0] and "fact-77" in hit[0]
        assert "goes to bed" not in hit[0]


@pytest.mark.asyncio
async def test_thread_cap_failure_warns(caplog, monkeypatch):
    monkeypatch.setattr("config.app_config.THREAD_SURFACING_ENABLED", True)
    store = MagicMock()
    store.list_open_threads.return_value = []
    store.enforce_cap.side_effect = RuntimeError("cap boom")
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value="[]")
    proc = _processor(mm=mm, thread_store=store)
    convs = [
        {"query": "Planning the week", "response": "ok", "timestamp": "2026-10-27T09:00:00"},
        {"query": "Also the dentist", "response": "ok", "timestamp": "2026-10-27T09:05:00"},
    ]
    with caplog.at_level(logging.DEBUG):
        await proc._process_open_threads(convs)
    store.enforce_cap.assert_called_once()
    msgs = [r.getMessage() for r in _warnings(caplog)]
    assert any("Thread cap enforcement failed" in m and "RuntimeError" in m for m in msgs), msgs
