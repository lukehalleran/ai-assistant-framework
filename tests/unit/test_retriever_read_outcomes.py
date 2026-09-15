"""memory/memory_retriever.py three read sites raise RetrievalError instead
of silently returning an empty list on failure (CGR-009 #123, #125, #126):
the reflection metadata fallback search, recent facts, and procedural
skills. Their consumers already flatten a typed list -- gatherer_knowledge's
`get_procedural_skills` does `return skills or []`, and
`get_reflections_hybrid` merges the fallback's results into a new list --
so a raise (not a typed OutcomeList) is the correct contract here.

Deployed-function tests: `MemoryRetriever.__new__` with the attributes each
method actually reads, MagicMock stores, no real embedder or cross-encoder.
One test drives the real `MemoryCoordinator.get_skills` passthrough and the
real `core.prompt.gatherer_knowledge.get_procedural_skills`. One drives the
real `ShutdownProcessor._gather_proposal_context`.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin
from memory.memory_coordinator import MemoryCoordinator
from memory.memory_retriever import MemoryRetriever, _metadata_fallback_search
from memory.shutdown_processor import ShutdownProcessor
from utils.retrieval_outcome import RetrievalError, outcome_status

MARKER = "F12C_R7Q2M"


def _retriever(chroma_store=None):
    r = MemoryRetriever.__new__(MemoryRetriever)
    r.chroma_store = chroma_store
    return r


# ---------------------------------------------------------------------------
# _metadata_fallback_search -- ANCHOR #123
# ---------------------------------------------------------------------------

class TestMetadataFallbackSearch:
    def test_query_raises_retrieval_error(self):
        coll = MagicMock()
        coll.query.side_effect = RuntimeError(f"boom {MARKER}")
        store = MagicMock()
        store.collections = {"reflections": coll}
        with pytest.raises(RetrievalError) as exc:
            _metadata_fallback_search(store, "quasar project details", set())
        assert exc.value.source == "reflection_metadata_fallback"
        assert exc.value.reason == "RuntimeError"
        assert MARKER not in str(exc.value)

    def test_no_query_returns_empty(self):
        store = MagicMock()
        assert _metadata_fallback_search(store, "", set()) == []

    def test_no_store_returns_empty(self):
        assert _metadata_fallback_search(None, "quasar project details", set()) == []

    def test_no_content_words_returns_empty(self):
        # Every word is <=3 chars after stripping punctuation, or a stopword.
        store = MagicMock()
        assert _metadata_fallback_search(store, "it is a", set()) == []
        store.collections.get.assert_not_called()

    def test_missing_collection_returns_empty(self):
        store = MagicMock()
        store.collections = {}
        assert _metadata_fallback_search(store, "quasar project details", set()) == []

    def test_healthy_query_returns_results(self):
        coll = MagicMock()
        coll.query.return_value = {
            "ids": [["r1"]],
            "documents": [["quasar project details discussed"]],
            "metadatas": [[{"topic": "quasar"}]],
            "distances": [[0.2]],
        }
        store = MagicMock()
        store.collections = {"reflections": coll}
        out = _metadata_fallback_search(store, "quasar project details", set())
        assert len(out) == 1
        assert out[0]["id"] == "r1"
        assert out[0]["content"] == "quasar project details discussed"


# ---------------------------------------------------------------------------
# get_recent_facts -- ANCHOR #125
# ---------------------------------------------------------------------------

class TestGetRecentFacts:
    def test_get_recent_raises_retrieval_error(self):
        store = MagicMock()
        store.get_recent.side_effect = RuntimeError(f"boom {MARKER}")
        r = _retriever(store)
        with pytest.raises(RetrievalError) as exc:
            asyncio.run(r.get_recent_facts(limit=5))
        assert exc.value.source == "recent_facts"
        assert exc.value.reason == "RuntimeError"
        assert MARKER not in str(exc.value)

    def test_empty_returns_empty_list_control(self):
        store = MagicMock()
        store.get_recent.return_value = []
        r = _retriever(store)
        assert asyncio.run(r.get_recent_facts(limit=5)) == []

    def test_healthy_returns_facts_control(self):
        store = MagicMock()
        store.get_recent.return_value = [{"content": "likes tea"}]
        r = _retriever(store)
        assert asyncio.run(r.get_recent_facts(limit=5)) == [{"content": "likes tea"}]


# ---------------------------------------------------------------------------
# get_skills -- ANCHOR #126
# ---------------------------------------------------------------------------

def _skills_store(count=1, query_raise=None, query_return=None):
    coll = MagicMock()
    coll.count.return_value = count
    store = MagicMock()
    store.collections = {"procedural_skills": coll}
    store.get_recent.return_value = []
    if query_raise is not None:
        store.query_collection.side_effect = query_raise
    else:
        store.query_collection.return_value = query_return or []
    return store


class TestGetSkills:
    def test_query_collection_raises_retrieval_error(self, monkeypatch):
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", True)
        store = _skills_store(count=1, query_raise=RuntimeError(f"boom {MARKER}"))
        r = _retriever(store)
        with pytest.raises(RetrievalError) as exc:
            asyncio.run(r.get_skills("quasar project", limit=5))
        assert exc.value.source == "procedural_skills"
        assert exc.value.reason == "RuntimeError"
        assert MARKER not in str(exc.value)

    def test_disabled_returns_empty_control(self, monkeypatch):
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", False)
        store = _skills_store(count=5)
        r = _retriever(store)
        assert asyncio.run(r.get_skills("quasar project", limit=5)) == []
        store.collections["procedural_skills"].count.assert_not_called()

    def test_empty_collection_returns_empty_control(self, monkeypatch):
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", True)
        store = _skills_store(count=0)
        r = _retriever(store)
        assert asyncio.run(r.get_skills("quasar project", limit=5)) == []

    def test_healthy_returns_skills_control(self, monkeypatch):
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", True)
        store = _skills_store(count=1, query_return=[
            {"id": "s1", "content": "chop onions", "metadata": {}},
        ])
        r = _retriever(store)
        out = asyncio.run(r.get_skills("quasar project", limit=5))
        assert len(out) == 1
        assert out[0]["id"] == "s1"


# ---------------------------------------------------------------------------
# Through the deployed MemoryCoordinator passthrough and
# core.prompt.gatherer_knowledge.get_procedural_skills
# ---------------------------------------------------------------------------

class _G(KnowledgeRetrievalMixin):
    """Bare mixin host -- only the attribute get_procedural_skills reads."""

    def __init__(self, memory_coordinator):
        self.memory_coordinator = memory_coordinator
        self.memory_id_map = {}


class TestThroughCoordinatorAndGatherer:
    @pytest.mark.asyncio
    async def test_skills_raise_propagates_as_failed_retrieval_error(self, monkeypatch):
        monkeypatch.setattr("config.app_config.PROCEDURAL_SKILLS_ENABLED", True)
        store = _skills_store(count=1, query_raise=RuntimeError(f"boom {MARKER}"))
        retriever = _retriever(store)

        # Real MemoryCoordinator.get_skills passthrough (no mock override).
        mc = MemoryCoordinator.__new__(MemoryCoordinator)
        mc._retriever = retriever

        g = _G(mc)
        result = await g.get_procedural_skills("quasar project", limit=5)
        status, reason = outcome_status(result)
        assert (status, reason) == ("failed", "RetrievalError")
        assert result == []
        assert MARKER not in reason


# ---------------------------------------------------------------------------
# Shutdown evidence -- the real _gather_proposal_context survives both
# raises via its own per-task catch (memory/shutdown_processor.py, not
# owned by this batch; read-only evidence that the raise is safe there).
# ---------------------------------------------------------------------------

class TestShutdownEvidence:
    @pytest.mark.asyncio
    async def test_gather_proposal_context_survives_skills_and_reflections_raising(self):
        mc = MagicMock()
        mc.get_memories = AsyncMock(return_value=[])
        mc.get_summaries_hybrid = MagicMock(return_value=[])
        mc.get_facts = AsyncMock(return_value=[])
        mc.get_skills = AsyncMock(
            side_effect=RetrievalError(source="procedural_skills", reason="RuntimeError")
        )
        mc.get_reflections_hybrid = AsyncMock(
            side_effect=RetrievalError(source="reflection_metadata_fallback", reason="RuntimeError")
        )

        sp = ShutdownProcessor(
            corpus_manager=MagicMock(),
            chroma_store=MagicMock(query_collection=MagicMock(return_value=[])),
            consolidator=MagicMock(),
            fact_extractor=MagicMock(),
            model_manager=MagicMock(),
            user_profile=None,
            storage=MagicMock(),
            session_start=datetime.now(),
            memory_coordinator=mc,
        )
        sess_items = [
            {"query": f"quasar project update {MARKER}", "response": "noted"},
        ] * 3
        result = await sp._gather_proposal_context(sess_items)
        assert isinstance(result, str)
        assert "Problem-Solving Patterns" not in result
        assert "Past Reflections" not in result
