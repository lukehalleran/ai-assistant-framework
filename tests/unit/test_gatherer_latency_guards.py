"""Latency guards in the context gatherer (2026-07-15 speedup batch).

1. Wiki semantic in-flight guard: the USB-backed FAISS search runs on a
   dedicated executor and a non-blocking semaphore — a search still stuck
   from a previous turn (asyncio.wait_for cannot kill blocking I/O) makes
   the current turn SKIP wiki chunks instead of queuing behind the zombie.
2. user_uploads existence gate: the ~0.9s hybrid retrieval is skipped
   entirely when a cheap metadata probe shows no uploads exist.
"""
import threading

import pytest
from unittest.mock import Mock, AsyncMock

import core.prompt.gatherer_knowledge as gk


@pytest.fixture(autouse=True)
def _fresh_inflight_semaphore(monkeypatch):
    """Isolate the module-level in-flight semaphore per test.

    Other test files (e.g. test_prompt_internal_methods) exercise real prompt
    builders whose wiki search can block past SEM_TIMEOUT_S in the test env —
    the worker thread then holds a slot for the REST OF THE PYTEST PROCESS
    (that's the by-design production behavior). The all-slots-free assertions
    here were failing only in batch runs because of that leaked slot
    (pre-existing; confirmed on a clean HEAD worktree 2026-08-05).
    """
    monkeypatch.setattr(
        gk, "_WIKI_SEM_INFLIGHT", threading.Semaphore(gk._WIKI_SEM_MAX_CONCURRENT)
    )


def _bare_gatherer():
    """KnowledgeRetrievalMixin instance without running any __init__."""
    class G(gk.KnowledgeRetrievalMixin):
        pass
    return G.__new__(G)


# ===========================================================================
# Wiki semantic in-flight guard
# ===========================================================================

class TestWikiInflightGuard:

    @pytest.mark.asyncio
    async def test_skips_when_all_slots_occupied(self, monkeypatch):
        # Exhaust the in-flight semaphore (simulates zombie searches from
        # previous turns still blocked on USB I/O).
        held = 0
        while gk._WIKI_SEM_INFLIGHT.acquire(blocking=False):
            held += 1
        try:
            called = {"n": 0}

            def fake_search(query, k):
                called["n"] += 1
                return []

            monkeypatch.setattr(gk, "semantic_search_with_neighbors", fake_search)
            g = _bare_gatherer()
            out = await g._get_semantic_chunks("some query", max_results=5)
            assert out == []
            assert called["n"] == 0  # search never even submitted
        finally:
            for _ in range(held):
                gk._WIKI_SEM_INFLIGHT.release()

    @pytest.mark.asyncio
    async def test_slot_released_after_completion(self, monkeypatch):
        monkeypatch.setattr(gk, "semantic_search_with_neighbors", lambda q, k: [])
        g = _bare_gatherer()
        out = await g._get_semantic_chunks("some query", max_results=5)
        assert out == []

        # All slots must be free again — the worker releases in its finally.
        acquired = 0
        while gk._WIKI_SEM_INFLIGHT.acquire(blocking=False):
            acquired += 1
        for _ in range(acquired):
            gk._WIKI_SEM_INFLIGHT.release()
        assert acquired == gk._WIKI_SEM_MAX_CONCURRENT

    @pytest.mark.asyncio
    async def test_slot_released_after_search_error(self, monkeypatch):
        def _boom(q, k):
            raise RuntimeError("faiss unhappy")

        monkeypatch.setattr(gk, "semantic_search_with_neighbors", _boom)
        g = _bare_gatherer()
        out = await g._get_semantic_chunks("some query", max_results=5)
        assert out == []

        acquired = 0
        while gk._WIKI_SEM_INFLIGHT.acquire(blocking=False):
            acquired += 1
        for _ in range(acquired):
            gk._WIKI_SEM_INFLIGHT.release()
        assert acquired == gk._WIKI_SEM_MAX_CONCURRENT


# ===========================================================================
# user_uploads existence gate
# ===========================================================================

def _gatherer_with_store(ids):
    g = _bare_gatherer()
    manager = Mock()
    coll = Mock()
    coll.get = Mock(return_value={"ids": ids})
    manager.chroma_store._get_collection = Mock(return_value=coll)
    g.reference_docs_manager = manager
    g.memory_id_map = {}
    return g, manager, coll


class TestUploadsExistenceGate:

    @pytest.mark.asyncio
    async def test_no_uploads_skips_retrieval(self):
        g, manager, coll = _gatherer_with_store(ids=[])
        manager.get_documents = AsyncMock()
        out = await g.get_user_uploads("query")
        assert out == []
        manager.get_documents.assert_not_called()
        coll.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_uploads_exist_runs_retrieval(self):
        g, manager, coll = _gatherer_with_store(ids=["u1"])
        manager.get_documents = AsyncMock(return_value=[])
        await g.get_user_uploads("query")
        manager.get_documents.assert_called_once()

    def test_negative_result_cached_within_ttl(self):
        g, manager, coll = _gatherer_with_store(ids=[])
        assert g._any_user_uploads_exist() is False
        assert g._any_user_uploads_exist() is False
        coll.get.assert_called_once()  # second answer came from the cache

    def test_positive_result_cached_for_session(self):
        g, manager, coll = _gatherer_with_store(ids=["u1"])
        assert g._any_user_uploads_exist() is True
        assert g._any_user_uploads_exist() is True
        coll.get.assert_called_once()

    def test_probe_failure_fails_open(self):
        g, manager, coll = _gatherer_with_store(ids=[])
        coll.get.side_effect = RuntimeError("chroma down")
        assert g._any_user_uploads_exist() is True
