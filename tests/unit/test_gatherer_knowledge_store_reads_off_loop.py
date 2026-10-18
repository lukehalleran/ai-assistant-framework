"""Knowledge-gatherer store reads run off the event loop.

2026-10-17, class BC-41 (blocking call on the event loop).

Live loop-stall records: `query_collection` for daemon self-notes (5178 ms,
2x), the thread store's `list_all` via get_unresolved_threads (2242 ms), and
ObsidianManager._keyword_search via get_notes (403 ms).  Each test drives THE
deployed async function; only the sync store seam is replaced by a recorder of
`threading.get_ident()`.  Item 2 also pins that list_open_threads pushes the
status filter into Chroma (`where`) with unchanged results.
"""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin
from knowledge.obsidian_manager import ObsidianManager


class _G(KnowledgeRetrievalMixin):
    def __init__(self):
        self.memory_coordinator = None
        self.memory_id_map = {}
        self._chroma_store = None


@pytest.mark.asyncio
async def test_self_notes_query_runs_off_loop():
    seen = {}

    def fake_query(name, query_text=None, n_results=3, **kw):
        seen["t"] = threading.get_ident()
        return [{"content": "Remember the quasar deadline.", "metadata": {}}]

    g = _G()
    g._chroma_store = SimpleNamespace(query_collection=fake_query)
    loop_thread = threading.get_ident()
    out = await g.get_daemon_self_notes("quasar", limit=3)
    assert out
    assert seen["t"] != loop_thread


@pytest.mark.asyncio
async def test_unresolved_threads_read_runs_off_loop(monkeypatch):
    monkeypatch.setattr("config.app_config.THREAD_SURFACING_ENABLED", True)
    seen = {}

    def fake_get(max_results=3):
        seen["t"] = threading.get_ident()
        return [{"topic": "quasar review"}]

    g = _G()
    g.memory_coordinator = SimpleNamespace(get_unresolved_threads=fake_get)
    loop_thread = threading.get_ident()
    out = await g.get_unresolved_threads(max_results=3)
    assert list(out) == [{"topic": "quasar review"}]
    assert seen["t"] != loop_thread


@pytest.mark.asyncio
async def test_obsidian_keyword_search_runs_off_loop():
    seen = {}
    mgr = ObsidianManager(chroma_store=MagicMock(), vault_path="/nonexistent")
    mgr.chroma_store.query_collection.return_value = []

    def fake_kw(query, limit=10):
        seen["t"] = threading.get_ident()
        return []

    mgr._keyword_search = fake_kw
    loop_thread = threading.get_ident()
    await mgr.get_notes("quasar planning", limit=6)
    assert seen["t"] != loop_thread


def test_list_open_threads_pushes_status_filter_into_chroma():
    from memory.thread_store import ThreadStore, COLLECTION_NAME
    from memory.thread_models import OpenThread, ThreadType

    open_t = OpenThread(topic="quasar review", summary="s", thread_type=ThreadType.DEADLINE)
    meta = open_t.to_metadata() if hasattr(open_t, "to_metadata") else None
    if meta is None:
        pytest.skip("OpenThread has no to_metadata")
    calls = []

    class Chroma:
        collections = {COLLECTION_NAME: object()}

        def list_all(self, name, where=None):
            calls.append((name, where))
            return [{"id": "1", "content": "x", "metadata": meta}]

    store = ThreadStore.__new__(ThreadStore)
    store.chroma_store = Chroma()
    store._ensure_collection = lambda: True
    threads = store.list_open_threads()
    assert calls == [(COLLECTION_NAME, {"status": "open"})]
    assert [t.topic for t in threads] == ["quasar review"]


def test_store_list_all_forwards_where_to_chroma_get():
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    store = MultiCollectionChromaStore.__new__(MultiCollectionChromaStore)
    store.collections = {"threads": object()}
    coll = MagicMock()
    coll.get.return_value = {"ids": ["1"], "documents": ["d"], "metadatas": [{"status": "open"}]}
    store._get_collection = lambda name: coll
    out = store.list_all("threads", where={"status": "open"})
    assert coll.get.call_args.kwargs["where"] == {"status": "open"}
    assert out[0]["metadata"] == {"status": "open"}
    coll.get.reset_mock()
    store.list_all("threads")
    assert "where" not in coll.get.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "what's in my homework document",
    "    what's in my\n    homework document\n",
])
async def test_upload_roster_fetch_runs_off_loop(query):
    """2026-10-18, BC-41: _fetch_upload_roster (chroma .get) blocked the loop 268 ms live."""
    from unittest.mock import AsyncMock
    from tests.unit.test_gatherer_outcomes_notes_docs_uploads import _bound

    seen = {}

    def fake_roster():
        seen["t"] = threading.get_ident()
        return [{"title": "syllabus.pdf", "date": "2026-10-17"}]

    mgr = MagicMock()
    mgr.get_documents = AsyncMock(return_value=[])
    g, method = _bound("user_uploads", manager=mgr)
    g._fetch_upload_roster = fake_roster
    loop_thread = threading.get_ident()
    result = await method(query, 10)
    assert seen["t"] != loop_thread
    assert result[0]["metadata"]["type"] == "upload_roster"


@pytest.mark.asyncio
async def test_upload_existence_probe_runs_off_loop():
    """2026-10-18, BC-41: the cold _any_user_uploads_exist probe (chroma .get) waited
    2.3 s on the loop behind another thread's Chroma read on the first turn after a restart."""
    from unittest.mock import AsyncMock
    from tests.unit.test_gatherer_outcomes_notes_docs_uploads import _bound

    seen = {}

    def fake_probe():
        seen["t"] = threading.get_ident()
        return False  # no uploads -> the method returns early, nothing else runs

    mgr = MagicMock()
    mgr.get_documents = AsyncMock(return_value=[])
    g, method = _bound("user_uploads", manager=mgr)
    g._any_user_uploads_exist = fake_probe
    loop_thread = threading.get_ident()
    result = await method("what's in my homework document", 10)
    assert seen["t"] != loop_thread
    assert list(result) == []
