"""F11a-1: thread store reads report failure instead of an empty list; a
failed thread write raises; the shutdown thread pass skips extraction (no
duplicate re-store) after a failed open-threads read.

ANCHORS: CGR-20260913-009 #121 (MemoryCoordinator.get_unresolved_threads),
#147 (ThreadStore.list_open_threads), #148 (ThreadStore.query_threads, no
production caller); CGR-20260913-010 #146 (ThreadStore.store_thread).

Drives the DEPLOYED ThreadStore, MemoryCoordinator.get_unresolved_threads,
the gatherer's get_unresolved_threads (bound to the real coordinator
method) and ShutdownProcessor._process_open_threads. Fakes only: a LOCAL
copy of test_thread_store.py's MockChromaStore/MockCollection shape (that
file's own fixture is untouched except the one FIXTURE RULE repair), plus
MagicMock/SimpleNamespace hosts and a class-level monkeypatch of
ThreadExtractor's two LLM-calling methods -- no real ChromaDB, embedder,
model or data path anywhere.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.thread_models import OpenThread, ThreadType, ThreadStatus
from memory.thread_store import ThreadStore, COLLECTION_NAME
from memory.memory_coordinator import MemoryCoordinator
from memory.shutdown_processor import ShutdownProcessor
from memory.thread_extractor import ThreadExtractor
from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin
from utils.retrieval_outcome import RetrievalError, StoreWriteError, OutcomeList, outcome_status

MARKER = "F11A1QZ7"


# --- Local fakes (FIXTURE RULE: a local copy, not an import, of
# test_thread_store.py's MockChromaStore/MockCollection shape) ---

class MockCollection:
    def __init__(self, items=None):
        self._items = items or []

    def count(self):
        return len(self._items)


class MockChromaStore:
    """Local copy of the MultiCollectionChromaStore subset ThreadStore uses."""

    def __init__(self, with_collection=True, items=None):
        self._items = items or []
        self._mock_collection = MockCollection(self._items) if with_collection else None
        self.collections = {COLLECTION_NAME: self._mock_collection} if with_collection else {}
        self._id_counter = 0

    def add_to_collection(self, name, text, metadata):
        self._id_counter += 1
        doc_id = f"doc_{self._id_counter}"
        item = {"id": doc_id, "content": text, "metadata": metadata}
        self._items.append(item)
        if self._mock_collection:
            self._mock_collection._items = self._items
        return doc_id

    def list_all(self, name):
        return list(self._items)

    def query_collection(self, name, query_text, n_results):
        return list(self._items[:n_results])

    def create_collection(self, name):
        coll = MockCollection(self._items)
        self.collections[name] = coll
        self._mock_collection = coll


class _CollectionsMissingOnGet(dict):
    """``in``/``[]`` behave like a normal dict (so ``_ensure_collection``
    sees the collection as present) but ``.get()`` always returns the
    default -- reaches ``query_threads``'s ``coll is None`` branch. Not a
    realistic Chroma shape; a direct constructor for that one line."""

    def get(self, key, default=None):
        return default


def _make_thread(topic="Test topic", thread_type=ThreadType.UNFINISHED,
                  status=ThreadStatus.OPEN, thread_id=None, urgency=0.5):
    kwargs = dict(topic=topic, thread_type=thread_type, urgency=urgency, status=status)
    if thread_id is not None:
        kwargs["thread_id"] = thread_id
    return OpenThread(**kwargs)


# === ThreadStore.store_thread (#146) ===

class TestStoreThreadOutcomes:
    def test_add_to_collection_raises_store_write_error(self):
        """#146: write raises StoreWriteError instead of returning None."""
        chroma = MockChromaStore()
        chroma.add_to_collection = MagicMock(side_effect=RuntimeError(f"boom {MARKER}"))
        store = ThreadStore(chroma_store=chroma)
        with pytest.raises(StoreWriteError) as exc_info:
            store.store_thread(_make_thread())
        err = exc_info.value
        assert err.source == "thread_store"
        assert err.reason == "RuntimeError"
        assert MARKER not in str(err) and MARKER not in err.reason

    def test_control_chroma_store_none_returns_none(self):
        """Control: deliberate skip still returns None."""
        store = ThreadStore(chroma_store=None)
        assert store.store_thread(_make_thread()) is None


# === ThreadStore.list_open_threads (#147) ===

class TestListOpenThreadsOutcomes:
    def test_list_all_raises_retrieval_error(self):
        """#147: read raises RetrievalError instead of returning []."""
        chroma = MockChromaStore()
        chroma.list_all = MagicMock(side_effect=RuntimeError(f"boom {MARKER}"))
        store = ThreadStore(chroma_store=chroma)
        with pytest.raises(RetrievalError) as exc_info:
            store.list_open_threads()
        err = exc_info.value
        assert err.source == "thread_store"
        assert err.reason == "list_open:RuntimeError"
        assert MARKER not in str(err)

    def test_control_empty_collection_returns_empty_list(self):
        """Control: a genuinely empty collection still returns []."""
        store = ThreadStore(chroma_store=MockChromaStore())
        assert store.list_open_threads() == []

    def test_control_malformed_item_is_skipped(self):
        """Control: the per-item parse skip is unchanged (kept, not failed)."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)
        store.store_thread(_make_thread(topic="Good", thread_id="good1"))
        chroma._items.append({
            "id": "bad1", "content": "junk",
            "metadata": {"thread_id": "bad1", "topic": "Bad", "status": "open",
                         "thread_type": "not_a_real_type"},
        })
        results = store.list_open_threads()
        assert len(results) == 1 and results[0].thread_id == "good1"


class TestGetTopThreadsPropagation:
    def test_raising_list_open_threads_propagates_unchanged(self):
        """Evidence: no try around list_open_threads, so a RetrievalError
        passes through unchanged (sibling, no code change)."""
        chroma = MockChromaStore()
        chroma.list_all = MagicMock(side_effect=RuntimeError(f"boom {MARKER}"))
        store = ThreadStore(chroma_store=chroma)
        with pytest.raises(RetrievalError) as exc_info:
            store.get_top_threads()
        assert exc_info.value.reason == "list_open:RuntimeError"


# === ThreadStore.query_threads (#148, no production caller) ===

class TestQueryThreadsOutcomes:
    def test_query_collection_raises_failed_outcome_list(self):
        """#148: a failing query returns OutcomeList.failed(...), == []."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)
        store.store_thread(_make_thread(topic="Something", thread_id="t1"))
        chroma.query_collection = MagicMock(side_effect=RuntimeError(f"boom {MARKER}"))
        result = store.query_threads("something")
        assert result == [] and isinstance(result, OutcomeList)
        assert outcome_status(result) == ("failed", "RuntimeError")
        assert MARKER not in result.reason

    def test_collection_missing_returns_unavailable(self):
        """coll is None (configured but not fetchable) -> unavailable, not []."""
        chroma = MockChromaStore(with_collection=True)
        chroma.collections = _CollectionsMissingOnGet({COLLECTION_NAME: chroma._mock_collection})
        store = ThreadStore(chroma_store=chroma)
        result = store.query_threads("anything")
        assert result == []
        assert outcome_status(result) == ("unavailable", "collection_missing")

    def test_control_empty_collection_count_zero_returns_empty_list(self):
        """Control: count() == 0 is unchanged, still plain []."""
        store = ThreadStore(chroma_store=MockChromaStore())
        assert store.query_threads("anything") == []

    def test_control_healthy_query_returns_threads(self):
        """Control: a healthy query still returns matching threads."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)
        store.store_thread(_make_thread(topic="Python project deadline", thread_id="t1"))
        results = store.query_threads("Python project", n_results=5)
        assert len(results) == 1 and isinstance(results[0], OpenThread)


# === MemoryCoordinator.get_unresolved_threads (#121) ===

def _make_coordinator(thread_store):
    coord = MemoryCoordinator.__new__(MemoryCoordinator)
    coord.thread_store = thread_store
    return coord


class TestCoordinatorUnresolvedThreads:
    def test_retrieval_error_propagates_unchanged(self):
        """#121: a RetrievalError (e.g. via get_top_threads) reaches the
        caller unchanged, not re-wrapped."""
        original = RetrievalError(source="thread_store", reason="list_open:RuntimeError")
        thread_store = SimpleNamespace(get_top_threads=MagicMock(side_effect=original))
        coord = _make_coordinator(thread_store)
        with pytest.raises(RetrievalError) as exc_info:
            coord.get_unresolved_threads(max_results=3)
        assert exc_info.value is original

    def test_other_exception_wrapped_as_retrieval_error(self):
        """#121: the swallow-to-[] is removed; a non-RetrievalError failure
        is wrapped instead of vanishing."""
        thread_store = SimpleNamespace(
            get_top_threads=MagicMock(side_effect=ValueError(f"boom {MARKER}")))
        coord = _make_coordinator(thread_store)
        with pytest.raises(RetrievalError) as exc_info:
            coord.get_unresolved_threads(max_results=3)
        err = exc_info.value
        assert err.source == "unresolved_threads" and err.reason == "ValueError"
        assert MARKER not in str(err)

    def test_control_no_thread_store_returns_empty(self):
        """Control: deliberate skip (no thread store) still returns []."""
        assert _make_coordinator(None).get_unresolved_threads() == []

    def test_control_healthy_returns_dicts(self):
        """Control: a healthy read still returns thread dicts."""
        thread = _make_thread(topic="Healthy", thread_id="h1")
        thread_store = SimpleNamespace(get_top_threads=MagicMock(return_value=[thread]))
        coord = _make_coordinator(thread_store)
        assert coord.get_unresolved_threads(max_results=3) == [thread.to_dict()]


# === Through the gatherer: the REAL coordinator method, fake store ===

class _BareGathererHost(KnowledgeRetrievalMixin):
    def __init__(self, memory_coordinator):
        self.memory_coordinator = memory_coordinator


class TestGathererThroughRealCoordinator:
    @pytest.mark.asyncio
    async def test_coordinator_raise_becomes_failed_outcome(self, monkeypatch):
        """F11a design note: the gatherer needs no code change -- a raise
        from the real coordinator method already reaches its typed except,
        recorded as failed with reason 'RetrievalError'."""
        monkeypatch.setattr("config.app_config.THREAD_SURFACING_ENABLED", True)
        thread_store = SimpleNamespace(
            get_top_threads=MagicMock(side_effect=RuntimeError(f"boom {MARKER}")))
        coord = MemoryCoordinator.__new__(MemoryCoordinator)
        coord.thread_store = thread_store
        result = await _BareGathererHost(coord).get_unresolved_threads(max_results=3)
        assert result == []
        assert outcome_status(result) == ("failed", "RetrievalError")
        assert MARKER not in result.reason


# === ShutdownProcessor._process_open_threads ===

def _make_processor(thread_store):
    proc = ShutdownProcessor.__new__(ShutdownProcessor)
    proc.thread_store = thread_store
    proc.model_manager = SimpleNamespace(generate_once=AsyncMock())
    return proc


_SESSION = [
    {"query": f"session msg one {MARKER}", "response": "ok"},
    {"query": f"session msg two {MARKER}", "response": "ok"},
]


class TestShutdownOpenThreadsReadDegrade:
    @pytest.mark.asyncio
    async def test_retrieval_error_skips_extraction_and_store_no_exception(
        self, monkeypatch, caplog
    ):
        """Contract 5: the typed RetrievalError from the open-threads read
        returns BEFORE resolution/extraction/store -- no duplicates."""
        monkeypatch.setattr("config.app_config.THREAD_SURFACING_ENABLED", True)
        monkeypatch.setattr(ThreadExtractor, "extract_new_threads",
                             AsyncMock(side_effect=AssertionError("must not be called")))
        monkeypatch.setattr(ThreadExtractor, "detect_resolutions",
                             AsyncMock(side_effect=AssertionError("must not be called")))
        thread_store = SimpleNamespace(
            list_open_threads=MagicMock(
                side_effect=RetrievalError(source="thread_store", reason="list_open:RuntimeError")),
            store_thread=MagicMock(), enforce_cap=MagicMock(),
        )
        proc = _make_processor(thread_store)
        with caplog.at_level("WARNING"):
            await proc._process_open_threads(list(_SESSION))
        assert not thread_store.store_thread.called
        assert not thread_store.enforce_cap.called
        assert "[Shutdown] Listing open threads failed" in caplog.text
        assert MARKER not in caplog.text

    @pytest.mark.asyncio
    async def test_control_genuinely_empty_open_list_extracts_and_stores(self, monkeypatch, caplog):
        """Control: a genuinely empty open list runs the pass as today."""
        monkeypatch.setattr("config.app_config.THREAD_SURFACING_ENABLED", True)
        new_threads = [_make_thread(topic="New one", thread_id="n1"),
                       _make_thread(topic="New two", thread_id="n2")]
        monkeypatch.setattr(ThreadExtractor, "extract_new_threads",
                             AsyncMock(return_value=new_threads))
        thread_store = SimpleNamespace(
            list_open_threads=MagicMock(return_value=[]),
            store_thread=MagicMock(side_effect=["doc1", "doc2"]),
            enforce_cap=MagicMock(return_value=0),
            touch_thread=MagicMock(), resolve_thread=MagicMock(),
        )
        proc = _make_processor(thread_store)
        with caplog.at_level("INFO"):
            await proc._process_open_threads(list(_SESSION))
        assert thread_store.store_thread.call_count == 2
        assert thread_store.enforce_cap.called
        assert "Stored 2 new thread(s), 0 failed" in caplog.text
        assert MARKER not in caplog.text

    @pytest.mark.asyncio
    async def test_one_write_failure_counted_second_still_stored(self, monkeypatch, caplog):
        """Contract 5: a StoreWriteError from one thread is counted, the
        loop continues, enforce_cap still runs, no exception."""
        monkeypatch.setattr("config.app_config.THREAD_SURFACING_ENABLED", True)
        new_threads = [_make_thread(topic="New one", thread_id="n1"),
                       _make_thread(topic="New two", thread_id="n2")]
        monkeypatch.setattr(ThreadExtractor, "extract_new_threads",
                             AsyncMock(return_value=new_threads))
        thread_store = SimpleNamespace(
            list_open_threads=MagicMock(return_value=[]),
            store_thread=MagicMock(side_effect=[
                StoreWriteError(source="thread_store", reason="RuntimeError"), "doc2"]),
            enforce_cap=MagicMock(return_value=0),
            touch_thread=MagicMock(), resolve_thread=MagicMock(),
        )
        proc = _make_processor(thread_store)
        with caplog.at_level("INFO"):
            await proc._process_open_threads(list(_SESSION))
        assert thread_store.store_thread.call_count == 2
        assert thread_store.enforce_cap.called
        assert "Stored 1 new thread(s), 1 failed" in caplog.text
        assert MARKER not in caplog.text
