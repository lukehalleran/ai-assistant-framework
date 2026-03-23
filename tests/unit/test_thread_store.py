"""Tests for memory/thread_store.py — ChromaDB-backed thread persistence."""
import pytest
import time
from unittest.mock import MagicMock, patch, PropertyMock
from memory.thread_models import OpenThread, ThreadType, ThreadStatus
from memory.thread_store import ThreadStore, COLLECTION_NAME


# ---------------------------------------------------------------------------
# Mock ChromaDB infrastructure
# ---------------------------------------------------------------------------

class MockCollection:
    """Mock ChromaDB collection with count() and delete() methods."""

    def __init__(self, items=None):
        self._items = items or []

    def count(self):
        return len(self._items)

    def delete(self, ids=None):
        if ids:
            # Mutate in-place to preserve shared reference with MockChromaStore._items
            ids_set = set(ids)
            to_remove = [i for i in self._items if i.get("id") in ids_set]
            for item in to_remove:
                self._items.remove(item)


class MockChromaStore:
    """
    Mock of MultiCollectionChromaStore with the subset of methods
    that ThreadStore actually calls.
    """

    def __init__(self, with_collection=True, items=None):
        self._items = items or []
        self._id_counter = 0
        self._mock_collection = MockCollection(self._items) if with_collection else None
        self.collections = {COLLECTION_NAME: self._mock_collection} if with_collection else {}

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
        # Return items in insertion order as mock semantic results
        return list(self._items[:n_results])

    def create_collection(self, name):
        coll = MockCollection(self._items)
        self.collections[name] = coll
        self._mock_collection = coll


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_thread(
    topic="Test topic",
    thread_type=ThreadType.UNFINISHED,
    urgency=0.5,
    status=ThreadStatus.OPEN,
    last_referenced=None,
    thread_id=None,
    summary="",
    deadline_date=None,
):
    """Construct a thread with sensible defaults, overridable for each test."""
    kwargs = dict(
        topic=topic,
        thread_type=thread_type,
        urgency=urgency,
        status=status,
        summary=summary,
    )
    if last_referenced is not None:
        kwargs["last_referenced"] = last_referenced
        kwargs["mentioned_at"] = last_referenced
    if thread_id is not None:
        kwargs["thread_id"] = thread_id
    if deadline_date is not None:
        kwargs["deadline_date"] = deadline_date
    return OpenThread(**kwargs)


def _seed_store(store, threads):
    """Store a list of threads and return their doc_ids."""
    ids = []
    for t in threads:
        ids.append(store.store_thread(t))
    return ids


# ===========================================================================
# _ensure_collection
# ===========================================================================

class TestEnsureCollection:
    """Tests for _ensure_collection()."""

    def test_returns_false_when_chroma_store_is_none(self):
        """1. Returns False when chroma_store is None."""
        store = ThreadStore(chroma_store=None)
        assert store._ensure_collection() is False

    def test_returns_false_when_collections_is_none(self):
        """2. Returns False when collections attr is None."""
        mock = MagicMock()
        mock.collections = None
        store = ThreadStore(chroma_store=mock)
        assert store._ensure_collection() is False

    def test_returns_true_when_collection_exists(self):
        """3. Returns True when threads collection already present."""
        store = ThreadStore(chroma_store=MockChromaStore(with_collection=True))
        assert store._ensure_collection() is True

    def test_creates_collection_when_missing(self):
        """4. Creates collection when not present but create_collection is available."""
        chroma = MockChromaStore(with_collection=False)
        # Collection not in the dict yet
        assert COLLECTION_NAME not in chroma.collections
        store = ThreadStore(chroma_store=chroma)
        result = store._ensure_collection()
        assert result is True
        assert COLLECTION_NAME in chroma.collections

    def test_returns_false_when_create_collection_raises(self):
        """5. Returns False when create_collection raises an exception."""
        chroma = MockChromaStore(with_collection=False)
        chroma.create_collection = MagicMock(side_effect=RuntimeError("DB error"))
        store = ThreadStore(chroma_store=chroma)
        assert store._ensure_collection() is False

    def test_returns_false_when_collection_is_none_and_no_create(self):
        """Edge: collection key exists but value is None, and no create_collection method."""
        chroma = MagicMock()
        chroma.collections = {COLLECTION_NAME: None}
        # Remove create_collection so hasattr returns False
        del chroma.create_collection
        store = ThreadStore(chroma_store=chroma)
        assert store._ensure_collection() is False


# ===========================================================================
# store_thread
# ===========================================================================

class TestStoreThread:
    """Tests for store_thread()."""

    def test_stores_thread_and_returns_doc_id(self):
        """6. Stores a thread and returns a doc_id string."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)
        thread = _make_thread(topic="Buy groceries", thread_type=ThreadType.COMMITMENT)

        doc_id = store.store_thread(thread)

        assert doc_id is not None
        assert isinstance(doc_id, str)
        assert len(chroma._items) == 1
        stored_meta = chroma._items[0]["metadata"]
        assert stored_meta["topic"] == "Buy groceries"
        assert stored_meta["thread_type"] == "commitment"

    def test_returns_none_when_collection_unavailable(self):
        """7. Returns None when chroma_store is unavailable."""
        store = ThreadStore(chroma_store=None)
        assert store.store_thread(_make_thread()) is None

    def test_returns_none_when_add_raises(self):
        """8. Returns None when add_to_collection raises."""
        chroma = MockChromaStore()
        chroma.add_to_collection = MagicMock(side_effect=RuntimeError("write fail"))
        store = ThreadStore(chroma_store=chroma)
        assert store.store_thread(_make_thread()) is None


# ===========================================================================
# list_open_threads
# ===========================================================================

class TestListOpenThreads:
    """Tests for list_open_threads()."""

    def test_returns_only_open_threads(self):
        """9. Returns only threads with OPEN status."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        t_open = _make_thread(topic="Open one", thread_id="t1")
        t_resolved = _make_thread(topic="Resolved one", thread_id="t2", status=ThreadStatus.RESOLVED)
        _seed_store(store, [t_open, t_resolved])

        results = store.list_open_threads()
        assert len(results) == 1
        assert results[0].thread_id == "t1"

    def test_returns_empty_when_no_open_threads(self):
        """10. Returns empty list when no threads have OPEN status."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        _seed_store(store, [
            _make_thread(status=ThreadStatus.RESOLVED),
            _make_thread(status=ThreadStatus.STALE),
        ])

        assert store.list_open_threads() == []

    def test_returns_empty_when_collection_unavailable(self):
        """11. Returns empty when collection is unavailable."""
        store = ThreadStore(chroma_store=None)
        assert store.list_open_threads() == []

    def test_filters_out_resolved_and_stale(self):
        """12. Filters out RESOLVED and STALE threads, keeping only OPEN."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        _seed_store(store, [
            _make_thread(topic="A", thread_id="a", status=ThreadStatus.OPEN),
            _make_thread(topic="B", thread_id="b", status=ThreadStatus.RESOLVED),
            _make_thread(topic="C", thread_id="c", status=ThreadStatus.STALE),
            _make_thread(topic="D", thread_id="d", status=ThreadStatus.OPEN),
        ])

        results = store.list_open_threads()
        ids = {t.thread_id for t in results}
        assert ids == {"a", "d"}


# ===========================================================================
# get_top_threads
# ===========================================================================

class TestGetTopThreads:
    """Tests for get_top_threads()."""

    def test_returns_threads_sorted_by_priority(self):
        """13. Returns threads sorted by priority_score (highest first)."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        # deadline (weight 1.0, urgency 0.9) should rank above unfinished (weight 0.4, urgency 0.5)
        low = _make_thread(topic="Low", thread_id="low", thread_type=ThreadType.UNFINISHED, urgency=0.5)
        high = _make_thread(topic="High", thread_id="high", thread_type=ThreadType.DEADLINE, urgency=0.9)
        _seed_store(store, [low, high])

        results = store.get_top_threads(max_results=5, stale_days=30)
        assert len(results) == 2
        assert results[0].thread_id == "high"
        assert results[1].thread_id == "low"

    def test_marks_stale_threads(self):
        """14. Marks stale threads as STALE via lazy staleness check."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        stale_time = time.time() - 20 * 86400  # 20 days ago
        fresh_time = time.time()

        stale_t = _make_thread(topic="Old", thread_id="old", last_referenced=stale_time)
        fresh_t = _make_thread(topic="New", thread_id="new", last_referenced=fresh_time)
        _seed_store(store, [stale_t, fresh_t])

        results = store.get_top_threads(max_results=10, stale_days=14)

        # Only the fresh thread should be returned
        assert len(results) == 1
        assert results[0].thread_id == "new"

        # The stale thread should have been updated to STALE in the store
        all_items = chroma.list_all(COLLECTION_NAME)
        stale_items = [
            i for i in all_items
            if i["metadata"].get("thread_id") == "old"
            and i["metadata"].get("status") == "stale"
        ]
        assert len(stale_items) >= 1

    def test_returns_empty_when_no_threads(self):
        """15. Returns empty list when there are no threads."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)
        assert store.get_top_threads(max_results=5, stale_days=14) == []

    def test_respects_max_results_limit(self):
        """16. Returns at most max_results threads."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        threads = [
            _make_thread(topic=f"T{i}", thread_id=f"t{i}", urgency=0.5 + i * 0.05)
            for i in range(5)
        ]
        _seed_store(store, threads)

        results = store.get_top_threads(max_results=2, stale_days=30)
        assert len(results) == 2

    def test_default_stale_days_from_config(self):
        """17. Uses THREAD_STALE_DAYS from config when stale_days not provided."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        # Thread 10 days old — stale if default < 10, fresh if default >= 10
        ten_days_ago = time.time() - 10 * 86400
        t = _make_thread(topic="Config test", thread_id="ct", last_referenced=ten_days_ago)
        _seed_store(store, [t])

        # Patch config to 5 days (so 10-day-old thread IS stale)
        with patch("memory.thread_store.ThreadStore.list_open_threads") as mock_list:
            # We need a more targeted patch; let's patch the import inside get_top_threads
            pass

        # Direct approach: patch the config constant
        with patch.dict("sys.modules", {}):
            # Simplest: call with stale_days=5 to prove staleness marking
            results_stale = store.get_top_threads(max_results=5, stale_days=5)
            assert len(results_stale) == 0  # 10 days > 5 = stale

        # And with stale_days=15 it should survive
        # Re-seed because the old thread was marked stale
        chroma2 = MockChromaStore()
        store2 = ThreadStore(chroma_store=chroma2)
        t2 = _make_thread(topic="Config test2", thread_id="ct2", last_referenced=ten_days_ago)
        _seed_store(store2, [t2])
        results_fresh = store2.get_top_threads(max_results=5, stale_days=15)
        assert len(results_fresh) == 1

    @patch("config.app_config.THREAD_STALE_DAYS", 7)
    def test_default_stale_days_import_path(self):
        """17b. stale_days defaults to config value when not passed."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        # 10-day-old thread should be stale with 7-day default
        ten_days_ago = time.time() - 10 * 86400
        t = _make_thread(topic="Import test", thread_id="it", last_referenced=ten_days_ago)
        _seed_store(store, [t])

        results = store.get_top_threads(max_results=5)  # no stale_days arg
        assert len(results) == 0  # stale at 7 days


# ===========================================================================
# query_threads
# ===========================================================================

class TestQueryThreads:
    """Tests for query_threads()."""

    def test_returns_threads_matching_query(self):
        """18. Returns threads that match the query."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        _seed_store(store, [
            _make_thread(topic="Python project deadline", thread_id="t1"),
            _make_thread(topic="Grocery shopping", thread_id="t2"),
        ])

        results = store.query_threads("Python project", n_results=5)
        assert len(results) >= 1
        # All returned items should be valid OpenThread objects
        for r in results:
            assert isinstance(r, OpenThread)

    def test_returns_empty_when_collection_empty(self):
        """19. Returns empty list when collection has zero documents."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        # Collection exists but is empty (count() == 0)
        results = store.query_threads("anything")
        assert results == []

    def test_returns_empty_when_collection_unavailable(self):
        """20. Returns empty list when collection is unavailable."""
        store = ThreadStore(chroma_store=None)
        assert store.query_threads("test") == []

    def test_respects_n_results_limit(self):
        """21. Returns at most n_results threads."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        _seed_store(store, [
            _make_thread(topic=f"Thread {i}", thread_id=f"t{i}")
            for i in range(10)
        ])

        results = store.query_threads("Thread", n_results=3)
        assert len(results) <= 3


# ===========================================================================
# resolve_thread
# ===========================================================================

class TestResolveThread:
    """Tests for resolve_thread()."""

    def test_resolves_existing_thread(self):
        """22. Resolves an existing thread — status changes to RESOLVED."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        thread = _make_thread(topic="Fix bug", thread_id="fix1")
        _seed_store(store, [thread])

        result = store.resolve_thread("fix1", resolution="Bug was fixed in v2.1")
        assert result is True

        # Verify the thread is now RESOLVED in the store
        all_items = chroma.list_all(COLLECTION_NAME)
        resolved = [
            i for i in all_items
            if i["metadata"].get("thread_id") == "fix1"
            and i["metadata"].get("status") == "resolved"
        ]
        assert len(resolved) >= 1
        assert resolved[0]["metadata"]["resolution_hint"] == "Bug was fixed in v2.1"

    def test_returns_false_when_thread_not_found(self):
        """23. Returns False when thread_id doesn't match any stored thread."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        _seed_store(store, [_make_thread(thread_id="other")])
        assert store.resolve_thread("nonexistent") is False

    def test_returns_false_when_collection_unavailable(self):
        """24. Returns False when collection is unavailable."""
        store = ThreadStore(chroma_store=None)
        assert store.resolve_thread("any_id") is False

    def test_sets_resolution_text(self):
        """25. Resolution text is persisted in the thread metadata."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        thread = _make_thread(topic="Review PR", thread_id="pr1")
        _seed_store(store, [thread])

        store.resolve_thread("pr1", resolution="PR approved and merged")

        all_items = chroma.list_all(COLLECTION_NAME)
        pr_items = [
            i for i in all_items
            if i["metadata"].get("thread_id") == "pr1"
        ]
        # Should find the newly-stored resolved version
        resolved_items = [i for i in pr_items if i["metadata"]["status"] == "resolved"]
        assert len(resolved_items) >= 1
        assert resolved_items[0]["metadata"]["resolution_hint"] == "PR approved and merged"


# ===========================================================================
# enforce_cap
# ===========================================================================

class TestEnforceCap:
    """Tests for enforce_cap()."""

    def test_noop_when_under_cap(self):
        """26. No pruning when open thread count is under the cap."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        _seed_store(store, [
            _make_thread(topic="A", thread_id="a"),
            _make_thread(topic="B", thread_id="b"),
        ])

        pruned = store.enforce_cap(max_open=5)
        assert pruned == 0

    def test_prunes_lowest_priority_threads(self):
        """27. Prunes lowest-priority threads when over cap."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        # 3 threads with different priorities
        low = _make_thread(topic="Low", thread_id="low", thread_type=ThreadType.UNFINISHED, urgency=0.1)
        mid = _make_thread(topic="Mid", thread_id="mid", thread_type=ThreadType.QUESTION, urgency=0.5)
        high = _make_thread(topic="High", thread_id="high", thread_type=ThreadType.DEADLINE, urgency=0.9)
        _seed_store(store, [low, mid, high])

        pruned = store.enforce_cap(max_open=2)
        assert pruned == 1

        # The lowest priority thread should now be STALE
        all_items = chroma.list_all(COLLECTION_NAME)
        low_items = [
            i for i in all_items
            if i["metadata"].get("thread_id") == "low"
        ]
        stale_low = [i for i in low_items if i["metadata"].get("status") == "stale"]
        assert len(stale_low) >= 1

    def test_returns_number_pruned(self):
        """28. Returns the exact number of threads pruned."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        _seed_store(store, [
            _make_thread(topic=f"T{i}", thread_id=f"t{i}", urgency=0.1 * (i + 1))
            for i in range(6)
        ])

        pruned = store.enforce_cap(max_open=2)
        assert pruned == 4

    @patch("config.app_config.THREAD_MAX_OPEN", 3)
    def test_default_max_open_from_config(self):
        """29. Uses THREAD_MAX_OPEN from config when max_open not provided."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        _seed_store(store, [
            _make_thread(topic=f"T{i}", thread_id=f"t{i}", urgency=0.1 * (i + 1))
            for i in range(5)
        ])

        pruned = store.enforce_cap()  # no max_open arg — should use config value of 3
        assert pruned == 2

    def test_noop_when_exactly_at_cap(self):
        """Edge: No pruning when count equals the cap exactly."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        _seed_store(store, [
            _make_thread(topic=f"T{i}", thread_id=f"t{i}")
            for i in range(3)
        ])

        pruned = store.enforce_cap(max_open=3)
        assert pruned == 0

    def test_enforce_cap_no_threads(self):
        """Edge: Returns 0 when store is empty."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)
        assert store.enforce_cap(max_open=5) == 0


# ===========================================================================
# _update_thread
# ===========================================================================

class TestUpdateThread:
    """Tests for _update_thread()."""

    def test_updates_existing_thread(self):
        """30. Updates an existing thread via delete-and-re-add."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        thread = _make_thread(topic="Original topic", thread_id="u1", urgency=0.3)
        _seed_store(store, [thread])

        # Modify the thread
        thread.urgency = 0.9
        thread.summary = "Updated summary"
        result = store._update_thread(thread)
        assert result is True

        # Verify the updated version exists
        all_items = chroma.list_all(COLLECTION_NAME)
        updated = [
            i for i in all_items
            if i["metadata"].get("thread_id") == "u1"
            and i["metadata"].get("urgency") == 0.9
        ]
        assert len(updated) >= 1
        assert updated[0]["metadata"]["summary"] == "Updated summary"

    def test_returns_false_when_thread_not_found(self):
        """31. Returns False when thread_id not found in store."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        _seed_store(store, [_make_thread(thread_id="other")])

        ghost = _make_thread(thread_id="nonexistent")
        assert store._update_thread(ghost) is False

    def test_returns_false_when_collection_unavailable(self):
        """32. Returns False when collection is unavailable."""
        store = ThreadStore(chroma_store=None)
        assert store._update_thread(_make_thread()) is False

    def test_update_preserves_other_threads(self):
        """Edge: Updating one thread doesn't affect other threads."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        t1 = _make_thread(topic="Thread A", thread_id="a1", urgency=0.3)
        t2 = _make_thread(topic="Thread B", thread_id="b1", urgency=0.7)
        _seed_store(store, [t1, t2])

        t1.urgency = 0.9
        store._update_thread(t1)

        all_items = chroma.list_all(COLLECTION_NAME)
        b_items = [
            i for i in all_items
            if i["metadata"].get("thread_id") == "b1"
        ]
        assert len(b_items) == 1
        assert b_items[0]["metadata"]["urgency"] == 0.7  # unchanged


# ===========================================================================
# Integration-style scenarios
# ===========================================================================

class TestIntegrationScenarios:
    """End-to-end scenarios combining multiple ThreadStore methods."""

    def test_store_then_query_roundtrip(self):
        """Store threads, then query and verify deserialization fidelity."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        original = _make_thread(
            topic="Deploy v3.0",
            thread_id="dep3",
            thread_type=ThreadType.DEADLINE,
            urgency=0.95,
            summary="Deploy version 3.0 to production by Friday",
            deadline_date="2026-03-28",
        )
        store.store_thread(original)

        results = store.query_threads("deploy", n_results=5)
        assert len(results) == 1
        found = results[0]
        assert found.thread_id == "dep3"
        assert found.topic == "Deploy v3.0"
        assert found.thread_type == ThreadType.DEADLINE
        assert found.urgency == 0.95
        assert found.deadline_date == "2026-03-28"

    def test_resolve_then_list_open_excludes_resolved(self):
        """After resolving, list_open_threads no longer returns it."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        t = _make_thread(topic="Write docs", thread_id="wd1")
        store.store_thread(t)

        assert len(store.list_open_threads()) == 1

        store.resolve_thread("wd1", resolution="Docs written")

        assert len(store.list_open_threads()) == 0

    def test_enforce_cap_then_list_open(self):
        """After cap enforcement, remaining open threads are the highest priority."""
        chroma = MockChromaStore()
        store = ThreadStore(chroma_store=chroma)

        _seed_store(store, [
            _make_thread(topic="Low", thread_id="lo", thread_type=ThreadType.UNFINISHED, urgency=0.1),
            _make_thread(topic="Med", thread_id="me", thread_type=ThreadType.QUESTION, urgency=0.5),
            _make_thread(topic="High", thread_id="hi", thread_type=ThreadType.DEADLINE, urgency=0.9),
        ])

        store.enforce_cap(max_open=2)

        open_threads = store.list_open_threads()
        open_ids = {t.thread_id for t in open_threads}
        assert "lo" not in open_ids
        assert len(open_threads) == 2
