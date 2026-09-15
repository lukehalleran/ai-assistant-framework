"""Tests for the CM-07 staged replacement in memory/thread_store.py.

`resolve_thread` and `_update_thread` must write the new version of a thread
BEFORE deleting the old document, and delete the old document only after a
confirmed write, so a failed re-store never loses the existing record
(failure_outcome_design.md, "[verified] Delete-then-store loses the record").

A local, in-memory fake store/collection is used throughout (no real Chroma).
The fake can be told to raise on the next `add_to_collection()` call and/or
on a `delete()` call naming a specific document id, so each staged-replacement
branch (write fails, delete fails and rolls back, success, missing id) can be
exercised directly.
"""
import logging

from memory.thread_models import OpenThread, ThreadType, ThreadStatus
from memory.thread_store import ThreadStore, COLLECTION_NAME

MARKER = "SYNTHETIC_test_thread_staged_replacement_marker"


# ---------------------------------------------------------------------------
# Fake ChromaDB infrastructure (records adds/deletes; raises on demand)
# ---------------------------------------------------------------------------

class FakeCollection:
    """Fake ChromaDB collection: count()/delete(), with injectable delete failures."""

    def __init__(self, items):
        self._items = items  # shared list reference with FakeChromaStore
        self._fail_delete_ids = set()
        self.delete_calls = []

    def count(self):
        return len(self._items)

    def fail_delete(self, doc_id):
        """The next delete() call naming doc_id raises instead of deleting."""
        self._fail_delete_ids.add(doc_id)

    def delete(self, ids=None):
        ids = list(ids or [])
        self.delete_calls.append(ids)
        if any(i in self._fail_delete_ids for i in ids):
            raise RuntimeError(f"{MARKER}_delete")
        matched = set(ids)
        self._items[:] = [item for item in self._items if item.get("id") not in matched]


class FakeChromaStore:
    """Fake of MultiCollectionChromaStore's subset used by ThreadStore."""

    def __init__(self, items=None):
        self._items = items if items is not None else []
        self._id_counter = 0
        self.collection = FakeCollection(self._items)
        self.collections = {COLLECTION_NAME: self.collection}
        self._add_outcomes = []  # queue of bools; True = next add() raises

    def queue_add_failure(self):
        self._add_outcomes.append(True)

    def add_to_collection(self, name, text, metadata):
        fail = self._add_outcomes.pop(0) if self._add_outcomes else False
        if fail:
            raise RuntimeError(f"{MARKER}_add")
        self._id_counter += 1
        doc_id = f"doc_{self._id_counter}"
        self._items.append({"id": doc_id, "content": text, "metadata": dict(metadata)})
        return doc_id

    def list_all(self, name):
        return list(self._items)

    def create_collection(self, name):
        return self.collection


def _make_thread(topic="Test topic", thread_id=None, status=ThreadStatus.OPEN, urgency=0.5):
    kwargs = dict(topic=topic, thread_type=ThreadType.UNFINISHED, urgency=urgency, status=status)
    if thread_id is not None:
        kwargs["thread_id"] = thread_id
    return OpenThread(**kwargs)


def _items_for(chroma, thread_id):
    return [i for i in chroma.list_all(COLLECTION_NAME)
            if i["metadata"].get("thread_id") == thread_id]


# ===========================================================================
# resolve_thread — staged replacement
# ===========================================================================

class TestResolveThreadStagedReplacement:

    def test_store_fails_keeps_old_document(self):
        """A raising re-store leaves the old (still-open) document untouched."""
        chroma = FakeChromaStore()
        store = ThreadStore(chroma_store=chroma)
        store.store_thread(_make_thread(topic="Fix bug", thread_id="r1"))

        chroma.queue_add_failure()
        assert store.resolve_thread("r1", resolution="Bug was fixed") is False

        items = _items_for(chroma, "r1")
        assert len(items) == 1
        assert items[0]["metadata"]["status"] == "open"
        assert chroma.collection.delete_calls == []  # no delete was attempted

    def test_delete_of_old_document_fails_rolls_back_new_document(self):
        """A delete failure rolls the new document back; exactly the old one remains."""
        chroma = FakeChromaStore()
        store = ThreadStore(chroma_store=chroma)
        store.store_thread(_make_thread(topic="Review PR", thread_id="r2"))
        old_id = _items_for(chroma, "r2")[0]["id"]
        chroma.collection.fail_delete(old_id)

        assert store.resolve_thread("r2", resolution="PR approved") is False

        items = _items_for(chroma, "r2")
        assert len(items) == 1
        assert items[0]["id"] == old_id  # the new (resolved) doc was rolled back
        assert items[0]["metadata"]["status"] == "open"

    def test_success_stages_replacement(self):
        """A clean write+delete: exactly one resolved document, a fresh id."""
        chroma = FakeChromaStore()
        store = ThreadStore(chroma_store=chroma)
        store.store_thread(_make_thread(topic="Deploy", thread_id="r3"))
        old_id = _items_for(chroma, "r3")[0]["id"]

        assert store.resolve_thread("r3", resolution="Deployed") is True

        items = _items_for(chroma, "r3")
        assert len(items) == 1
        assert items[0]["id"] != old_id  # a fresh uuid-style id, not an in-place edit
        assert items[0]["metadata"]["status"] == "resolved"
        assert items[0]["metadata"]["resolution_hint"] == "Deployed"

    def test_item_without_id_returns_false_and_writes_nothing(self):
        """Today this silently stores a duplicate; staged replacement must refuse."""
        chroma = FakeChromaStore()
        store = ThreadStore(chroma_store=chroma)
        store.store_thread(_make_thread(topic="No id", thread_id="r4"))
        _items_for(chroma, "r4")[0]["id"] = None

        assert store.resolve_thread("r4") is False

        items = _items_for(chroma, "r4")
        assert len(items) == 1  # nothing new was stored
        assert items[0]["id"] is None
        assert items[0]["metadata"]["status"] == "open"  # nothing was deleted either
        assert chroma.collection.delete_calls == []

    def test_not_found_returns_false(self):
        """Control: unmatched thread_id behaves exactly as before."""
        chroma = FakeChromaStore()
        store = ThreadStore(chroma_store=chroma)
        store.store_thread(_make_thread(topic="Other", thread_id="other"))

        assert store.resolve_thread("nonexistent") is False


# ===========================================================================
# _update_thread / touch_thread — staged replacement
# ===========================================================================

class TestUpdateThreadStagedReplacement:

    def test_store_fails_keeps_old_document(self):
        chroma = FakeChromaStore()
        store = ThreadStore(chroma_store=chroma)
        thread = _make_thread(topic="Original", thread_id="u1", urgency=0.3)
        store.store_thread(thread)

        thread.urgency = 0.9
        chroma.queue_add_failure()
        assert store._update_thread(thread) is False

        items = _items_for(chroma, "u1")
        assert len(items) == 1
        assert items[0]["metadata"]["urgency"] == 0.3  # old version, untouched
        assert chroma.collection.delete_calls == []

    def test_delete_of_old_document_fails_rolls_back_new_document(self):
        chroma = FakeChromaStore()
        store = ThreadStore(chroma_store=chroma)
        thread = _make_thread(topic="Original", thread_id="u2", urgency=0.3)
        store.store_thread(thread)
        old_id = _items_for(chroma, "u2")[0]["id"]
        chroma.collection.fail_delete(old_id)

        thread.urgency = 0.9
        assert store._update_thread(thread) is False

        items = _items_for(chroma, "u2")
        assert len(items) == 1
        assert items[0]["id"] == old_id
        assert items[0]["metadata"]["urgency"] == 0.3  # rolled back to the old value

    def test_touch_thread_success_stages_replacement(self):
        chroma = FakeChromaStore()
        store = ThreadStore(chroma_store=chroma)
        thread = _make_thread(topic="Tracked task", thread_id="u3")
        store.store_thread(thread)
        old_id = _items_for(chroma, "u3")[0]["id"]

        assert store.touch_thread(thread) is True

        items = _items_for(chroma, "u3")
        assert len(items) == 1
        assert items[0]["id"] != old_id

    def test_enforce_cap_counts_only_successful_updates(self):
        """One update fails, one succeeds → enforce_cap reports exactly 1 pruned."""
        chroma = FakeChromaStore()
        store = ThreadStore(chroma_store=chroma)
        store.store_thread(_make_thread(topic="Low", thread_id="lo", urgency=0.1))
        store.store_thread(_make_thread(topic="Lower", thread_id="lower", urgency=0.05))

        chroma.queue_add_failure()  # the first of the two prune-triggered re-stores fails
        pruned = store.enforce_cap(max_open=0)

        assert pruned == 1


# ===========================================================================
# Privacy: injected markers never reach the new log lines
# ===========================================================================

def _staged_replacement_lines(caplog):
    """Only the new _replace_stored_thread() log lines — not the pre-existing,
    out-of-scope store_thread()/resolve_thread() outer-except lines, which
    interpolate the raw exception and are untouched by this batch."""
    return [r.message for r in caplog.records if "Staged replacement" in r.message]


class TestStagedReplacementPrivacy:

    def test_store_failure_marker_not_in_logs(self, caplog):
        chroma = FakeChromaStore()
        store = ThreadStore(chroma_store=chroma)
        store.store_thread(_make_thread(topic="Fix bug", thread_id="p1"))

        chroma.queue_add_failure()
        with caplog.at_level(logging.ERROR):
            assert store.resolve_thread("p1", resolution="fixed") is False

        lines = _staged_replacement_lines(caplog)
        assert lines  # the new log line fired
        assert not any(MARKER in line for line in lines)
        assert any("StoreWriteError" in line for line in lines)

    def test_delete_failure_marker_not_in_logs(self, caplog):
        chroma = FakeChromaStore()
        store = ThreadStore(chroma_store=chroma)
        store.store_thread(_make_thread(topic="Fix bug", thread_id="p2"))
        old_id = _items_for(chroma, "p2")[0]["id"]
        chroma.collection.fail_delete(old_id)

        with caplog.at_level(logging.ERROR):
            assert store.resolve_thread("p2", resolution="fixed") is False

        lines = _staged_replacement_lines(caplog)
        assert lines
        assert not any(MARKER in line for line in lines)
