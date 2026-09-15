"""Regression tests for CGR-20260913-009 anchor #139 (F9a).

``MultiCollectionChromaStore.get_by_id`` used to swallow ANY exception from
``coll.get(...)`` into the same ``None`` it returns for a genuine not-found
(memory/storage/multi_collection_chroma_store.py, previously 509-510). A
caller cannot tell "no such document" from "the read failed" — and
``ClaimIndex.cascade_staleness`` (memory/claim_tracker.py 290-317) treated a
``None`` as "doesn't exist (deleted?)" and DELETED the index entry, so a
transient Chroma failure silently dropped tracked-claim bookkeeping for a
document that still exists.

These tests drive the DEPLOYED ``get_by_id`` (via a fake Chroma collection
behind the same ``__new__`` + manual ``.collections`` construction used by
tests/unit/test_chroma_store_no_autodelete.py and
tests/unit/test_sep09_storage_repairs.py — never a real Chroma client here)
and the DEPLOYED ``ClaimIndex.cascade_staleness`` — first through a mocked
``get_by_id`` (isolating the cascade's own exception handling), then through
the real edited store (proving the store fix and the cascade together).

Contract (docs/execution/generalization/briefs/F9a.md):
  1. Unknown collection -> None (unchanged).
  2. Genuine not-found (empty ``documents``) -> None (unchanged).
  3. ``coll.get(...)`` raises -> ``RetrievalError(source="chroma_get_by_id",
     reason=type(e).__name__)`` — never the exception's message text.
  4. Found -> the same dict as today.
  5. A failed read must not reach ``ClaimIndex.remove_document``; a genuine
     not-found still removes the entry; a healthy doc still updates.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from memory.claim_tracker import ClaimIndex, ClaimKey
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from utils.retrieval_outcome import RetrievalError


# ---------------------------------------------------------------------------
# Fakes — no real Chroma client anywhere in this file
# ---------------------------------------------------------------------------


class _FakeCollection:
    """Stand-in for a Chroma collection's ``.get(...)``."""

    def __init__(self, *, raises=None, documents=None, metadatas=None):
        self._raises = raises
        self._documents = documents if documents is not None else []
        self._metadatas = metadatas if metadatas is not None else []

    def get(self, ids=None, include=None):
        if self._raises is not None:
            raise self._raises
        return {"documents": self._documents, "metadatas": self._metadatas}


def _make_store(collections):
    """A MultiCollectionChromaStore wired to fake collections, bypassing
    __init__ (the test_chroma_store_no_autodelete.py / test_sep09_storage_repairs.py
    pattern) so no real Chroma client or SentenceTransformer is constructed."""
    store = MultiCollectionChromaStore.__new__(MultiCollectionChromaStore)
    store.collections = collections
    return store


def _populated_index() -> ClaimIndex:
    idx = ClaimIndex()
    idx.add_claims("doc_1", "facts", [ClaimKey(subject="alex", relation="lives_in")])
    return idx


# ---------------------------------------------------------------------------
# Store: MultiCollectionChromaStore.get_by_id
# ---------------------------------------------------------------------------


class TestGetByIdOutcomes:
    def test_failed_read_raises_retrieval_error(self):
        store = _make_store({"facts": _FakeCollection(raises=RuntimeError("backend exploded"))})

        with pytest.raises(RetrievalError) as exc_info:
            store.get_by_id("facts", "doc_1")

        err = exc_info.value
        assert err.source == "chroma_get_by_id"
        assert err.reason == "RuntimeError"
        assert "backend exploded" not in err.reason
        assert "backend exploded" not in str(err)

    def test_genuine_not_found_returns_none(self):
        store = _make_store({"facts": _FakeCollection(documents=[], metadatas=[])})
        assert store.get_by_id("facts", "doc_1") is None

    def test_found_returns_dict(self):
        store = _make_store({"facts": _FakeCollection(documents=["hello"], metadatas=[{"a": 1}])})
        assert store.get_by_id("facts", "doc_1") == {
            "id": "doc_1", "content": "hello", "metadata": {"a": 1},
        }

    def test_unknown_collection_returns_none(self):
        store = _make_store({})
        assert store.get_by_id("facts", "doc_1") is None


# ---------------------------------------------------------------------------
# ClaimIndex.cascade_staleness with a MOCKED get_by_id (isolates the
# cascade's own exception handling from the store fix).
# ---------------------------------------------------------------------------


class TestCascadeStalenessMockedGetById:
    def test_raising_get_by_id_does_not_delete_index_entry(self):
        idx = _populated_index()
        mock_chroma = MagicMock(spec=MultiCollectionChromaStore)
        mock_chroma.get_by_id.side_effect = RetrievalError(
            source="chroma_get_by_id", reason="RuntimeError",
        )
        ck = ClaimKey(subject="alex", relation="lives_in")

        with patch.object(idx, "remove_document") as mock_remove:
            results = idx.cascade_staleness(ck, chroma_store=mock_chroma)

        mock_remove.assert_not_called()
        assert "doc_1" in idx._doc_claims  # index entry retained
        assert results == []  # the failed doc gets no result entry
        mock_chroma.update_metadata.assert_not_called()

    def test_control_genuine_not_found_removes_entry(self):
        """Paired control: a genuine not-found (None) still removes — unchanged."""
        idx = _populated_index()
        mock_chroma = MagicMock(spec=MultiCollectionChromaStore)
        mock_chroma.get_by_id.return_value = None
        ck = ClaimKey(subject="alex", relation="lives_in")

        idx.cascade_staleness(ck, chroma_store=mock_chroma)

        assert "doc_1" not in idx._doc_claims  # removed, as before

    def test_control_healthy_doc_updates_metadata(self):
        """Paired control: a healthy read still updates metadata and returns a result."""
        idx = _populated_index()
        mock_chroma = MagicMock(spec=MultiCollectionChromaStore)
        mock_chroma.get_by_id.return_value = {
            "id": "doc_1", "content": "x",
            "metadata": {"staleness_ratio": 0.0, "stale_claims": ""},
        }
        ck = ClaimKey(subject="alex", relation="lives_in")

        results = idx.cascade_staleness(ck, chroma_store=mock_chroma)

        assert len(results) == 1
        mock_chroma.update_metadata.assert_called_once()


# ---------------------------------------------------------------------------
# ClaimIndex.cascade_staleness through the REAL edited store (proves the
# store fix and the cascade together, not just a mocked get_by_id).
# ---------------------------------------------------------------------------


class TestCascadeStalenessRealStore:
    def test_failed_read_retains_index_entry(self):
        idx = _populated_index()
        store = _make_store({"facts": _FakeCollection(raises=RuntimeError("backend exploded"))})
        ck = ClaimKey(subject="alex", relation="lives_in")

        results = idx.cascade_staleness(ck, chroma_store=store)

        assert "doc_1" in idx._doc_claims  # retained: no deletion on a failed read
        assert results == []

    def test_failed_read_warning_has_no_message_leak(self, caplog):
        marker = "SYNTH_MARKER_7f2a9c1b"
        idx = _populated_index()
        store = _make_store({
            "facts": _FakeCollection(raises=RuntimeError(f"synthetic backend detail {marker}")),
        })
        ck = ClaimKey(subject="alex", relation="lives_in")

        with caplog.at_level(logging.WARNING):
            idx.cascade_staleness(ck, chroma_store=store)

        assert marker not in caplog.text
