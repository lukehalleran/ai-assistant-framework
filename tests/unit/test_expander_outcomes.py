"""Regression tests for CGR-20260913-009 anchor #122 (F9b).

``MemoryExpander._fetch_conversations_in_range`` used to swallow ANY
exception from ``get_ids_by_timestamp_range()`` into ``[]``
(memory/memory_expander.py, previously 359-365) — the same shape as the
``get_ids_by_timestamp_range incident`` this anchor is named for. A failed
range read was indistinguishable from "no conversations in this window", so
a transient store failure silently produced the "no source_doc_ids or
temporal anchors" fallback text as if the summary genuinely had no linkage.
``MemoryExpander.expand()`` then cached that wrong result
(``docs/execution/generalization/failure_outcome_design.md`` amendment
"[map] The expander caches its wrong error").

The producer side (``MultiCollectionChromaStore.get_ids_by_timestamp_range``)
had the matching shape: both its numeric-query and legacy-page passes
swallowed any exception into a silently partial (or empty) union, reported
as if complete.

These tests drive the DEPLOYED ``get_ids_by_timestamp_range`` (via a fake
Chroma collection behind the same ``__new__`` + manual ``.collections``
construction used by F9a's tests/unit/test_store_get_by_id_outcomes.py and
tests/unit/test_sep09_storage_repairs.py — never a real Chroma client here)
and the DEPLOYED ``MemoryExpander.expand()`` / ``_fetch_conversations_in_range``
(via a fake store like tests/unit/test_memory_expander.py's, extended to
raise on demand with call counters).

Contract (docs/execution/generalization/briefs/F9b.md):
  1. ``get_ids_by_timestamp_range``:
     - numeric-query failure -> ``RetrievalError(source="timestamp_range",
       reason=f"numeric:{type(e).__name__}")``;
     - legacy-page failure -> ``RetrievalError(source="timestamp_range",
       reason=f"legacy_page:{type(e).__name__}")``;
     - ``count()`` failure, unknown collection, malformed bounds, the
       malformed-row skip and the healthy union are all unchanged.
  2. ``_fetch_conversations_in_range``:
     - a ``RetrievalError`` from the producer propagates unchanged;
     - any OTHER exception from the range call is wrapped:
       ``RetrievalError(source="timestamp_range", reason=type(e).__name__)``;
     - a genuine empty id list is still ``[]``;
     - a per-id ``get_by_id`` that raises propagates, with no per-id guard
       (one failed read fails the whole range fetch — a silently partial
       turn list is the defect class);
     - a genuine per-id not-found is still skipped.
  3. ``expand()`` wraps ONLY the ``_do_expand(...)`` call: a
     ``RetrievalError`` returns an explicit error dict
     (``error=f"expansion_failed: {e.source}: {e.reason}"``) and is NEVER
     cached; any other exception still propagates; a success, a genuine
     not-found, a suppressed anchor or the summary linkage fallback are
     still cached exactly as before.
  4. Privacy: exception message text never reaches ``RetrievalError.reason``
     or the returned ``error`` string — labels and class names only.
"""

from datetime import datetime, timedelta

import pytest

from memory.memory_expander import MemoryExpander
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from utils.retrieval_outcome import RetrievalError


# ---------------------------------------------------------------------------
# Producer: MultiCollectionChromaStore.get_ids_by_timestamp_range
# ---------------------------------------------------------------------------


class _FakeRangeCollection:
    """Stand-in for a Chroma collection's ``.get()``/``.count()``, split
    into the two ``get_ids_by_timestamp_range()`` passes (numeric ``where=``
    query vs. the legacy paged scan), each independently controllable."""

    def __init__(
        self,
        *,
        numeric_raises=None,
        legacy_raises=None,
        count_raises=None,
        numeric_ids=None,
        legacy_rows=None,
    ):
        self._numeric_raises = numeric_raises
        self._legacy_raises = legacy_raises
        self._count_raises = count_raises
        self._numeric_ids = numeric_ids if numeric_ids is not None else []
        # legacy_rows: [(doc_id, metadata_without_timestamp_epoch), ...]
        self._legacy_rows = legacy_rows if legacy_rows is not None else []

    def get(self, *, where=None, include=None, limit=None, offset=None, ids=None):
        if where is not None:
            # Pass 1: numeric timestamp_epoch query.
            if self._numeric_raises is not None:
                raise self._numeric_raises
            return {"ids": list(self._numeric_ids)}
        # Pass 2: legacy paged metadata scan.
        if self._legacy_raises is not None:
            raise self._legacy_raises
        offset = offset or 0
        page = self._legacy_rows[offset : offset + (limit or len(self._legacy_rows))]
        return {"ids": [r[0] for r in page], "metadatas": [r[1] for r in page]}

    def count(self):
        if self._count_raises is not None:
            raise self._count_raises
        return len(self._legacy_rows)


def _make_range_store(coll):
    """A MultiCollectionChromaStore wired to one fake collection, bypassing
    __init__ (the F9a pattern) so no real Chroma client is constructed."""
    store = MultiCollectionChromaStore.__new__(MultiCollectionChromaStore)
    store.collections = {"conversations": coll}
    return store


class TestTimestampRangeProducerOutcomes:
    def test_numeric_query_failure_raises_retrieval_error(self):
        marker = "SYNTH_MARKER_numeric_9f3c"
        coll = _FakeRangeCollection(numeric_raises=RuntimeError(f"backend down {marker}"))
        store = _make_range_store(coll)

        with pytest.raises(RetrievalError) as exc_info:
            store.get_ids_by_timestamp_range(
                "conversations", "2026-09-09T00:00:00", "2026-09-09T01:00:00"
            )

        err = exc_info.value
        assert err.source == "timestamp_range"
        assert err.reason == "numeric:RuntimeError"
        assert marker not in err.reason
        assert marker not in str(err)

    def test_legacy_page_failure_raises_retrieval_error(self):
        marker = "SYNTH_MARKER_legacy_2b7e"
        coll = _FakeRangeCollection(legacy_raises=RuntimeError(f"page fetch broke {marker}"))
        store = _make_range_store(coll)

        with pytest.raises(RetrievalError) as exc_info:
            store.get_ids_by_timestamp_range(
                "conversations", "2026-09-09T00:00:00", "2026-09-09T01:00:00"
            )

        err = exc_info.value
        assert err.source == "timestamp_range"
        assert err.reason == "legacy_page:RuntimeError"
        assert marker not in err.reason
        assert marker not in str(err)

    def test_count_failure_still_returns_union(self):
        """Control: count() stays advisory-only (KEEP)."""
        coll = _FakeRangeCollection(
            count_raises=RuntimeError("count broke"),
            numeric_ids=["n1"],
            legacy_rows=[("l1", {"timestamp": "2026-09-09T00:30:00"})],
        )
        store = _make_range_store(coll)

        ids = store.get_ids_by_timestamp_range(
            "conversations", "2026-09-09T00:00:00", "2026-09-09T01:00:00"
        )
        assert ids == sorted(["n1", "l1"])

    def test_malformed_bounds_returns_empty(self):
        """Control: unchanged."""
        store = _make_range_store(_FakeRangeCollection())
        assert store.get_ids_by_timestamp_range("conversations", "garbage", "also garbage") == []

    def test_healthy_numeric_plus_legacy_mix_sorted_union(self):
        """Control: unchanged healthy path — sorted union of both passes."""
        coll = _FakeRangeCollection(
            numeric_ids=["b_new"],
            legacy_rows=[("a_legacy", {"timestamp": "2026-09-09T00:30:00"})],
        )
        store = _make_range_store(coll)

        ids = store.get_ids_by_timestamp_range(
            "conversations", "2026-09-09T00:00:00", "2026-09-09T01:00:00"
        )
        assert ids == sorted(["b_new", "a_legacy"])


# ---------------------------------------------------------------------------
# Expander: MemoryExpander.expand() / _fetch_conversations_in_range()
# ---------------------------------------------------------------------------


class FakeChromaStore:
    """Like tests/unit/test_memory_expander.py's FakeChromaStore, extended
    to raise on demand from get_by_id() / get_ids_by_timestamp_range(),
    with call counters to prove cache-hit vs. re-fetch."""

    def __init__(self, data: dict | None = None):
        self._data = data or {}
        self._get_by_id_raises: dict = {}  # doc_id -> exception instance
        self._range_raises = None  # exception instance or None
        self.get_by_id_calls = 0
        self.range_calls = 0

    def get_by_id(self, collection_name: str, doc_id: str):
        self.get_by_id_calls += 1
        if doc_id in self._get_by_id_raises:
            raise self._get_by_id_raises[doc_id]
        for doc in self._data.get(collection_name, []):
            if doc["id"] == doc_id:
                return dict(doc)
        return None

    def list_all(self, collection_name: str):
        return [dict(d) for d in self._data.get(collection_name, [])]

    def get_ids_by_timestamp_range(self, collection_name: str, start_iso: str, end_iso: str):
        self.range_calls += 1
        if self._range_raises is not None:
            raise self._range_raises
        try:
            start = datetime.fromisoformat(start_iso)
            end = datetime.fromisoformat(end_iso)
        except (ValueError, TypeError):
            return []
        ids = []
        for doc in self._data.get(collection_name, []):
            ts_str = (doc.get("metadata") or {}).get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue
            if start <= ts <= end:
                ids.append(doc["id"])
        return ids


def _make_doc(doc_id: str, ts: str, content: str = "", **extra_meta) -> dict:
    meta = {"timestamp": ts}
    meta.update(extra_meta)
    if not content:
        content = (
            f"This is a detailed reflection or summary for {doc_id} with enough "
            "text to pass validation checks and be considered real content."
        )
    return {"id": doc_id, "content": content, "metadata": meta}


def _ts(offset_minutes: int) -> str:
    base = datetime(2025, 6, 15, 12, 0, 0)
    return (base + timedelta(minutes=offset_minutes)).isoformat()


class TestExpandRetrievalErrorOutcomes:
    def _make_expander(self, data):
        store = FakeChromaStore(data)
        return store, MemoryExpander(store)

    def test_range_producer_retrieval_error_not_cached(self):
        """A RetrievalError from the range producer propagates through
        _fetch_conversations_in_range unchanged and is reported by expand(),
        never cached — a SECOND call hits the store again."""
        convos = [_make_doc(f"c{i}", _ts(i * 10)) for i in range(3)]
        summary = _make_doc(
            "s1", _ts(50),
            "Summary of early conversations covering the first block of messages.",
            type="summary",
            temporal_anchor_start=_ts(0), temporal_anchor_end=_ts(20),
        )
        store, expander = self._make_expander({"summaries": [summary], "conversations": convos})
        store._range_raises = RetrievalError(source="timestamp_range", reason="numeric:RuntimeError")

        result = expander.expand("s1", collection="summaries")

        assert result["error"].startswith("expansion_failed: timestamp_range: numeric:RuntimeError")
        assert result["turns"] == []
        assert result["total_in_collection"] == 0
        assert result["expansion_method"] == "timestamp_window"
        assert result["anchor_id"] == "s1"
        assert result["collection"] == "summaries"

        calls_after_first = store.range_calls
        result2 = expander.expand("s1", collection="summaries")
        assert store.range_calls > calls_after_first  # re-fetched, not cached
        assert result2["error"].startswith("expansion_failed: timestamp_range:")

    def test_range_producer_non_retrieval_error_wrapped(self):
        """A non-RetrievalError exception from the range call is wrapped by
        _fetch_conversations_in_range, and the raw message never leaks."""
        marker = "SYNTH_MARKER_range_5d1a"
        convos = [_make_doc(f"c{i}", _ts(i * 10)) for i in range(3)]
        summary = _make_doc(
            "s1", _ts(50),
            "Summary of early conversations covering the first block of messages.",
            type="summary",
            temporal_anchor_start=_ts(0), temporal_anchor_end=_ts(20),
        )
        store, expander = self._make_expander({"summaries": [summary], "conversations": convos})
        store._range_raises = ValueError(f"driver exploded {marker}")

        result = expander.expand("s1", collection="summaries")

        assert result["error"].startswith("expansion_failed: timestamp_range: ValueError")
        assert marker not in result["error"]

    def test_anchor_get_by_id_retrieval_error_not_cached(self):
        """The anchor get_by_id raising RetrievalError -> explicit error,
        never cached."""
        summary = _make_doc(
            "s1", _ts(50),
            "Orphan-looking summary whose anchor read itself will fail.",
            type="summary",
            source_doc_ids="c1",
        )
        store, expander = self._make_expander({"summaries": [summary]})
        store._get_by_id_raises["s1"] = RetrievalError(source="chroma_get_by_id", reason="RuntimeError")

        result = expander.expand("s1", collection="summaries")
        assert result["error"].startswith("expansion_failed: chroma_get_by_id: RuntimeError")
        assert result["turns"] == []

        calls_after_first = store.get_by_id_calls
        expander.expand("s1", collection="summaries")
        assert store.get_by_id_calls > calls_after_first  # re-fetched, not cached

    def test_per_id_get_by_id_retrieval_error_aborts_range_fetch(self):
        """One failed per-id read inside the range loop fails the whole
        range fetch instead of silently dropping that turn — no per-id
        guard (per-id abort decision)."""
        convos = [_make_doc(f"c{i}", _ts(i * 10)) for i in range(3)]
        summary = _make_doc(
            "s1", _ts(50),
            "Summary of early conversations covering the first block of messages.",
            type="summary",
            temporal_anchor_start=_ts(0), temporal_anchor_end=_ts(20),
        )
        store, expander = self._make_expander({"summaries": [summary], "conversations": convos})
        store._get_by_id_raises["c1"] = RetrievalError(source="chroma_get_by_id", reason="RuntimeError")

        result = expander.expand("s1", collection="summaries")
        assert result["error"].startswith("expansion_failed: chroma_get_by_id: RuntimeError")
        assert result["turns"] == []

    def test_non_retrieval_error_from_store_propagates_out_of_expand(self):
        """Control: a non-RetrievalError exception is NOT swallowed into an
        error dict — it propagates out of expand() unchanged, as today."""
        docs = [_make_doc(f"d{i}", _ts(i * 10)) for i in range(5)]
        store, expander = self._make_expander({"conversations": docs})
        store._get_by_id_raises["d2"] = ValueError("not a RetrievalError")

        with pytest.raises(ValueError):
            expander.expand("d2", collection="conversations")

    def test_control_genuine_empty_range_uses_fallback_and_is_cached(self):
        """Control: a genuinely empty range still produces today's fallback
        text, and IS cached (unlike a RetrievalError)."""
        convos = [_make_doc(f"c{i}", _ts(100 + i * 10)) for i in range(3)]
        summary = _make_doc(
            "s1", _ts(50),
            "Summary text with a temporal anchor range that has no matching conversations.",
            type="summary",
            temporal_anchor_start=_ts(0), temporal_anchor_end=_ts(20),
        )
        store, expander = self._make_expander({"summaries": [summary], "conversations": convos})

        r1 = expander.expand("s1", collection="summaries")
        assert "no source_doc_ids" in r1["error"].lower()
        r2 = expander.expand("s1", collection="summaries")
        assert r1 is r2  # served from cache

    def test_control_healthy_expansion_is_cached(self):
        """Control: a healthy expansion is cached exactly as before."""
        docs = [_make_doc(f"d{i}", _ts(i * 10)) for i in range(5)]
        store, expander = self._make_expander({"conversations": docs})

        r1 = expander.expand("d2", window=1, collection="conversations")
        assert r1["error"] is None
        r2 = expander.expand("d2", window=1, collection="conversations")
        assert r1 is r2
