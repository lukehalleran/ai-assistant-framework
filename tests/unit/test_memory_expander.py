"""
Unit tests for memory/memory_expander.py — MemoryExpander.

Tests cover:
    - Basic window slicing (anchor in middle)
    - Boundary: anchor is first item (only forward expansion)
    - Boundary: anchor is last item (only backward expansion)
    - Same-timestamp tie-breaking via doc_id
    - Cache hit / miss behaviour
    - Unknown doc_id returns error
    - Unknown collection tries all expandable collections
    - Non-expandable collection returns anchor only with message
    - clear_cache() works
    - Summary expansion via temporal anchors (source_docs strategy)
    - Summary expansion via source_doc_ids metadata
    - Summary fallback when no linkage metadata
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Fake ChromaDB store
# ---------------------------------------------------------------------------

class FakeChromaStore:
    """Minimal stand-in for MultiCollectionChromaStore."""

    def __init__(self, data: dict[str, list[dict]] | None = None):
        # data: {collection_name: [doc_dicts]}
        self._data = data or {}

    def get_by_id(self, collection_name: str, doc_id: str):
        for doc in self._data.get(collection_name, []):
            if doc["id"] == doc_id:
                return dict(doc)
        return None

    def list_all(self, collection_name: str):
        return [dict(d) for d in self._data.get(collection_name, [])]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(doc_id: str, ts: str, content: str = "", **extra_meta) -> dict:
    meta = {"timestamp": ts}
    meta.update(extra_meta)
    return {
        "id": doc_id,
        "content": content or f"content of {doc_id}",
        "metadata": meta,
    }


def _ts(offset_minutes: int) -> str:
    """ISO timestamp offset from a base time."""
    base = datetime(2025, 6, 15, 12, 0, 0)
    return (base + timedelta(minutes=offset_minutes)).isoformat()


# ---------------------------------------------------------------------------
# Tests — Temporal Window Expansion (non-summary collections)
# ---------------------------------------------------------------------------

class TestTemporalWindowExpansion:
    """Tests for standard timestamp-window expansion."""

    def _make_expander(self, data):
        store = FakeChromaStore(data)
        from memory.memory_expander import MemoryExpander
        return MemoryExpander(store)

    def test_basic_window_anchor_in_middle(self):
        """Anchor in the middle with window=2 → 5 turns total."""
        docs = [_make_doc(f"d{i}", _ts(i * 10)) for i in range(10)]
        expander = self._make_expander({"conversations": docs})

        result = expander.expand("d5", window=2, collection="conversations")

        assert result["error"] is None
        assert result["collection"] == "conversations"
        assert result["anchor_id"] == "d5"
        assert len(result["turns"]) == 5
        ids = [t["id"] for t in result["turns"]]
        assert ids == ["d3", "d4", "d5", "d6", "d7"]
        anchors = [t for t in result["turns"] if t["is_anchor"]]
        assert len(anchors) == 1
        assert anchors[0]["id"] == "d5"

    def test_boundary_anchor_is_first(self):
        """Anchor at index 0 → only forward expansion."""
        docs = [_make_doc(f"d{i}", _ts(i * 10)) for i in range(5)]
        expander = self._make_expander({"conversations": docs})

        result = expander.expand("d0", window=3, collection="conversations")

        assert result["error"] is None
        ids = [t["id"] for t in result["turns"]]
        assert ids[0] == "d0"
        assert len(ids) == 4  # d0, d1, d2, d3

    def test_boundary_anchor_is_last(self):
        """Anchor at last index → only backward expansion."""
        docs = [_make_doc(f"d{i}", _ts(i * 10)) for i in range(5)]
        expander = self._make_expander({"conversations": docs})

        result = expander.expand("d4", window=3, collection="conversations")

        assert result["error"] is None
        ids = [t["id"] for t in result["turns"]]
        assert ids[-1] == "d4"
        assert len(ids) == 4  # d1, d2, d3, d4

    def test_same_timestamp_tiebreak_by_doc_id(self):
        """When timestamps are identical, sort by doc_id for determinism."""
        same_ts = _ts(0)
        docs = [
            _make_doc("ccc", same_ts),
            _make_doc("aaa", same_ts),
            _make_doc("bbb", same_ts),
        ]
        expander = self._make_expander({"conversations": docs})

        result = expander.expand("bbb", window=5, collection="conversations")

        assert result["error"] is None
        ids = [t["id"] for t in result["turns"]]
        assert ids == ["aaa", "bbb", "ccc"]

    def test_reflections_use_temporal_window(self):
        """Reflections (non-summary expandable) use temporal window."""
        docs = [_make_doc(f"r{i}", _ts(i * 60)) for i in range(5)]
        expander = self._make_expander({"reflections": docs})

        result = expander.expand("r2", window=1, collection="reflections")

        assert result["error"] is None
        assert result["expansion_method"] == "timestamp_window"
        assert len(result["turns"]) == 3
        assert result["total_in_collection"] == 5


# ---------------------------------------------------------------------------
# Tests — Cache
# ---------------------------------------------------------------------------

class TestCache:
    """Tests for result caching."""

    def _make_expander(self, data):
        store = FakeChromaStore(data)
        from memory.memory_expander import MemoryExpander
        return MemoryExpander(store)

    def test_cache_hit(self):
        """Same (id, window, collection) returns cached result."""
        docs = [_make_doc(f"d{i}", _ts(i * 10)) for i in range(5)]
        expander = self._make_expander({"conversations": docs})

        r1 = expander.expand("d2", window=1, collection="conversations")
        r2 = expander.expand("d2", window=1, collection="conversations")
        assert r1 is r2

    def test_cache_miss_different_window(self):
        """Different window → fresh result."""
        docs = [_make_doc(f"d{i}", _ts(i * 10)) for i in range(5)]
        expander = self._make_expander({"conversations": docs})

        r1 = expander.expand("d2", window=1, collection="conversations")
        r2 = expander.expand("d2", window=2, collection="conversations")
        assert r1 is not r2
        assert len(r1["turns"]) < len(r2["turns"])

    def test_clear_cache(self):
        """clear_cache() forces fresh fetch on next call."""
        docs = [_make_doc(f"d{i}", _ts(i * 10)) for i in range(5)]
        expander = self._make_expander({"conversations": docs})

        r1 = expander.expand("d2", window=1, collection="conversations")
        expander.clear_cache()
        r2 = expander.expand("d2", window=1, collection="conversations")
        assert r1 is not r2


# ---------------------------------------------------------------------------
# Tests — Error Cases
# ---------------------------------------------------------------------------

class TestErrorCases:
    """Tests for error handling."""

    def _make_expander(self, data):
        store = FakeChromaStore(data)
        from memory.memory_expander import MemoryExpander
        return MemoryExpander(store)

    def test_unknown_doc_id(self):
        """Unknown ID returns error dict."""
        docs = [_make_doc("d0", _ts(0))]
        expander = self._make_expander({"conversations": docs})

        result = expander.expand("nonexistent", collection="conversations")
        assert result["error"] is not None
        assert "not found" in result["error"].lower()
        assert result["turns"] == []

    def test_unknown_collection_tries_all(self):
        """No collection given → scans all expandable collections."""
        docs = [_make_doc("target", _ts(0), "found it")]
        expander = self._make_expander({"reflections": docs})

        result = expander.expand("target", window=1, collection=None)
        assert result["error"] is None
        assert result["collection"] == "reflections"
        assert len(result["turns"]) == 1

    def test_non_expandable_collection(self):
        """Non-expandable collection returns anchor only with message."""
        docs = [_make_doc("p1", _ts(0), "some proc")]
        expander = self._make_expander({"procedural": docs})

        result = expander.expand("p1", collection="procedural")
        assert result["error"] is not None
        assert "does not support expansion" in result["error"]
        assert len(result["turns"]) == 1
        assert result["turns"][0]["is_anchor"] is True


# ---------------------------------------------------------------------------
# Tests — Content truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    """Tests for content char limits."""

    def _make_expander(self, data):
        store = FakeChromaStore(data)
        from memory.memory_expander import MemoryExpander
        return MemoryExpander(store)

    def test_anchor_uses_anchor_char_limit(self):
        """Anchor content is truncated to EXPAND_ANCHOR_CHAR_LIMIT."""
        long_content = "x" * 1000
        docs = [_make_doc("d0", _ts(0), long_content)]
        expander = self._make_expander({"conversations": docs})

        with patch("memory.memory_expander.cfg") as mock_cfg:
            mock_cfg.EXPAND_MAX_WINDOW = 5
            mock_cfg.EXPAND_ANCHOR_CHAR_LIMIT = 100
            mock_cfg.EXPAND_CONTEXT_CHAR_LIMIT = 50
            result = expander.expand("d0", window=1, collection="conversations")

        anchor_turn = [t for t in result["turns"] if t["is_anchor"]][0]
        assert len(anchor_turn["content"]) <= 104  # 100 + "..."

    def test_context_uses_context_char_limit(self):
        """Non-anchor content is truncated to EXPAND_CONTEXT_CHAR_LIMIT."""
        long_content = "y" * 1000
        docs = [
            _make_doc("d0", _ts(0), long_content),
            _make_doc("d1", _ts(10), "anchor"),
        ]
        expander = self._make_expander({"conversations": docs})

        with patch("memory.memory_expander.cfg") as mock_cfg:
            mock_cfg.EXPAND_MAX_WINDOW = 5
            mock_cfg.EXPAND_ANCHOR_CHAR_LIMIT = 600
            mock_cfg.EXPAND_CONTEXT_CHAR_LIMIT = 50
            result = expander.expand("d1", window=1, collection="conversations")

        context_turn = [t for t in result["turns"] if not t["is_anchor"]][0]
        assert len(context_turn["content"]) <= 54  # 50 + "..."

    def test_window_clamped_to_max(self):
        """Window larger than EXPAND_MAX_WINDOW is clamped."""
        docs = [_make_doc(f"d{i}", _ts(i * 10)) for i in range(20)]
        expander = self._make_expander({"conversations": docs})

        with patch("memory.memory_expander.cfg") as mock_cfg:
            mock_cfg.EXPAND_MAX_WINDOW = 2
            mock_cfg.EXPAND_ANCHOR_CHAR_LIMIT = 600
            mock_cfg.EXPAND_CONTEXT_CHAR_LIMIT = 300
            result = expander.expand("d10", window=100, collection="conversations")

        # window clamped to 2 → max 5 turns (10-2=8 to 10+2=12, inclusive)
        assert len(result["turns"]) == 5


# ---------------------------------------------------------------------------
# Tests — Summary Expansion (source_docs strategy)
# ---------------------------------------------------------------------------

class TestSummaryExpansion:
    """Tests for summary → source conversations drill-down."""

    def _make_expander(self, data):
        store = FakeChromaStore(data)
        from memory.memory_expander import MemoryExpander
        return MemoryExpander(store)

    def test_summary_with_temporal_anchors(self):
        """Summary with temporal_anchor_start/end fetches conversations in range."""
        # 5 conversations spanning 0–40 minutes
        convos = [_make_doc(f"c{i}", _ts(i * 10), f"User: q{i}\nAssistant: a{i}") for i in range(5)]
        # Summary covering the first 3 conversations (0–20 min)
        summary = _make_doc(
            "s1", _ts(50), "Summary of early conversations",
            type="summary",
            temporal_anchor_start=_ts(0),
            temporal_anchor_end=_ts(20),
        )
        expander = self._make_expander({"summaries": [summary], "conversations": convos})

        result = expander.expand("s1", collection="summaries")

        assert result["error"] is None
        assert result["expansion_method"] == "source_docs"
        assert result["collection"] == "summaries"
        # First turn is the summary anchor
        assert result["turns"][0]["is_anchor"] is True
        # Source conversations are c0, c1, c2 (within 0–20 min range)
        source_turns = [t for t in result["turns"] if not t["is_anchor"]]
        source_ids = [t["id"] for t in source_turns]
        assert "c0" in source_ids
        assert "c1" in source_ids
        assert "c2" in source_ids
        assert "c3" not in source_ids  # outside range
        assert "c4" not in source_ids

    def test_summary_with_source_doc_ids(self):
        """Summary with explicit source_doc_ids fetches those docs directly."""
        convos = [_make_doc(f"c{i}", _ts(i * 10), f"convo {i}") for i in range(5)]
        summary = _make_doc(
            "s1", _ts(50), "Summary text",
            type="summary",
            source_doc_ids="c1,c3",
        )
        expander = self._make_expander({"summaries": [summary], "conversations": convos})

        result = expander.expand("s1", collection="summaries")

        assert result["error"] is None
        assert result["expansion_method"] == "source_docs"
        source_turns = [t for t in result["turns"] if not t["is_anchor"]]
        source_ids = [t["id"] for t in source_turns]
        assert source_ids == ["c1", "c3"]

    def test_summary_source_doc_ids_takes_priority(self):
        """source_doc_ids is checked before temporal anchors."""
        convos = [_make_doc(f"c{i}", _ts(i * 10), f"convo {i}") for i in range(5)]
        summary = _make_doc(
            "s1", _ts(50), "Summary text",
            type="summary",
            source_doc_ids="c4",
            temporal_anchor_start=_ts(0),
            temporal_anchor_end=_ts(20),
        )
        expander = self._make_expander({"summaries": [summary], "conversations": convos})

        result = expander.expand("s1", collection="summaries")

        source_turns = [t for t in result["turns"] if not t["is_anchor"]]
        # Should use source_doc_ids (c4), not temporal range (c0-c2)
        assert len(source_turns) == 1
        assert source_turns[0]["id"] == "c4"

    def test_summary_no_linkage_metadata(self):
        """Summary without source_doc_ids or temporal anchors returns error."""
        summary = _make_doc("s1", _ts(0), "Orphan summary", type="summary")
        expander = self._make_expander({"summaries": [summary]})

        result = expander.expand("s1", collection="summaries")

        assert result["error"] is not None
        assert "no source_doc_ids" in result["error"].lower()
        # Still returns the summary anchor turn
        assert len(result["turns"]) == 1
        assert result["turns"][0]["is_anchor"] is True

    def test_summary_temporal_anchors_no_matching_convos(self):
        """Summary with temporal anchors but no conversations in range."""
        convos = [_make_doc(f"c{i}", _ts(100 + i * 10), f"convo {i}") for i in range(3)]
        summary = _make_doc(
            "s1", _ts(50), "Summary text",
            type="summary",
            temporal_anchor_start=_ts(0),
            temporal_anchor_end=_ts(20),
        )
        expander = self._make_expander({"summaries": [summary], "conversations": convos})

        result = expander.expand("s1", collection="summaries")

        # No convos in range → falls through to fallback
        assert result["error"] is not None
        source_turns = [t for t in result["turns"] if not t["is_anchor"]]
        assert len(source_turns) == 0

    def test_summary_source_conversations_sorted_by_time(self):
        """Source conversations are returned in chronological order."""
        convos = [
            _make_doc("c3", _ts(30), "third"),
            _make_doc("c1", _ts(10), "first"),
            _make_doc("c2", _ts(20), "second"),
        ]
        summary = _make_doc(
            "s1", _ts(50), "Summary",
            type="summary",
            temporal_anchor_start=_ts(0),
            temporal_anchor_end=_ts(40),
        )
        expander = self._make_expander({"summaries": [summary], "conversations": convos})

        result = expander.expand("s1", collection="summaries")

        source_turns = [t for t in result["turns"] if not t["is_anchor"]]
        source_ids = [t["id"] for t in source_turns]
        assert source_ids == ["c1", "c2", "c3"]

    def test_summary_expansion_total_in_collection(self):
        """total_in_collection reports number of source conversations found."""
        convos = [_make_doc(f"c{i}", _ts(i * 10), f"convo {i}") for i in range(4)]
        summary = _make_doc(
            "s1", _ts(50), "Summary",
            type="summary",
            temporal_anchor_start=_ts(0),
            temporal_anchor_end=_ts(30),
        )
        expander = self._make_expander({"summaries": [summary], "conversations": convos})

        result = expander.expand("s1", collection="summaries")

        assert result["total_in_collection"] == 4  # all 4 convos in range
