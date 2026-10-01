"""
tests/unit/test_sep27b_B6.py

Lane B batch B6 (2026-09-27, class BC-21 datetime convention / BC-58 sibling
gap): four bare `.replace(tzinfo=None)` timestamp strips migrated onto the
shared `utils.date_coerce.to_naive_local` convention (convert-THEN-strip, so
an aware value is read at its correct local wall-clock time rather than
misread by the raw UTC offset), plus the dead `get_last_summary_meta` branch
in `memory/memory_consolidator.py::maybe_consolidate` removed (the method was
never implemented on any CorpusManager, so `getattr(..., lambda: None)()`
always fell through to the default — the branch never fired on its own, but
silently re-armed the instant anything else attached that attribute name).

Sites covered here: main.py (refresh-narrative corpus-summary fallback),
memory/graph_memory.py (`_edge_is_stale_transient`), memory/pattern_engine.py
(`_parse_ts`), memory/memory_consolidator.py (`maybe_consolidate`).
Debt-file lines in memory_retriever.py/memory_scorer.py/core/orchestrator.py
are explicitly OUT of scope for this batch and untouched.
"""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import memory.relation_classifier as relation_classifier
from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge
from memory.pattern_engine import _parse_ts


# ---------------------------------------------------------------------------
# memory/pattern_engine.py::_parse_ts — aware input must convert to the
# correct local wall-clock reading, not a bare UTC-offset-blind tzinfo strip.
# ---------------------------------------------------------------------------

def _local_naive_equivalent(aware_dt: datetime) -> datetime:
    """Ground truth: the naive local-clock reading of the same real instant."""
    return datetime.fromtimestamp(aware_dt.timestamp())


def test_parse_ts_aware_datetime_object_matches_local_wallclock():
    aware = datetime.now(timezone.utc) - timedelta(hours=3)
    result = _parse_ts(aware)
    assert result is not None
    assert result.tzinfo is None
    assert result == _local_naive_equivalent(aware)


def test_parse_ts_aware_z_string_matches_local_wallclock():
    aware = datetime.now(timezone.utc) - timedelta(hours=3)
    iso_z = aware.isoformat().replace("+00:00", "Z")
    result = _parse_ts(iso_z)
    assert result is not None
    assert result.tzinfo is None
    assert result == _local_naive_equivalent(aware)


def test_parse_ts_naive_input_unchanged():
    naive = datetime.now().replace(microsecond=0)
    assert _parse_ts(naive) == naive
    assert _parse_ts(naive).tzinfo is None


# ---------------------------------------------------------------------------
# memory/graph_memory.py::GraphMemory._edge_is_stale_transient — same
# convert-then-strip requirement, exercised through the deployed staticmethod
# with a real ephemeral relation ("meeting") and its real TTL.
# ---------------------------------------------------------------------------

def _make_edge(last_seen: datetime) -> GraphEdge:
    return GraphEdge(
        source_id="user",
        relation="meeting",
        target_id="dentist",
        first_seen=last_seen,
        last_seen=last_seen,
    )


def test_edge_is_stale_transient_aware_timestamp_uses_local_offset():
    ttl = relation_classifier.ephemeral_ttl_hours("meeting")
    assert ttl is not None and ttl > 0

    offset = datetime.now().astimezone().utcoffset() or timedelta(0)
    offset_hours = offset.total_seconds() / 3600.0
    if offset_hours == 0:
        pytest.skip(
            "host local timezone is UTC — a bare .replace(tzinfo=None) strip "
            "is indistinguishable from the correct conversion here"
        )

    # A bare tzinfo strip is off by exactly the local UTC offset (age_buggy =
    # age_correct + offset_hours). Choose the TRUE age so the two methods
    # land on opposite sides of the TTL boundary — this is the only shape of
    # input that can tell a converting implementation apart from a stripping
    # one via the staticmethod's boolean return.
    if offset_hours < 0:
        # local behind UTC (e.g. US timezones): the bug UNDER-counts age.
        true_age_hours = ttl + (abs(offset_hours) / 2.0)
        expect_stale = True  # correct conversion: truly past TTL
    else:
        # local ahead of UTC: the bug OVER-counts age.
        true_age_hours = max(ttl - (offset_hours / 2.0), 0.1)
        expect_stale = False  # correct conversion: truly within TTL

    aware_last_seen = datetime.now(timezone.utc) - timedelta(hours=true_age_hours)
    edge = _make_edge(aware_last_seen)

    result = GraphMemory._edge_is_stale_transient(edge)
    assert result is expect_stale


def test_edge_is_stale_transient_naive_timestamp_unaffected():
    """Naive last_seen (the common case) must behave exactly as before —
    the fix only changes the aware-input path."""
    ttl = relation_classifier.ephemeral_ttl_hours("meeting")
    fresh = _make_edge(datetime.now() - timedelta(hours=ttl / 2))
    stale = _make_edge(datetime.now() - timedelta(hours=ttl * 2))
    assert GraphMemory._edge_is_stale_transient(fresh) is False
    assert GraphMemory._edge_is_stale_transient(stale) is True


# ---------------------------------------------------------------------------
# main.py refresh-narrative corpus-summary fallback — this code lives inside
# `elif mode == "refresh-narrative":` under `if __name__ == "__main__":`, so
# it is only ever defined at CLI-invocation time and cannot be imported or
# called directly from a unit test. Validated statically against the
# DEPLOYED file's own source (not a reproduction): the nested
# `refresh_narrative_context` function must call the shared `to_naive_local`
# helper and must no longer contain the bare `.replace(tzinfo=None)` strip.
# ---------------------------------------------------------------------------

def _main_py_source() -> str:
    return Path(__file__).resolve().parents[2].joinpath("main.py").read_text()


def _refresh_narrative_segment(source: str) -> str:
    tree = ast.parse(source, filename="main.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "refresh_narrative_context":
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError("refresh_narrative_context not found in main.py")


def test_main_py_imports_to_naive_local():
    source = _main_py_source()
    assert "from utils.date_coerce import to_naive_local" in source


def test_main_py_refresh_narrative_uses_to_naive_local_not_bare_strip():
    segment = _refresh_narrative_segment(_main_py_source())
    assert "to_naive_local(ts)" in segment
    assert "ts = ts.replace(tzinfo=None)" not in segment


# ---------------------------------------------------------------------------
# memory/memory_consolidator.py::maybe_consolidate — the dead
# `get_last_summary_meta` time-throttle branch is gone. Before this fix, any
# corpus_manager exposing that attribute (even incidentally — the method was
# never part of any real CorpusManager implementation) forced an early
# "min gap not elapsed" False regardless of how many exchanges had piled up.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_maybe_consolidate_ignores_incidental_get_last_summary_meta_attribute():
    import tempfile
    from unittest.mock import AsyncMock, Mock

    from memory.corpus_manager import CorpusManager
    from memory.memory_consolidator import MemoryConsolidator

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = CorpusManager(corpus_file=str(Path(tmpdir) / "corpus.json"))
        corpus.get_last_summary_meta = lambda: {"timestamp": datetime.now()}

        for i in range(15):
            corpus.add_entry(f"Q{i}", f"A{i}")

        mm = Mock()
        mm.generate_once = AsyncMock(return_value="Summary of recent conversations")
        consolidator = MemoryConsolidator(consolidation_threshold=10, model_manager=mm)

        result = await consolidator.maybe_consolidate(corpus)

        assert result is True
        mm.generate_once.assert_awaited()


def test_memory_consolidator_no_longer_looks_up_get_last_summary_meta():
    """Structural guard: the removed branch's only functional reference was
    `getattr(corpus_manager, "get_last_summary_meta", ...)` — assert that
    lookup is gone from the deployed module (an explanatory comment naming
    the removed method for future readers is fine; the attribute lookup
    itself is not)."""
    source = Path(__file__).resolve().parents[2].joinpath(
        "memory", "memory_consolidator.py"
    ).read_text()
    assert 'getattr(corpus_manager, "get_last_summary_meta"' not in source
