"""Deployed-function tests for CGR-20260913-007 anchor #80
(`_get_wiki_content_timed`'s fallback outer except, dm18) and its two
BC-58 siblings: `_get_semantic_chunks_timed` (never reads the FAISS
producer's F4 `.status` before its falsy check) and
`SemanticSearchIndex.search`'s total row-read failure (F4 parent-review
follow-up; `_read_rows` stays UNCHANGED -- only `search`'s post-`_read_rows`
check changes). F7b, docs/execution/generalization/failure_outcome_design.md,
"F7 split and gatherer outcome shape".

Drives the DEPLOYED `_get_wiki_content`/`_get_semantic_chunks` wrappers and
`SemanticSearchIndex.search`, reusing fakes/techniques from
test_sep09_latency_metrics.py (semaphore monkeypatching, `sys.modules`
stubs for the lazy WikiManager/wiki_tracker imports, `_timing_line`),
test_audit0831_fixes.py's `TestWikiTimeoutSkip`, and
test_semantic_visual_failure_outcomes.py's `_ready_index` helper. No real
FAISS index, ChromaDB, embedder, Wikipedia API or network anywhere:
`get_index()` is never called; `SemanticSearchIndex` is instantiated
directly with fakes wired onto it.

FAILING FIRST: run against the UNEDITED core/prompt/gatherer_knowledge.py
and knowledge/semantic_search.py (digests recorded in the same command as
this run); failures are recorded in batches/F7b.md.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

import knowledge.doc_cooccurrence as dc
import knowledge.semantic_search as sem_mod
from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin
from knowledge.semantic_search import SemanticSearchIndex
from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits
from utils.retrieval_outcome import OutcomeList, RetrievalError, outcome_status

# Distinctive markers: a reason must never carry the query or exception text.
PRIVATE_EXC_TEXT = "leaked exception detail F7BQX9"
PRIVATE_QUERY = "sensitive question about TITLEF7BQX9 health and stellar quasars in depth"
Q_MARK = "queryF7BQX9"
A, B = "homeostasisF7BQX9", "thermodynamicsF7BQX9"

# Long enough (>=2 words of >=4 alpha chars) to clear _should_skip_wikipedia,
# and checked against every entry in its conversational_patterns list for a
# hidden substring match (that list is matched by raw substring, not word
# boundary -- "history" contains "hi", "luminous"/"phenomena" contain "no").
WIKI_QUERY = "describe ancient roman aqueduct engineering techniques"
WIKI_TERM = "quasar"


class _TestBoom(RuntimeError):
    """A distinctly-named exception reused at every raising site below, so
    each site's `type(e).__name__` reason is proven, not assumed."""


# --------------------------------------------------------------- fixtures

@pytest.fixture
def gk_module(monkeypatch):
    import core.prompt.gatherer_knowledge as gk

    logger = Mock()
    monkeypatch.setattr(gk, "logger", logger)
    # Full concurrency by default; a test narrows a semaphore to 0 for busy.
    monkeypatch.setattr(gk, "_WIKI_CHROMA_INFLIGHT", threading.Semaphore(2))
    monkeypatch.setattr(gk, "_WIKI_SEM_INFLIGHT", threading.Semaphore(2))
    monkeypatch.setitem(sys.modules, "knowledge.WikiManager", SimpleNamespace(
        _keywords_from_query=Mock(return_value=[WIKI_TERM]),
    ))
    monkeypatch.setitem(sys.modules, "knowledge.wiki_tracker", SimpleNamespace(
        WikiArticleTracker=SimpleNamespace(get_instance=lambda: Mock()),
    ))
    return gk, logger


@pytest.fixture
def gatherer():
    g = KnowledgeRetrievalMixin()
    g.memory_coordinator = SimpleNamespace(chroma_store=None)
    return g


def _timing_line(logger, task):
    calls = [c for c in logger.debug.call_args_list
              if c.args and c.args[0] == f"[WikiTiming] task={task} %s"]
    assert len(calls) == 1
    return calls[0].args[1]


# ---- Shared across both legs: the busy guard and the timeout guard ----

@pytest.mark.asyncio
@pytest.mark.parametrize("leg", ["wiki", "semantic"])
async def test_busy_inflight_guard_is_unavailable(gk_module, gatherer, monkeypatch, leg):
    gk, logger = gk_module
    if leg == "wiki":
        gatherer.memory_coordinator.chroma_store = SimpleNamespace(
            collections={"wiki_knowledge": Mock()})
        monkeypatch.setattr(gk, "_WIKI_CHROMA_INFLIGHT", threading.Semaphore(0))
        result = await gatherer._get_wiki_content(WIKI_QUERY)
    else:
        monkeypatch.setattr(gk, "_WIKI_SEM_INFLIGHT", threading.Semaphore(0))
        result = await gatherer._get_semantic_chunks(WIKI_QUERY)

    assert isinstance(result, OutcomeList)
    assert outcome_status(result) == ("unavailable", "in_flight")
    assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize("leg", ["wiki", "semantic"])
async def test_timeout_is_unavailable_and_timed_out_recorded(gk_module, gatherer, monkeypatch, leg):
    gk, logger = gk_module
    release = threading.Event()

    def blocked(*args, **kwargs):
        release.wait(2)
        return []

    if leg == "wiki":
        coll = Mock()
        coll.count.return_value = 1
        gatherer.memory_coordinator.chroma_store = SimpleNamespace(
            collections={"wiki_knowledge": coll}, query_collection=blocked)
        monkeypatch.setattr(gk, "WIKI_CHROMA_TIMEOUT_S", 0.01)
        call = gatherer._get_wiki_content
    else:
        monkeypatch.setattr(gk, "semantic_search_with_neighbors", blocked)
        monkeypatch.setattr(gk, "SEM_TIMEOUT_S", 0.01)
        call = gatherer._get_semantic_chunks
    try:
        result = await call(WIKI_QUERY)
        assert isinstance(result, OutcomeList)
        assert outcome_status(result) == ("unavailable", "timeout")
        assert _timing_line(logger, leg)["timed_out"] is True
    finally:
        release.set()
        await asyncio.sleep(0.6)


# ---- Contract point 1: `_get_wiki_content_timed` via `_get_wiki_content` --

class TestWikiOutcomes:
    @pytest.mark.asyncio
    async def test_fallback_raising_is_failed_with_exception_class(self, gk_module, gatherer):
        # No chroma -- straight to the live fallback, whose snippet fetch
        # raises. ANCHOR #80's site. Privacy: marker query + exception text.
        gk, logger = gk_module
        gatherer._get_wiki_snippet_cached = AsyncMock(
            side_effect=_TestBoom(PRIVATE_EXC_TEXT))

        result = await gatherer._get_wiki_content(PRIVATE_QUERY)

        assert isinstance(result, OutcomeList)
        assert outcome_status(result) == ("failed", "_TestBoom")
        assert result == []
        assert PRIVATE_EXC_TEXT not in result.reason
        assert PRIVATE_QUERY not in result.reason

    @pytest.mark.asyncio
    @pytest.mark.parametrize("snippet,expected", [
        (None, ("failed", "chroma:_TestBoom")),
        ({"content": "A stellar object."}, ("succeeded", "")),
    ])
    async def test_chroma_raises_fallback_outcome(self, gk_module, gatherer, snippet, expected):
        gk, logger = gk_module
        coll = Mock()
        coll.count.return_value = 1
        gatherer.memory_coordinator.chroma_store = SimpleNamespace(
            collections={"wiki_knowledge": coll},
            query_collection=Mock(side_effect=_TestBoom(PRIVATE_EXC_TEXT)))
        gatherer._get_wiki_snippet_cached = AsyncMock(return_value=snippet)

        result = await gatherer._get_wiki_content(PRIVATE_QUERY)

        assert outcome_status(result) == expected
        if snippet is None:
            assert result == []
            assert PRIVATE_EXC_TEXT not in result.reason
            assert PRIVATE_QUERY not in result.reason
        else:
            assert result == [snippet]  # chroma raised but the fallback delivered -- succeeded

    @pytest.mark.asyncio
    async def test_healthy_chroma_returns_todays_results_control(self, gk_module, gatherer):
        gk, logger = gk_module
        coll = Mock()
        coll.count.return_value = 1
        hit = {"content": "Quasars are extremely luminous.",
               "metadata": {"title": "Quasar"}, "relevance_score": 0.9}
        gatherer.memory_coordinator.chroma_store = SimpleNamespace(
            collections={"wiki_knowledge": coll},
            query_collection=Mock(return_value=[hit]))

        result = await gatherer._get_wiki_content(WIKI_QUERY)

        assert outcome_status(result) == ("succeeded", "")
        assert result[0]["content"] == hit["content"]
        assert result[0]["source"] == "wiki_knowledge"


# -- Contract point 2: `_get_semantic_chunks_timed` via `_get_semantic_chunks` --

class TestSemanticOutcomes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("producer,expected", [
        (OutcomeList.failed("StoreDown"), ("failed", "StoreDown")),
        (OutcomeList.unavailable("index_not_loaded"), ("unavailable", "index_not_loaded")),
    ])
    async def test_producer_status_is_read_before_falsy_check(
        self, gk_module, gatherer, monkeypatch, producer, expected
    ):
        gk, logger = gk_module
        monkeypatch.setattr(gk, "semantic_search_with_neighbors", Mock(return_value=producer))

        result = await gatherer._get_semantic_chunks(WIKI_QUERY)

        assert outcome_status(result) == expected
        assert result == []

    @pytest.mark.asyncio
    async def test_unexpected_pipeline_exception_is_failed(self, gk_module, gatherer, monkeypatch):
        # Privacy: marker query + exception text through a non-producer,
        # non-timeout failure site (the disambiguation filter).
        gk, logger = gk_module
        hit = {"title": "Quasar", "content": "Quasars are luminous.",
               "similarity": 0.9, "metadata": {}}
        monkeypatch.setattr(gk, "semantic_search_with_neighbors", Mock(return_value=[hit]))
        monkeypatch.setattr(gk, "looks_like_disambiguation_text",
                             Mock(side_effect=_TestBoom(PRIVATE_EXC_TEXT)))

        result = await gatherer._get_semantic_chunks(PRIVATE_QUERY)

        assert outcome_status(result) == ("failed", "_TestBoom")
        assert result == []
        assert PRIVATE_EXC_TEXT not in result.reason
        assert PRIVATE_QUERY not in result.reason

    @pytest.mark.asyncio
    @pytest.mark.parametrize("similarity,expected_status", [(0.9, "succeeded"), (0.01, "no_results")])
    async def test_healthy_threshold_gate(
        self, gk_module, gatherer, monkeypatch, similarity, expected_status
    ):
        gk, logger = gk_module
        hit = {"title": "Quasar", "content": "Quasars are luminous.",
               "similarity": similarity, "metadata": {}}
        monkeypatch.setattr(gk, "semantic_search_with_neighbors", Mock(return_value=[hit]))

        result = await gatherer._get_semantic_chunks(WIKI_QUERY)

        assert outcome_status(result) == (expected_status, "")
        if expected_status == "succeeded":
            assert result[0]["title"] == "Quasar"
        else:
            assert result == []


# -- Contract point 3: `SemanticSearchIndex.search`'s total row-read failure
# (F4 parent-review follow-up). `_read_rows` stays UNCHANGED (ownership):
# these tests drive it for real through a fake `_pq_file.read_row_group`
# that raises, rather than mocking `_read_rows` itself.

class _FakeTable:
    def __init__(self, data):
        self._data = data

    def column(self, name):
        return [SimpleNamespace(as_py=lambda v=v: v) for v in self._data[name]]


class _FakePQFile:
    """Fakes just enough of pyarrow.parquet.ParquetFile for the real,
    unchanged `_read_rows` to drive `search`'s row-read-failure logic."""

    def __init__(self, row_groups, raise_groups=()):
        # row_groups: {rg_idx: {local_offset: {col: value}}}
        self._row_groups = row_groups
        self._raise_groups = set(raise_groups)
        cols = {c for rows in row_groups.values() for row in rows.values() for c in row}
        self.schema_arrow = SimpleNamespace(names=sorted(cols))

    def read_row_group(self, rg_idx, columns):
        if rg_idx in self._raise_groups:
            raise _TestBoom(f"row group {rg_idx} unreadable")
        rows = self._row_groups[rg_idx]
        width = max(rows.keys()) + 1
        return _FakeTable(
            {col: [rows.get(i, {}).get(col) for i in range(width)] for col in columns}
        )


def _ready_index(**attrs) -> SemanticSearchIndex:
    """A SemanticSearchIndex that skips load() (already 'loaded') with a
    fake encoder -- never touches the real multi-GB index/parquet."""
    idx = SemanticSearchIndex()
    idx.loaded = True
    idx._encode_query = lambda q: np.zeros((1, 4), dtype=np.float32)
    for k, v in attrs.items():
        setattr(idx, k, v)
    return idx


def _faiss_hits(*index_score_pairs):
    idxs = [p[0] for p in index_score_pairs]
    scores = [p[1] for p in index_score_pairs]
    return (np.array([scores], dtype=np.float32), np.array([idxs], dtype=np.int64))


class TestSemanticSearchIndexRowReadFailure:
    @pytest.mark.parametrize("raise_groups,rg_offsets,row_groups,expected", [
        # All hits land in the one row group that raises -> total failure.
        ({0}, [0],
         {0: {0: {"text": "doc0", "title": "t0"}, 1: {"text": "doc1", "title": "t1"}}},
         ("failed", "row_read_failed", 0)),
        # Hits split across two row groups; only the second raises -> the
        # readable rows still come back, succeeded (control).
        ({1}, [0, 1],
         {0: {0: {"text": "doc0", "title": "t0"}}, 1: {0: {"text": "doc1", "title": "t1"}}},
         ("succeeded", "", 1)),
    ])
    def test_total_vs_partial_row_read_failure(self, raise_groups, rg_offsets, row_groups, expected):
        # Privacy: a marker query proves a constant/empty reason never echoes it.
        pq = _FakePQFile(row_groups=row_groups, raise_groups=raise_groups)
        idx = _ready_index(
            index=SimpleNamespace(search=lambda q, k: _faiss_hits((0, 0.9), (1, 0.8))),
            _pq_file=pq, _rg_offsets=rg_offsets, _total_rows=2,
        )

        result = idx.search(PRIVATE_QUERY, k=3)

        status, reason, length = expected
        assert isinstance(result, OutcomeList)
        assert result.status == status and result.reason == reason
        assert len(result) == length
        assert PRIVATE_QUERY not in result.reason
        if status == "succeeded":
            assert result[0]["title"] == "t0"

    def test_pq_file_none_with_hits_is_unavailable_metadata_unavailable(self):
        idx = _ready_index(
            index=SimpleNamespace(search=lambda q, k: _faiss_hits((0, 0.9))),
            _pq_file=None, _total_rows=2,
        )

        result = idx.search(Q_MARK, k=3)

        assert result.status == "unavailable" and result.reason == "metadata_unavailable"
        assert result == []

    def test_end_to_end_doc_cooccurrence_raises_semantic_failed(self, monkeypatch):
        pq = _FakePQFile(
            row_groups={0: {0: {"text": "doc0", "title": "t0"},
                            1: {"text": "doc1", "title": "t1"}}},
            raise_groups={0},
        )
        fake_idx = _ready_index(
            index=SimpleNamespace(search=lambda q, k: _faiss_hits((0, 0.9), (1, 0.8))),
            _pq_file=pq, _rg_offsets=[0], _total_rows=2,
        )
        monkeypatch.setattr(sem_mod, "get_index", lambda: fake_idx)

        with pytest.raises(RetrievalError) as exc_info:
            dc.doc_cooccurrence(A, B, depth=5)

        assert exc_info.value.reason == "semantic_failed"
        assert exc_info.value.source == "doc_cooccurrence"


# ------------------------------------- Through the deployed builder (F5) --

class TestThroughBuilder:
    @pytest.mark.asyncio
    async def test_busy_semantic_slot_marks_section_unavailable_in_flight(self, monkeypatch):
        import core.prompt.gatherer_knowledge as gk
        monkeypatch.setattr(gk, "_WIKI_SEM_INFLIGHT", threading.Semaphore(0))

        builder = full_builder(
            monkeypatch, [{"query": "hi", "response": "hello"}], budget=10000)
        semantic_gatherer = KnowledgeRetrievalMixin()
        builder.context_gatherer._get_semantic_chunks = semantic_gatherer._get_semantic_chunks

        overrides = {**retrieval_limits(), "max_semantic": 5}
        # A short 2-word query (e.g. "Synthetic question") reads as a bare
        # fragment continuation (utils/query_checker.is_fragment_continuation,
        # max_words=4) and builder.py suppresses eff_max_semantic to 0 for
        # those -- no "semantic" task, no section outcome. This query is
        # long and question-shaped enough to clear that gate.
        result = await builder.build_prompt(
            "What is the history of ancient Roman aqueduct engineering?",
            retrieval_overrides=overrides)

        outcomes = result["_section_outcomes"]
        assert outcomes["semantic"] == {"status": "unavailable", "reason": "in_flight"}
        assert result["semantic_chunks"] == []
