"""Deployed-function tests for CGR-20260913-008 anchors #107-#109.

FAISS `SemanticSearchIndex.search` (#107 FAISS search error, #108 result
assembly error) and `VisualMemoryStore.search_by_text` (#109 ChromaDB text
search error) must return an `OutcomeList` (utils.retrieval_outcome, the F1
leaf) distinguishing failure from an ordinary empty match, matching F2/F3a/
F3b's precedent. `VisualRetriever.retrieve_visual_memories` exposes the
text-search leg's status as two additive dict keys.
`knowledge.doc_cooccurrence.doc_cooccurrence` raises `RetrievalError` when
either wiki leg fails or is unavailable, instead of quietly reading as
`known=False`; `knowledge.literature_oracle`'s existing (untouched, F4
read-only) non-fatal `except` around that call is proven to absorb it.

No real FAISS index, ChromaDB, CLIP model, LLM or network anywhere in this
file: every store/index/model boundary is a fake or a mock.
`knowledge.semantic_search.get_index()` — the real singleton loader that
reads the multi-GB FAISS index and parquet from data/ — is never called;
`get_index` itself is monkeypatched to a fake where its return value matters.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import knowledge.doc_cooccurrence as dc
import knowledge.semantic_search as ss
from knowledge.semantic_search import SemanticSearchIndex, semantic_search_with_neighbors
from knowledge.visual_memory_store import VisualMemoryStore
from knowledge.visual_retrieval import VisualRetriever
from utils.retrieval_outcome import OutcomeList, RetrievalError, outcome_status

# Distinctive markers: proves a `.reason`/log line never carries query, concept,
# caption or exception-message text (privacy contract, point 5).
Q_MARK = "queryEXCZQX9"
A, B = "homeostasisEXCZQX9", "thermodynamicsEXCZQX9"


class _FaissBoom(RuntimeError):
    """Distinguishes the FAISS search-call exception path in assertions."""


class _AssemblyBoom(ValueError):
    """Distinguishes the result-assembly exception path in assertions."""


class _ChromaBoom(RuntimeError):
    """Distinguishes the ChromaDB query_collection exception path in assertions."""


def _ready_index(**attrs) -> SemanticSearchIndex:
    """A SemanticSearchIndex that skips load() (already 'loaded') with a fake encoder."""
    idx = SemanticSearchIndex()
    idx.loaded = True
    idx._encode_query = lambda q: np.zeros((1, 4), dtype=np.float32)
    for k, v in attrs.items():
        setattr(idx, k, v)
    return idx


# --------------------------------------------------------- SemanticSearchIndex.search
class TestSemanticSearchIndexSearch:
    def test_not_loaded_returns_unavailable(self):
        idx = SemanticSearchIndex()
        idx.loaded = False
        idx.load = lambda: None  # stays not-loaded, e.g. FAISS missing
        result = idx.search(Q_MARK, k=3)
        assert isinstance(result, OutcomeList)
        assert result.status == "unavailable" and result.reason == "index_not_loaded"
        assert result == []

    def test_index_search_raises_returns_failed_with_exception_class_name(self):
        idx = _ready_index(index=MagicMock())
        idx.index.search.side_effect = _FaissBoom("faiss internals exploded")
        result = idx.search(Q_MARK, k=3)
        assert result.status == "failed" and result.reason == "_FaissBoom"
        assert "exploded" not in result.reason

    def test_row_to_result_raises_returns_failed(self):
        D, I = np.array([[0.5]], dtype=np.float32), np.array([[0]], dtype=np.int64)
        idx = _ready_index(
            index=MagicMock(search=MagicMock(return_value=(D, I))),
            _total_rows=10,
            _read_rows=lambda indices, columns=None: {0: {"text": "doc0", "title": "t0"}},
            _row_to_result=MagicMock(side_effect=_AssemblyBoom("assembly exploded")),
        )
        result = idx.search(Q_MARK, k=3)
        assert result.status == "failed" and result.reason == "_AssemblyBoom"
        assert "exploded" not in result.reason

    def test_hits_return_succeeded_with_rows(self):
        D, I = np.array([[0.5]], dtype=np.float32), np.array([[0]], dtype=np.int64)
        idx = _ready_index(
            index=MagicMock(search=MagicMock(return_value=(D, I))),
            _total_rows=10,
            _read_rows=lambda indices, columns=None: {0: {"text": "doc0", "title": "t0"}},
        )
        result = idx.search(Q_MARK, k=3)
        assert result.status == "succeeded" and result.reason == ""
        assert len(result) == 1 and result[0]["title"] == "t0"

    def test_no_hits_returns_no_results(self):
        D, I = np.array([[0.1]], dtype=np.float32), np.array([[-1]], dtype=np.int64)
        idx = _ready_index(index=MagicMock(search=MagicMock(return_value=(D, I))), _total_rows=10)
        result = idx.search(Q_MARK, k=3)
        assert result.status == "no_results" and result == []

    def test_empty_query_returns_no_results(self):
        result = SemanticSearchIndex().search("", k=3)
        assert isinstance(result, OutcomeList)
        assert result.status == "no_results" and result == []


# ----------------------------------------------------- semantic_search_with_neighbors
class TestSemanticSearchWithNeighbors:
    def test_passes_failed_status_through(self, monkeypatch):
        idx = _ready_index(index=MagicMock())
        idx.index.search.side_effect = _FaissBoom("down")
        monkeypatch.setattr(ss, "get_index", lambda: idx)
        status, reason = outcome_status(semantic_search_with_neighbors(Q_MARK, k=5))
        assert (status, reason) == ("failed", "_FaissBoom")

    def test_passes_succeeded_status_through(self, monkeypatch):
        D, I = np.array([[0.5]], dtype=np.float32), np.array([[0]], dtype=np.int64)
        idx = _ready_index(
            index=MagicMock(search=MagicMock(return_value=(D, I))),
            _total_rows=10,
            _read_rows=lambda indices, columns=None: {0: {"text": "doc0", "title": "t0"}},
        )
        monkeypatch.setattr(ss, "get_index", lambda: idx)
        result = semantic_search_with_neighbors(Q_MARK, k=5)
        assert outcome_status(result) == ("succeeded", "")


# --------------------------------------------------- VisualMemoryStore.search_by_text
class TestSearchByText:
    def test_chroma_none_returns_unavailable(self, tmp_path):
        store = VisualMemoryStore(chroma_store=None, data_dir=str(tmp_path))
        result = store.search_by_text(Q_MARK, k=5)
        assert result.status == "unavailable" and result.reason == "chroma_unavailable"
        assert result == []

    def test_query_collection_raises_returns_failed(self, tmp_path):
        chroma = MagicMock(query_collection=MagicMock(side_effect=_ChromaBoom("boom")))
        store = VisualMemoryStore(chroma_store=chroma, data_dir=str(tmp_path))
        result = store.search_by_text(Q_MARK, k=5)
        assert result.status == "failed" and result.reason == "_ChromaBoom"
        assert "boom" not in result.reason

    def test_results_return_succeeded(self, tmp_path):
        chroma = MagicMock(query_collection=MagicMock(return_value=[
            {"id": "d1", "content": "a cat", "metadata": {"image_path": "/cat.jpg"},
             "relevance_score": 0.7}
        ]))
        store = VisualMemoryStore(chroma_store=chroma, data_dir=str(tmp_path))
        result = store.search_by_text(Q_MARK, k=5)
        assert result.status == "succeeded" and len(result) == 1

    def test_empty_results_return_no_results(self, tmp_path):
        chroma = MagicMock(query_collection=MagicMock(return_value=[]))
        store = VisualMemoryStore(chroma_store=chroma, data_dir=str(tmp_path))
        result = store.search_by_text(Q_MARK, k=5)
        assert result.status == "no_results" and result == []


# --------------------------------------------- VisualRetriever.retrieve_visual_memories
class _FakeClipNone:
    """CLIP manager stand-in: no embedding -> _search_clip short-circuits to []."""

    def encode_text(self, query):
        return None


class _FakeVisualStore:
    def __init__(self, text_result):
        self._text_result = text_result

    def search_by_text(self, query, k=5):
        return self._text_result

    def search_by_clip(self, query_embedding, k=5):
        return []

    def get_by_entity(self, entity_ids):
        return []


class TestRetrieveVisualMemories:
    async def test_failed_text_search_surfaces_status_keys(self):
        store = _FakeVisualStore(OutcomeList.failed("_ChromaBoom"))
        retriever = VisualRetriever(_FakeClipNone(), store)
        result = await retriever.retrieve_visual_memories(Q_MARK, k=3)
        assert result["text_search_status"] == "failed"
        assert result["text_search_reason"] == "_ChromaBoom"
        # existing keys unchanged
        assert result["text_results"] == [] and result["images"] == []

    async def test_healthy_text_search_status_succeeded(self):
        item = {"doc_id": "d1", "image_path": "/cat.jpg", "caption": "a cat", "source": "s",
                "entity_ids": [], "media_type": "", "timestamp": "", "score": 0.9}
        store = _FakeVisualStore(OutcomeList([item]))
        retriever = VisualRetriever(_FakeClipNone(), store)
        result = await retriever.retrieve_visual_memories(Q_MARK, k=3)
        assert result["text_search_status"] == "succeeded"
        assert result["text_search_reason"] == ""
        assert result["text_results"][0]["caption"] == "a cat"

    async def test_clip_failure_is_not_swallowed(self):
        """Sibling determination: unlike search_by_text, _search_clip/search_by_clip
        have no try/except of their own — a raise propagates straight out of
        retrieve_visual_memories instead of degrading to an empty result."""

        class _FakeClipRaises:
            def encode_text(self, query):
                raise RuntimeError("clip down")

        store = _FakeVisualStore(OutcomeList([]))
        retriever = VisualRetriever(_FakeClipRaises(), store)
        with pytest.raises(RuntimeError):
            await retriever.retrieve_visual_memories(Q_MARK, k=3)


# --------------------------------------------------------------- doc_cooccurrence
def _patch_legs(monkeypatch, ra, rb):
    monkeypatch.setattr(dc, "semantic_search_with_neighbors",
                         lambda q, k=40: ra if q == A else rb)


class TestDocCooccurrence:
    def test_failed_leg_raises_retrieval_error(self, monkeypatch):
        _patch_legs(monkeypatch, OutcomeList.failed("_FaissBoom"), OutcomeList([]))
        with pytest.raises(RetrievalError) as ei:
            dc.doc_cooccurrence(A, B)
        assert ei.value.source == "doc_cooccurrence"
        assert ei.value.reason == "semantic_failed"

    def test_unavailable_leg_raises_retrieval_error(self, monkeypatch):
        _patch_legs(monkeypatch, OutcomeList([]), OutcomeList.unavailable("index_not_loaded"))
        with pytest.raises(RetrievalError) as ei:
            dc.doc_cooccurrence(A, B)
        assert ei.value.reason == "semantic_unavailable"

    def test_failed_outranks_unavailable_in_reason(self, monkeypatch):
        _patch_legs(monkeypatch, OutcomeList.unavailable("index_not_loaded"),
                    OutcomeList.failed("_FaissBoom"))
        with pytest.raises(RetrievalError) as ei:
            dc.doc_cooccurrence(A, B)
        assert ei.value.reason == "semantic_failed"  # worse leg status only

    def test_both_legs_no_results_control(self, monkeypatch):
        _patch_legs(monkeypatch, OutcomeList([]), OutcomeList([]))
        assert dc.doc_cooccurrence(A, B) == dc.DocCooccurResult(0, [], False, False)

    def test_shared_titles_known_true_control(self, monkeypatch):
        item = {"title": "Shared Title", "content": "irrelevant"}
        _patch_legs(monkeypatch, OutcomeList([item]), OutcomeList([item]))
        result = dc.doc_cooccurrence(A, B)
        assert result.known is True and result.shared == 1

    def test_is_known_inherits_the_raise(self, monkeypatch):
        _patch_legs(monkeypatch, OutcomeList.failed("_FaissBoom"), OutcomeList([]))
        with pytest.raises(RetrievalError):
            dc.is_known(A, B)


# ------------------------------------------------------------ privacy (no leaked text)
class TestPrivacyNoLeakedText:
    def test_semantic_search_index_reason_has_no_query_or_exception_text(self):
        idx = _ready_index(index=MagicMock())
        idx.index.search.side_effect = _FaissBoom("EXCZQX9 leak")
        result = idx.search(Q_MARK, k=3)
        assert "EXCZQX9" not in result.reason

    def test_search_by_text_reason_has_no_query_or_exception_text(self, tmp_path):
        chroma = MagicMock(query_collection=MagicMock(side_effect=_ChromaBoom("EXCZQX9 leak")))
        store = VisualMemoryStore(chroma_store=chroma, data_dir=str(tmp_path))
        result = store.search_by_text(Q_MARK, k=3)
        assert "EXCZQX9" not in result.reason

    def test_doc_cooccurrence_raise_reason_has_no_concept_text(self, monkeypatch):
        _patch_legs(monkeypatch, OutcomeList.failed("_FaissBoom"), OutcomeList([]))
        with pytest.raises(RetrievalError) as ei:
            dc.doc_cooccurrence(A, B)
        assert A not in ei.value.reason and B not in ei.value.reason


# --------------------------------------------- literature_oracle deployed consumer
class TestLiteratureOracleConsumer:
    """Read-only deployed-consumer drive of knowledge/literature_oracle.py:738-746
    (untouched, read-only per ownership). Proves the existing non-fatal
    `except Exception` there absorbs our new RetrievalError exactly like any
    other exception, leaving `aux_doc_cooccurrence` unset instead of a false
    `known=False`. Fully offline: model_manager.generate_once and
    retrieve_arxiv are mocked/stubbed; web_search_manager=None short-circuits
    retrieve_tavily with no network."""

    @staticmethod
    def _mk_oracle(**kw):
        from knowledge.literature_oracle import LiteratureOracle
        query_json = json.dumps(
            {"queries": ["q1 xyz", "q2 xyz"], "canonical_idx": 0, "relationship_idx": 1})
        adj_json = json.dumps({"verdict": "NOT_FOUND", "name": "", "citations": [],
                                "prior_verdict": "", "prior_citation_claim": "",
                                "reasoning": "because"})
        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=[query_json, adj_json])
        oracle = LiteratureOracle(mm, web_search_manager=None, model_name="test-model",
                                   escalate=False, enable_aux=True, **kw)
        oracle.retrieve_arxiv = AsyncMock(return_value=[])
        return oracle

    async def test_failed_leg_leaves_aux_none_and_logs_nonfatal(self, caplog):
        oracle = self._mk_oracle()
        err = RetrievalError(source="doc_cooccurrence", reason="semantic_failed")
        with patch("knowledge.doc_cooccurrence.doc_cooccurrence", side_effect=err):
            with caplog.at_level(logging.WARNING):
                rec = await oracle.score_candidate(
                    "c1", "aEXCZQX9", "bEXCZQX9", "claim text", run_id="r1")
        assert rec.aux_doc_cooccurrence is None
        assert any("aux doc_cooccurrence failed" in r.getMessage() for r in caplog.records)
        assert not any("EXCZQX9" in r.getMessage() for r in caplog.records)

    async def test_healthy_leg_populates_aux_control(self):
        from knowledge.doc_cooccurrence import DocCooccurResult
        ok = DocCooccurResult(1, ["shared title"], True, True)
        oracle = self._mk_oracle()
        with patch("knowledge.doc_cooccurrence.doc_cooccurrence", return_value=ok):
            rec = await oracle.score_candidate("c1", "a", "b", "claim", run_id="r1")
        assert rec.aux_doc_cooccurrence == {"shared": 1, "mention": True, "known": True}
