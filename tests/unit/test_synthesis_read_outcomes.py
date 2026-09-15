"""memory/synthesis_memory.py three read sites raise RetrievalError instead
of silently returning an empty list on failure (CGR-009 #141, #142, #143):
find_similar, get_recurring and get_all_results. A raise (not a typed
OutcomeList) is the correct contract here:

- store_result calls find_similar with no try (memory/synthesis_memory.py
  ~107): a raise leaves BEFORE the insert branch, so a failed similarity
  read can never store a duplicate.
- get_known_connections / get_ungraded / get_graded / get_audit_stats all
  call get_all_results with no try: the raise propagates to their own
  callers instead of being consumed as "no items".
- knowledge/synthesis_filter.py's _stage_4_novelty_internal calls
  find_similar with no try, but process_candidate's own per-stage
  try/except (fail-closed by construction) turns the raise into a labelled
  stage rejection.

Scope note (SIZE split, batches/F12d.md "SIZE checkpoint"): the bundled
shutdown-audit fail-closed change (memory/shutdown_processor.py
_run_synthesis_dreaming) is a separate owner decision, not a CGR-009 scan
anchor, and is deferred to a follow-up batch -- not covered by this file.

Deployed-function tests: a LOCAL copy of tests/unit/test_synthesis_audit.py's
FakeChromaStore (d25b3d549f7a9c7c7705f934c34f5e9651469e1d0aeedfa56fb0bfd5aa0a6abc,
in-file class at line 74) extended with a raising mode and call counters (no
edit to that file).
"""
from unittest.mock import MagicMock

import pytest

from knowledge.synthesis_models import (
    CandidateStatus,
    SynthesisCandidate,
    SynthesisResult,
)
from memory.synthesis_memory import SynthesisMemory
from utils.retrieval_outcome import RetrievalError

MARKER = "F12D_Q8XZK3"


# ---------------------------------------------------------------------------
# Local fixture (a copy of test_synthesis_audit.py's FakeChromaStore, plus a
# raising mode and call counters -- that file is not edited by this batch)
# ---------------------------------------------------------------------------

class FakeChromaStore:
    """Minimal in-memory ChromaDB mock for the synthesis_results collection."""

    def __init__(self, query_raise=None, relevance_score=1.0):
        self._docs = {}
        self._counter = 0
        self._query_raise = query_raise
        self._relevance_score = relevance_score
        self.add_calls = 0
        self.update_calls = 0

    def add_to_collection(self, name, text, metadata):
        self.add_calls += 1
        self._counter += 1
        doc_id = f"synth_{self._counter}"
        self._docs[doc_id] = {"text": text, "metadata": dict(metadata)}
        return doc_id

    def query_collection(self, collection_name, query_text, n_results=10):
        if self._query_raise is not None:
            raise self._query_raise
        items = []
        for doc_id, doc in self._docs.items():
            items.append({
                "id": doc_id,
                "content": doc["text"],
                "metadata": dict(doc["metadata"]),
                "relevance_score": self._relevance_score,  # 1.0 => similarity 1.0
            })
        return items[:n_results]

    def update_metadata(self, collection_name, doc_id, metadata_updates):
        self.update_calls += 1
        if doc_id in self._docs:
            self._docs[doc_id]["metadata"].update(metadata_updates)

    def get_collection_stats(self):
        return {"synthesis_results": {"count": len(self._docs)}}


def _make_candidate(concept_a="my dog", concept_b="Pavlov", claim="conditioning"):
    return SynthesisCandidate(
        concept_a=concept_a,
        concept_b=concept_b,
        connection_claim=claim,
        walk_path=["a", "b"],
        source_domains={"personal", "psychology"},
        endpoint_distance=0.5,
    )


def _make_result(claim="conditioning parallel", status=CandidateStatus.ACCEPTED):
    r = SynthesisResult(candidate=_make_candidate(claim=claim))
    r.status = status
    return r


# ---------------------------------------------------------------------------
# find_similar -- ANCHOR #141
# ---------------------------------------------------------------------------

class TestFindSimilar:
    def test_query_raises_retrieval_error(self):
        store = FakeChromaStore(query_raise=RuntimeError(f"boom {MARKER}"))
        sm = SynthesisMemory(store)
        with pytest.raises(RetrievalError) as exc:
            sm.find_similar("some claim")
        assert exc.value.source == "synthesis_memory"
        assert exc.value.reason == "find_similar:RuntimeError"
        assert MARKER not in str(exc.value)

    def test_empty_collection_returns_empty_control(self):
        store = FakeChromaStore()
        sm = SynthesisMemory(store)
        assert sm.find_similar("some claim") == []

    def test_healthy_returns_match_control(self):
        store = FakeChromaStore(relevance_score=1.0)
        store.add_to_collection("synthesis_results", "existing claim",
                                 _make_result(claim="existing claim").to_metadata())
        sm = SynthesisMemory(store)
        matches = sm.find_similar("existing claim")
        assert len(matches) == 1
        assert matches[0][1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_recurring -- ANCHOR #142
# ---------------------------------------------------------------------------

class TestGetRecurring:
    def test_query_raises_retrieval_error(self):
        store = FakeChromaStore(query_raise=RuntimeError(f"boom {MARKER}"))
        sm = SynthesisMemory(store)
        with pytest.raises(RetrievalError) as exc:
            sm.get_recurring()
        assert exc.value.source == "synthesis_memory"
        assert exc.value.reason == "recurring:RuntimeError"
        assert MARKER not in str(exc.value)

    def test_empty_collection_returns_empty_control(self):
        store = FakeChromaStore()
        sm = SynthesisMemory(store)
        assert sm.get_recurring() == []

    def test_healthy_returns_converging_control(self):
        store = FakeChromaStore()
        r = _make_result(claim="converging claim", status=CandidateStatus.CONVERGING)
        r.unique_paths = {"p1", "p2", "p3"}
        r.unique_sources = {"s1", "s2"}
        store.add_to_collection("synthesis_results", r.candidate.connection_claim, r.to_metadata())
        sm = SynthesisMemory(store)
        recurring = sm.get_recurring()
        assert len(recurring) == 1
        assert recurring[0].candidate.connection_claim == "converging claim"


# ---------------------------------------------------------------------------
# get_all_results -- ANCHOR #143
# ---------------------------------------------------------------------------

class TestGetAllResults:
    def test_query_raises_retrieval_error(self):
        store = FakeChromaStore(query_raise=RuntimeError(f"boom {MARKER}"))
        sm = SynthesisMemory(store)
        with pytest.raises(RetrievalError) as exc:
            sm.get_all_results()
        assert exc.value.source == "synthesis_memory"
        assert exc.value.reason == "all_results:RuntimeError"
        assert MARKER not in str(exc.value)

    def test_empty_collection_returns_empty_control(self):
        store = FakeChromaStore()
        sm = SynthesisMemory(store)
        assert sm.get_all_results() == []

    def test_healthy_returns_results_control(self):
        store = FakeChromaStore()
        r = _make_result(claim="a result")
        store.add_to_collection("synthesis_results", r.candidate.connection_claim, r.to_metadata())
        sm = SynthesisMemory(store)
        results = sm.get_all_results()
        assert len(results) == 1
        assert results[0][1].candidate.connection_claim == "a result"


# ---------------------------------------------------------------------------
# store_result: a failed similarity read must never store a duplicate
# ---------------------------------------------------------------------------

class TestStoreResultNoDuplicateOnFailedSimilarityRead:
    def test_raising_similarity_read_raises_and_never_inserts(self):
        store = FakeChromaStore(query_raise=RuntimeError(f"boom {MARKER}"))
        sm = SynthesisMemory(store)
        with pytest.raises(RetrievalError) as exc:
            sm.store_result(_make_result())
        assert exc.value.reason == "find_similar:RuntimeError"
        assert store.add_calls == 0

    def test_healthy_no_match_inserts_once_control(self):
        store = FakeChromaStore()
        sm = SynthesisMemory(store)
        doc_id = sm.store_result(_make_result(claim="brand new claim"))
        assert doc_id
        assert store.add_calls == 1

    def test_healthy_match_updates_convergence_no_insert_control(self):
        store = FakeChromaStore(relevance_score=1.0)
        sm = SynthesisMemory(store)
        existing = _make_result(claim="shared claim")
        store.add_to_collection("synthesis_results", existing.candidate.connection_claim,
                                 existing.to_metadata())
        assert store.add_calls == 1
        sm.store_result(_make_result(claim="shared claim"))
        assert store.add_calls == 1  # unchanged -- no duplicate insert
        assert store.update_calls == 1  # convergence updated instead


# ---------------------------------------------------------------------------
# Through the deployed filter: SynthesisFilter stage 4 (novelty_internal)
# ---------------------------------------------------------------------------

class TestFilterStage4NoveltyInternal:
    """Drives the REAL process_candidate / _stage_4_novelty_internal with a
    REAL SynthesisMemory backed by the fake store. Stages 0-3 need real
    embeddings/FAISS to exercise meaningfully (out of this batch's scope,
    per the brief's "record read-only evidence instead" fallback), so
    `_stages` is narrowed to just novelty_internal on the instance --
    process_candidate's per-stage try/except (the code path under test) and
    _stage_4_novelty_internal itself are both real and unmocked.
    """

    def _filter(self, store):
        from knowledge.synthesis_filter import SynthesisFilter
        sm = SynthesisMemory(store)
        filt = SynthesisFilter(chroma_store=store, model_manager=MagicMock(), synthesis_memory=sm)
        filt._stages = [("novelty_internal", filt._stage_4_novelty_internal)]
        return filt

    @pytest.mark.asyncio
    async def test_raising_find_similar_rejects_candidate_labels_only(self):
        store = FakeChromaStore(query_raise=RuntimeError(f"boom {MARKER}"))
        filt = self._filter(store)
        result = await filt.process_candidate(_make_candidate())

        assert result.status == CandidateStatus.REJECTED
        assert result.rejection_stage == "novelty_internal"
        assert result.rejection_reason == "Stage error: synthesis_memory: find_similar:RuntimeError"
        assert MARKER not in result.rejection_reason

    @pytest.mark.asyncio
    async def test_healthy_no_match_passes_control(self):
        store = FakeChromaStore()  # empty -- no prior insight to collide with
        filt = self._filter(store)
        result = await filt.process_candidate(_make_candidate())

        assert result.status == CandidateStatus.ACCEPTED
        assert result.stage_results[0].stage_name == "novelty_internal"
        assert result.stage_results[0].passed is True


# ---------------------------------------------------------------------------
# get_all_results consumers propagate its raise (never turned back into an
# empty view)
# ---------------------------------------------------------------------------

class TestConsumersPropagateAllResultsFailure:
    def _raising_memory(self):
        store = FakeChromaStore(query_raise=RuntimeError(f"boom {MARKER}"))
        return SynthesisMemory(store)

    def test_get_known_connections_propagates(self):
        sm = self._raising_memory()
        with pytest.raises(RetrievalError) as exc:
            sm.get_known_connections()
        assert exc.value.reason == "all_results:RuntimeError"
        assert MARKER not in str(exc.value)

    def test_get_ungraded_propagates(self):
        sm = self._raising_memory()
        with pytest.raises(RetrievalError) as exc:
            sm.get_ungraded()
        assert exc.value.reason == "all_results:RuntimeError"

    def test_get_graded_propagates(self):
        sm = self._raising_memory()
        with pytest.raises(RetrievalError) as exc:
            sm.get_graded()
        assert exc.value.reason == "all_results:RuntimeError"

    def test_get_audit_stats_propagates(self):
        sm = self._raising_memory()
        with pytest.raises(RetrievalError) as exc:
            sm.get_audit_stats()
        assert exc.value.reason == "all_results:RuntimeError"
