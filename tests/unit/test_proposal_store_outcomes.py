# tests/unit/test_proposal_store_outcomes.py
"""CGR-20260913-009 (#132 query_proposals, #133 get_proposal, #134 get_pending,
#136 get_pending_and_approved) / CGR-20260913-010 (#131 store_proposal): a
ProposalStore read that fails must raise ``RetrievalError`` instead of returning
an empty list / None indistinguishable from "not found" or "no proposals"; a
write that fails must raise ``StoreWriteError`` instead of returning None
indistinguishable from the deliberate "chroma not configured" skip.

SIZE SPLIT (F11b-1, 2026-09-14): the brief's pre-authorized split ("store +
filter/gatherer evidence vs the two loop callers + CLI") applies — isolating
this batch's own edits from earlier batches' overlapping hunks in
memory/shutdown_processor.py and main.py put the full scope at 508 changed
lines, over the 450 hard cap. This file covers ONLY the ProposalStore fix
(all five methods) plus the filter/gatherer propagation evidence. The two
batch-loop callers (shutdown ``_generate_proposals``, ``agent_branch``
``ingest_survivors``) and the ``main.py`` ``check-proposals`` CLI wrap are
DEFERRED to a follow-up batch — see batches/F11b-1.md "Size and split".

Also covers ``ShutdownProcessor._check_implementation_tracking`` (read-only —
OWNERSHIP leaves this method untouched in both parts of the split): its
existing, unmodified outer ``except Exception`` already catches the new
``RetrievalError`` from ``get_pending_and_approved`` the same way it catches
anything else, so this evidence needs no source change and has no Part B
dependency.

Fakes only: a LOCAL copy of ``test_proposal_store.py``'s
``MockChromaStore``/``MockChromaCollection`` shape (extended with switches to
raise on ``add_to_collection`` / ``list_all`` / ``query_collection`` on
demand), plus the DEPLOYED ``ProposalFilter``, ``KnowledgeRetrievalMixin`` and
``ShutdownProcessor`` driven directly. No real Chroma, embedder, model,
filesystem scan or data path anywhere.
"""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory.code_proposal import CodeProposal, ProposalType
from memory.proposal_store import ProposalStore
from utils.retrieval_outcome import RetrievalError, StoreWriteError, outcome_status

MARKER = "F11BQZ7"  # synthetic marker; must never appear in a NEW error/log


# ---------------------------------------------------------------------------
# Local raising store — LOCAL copy of test_proposal_store.py's MockChromaStore
# / MockChromaCollection shape, extended with switches to raise on demand.
# ---------------------------------------------------------------------------


class _RaisingCollection:
    def __init__(self):
        self._docs = {}

    def count(self):
        return len(self._docs)

    def delete(self, ids=None):
        for doc_id in (ids or []):
            self._docs.pop(doc_id, None)


class _RaisingChromaStore:
    """Mock of MultiCollectionChromaStore, able to raise on add / list / query."""

    def __init__(self):
        self.collections = {"proposals": _RaisingCollection()}
        self.raise_on_add = False
        self.raise_on_list = False
        self.raise_on_query = False

    def create_collection(self, name):
        self.collections[name] = _RaisingCollection()

    def add_to_collection(self, name, text, metadata):
        if self.raise_on_add:
            raise RuntimeError(f"add failed {MARKER}")
        import uuid
        doc_id = str(uuid.uuid4())
        coll = self.collections.get(name)
        if coll:
            coll._docs[doc_id] = {"document": text, "metadata": dict(metadata)}
        return doc_id

    def list_all(self, collection_name):
        if self.raise_on_list:
            raise RuntimeError(f"list_all failed {MARKER}")
        coll = self.collections.get(collection_name)
        if not coll:
            return []
        return [
            {"id": doc_id, "content": data["document"], "metadata": data["metadata"]}
            for doc_id, data in coll._docs.items()
        ]

    def query_collection(self, collection_name, query_text, n_results=5):
        if self.raise_on_query:
            raise RuntimeError(f"query failed {MARKER}")
        items = self.list_all(collection_name)
        results = []
        for i, item in enumerate(items[:n_results]):
            results.append({
                "id": item["id"],
                "content": item["content"],
                "metadata": item["metadata"],
                "relevance_score": 0.9 - (i * 0.1),
                "collection": collection_name,
                "rank": i + 1,
            })
        return results


def _sample_proposal(title="Add caching layer"):
    return CodeProposal(
        title=title, proposal_type=ProposalType.FEATURE, priority=8,
        reasoning="API responses are slow", tags=["performance"],
    )


# ---------------------------------------------------------------------------
# CGR-20260913-010 #131 — store_proposal
# ---------------------------------------------------------------------------


class TestStoreProposalRaises:
    def test_add_raises_raises_store_write_error(self):
        store = ProposalStore(chroma_store=_RaisingChromaStore())
        store.chroma_store.raise_on_add = True
        with pytest.raises(StoreWriteError) as exc:
            store.store_proposal(_sample_proposal())
        assert exc.value.source == "proposal_store"
        assert MARKER not in str(exc.value)
        assert MARKER not in exc.value.reason

    def test_skip_control_returns_none(self):
        """Paired control: the deliberate skip (no chroma) is unchanged."""
        store = ProposalStore(chroma_store=None)
        assert store.store_proposal(_sample_proposal()) is None


# ---------------------------------------------------------------------------
# CGR-20260913-009 #132 — query_proposals
# ---------------------------------------------------------------------------


class TestQueryProposalsRaises:
    def test_query_raises_raises_retrieval_error(self):
        chroma = _RaisingChromaStore()
        store = ProposalStore(chroma_store=chroma)
        store.store_proposal(_sample_proposal())  # non-empty collection
        chroma.raise_on_query = True
        with pytest.raises(RetrievalError) as exc:
            store.query_proposals("caching")
        assert exc.value.reason.startswith("query:")
        assert MARKER not in str(exc.value)

    def test_empty_collection_control_returns_empty_list(self):
        """Paired control: genuine empty (count() == 0) stays []."""
        store = ProposalStore(chroma_store=_RaisingChromaStore())
        assert store.query_proposals("anything") == []

    def test_no_chroma_control_returns_empty_list(self):
        """Paired control: the deliberate skip stays []."""
        store = ProposalStore(chroma_store=None)
        assert store.query_proposals("anything") == []


# ---------------------------------------------------------------------------
# CGR-20260913-009 #133 — get_proposal
# ---------------------------------------------------------------------------


class TestGetProposalRaises:
    def test_list_all_raises_raises_retrieval_error(self):
        chroma = _RaisingChromaStore()
        store = ProposalStore(chroma_store=chroma)
        chroma.raise_on_list = True
        with pytest.raises(RetrievalError) as exc:
            store.get_proposal("some-id")
        assert exc.value.reason.startswith("get:")
        assert MARKER not in str(exc.value)

    def test_corrupt_matching_record_raises_retrieval_error(self):
        """A record that matches by id but fails to deserialize looks like
        'not found' today; it must raise instead of silently vanishing."""
        chroma = _RaisingChromaStore()
        store = ProposalStore(chroma_store=chroma)
        chroma.collections["proposals"]._docs["doc-1"] = {
            "document": "x",
            "metadata": {"proposal_id": "corrupt-1", "steps_json": "{not valid json"},
        }
        with pytest.raises(RetrievalError):
            store.get_proposal("corrupt-1")

    def test_not_found_control_returns_none(self):
        """Paired control: a genuine miss stays None."""
        store = ProposalStore(chroma_store=_RaisingChromaStore())
        assert store.get_proposal("missing") is None

    def test_found_control_returns_proposal(self):
        """Paired control: a healthy read still returns the proposal."""
        store = ProposalStore(chroma_store=_RaisingChromaStore())
        p = _sample_proposal()
        store.store_proposal(p)
        found = store.get_proposal(p.id)
        assert found is not None
        assert found.title == "Add caching layer"


# ---------------------------------------------------------------------------
# CGR-20260913-009 #134 — get_pending / #136 — get_pending_and_approved
# ---------------------------------------------------------------------------


class TestGetPendingRaises:
    def test_list_all_raises_raises_retrieval_error(self):
        chroma = _RaisingChromaStore()
        store = ProposalStore(chroma_store=chroma)
        chroma.raise_on_list = True
        with pytest.raises(RetrievalError) as exc:
            store.get_pending()
        assert exc.value.reason.startswith("pending:")
        assert MARKER not in str(exc.value)

    def test_empty_control_returns_empty_list(self):
        store = ProposalStore(chroma_store=_RaisingChromaStore())
        assert store.get_pending() == []


class TestGetPendingAndApprovedRaises:
    def test_list_all_raises_raises_retrieval_error(self):
        chroma = _RaisingChromaStore()
        store = ProposalStore(chroma_store=chroma)
        chroma.raise_on_list = True
        with pytest.raises(RetrievalError) as exc:
            store.get_pending_and_approved()
        assert exc.value.reason.startswith("pending_approved:")
        assert MARKER not in str(exc.value)

    def test_empty_control_returns_empty_list(self):
        store = ProposalStore(chroma_store=_RaisingChromaStore())
        assert store.get_pending_and_approved() == []


# ---------------------------------------------------------------------------
# Through the deployed filter and gatherer: a query_proposals raise must
# reach get_proposed_features as a typed "failed" outcome, not [] silently.
# ---------------------------------------------------------------------------


class TestFilterAndGathererPropagation:
    @pytest.mark.asyncio
    async def test_query_raise_reaches_gatherer_as_failed(self, monkeypatch):
        from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin
        from core.prompt.proposal_filter import ProposalFilter

        monkeypatch.setattr("config.app_config.CODE_PROPOSALS_PROMPT_ENABLED", True)

        class _RaisingStore:
            def query_proposals(self, *a, **kw):
                raise RetrievalError(source="proposal_store", reason="query:RuntimeError")

        pf = ProposalFilter()
        pf._proposal_store = _RaisingStore()

        class _Host(KnowledgeRetrievalMixin):
            def __init__(self):
                self.memory_id_map = {}
                self._proposal_filter = pf

        host = _Host()
        result = await host.get_proposed_features("refactor the caching pipeline", limit=3)
        assert outcome_status(result) == ("failed", "RetrievalError")
        assert result == []


# ---------------------------------------------------------------------------
# _check_implementation_tracking evidence (read-only region, both parts of
# the split): a typed read failure is logged and the detector is never
# constructed. No source change here or dependency on the deferred Part B.
# ---------------------------------------------------------------------------


def _shutdown_processor():
    from memory.shutdown_processor import ShutdownProcessor
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value="")
    return ShutdownProcessor(
        corpus_manager=MagicMock(),
        chroma_store=MagicMock(),
        consolidator=MagicMock(),
        fact_extractor=MagicMock(),
        model_manager=mm,
        user_profile=MagicMock(),
        storage=MagicMock(),
        session_start=datetime.now(),
        memory_coordinator=None,
    )


class TestCheckImplementationTrackingEvidence:
    @pytest.mark.asyncio
    async def test_pending_and_approved_raise_logs_and_skips_detector(self, monkeypatch, caplog):
        monkeypatch.setattr("config.app_config.IMPL_TRACKING_ENABLED", True)
        monkeypatch.setattr("config.app_config.IMPL_TRACKING_AT_SHUTDOWN", True)

        proc = _shutdown_processor()
        store = MagicMock()
        store.get_pending_and_approved.side_effect = RetrievalError(
            source="proposal_store", reason="pending_approved:RuntimeError"
        )
        with patch("memory.proposal_store.ProposalStore", return_value=store):
            with patch("knowledge.implementation_detector.ImplementationDetector") as mock_det:
                with caplog.at_level("WARNING"):
                    await proc._check_implementation_tracking()
                mock_det.assert_not_called()
        assert "Implementation tracking failed" in caplog.text
