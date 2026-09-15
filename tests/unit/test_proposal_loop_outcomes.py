# tests/unit/test_proposal_loop_outcomes.py
"""Regression tests for CGR-20260913-010 anchor #131's caller completion (F11b-1b).

Completes F11b-1's deferred Part B: ``ShutdownProcessor._generate_proposals``'s
proposal-storage loop and ``agent_branch.proposal_bridge.ingest_survivors`` now
catch the typed ``StoreWriteError`` that ``ProposalStore.store_proposal``
raises (as of F11b-1) per item, instead of letting one failed write abort the
rest of the batch via each caller's pre-existing, unrelated outer exception
handler (F11b-1's disclosed interim window; see
class_guard_responses/CGR-20260913-010-4.md).

Contract (briefs/F11b-1b.md):
- ``_generate_proposals``: a ``failed`` counter beside ``kept``; only a
  ``StoreWriteError`` is caught per item (``failed += 1; continue``); a
  similarity-dedup skip is never counted as failed; the summary log fires on
  ``kept or failed`` and reports both counts (labels only).
- ``ingest_survivors``: a ``StoreWriteError`` per survivor logs one warning
  (labels only) and is added to neither ``seen`` nor ``stored``; the return
  shape (``List[str]``) is unchanged.

Fixtures: ``_make_shutdown_processor`` builds ``ShutdownProcessor`` via
``__new__`` with only the attributes ``_generate_proposals`` /
``_generate_proposals_cold`` read (``model_manager``, ``chroma_store``,
``memory_coordinator``) -- no real corpus, Chroma, model, consolidator or
profile, following tests/unit/test_skill_and_summary_outcomes.py's
established pattern. ``_report()`` is a local, trimmed copy of
tests/agent_branch/test_proposal_bridge.py's ``_report`` fixture shape (that
file is top-level and is never run here; read only, per R_common_rules)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_branch.proposal_bridge import ingest_survivors
from agent_branch.scoring import RankedPortfolio, ScoredBranch
from memory.shutdown_processor import ShutdownProcessor
from utils.retrieval_outcome import StoreWriteError

_MARKER = "SYNTH_MARKER_f11b1b_4e2a91"


def _raising_write_error():
    """A StoreWriteError shaped like memory/proposal_store.py's real raise
    (``raise StoreWriteError(source=..., reason=type(e).__name__) from e``):
    `reason` is a clean type-name label; the marker lives only in the
    underlying (never-logged) exception's message. So a caplog assertion
    that the marker never leaks is a real privacy check on what callers
    log (`e.reason`), not a tautology about what this fixture typed in."""
    underlying = RuntimeError(f"connection reset ({_MARKER})")
    err = StoreWriteError(source="proposal_store", reason=type(underlying).__name__)
    err.__cause__ = underlying
    return err


# ----------------------------------------------------------------------
# ShutdownProcessor._generate_proposals
# ----------------------------------------------------------------------


def _make_session_items(n=3):
    return [
        {"query": f"test query {i}", "response": f"test response {i}"}
        for i in range(n)
    ]


def _make_shutdown_processor(*, model_manager=None, chroma_store=None, memory_coordinator=None):
    sp = ShutdownProcessor.__new__(ShutdownProcessor)
    sp.model_manager = model_manager if model_manager is not None else MagicMock()
    sp.chroma_store = chroma_store if chroma_store is not None else MagicMock()
    sp.memory_coordinator = memory_coordinator
    return sp


class TestShutdownGenerateProposalsBatchLoop:
    @pytest.fixture(autouse=True)
    def _proposals_enabled(self, monkeypatch):
        # Pin the gate ON: these tests exercise _generate_proposals' loop
        # behavior, which must not depend on the owner's live config.yaml
        # toggle (same precaution as test_shutdown_pipeline_proposals.py).
        monkeypatch.setattr("config.app_config.CODE_PROPOSALS_ENABLED", True)

    @pytest.mark.asyncio
    async def test_one_failed_write_is_counted_second_still_stored(self, caplog):
        p1 = MagicMock(title="Proposal A")
        p2 = MagicMock(title="Proposal B")
        mock_gen = MagicMock()
        mock_gen.generate_proposals = AsyncMock(return_value=[p1, p2])

        mock_store = MagicMock()
        mock_store.get_for_dedup.return_value = ""
        mock_store.check_similarity.return_value = None
        mock_store.store_proposal.side_effect = [_raising_write_error(), "doc-2"]

        sp = _make_shutdown_processor()
        with patch("knowledge.proposal_generator.GoalDirectedGenerator", return_value=mock_gen):
            with patch("memory.proposal_store.ProposalStore", return_value=mock_store):
                with caplog.at_level("INFO"):
                    await sp._generate_proposals(_make_session_items())

        # Both proposals were attempted -- the first failure did not abort
        # the loop before the second was tried.
        assert mock_store.store_proposal.call_count == 2
        assert "Generated 1 proposal(s), 1 failed" in caplog.text
        assert _MARKER not in caplog.text

    @pytest.mark.asyncio
    async def test_similarity_skip_control_not_counted_as_failed(self, caplog):
        """Paired control: a dedup skip via check_similarity never reaches
        store_proposal, and is not counted toward `failed`."""
        p1 = MagicMock(title="Proposal A")
        p2 = MagicMock(title="Proposal B")
        mock_gen = MagicMock()
        mock_gen.generate_proposals = AsyncMock(return_value=[p1, p2])

        mock_store = MagicMock()
        mock_store.get_for_dedup.return_value = ""
        mock_store.check_similarity.side_effect = ["existing-doc-1", None]
        mock_store.store_proposal.return_value = "doc-1"

        sp = _make_shutdown_processor()
        with patch("knowledge.proposal_generator.GoalDirectedGenerator", return_value=mock_gen):
            with patch("memory.proposal_store.ProposalStore", return_value=mock_store):
                with caplog.at_level("INFO"):
                    await sp._generate_proposals(_make_session_items())

        assert mock_store.store_proposal.call_count == 1
        assert "Generated 1 proposal(s), 0 failed" in caplog.text

    @pytest.mark.asyncio
    async def test_all_healthy_control_two_kept_zero_failed(self, caplog):
        """Control: every write succeeds -> 2 kept, 0 failed."""
        p1 = MagicMock(title="Proposal A")
        p2 = MagicMock(title="Proposal B")
        mock_gen = MagicMock()
        mock_gen.generate_proposals = AsyncMock(return_value=[p1, p2])

        mock_store = MagicMock()
        mock_store.get_for_dedup.return_value = ""
        mock_store.check_similarity.return_value = None
        mock_store.store_proposal.side_effect = ["doc-1", "doc-2"]

        sp = _make_shutdown_processor()
        with patch("knowledge.proposal_generator.GoalDirectedGenerator", return_value=mock_gen):
            with patch("memory.proposal_store.ProposalStore", return_value=mock_store):
                with caplog.at_level("INFO"):
                    await sp._generate_proposals(_make_session_items())

        assert mock_store.store_proposal.call_count == 2
        assert "Generated 2 proposal(s), 0 failed" in caplog.text


# ----------------------------------------------------------------------
# agent_branch.proposal_bridge.ingest_survivors
# ----------------------------------------------------------------------


def _report(branch_id, *, touched, diff="+x\n"):
    """Trimmed local copy of tests/agent_branch/test_proposal_bridge.py's
    ``_report`` fixture shape -- only the fields
    ``branch_report_to_proposal`` reads."""
    return SimpleNamespace(
        branch_id=branch_id,
        strategy="surgical",
        diff_excerpt=diff,
        static_gate=SimpleNamespace(touched_paths=list(touched), added_lines=1, removed_lines=0),
        run_stats=SimpleNamespace(tokens_spent=10, wallclock_elapsed_s=1.0),
        sandbox_eval=SimpleNamespace(reason="trusted tests passed", branch_evidence=None),
    )


def _portfolio(*ranked):
    return RankedPortfolio(ranked=list(ranked))


class _AlwaysSucceedsStore:
    """ProposalStore-like fake: no real Chroma, model or filesystem."""

    def __init__(self):
        self.chroma_store = None  # _existing_signatures degrades to empty
        self.stored = []

    def store_proposal(self, proposal):
        self.stored.append(proposal)
        return proposal.id


class _RaisingFirstStore:
    """ProposalStore-like fake whose store_proposal raises StoreWriteError on
    the first call only, then succeeds. No real Chroma, model or
    filesystem. ``ingest_survivors`` legitimately logs `e.reason` per
    contract, so the injected error uses `_raising_write_error()`'s
    realistic clean-label shape -- the marker must never reach that log
    line even though `e.reason` itself is logged."""

    def __init__(self):
        self.chroma_store = None
        self.calls = 0
        self.attempts = []

    def store_proposal(self, proposal):
        self.calls += 1
        self.attempts.append(proposal)
        if self.calls == 1:
            raise _raising_write_error()
        return proposal.id


class TestIngestSurvivorsBatchLoop:
    def test_one_failed_write_is_skipped_second_still_stored(self, caplog):
        reports = [
            _report("a", touched=["sandbox/a.py"], diff="+a\n"),
            _report("b", touched=["sandbox/b.py"], diff="+b\n"),
        ]
        portfolio = _portfolio(
            ScoredBranch(branch_id="a", rank=1, survived=True, objective_met=True),
            ScoredBranch(branch_id="b", rank=2, survived=True, objective_met=True),
        )
        store = _RaisingFirstStore()

        with caplog.at_level("WARNING"):
            ids = ingest_survivors("implement subtract", reports, portfolio, store)

        # Both survivors were attempted -- the first failure did not abort
        # the rest of the batch.
        assert store.calls == 2
        # Only the second survivor's id was returned.
        assert ids == [store.attempts[1].id]
        assert _MARKER not in caplog.text

    def test_all_healthy_control_both_ids_returned(self):
        reports = [
            _report("a", touched=["sandbox/a.py"], diff="+a\n"),
            _report("b", touched=["sandbox/b.py"], diff="+b\n"),
        ]
        portfolio = _portfolio(
            ScoredBranch(branch_id="a", rank=1, survived=True, objective_met=True),
            ScoredBranch(branch_id="b", rank=2, survived=True, objective_met=True),
        )
        store = _AlwaysSucceedsStore()

        ids = ingest_survivors("implement subtract", reports, portfolio, store)

        assert len(ids) == 2
        assert set(ids) == {p.id for p in store.stored}
