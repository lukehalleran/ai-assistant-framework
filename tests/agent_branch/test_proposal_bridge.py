# tests/agent_branch/test_proposal_bridge.py
"""Tests for the agent_branch -> Proposals-tab bridge (the M3 hand-off).

Pure (no podman): the bridge is a leaf that converts duck-typed BranchReport-like
objects into CodeProposals and ingests ranked survivors into a ProposalStore-like
object. Covers: the conversion mapping (source/status/priority/risk/provenance),
risk classification driving the GUI acknowledge gate, and that ONLY ranked
survivors are ingested (killed/rejected branches never become proposals)."""

from types import SimpleNamespace

import json

from agent_branch.proposal_bridge import branch_report_to_proposal, ingest_survivors
from agent_branch.scoring import RankedPortfolio, ScoredBranch
from memory.code_proposal import ProposalSource, ProposalStatus, RiskLevel


# -- stubs -------------------------------------------------------------------

def _report(branch_id, *, touched, added=10, removed=2, diff="+x\n",
            strategy="surgical", tokens=500, wallclock=12.0,
            proof="trusted tests passed", test_files=None, ev_passed=True):
    evidence = None
    if test_files:
        evidence = SimpleNamespace(passed=ev_passed, test_files=list(test_files))
    return SimpleNamespace(
        branch_id=branch_id,
        strategy=strategy,
        diff_excerpt=diff,
        static_gate=SimpleNamespace(touched_paths=list(touched),
                                    added_lines=added, removed_lines=removed),
        run_stats=SimpleNamespace(tokens_spent=tokens, wallclock_elapsed_s=wallclock),
        sandbox_eval=SimpleNamespace(reason=proof, branch_evidence=evidence),
    )


class _FakeChroma:
    """Minimal chroma_store stub exposing list_all (for content-dedup tests)."""
    def __init__(self, existing_meta=()):
        self._existing = list(existing_meta)  # list of metadata dicts

    def list_all(self, _collection):
        return [{"metadata": m} for m in self._existing]


class _FakeStore:
    """ProposalStore-like: records stores; exposes chroma_store.list_all so the
    bridge's content-signature dedup can see prior runs."""
    def __init__(self, existing_meta=()):
        self.stored = []
        self.chroma_store = _FakeChroma(existing_meta)

    def store_proposal(self, proposal):
        self.stored.append(proposal)
        return proposal.id


# -- conversion mapping ------------------------------------------------------

def test_converter_basic_mapping():
    p = branch_report_to_proposal(
        _report("surgical", touched=["sandbox/calc.py"]),
        objective="implement subtract", rank=1, total_survivors=3,
    )
    assert p.source == ProposalSource.AGENT_BRANCH
    assert p.status == ProposalStatus.PENDING
    assert p.priority == 10                       # rank 1 -> top
    assert p.affected_files == ["sandbox/calc.py"]
    assert "agent_branch" in p.tags and "rank-1" in p.tags
    assert "branch:surgical" in p.tags
    assert "implement subtract" in p.title and "surgical" in p.title
    assert "```diff" in p.description            # supervisor diff embedded
    # a plain sandbox change is neither core nor critical
    assert p.touches_core_system is False
    assert p.risk_level == RiskLevel.MEDIUM


def test_priority_steps_down_with_rank():
    p3 = branch_report_to_proposal(_report("b3", touched=["sandbox/x.py"]),
                                   objective="o", rank=3, total_survivors=3)
    assert p3.priority == 8                       # 11 - 3


def test_core_touch_is_high_and_flagged_for_ack():
    p = branch_report_to_proposal(
        _report("b", touched=["core/orchestrator.py"]),
        objective="touch the spine", rank=1,
    )
    assert p.touches_core_system is True
    assert p.risk_level == RiskLevel.HIGH


def test_safety_touch_is_critical():
    p = branch_report_to_proposal(
        _report("b", touched=["utils/python_fs_guard.py"]),
        objective="weaken a guard", rank=1,
    )
    assert p.touches_core_system is True
    assert p.risk_level == RiskLevel.CRITICAL


def test_import_in_diff_trips_core_even_on_clean_path():
    rep = _report("b", touched=["plugins/helper.py"],
                  diff="+from core.orchestrator import Orchestrator\n")
    p = branch_report_to_proposal(rep, objective="sneaky", rank=1)
    assert p.touches_core_system is True
    assert p.risk_level == RiskLevel.HIGH


def test_branch_evidence_surfaced():
    rep = _report("b", touched=["sandbox/x.py"],
                  test_files=["sandbox/test_mine.py"], ev_passed=True)
    p = branch_report_to_proposal(rep, objective="o", rank=1)
    assert "sandbox/test_mine.py" in p.test_files
    assert "evidence" in p.description.lower()


def test_empty_objective_gets_fallback_title():
    p = branch_report_to_proposal(_report("b", touched=["sandbox/x.py"], strategy=""),
                                  objective="", rank=1)
    assert "agent-branch change" in p.title


# -- ingest: only ranked survivors become proposals --------------------------

def _portfolio(*ranked):
    return RankedPortfolio(ranked=list(ranked))


def test_ingest_stores_only_ranked_survivors():
    reports = [
        _report("surgical", touched=["sandbox/a.py"]),
        _report("refactor", touched=["sandbox/b.py"]),
        _report("saboteur", touched=["config/defaults.py"]),  # killed — not ranked
    ]
    portfolio = _portfolio(
        ScoredBranch(branch_id="surgical", rank=1, survived=True, objective_met=True),
        ScoredBranch(branch_id="refactor", rank=2, survived=True, objective_met=True),
    )
    store = _FakeStore()
    ids = ingest_survivors("implement subtract", reports, portfolio, store)

    assert len(ids) == 2
    def _branch_tag(p):
        return next(t for t in p.tags if t.startswith("branch:"))
    stored_branches = {_branch_tag(p) for p in store.stored}
    assert stored_branches == {"branch:surgical", "branch:refactor"}
    # the killed saboteur never became a proposal
    assert "branch:saboteur" not in stored_branches
    # rank -> priority ordering preserved
    by_branch = {_branch_tag(p): p.priority for p in store.stored}
    assert by_branch["branch:surgical"] == 10 and by_branch["branch:refactor"] == 9


def test_divergent_survivors_same_files_different_diff_all_kept():
    # The whole point: two survivors touching the SAME file with DIFFERENT diffs
    # are distinct candidates — both must be kept (semantic/title dedup would have
    # collapsed them, the bug content-signature dedup fixes).
    reports = [
        _report("surgical", touched=["sandbox/calc.py"], diff="+    return a - b\n"),
        _report("refactor", touched=["sandbox/calc.py"], diff="+    return -(b - a)\n"),
    ]
    portfolio = _portfolio(
        ScoredBranch(branch_id="surgical", rank=1, survived=True, objective_met=True),
        ScoredBranch(branch_id="refactor", rank=2, survived=True, objective_met=True),
    )
    store = _FakeStore()
    ids = ingest_survivors("implement subtract", reports, portfolio, store)
    assert len(ids) == 2


def test_identical_survivors_deduped_within_run():
    # Two survivors with identical touched files AND diff -> same signature -> one.
    reports = [
        _report("a", touched=["sandbox/calc.py"], diff="+    return a - b\n"),
        _report("b", touched=["sandbox/calc.py"], diff="+    return a - b\n"),
    ]
    portfolio = _portfolio(
        ScoredBranch(branch_id="a", rank=1, survived=True, objective_met=True),
        ScoredBranch(branch_id="b", rank=2, survived=True, objective_met=True),
    )
    store = _FakeStore()
    ids = ingest_survivors("o", reports, portfolio, store)
    assert len(ids) == 1


def test_ingest_skips_content_already_in_store():
    # A weekly re-run that reproduces the same diff is skipped (cross-run dedup
    # via the store's existing sig tags).
    rep = _report("surgical", touched=["sandbox/calc.py"], diff="+    return a - b\n")
    prior = branch_report_to_proposal(rep, objective="o", rank=1)
    sig = next(t for t in prior.tags if t.startswith("sig:"))
    store = _FakeStore(existing_meta=[{"tags_json": json.dumps([sig])}])
    portfolio = _portfolio(
        ScoredBranch(branch_id="surgical", rank=1, survived=True, objective_met=True))
    ids = ingest_survivors("o", [rep], portfolio, store)
    assert ids == [] and store.stored == []


def test_ingest_handles_missing_report_for_survivor():
    # A survivor with no matching BranchReport is skipped, not a crash.
    portfolio = _portfolio(
        ScoredBranch(branch_id="ghost", rank=1, survived=True, objective_met=True))
    store = _FakeStore()
    assert ingest_survivors("o", [], portfolio, store) == []
