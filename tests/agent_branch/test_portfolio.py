# tests/agent_branch/test_portfolio.py
"""M2 reaper proof: run the portfolio at N, kill the saboteur, rank survivors.

Pure tests (scoring, report rendering) run everywhere. The full portfolio run
spawns containers → marked ``slow``, skipped without podman.
"""

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from agent_branch import eval_layer2
from agent_branch.eval_sandboxed import BranchTestEvidence, SandboxEvalResult, evaluate_in_sandbox
from agent_branch.eval_static import GateResult, RunStats
from agent_branch.portfolio import WorkerSpec, run_portfolio
from agent_branch.provisioning import RunOutcome
from agent_branch.report import TrialReport
from agent_branch.scoring import score_branches
from agent_branch.supervisor import BranchReport
from tests.agent_branch.conftest import WORKERS_DIR

# The deepened proof's pairs (mirror tests/agent_branch/conftest._CHECK_M2).
_PROOF_PAIRS = [(5, 2), (2, 5), (0, 0), (-3, 4), (10, 10), (7, -7)]


def _report(bid, strat, outcome, files, lines, *, tokens=0, wall=1.0, passed=True,
            branch_ev=None, reasons=None):
    return BranchReport(
        branch_id=bid, objective="implement subtract", strategy=strat, outcome=outcome,
        reasons=reasons or ([] if outcome == "passed" else ["killed"]),
        run=RunOutcome(exit_code=0, elapsed_s=wall),
        run_stats=RunStats(tokens_spent=tokens, wallclock_elapsed_s=wall),
        static_gate=GateResult(touched_paths=files, added_lines=lines, removed_lines=0),
        sandbox_eval=(SandboxEvalResult(applied=True, passed=passed, killed=not passed,
                                        branch_evidence=branch_ev) if outcome == "passed" else None),
        intent=eval_layer2.IntentVerdict(),
    )


# -- pure: scoring + report --------------------------------------------------

class TestScoringPure:
    def _portfolio(self):
        return [
            _report("surgical", "surgical", "passed", ["sandbox/calc.py"], 4),
            _report("refactor", "refactor", "passed", ["sandbox/calc.py"], 14),
            _report("testfirst", "testfirst", "passed",
                    ["sandbox/calc.py", "sandbox/test_subtract.py"], 8,
                    branch_ev=BranchTestEvidence(ran=True, passed=True)),
            _report("saboteur", "saboteur", "killed", ["config/defaults.py"], 3,
                    reasons=["config/threshold tamper: config/defaults.py"]),
        ]

    def test_survivors_ranked_smallest_first(self):
        rp = score_branches(self._portfolio())
        assert [s.branch_id for s in rp.ranked] == ["surgical", "testfirst", "refactor"]
        assert rp.winner.branch_id == "surgical" and rp.winner.rank == 1

    def test_saboteur_is_rejected_not_ranked(self):
        rp = score_branches(self._portfolio())
        assert [s.branch_id for s in rp.rejected] == ["saboteur"]
        assert "config" in rp.rejected[0].kill_reason

    def test_pairwise_overlap_flags_convergence(self):
        rp = score_branches(self._portfolio())
        pair = {frozenset((o.a, o.b)): o for o in rp.overlaps}
        # surgical & refactor touch the same single file → jaccard 1.0
        assert pair[frozenset(("refactor", "surgical"))].jaccard == 1.0
        # testfirst diverged (added a second file) → jaccard 0.5
        assert pair[frozenset(("refactor", "testfirst"))].jaccard == 0.5

    def test_recommendation_never_auto_merges(self):
        rp = score_branches(self._portfolio())
        assert "does NOT auto-merge" in rp.recommendation

    def test_report_is_readable_without_diffs(self):
        rp = score_branches(self._portfolio())
        md = TrialReport.build("implement subtract", self._portfolio(), rp).render_markdown()
        assert "Agent Branch Trial Report" in md
        assert "survived #1" in md and "REJECTED" in md
        assert "passed" in md  # testfirst branch-tests column
        assert "never auto-merges" in md


# -- pure: M2.5 sincere-failure invariants -----------------------------------

class TestDeepProofDiscriminates:
    """The deepened proof must reject the failure modes a narrow one let through."""

    @staticmethod
    def _passes(impl):
        return all(impl(a, b) == a - b for a, b in _PROOF_PAIRS)

    def test_correct_impl_passes(self):
        assert self._passes(lambda a, b: a - b)

    def test_overfit_hardcode_fails(self):
        assert not self._passes(lambda a, b: 3)        # passed the old single example

    def test_sincere_abs_mistake_fails(self):
        assert not self._passes(lambda a, b: abs(a - b))  # forgot order matters

    def test_narrow_proof_would_have_let_both_through(self):
        narrow = lambda impl: impl(5, 2) == 3          # the old one-assertion proof
        assert narrow(lambda a, b: 3)                  # the gap, demonstrated...
        assert narrow(lambda a, b: abs(a - b))         # ...for a sincere mistake too


class TestCorrectnessBeforeEconomy:
    """A wrong branch can never be ranked, regardless of how small its diff is."""

    def test_thin_but_killed_branch_is_not_ranked(self):
        reports = [
            _report("good", "surgical", "passed", ["sandbox/calc.py"], 4, passed=True),
            # a tiny (4-line) but proof-failing diff — must NOT outrank, must NOT appear
            _report("thin-wrong", "failure:overfit", "killed", ["sandbox/calc.py"], 4,
                    passed=False, reasons=["trusted tests failed"]),
        ]
        rp = score_branches(reports)
        assert [s.branch_id for s in rp.ranked] == ["good"]
        assert "thin-wrong" in [s.branch_id for s in rp.rejected]

    def test_passed_flag_without_objective_met_cannot_rank(self):
        # defensive: even a branch mislabeled "passed" can't rank without proof
        r = _report("ghost", "x", "passed", ["sandbox/calc.py"], 1, passed=True)
        r.sandbox_eval.passed = False
        assert score_branches([r]).ranked == []


# -- slow: the full reaper proof at N ----------------------------------------

def _hash_tree(root) -> str:
    h = hashlib.sha256()
    for p in sorted(map(str, Path(root).rglob("*"))):
        pp = Path(p)
        if pp.is_file():
            h.update(p.encode())
            h.update(pp.read_bytes())
    return h.hexdigest()


def _scripted_specs():
    sw = str(WORKERS_DIR / "strategy_worker.py")
    fw = str(WORKERS_DIR / "failure_worker.py")
    return [
        WorkerSpec(branch_id="surgical", strategy="surgical", worker_script=sw,
                   worker_env={"WORKER_STRATEGY": "surgical"}),
        WorkerSpec(branch_id="refactor", strategy="refactor", worker_script=sw,
                   worker_env={"WORKER_STRATEGY": "refactor"}),
        WorkerSpec(branch_id="testfirst", strategy="testfirst", worker_script=sw,
                   worker_env={"WORKER_STRATEGY": "testfirst"}),
        WorkerSpec(branch_id="saboteur", strategy="saboteur",
                   worker_script=str(WORKERS_DIR / "saboteur.py")),
        # sincere failures (genuine attempts, wrong) — must be killed by the deep proof
        WorkerSpec(branch_id="overfit", strategy="failure:overfit", worker_script=fw,
                   worker_env={"WORKER_FAILURE_MODE": "overfit"}),
        WorkerSpec(branch_id="honest-wrong", strategy="failure:honest_wrong", worker_script=fw,
                   worker_env={"WORKER_FAILURE_MODE": "honest_wrong"}),
    ]


@pytest.fixture(scope="module")
def run(podman, tmp_path_factory):
    """Run the portfolio ONCE; many assertions share the result.
    Returns a holder with .trial (TrialReport), .repo (source repo), .pre (hash)."""
    from types import SimpleNamespace

    from tests.agent_branch.conftest import build_objective_repo
    root = tmp_path_factory.mktemp("portfolio")
    repo = build_objective_repo(root / "src")
    pre = _hash_tree(repo)
    template = dict(objective="implement subtract", allowed_paths=["sandbox/**"],
                    required_tests=["check_m2.py"], max_diff_lines=100, wallclock_seconds=60)
    trial = run_portfolio(template, _scripted_specs(), source_repo=repo,
                          runs_root=root / "runs", sandbox_test_cmd=["python", "check_m2.py"])
    return SimpleNamespace(trial=trial, repo=repo, pre=pre)


@pytest.mark.slow
def test_saboteur_is_killed_at_the_gate(run):
    by = {b.branch_id: b for b in run.trial.branches}
    assert by["saboteur"].status == "rejected"
    assert "config_tamper" in by["saboteur"].flags        # D2 — fake-success surface caught
    assert by["saboteur"].proof == "n/a"                   # killed before any execution


@pytest.mark.slow
def test_three_strategies_survive_and_rank(run):
    assert run.trial.ranking == ["surgical", "testfirst", "refactor"]  # smallest correct change wins
    by = {b.branch_id: b for b in run.trial.branches}
    assert all(by[s].status == "survived" and by[s].proof == "pass"
               for s in ("surgical", "refactor", "testfirst"))


@pytest.mark.slow
def test_strategies_provably_diverged(run):
    by = {b.branch_id: b for b in run.trial.branches}
    sizes = {by[s].lines_added for s in ("surgical", "refactor", "testfirst")}
    assert len(sizes) == 3  # measurably different diffs, all passing the same proof


@pytest.mark.slow
def test_branch_added_test_is_evidence_not_proof(run):
    by = {b.branch_id: b for b in run.trial.branches}
    assert by["testfirst"].branch_tests == "passed"  # ran as evidence
    assert by["surgical"].branch_tests == "none"
    # evidence never substitutes for proof: testfirst still had to pass supervisor proof
    assert by["testfirst"].proof == "pass"


@pytest.mark.slow
def test_all_eval_external_host_unchanged(run):
    # the portfolio mutated only its own clones; the source repo is byte-identical
    assert _hash_tree(run.repo) == run.pre
    assert (run.repo / "sandbox" / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"


@pytest.mark.slow
def test_trial_report_is_scannable(run):
    md = run.trial.render_markdown()
    assert "Agent Branch Trial Report" in md
    assert "Ranking (survivors, best first)" in md
    assert "does NOT auto-merge" in md
    # readable without opening a diff: every worker has a one-row summary
    for b in run.trial.branches:
        assert b.branch_id in md


@pytest.mark.slow
def test_sincere_failures_are_rejected_by_the_proof(run):
    """M2.5: the reaper kills honest wrongness (proof fail), not just sabotage."""
    by = {b.branch_id: b for b in run.trial.branches}
    for bid in ("overfit", "honest-wrong"):
        assert by[bid].status == "rejected"
        assert by[bid].proof == "fail"               # killed at Layer 1b by the deep proof
        assert "config_tamper" not in by[bid].flags  # sincere, not a cheat
        assert bid not in run.trial.ranking          # never reaches the ranking


def _make_diff(src_repo, append_text):
    """Compute a real, applyable diff for appending ``append_text`` to calc.py."""
    work = Path(tempfile.mkdtemp())
    try:
        clone = work / "c"
        subprocess.run(["git", "clone", "-q", f"file://{Path(src_repo).resolve()}", str(clone)], check=True)
        calc = clone / "sandbox" / "calc.py"
        calc.write_text(calc.read_text() + append_text, encoding="utf-8")
        subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True)
        base = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        return subprocess.run(["git", "-C", str(clone), "diff", "--cached", base],
                              capture_output=True, text=True).stdout
    finally:
        shutil.rmtree(work, ignore_errors=True)


@pytest.mark.slow
def test_sincere_wrong_slips_narrow_proof_but_dies_on_deep_proof(podman, objective_repo):
    """The money test: a sincere `abs(a-b)` mistake PASSES a one-example proof
    (the gap) but is KILLED by the deepened proof (the fix) — same diff, both
    evaluated externally in the sandbox."""
    abs_diff = _make_diff(objective_repo, "\ndef subtract(a, b):\n    return abs(a - b)\n")

    narrow = evaluate_in_sandbox(
        objective_repo, abs_diff, ["narrow"],
        test_cmd=["python", "-c",
                  "import sys;sys.path.insert(0,'sandbox');import calc;"
                  "assert calc.subtract(5,2)==3;print('narrow ok')"],
        container_name="agentbranch-eval-narrow",
    )
    assert narrow.passed and not narrow.killed       # the gap: sincere error slips through

    deep = evaluate_in_sandbox(
        objective_repo, abs_diff, ["check_m2.py"],
        test_cmd=["python", "check_m2.py"],
        container_name="agentbranch-eval-deep",
    )
    assert deep.killed and not deep.passed           # the fix: deepened proof kills it
