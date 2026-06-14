# agent_branch/scoring.py
"""
Mechanical, transparent ranking of portfolio branches (M2).

Module Contract
- Purpose: turn N per-branch results into a ranked portfolio a human can act on
  WITHOUT reading any diff. Ranking is deliberately mechanical — NOT an LLM judge
  (LLM judgment is the eval_layer2 stub, deferred to M3). Since merging is not the
  goal, the signal is: survive the gates, then prefer the cheapest correct change.
- Ranking key (survivors only, all of which passed the proof tests):
    1. objective met (proof passed)  — true for all survivors, kept explicit
    2. diff economy: fewer changed lines, then fewer files
    3. cost: fewer tokens, then less wallclock
- Also reports pairwise file-overlap among survivors — the honest check on whether
  the strategies actually diverged (high overlap ⇒ they converged on the same edit).
- Inputs: a sequence of BranchReport-like objects (duck-typed; this stays a pure
  leaf module and does not import the podman-heavy supervisor at runtime).
- Output: RankedPortfolio (ranked survivors, rejected branches, overlaps, a
  human recommendation that never auto-merges).
- Side effects: none.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, List, Optional, Sequence

from pydantic import BaseModel, Field

from utils.logging_utils import get_logger

if TYPE_CHECKING:  # annotation only — no runtime import / no cycle
    from agent_branch.supervisor import BranchReport

logger = get_logger("agent_branch.scoring")


class ScoredBranch(BaseModel):
    branch_id: str
    strategy: str = ""
    survived: bool = False
    rank: Optional[int] = None        # 1 = best; None for rejected branches
    objective_met: bool = False
    diff_lines: int = 0
    diff_files: int = 0
    tokens: int = 0
    wallclock_s: float = 0.0
    kill_reason: str = ""
    rationale: str = ""


class PairOverlap(BaseModel):
    a: str
    b: str
    shared_files: List[str] = Field(default_factory=list)
    jaccard: float = 0.0               # file-set overlap (1.0 = same files touched)
    content_similarity: float = 0.0    # diff-content similarity (1.0 = identical change)


class RankedPortfolio(BaseModel):
    ranked: List[ScoredBranch] = Field(default_factory=list)    # survivors, best first
    rejected: List[ScoredBranch] = Field(default_factory=list)
    overlaps: List[PairOverlap] = Field(default_factory=list)
    recommendation: str = ""

    @property
    def winner(self) -> Optional[ScoredBranch]:
        return self.ranked[0] if self.ranked else None


def _score_one(report: "BranchReport") -> ScoredBranch:
    gate = report.static_gate
    survived = report.outcome == "passed"
    objective_met = bool(report.sandbox_eval and report.sandbox_eval.passed)
    return ScoredBranch(
        branch_id=report.branch_id,
        strategy=getattr(report, "strategy", "") or "",
        survived=survived,
        objective_met=objective_met,
        diff_lines=gate.added_lines + gate.removed_lines,
        diff_files=len(gate.touched_paths),
        tokens=report.run_stats.tokens_spent,
        wallclock_s=round(report.run_stats.wallclock_elapsed_s, 2),
        kill_reason="" if survived else "; ".join(report.reasons[:3]),
    )


def _added_text(diff: str) -> str:
    """The added lines of a unified diff (markers stripped) — the actual new content.
    Used to measure how much two survivors' CHANGES differ, not just whether they
    touched the same files (file-overlap is always 1.0 on a single-file objective,
    so it can't tell convergence from genuine divergence — content can)."""
    return "\n".join(l[1:] for l in (diff or "").splitlines()
                     if l.startswith("+") and not l.startswith("+++"))


def _pairwise_overlaps(reports: "Sequence[BranchReport]", survivor_ids: set) -> List[PairOverlap]:
    survs = {r.branch_id: r for r in reports if r.branch_id in survivor_ids}
    files = {bid: set(r.static_gate.touched_paths) for bid, r in survs.items()}
    added = {bid: _added_text(getattr(r, "diff_excerpt", "")) for bid, r in survs.items()}
    ids = sorted(survs)
    out: List[PairOverlap] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            inter, union = files[a] & files[b], files[a] | files[b]
            sim = (difflib.SequenceMatcher(None, added[a], added[b]).ratio()
                   if (added[a] or added[b]) else 0.0)
            out.append(PairOverlap(
                a=a, b=b, shared_files=sorted(inter),
                jaccard=round(len(inter) / len(union), 3) if union else 0.0,
                content_similarity=round(sim, 3),
            ))
    return out


def _recommendation(survivors: List[ScoredBranch]) -> str:
    if not survivors:
        return ("No survivor — every branch was rejected. Do NOT proceed; investigate the "
                "objective or the workers before scaling.")
    best = survivors[0]
    return (f"Top survivor: {best.branch_id} (strategy={best.strategy or 'n/a'}) — the smallest "
            f"correct change ({best.diff_lines} lines / {best.diff_files} file(s)). "
            f"Human review required; this harness does NOT auto-merge.")


def score_branches(reports: "Sequence[BranchReport]") -> RankedPortfolio:
    """Rank survivors mechanically and surface rejected branches + divergence."""
    scored = [_score_one(r) for r in reports]
    # Correctness precedes economy: a branch may be RANKED only if it both passed
    # the gates AND met the objective (proof tests passed). A thin-but-wrong diff
    # cannot reach the ranking — diff economy is only ever a tiebreaker among
    # genuinely-correct survivors, never a path to the top (M2.5).
    survivors = [s for s in scored if s.survived and s.objective_met]
    rejected = [s for s in scored if not (s.survived and s.objective_met)]

    survivors.sort(key=lambda s: (s.diff_lines, s.diff_files, s.tokens, s.wallclock_s))
    for i, s in enumerate(survivors, 1):
        s.rank = i
        s.rationale = (f"rank {i}: {s.diff_lines} diff lines across {s.diff_files} file(s), "
                       f"{s.tokens} tokens, {s.wallclock_s:.1f}s")

    overlaps = _pairwise_overlaps(reports, {s.branch_id for s in survivors})
    portfolio = RankedPortfolio(
        ranked=survivors, rejected=rejected, overlaps=overlaps,
        recommendation=_recommendation(survivors),
    )
    logger.info("scored portfolio: %d survivors, %d rejected", len(survivors), len(rejected))
    return portfolio
