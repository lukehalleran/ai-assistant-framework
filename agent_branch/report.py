# agent_branch/report.py
"""
Agent Branch Trial Report — the single human-scannable artifact (M2).

Module Contract
- Purpose: aggregate N per-branch results + the ranking into ONE report a human
  can act on WITHOUT reading any diff. Markdown for humans, JSON for machines.
- Per branch it records: strategy, status (survived/rejected/error) + kill reason,
  proof result (supervisor tests), branch-test evidence (worker's own tests, kept
  visibly distinct from proof), changed files / line counts, gate flags, and cost
  (tokens / wallclock). Then a ranking, a divergence check, and a recommendation
  that explicitly does NOT auto-merge.
- Inputs: objective, the BranchReports, and the RankedPortfolio from scoring.
- Side effects: none (rendering only; the caller writes files).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import BaseModel, Field

from agent_branch.scoring import RankedPortfolio

if TYPE_CHECKING:
    from agent_branch.supervisor import BranchReport


class BranchSummary(BaseModel):
    branch_id: str
    strategy: str = ""
    status: str = "rejected"          # survived | rejected | error
    rank: Optional[int] = None
    kill_reason: str = ""
    files: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    proof: str = "n/a"                # pass | fail | n/a
    branch_tests: str = "none"        # passed | failed | none
    flags: List[str] = Field(default_factory=list)
    tokens: int = 0
    wallclock_s: float = 0.0


class TrialReport(BaseModel):
    objective: str
    created_at: str = ""
    branches: List[BranchSummary] = Field(default_factory=list)
    ranking: List[str] = Field(default_factory=list)   # survivor branch_ids, best first
    overlaps: List[Dict] = Field(default_factory=list)
    recommendation: str = ""

    # ---- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        objective: str,
        reports: "List[BranchReport]",
        portfolio: RankedPortfolio,
        *,
        created_at: str = "",
    ) -> "TrialReport":
        rank_by_id = {s.branch_id: s.rank for s in portfolio.ranked}
        summaries: List[BranchSummary] = []
        for r in reports:
            gate = r.static_gate
            se = r.sandbox_eval
            ev = se.branch_evidence if se else None
            status = {"passed": "survived", "killed": "rejected", "error": "error"}.get(r.outcome, "rejected")
            summaries.append(BranchSummary(
                branch_id=r.branch_id, strategy=r.strategy, status=status,
                rank=rank_by_id.get(r.branch_id),
                kill_reason="; ".join(r.reasons[:2]) if status != "survived" else "",
                files=len(gate.touched_paths),
                lines_added=gate.added_lines, lines_removed=gate.removed_lines,
                proof=("pass" if se and se.passed else "fail") if se else "n/a",
                branch_tests=("passed" if ev and ev.passed else "failed") if ev and ev.ran else "none",
                flags=[f.value for f in gate.flags],
                tokens=r.run_stats.tokens_spent,
                wallclock_s=round(r.run_stats.wallclock_elapsed_s, 1),
            ))
        return cls(
            objective=objective, created_at=created_at, branches=summaries,
            ranking=[s.branch_id for s in portfolio.ranked],
            overlaps=[o.model_dump() for o in portfolio.overlaps],
            recommendation=portfolio.recommendation,
        )

    # ---- rendering ---------------------------------------------------------

    def render_markdown(self) -> str:
        survived = [b for b in self.branches if b.status == "survived"]
        rejected = [b for b in self.branches if b.status != "survived"]
        out: List[str] = [
            "# Agent Branch Trial Report",
            "",
            f"**Objective:** {self.objective}  ",
            f"**Workers:** {len(self.branches)} — {len(survived)} survived, {len(rejected)} rejected"
            + (f"  \n**Generated:** {self.created_at}" if self.created_at else ""),
            "",
            "## Ranking (survivors, best first)",
        ]
        if self.ranking:
            by_id = {b.branch_id: b for b in self.branches}
            for i, bid in enumerate(self.ranking, 1):
                b = by_id[bid]
                out.append(f"{i}. **{bid}** ({b.strategy or 'n/a'}) — "
                           f"{b.lines_added + b.lines_removed} lines / {b.files} file(s), "
                           f"{b.tokens} tokens, {b.wallclock_s}s")
        else:
            out.append("_No survivors._")

        out += ["", "## Branches", "",
                "| branch | strategy | status | proof | branch-tests | files | +/− | cost | note |",
                "|---|---|---|---|---|---|---|---|---|"]
        for b in self.branches:
            status = f"survived #{b.rank}" if b.status == "survived" else b.status.upper()
            note = b.kill_reason if b.status != "survived" else (",".join(b.flags) or "—")
            out.append(
                f"| {b.branch_id} | {b.strategy or '—'} | {status} | {b.proof} | "
                f"{b.branch_tests} | {b.files} | +{b.lines_added}/−{b.lines_removed} | "
                f"{b.tokens}t/{b.wallclock_s}s | {note} |"
            )

        if self.overlaps:
            out += ["", "## Divergence (pairwise file overlap among survivors)"]
            for o in self.overlaps:
                shared = ", ".join(o.get("shared_files") or []) or "none"
                out.append(f"- {o['a']} vs {o['b']}: jaccard {o['jaccard']} (shared: {shared})")

        out += ["", "## Recommendation", "", self.recommendation,
                "", "_This harness evaluates and ranks only — it never auto-merges._"]
        return "\n".join(out) + "\n"

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
