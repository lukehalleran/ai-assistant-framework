# agent_branch/proposal_bridge.py
"""
Bridge — agent_branch Trial Report survivors -> CodeProposal in the live store.

Module Contract
- Purpose: the M3 hand-off. Convert SURVIVING branches (passed both gates AND met
  the objective) from a portfolio run into ``CodeProposal`` objects and store them
  in the same ``ProposalStore`` the GUI reads, so a supervisor-VERIFIED diff flows
  into the Proposals tab and is displayed / risk-badged / scored / sorted exactly
  like a self-generated proposal — then reviewed under the same
  acknowledge-before-approve gate. Nothing here merges or commits; the proposal is
  a PENDING review artifact awaiting human approval.
- This is the ONE place Track B (agent_branch isolation/eval) reaches into Track A
  (the live proposal store). Only RANKED SURVIVORS become proposals; killed /
  rejected branches are recorded in the Trial Report but never stored as proposals
  (they are noise, not candidates).
- Inputs: duck-typed ``BranchReport``-like objects (this stays a leaf module and
  does NOT import the podman-heavy supervisor at runtime — same pattern as
  scoring.py), a ``RankedPortfolio`` (its ``ranked`` survivors carry the rank), and
  a ProposalStore-like object exposing ``check_similarity`` + ``store_proposal``.
- Outputs: the list of stored proposal IDs.
- Mapping: source=AGENT_BRANCH, status=PENDING; priority from rank (rank 1 -> 10,
  so the best survivor sorts to the top); risk_level / touches_core_system computed
  by ``classify_proposal`` over the touched paths + the diff text (so a core/safety
  touch is flagged for the acknowledge gate); affected_files + per-file steps from
  the static gate; the supervisor-computed diff + proof result in the description;
  provenance (strategy, rank, branch id) in tags.
- Side effects: only via the injected store (store_proposal writes to ChromaDB).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, List, Sequence, Set

from memory.code_proposal import (
    CodeProposal,
    ImplementationStep,
    ProposalSource,
    ProposalStatus,
    ProposalType,
)
from memory.proposal_risk import classify_proposal
from utils.logging_utils import get_logger

if TYPE_CHECKING:  # annotation only — no runtime import / no podman dependency
    from agent_branch.scoring import RankedPortfolio
    from agent_branch.supervisor import BranchReport
    from memory.proposal_store import ProposalStore

logger = get_logger("agent_branch.proposal_bridge")


def _signature(touched: Sequence[str], diff: str) -> str:
    """A stable content signature over (touched files + diff) used for CONTENT
    dedup. Divergent survivors of the same objective have different diffs -> different
    signatures -> all kept (the whole point of a portfolio). An identical re-run
    produces the same signature -> skipped. NOT the LLM-generator's semantic/title
    dedup, which wrongly collapses divergent verified survivors into one."""
    h = hashlib.sha256()
    h.update("\n".join(sorted(touched)).encode("utf-8"))
    h.update(b"\x00")
    h.update((diff or "").encode("utf-8"))
    return "sig:" + h.hexdigest()[:16]


def _existing_signatures(store: "ProposalStore") -> Set[str]:
    """Content signatures of agent-branch proposals already in the store (read via
    the same list_all the GUI uses), so a weekly re-run doesn't re-store identical
    survivors. Degrades to empty if the store can't be listed."""
    sigs: Set[str] = set()
    chroma = getattr(store, "chroma_store", None)
    if chroma is None or not hasattr(chroma, "list_all"):
        return sigs
    try:
        for item in chroma.list_all("proposals"):
            md = item.get("metadata") or {}
            for tag in json.loads(md.get("tags_json", "[]") or "[]"):
                if isinstance(tag, str) and tag.startswith("sig:"):
                    sigs.add(tag)
    except Exception:  # noqa: BLE001 — dedup is best-effort, never block ingest
        pass
    return sigs


def _added_code(diff: str) -> str:
    """The ADDED lines of a unified diff with the leading '+' stripped, so
    import-based risk detection (classify_proposal) sees real code rather than
    '+from core.orchestrator import ...' (the '+' prefix would defeat the import
    regex and let a core/safety import through unflagged)."""
    out = []
    for line in (diff or "").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
    return "\n".join(out)


def _complexity_for(diff_lines: int) -> str:
    if diff_lines < 20:
        return "low"
    if diff_lines < 100:
        return "medium"
    return "high"


def branch_report_to_proposal(
    report: "BranchReport",
    *,
    objective: str,
    rank: int,
    total_survivors: int = 1,
) -> CodeProposal:
    """Convert one surviving ``BranchReport`` into a PENDING agent-branch proposal.

    Duck-typed: ``report`` need only expose ``branch_id``, ``static_gate``
    (``touched_paths`` / ``added_lines`` / ``removed_lines``), ``run_stats``
    (``tokens_spent`` / ``wallclock_elapsed_s``), ``sandbox_eval`` (``reason`` /
    ``branch_evidence``), ``diff_excerpt`` and (optionally) ``strategy``."""
    objective = (objective or "agent-branch change").strip()
    gate = report.static_gate
    touched = list(getattr(gate, "touched_paths", []) or [])
    added = int(getattr(gate, "added_lines", 0) or 0)
    removed = int(getattr(gate, "removed_lines", 0) or 0)
    diff_lines = added + removed
    diff = getattr(report, "diff_excerpt", "") or ""
    strategy = getattr(report, "strategy", "") or ""

    stats = getattr(report, "run_stats", None)
    tokens = int(getattr(stats, "tokens_spent", 0) or 0) if stats else 0
    wallclock = float(getattr(stats, "wallclock_elapsed_s", 0.0) or 0.0) if stats else 0.0

    sev = getattr(report, "sandbox_eval", None)
    proof_reason = getattr(sev, "reason", "") if sev else ""
    evidence = getattr(sev, "branch_evidence", None) if sev else None
    test_files = list(getattr(evidence, "test_files", []) or []) if evidence else []

    # rank 1 -> priority 10 (sorts to the top of the tab, like a high-priority
    # self-generated proposal); lower ranks step down but never below 1.
    rank = max(1, int(rank))
    priority = max(1, min(10, 11 - rank))

    # Risk drives the GUI acknowledge gate. Pass the ADDED code (diff markers
    # stripped) as code_texts so an import of a core/safety/supervision module
    # trips even when the file's own path looks clean.
    touches_core, risk = classify_proposal(
        touched, code_texts=[_added_code(diff)], title=objective,
        proposal_type=ProposalType.FEATURE,
    )

    steps = [
        ImplementationStep(
            order=i + 1, action="modify", file_path=p,
            description="agent-branch verified change",
        )
        for i, p in enumerate(touched)
    ]

    reasoning = (
        f"Survived the static gate and the sandboxed proof. Ranked {rank}/"
        f"{max(total_survivors, rank)} by diff economy ({added}+/{removed}- across "
        f"{len(touched)} file(s)), cost {tokens} tokens / {wallclock:.0f}s. "
        f"Worker strategy: {strategy or 'n/a'}."
    )

    desc_parts = [
        f"Agent-branch survivor for objective: {objective}.",
        f"Proof: {proof_reason or 'trusted tests passed'}.",
    ]
    if test_files:
        ev_passed = bool(getattr(evidence, "passed", False))
        desc_parts.append(
            f"Branch-added tests (evidence, not proof): "
            f"{'passed' if ev_passed else 'failed'} — {', '.join(test_files)}."
        )
    if diff:
        desc_parts.append("Supervisor-computed diff:\n```diff\n" + diff + "\n```")
    description = "\n\n".join(desc_parts)

    title = f"[agent-branch] {objective}"
    if strategy:
        title += f" — {strategy}"
    title = title[:200]

    tags = ["agent_branch", "verified"]
    if strategy:
        tags.append(strategy)
    tags += [f"rank-{rank}", f"branch:{report.branch_id}", _signature(touched, diff)]

    return CodeProposal(
        title=title,
        proposal_type=ProposalType.FEATURE,
        status=ProposalStatus.PENDING,
        source=ProposalSource.AGENT_BRANCH,
        priority=priority,
        reasoning=reasoning,
        description=description,
        implementation_steps=steps,
        affected_files=touched,
        tags=tags,
        estimated_complexity=_complexity_for(diff_lines),
        requires_tests=True,
        risk_level=risk,
        touches_core_system=touches_core,
        test_files=test_files,
    )


def ingest_survivors(
    objective: str,
    branch_reports: "Sequence[BranchReport]",
    portfolio: "RankedPortfolio",
    store: "ProposalStore",
) -> List[str]:
    """Store every ranked survivor as a PENDING agent-branch proposal.

    Only ``portfolio.ranked`` (survivors that passed the gates AND met the
    objective) are ingested — rejected/killed branches are never stored. ALL
    divergent survivors are kept (they are the point of a portfolio); dedup is by
    CONTENT signature (touched files + diff), so an identical re-run is skipped but
    a different diff is not. Deliberately NOT the semantic/title dedup, which would
    collapse divergent survivors of one objective into a single proposal. Returns
    the list of stored proposal IDs."""
    by_id = {r.branch_id: r for r in branch_reports}
    survivors = list(getattr(portfolio, "ranked", []) or [])
    total = len(survivors)
    seen = _existing_signatures(store)  # across prior runs
    stored: List[str] = []

    for sb in survivors:
        report = by_id.get(sb.branch_id)
        if report is None:
            logger.warning("ingest: no BranchReport for survivor %s", sb.branch_id)
            continue
        proposal = branch_report_to_proposal(
            report, objective=objective, rank=sb.rank or 1, total_survivors=total,
        )
        sig = next((t for t in proposal.tags if t.startswith("sig:")), None)
        if sig and sig in seen:
            logger.info("ingest: skipped identical-content survivor %s (%s)", sig, sb.branch_id)
            continue
        doc_id = store.store_proposal(proposal)
        if doc_id:
            if sig:
                seen.add(sig)  # also dedup within this run
            stored.append(proposal.id)
            logger.info("ingest: stored agent-branch proposal %s (rank %s, branch %s)",
                        proposal.id, sb.rank, sb.branch_id)

    logger.info("ingest: %d/%d survivor(s) stored as proposals", len(stored), total)
    return stored
