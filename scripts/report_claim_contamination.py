#!/usr/bin/env python3
"""
Dry-run contamination report over `data/user_profile.json` — READ-ONLY.

No `--apply` flag exists at all: this script never writes to the profile,
never touches ChromaDB, never touches the knowledge graph. It runs the
DEPLOYED provenance/temporal-kind functions (memory.fact_source) against
every `is_current` profile fact's OWN stored `source_excerpt` and reports
which facts would NOT be admitted under the current (post-2026-09-06,
clause-level-negation-aware) rules — the repair path for legacy records that
predate a fix (see docs/HANDOFF_20260906_phaseAB_contracts.md Phase B / B1,
B8).

Checks per `is_current` fact (subject is always implicitly "user" for a
profile fact):
  unsupported_by_own_excerpt — memory.fact_source.find_supporting_user_span
      run with the fact's OWN source_excerpt as the ONLY user message
      returns None (the stored excerpt, taken alone, does not corroborate
      the (user, relation, value) triple under current rules).
  negated_object_clause — the excerpt's object-bearing clause is negated
      (memory.fact_source._clause_is_negated) — an affirmative fact whose
      only cited evidence is a denial.
  past_event_current — memory.fact_source.classify_claim_time on the
      excerpt classifies as a discrete PAST EVENT whose resolved date is
      before today, yet the fact is still marked is_current=True.
  origin_test — the excerpt contains a [test]...[/test] block
      (memory.fact_source.contains_test_block) — synthetic/replay content.

Each flagged fact gets a `proposed_action`:
  quarantine — the evidence itself contradicts or never supported the claim
      (unsupported_by_own_excerpt / negated_object_clause / origin_test).
  supersede  — the claim is well-evidenced but is a one-time PAST event,
      not durable current state (past_event_current alone).
  review     — flagged for a reason this script does not yet map to a
      confident action (defensive fallback; unreachable with the current
      reason set, kept for forward-compatibility).

`dependent hints` is a `subject|relation|value` string a human reviewer can
grep the knowledge graph for — this script never reads or writes the graph.

Usage:
    python scripts/report_claim_contamination.py --out /tmp/report.json
    python scripts/report_claim_contamination.py --profile data/user_profile.json --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_PROFILE_PATH = "data/user_profile.json"

QUARANTINE_REASONS = frozenset({
    "unsupported_by_own_excerpt", "negated_object_clause", "origin_test",
})
SUPERSEDE_REASONS = frozenset({"past_event_current"})


def _daemon_running() -> bool:
    """Read-only report — a live Daemon changes nothing here, so this is a
    WARNING only, never a refusal (unlike the --apply scripts)."""
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without touching data/user_profile.json)
# ---------------------------------------------------------------------------

def evaluate_fact(fact: Dict[str, Any], *, observed_at: Optional[datetime] = None) -> List[str]:
    """Reasons THE DEPLOYED provenance/temporal functions flag this fact for
    — empty when the fact is clean under current rules."""
    from memory.fact_source import (
        _clause_is_negated,
        _object_bearing_clause,
        _split_clauses,
        _tokens,
        classify_claim_time,
        contains_test_block,
        find_supporting_user_span,
    )

    reasons: List[str] = []
    relation = str(fact.get("relation") or "").strip()
    value = str(fact.get("value") or "").strip()
    excerpt = str(fact.get("source_excerpt") or "").strip()
    if not relation or not value or not excerpt:
        # Nothing to evaluate against — nothing to flag either (an absent
        # excerpt is a separate, pre-existing data-quality question this
        # report does not invent an opinion about).
        return reasons

    triple = {"subject": "user", "relation": relation, "object": value}

    if find_supporting_user_span(triple, [excerpt]) is None:
        reasons.append("unsupported_by_own_excerpt")

    object_tokens = _tokens(value)
    clauses = _split_clauses(excerpt)
    object_clause = _object_bearing_clause(clauses, value, object_tokens) or excerpt
    if _clause_is_negated(object_clause, value, object_tokens):
        reasons.append("negated_object_clause")

    claim_time = classify_claim_time(excerpt, observed_at=observed_at)
    if claim_time.kind == "event" and claim_time.event_date is not None:
        today = (observed_at or datetime.now()).date()
        if claim_time.event_date < today:
            reasons.append("past_event_current")

    if contains_test_block(excerpt):
        reasons.append("origin_test")

    return reasons


def propose_action(reasons: List[str]) -> str:
    reason_set = set(reasons)
    if reason_set & QUARANTINE_REASONS:
        return "quarantine"
    if reason_set & SUPERSEDE_REASONS:
        return "supersede"
    return "review"


def _truncate(value: str, limit: int = 80) -> str:
    value = value or ""
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def build_report_rows(
    categories: Dict[str, List[Dict[str, Any]]],
    *,
    observed_at: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """One row per flagged is_current fact, across every category."""
    rows: List[Dict[str, Any]] = []
    for category, facts in (categories or {}).items():
        for fact in facts or []:
            if not isinstance(fact, dict):
                continue
            if not fact.get("is_current", True):
                continue
            reasons = evaluate_fact(fact, observed_at=observed_at)
            if not reasons:
                continue
            relation = str(fact.get("relation") or "")
            value = str(fact.get("value") or "")
            rows.append({
                "fact_id": fact.get("fact_id", ""),
                "category": category,
                "relation": relation,
                "value": _truncate(value, 80),
                "reasons": reasons,
                "proposed_action": propose_action(reasons),
                "dependent_hint": f"user|{relation}|{value}",
            })
    return rows


def count_table(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    by_reason: Dict[str, int] = {}
    by_action: Dict[str, int] = {}
    for row in rows:
        for reason in row["reasons"]:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        action = row["proposed_action"]
        by_action[action] = by_action.get(action, 0) + 1
    return {"by_reason": by_reason, "by_action": by_action}


def scanned_current_count(categories: Dict[str, List[Dict[str, Any]]]) -> int:
    total = 0
    for facts in (categories or {}).values():
        for fact in facts or []:
            if isinstance(fact, dict) and fact.get("is_current", True):
                total += 1
    return total


def print_report(rows: List[Dict[str, Any]], scanned: int) -> None:
    print(f"Scanned {scanned} is_current fact(s).")
    print(f"Flagged {len(rows)} fact(s).")
    counts = count_table(rows)
    if counts["by_reason"]:
        print("\nBy reason:")
        for reason, n in sorted(counts["by_reason"].items()):
            print(f"  {reason}: {n}")
    if counts["by_action"]:
        print("\nProposed actions:")
        for action, n in sorted(counts["by_action"].items()):
            print(f"  {action}: {n}")
    for row in rows:
        print(
            f"  - [{row['proposed_action']}] {row['category']}/{row['relation']}="
            f"{row['value']!r} ({', '.join(row['reasons'])}) "
            f"fact_id={row['fact_id']} hint={row['dependent_hint']}"
        )


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--profile", default=DEFAULT_PROFILE_PATH,
                     help=f"path to a user_profile.json (default: {DEFAULT_PROFILE_PATH})")
    ap.add_argument("--out", required=True, help="JSON report output path (the only file this script writes)")
    args = ap.parse_args()

    if _daemon_running():
        print("WARNING: a live Daemon main.py process is running. This report is "
              "READ-ONLY (no --apply exists), so this is informational only — the "
              "report may not reflect writes the live instance makes after this run.")

    from memory.user_profile import UserProfile

    profile = UserProfile(profile_path=args.profile)
    categories = profile.profile.get("categories", {})
    observed_at = datetime.now()

    rows = build_report_rows(categories, observed_at=observed_at)
    scanned = scanned_current_count(categories)
    print_report(rows, scanned)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": observed_at.isoformat(),
        "profile_path": str(args.profile),
        "scanned_current_facts": scanned,
        "flagged_count": len(rows),
        "counts": count_table(rows),
        "rows": rows,
    }
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
