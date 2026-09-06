#!/usr/bin/env python3
"""Read-only probe: gradable evidence-relevance data for insight-mode sweeps.

Purpose (Codex audit, 2026-09-06): broad keyword facets (e.g. a bare "work"
keyword) pull irrelevant philosophical/technical passages into insight-mode
evidence. The owner rule is "regression data before broad threshold changes"
— this script does NOT change any threshold or gate. It runs THE deployed
`core.insight.facets.decompose` (or its deterministic fallback,
`core.insight.facets._fallback_plan`, under --no-llm) and THE deployed
`core.insight.sweep.run_sweep` against the real stores, and emits one JSONL
row per evidence item with a blank `grade` field for a human to fill in.

READ-ONLY: this script opens the same stores `main.build_orchestrator()`
wires up for the live daemon and only calls query/read methods
(`chroma_store.query_collection`, corpus keyword search, graph BFS, temporal
expansion — all read paths inside the deployed `run_sweep`). It never calls
any store-mutating method (no writes of any kind to any collection). It
is safe to run at any time, whether or not a live daemon is also running —
reads don't race with the daemon's writes the way a script that later
applies a mutation would (see `utils/daemon_guard.py` for that guard, which
does not apply here).

Usage:
    python -m scripts.probe_insight_sweep_relevance --last 10
    python -m scripts.probe_insight_sweep_relevance --query "how has work been going"
    python -m scripts.probe_insight_sweep_relevance --no-llm --query "..." --out /tmp/rows.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TURN_RECORDS_PATH = REPO_ROOT / "logs" / "turn_records.jsonl"


# --------------------------------------------------------------------- #
# Pure helpers (unit-tested directly with synthetic inputs — no stores,
# no LLM, no filesystem beyond what the caller passes in).
# --------------------------------------------------------------------- #

def is_insight_record(rec: Any) -> bool:
    """True when a turn_records.jsonl row is an insight-mode turn.

    Matches either `mode` starting with "insight" (e.g. "insight-assembly")
    or a `gate_reason` containing "insight-mode" (e.g. the pattern_temporal
    facet's "insight-mode: pattern_temporal").
    """
    if not isinstance(rec, dict):
        return False
    mode = rec.get("mode")
    if isinstance(mode, str) and mode.startswith("insight"):
        return True
    reason = rec.get("gate_reason")
    if isinstance(reason, str) and "insight-mode" in reason:
        return True
    return False


def load_insight_queries(path: Path, last_n: int = 10) -> List[Dict[str, Any]]:
    """Read insight-mode turn records from a turn_records.jsonl-shaped file.

    The file is append-only (chronological by line order), so the last
    ``last_n`` matching records are the most recent. ``last_n <= 0`` returns
    every matching record. Malformed lines are skipped, never raised.
    """
    path = Path(path)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if is_insight_record(rec):
                records.append(rec)
    if last_n and last_n > 0:
        records = records[-last_n:]
    return records


def _apply_redaction(text: str, redact: bool):
    if not redact:
        return text
    from utils.privacy_redaction import redact_text
    return redact_text(text)


def build_evidence_row(
    *,
    query: str,
    facet_name: str,
    facet_keywords: Optional[Iterable[str]],
    item: Any,
    rank: int,
    redact: bool = True,
    head_chars: int = 200,
) -> Dict[str, Any]:
    """One gradable JSONL row for a single evidence item.

    ``item`` is an EvidenceItem (or anything exposing the same attributes
    via getattr — synthetic test doubles included). Text is truncated to
    ``head_chars`` BEFORE redaction is measured (redaction may change
    length, e.g. an email collapses to "[REDACTED EMAIL]").
    """
    text = getattr(item, "text", "") or ""
    head = text[:head_chars]
    query_out = _apply_redaction(query, redact)
    head_out = _apply_redaction(head, redact)
    return {
        "query": query_out,
        "facet": facet_name or "",
        "facet_keywords": list(facet_keywords or []),
        "collection": getattr(item, "collection", "") or "",
        "stance_label": getattr(item, "stance_label", "") or "",
        "speaker": getattr(item, "speaker", "") or "",
        "date": getattr(item, "date", None),
        "doc_id": getattr(item, "doc_id", None),
        "text_head": head_out,
        "rank": rank,
        "grade": "",
    }


def summarize_items(items: Iterable[Any]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Item counts grouped by collection and by facet."""
    per_collection: Dict[str, int] = {}
    per_facet: Dict[str, int] = {}
    for item in items:
        coll = getattr(item, "collection", "") or "?"
        facet = getattr(item, "facet", "") or "?"
        per_collection[coll] = per_collection.get(coll, 0) + 1
        per_facet[facet] = per_facet.get(facet, 0) + 1
    return per_collection, per_facet


def build_summary_row(
    *, query: str, per_collection: Dict[str, int], per_facet: Dict[str, int], total: int,
) -> Dict[str, Any]:
    """One JSONL summary row per query (items per collection/facet)."""
    return {
        "type": "summary",
        "query": query,
        "total_items": total,
        "per_collection": dict(per_collection),
        "per_facet": dict(per_facet),
    }


def format_table(query: str, items: List[Any], per_collection: Dict[str, int], per_facet: Dict[str, int]) -> str:
    """Compact human-readable table for stdout."""
    lines = [f"\n=== {query[:90]!r} — {len(items)} evidence items ==="]
    if per_collection:
        lines.append("  by collection: " + ", ".join(f"{k}={v}" for k, v in sorted(per_collection.items())))
    if per_facet:
        lines.append("  by facet:      " + ", ".join(f"{k}={v}" for k, v in sorted(per_facet.items())))
    for i, item in enumerate(items[:15], 1):
        head = (getattr(item, "text", "") or "")[:80].replace("\n", " ")
        lines.append(
            f"    [{i:>2}] {getattr(item, 'collection', ''):<14} "
            f"{getattr(item, 'stance_label', ''):<20} {head}"
        )
    if len(items) > 15:
        lines.append(f"    … {len(items) - 15} more")
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# Store-touching functions (import-only at call time; never executed by
# the unit tests, which exercise only the pure helpers above).
# --------------------------------------------------------------------- #

async def decompose_facets(query: str, model_manager, no_llm: bool):
    """Facet plan via THE deployed decompose(), or its deterministic
    fallback under --no-llm / when no model manager is available."""
    from core.insight.facets import _fallback_plan, decompose
    from core.insight.types import InsightIntent

    intent = InsightIntent(kind="theme_sweep", theme=query, raw_query=query)
    if no_llm or model_manager is None:
        return _fallback_plan(intent), intent
    plan = await decompose(intent, model_manager)
    return plan, intent


async def run_one_query(
    query: str, *, orch, model_manager, no_llm: bool, redact: bool, out_fh,
) -> None:
    """Decompose + sweep one query and write its gradable rows + summary."""
    from core.insight.sweep import run_sweep

    plan, _intent = await decompose_facets(query, model_manager, no_llm)

    mem = getattr(orch, "memory_system", None)
    chroma_store = getattr(mem, "chroma_store", None)
    corpus_manager = getattr(mem, "corpus_manager", None) or getattr(orch, "corpus_manager", None)
    graph_memory = getattr(mem, "graph_memory", None) or getattr(orch, "graph_memory", None)
    entity_resolver = getattr(mem, "entity_resolver", None)
    memory_expander = getattr(mem, "memory_expander", None)

    items = await run_sweep(
        plan,
        chroma_store=chroma_store,
        corpus_manager=corpus_manager,
        graph_memory=graph_memory,
        entity_resolver=entity_resolver,
        memory_expander=memory_expander,
        request_text=query,
    )

    # Another agent may be adding evidence_layout concurrently — apply it
    # for reference if present, but this probe does not depend on it.
    try:
        from core.insight.evidence_layout import layout_evidence
        items = layout_evidence(items) or items
        print("  [evidence_layout applied]")
    except ImportError:
        print("  [evidence_layout not available yet — skipped]")

    facet_by_name = {f.name: f for f in plan.facets}
    per_collection, per_facet = summarize_items(items)
    print(format_table(query, items, per_collection, per_facet))

    for rank, item in enumerate(items, 1):
        facet_name = getattr(item, "facet", "") or ""
        facet = facet_by_name.get(facet_name)
        row = build_evidence_row(
            query=query,
            facet_name=facet_name,
            facet_keywords=facet.keywords if facet else [],
            item=item,
            rank=rank,
            redact=redact,
        )
        out_fh.write(json.dumps(row) + "\n")

    summary = build_summary_row(query=query, per_collection=per_collection, per_facet=per_facet, total=len(items))
    out_fh.write(json.dumps(summary) + "\n")


def default_out_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "eval" / "runs" / f"insight_sweep_probe_{ts}.jsonl"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--last", type=int, default=10,
                   help="Number of most-recent insight-mode turn_records.jsonl entries to probe (default 10)")
    p.add_argument("--query", type=str, default=None,
                   help="Run one ad-hoc query instead of reading turn_records.jsonl")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSONL path (default eval/runs/insight_sweep_probe_<timestamp>.jsonl)")
    p.add_argument("--no-llm", action="store_true",
                   help="Use the deterministic facet fallback instead of an LLM decompose call")
    p.add_argument("--redact", dest="redact", action="store_true", default=True,
                   help="Redact PII in output rows before writing (default ON — this file may leave the machine)")
    p.add_argument("--no-redact", dest="redact", action="store_false",
                   help="Disable redaction (only for local-only grading; never share the resulting file)")
    p.add_argument("--turn-records", type=str, default=str(DEFAULT_TURN_RECORDS_PATH),
                   help="Path to turn_records.jsonl")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.query:
        queries = [args.query]
    else:
        records = load_insight_queries(Path(args.turn_records), args.last)
        queries = [r.get("query", "") for r in records if r.get("query")]
        if not queries:
            print(
                "No insight-mode turn records found in "
                f"{args.turn_records} (or their query fields were empty). "
                "Use --query to run an ad-hoc probe instead."
            )
            return 1

    out_path = Path(args.out) if args.out else default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building orchestrator (read-only; loads models — ~1 min)...")
    from main import build_orchestrator  # lazy import: heavy startup cost
    orch = build_orchestrator()
    model_manager = getattr(orch, "model_manager", None)

    with out_path.open("w", encoding="utf-8") as out_fh:
        for query in queries:
            asyncio.run(run_one_query(
                query, orch=orch, model_manager=model_manager,
                no_llm=args.no_llm, redact=args.redact, out_fh=out_fh,
            ))

    print(f"\nWrote gradable rows to {out_path}")
    print('Fill in the blank "grade" field per row (e.g. relevant/borderline/irrelevant) for the audit.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
