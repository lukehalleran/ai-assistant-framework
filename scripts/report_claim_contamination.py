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
  recurrence_unsupported — (2026-10-16, BC-51) a fact whose RELATION names a
      recurrence (habit/routine/pattern token) but whose OWN excerpt, through
      THE deployed memory.fact_source.recurrence_claim_supported
      (classify_claim_time_for_object + the sentence frame check), does not
      assert a present-tense recurrence.
  recurrence_no_evidence — the same relation family with an EMPTY excerpt.

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

Disposition rows (2026-10-16, CLS-5).  For every recurrence candidate (and
every fact_id listed in --inventory, whatever the detector says — this covers
non-family candidates such as the *_environment facts) the report builds one
row naming where the same claim lives in each store:
  profile      present:<fact_id> | absent
  chroma facts present:<doc ids> | absent | unresolved:<why>   (+ quarantine state)
  graph edges  present:<edge keys> | absent | unresolved:<why>  (+ suppression state)
  graph node attributes (memory_storage stores non-entity values as node
  metadata) present:<node>.<attr> | absent | unresolved:<why>
"unresolved" is NEVER absent: the store was not scanned, could not be read, or
the match was ambiguous.  The Chroma/graph scans are opt-in (--chroma-path /
--graph / --with-stores) because opening a live store is not free.

The owner curates the TSV (delete rows or comment them with '#'), then
--emit-id-files writes tool-compatible files into a scratch dir the owner names:
  profile_ids.txt     for scripts/purge_profile_facts.py      (one fact_id per line)
  chroma_ids.txt      for scripts/quarantine_facts.py         (one doc id per line)
  graph_edges.txt     for scripts/quarantine_graph_edges.py   (source|relation|target)
  owner_decision.txt  node-attribute hits + unresolved cells (no existing tool)
Nothing here writes to any Daemon store.

Usage:
    python scripts/report_claim_contamination.py --out /tmp/report.json
    python scripts/report_claim_contamination.py --profile data/user_profile.json --out report.json
    python scripts/report_claim_contamination.py --out r.json --disposition-tsv d.tsv --with-stores
    python scripts/report_claim_contamination.py --emit-id-files d_curated.tsv --scratch-dir /tmp/ids
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_PROFILE_PATH = "data/user_profile.json"

RECURRENCE_REASONS = frozenset({"recurrence_unsupported", "recurrence_no_evidence"})
QUARANTINE_REASONS = frozenset({
    "unsupported_by_own_excerpt", "negated_object_clause", "origin_test",
}) | RECURRENCE_REASONS
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
        classify_claim_time_for_object,
        contains_test_block,
        find_supporting_user_span,
        is_recurrence_relation,
        recurrence_claim_supported,
    )

    reasons: List[str] = []
    relation = str(fact.get("relation") or "").strip()
    value = str(fact.get("value") or "").strip()
    excerpt = str(fact.get("source_excerpt") or "").strip()
    if relation and value and not excerpt and is_recurrence_relation(relation):
        # CLS-5: a recurrence-named fact with NO stored evidence at all.
        reasons.append("recurrence_no_evidence")
        return reasons
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

    if is_recurrence_relation(relation) and not recurrence_claim_supported(
            relation, excerpt, object_clause, observed_at=observed_at):
        reasons.append("recurrence_unsupported")

    claim_time = classify_claim_time_for_object(excerpt, object_clause, observed_at=observed_at)
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
# CLS-5 (2026-10-16): per-store disposition rows.  READ-ONLY.
# ---------------------------------------------------------------------------

QUARANTINE_KEY = "curation_quarantined"
TSV_COLUMNS = [
    "fact_id", "category", "relation", "value", "origin", "reasons",
    "profile", "chroma", "chroma_state", "graph_edges", "graph_state", "graph_node_attrs",
]
_ABSENT = "absent"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", _norm(text)))


def _loosely_same(a: str, b: str) -> bool:
    """Ambiguity test (NOT identity): one contains the other, or token Jaccard >= 0.5."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.5


def parse_fact_content(content: str) -> Optional[Tuple[str, str, str]]:
    """``"subject | relation | object"`` -> (subject, relation, object), or None."""
    parts = [p.strip() for p in str(content or "").split(" | ", 2)]
    if len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


def chroma_cells(relation: str, value: str, docs: Optional[List[Dict[str, Any]]],
                 error: str = "") -> Tuple[str, str]:
    """(cell, state) for the Chroma ``facts`` copies of (user, relation, value)."""
    if docs is None:
        return f"unresolved:{error or 'not_scanned'}", ""
    rel, val = _norm(relation), _norm(value)
    exact: List[Dict[str, Any]] = []
    loose: List[str] = []
    for doc in docs:
        parsed = parse_fact_content(doc.get("content", ""))
        if not parsed or _norm(parsed[0]) != "user" or _norm(parsed[1]) != rel:
            continue
        if _norm(parsed[2]) == val:
            exact.append(doc)
        elif _loosely_same(parsed[2], value):
            loose.append(str(doc.get("id")))
    if exact:
        ids = [str(d.get("id")) for d in exact]
        states = [
            f"{d.get('id')}={'quarantined' if (d.get('metadata') or {}).get(QUARANTINE_KEY) else 'live'}"
            for d in exact
        ]
        return "present:" + ";".join(ids), ";".join(states)
    if loose:
        return "unresolved:ambiguous_object(" + ";".join(loose) + ")", ""
    return _ABSENT, ""


def _canon_relation(relation: str) -> set:
    rels = {_norm(relation)}
    try:
        from memory.entity_resolver import normalize_relation
        rels.add(_norm(normalize_relation(relation)))
    except Exception:  # degrades: only the raw relation spelling is searched; absent is then NOT claimed
        pass
    return rels


def graph_cells(relation: str, value: str, gm: Any, error: str = "") -> Tuple[str, str, str]:
    """(edge cell, suppression state, node-attribute cell) for one claim."""
    if gm is None:
        why = f"unresolved:{error or 'not_scanned'}"
        return why, "", why
    rels = _canon_relation(relation)
    val = _norm(value)
    exact: List[str] = []
    loose: List[str] = []
    states: List[str] = []
    for key, edge in list(gm._edge_index.items()):
        if _norm(edge.source_id) != "user" or _norm(edge.relation) not in rels:
            continue
        node = gm.get_entity(edge.target_id)
        names = {_norm(edge.target_id), _norm(getattr(node, "display_name", "") or "")}
        if val in names:
            exact.append(key)
            states.append(f"{key}={'suppressed' if gm.edge_is_suppressed(edge) else 'visible'}")
        elif any(_loosely_same(n, value) for n in names if n):
            loose.append(key)
    if exact:
        edge_cell = "present:" + ";".join(exact)
    elif loose:
        edge_cell = "unresolved:ambiguous_target(" + ";".join(loose) + ")"
    else:
        edge_cell = _ABSENT

    attr_hits: List[str] = []
    attr_loose: List[str] = []
    for nid in list(gm.graph.nodes):
        node = gm.get_entity(nid)
        md = (getattr(node, "metadata", None) or {}) if node is not None else {}
        for attr, held in md.items():
            if _norm(attr) not in rels:
                continue
            if _norm(held) == val:
                attr_hits.append(f"{nid}.{attr}")
            elif _loosely_same(str(held), value):
                attr_loose.append(f"{nid}.{attr}")
    if attr_hits:
        attr_cell = "present:" + ";".join(attr_hits)
    elif attr_loose:
        attr_cell = "unresolved:ambiguous_value(" + ";".join(attr_loose) + ")"
    else:
        attr_cell = _ABSENT
    return edge_cell, ";".join(states), attr_cell


def load_chroma_fact_docs(path: str) -> List[Dict[str, Any]]:
    """All ``facts`` documents from a Chroma directory (read-only ``get``).
    Raises on a missing directory/collection — the caller records ``unresolved``."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"chroma directory not found: {path}")
    import chromadb  # lazy import: startup-cost (heavy; opt-in scan only)
    coll = chromadb.PersistentClient(path=path).get_collection("facts", embedding_function=None)
    res = coll.get(include=["documents", "metadatas"])
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    return [
        {"id": ids[i], "content": docs[i] if i < len(docs) else "",
         "metadata": (metas[i] if i < len(metas) else {}) or {}}
        for i in range(len(ids))
    ]


def load_graph(path: str) -> Any:
    """A GraphMemory over ``path`` — only after proving the file exists and
    parses, because GraphMemory starts EMPTY on a missing file (which would
    read as a false ``absent``) and QUARANTINES a corrupt one (a write)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"graph file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        json.load(fh)
    from memory.graph_memory import GraphMemory
    return GraphMemory(persist_path=path)


def parse_inventory(text: str) -> List[str]:
    ids: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and line.split()[0] not in ids:
            ids.append(line.split()[0])
    return ids


def build_disposition_rows(
    categories: Dict[str, List[Dict[str, Any]]],
    *,
    inventory_ids: Optional[Iterable[str]] = None,
    observed_at: Optional[datetime] = None,
    chroma_docs: Optional[List[Dict[str, Any]]] = None,
    chroma_error: str = "",
    graph: Any = None,
    graph_error: str = "",
) -> List[Dict[str, str]]:
    """One row per recurrence candidate or inventoried fact_id."""
    inventory = list(dict.fromkeys(inventory_ids or []))
    by_id: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for category, facts in (categories or {}).items():
        for fact in facts or []:
            if isinstance(fact, dict) and fact.get("fact_id"):
                by_id[str(fact["fact_id"])] = (category, fact)

    selected: List[Tuple[str, str]] = []   # (fact_id, origin)
    for category, facts in (categories or {}).items():
        for fact in facts or []:
            if not isinstance(fact, dict) or not fact.get("is_current", True):
                continue
            if set(evaluate_fact(fact, observed_at=observed_at)) & RECURRENCE_REASONS:
                selected.append((str(fact.get("fact_id", "")), "detector"))
    seen = {fid for fid, _ in selected}
    selected = [(fid, "detector+inventory" if fid in inventory else "detector") for fid, _ in selected]
    selected += [(fid, "inventory") for fid in inventory if fid not in seen]

    rows: List[Dict[str, str]] = []
    for fid, origin in selected:
        category, fact = by_id.get(fid, ("", {}))
        relation = str(fact.get("relation") or "")
        value = str(fact.get("value") or "")
        row = {c: "" for c in TSV_COLUMNS}
        row.update({
            "fact_id": fid, "category": category, "relation": relation, "value": value,
            "origin": origin,
            "reasons": ",".join(evaluate_fact(fact, observed_at=observed_at)) if fact else "",
            "profile": f"present:{fid}" if fact else _ABSENT,
        })
        if not fact:
            why = "unresolved:no_profile_fact_to_search_by"
            row.update({"chroma": why, "graph_edges": why, "graph_node_attrs": why})
        else:
            row["chroma"], row["chroma_state"] = chroma_cells(relation, value, chroma_docs, chroma_error)
            row["graph_edges"], row["graph_state"], row["graph_node_attrs"] = graph_cells(
                relation, value, graph, graph_error)
        rows.append(row)
    return rows


def _cell(value: str) -> str:
    return re.sub(r"[\t\r\n]+", " ", str(value or "")).strip()


def rows_to_tsv(rows: List[Dict[str, str]]) -> str:
    lines = ["\t".join(TSV_COLUMNS)]
    lines += ["\t".join(_cell(r.get(c, "")) for c in TSV_COLUMNS) for r in rows]
    return "\n".join(lines) + "\n"


def _present_values(cell: str) -> List[str]:
    return [v for v in cell[len("present:"):].split(";") if v] if cell.startswith("present:") else []


def write_id_files(curated_tsv: str, scratch_dir: str) -> Dict[str, List[str]]:
    """Tool-compatible id files from a CURATED disposition TSV ('#' rows and
    the header are skipped).  Returns what was written."""
    out: Dict[str, List[str]] = {"profile_ids": [], "chroma_ids": [], "graph_edges": [], "owner_decision": []}
    for raw in (curated_tsv or "").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#") or raw.startswith("fact_id\t"):
            continue
        cells = dict(zip(TSV_COLUMNS, raw.split("\t")))
        for fid in _present_values(cells.get("profile", "")):
            if fid not in out["profile_ids"]:
                out["profile_ids"].append(fid)
        for cid in _present_values(cells.get("chroma", "")):
            if cid not in out["chroma_ids"]:
                out["chroma_ids"].append(cid)
        for key in _present_values(cells.get("graph_edges", "")):
            if key.lower() not in out["graph_edges"]:
                out["graph_edges"].append(key.lower())
        label = f"{cells.get('fact_id', '')} {cells.get('relation', '')}={cells.get('value', '')}"
        for col in ("graph_node_attrs", "chroma", "graph_edges"):
            cell = cells.get(col, "")
            if col == "graph_node_attrs" and cell.startswith("present:"):
                out["owner_decision"].append(f"node attribute (no tool): {label} -> {cell}")
            elif cell.startswith("unresolved"):
                out["owner_decision"].append(f"unresolved {col}: {label} -> {cell}")
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "profile_ids.txt").write_text("".join(f"{i}\n" for i in out["profile_ids"]))
    (scratch / "chroma_ids.txt").write_text("".join(f"{i}\n" for i in out["chroma_ids"]))
    (scratch / "graph_edges.txt").write_text("".join(f"{i}\n" for i in out["graph_edges"]))
    (scratch / "owner_decision.txt").write_text("".join(f"{i}\n" for i in out["owner_decision"]))
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--profile", default=DEFAULT_PROFILE_PATH,
                     help=f"path to a user_profile.json (default: {DEFAULT_PROFILE_PATH})")
    ap.add_argument("--out", help="JSON report output path")
    ap.add_argument("--inventory", help="file of profile fact_ids (one per line, '#' comments): "
                    "a disposition row is built for EVERY id, whatever the detector says")
    ap.add_argument("--disposition-tsv", help="write the per-store disposition rows here (owner curates)")
    ap.add_argument("--chroma-path", help="Chroma directory to scan for facts-collection copies")
    ap.add_argument("--graph", help="knowledge_graph.json to scan for edges / node attributes")
    ap.add_argument("--with-stores", action="store_true",
                    help="scan the config-default Chroma + graph paths (opens live stores read-only)")
    ap.add_argument("--emit-id-files", metavar="CURATED_TSV",
                    help="write tool-compatible id files from a curated disposition TSV, then exit")
    ap.add_argument("--scratch-dir", help="directory for --emit-id-files output (the owner names it)")
    args = ap.parse_args()

    if args.emit_id_files:
        if not args.scratch_dir:
            ap.error("--emit-id-files needs --scratch-dir")
        written = write_id_files(Path(args.emit_id_files).read_text(), args.scratch_dir)
        for name, items in written.items():
            print(f"  {name}: {len(items)}")
        print(f"Wrote id files into {args.scratch_dir} (nothing applied; each tool is dry-run first).")
        return 0
    if not args.out:
        ap.error("--out is required")

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

    if args.disposition_tsv or args.inventory:
        chroma_path, graph_path = args.chroma_path, args.graph
        if args.with_stores:
            from config.app_config import CHROMA_PATH, KNOWLEDGE_GRAPH_PERSIST_PATH
            chroma_path = chroma_path or CHROMA_PATH
            graph_path = graph_path or KNOWLEDGE_GRAPH_PERSIST_PATH
        chroma_docs, chroma_error, graph, graph_error = None, "", None, ""
        if chroma_path:
            try:
                chroma_docs = load_chroma_fact_docs(chroma_path)
            except Exception as e:  # degrades: Chroma cells read "unresolved", never "absent"
                chroma_error = f"chroma_unreadable({type(e).__name__})"
        if graph_path:
            try:
                graph = load_graph(graph_path)
            except Exception as e:  # degrades: graph cells read "unresolved", never "absent"
                graph_error = f"graph_unreadable({type(e).__name__})"
        inventory = parse_inventory(Path(args.inventory).read_text()) if args.inventory else []
        drows = build_disposition_rows(
            categories, inventory_ids=inventory, observed_at=observed_at,
            chroma_docs=chroma_docs, chroma_error=chroma_error,
            graph=graph, graph_error=graph_error,
        )
        print(f"Disposition rows: {len(drows)}")
        if args.disposition_tsv:
            tsv_path = Path(args.disposition_tsv)
            tsv_path.parent.mkdir(parents=True, exist_ok=True)
            tsv_path.write_text(rows_to_tsv(drows))
            print(f"Disposition TSV written to {tsv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
