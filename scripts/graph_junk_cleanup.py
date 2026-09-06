#!/usr/bin/env python3
"""
scripts/graph_junk_cleanup.py

Curated, dry-run-first cleanup of junk nodes in the knowledge graph
(data/knowledge_graph.json).  Fact ingestion has accreted many low-value
nodes — numeric/temporal fragments ("4pm", "11k"), conversational snippets
("already_ate", "a_short_walk"), and deg-0 orphan wiki titles — which pollute
graph query expansion and graph-boosted scoring.

Safety model (per CLAUDE.md "NEVER auto-delete user data"):
  * Default run is DRY-RUN.  It only ANALYZES and writes a *candidate file*
    you review/edit — it never touches the graph.
  * --apply removes ONLY the entity_ids left UNCOMMENTED in the candidate
    file, and backs up the graph JSON first.  You are the curator: comment
    out (prefix with '#') anything you want to keep before applying.

Workflow:
    python scripts/graph_junk_cleanup.py                 # dry-run -> writes candidate file
    #   ... review / edit data/graph_junk_candidates.txt ...
    python scripts/graph_junk_cleanup.py --apply         # remove uncommented ids (backs up first)

A dry run never overwrites an existing candidate file (it may hold curation);
fresh scans go to a timestamped `.rescan-<ts>` sibling instead. (2026-08-14: a
pre-apply dry run clobbered a curated file's uncommented ids, so --apply
silently removed nothing.)

Tiers in the candidate file:
    T1 (uncommented) — high-confidence mechanical junk: numeric/temporal/
                       measurement fragments, bare stopwords, verb-phrase
                       fragments, and an explicit fragment word list.
    #T2#             — conversational snippets (leading filler word + phrase).
                       Commented out: likely junk, but review before removing.
    #T3#             — deg-0 orphan nodes (no edges; dead weight). Commented.

Never flagged: the "user" node, personal entity types (person/pet/org/
location), and any node with degree >= 3 (well-connected → probably real).
"""

import argparse
import os
import re
import sys

# Make repo root importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.graph_memory import GraphMemory  # noqa: E402
from memory.graph_utils import (  # noqa: E402
    _TEMPORAL_RE, _MEASUREMENT_RE, _FREQUENCY_RE, _VERB_STEMS, _STOPWORDS,
)

DEFAULT_GRAPH = os.path.join("data", "knowledge_graph.json")
DEFAULT_CANDIDATES = os.path.join("data", "graph_junk_candidates.txt")

PROTECTED_ID = "user"
PROTECTED_TYPES = frozenset({"person", "pet", "organization", "location"})
KEEP_DEGREE = 3  # nodes with degree >= this are assumed real, never flagged

# Explicit common-noun/participle fragments that were ingested as entities.
FRAGMENT_WORDS = frozenset({
    "poor", "support", "tests", "test", "videos", "video", "done", "homework",
    "stuff", "applications", "application", "design", "designs", "and", "the",
    "personal", "thing", "things", "point", "points", "part", "parts",
})

# Leading filler words that mark a display name as a conversational snippet
# rather than a named entity (only when followed by more tokens).
LEADING_FILLER = frozenset({
    "a", "an", "the", "about", "after", "along", "already", "another", "any",
    "anything", "at", "before", "being", "doing", "for", "from", "getting",
    "going", "had", "has", "have", "having", "his", "her", "hers", "i", "in",
    "is", "it", "its", "just", "maybe", "me", "more", "my", "no", "not", "of",
    "on", "once", "only", "or", "over", "some", "still", "that", "their",
    "them", "then", "there", "this", "to", "too", "until", "up", "very",
    "was", "way", "we", "went", "were", "what", "when", "with", "you", "your",
})

_PURE_NUM_RE = re.compile(r"^\d[\d.,]*[a-z%]{0,3}$", re.IGNORECASE)   # 11k, 4pm, 100
_ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE)         # 1st, 22nd


def _tokens(name: str) -> list:
    return [t for t in re.split(r"[\s_]+", name.strip().lower()) if t]


def _is_mechanical_junk(nid: str, dn: str) -> bool:
    """T1: deterministic fragment junk (numeric/temporal/measurement/etc.)."""
    for cand in (dn, nid):
        c = cand.strip().lower()
        if not c:
            continue
        if c in FRAGMENT_WORDS or c in _STOPWORDS:
            return True
        if _PURE_NUM_RE.match(c) or _ORDINAL_RE.match(c):
            return True
        if _TEMPORAL_RE.match(c) or _MEASUREMENT_RE.match(c) or _FREQUENCY_RE.match(c):
            return True
        if c[0].isdigit():
            return True
        toks = _tokens(c)
        if toks and toks[0] in _VERB_STEMS:
            return True
    return False


def _is_snippet(nid: str, dn: str) -> bool:
    """T2: conversational snippet — leading filler word plus more tokens."""
    for cand in (dn, nid):
        toks = _tokens(cand)
        if len(toks) >= 2 and toks[0] in LEADING_FILLER:
            return True
    return False


def classify(gm: GraphMemory):
    """Return {tier: [(entity_id, display_name, degree, edge_summary)]}."""
    tiers = {"T1": [], "T2": [], "T3": []}
    for nid in list(gm.graph.nodes()):
        if nid == PROTECTED_ID:
            continue
        data = gm.graph.nodes[nid]
        etype = data.get("entity_type", "other")
        if etype in PROTECTED_TYPES:
            continue
        deg = gm.graph.degree(nid)
        if deg >= KEEP_DEGREE:
            continue
        dn = data.get("display_name", nid)
        rels = gm.get_relations(nid, direction="both")
        edge_summary = ", ".join(
            f"{e.relation}->{e.target_id}" if e.source_id == nid
            else f"{e.source_id}-{e.relation}->" for e in rels[:4]
        ) or "(no edges)"

        if _is_mechanical_junk(nid, dn):
            tiers["T1"].append((nid, dn, deg, edge_summary))
        elif _is_snippet(nid, dn):
            tiers["T2"].append((nid, dn, deg, edge_summary))
        elif deg == 0:
            tiers["T3"].append((nid, dn, deg, edge_summary))
    for t in tiers:
        tiers[t].sort(key=lambda r: r[0])
    return tiers


def write_candidates(tiers, path):
    lines = [
        "# graph_junk_cleanup candidates — review then run: "
        "python scripts/graph_junk_cleanup.py --apply",
        "# UNCOMMENTED entity_ids are removed on --apply. Comment out (add '#') to KEEP.",
        "# Uncomment a #T2#/#T3# line to ALSO remove it.",
        f"# T1 mechanical junk (default remove): {len(tiers['T1'])}",
        f"# T2 conversational snippets (review):  {len(tiers['T2'])}",
        f"# T3 deg-0 orphans (review):            {len(tiers['T3'])}",
        "",
    ]
    for nid, dn, deg, es in tiers["T1"]:
        lines.append(f"{nid}    # dn={dn!r} deg={deg} | {es}")
    lines.append("")
    lines.append("# ---- T2: conversational snippets (uncomment to remove) ----")
    for nid, dn, deg, es in tiers["T2"]:
        lines.append(f"#T2# {nid}    # dn={dn!r} deg={deg} | {es}")
    lines.append("")
    lines.append("# ---- T3: deg-0 orphan nodes (uncomment to remove) ----")
    for nid, dn, deg, es in tiers["T3"]:
        lines.append(f"#T3# {nid}    # dn={dn!r} deg={deg} | {es}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def read_approved(path) -> list:
    """Return entity_ids from uncommented lines of the candidate file."""
    approved = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # id is the first whitespace-delimited token (before any '# ...' note)
            approved.append(line.split()[0])
    return approved


def backup(src) -> str:
    """Byte-copy the graph JSON to a timestamped .bak (avoids fs-guard)."""
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = f"{src}.bak-{stamp}"
    with open(src, "rb") as r, open(dst, "wb") as w:
        w.write(r.read())
    return dst


def main():
    ap = argparse.ArgumentParser(description="Curated knowledge-graph junk cleanup.")
    ap.add_argument("--graph", default=DEFAULT_GRAPH, help="graph JSON path")
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES, help="candidate file path")
    ap.add_argument("--apply", action="store_true",
                    help="remove uncommented ids in the candidate file (backs up first)")
    args = ap.parse_args()

    gm = GraphMemory(persist_path=args.graph)
    print(f"[graph] {gm.node_count()} nodes, {gm.edge_count()} edges @ {args.graph}")

    if not args.apply:
        tiers = classify(gm)
        # NEVER overwrite an existing candidate file — it may hold the
        # owner's curation (2026-08-14: a pre-apply dry run clobbered a
        # curated file's uncommented ids, so the subsequent --apply removed
        # nothing). Fresh scans go to a timestamped sibling instead.
        out_path = args.candidates
        if os.path.exists(out_path):
            from datetime import datetime
            base, ext = os.path.splitext(out_path)
            out_path = f"{base}.rescan-{datetime.now():%Y%m%d_%H%M%S}{ext}"
            print(f"[candidates] {args.candidates} exists (possibly curated) — "
                  f"writing fresh scan to {out_path} instead")
        write_candidates(tiers, out_path)
        print("\n=== DRY-RUN (no changes made) ===")
        print(f"  T1 mechanical junk (remove by default): {len(tiers['T1'])}")
        print(f"  T2 conversational snippets (review):     {len(tiers['T2'])}")
        print(f"  T3 deg-0 orphans (review):               {len(tiers['T3'])}")
        for t in ("T1", "T2", "T3"):
            sample = tiers[t][:8]
            if sample:
                print(f"\n  {t} sample:")
                for nid, dn, deg, es in sample:
                    print(f"    {nid!r:34} dn={dn!r:30} deg={deg}")
        print(f"\nCandidate file written: {out_path}")
        print("Review/edit it, then re-run with --apply to remove uncommented ids.")
        return

    # --apply — refuse while a live Daemon holds the graph in memory (its next
    # save would re-write the removed nodes; 2026-09-05: this script had no
    # guard while the other store-writing scripts did).
    try:
        from utils.daemon_guard import daemon_running
        if daemon_running():
            print("ERROR: a live Daemon (main.py) is running from this repo — shut it down "
                  "before --apply, or the in-memory graph will re-save the removed nodes.")
            sys.exit(2)
    except ImportError:
        pass
    if not os.path.exists(args.candidates):
        print(f"ERROR: candidate file not found: {args.candidates}\n"
              f"Run a dry-run first (no --apply) to generate it.")
        sys.exit(1)
    approved = read_approved(args.candidates)
    approved = [a for a in approved if a and a != PROTECTED_ID]
    if not approved:
        print("No uncommented entity_ids in candidate file — nothing to remove.")
        return

    bak = backup(args.graph)
    print(f"[backup] {bak}")
    removed = 0
    for nid in approved:
        if gm.remove_entity(nid):
            removed += 1
        else:
            print(f"  (skip, not found) {nid}")
    gm.save()
    print(f"\n[applied] removed {removed}/{len(approved)} nodes; "
          f"graph now {gm.node_count()} nodes, {gm.edge_count()} edges.")
    print(f"Backup preserved at {bak} — restore with: cp {bak} {args.graph}")


if __name__ == "__main__":
    main()
