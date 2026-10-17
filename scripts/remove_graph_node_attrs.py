#!/usr/bin/env python3
"""
Remove curated knowledge-graph NODE ATTRIBUTES by key — DRY-RUN FIRST, pre-image, undoable.

Graph-node counterpart of scripts/quarantine_graph_edges.py. A fact whose object
was not graph-worthy is stored as an attribute in the subject node's `metadata`
(memory_storage._ingest_fact_to_graph: `metadata={canon_rel: obj}`), e.g.
`user.social_habit`. Profile rows, Chroma facts and graph EDGES each have an
owner tool; node attributes had none (2026-10-16 habit cleanup).

Why REMOVAL with a pre-image, not a quarantine flag: no reader honours a flag on
attributes — and no reader renders them into a prompt either (the only consumer
that exposes them is the /api/graph SPA export, api/routes/system.py). Adding a
flag would need reader changes for no prompt-side gain, so the ladder's
"reversible by restore" rung is used: every removed (node, attr, value) is
written to a pre-image JSONL first and `--undo --from-preimage` puts it back.

Keys: `<node_id>.<attr>`, one per line ('#' comments) or repeated --key. The
node id is everything before the LAST '.', so ids containing dots work.
The owner_decision.txt lines written by scripts/report_claim_contamination.py
--emit-id-files are accepted as-is, e.g.
    node attribute (no tool): <label> -> present:user.social_habit;user.study_habit
(every `node.attr` after `present:` is taken; other text on the line is ignored).

Safety:
  - Default is DRY RUN: prints node, attr, current value; changes nothing.
  - --apply REFUSES while a live Daemon main.py runs (it holds the graph in memory
    and would re-save the old state). Pre-image first, then GraphMemory.save() (atomic).
  - Unknown node / attribute keys are reported and skipped (not fatal).
  - --undo restores only attributes that are currently ABSENT; a present, different
    value is reported and left alone.

Usage:
    python scripts/remove_graph_node_attrs.py --key user.social_habit --key user.study_habit
    python scripts/remove_graph_node_attrs.py --from-file owner_decision.txt --apply
    python scripts/remove_graph_node_attrs.py --undo --from-preimage data/backups/remove_graph_node_attrs_preimage_<ts>.jsonl --apply
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PREIMAGE_PREFIX = "remove_graph_node_attrs_preimage_"


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception as exc:  # fail CLOSED: without the real guard we cannot prove the Daemon is down
        print(f"[daemon-guard] utils.daemon_guard unavailable ({exc!r}) — treating the Daemon as RUNNING; "
              f"--apply is refused. Run from the repo root with the project interpreter.", file=sys.stderr)
        return True


def _split_key(token: str):
    token = token.strip()
    if "." not in token:
        return None
    node_id, attr = token.rsplit(".", 1)
    node_id, attr = node_id.strip().lower(), attr.strip()
    return (node_id, attr) if node_id and attr else None


def parse_keys(text: str) -> list:
    """Plain `node.attr` lines and report_claim_contamination owner_decision lines."""
    out = []
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "present:" in line:
            tokens = [t for t in line.split("present:", 1)[1].split(";")]
        elif " " in line:
            print(f"  ! skipping malformed line: {raw!r}")
            continue
        else:
            tokens = [line]
        for tok in tokens:
            if not tok.strip():
                continue
            pair = _split_key(tok)
            if pair is None:
                print(f"  ! skipping malformed key: {tok!r}")
                continue
            if pair not in out:
                out.append(pair)
    return out


def _node_meta(gm, node_id):
    if not gm.graph.has_node(node_id):
        return None
    md = gm.graph.nodes[node_id].get("metadata")
    return md if isinstance(md, dict) else {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", action="append", default=[], help="node_id.attr (repeatable)")
    ap.add_argument("--from-file", help="file of keys / owner_decision lines")
    ap.add_argument("--graph", default=None, help="graph JSON path (default: config KNOWLEDGE_GRAPH_PERSIST_PATH)")
    ap.add_argument("--backup-dir", default="data/backups", help="pre-image directory (default data/backups)")
    ap.add_argument("--undo", action="store_true", help="restore attributes from --from-preimage")
    ap.add_argument("--from-preimage", help="with --undo: pre-image JSONL written by a previous --apply")
    ap.add_argument("--apply", action="store_true", help="actually modify the graph (pre-image first)")
    ap.add_argument("--force", action="store_true", help="with --apply: run even if a live Daemon is detected")
    args = ap.parse_args(argv)

    values = {}  # (node, attr) -> value, only for undo
    if args.undo:
        if not args.from_preimage:
            print("--undo needs --from-preimage <file>."); return 2
        keys = []
        for raw in Path(args.from_preimage).read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            rec = json.loads(raw)
            pair = (rec["node_id"], rec["attr"])
            keys.append(pair)
            values[pair] = rec["value"]
    else:
        keys = parse_keys("\n".join(f"{k}" for k in args.key))
        if args.from_file:
            for pair in parse_keys(Path(args.from_file).read_text(encoding="utf-8")):
                if pair not in keys:
                    keys.append(pair)
    if not keys:
        print("No keys given."); return 2

    if args.apply and _daemon_running() and not args.force:
        print("REFUSED: a live Daemon main.py is running — it holds the graph in memory and "
              "would re-save the pre-change state. Shut it down, then re-run with --apply.")
        return 3

    graph_path = args.graph
    if not graph_path:
        try:
            from config.app_config import KNOWLEDGE_GRAPH_PERSIST_PATH as graph_path  # noqa: N813
        except Exception:  # degrades: falls back to the conventional graph path
            graph_path = "data/knowledge_graph.json"
    from memory.graph_memory import GraphMemory
    gm = GraphMemory(persist_path=graph_path)
    gm.load()
    print(f"Graph: {graph_path} ({gm.node_count()} nodes, {gm.edge_count()} edges)")

    verb = "RESTORE" if args.undo else "REMOVE"
    todo = []
    for node_id, attr in keys:
        md = _node_meta(gm, node_id)
        if md is None:
            print(f"  ? node not found: {node_id}  (key {node_id}.{attr})")
            continue
        if args.undo:
            if attr in md:
                same = md[attr] == values[(node_id, attr)]
                print(f"  = {node_id}.{attr} already present ({'same value' if same else 'DIFFERENT value, left alone'})")
                continue
            print(f"  {node_id}.{attr} = {values[(node_id, attr)]!r}  -> would {verb}")
            todo.append((node_id, attr, values[(node_id, attr)]))
        else:
            if attr not in md:
                print(f"  ? attribute not found: {node_id}.{attr}")
                continue
            print(f"  {node_id}.{attr} = {md[attr]!r}  -> would {verb}")
            todo.append((node_id, attr, md[attr]))

    if not args.apply:
        print(f"\nDRY RUN — nothing was modified. Re-run with --apply (Daemon DOWN) to {verb} {len(todo)} attribute(s).")
        return 0
    if not todo:
        print("Nothing to do."); return 0

    if not args.undo:
        bdir = Path(args.backup_dir)
        bdir.mkdir(parents=True, exist_ok=True)
        pre = bdir / f"{PREIMAGE_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(pre, "w", encoding="utf-8") as fh:
            for node_id, attr, value in todo:
                fh.write(json.dumps({"node_id": node_id, "attr": attr, "value": value}, default=str) + "\n")
        print(f"Pre-image written: {pre}")

    for node_id, attr, value in todo:
        data = gm.graph.nodes[node_id]
        md = data.get("metadata")
        if not isinstance(md, dict):
            md = {}
            data["metadata"] = md
        if args.undo:
            md[attr] = value
        else:
            md.pop(attr, None)
    gm._mark_dirty()
    gm.save(raise_on_error=True)
    tail = "" if args.undo else " Undo with --undo --from-preimage <that file> --apply."
    print(f"Applied: {verb} on {len(todo)} attribute(s).{tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
