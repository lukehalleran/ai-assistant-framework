#!/usr/bin/env python3
"""
Quarantine (or un-quarantine) specific `facts` documents by id — REVERSIBLE, DRY-RUN FIRST.

Instrument ladder (docs/AUTONOMOUS_CURATION_DESIGN.md): reversible metadata
beats deletion. A quarantined fact stays on disk but never surfaces — every
retrieval filter site checks `memory.utils.is_quarantined` (metadata
`curation_quarantined=True`). This is the owner-side by-id counterpart of the
Curation Center's quarantine flip, for facts a provenance audit names
directly (2026-09-02: `user | uses | WHOOP` sourced from a quoted model
comparison; `user | has_dog | Mochi/Waffles` invented from "playing with
Mochi and Waffles" — older graph metadata calls them cats).

Candidate file: one fact id per line ('#' comments allowed), or JSONL rows
carrying an "id" field (the purge scripts' candidate files work as-is).

Safety:
  - Default is DRY RUN (read-only): resolves every id, prints content + current
    quarantine state, modifies nothing.
  - --apply REFUSES to run while a live Daemon main.py is detected (it holds
    the store open). Writes a pre-image JSONL of each doc's metadata to
    data/backups/quarantine_facts_preimage_<ts>.jsonl BEFORE flipping.
  - --undo (with --apply) clears the flag instead of setting it.
  - Nothing is ever deleted; the knowledge graph is not touched (graph edges
    are handled by scripts/graph_junk_cleanup.py / purge_calibration_facts.py).

Usage:
    python scripts/quarantine_facts.py --from-file data/audit_fact_quarantine_candidates_20260902.jsonl
    python scripts/quarantine_facts.py --from-file ... --reason "provenance audit 2026-09-02" --apply
    python scripts/quarantine_facts.py --from-file ... --undo --apply
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUARANTINE_KEY = "curation_quarantined"


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without Chroma)
# ---------------------------------------------------------------------------

def parse_id_file(text: str):
    """Ids from a plain list (one per line, '#' comments) or JSONL rows with "id"."""
    ids = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            fid = str(row.get("id") or "").strip()
        else:
            fid = line.split("#", 1)[0].strip()
            if any(ch.isspace() for ch in fid):
                continue  # ids never contain whitespace — malformed line, skip
        if fid and fid not in ids:
            ids.append(fid)
    return ids


def plan_quarantine(docs, ids, *, undo=False):
    """Split requested ids into (to_change, already_in_state, missing)."""
    by_id = {d.get("id"): d for d in (docs or []) if d.get("id")}
    to_change, already, missing = [], [], []
    for fid in ids:
        doc = by_id.get(fid)
        if doc is None:
            missing.append(fid)
            continue
        flagged = bool((doc.get("metadata") or {}).get(QUARANTINE_KEY))
        if flagged != undo:
            already.append(doc)
        else:
            to_change.append(doc)
    return to_change, already, missing


def metadata_update(*, undo: bool, reason: str, ts: str) -> dict:
    if undo:
        return {QUARANTINE_KEY: False, "curation_unquarantined_at": ts}
    return {
        QUARANTINE_KEY: True,
        "curation_quarantine_reason": (reason or "owner_quarantine")[:120],
        "curation_quarantined_at": ts,
    }


def apply_plan(store, to_change, *, undo: bool, reason: str, ts: str) -> int:
    """Flip the flag on each doc via THE deployed update_metadata. Returns count."""
    update = metadata_update(undo=undo, reason=reason, ts=ts)
    changed = 0
    for doc in to_change:
        if store.update_metadata("facts", doc["id"], dict(update)):
            changed += 1
    return changed


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--from-file", required=True, help="candidate file (ids per line or JSONL with id)")
    ap.add_argument("--reason", default="owner_quarantine", help="stored as curation_quarantine_reason")
    ap.add_argument("--undo", action="store_true", help="clear the quarantine flag instead of setting it")
    ap.add_argument("--apply", action="store_true", help="actually flip metadata (pre-image backup first)")
    ap.add_argument("--force", action="store_true", help="with --apply: run even if a live Daemon is detected")
    args = ap.parse_args()

    if args.apply and _daemon_running() and not args.force:
        print("ABORT: a live Daemon main.py process is running — it holds ChromaDB open. "
              "Shut it down first (or --force if the match is stale).")
        return 1

    ids = parse_id_file(Path(args.from_file).read_text())
    if not ids:
        print("No ids in candidate file; nothing to do.")
        return 0

    from config.app_config import CHROMA_PATH
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
    docs = store.list_all("facts")
    to_change, already, missing = plan_quarantine(docs, ids, undo=args.undo)
    verb = "un-quarantine" if args.undo else "quarantine"

    print(f"Requested {len(ids)} id(s): {len(to_change)} to {verb}, "
          f"{len(already)} already in target state, {len(missing)} not found.")
    for d in to_change:
        md = d.get("metadata") or {}
        print(f"  - {d['id']}: {str(d.get('content', ''))[:80]!r} "
              f"(source={md.get('source')}, quarantined={bool(md.get(QUARANTINE_KEY))})")
    for fid in missing:
        print(f"  ? not found: {fid}")

    if not to_change:
        print("Nothing to change.")
        return 0
    if not args.apply:
        print(f"\nDRY RUN — nothing was modified. Re-run with --apply (Daemon DOWN) to {verb}.")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("data") / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    preimage = backup_dir / f"quarantine_facts_preimage_{ts}.jsonl"
    with open(preimage, "w") as f:
        for d in to_change:
            f.write(json.dumps(d, default=str) + "\n")
    print(f"Pre-image backup written to {preimage}")

    changed = apply_plan(store, to_change, undo=args.undo, reason=args.reason,
                         ts=datetime.now().isoformat())
    print(f"{verb.capitalize()}d {changed}/{len(to_change)} fact(s). Reversible: re-run with "
          f"{'--apply (without --undo)' if args.undo else '--undo --apply'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
