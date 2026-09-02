#!/usr/bin/env python3
"""Remove poisoned learned exemplars from data/adaptive_exemplars.json (DRY-RUN FIRST).

2026-08-15: the tone-corroborated gate veto's no_search teacher recorded an
explicit search command ("Look it up it's pretty funny") and a third-party
news share (the Strait of Hormuz message) as web_search/no_search anchors —
the veto itself was wrong (sticky tone floor + any-statement test), so the
teacher poisoned the store. The inflow is fixed in core/agentic/gate.py
(vent-shape gate on both the veto and the teacher); this script removes the
already-stored entries.

Safety model:
  * Default is DRY RUN — prints exactly which exemplars match and exits.
  * --apply writes a pre-image backup of the full store to
    data/backups/adaptive_exemplars_preimage_<ts>.json first, then removes
    only the matched entries via an atomic safe_json write.
  * Refuses --apply while a live Daemon main.py is detected — the running
    instance holds the store in memory and its next accepted record would
    re-save the pre-purge contents (2026-08-05 profile-clobber lesson).

Matching: case-insensitive substring against the exemplar text, scoped to
one domain/label. Every --match term must match at least one entry or the
script aborts (a typo must not silently purge nothing).

2026-08-21 extension (--non-vent): the 08-18 evening session taught 7+
no_search exemplars from political/news commentary through the
first-person-anywhere hole ("I mean it seems possible this was censored…",
"I think people would be arrested…"). Instead of hand-matching each, --non-vent
scores every learned web_search/no_search entry against THE DEPLOYED
core.agentic.gate._is_vent_shaped (which now strips epistemic markers) and
selects the entries that are NOT vent-shaped — i.e. entries the current
teacher would refuse to learn. Composable with --match.

Usage:
    python scripts/purge_adaptive_exemplars.py --list
    python scripts/purge_adaptive_exemplars.py --non-vent
    python scripts/purge_adaptive_exemplars.py --non-vent --apply
    python scripts/purge_adaptive_exemplars.py --match "strait of hormuz" --match "look it up"
    python scripts/purge_adaptive_exemplars.py --match "strait of hormuz" --match "look it up" --apply
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STORE_PATH = Path("data/adaptive_exemplars.json")


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception:
        pass
    # Fallback (guard module unavailable): old cmdline heuristic. Known hole:
    # a relative-path launch has no repo name in its cmdline (2026-08-21).
    try:
        out = subprocess.run(["pgrep", "-af", "main.py"], capture_output=True, text=True).stdout
        return any("Daemon_v1" in line for line in out.splitlines())
    except Exception:
        return False


def find_matches(data: dict, domain: str, label: str, terms: list) -> tuple:
    """Return (matched_entries, unmatched_terms). An entry matches if ANY term
    is a case-insensitive substring of its text."""
    entries = data.get(domain, {}).get(label, [])
    lowered = [t.lower() for t in terms]
    matched = []
    hit_terms = set()
    for e in entries:
        text = (e.get("text") or "").lower()
        hits = [t for t in lowered if t in text]
        if hits:
            matched.append(e)
            hit_terms.update(hits)
    unmatched = [t for t in lowered if t not in hit_terms]
    return matched, unmatched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="web_search")
    ap.add_argument("--label", default="no_search")
    ap.add_argument("--match", action="append", default=[],
                    help="case-insensitive substring of the exemplar text (repeatable)")
    ap.add_argument("--list", action="store_true", help="list all entries in domain/label and exit")
    ap.add_argument("--all", action="store_true",
                    help="select EVERY learned entry in domain/label (2026-09-01: "
                         "the intent/temporal_recall label was wholesale-poisoned by "
                         "pre-08-21 crisis-day teaching — sometimes the whole label "
                         "must reset to seeds)")
    ap.add_argument("--non-vent", action="store_true",
                    help="select web_search/no_search entries that fail the DEPLOYED "
                         "gate._is_vent_shaped test (entries the current teacher "
                         "would refuse to learn)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not STORE_PATH.exists():
        print(f"{STORE_PATH} does not exist — nothing to do.")
        return 0
    data = json.loads(STORE_PATH.read_text())
    entries = data.get(args.domain, {}).get(args.label, [])

    if args.list:
        print(f"{args.domain}/{args.label}: {len(entries)} learned exemplars")
        for i, e in enumerate(entries):
            print(f"  [{i}] ({e.get('source')}, {e.get('ts', '')[:19]}) {e.get('text')!r}")
        return 0

    if not args.match and not args.non_vent and not args.all:
        print("No --match terms, --non-vent, or --all given (use --list to inspect the store).")
        return 1

    matched, unmatched = ([], [])
    if args.all:
        entries = data.get(args.domain, {}).get(args.label, [])
        matched, unmatched = list(entries), []
    elif args.match:
        matched, unmatched = find_matches(data, args.domain, args.label, args.match)
    if args.non_vent:
        # Score against THE DEPLOYED vent-shape test — never a re-derivation.
        from core.agentic.gate import _is_vent_shaped
        if args.domain != "web_search" or args.label != "no_search":
            print("--non-vent only applies to web_search/no_search.")
            return 1
        _matched_ids = {id(e) for e in matched}
        for e in entries:
            if id(e) not in _matched_ids and not _is_vent_shaped(e.get("text") or ""):
                matched.append(e)
    print(f"{args.domain}/{args.label}: {len(entries)} entries, {len(matched)} matched\n")
    for e in matched:
        print(f"  REMOVE ({e.get('source')}, {e.get('ts', '')[:19]}) {e.get('text')!r}")
    if unmatched:
        print(f"\nABORT: no entry matched: {unmatched}")
        return 1
    if not matched:
        print("Nothing matched — nothing to do.")
        return 0

    if not args.apply:
        print("\nDRY RUN — re-run with --apply to back up + remove the matched entries.")
        return 0

    if _daemon_running():
        print("Refusing to run --apply: a live Daemon main.py is detected — it holds "
              "the store in memory and would re-save the pre-purge contents. "
              "Shut Daemon down first.")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("data/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"adaptive_exemplars_preimage_{ts}.json"
    shutil.copy2(STORE_PATH, backup_path)
    print(f"\nPre-image backup: {backup_path}")

    removed_ids = {id(e) for e in matched}
    data[args.domain][args.label] = [e for e in entries if id(e) not in removed_ids]

    from utils.safe_json import atomic_write_json
    atomic_write_json(str(STORE_PATH), data)
    print(f"Removed {len(matched)} entries and saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
