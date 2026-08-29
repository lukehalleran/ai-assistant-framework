#!/usr/bin/env python3
"""
Repair the 2026-08-23 wrong Zelphex-duration claims ("day 8" / "a week" when
the true figure was ~six weeks).

Incident: on 2026-08-23 (11:53-13:44), before Luke's correction at 13:46
("6 weeks off vryalr not 1"), three assistant responses asserted a ~1-week
current duration off Zelphex, and the 11:55 shutdown LLM extractor minted a
fact from the wrong response. The correction subsystem was dead on the GUI
path at the time (fixed 2026-08-23/24), so nothing updated. Inventory
(verified against live data 2026-08-24):

  corpus_v4.json           3 responses (ts 11:53:36, 13:39:09, 13:44:44)
  chroma conversations     3 mirror docs of the same turns
  chroma facts             fact_89f080f5_20260823_115553953
                           ("user | doctor_communication | uncertain sleep
                            and day 8 off Zelphex", is_current absent=true)
  user_profile.json        fact_id f353172c-4bd8-4ede-ad0b-8cba41a9a676
                           (categories.health, is_current=true — it also
                            superseded the 08-08 "prescriber acknowledged"
                            status, which stays non-current: it is itself a
                            stale moment-in-time report)

NOT touched (deliberately):
  - The July "day 7/8/9" conversations, summaries and profile excerpts —
    those were TRUE when said; only claims dated 2026-08-23 asserting a
    CURRENT ~1-week duration are wrong. Date scoping + exact phrases keep
    them out (the July doc says "day 8 or 9", which no pattern matches).
  - The 13:46 correction turn itself — it is the truth record.
  - profile raw_log — append-only history, never rendered as current state.

Repair doctrine (matches the 08-03 dedup-queue rules):
  - Conversation docs are ANNOTATED, never rewritten or deleted — a
    [CORRECTION ...] line is appended so any future retrieval of these turns
    is self-correcting, while the record of what was actually said stands.
  - The wrong fact is SUPERSEDED (is_current=False + reason), never deleted;
    the facts-collection supersession filter already drops non-current facts
    at retrieval.

Usage:
    python scripts/repair_zelphex_duration_claims.py           # dry run
    python scripts/repair_zelphex_duration_claims.py --apply   # write

Safety:
    - Dry run by default; --apply refuses while Daemon (main.py) is running
      (a live instance holds these stores in memory and would clobber the
      repair on its next save).
    - corpus_v4.json and user_profile.json are copied to timestamped .bak
      files before writing; writes are atomic (tmp + os.replace).
    - ChromaDB pre-images of every updated doc go to
      data/backups/zelphex_duration_preimage.<ts>.jsonl before updating.
    - Every change is idempotent: already-annotated docs and already-
      superseded facts are skipped, so re-running is safe.
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATE_PREFIX = "2026-08-23"

# Exact wrong-claim phrases from the three pre-correction responses. Kept
# narrow on purpose: the July "day 8 or 9" doc and the correction turn's own
# "six weeks off, not one" must never match.
WRONG_PATTERNS = [
    re.compile(r"day\s*8-?ish\s+off\s+Zelphex", re.IGNORECASE),
    re.compile(r"day\s*8\s+off\s+Zelphex", re.IGNORECASE),
    re.compile(r"a\s+week\s+out\s+from\s+quitting\s+Zelphex", re.IGNORECASE),
]

MARKER_PREFIX = "[CORRECTION 2026-08-23:"
CORRECTION_MARKER = (
    "\n\n[CORRECTION 2026-08-23: the timeline above is wrong — Luke was about "
    "six weeks off Zelphex at this point, not one week / \"day 8\". He "
    "corrected this later the same session (\"6 weeks off vryalr not 1\").]"
)

WRONG_FACT_CHROMA_ID = "fact_89f080f5_20260823_115553953"
WRONG_FACT_PROFILE_ID = "f353172c-4bd8-4ede-ad0b-8cba41a9a676"
SUPERSEDED_REASON = (
    "owner repair 2026-08-24: extracted from a response whose timeline the "
    "user corrected same-session (~six weeks off Zelphex, not day 8)"
)

CHROMA_COLLECTIONS = ["conversations", "summaries", "reflections"]


def _matches_wrong_claim(text: str) -> bool:
    return bool(text) and any(p.search(text) for p in WRONG_PATTERNS)


def daemon_is_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception:
        return False


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def repair_corpus(corpus_path: Path, apply: bool) -> int:
    print(f"\n== corpus: {corpus_path}")
    data = json.loads(corpus_path.read_text())
    if not isinstance(data, list):
        print(f"  !! Unexpected corpus shape ({type(data).__name__}) — skipping")
        return 0

    changed = 0
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        ts = str(entry.get("timestamp", ""))
        if not ts.startswith(DATE_PREFIX):
            continue
        for field in ("response", "content"):
            val = entry.get(field)
            if not isinstance(val, str) or not _matches_wrong_claim(val):
                continue
            if MARKER_PREFIX in val:
                print(f"  entry {i} [{field}] already annotated — skipped")
                continue
            snippet = next(
                p.search(val).group(0) for p in WRONG_PATTERNS if p.search(val)
            )
            print(f"  entry {i} ts={ts} [{field}]: contains {snippet!r}"
                  f" -> {'annotating' if apply else 'would annotate'}")
            if apply:
                entry[field] = val + CORRECTION_MARKER
            changed += 1

    if apply and changed:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup = corpus_path.with_suffix(f".json.bak.{ts}")
        shutil.copy2(corpus_path, backup)
        print(f"  Backed up corpus to {backup.name}")
        _atomic_write(corpus_path, json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  Wrote corpus ({changed} fields annotated)")
    return changed


def repair_chroma(apply: bool) -> int:
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from config.app_config import CHROMA_PATH

    persist_dir = CHROMA_PATH or "data/chroma_db_v4"
    print(f"\n== chroma: {persist_dir}")
    store = MultiCollectionChromaStore(persist_directory=persist_dir)

    backup_fh = None
    if apply:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup_path = PROJECT_ROOT / "data" / "backups" / \
            f"zelphex_duration_preimage.{ts}.jsonl"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_fh = backup_path.open("w", encoding="utf-8")
        print(f"  Pre-image backup: {backup_path}")

    changed = 0

    # --- Conversation-shaped docs: annotate ---
    for cname in CHROMA_COLLECTIONS:
        col = store._get_collection(cname)
        if col is None:
            print(f"  [{cname}] unavailable — skipped")
            continue
        got = col.get(include=["documents", "metadatas"])
        upd_ids, upd_docs, upd_metas, pre = [], [], [], []
        for doc_id, doc, meta in zip(got["ids"], got["documents"],
                                     got["metadatas"]):
            doc = doc or ""
            meta = meta or {}
            if not _matches_wrong_claim(doc):
                continue
            ts = str(meta.get("timestamp", ""))
            if ts and not ts.startswith(DATE_PREFIX):
                print(f"  [{cname}] {doc_id}: matches phrase but dated {ts[:10]}"
                      " — historically accurate, skipped")
                continue
            if not ts:
                print(f"  [{cname}] {doc_id}: matches phrase but has NO "
                      "timestamp — skipped for safety, review manually")
                continue
            if MARKER_PREFIX in doc:
                print(f"  [{cname}] {doc_id} already annotated — skipped")
                continue
            snippet = next(
                p.search(doc).group(0) for p in WRONG_PATTERNS if p.search(doc)
            )
            print(f"  [{cname}] {doc_id} ts={ts}: contains {snippet!r}"
                  f" -> {'annotating' if apply else 'would annotate'}")
            upd_ids.append(doc_id)
            upd_docs.append(doc + CORRECTION_MARKER)
            upd_metas.append(meta)
            pre.append({"collection": cname, "id": doc_id,
                        "document": doc, "metadata": meta})
        if apply and upd_ids:
            for row in pre:
                backup_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            backup_fh.flush()
            # update() with documents re-embeds via the collection's embedding fn
            col.update(ids=upd_ids, documents=upd_docs, metadatas=upd_metas)
            print(f"  [{cname}] updated {len(upd_ids)} docs (re-embedded)")
        changed += len(upd_ids)

    # --- The wrong fact: supersede (is_current=False), never delete ---
    col = store._get_collection("facts")
    if col is None:
        print("  [facts] unavailable — skipped")
    else:
        got = col.get(ids=[WRONG_FACT_CHROMA_ID],
                      include=["documents", "metadatas"])
        if not got["ids"]:
            print(f"  [facts] {WRONG_FACT_CHROMA_ID} not found — skipped")
        else:
            doc = got["documents"][0] or ""
            meta = dict(got["metadatas"][0] or {})
            if not _matches_wrong_claim(doc):
                print(f"  [facts] {WRONG_FACT_CHROMA_ID} content does not "
                      f"match the wrong claim ({doc[:80]!r}) — skipped")
            elif meta.get("is_current") is False:
                print(f"  [facts] {WRONG_FACT_CHROMA_ID} already superseded"
                      " — skipped")
            else:
                print(f"  [facts] {WRONG_FACT_CHROMA_ID}: {doc!r}"
                      f" -> {'superseding' if apply else 'would supersede'}"
                      " (is_current=False)")
                if apply:
                    backup_fh.write(json.dumps(
                        {"collection": "facts", "id": WRONG_FACT_CHROMA_ID,
                         "document": doc, "metadata": meta},
                        ensure_ascii=False) + "\n")
                    backup_fh.flush()
                    # chroma update() replaces metadata wholesale — merge first
                    meta.update({
                        "is_current": False,
                        "superseded_reason": SUPERSEDED_REASON,
                        "superseded_at": datetime.now().isoformat(),
                    })
                    col.update(ids=[WRONG_FACT_CHROMA_ID], metadatas=[meta])
                    print("  [facts] superseded")
                changed += 1

    if backup_fh is not None:
        backup_fh.close()
    return changed


def repair_profile(profile_path: Path, apply: bool) -> int:
    print(f"\n== profile: {profile_path}")
    data = json.loads(profile_path.read_text())
    changed = 0
    for cat, facts in (data.get("categories") or {}).items():
        if not isinstance(facts, list):
            continue
        for fc in facts:
            if not isinstance(fc, dict) or \
                    fc.get("fact_id") != WRONG_FACT_PROFILE_ID:
                continue
            if not _matches_wrong_claim(str(fc.get("value", ""))):
                print(f"  {WRONG_FACT_PROFILE_ID} found in {cat} but value "
                      f"does not match the wrong claim "
                      f"({fc.get('value')!r}) — skipped")
                continue
            if fc.get("is_current") is False:
                print(f"  {WRONG_FACT_PROFILE_ID} already superseded — skipped")
                continue
            print(f"  categories.{cat} fact {WRONG_FACT_PROFILE_ID}: "
                  f"{fc.get('relation')}={fc.get('value')!r}"
                  f" -> {'superseding' if apply else 'would supersede'}"
                  " (is_current=False)")
            if apply:
                fc["is_current"] = False
                fc["superseded_reason"] = SUPERSEDED_REASON
                fc["superseded_at"] = datetime.now().isoformat()
            changed += 1

    if apply and changed:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup = profile_path.with_suffix(f".json.bak.{ts}")
        shutil.copy2(profile_path, backup)
        print(f"  Backed up profile to {backup.name}")
        _atomic_write(profile_path, json.dumps(data, indent=2,
                                               ensure_ascii=False))
        print(f"  Wrote profile ({changed} fact superseded)")
    return changed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default: dry run)")
    ap.add_argument("--corpus-path", default="data/corpus_v4.json")
    ap.add_argument("--profile-path", default="data/user_profile.json")
    args = ap.parse_args()

    if args.apply and daemon_is_running():
        print("REFUSING --apply: Daemon (main.py) appears to be running. "
              "A live instance holds these stores in memory and its next "
              "save would clobber the repair. Shut it down first.")
        sys.exit(1)

    mode = "APPLY" if args.apply else "DRY RUN (no writes)"
    print(f"Zelphex-duration claim repair — {mode}")

    total = 0
    total += repair_corpus(Path(args.corpus_path), args.apply)
    total += repair_profile(Path(args.profile_path), args.apply)
    total += repair_chroma(args.apply)

    print(f"\n{'Applied' if args.apply else 'Would apply'} {total} changes.")
    if not args.apply:
        print("Re-run with --apply (Daemon must be shut down) to write.")


if __name__ == "__main__":
    main()
