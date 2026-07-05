#!/usr/bin/env python3
"""
Repair thinking-tag pollution in stored memories.

Historical leaks persisted literal <thinking>...</thinking> blocks and empty
<thinking></thinking> stream markers into the conversations/summaries/reflections
ChromaDB collections and corpus_v4.json. Those get replayed into prompts as
conversation history, which teaches the model to emit the tags itself.
As of 2026-06-10: 752/5920 conversation docs, 429/2000 corpus entries,
16 reflections, 1 summary were polluted.

This script strips the artifacts IN PLACE (content update — never deletes a
document, never drops a row). Documents are updated through the project's
MultiCollectionChromaStore so the collection's embedding function re-embeds
the cleaned text.

Usage:
    python scripts/repair_thinking_leaks.py            # dry run (default): report + samples
    python scripts/repair_thinking_leaks.py --apply    # actually write changes
    python scripts/repair_thinking_leaks.py --apply --skip-corpus   # ChromaDB only

Safety:
    - Dry run by default; --apply required to write anything.
    - Refuses to run with --apply while Daemon (main.py) is running.
    - corpus_v4.json is backed up to corpus_v4.json.bak.<timestamp> before writing.
    - ChromaDB pre-images of every updated doc are written to
      logs/repair_thinking_leaks_preimage.<timestamp>.jsonl before updating.
    - Content-bearing blocks are deleted only when response-LEADING (the
      verified leak shape); a mid-text <tag>...</tag> pair — possibly a
      conversation quoting the tags — loses only the tag literals, never
      the text between them.
    - A document whose cleaned text would be empty is left untouched and reported.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Tag patterns (mirror core/response_parser.py). _TAG covers both the
# think/thinking family and the reasoning/reason family (some models leak
# literal <reasoning> blocks into the content channel — observed 2026-07-03).
_TAG = r'(?:think(?:ing)?|reason(?:ing)?)'
EMPTY_PAIR_RE = re.compile(rf'<\|?{_TAG}\|?>\s*<\|?/{_TAG}\|?>', re.IGNORECASE)
# Content-bearing blocks are deleted ONLY when response-leading — at the start
# of the text, or right after an "Assistant:" marker in doc-shaped entries
# (every one of the 144 verified corpus leaks was response-leading reasoning).
# A mid-text <tag>...</tag> pair may be a stored conversation QUOTING the tags
# (e.g. a debugging session about this very bug) — deleting the span would
# destroy legitimate content, so mid-text pairs get just their tag literals
# stripped by ORPHAN_TAG_RE instead (kills the imitation vector, keeps the text).
LEADING_BLOCK_RE = re.compile(
    rf'(^\s*|Assistant:\s*)<{_TAG}>[\s\S]*?</{_TAG}>\s*', re.IGNORECASE
)
ORPHAN_TAG_RE = re.compile(rf'<\|?/?{_TAG}\|?>', re.IGNORECASE)
# A bare 'thinking>'/'reasoning>' fragment (opening '<' lost) is only treated
# as tag debris at the very start of the text — mid-text it's likely prose.
LEADING_FRAGMENT_RE = re.compile(rf'^\s*/?{_TAG}\|?>', re.IGNORECASE)
ANY_TAG_RE = re.compile(rf'</?{_TAG}>', re.IGNORECASE)

CHROMA_COLLECTIONS = ["conversations", "summaries", "reflections"]


def scrub_thinking(text: str) -> str:
    """Remove thinking artifacts from stored text.

    Order matters: empty pairs first (so they don't match as leading blocks and
    leave nothing suspicious behind), then response-leading content blocks
    (looped — a leak can be several back-to-back blocks), then remaining tag
    literals. Mid-text blocks deliberately lose only their tags, never their
    content (see LEADING_BLOCK_RE comment).
    """
    if not text:
        return text
    cleaned = EMPTY_PAIR_RE.sub('', text)
    while True:
        stripped = LEADING_BLOCK_RE.sub(r'\1', cleaned, count=1)
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = LEADING_FRAGMENT_RE.sub('', cleaned)
    cleaned = ORPHAN_TAG_RE.sub('', cleaned)
    # Collapse whitespace damage left by the removals
    cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def scrub_tags_only(text: str) -> str:
    """Fallback for documents that are ENTIRELY thinking content: removing the
    block would empty them (and we never delete), so strip just the literal
    tags. The reasoning text stays, but the model can no longer imitate the
    tag syntax from replayed history."""
    if not text:
        return text
    cleaned = LEADING_FRAGMENT_RE.sub('', text)
    cleaned = ORPHAN_TAG_RE.sub('', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def daemon_is_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-af", "python.*main.py"], capture_output=True, text=True
        ).stdout
        return any(
            line.strip().endswith(" main.py") or "Daemon_v1" in line
            for line in out.splitlines()
        )
    except Exception:
        return False  # pgrep unavailable — proceed, user was warned in docs


def repair_corpus(corpus_path: Path, apply: bool) -> dict:
    stats = {"scanned": 0, "polluted": 0, "repaired": 0, "skipped_empty": 0}
    samples = []
    data = json.loads(corpus_path.read_text())
    if not isinstance(data, list):
        print(f"  !! Unexpected corpus shape ({type(data).__name__}) — skipping")
        return stats

    for entry in data:
        if not isinstance(entry, dict):
            continue
        stats["scanned"] += 1
        for field in ("response", "query"):
            val = entry.get(field) or ""
            if not ANY_TAG_RE.search(val):
                continue
            stats["polluted"] += 1
            cleaned = scrub_thinking(val)
            if not cleaned:
                # All-thinking entry — keep content, strip only the tags
                cleaned = scrub_tags_only(val)
                stats["tags_only"] = stats.get("tags_only", 0) + 1
                if not cleaned:
                    stats["skipped_empty"] += 1
                    continue
            if len(samples) < 3:
                samples.append((val[:160], cleaned[:160]))
            entry[field] = cleaned
            stats["repaired"] += 1

    for before, after in samples:
        print(f"    BEFORE: {before!r}")
        print(f"    AFTER:  {after!r}\n")

    if apply and stats["repaired"]:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup = corpus_path.with_suffix(f".json.bak.{ts}")
        shutil.copy2(corpus_path, backup)
        print(f"  Backed up corpus to {backup.name}")
        corpus_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  Wrote repaired corpus ({stats['repaired']} fields cleaned)")
    return stats


def repair_chroma(apply: bool) -> dict:
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    from config.app_config import CHROMA_PATH

    persist_dir = CHROMA_PATH or "data/chroma_db_v4"
    print(f"  ChromaDB path: {persist_dir}")
    store = MultiCollectionChromaStore(persist_directory=persist_dir)

    backup_fh = None
    if apply:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup_path = PROJECT_ROOT / "logs" / f"repair_thinking_leaks_preimage.{ts}.jsonl"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_fh = backup_path.open("w", encoding="utf-8")
        print(f"  Pre-image backup: {backup_path}")

    totals = {"scanned": 0, "polluted": 0, "repaired": 0, "skipped_empty": 0}
    for cname in CHROMA_COLLECTIONS:
        col = store._get_collection(cname)
        got = col.get(include=["documents", "metadatas"])
        ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
        totals["scanned"] += len(ids)

        upd_ids, upd_docs, upd_metas = [], [], []
        shown = 0
        for doc_id, doc, meta in zip(ids, docs, metas):
            doc = doc or ""
            meta = dict(meta or {})
            meta_resp = meta.get("response") or ""
            meta_query = meta.get("query") or ""
            if not (ANY_TAG_RE.search(doc) or ANY_TAG_RE.search(meta_resp)
                    or ANY_TAG_RE.search(meta_query)):
                continue
            totals["polluted"] += 1

            new_meta = meta
            if cname == "conversations" and meta_resp:
                # Rebuild the doc from cleaned parts to keep the canonical
                # "User: ...\nAssistant: ..." shape used at write time.
                clean_resp = scrub_thinking(meta_resp)
                clean_query = scrub_thinking(meta_query) if meta_query else meta_query
                if not clean_resp:
                    # All-thinking response — keep content, strip only the tags
                    clean_resp = scrub_tags_only(meta_resp)
                    totals["tags_only"] = totals.get("tags_only", 0) + 1
                    if not clean_resp:
                        totals["skipped_empty"] += 1
                        continue
                new_doc = f"User: {clean_query}\nAssistant: {clean_resp}"
                new_meta = {**meta, "response": clean_resp, "query": clean_query}
            else:
                new_doc = scrub_thinking(doc)
                if not new_doc:
                    new_doc = scrub_tags_only(doc)
                    totals["tags_only"] = totals.get("tags_only", 0) + 1
                    if not new_doc:
                        totals["skipped_empty"] += 1
                        continue

            if shown < 2:
                print(f"    [{cname}] {doc_id}")
                print(f"      BEFORE: {doc[:140]!r}")
                print(f"      AFTER:  {new_doc[:140]!r}")
                shown += 1
            upd_ids.append(doc_id)
            upd_docs.append(new_doc)
            upd_metas.append(new_meta)

        if upd_ids and apply:
            # Write the pre-image of every doc we are about to change, so the
            # repair is reversible (ChromaDB has no backup of its own here).
            id_set = set(upd_ids)
            for doc_id, doc, meta in zip(ids, docs, metas):
                if doc_id in id_set:
                    backup_fh.write(json.dumps(
                        {"collection": cname, "id": doc_id,
                         "document": doc, "metadata": meta},
                        ensure_ascii=False) + "\n")
            backup_fh.flush()
            # update() with documents re-embeds via the collection's embedding fn
            BATCH = 100
            for i in range(0, len(upd_ids), BATCH):
                col.update(
                    ids=upd_ids[i:i + BATCH],
                    documents=upd_docs[i:i + BATCH],
                    metadatas=upd_metas[i:i + BATCH],
                )
            print(f"  [{cname}] updated {len(upd_ids)} docs (re-embedded)")
        elif upd_ids:
            print(f"  [{cname}] would update {len(upd_ids)} docs")
        totals["repaired"] += len(upd_ids)
    if backup_fh is not None:
        backup_fh.close()
    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="Write changes (default is dry run)")
    ap.add_argument("--skip-corpus", action="store_true", help="Skip corpus JSON repair")
    ap.add_argument("--skip-chroma", action="store_true", help="Skip ChromaDB repair")
    ap.add_argument("--corpus", default=str(PROJECT_ROOT / "data" / "corpus_v4.json"),
                    help="Corpus JSON path")
    args = ap.parse_args()

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"=== Thinking-leak repair ({mode}) ===\n")

    if args.apply and daemon_is_running():
        print("ABORT: Daemon (main.py) appears to be running. Close it first — "
              "concurrent writes to ChromaDB from two processes are unsafe.")
        sys.exit(1)

    if not args.skip_corpus:
        corpus_path = Path(args.corpus)
        print(f"-- Corpus: {corpus_path}")
        if corpus_path.exists():
            stats = repair_corpus(corpus_path, args.apply)
            print(f"  corpus: {stats}\n")
        else:
            print("  not found, skipping\n")

    if not args.skip_chroma:
        print("-- ChromaDB collections: " + ", ".join(CHROMA_COLLECTIONS))
        stats = repair_chroma(args.apply)
        print(f"  chroma: {stats}\n")

    if not args.apply:
        print("Dry run complete. Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
