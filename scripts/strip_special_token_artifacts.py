#!/usr/bin/env python3
"""Repair stored responses that begin/end with leaked chat-template special
tokens (<|sep|> etc.) — DRY-RUN FIRST.

2026-08-21: kimi-3 via OpenRouter intermittently emits <|sep|> as the first
content chunk; 11 corpus replies (plus their chroma copies) start
"<|sep|>That's ...". The inflow is fixed (ResponseParser.
strip_stream_special_tokens, folded into strip_trailing_stream_artifact so
storage + all display paths inherit); this script repairs the stored docs
by applying THE DEPLOYED strip — never a re-derivation.

Safety model:
  * Default is DRY RUN — prints exactly what would change and exits.
  * --apply writes a pre-image JSONL backup of every doc/entry it is about
    to modify, then rewrites in place (corpus via atomic safe_json write,
    chroma via update).
  * Refuses --apply while a live Daemon main.py is detected
    (utils.daemon_guard — cwd-based check).

Usage:
    python scripts/strip_special_token_artifacts.py           # dry run
    python scripts/strip_special_token_artifacts.py --apply
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CORPUS_PATH = Path("data/corpus_v4.json")


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    from core.response_parser import ResponseParser
    strip = ResponseParser.strip_stream_special_tokens

    # ── Corpus ────────────────────────────────────────────────────────
    corpus = json.loads(CORPUS_PATH.read_text())
    entries = corpus if isinstance(corpus, list) else corpus.get("entries", [])
    corpus_hits = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        resp = e.get("response") or ""
        fixed = strip(resp)
        if fixed != resp:
            corpus_hits.append((e, resp, fixed))
    print(f"Corpus: {len(corpus_hits)} entr(ies) with edge special tokens")
    for e, old, new in corpus_hits:
        print(f"  - [{str(e.get('timestamp'))[:19]}] {old[:60]!r} -> {new[:50]!r}")

    # ── Chroma conversations ──────────────────────────────────────────
    # Chroma docs embed the reply mid-document ("User: ... Assistant: <|sep|>...")
    # so the leaked token is not at a document EDGE — apply THE deployed strip
    # to the assistant segment specifically.
    def _strip_doc(doc: str) -> str:
        for label in ("Assistant: ", "Daemon: "):
            idx = doc.find(label)
            if idx >= 0:
                head = doc[: idx + len(label)]
                return head + strip(doc[idx + len(label):])
        return strip(doc)

    chroma_hits = []
    store = None
    try:
        from config.app_config import CHROMA_PATH
        from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
        store = MultiCollectionChromaStore(persist_directory=CHROMA_PATH)
        for d in store.list_all("conversations"):
            doc = d.get("content") or ""
            fixed = _strip_doc(doc)
            if fixed != doc:
                chroma_hits.append((d.get("id"), doc, fixed))
    except Exception as ex:
        print(f"Chroma scan failed: {ex}")
    print(f"Chroma conversations: {len(chroma_hits)} doc(s) with edge special tokens")
    for doc_id, old, new in chroma_hits[:20]:
        idx = old.find("<|")
        print(f"  - {doc_id[:12]}… ...{old[max(0, idx - 30):idx + 20]!r}...")

    if not corpus_hits and not chroma_hits:
        print("Nothing to repair.")
        return 0
    if not args.apply:
        print("\nDRY RUN — re-run with --apply to back up + repair.")
        return 0
    if _daemon_running():
        print("Refusing --apply: a live Daemon main.py is detected — it holds "
              "stores in memory and would re-save pre-repair contents. "
              "Shut Daemon down first.")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("data/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"special_token_repair_preimage_{ts}.jsonl"
    with open(backup_path, "w") as f:
        for e, old, _ in corpus_hits:
            f.write(json.dumps({"store": "corpus", "timestamp": str(e.get("timestamp")),
                                "response": old}) + "\n")
        for doc_id, old, _ in chroma_hits:
            f.write(json.dumps({"store": "chroma_conversations", "id": doc_id,
                                "document": old}) + "\n")
    print(f"\nPre-image backup: {backup_path}")

    for e, _, fixed in corpus_hits:
        e["response"] = fixed
    from utils.safe_json import atomic_write_json
    atomic_write_json(str(CORPUS_PATH), corpus)
    print(f"Corpus: repaired {len(corpus_hits)} entr(ies), saved atomically.")

    if chroma_hits:
        col = store._get_collection("conversations")
        for doc_id, _, fixed in chroma_hits:
            col.update(ids=[doc_id], documents=[fixed])
        print(f"Chroma: repaired {len(chroma_hits)} doc(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
