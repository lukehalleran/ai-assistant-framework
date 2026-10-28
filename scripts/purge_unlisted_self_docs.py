#!/usr/bin/env python3
"""Remove unreviewed repo-docs chunks from reference_docs (DRY-RUN FIRST).  class: BC-59, BC-58

Background (2026-10-27): the live seed globbed every top-level docs/*.md into
reference_docs (90 files incl. gitignored handoffs and private drafts) while the
reviewed self-docs set is ``config/reference_docs_manifest.txt``. The seed is fixed
(``gui.launch._seed_reference_docs``); this removes the historical pile.

Selection: reference_docs chunks whose metadata satisfies
``utils.self_docs_manifest.is_unlisted_repo_doc`` (file inside THIS install's docs/ dir
and not a valid manifest entry). Never a user_upload, never a chunk outside docs/.
Doc-shaped chunks recorded under some OTHER path (an older install, a restored store)
are listed for manual review and never selected.

Safety:
  - Refuses (dry run included) unless the manifest is fully valid
    (``manifest_problems()`` empty): an invalid entry would make reviewed docs look unlisted.
  - Default is DRY RUN: prints each title with its chunk count and a total.
  - --apply refuses while a live Daemon is detected (fail closed - no override), writes
    a pre-image JSONL (ids, documents, metadatas) to
    data/backups/purge_unlisted_self_docs_preimage_<ts>.jsonl, THEN deletes the ids.

Usage:
    python scripts/purge_unlisted_self_docs.py            # dry run
    python scripts/purge_unlisted_self_docs.py --apply    # pre-image + delete
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _daemon_running() -> bool:
    try:
        from utils.daemon_guard import daemon_running
        return daemon_running()
    except Exception as exc:  # fail CLOSED: without the real guard we cannot prove the Daemon is down
        print(f"[daemon-guard] utils.daemon_guard unavailable ({exc!r}) - treating the Daemon as RUNNING; "
              f"--apply is refused. Run from the repo root with the project interpreter.", file=sys.stderr)
        return True


def _open_collection():
    from config.app_config import CHROMA_PATH
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

    chroma = Path(CHROMA_PATH)  # a relative config path means the repo root, never the cwd
    store = MultiCollectionChromaStore(
        persist_directory=str(chroma if chroma.is_absolute() else ROOT / chroma))
    return store._get_collection("reference_docs")


def _rows(got):
    ids = got.get("ids") or []
    docs = got.get("documents") or [None] * len(ids)
    metas = got.get("metadatas") or [None] * len(ids)
    return [{"id": i, "document": doc, "metadata": meta or {}} for i, doc, meta in zip(ids, docs, metas)]


def select_unlisted(got) -> list:
    """Chunks (id, document, metadata) from a Chroma ``get`` result that are unlisted repo docs."""
    from utils.self_docs_manifest import is_unlisted_repo_doc

    return [r for r in _rows(got) if is_unlisted_repo_doc(r["metadata"])]


def foreign_docs(got) -> list:
    """Doc-shaped chunks from another install path: reported for review, never selected."""
    from utils.self_docs_manifest import is_foreign_repo_doc

    return [r for r in _rows(got) if is_foreign_repo_doc(r["metadata"])]


def run(coll, apply: bool, backup_dir: Path, daemon_running=_daemon_running) -> int:
    if apply and daemon_running():
        print("ABORT: a live Daemon main.py is detected (or could not be ruled out). "
              "Shut it down first; there is no override.")
        return 1

    from utils.self_docs_manifest import manifest_entries, manifest_problems
    problems = manifest_problems()
    if problems or not manifest_entries():
        # An unreadable, empty or partly invalid manifest makes reviewed docs look
        # "unlisted" - refuse rather than select (and on --apply delete) them.
        print("ABORT: config/reference_docs_manifest.txt is not fully valid; nothing selected, "
              "nothing deleted:\n  " + "\n  ".join(problems or ("no valid entries",)))
        return 1

    got = coll.get(include=["metadatas", "documents"])
    chosen = select_unlisted(got)
    per_title = Counter(str(c["metadata"].get("title") or "(untitled)") for c in chosen)
    for title in sorted(per_title):
        print(f"{title[:48]:<48} {per_title[title]:>6} chunks")
    print(f"\nTotal: {len(chosen)} chunks in {len(per_title)} unlisted repo docs.")
    other = foreign_docs(got)
    if other:
        paths = Counter(str(c["metadata"].get("file_path")) for c in other)
        print(f"\nNOT selected - {len(other)} doc-shaped chunks recorded under another install path "
              "(review by hand; [DAEMON DOCUMENTATION] and insight already hide them, the agentic tools do not):")
        for p in sorted(paths):
            print(f"  {p}  ({paths[p]} chunks)")

    if not chosen:
        print("Nothing to purge.")
        return 0
    if not apply:
        print("\nDRY RUN - nothing was deleted. Re-run with --apply (Daemon down) to purge "
              "(a pre-image JSONL is written first).")
        return 0

    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    preimage = backup_dir / f"purge_unlisted_self_docs_preimage_{ts}.jsonl"
    with open(preimage, "w") as fh:
        for c in chosen:
            fh.write(json.dumps(c, default=str) + "\n")
    print(f"Pre-image written to {preimage}")

    ids = [c["id"] for c in chosen]
    for i in range(0, len(ids), 500):
        coll.delete(ids=ids[i:i + 500])
    print(f"Deleted {len(ids)} chunks from reference_docs.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="Delete (after a pre-image backup). Default: dry run.")
    args = ap.parse_args()
    if args.apply and _daemon_running():  # refuse BEFORE opening the live store
        print("ABORT: a live Daemon main.py is detected (or could not be ruled out). "
              "Shut it down first; there is no override.")
        return 1
    return run(_open_collection(), args.apply, ROOT / "data" / "backups")


if __name__ == "__main__":
    sys.exit(main())
