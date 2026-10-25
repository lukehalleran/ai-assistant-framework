"""
# knowledge/git_memory_loader.py

Module Contract
- Purpose: Populate and maintain the PROCEDURAL ChromaDB collection from git commit history.
- Class: GitMemoryLoader(chroma_store, repo_path)
- Key methods:
  - backfill(limit, include_diffs) -> int  [initial population, returns count stored]
  - incremental_update(include_diffs, max_commits, cancel) -> int  [THE sync method: new commits since
    the last hash; if the anchor is missing/unreachable (rewritten history) re-anchors at HEAD and
    indexes the HEAD commits absent from the collection, newest first, capped; returns count stored]
  - clear() -> int  [wipe procedural collection, returns count deleted]
- CLI usage (via main.py):
  - python main.py git-backfill [LIMIT]   # initial load
  - python main.py git-update             # incremental since last sync
  - python main.py git-status             # show collection stats
  - python main.py git-clear              # wipe procedural collection
- State:
  - Last synced hash persisted at <data root>/git_memory_last_hash.txt (data root =
    utils.bootstrap.get_user_data_dir(), never the CWD); repo root defaults to this checkout
- Dependencies:
  - knowledge.git_memory.GitMemoryExtractor (commit extraction)
  - memory.storage.multi_collection_chroma_store.MultiCollectionChromaStore (vector storage)
  - config.app_config.CHROMA_PATH (default store path)
- Side effects:
  - ChromaDB writes to 'procedural' collection
  - File I/O for last-hash tracking at <data root>/git_memory_last_hash.txt
- Incident (2026-10-24, class: BC-58 / BC-16): the anchor named a commit the 2026-09-03 history rewrite
  removed, so `git rev-list <anchor>..HEAD` failed and every sync was a silent "No new commits" for
  7 weeks; nothing ran it at startup/shutdown either (main.py now does, in-process, on the live store).
"""

import os
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set

from knowledge.git_memory import GitMemoryExtractor
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
import utils.bootstrap as bootstrap
from utils.safe_json import atomic_write_text

logger = logging.getLogger(__name__)

_LAST_HASH_NAME = "git_memory_last_hash.txt"
_COLLECTION = "procedural"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_META_PAGE = 5000
_ADD_CHUNK = 50  # commits per store write: a timed-out run keeps the chunks that landed


def last_hash_path() -> str:
    """Anchor file under the live data root (resolved per call, BC-82; never CWD-relative)."""
    return os.path.join(bootstrap.get_user_data_dir(), _LAST_HASH_NAME)


class GitMemoryLoader:
    """Load git commits into the PROCEDURAL ChromaDB collection."""

    def __init__(
        self,
        chroma_store: Optional[MultiCollectionChromaStore] = None,
        repo_path: Optional[str] = None,
    ):
        if chroma_store is None:
            from config.app_config import CHROMA_PATH  # lazy import: live-config
            chroma_store = MultiCollectionChromaStore(CHROMA_PATH)
        self.chroma_store = chroma_store
        # Repo root from the module location, not the CWD (BC-16): the daemon
        # may be launched from anywhere.
        self.extractor = GitMemoryExtractor(str(repo_path) if repo_path else str(_REPO_ROOT))

    def backfill(self, limit: int = 500, include_diffs: bool = True) -> int:
        """
        Initial population of PROCEDURAL collection.

        Args:
            limit: Max commits to ingest.
            include_diffs: Include --stat diff summaries (default True).

        Returns:
            Number of commits stored.
        """
        logger.info(f"Starting git memory backfill (limit={limit}, diffs={include_diffs})")

        commits = self.extractor.extract_commits(
            limit=limit,
            include_diffs=include_diffs,
        )

        # Batch add commits (single embedding pass + disk write)
        texts = [c["content"] for c in commits]
        metas = [c["metadata"] for c in commits]
        try:
            if hasattr(self.chroma_store, 'add_batch_to_collection') and texts:
                self.chroma_store.add_batch_to_collection(_COLLECTION, texts, metas)
                stored = len(commits)
            else:
                stored = 0
                for commit in commits:
                    try:
                        self.chroma_store.add_to_collection(
                            name=_COLLECTION,
                            text=commit["content"],
                            metadata=commit["metadata"],
                        )
                        stored += 1
                    except Exception as e:
                        logger.error(f"Failed to store commit {commit['id']}: {e}")
        except Exception as e:
            logger.error(f"Batch backfill failed, falling back to individual: {e}")
            stored = 0
            for commit in commits:
                try:
                    self.chroma_store.add_to_collection(
                        name=_COLLECTION, text=commit["content"], metadata=commit["metadata"],
                    )
                    stored += 1
                except Exception as exc:
                    logger.error(f"Failed to store commit {commit['id']}: {exc}")

        # Save last hash for incremental updates (first = newest)
        if commits:
            self._save_last_hash(commits[0]["metadata"]["full_hash"])

        logger.info(f"Backfill complete: {stored}/{len(commits)} commits stored")
        return stored

    def incremental_update(
        self,
        include_diffs: bool = True,
        max_commits: Optional[int] = None,
        cancel: Optional[threading.Event] = None,
    ) -> int:
        """
        Add new commits since last sync.

        A missing or unreachable anchor (history rewritten) is recovered, not
        treated as "nothing new": see ``_recover_from_head``.

        Args:
            include_diffs: Include --stat diff summaries.
            max_commits: Recovery cap (default config git_memory.max_commits_per_sync).
            cancel: Set by a caller that timed out; checked before the store
                write so a late thread neither writes nor moves the anchor.

        Returns:
            Number of new commits stored.
        """
        last_hash = self._load_last_hash()
        if not last_hash:
            return self._recover_from_head("anchor missing", include_diffs, max_commits, cancel)
        if not self.extractor.is_reachable(last_hash):
            return self._recover_from_head(
                "anchor unreachable (history rewritten?)", include_diffs, max_commits, cancel
            )

        commits = self.extractor.get_recent_since_hash(last_hash)
        if not commits:
            logger.info("No new commits since last update")
            return 0

        # Re-extract with diffs if requested (get_recent_since_hash uses
        # extract_commits internally but without diffs by default)
        if include_diffs:
            commits = self.extractor.extract_commits(
                limit=len(commits),
                include_diffs=True,
            )

        # Batch add new commits
        texts = [c["content"] for c in commits]
        metas = [c["metadata"] for c in commits]
        try:
            if hasattr(self.chroma_store, 'add_batch_to_collection') and texts:
                self.chroma_store.add_batch_to_collection(_COLLECTION, texts, metas)
                stored = len(commits)
            else:
                stored = 0
                for commit in commits:
                    try:
                        self.chroma_store.add_to_collection(
                            name=_COLLECTION, text=commit["content"], metadata=commit["metadata"],
                        )
                        stored += 1
                    except Exception as exc:
                        logger.error(f"Failed to store commit {commit['id']}: {exc}")
        except Exception as e:
            logger.error(f"Batch incremental failed: {e}")
            stored = 0
            for commit in commits:
                try:
                    self.chroma_store.add_to_collection(
                        name=_COLLECTION, text=commit["content"], metadata=commit["metadata"],
                    )
                    stored += 1
                except Exception as exc:
                    logger.error(f"Failed to store commit {commit['id']}: {exc}")

        if commits:
            self._save_last_hash(commits[0]["metadata"]["full_hash"])

        logger.info(f"Incremental update: {stored} new commits stored")
        return stored

    def clear(self) -> int:
        """
        Wipe the PROCEDURAL collection and reset sync state.

        Returns:
            Number of entries that were removed.
        """
        coll = self.chroma_store.collections.get(_COLLECTION)
        if coll is None:
            return 0

        count = coll.count()
        if count > 0:
            # Get all IDs and delete them
            all_data = coll.get(include=[])
            ids = all_data.get("ids", [])
            if ids:
                coll.delete(ids=ids)
            logger.info(f"Cleared {count} entries from PROCEDURAL collection")

        # Remove last-hash file
        try:
            os.remove(last_hash_path())
        except FileNotFoundError:
            pass

        return count

    # ------------------------------------------------------------------
    # Anchor recovery
    # ------------------------------------------------------------------

    def _existing_hashes(self):
        """(full hashes, short hashes) already in the collection. Metadata only —
        documents are never loaded. Raises when the collection cannot be read:
        without the dedup set a recovery would double-index."""
        getter = getattr(self.chroma_store, "_get_collection", None)
        coll = getter(_COLLECTION) if callable(getter) else self.chroma_store.collections.get(_COLLECTION)
        full: Set[str] = set()
        short: Set[str] = set()
        if coll is None:
            return full, short
        offset = 0
        while True:
            page = coll.get(include=["metadatas"], limit=_META_PAGE, offset=offset)
            metas = page.get("metadatas") or []
            for md in metas:
                md = md or {}
                if md.get("full_hash"):
                    full.add(str(md["full_hash"]))
                if md.get("commit_hash"):
                    short.add(str(md["commit_hash"]))
            if len(metas) < _META_PAGE:
                break
            offset += _META_PAGE
        return full, short

    def _recover_from_head(
        self,
        reason: str,
        include_diffs: bool,
        max_commits: Optional[int],
        cancel: Optional[threading.Event],
    ) -> int:
        """Index HEAD commits missing from the collection, then re-anchor at HEAD.

        Existing entries are never deleted (an entry whose commit left HEAD is
        only counted in the log line). The anchor moves only after the batch
        landed, so a failed/cancelled run is retried next time.
        """
        if max_commits is None:
            from config import app_config  # lazy import: live-config (BC-82)
            max_commits = app_config.GIT_MEMORY_MAX_COMMITS_PER_SYNC
        max_commits = max(1, int(max_commits))

        head_hashes = self.extractor.list_head_hashes()
        if not head_hashes:
            logger.warning(f"[GitSync] {reason} and HEAD has no readable history — nothing indexed")
            return 0

        existing_full, existing_short = self._existing_hashes()

        def _present(h: str) -> bool:
            return h in existing_full or h[:8] in existing_short

        missing_ordered = [h for h in head_hashes if not _present(h)]
        skipped = len(head_hashes) - len(missing_ordered)
        unreachable = len(existing_full - set(head_hashes))
        selected = missing_ordered[:max_commits]
        selected_set = set(selected)

        commits: List[Dict] = []
        if selected:
            window = head_hashes.index(selected[-1]) + 1
            extracted = self.extractor.extract_commits(limit=window, include_diffs=include_diffs)
            commits = [c for c in extracted if c["metadata"]["full_hash"] in selected_set]
            if len(commits) < len(selected):
                logger.warning(
                    f"[GitSync] extracted {len(commits)}/{len(selected)} wanted commits — anchor unchanged"
                )
                return 0

        landed = 0
        for start in range(0, len(commits), _ADD_CHUNK):
            if cancel is not None and cancel.is_set():
                logger.warning(
                    f"[GitSync] cancelled after {landed}/{len(commits)} commits — anchor unchanged, "
                    "landed chunks are kept (deduped next run)"
                )
                return landed
            chunk = commits[start:start + _ADD_CHUNK]
            texts = [c["content"] for c in chunk]
            metas = [c["metadata"] for c in chunk]
            if hasattr(self.chroma_store, "add_batch_to_collection"):
                self.chroma_store.add_batch_to_collection(_COLLECTION, texts, metas)
            else:
                for c in chunk:
                    self.chroma_store.add_to_collection(
                        name=_COLLECTION, text=c["content"], metadata=c["metadata"],
                    )
            landed += len(chunk)

        still_missing = len(missing_ordered) - landed
        if still_missing > 0:
            # Never advance the anchor past work that did not land (BC-27/BC-95):
            # the next run re-enters recovery, dedups, and drains the next batch.
            logger.info(
                f"[GitSync] {reason} — re-anchor deferred: {still_missing} commits still missing "
                f"(cap {max_commits}); indexed {landed} this run, skipped {skipped} already present"
            )
            return landed

        self._save_last_hash(head_hashes[0])
        logger.info(
            f"[GitSync] {reason} — re-anchored at {head_hashes[0][:8]}, indexed {landed}, "
            f"skipped {skipped} already present; {unreachable} indexed hashes no longer on HEAD (kept)"
        )
        return landed

    # ------------------------------------------------------------------
    # Last-hash persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _save_last_hash(hash_val: str) -> None:
        atomic_write_text(last_hash_path(), hash_val)

    @staticmethod
    def _load_last_hash() -> Optional[str]:
        try:
            with open(last_hash_path(), "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
