"""
# knowledge/git_memory_loader.py

Module Contract
- Purpose: Populate and maintain the PROCEDURAL ChromaDB collection from git commit history.
- Class: GitMemoryLoader(chroma_store, repo_path)
- Key methods:
  - backfill(limit, include_diffs) -> int  [initial population, returns count stored]
  - incremental_update(include_diffs) -> int  [sync since last hash, returns count stored]
  - clear() -> int  [wipe procedural collection, returns count deleted]
- CLI usage (via main.py):
  - python main.py git-backfill [LIMIT]   # initial load
  - python main.py git-update             # incremental since last sync
  - python main.py git-status             # show collection stats
  - python main.py git-clear              # wipe procedural collection
- State:
  - Last synced hash persisted at data/git_memory_last_hash.txt
- Dependencies:
  - knowledge.git_memory.GitMemoryExtractor (commit extraction)
  - memory.storage.multi_collection_chroma_store.MultiCollectionChromaStore (vector storage)
  - config.app_config.CHROMA_PATH (default store path)
- Side effects:
  - ChromaDB writes to 'procedural' collection
  - File I/O for last-hash tracking at data/git_memory_last_hash.txt
"""

import os
import logging
from typing import Optional

from knowledge.git_memory import GitMemoryExtractor
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore

logger = logging.getLogger(__name__)

_LAST_HASH_FILE = os.path.join("data", "git_memory_last_hash.txt")
_COLLECTION = "procedural"


class GitMemoryLoader:
    """Load git commits into the PROCEDURAL ChromaDB collection."""

    def __init__(
        self,
        chroma_store: Optional[MultiCollectionChromaStore] = None,
        repo_path: str = ".",
    ):
        if chroma_store is None:
            from config.app_config import CHROMA_PATH
            chroma_store = MultiCollectionChromaStore(CHROMA_PATH)
        self.chroma_store = chroma_store
        self.extractor = GitMemoryExtractor(repo_path)

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

    def incremental_update(self, include_diffs: bool = True) -> int:
        """
        Add new commits since last sync.

        Returns:
            Number of new commits stored.
        """
        last_hash = self._load_last_hash()
        if not last_hash:
            logger.warning("No last hash found — run backfill first")
            return 0

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
            os.remove(_LAST_HASH_FILE)
        except FileNotFoundError:
            pass

        return count

    # ------------------------------------------------------------------
    # Last-hash persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _save_last_hash(hash_val: str) -> None:
        os.makedirs(os.path.dirname(_LAST_HASH_FILE), exist_ok=True)
        with open(_LAST_HASH_FILE, "w") as f:
            f.write(hash_val)

    @staticmethod
    def _load_last_hash() -> Optional[str]:
        try:
            with open(_LAST_HASH_FILE, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
