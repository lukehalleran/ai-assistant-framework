"""
Loader to populate PROCEDURAL ChromaDB collection from git history.

Usage:
    python main.py git-backfill [LIMIT]   # initial load (default: all commits)
    python main.py git-update             # incremental since last sync
    python main.py git-status             # show collection stats
    python main.py git-clear              # wipe procedural collection
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
