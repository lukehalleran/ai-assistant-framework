# memory/thread_store.py
"""
ChromaDB-backed storage and retrieval for open threads.

Module Contract
- Purpose: Persists OpenThread objects in the 'threads' ChromaDB collection
  with semantic search, status filtering, priority ranking, and cap enforcement.
- Inputs:
  - chroma_store: MultiCollectionChromaStore instance
  - OpenThread objects for storage/retrieval
- Outputs:
  - Stored threads with document IDs
  - Priority-ranked open threads for session surfacing
  - Semantic query results as OpenThread lists
- Key behaviors:
  - Embedding text is topic + summary (for semantic matching)
  - Full thread data stored in metadata via to_metadata()
  - Status updates via delete-and-re-add (ChromaDB lacks native update)
  - Lazy staleness: is_stale() checked at retrieval time
  - Cap enforcement: oldest low-priority threads pruned when over limit
  - Duplicate detection helpers: topics_equivalent() (strict, for cascading a
    resolution to duplicate threads) and threads_duplicate() (looser, for
    dropping just-extracted threads that duplicate an existing/resolved one);
    _norm_keywords() is the single keyword normalizer (digit tokens kept) —
    quick-resolution matching uses it on both message and thread sides too
  - check_quick_resolutions (2026-07-25): resolution signals include past-action
    verbs ("did the"/"sent"/"emailed" — question + second-person guarded) and a
    signal-adjacent-object rule with a digit-conflict guard ("I did the email
    yesterday" resolves the email thread; "did hw6" cannot resolve the hw7 thread)
  - touch_thread(): refresh last_referenced when a duplicate extraction shows
    the tracked task is still live
- Dependencies:
  - memory.storage.multi_collection_chroma_store (vector storage)
  - memory.thread_models (data models)
  - config.app_config (feature flags, thresholds)
"""

import re
import time
from typing import List, Optional

from utils.logging_utils import get_logger
from memory.thread_models import OpenThread, ThreadStatus

logger = get_logger("thread_store")

COLLECTION_NAME = "threads"

# ---------------------------------------------------------------------------
# Lightweight per-turn resolution detection (pure regex, no LLM)
# ---------------------------------------------------------------------------

_COMPLETION_SIGNALS = re.compile(
    r"\b(?:submitted|turned\s+in|finished|completed|done\s+with|handed\s+in|"
    r"wrapped\s+up|knocked\s+out|got\s+it\s+done|already\s+(?:did|done|submitted|finished)|"
    r"just\s+(?:submitted|finished|turned\s+in)|"
    # 2026-07-25: plain past-action verbs — "I did the email yesterday"
    # resolved nothing (no signal matched AND bag-of-words overlap was 1 < 2,
    # so the email thread kept surfacing past its deadline). Second-person
    # forms ("did you send…", "you sent…") are excluded: those are questions
    # about Daemon's actions, not the user reporting their own completion.
    r"(?<!you\s)(?:sent|emailed)\b|replied\s+to\b|"
    r"(?<!you\s)did\s+(?:the|my|that|it)\b"
    r")",
    re.IGNORECASE,
)

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in",
    "on", "for", "my", "user", "today", "due", "thread", "type", "summary",
})


def _norm_keywords(text: str) -> set:
    """Lowercased keyword set for duplicate matching.

    Keeps 3+ letter words plus any token containing a digit ("hw6", "10") —
    numeric tokens are what distinguish "Week 10 homework" from "Week 11
    homework", so they must survive normalization.
    """
    words = re.findall(r"[a-z]*\d[a-z\d]*|[a-z]{3,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def topics_equivalent(a: OpenThread, b: OpenThread) -> bool:
    """True when two threads' topics name the same task.

    Conservative near-subset match on topic keywords: the smaller topic's
    keywords must be ~fully contained in the other's ("Homework and paragraph
    due" vs "Homework and paragraph due Friday"). Used to cascade a resolution
    from one thread to its duplicate siblings, so false positives resolve a
    real task's thread — keep this strict.

    Min-side containment alone makes a 1-keyword topic ("Taxes") a wildcard
    that matches every topic naming that word ("File taxes by April 15") and
    cascade-resolves live, distinct tasks — so unless the keyword sets are
    identical, require >=2 shared keywords and the larger topic to be at
    least half covered.
    """
    ka, kb = _norm_keywords(a.topic), _norm_keywords(b.topic)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    shared = ka & kb
    if len(shared) < 2:
        return False
    return (len(shared) / min(len(ka), len(kb)) >= 0.9
            and len(shared) / max(len(ka), len(kb)) >= 0.5)


def threads_duplicate(a: OpenThread, b: OpenThread) -> bool:
    """True when two threads likely track the same underlying task.

    Looser than topics_equivalent(): also matches on strong topic+summary
    keyword Jaccard. Used to drop a just-extracted thread that duplicates a
    thread that is already open or was resolved this shutdown pass — a false
    positive here only skips storing a redundant row.
    """
    if topics_equivalent(a, b):
        return True
    ka = _norm_keywords(f"{a.topic} {a.summary}")
    kb = _norm_keywords(f"{b.topic} {b.summary}")
    if not ka or not kb:
        return False
    jaccard = len(ka & kb) / len(ka | kb)
    return jaccard >= 0.6


def check_quick_resolutions(user_message: str, open_threads: List[OpenThread]) -> List[str]:
    """Check if user's message signals completion of any open threads.

    Pure regex — no LLM, no DB queries. ~1ms.

    Args:
        user_message: The user's raw input text.
        open_threads: Currently open threads to check against.

    Returns:
        List of thread_ids that should be resolved.
    """
    if not user_message or not open_threads:
        return []

    # Question sentences are not completion reports ("What did the email
    # say?" must not resolve the email thread) — only declaratives count.
    sentences = re.split(r"(?<=[.!?])\s+", user_message)
    declarative = " . ".join(p for p in sentences if not p.rstrip().endswith("?"))
    msg_lower = declarative.lower()

    # Must contain at least one completion signal
    if not _COMPLETION_SIGNALS.search(msg_lower):
        return []

    # Signal-adjacent objects: the tokens right after a completion verb name
    # WHAT was completed ("did the EMAIL yesterday"). Lets a 1-keyword match
    # resolve when it is the completed object itself, without loosening the
    # global 2-keyword floor into a wildcard.
    object_words = set()
    for m in _COMPLETION_SIGNALS.finditer(msg_lower):
        tail = msg_lower[m.end():]
        object_words |= _norm_keywords(" ".join(tail.split()[:4]))

    # Same normalization on both sides (digit tokens kept: "hw6" is what
    # distinguishes numbered tasks, so "done with hw6" can't match hw7's thread
    # on generic words alone).
    msg_words = _norm_keywords(msg_lower)
    resolved_ids = []

    for thread in open_threads:
        keywords = _norm_keywords(f"{thread.topic} {thread.summary}")
        if not keywords:
            continue
        # Require at least 2 keyword overlaps (or 1 if topic has ≤2 keywords),
        # OR a direct hit on the completed object. The object shortcut is
        # blocked on a digit-token mismatch: numeric tokens are what
        # distinguish "homework 6" from hw7's thread, so a generic object
        # word ("homework") must not bridge conflicting numbers.
        overlap = msg_words & keywords
        min_required = min(2, len(keywords))
        _msg_digits = {w for w in msg_words if any(c.isdigit() for c in w)}
        _kw_digits = {w for w in keywords if any(c.isdigit() for c in w)}
        _digit_conflict = bool(_msg_digits and _kw_digits and not (_msg_digits & _kw_digits))
        _object_hit = bool(object_words & keywords) and not _digit_conflict
        if len(overlap) >= min_required or _object_hit:
            resolved_ids.append(thread.thread_id)
            logger.info(
                f"[QuickResolve] Thread '{thread.topic}' matched: "
                f"signal + keywords {overlap or (object_words & keywords)}"
            )

    return resolved_ids


class ThreadStore:
    """
    ChromaDB-backed store for open threads.

    Uses MultiCollectionChromaStore.add_to_collection() and
    query_collection() for all vector operations.
    """

    def __init__(self, chroma_store=None):
        self.chroma_store = chroma_store

    def _ensure_collection(self) -> bool:
        """Ensure the threads collection exists. Returns False if unavailable."""
        if not self.chroma_store:
            return False

        collections = getattr(self.chroma_store, "collections", None)
        if collections is None:
            return False

        if COLLECTION_NAME not in collections or collections[COLLECTION_NAME] is None:
            if hasattr(self.chroma_store, "create_collection"):
                try:
                    self.chroma_store.create_collection(COLLECTION_NAME)
                except Exception as e:
                    logger.error(f"[ThreadStore] Failed to create collection: {e}")
                    return False
            else:
                return False

        return True

    def store_thread(self, thread: OpenThread) -> Optional[str]:
        """
        Store a thread in ChromaDB.

        Args:
            thread: OpenThread to store

        Returns:
            Document ID if stored, None if failed
        """
        if not self._ensure_collection():
            logger.warning("[ThreadStore] ChromaDB not available, cannot store")
            return None

        try:
            embedding_text = thread.to_embedding_text()
            metadata = thread.to_metadata()

            doc_id = self.chroma_store.add_to_collection(
                COLLECTION_NAME, embedding_text, metadata
            )

            logger.info(
                f"[ThreadStore] Stored thread {doc_id}: "
                f"'{thread.topic}' ({thread.thread_type.value})"
            )
            return doc_id

        except Exception as e:
            logger.error(f"[ThreadStore] Failed to store thread: {e}")
            return None

    def list_open_threads(self) -> List[OpenThread]:
        """Get all threads with OPEN status."""
        if not self._ensure_collection():
            return []

        try:
            all_items = self.chroma_store.list_all(COLLECTION_NAME)
            threads = []
            for item in all_items:
                meta = item.get("metadata") or {}
                if meta.get("status") == ThreadStatus.OPEN.value:
                    try:
                        threads.append(OpenThread.from_metadata(meta))
                    except Exception:
                        continue
            return threads
        except Exception as e:
            logger.error(f"[ThreadStore] list_open_threads failed: {e}")
            return []

    def get_top_threads(
        self,
        max_results: int = 3,
        stale_days: Optional[int] = None,
        deadline_grace_hours: Optional[int] = None,
    ) -> List[OpenThread]:
        """
        Get top priority open threads for session surfacing.

        Filters out stale threads (marks them STALE lazily),
        then ranks remaining by priority_score().

        Args:
            max_results: Maximum threads to return
            stale_days: Days without reference before marking stale (default from config)
            deadline_grace_hours: Hours past deadline before marking stale (default from config)

        Returns:
            List of OpenThread objects, ranked by priority score
        """
        if stale_days is None:
            try:
                from config.app_config import THREAD_STALE_DAYS
                stale_days = THREAD_STALE_DAYS
            except ImportError:
                stale_days = 14

        if deadline_grace_hours is None:
            try:
                from config.app_config import THREAD_DEADLINE_GRACE_HOURS
                deadline_grace_hours = THREAD_DEADLINE_GRACE_HOURS
            except ImportError:
                deadline_grace_hours = 48

        open_threads = self.list_open_threads()
        if not open_threads:
            return []

        # Lazy staleness check — mark stale threads and persist
        active = []
        for thread in open_threads:
            if thread.is_stale(stale_days, deadline_grace_hours):
                thread.mark_stale()
                self._update_thread(thread)
            else:
                active.append(thread)

        # Sort by priority score (highest first)
        active.sort(key=lambda t: t.priority_score(), reverse=True)
        return active[:max_results]

    def query_threads(self, query: str, n_results: int = 5) -> List[OpenThread]:
        """
        Semantic search for threads matching a query.

        Args:
            query: Search text
            n_results: Maximum results to return

        Returns:
            List of OpenThread objects, ranked by relevance
        """
        if not self._ensure_collection():
            return []

        try:
            coll = self.chroma_store.collections.get(COLLECTION_NAME)
            if coll is None or coll.count() == 0:
                return []

            results = self.chroma_store.query_collection(
                COLLECTION_NAME,
                query_text=query,
                n_results=min(n_results, coll.count()),
            )

            threads = []
            for match in results:
                meta = match.get("metadata") or {}
                if not meta.get("thread_id"):
                    continue
                try:
                    threads.append(OpenThread.from_metadata(meta))
                except Exception:
                    continue

            return threads[:n_results]

        except Exception as e:
            logger.error(f"[ThreadStore] Query failed: {e}")
            return []

    def resolve_thread(self, thread_id: str, resolution: str = "") -> bool:
        """
        Mark a thread as resolved.

        Args:
            thread_id: ID of the thread to resolve
            resolution: Optional resolution description

        Returns:
            True if resolved successfully
        """
        if not self._ensure_collection():
            return False

        try:
            all_items = self.chroma_store.list_all(COLLECTION_NAME)
            for item in all_items:
                meta = item.get("metadata") or {}
                if meta.get("thread_id") == thread_id:
                    thread = OpenThread.from_metadata(meta)
                    thread.mark_resolved(resolution)
                    # Delete old and re-store
                    doc_id = item.get("id")
                    coll = self.chroma_store.collections.get(COLLECTION_NAME)
                    if coll and doc_id:
                        coll.delete(ids=[doc_id])
                    self.store_thread(thread)
                    logger.info(f"[ThreadStore] Resolved thread {thread_id}: '{thread.topic}'")
                    return True

            logger.warning(f"[ThreadStore] Thread {thread_id} not found for resolution")
            return False

        except Exception as e:
            logger.error(f"[ThreadStore] resolve_thread failed: {e}")
            return False

    def touch_thread(self, thread: OpenThread) -> bool:
        """
        Refresh a thread's last_referenced timestamp and persist.

        Called when a new extraction duplicates an already-open thread: the
        duplicate is dropped, but the mention proves the task is still live,
        so keep the tracked thread from going stale.

        Args:
            thread: The open thread to refresh

        Returns:
            True if updated successfully
        """
        thread.last_referenced = time.time()
        return self._update_thread(thread)

    def enforce_cap(self, max_open: Optional[int] = None) -> int:
        """
        Enforce maximum open thread count by pruning lowest-priority threads.

        Args:
            max_open: Maximum number of open threads to keep (default from config)

        Returns:
            Number of threads pruned
        """
        if max_open is None:
            try:
                from config.app_config import THREAD_MAX_OPEN
                max_open = THREAD_MAX_OPEN
            except ImportError:
                max_open = 50

        open_threads = self.list_open_threads()
        if len(open_threads) <= max_open:
            return 0

        # Sort by priority (lowest first for pruning)
        open_threads.sort(key=lambda t: t.priority_score())
        to_prune = len(open_threads) - max_open
        pruned = 0

        for thread in open_threads[:to_prune]:
            thread.mark_stale()
            if self._update_thread(thread):
                pruned += 1

        if pruned:
            logger.info(f"[ThreadStore] Pruned {pruned} low-priority threads (cap={max_open})")

        return pruned

    def _update_thread(self, thread: OpenThread) -> bool:
        """
        Update a thread in ChromaDB (delete-and-re-add pattern).

        Args:
            thread: Thread with updated fields

        Returns:
            True if updated successfully
        """
        if not self._ensure_collection():
            return False

        try:
            all_items = self.chroma_store.list_all(COLLECTION_NAME)
            for item in all_items:
                meta = item.get("metadata") or {}
                if meta.get("thread_id") == thread.thread_id:
                    doc_id = item.get("id")
                    coll = self.chroma_store.collections.get(COLLECTION_NAME)
                    if coll and doc_id:
                        coll.delete(ids=[doc_id])
                    self.store_thread(thread)
                    return True
            return False
        except Exception as e:
            logger.error(f"[ThreadStore] _update_thread failed: {e}")
            return False
