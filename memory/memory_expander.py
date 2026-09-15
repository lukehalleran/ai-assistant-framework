"""
Memory Expander — temporal-window expansion around a ChromaDB document.

Contract:
    - Given a doc ID, fetches the anchor document and its chronological
      neighbors from the same collection.
    - **Summaries** get special treatment: instead of showing neighboring
      summaries, the expander retrieves the original conversation turns
      that were compressed into the summary (via temporal_anchor_start/end
      or source_doc_ids metadata).
    - Returns a dict with turns (each marked ``is_anchor``), collection,
      expansion method, and error info.
    - Only expands collections that store timestamped turns/entries.
    - Hygiene (quarantine/junk/supersession) is enforced on the ANCHOR too
      (F08, 2026-09-09) — not just neighbors — across every expansion path
      including the non-expandable-collection fallback and the summary
      anchor.
    - Caches results per ``(memory_id, window, collection)`` tuple, keyed
      additionally to a content+metadata fingerprint of the anchor and
      bounded by ``EXPANSION_CACHE_TTL_S``; a chroma mutation made via the
      curation engine notifies every live expander through
      ``notify_chroma_mutation()`` so no manual ``clear_cache()`` call is
      needed at the call site.

Public Interface:
    - MemoryExpander.expand(memory_id, window, collection) -> dict
    - MemoryExpander.clear_cache()
    - register_expander(expander) / notify_chroma_mutation(doc_id)

Dependencies:
    - memory.storage.multi_collection_chroma_store.MultiCollectionChromaStore
    - config.app_config (EXPAND_* constants)
"""

import hashlib
import logging
import threading
import time
import weakref
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import app_config as cfg
from memory.utils import is_junk_conversation_doc, is_junk_summary, is_quarantined
from utils.retrieval_outcome import RetrievalError

logger = logging.getLogger(__name__)

# Collections where temporal expansion is meaningful
EXPANDABLE_COLLECTIONS = frozenset({
    "conversations", "summaries", "reflections", "facts", "obsidian_notes",
})

# Second bound on cached expansions, beside fingerprint invalidation (F08,
# 2026-09-09): an expansion older than this is recomputed unconditionally,
# even if re-fetching the anchor to check its fingerprint were to fail.
EXPANSION_CACHE_TTL_S = 300

# ---------------------------------------------------------------------------
# Cross-mutation cache invalidation (F08): the curation engine mutates chroma
# documents directly (quarantine flips, content repairs) outside any
# MemoryExpander instance's knowledge. Registered expanders drop their whole
# cache when notified — simple and always correct, since a stale cache is
# the only failure mode being defended against here.
# ---------------------------------------------------------------------------
_REGISTERED_EXPANDERS: "weakref.WeakSet" = weakref.WeakSet()
_REGISTRY_LOCK = threading.Lock()


def register_expander(expander: "MemoryExpander") -> None:
    """Register an expander instance for cross-mutation cache invalidation.
    Weak reference only — no lifecycle coupling with the registry."""
    with _REGISTRY_LOCK:
        _REGISTERED_EXPANDERS.add(expander)


def notify_chroma_mutation(doc_id: str) -> None:
    """Invalidate every registered expander's cache after a chroma document
    mutation made outside the normal expand() path (2026-09-09, F08) — the
    curation engine's apply_change()/revert_change() flip quarantine flags
    and replace content directly on the collection, and an expander's cache
    has no other way to learn about it before its TTL lapses. *doc_id* is
    accepted for a future doc-scoped invalidation; today it clears the
    whole cache, which is simple and always correct.
    """
    with _REGISTRY_LOCK:
        expanders = list(_REGISTERED_EXPANDERS)
    for expander in expanders:
        try:
            expander.clear_cache()
        except Exception:
            logger.debug(
                "[MemoryExpander] notify_chroma_mutation: clear_cache failed", exc_info=True
            )


class MemoryExpander:
    """Expand a single memory hit into its surrounding temporal window."""

    def __init__(self, chroma_store):
        self._store = chroma_store
        # cache_key -> (anchor_fingerprint, result, cached_at_monotonic)
        self._cache: Dict[Tuple[str, int, Optional[str]], Tuple[Optional[str], dict, float]] = {}
        register_expander(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(
        self,
        memory_id: str,
        window: int = 3,
        collection: Optional[str] = None,
    ) -> dict:
        """Return the anchor document plus *window* neighbors on each side.

        For **summaries**, returns the original conversation turns that
        were compressed into the summary (using temporal anchor metadata).

        Args:
            memory_id: ChromaDB document ID to expand around.
            window: Number of neighbors on each side (clamped to
                ``EXPAND_MAX_WINDOW``).
            collection: Collection to search. If ``None``, tries every
                expandable collection until the doc is found.

        Returns:
            dict with keys:
                anchor_id, collection, expansion_method, turns,
                total_in_collection, error
        """
        window = max(1, min(window, cfg.EXPAND_MAX_WINDOW))

        cache_key = (memory_id, window, collection)
        cached = self._cache.get(cache_key)
        if cached is not None:
            cached_fingerprint, cached_result, cached_at = cached
            fresh_enough = (time.time() - cached_at) < EXPANSION_CACHE_TTL_S
            current_fingerprint = self._anchor_fingerprint(
                memory_id, cached_result.get("collection") or collection
            )
            if fresh_enough and current_fingerprint == cached_fingerprint:
                logger.debug("[MemoryExpander] Cache hit for %s", memory_id[:8])
                return cached_result
            # Stale by TTL or the anchor changed underneath us — drop and
            # recompute rather than serve a possibly-wrong cached result.
            self._cache.pop(cache_key, None)

        try:
            result = self._do_expand(memory_id, window, collection)
        except RetrievalError as e:
            # A failed store read (not-found stays None/[] elsewhere) — an
            # explicit, uncached error rather than caching the anchor's
            # last-known-good result under this failure (design doc: "the
            # expander caches its wrong error"). Any other exception still
            # propagates untouched.
            return {
                "anchor_id": memory_id,
                "collection": collection,
                "expansion_method": "timestamp_window",
                "turns": [],
                "total_in_collection": 0,
                "error": f"expansion_failed: {e.source}: {e.reason}",
            }
        fingerprint = self._anchor_fingerprint(memory_id, result.get("collection") or collection)
        self._cache[cache_key] = (fingerprint, result, time.time())
        return result

    def clear_cache(self) -> None:
        """Drop cached expansions (call between ReAct sessions)."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _anchor_fingerprint(self, memory_id: str, collection: Optional[str]) -> Optional[str]:
        """Cheap content+metadata hash of the anchor doc, used to detect a
        chroma mutation between an expand() call and a later cache hit
        (F08). One `get_by_id` — no window fetch. Returns None when the
        collection is unknown or the doc can no longer be found (a None
        never equals a real hash, so it forces a recompute)."""
        if not collection:
            return None
        try:
            doc = self._store.get_by_id(collection, memory_id)
        except Exception:
            doc = None
        if not doc:
            return None
        content = doc.get("content", "") or ""
        metadata = doc.get("metadata") or {}
        try:
            meta_items = sorted((str(k), str(v)) for k, v in metadata.items())
        except Exception:
            meta_items = []
        payload = f"{content}|{meta_items!r}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _hygiene_block_reason(content: str, metadata: dict, collection: str = "") -> Optional[str]:
        """Return a short reason string when *content*/*metadata* should
        never surface at retrieval, else None.

        Blocks on:
        - The document is quarantined by the curation engine
        - The document is junk (based on content and collection type)
        - The document is superseded (facts only)
        """
        if not content:
            return "empty content"

        metadata = metadata or {}

        if is_quarantined(metadata):
            return "quarantined"

        if collection == "conversations":
            if is_junk_conversation_doc(content=content):
                return "junk"
        elif collection in ("summaries", "reflections"):
            if is_junk_summary(content):
                return "junk"
        elif collection == "facts":
            if metadata.get("is_current") is False or metadata.get("superseded_by"):
                return "superseded"

        return None

    @staticmethod
    def _passes_hygiene(content: str, metadata: dict, collection: str = "") -> bool:
        """Check if a document passes basic hygiene filters.

        Returns False when:
        - The document is quarantined by the curation engine
        - The document is junk (based on content and collection type)
        - The document is superseded (facts only)
        """
        return MemoryExpander._hygiene_block_reason(content, metadata, collection) is None

    def _do_expand(
        self, memory_id: str, window: int, collection: Optional[str]
    ) -> dict:
        error_template = {
            "anchor_id": memory_id,
            "collection": collection,
            "expansion_method": "timestamp_window",
            "turns": [],
            "total_in_collection": 0,
            "error": None,
        }

        # --- resolve collection if not provided ---
        if collection:
            anchor_doc = self._store.get_by_id(collection, memory_id)
            if not anchor_doc:
                return {**error_template, "error": f"Document {memory_id[:8]} not found in {collection}"}
        else:
            anchor_doc, collection = self._find_doc_across_collections(memory_id)
            if not anchor_doc:
                return {**error_template, "error": f"Document {memory_id[:8]} not found in any expandable collection"}
            error_template["collection"] = collection

        # --- anchor hygiene (F08, 2026-09-09): a quarantined/superseded/
        # junk anchor must never be returned, regardless of collection —
        # this used to apply only to NEIGHBORS, so an explicitly requested
        # quarantined/superseded doc was served anyway. ---
        anchor_content = anchor_doc.get("content", "")
        anchor_meta = anchor_doc.get("metadata") or {}
        block_reason = self._hygiene_block_reason(anchor_content, anchor_meta, collection)
        if block_reason:
            return {
                **error_template,
                "collection": collection,
                "turns": [],
                "error": f"anchor suppressed: {block_reason}",
            }

        # --- check expandable ---
        if collection not in EXPANDABLE_COLLECTIONS:
            turn = self._doc_to_turn(anchor_doc, is_anchor=True, collection=collection)
            return {
                **error_template,
                "collection": collection,
                "turns": [turn],
                "total_in_collection": 1,
                "error": f"Collection '{collection}' does not support expansion; returning anchor only",
            }

        # --- summary special case: expand to source conversations ---
        if collection == "summaries":
            return self._expand_summary(anchor_doc, memory_id)

        # --- standard temporal window expansion ---
        return self._expand_temporal_window(anchor_doc, memory_id, window, collection)

    def _expand_summary(self, anchor_doc: dict, memory_id: str) -> dict:
        """Expand a summary to its source conversation turns.

        Strategy (in order):
        1. If metadata has ``source_doc_ids``, fetch those directly.
        2. If metadata has ``temporal_anchor_start`` / ``temporal_anchor_end``,
           fetch all conversations in that time range.
        3. Fall back to returning the summary anchor only with a note.
        """
        meta = anchor_doc.get("metadata") or {}
        anchor_turn = self._doc_to_turn(anchor_doc, is_anchor=True, collection="summaries")

        base = {
            "anchor_id": memory_id,
            "collection": "summaries",
            "expansion_method": "source_docs",
            "turns": [anchor_turn],
            "total_in_collection": 0,
            "error": None,
        }

        # --- Strategy 1: explicit source_doc_ids ---
        source_ids_raw = meta.get("source_doc_ids", "")
        if source_ids_raw:
            source_ids = [s.strip() for s in source_ids_raw.split(",") if s.strip()]
            if source_ids:
                turns = self._fetch_docs_by_ids("conversations", source_ids, memory_id)
                if turns:
                    base["turns"] = [anchor_turn] + turns
                    base["total_in_collection"] = len(turns)
                    return base

        # --- Strategy 2: temporal anchor range ---
        ts_start = meta.get("temporal_anchor_start", "")
        ts_end = meta.get("temporal_anchor_end", "")
        if ts_start and ts_end:
            turns = self._fetch_conversations_in_range(ts_start, ts_end, memory_id)
            if turns:
                base["turns"] = [anchor_turn] + turns
                base["total_in_collection"] = len(turns)
                return base

        # --- Fallback: no linkage metadata available ---
        base["error"] = (
            "Summary has no source_doc_ids or temporal anchors; "
            "returning summary text only"
        )
        return base

    def _fetch_docs_by_ids(
        self, collection: str, doc_ids: List[str], anchor_id: str
    ) -> List[dict]:
        """Fetch specific docs by ID and return as turn dicts."""
        turns = []
        for did in doc_ids:
            doc = self._store.get_by_id(collection, did)
            if doc:
                content = doc.get("content", "")
                metadata = doc.get("metadata") or {}
                # Skip docs that fail hygiene checks
                if not self._passes_hygiene(content, metadata, collection):
                    continue
                turns.append(self._doc_to_turn(doc, is_anchor=False, collection=collection))
        turns.sort(key=lambda t: (t.get("timestamp", ""), t.get("id", "")))
        return turns

    def _fetch_conversations_in_range(
        self, ts_start: str, ts_end: str, anchor_id: str
    ) -> List[dict]:
        """Fetch all conversations whose timestamp falls within [start, end].

        F05 (2026-09-09): uses the store's indexed
        `get_ids_by_timestamp_range()` + per-id fetch instead of
        `list_all()` — the prior implementation pulled the ENTIRE
        conversations collection into memory on every summary expansion
        that fell back to the temporal-anchor strategy.

        F9b (2026-09-14): a failed range read or a failed per-id read is no
        longer swallowed into `[]` (indistinguishable from a genuinely empty
        window) — both propagate as `RetrievalError` so `expand()` can
        report an explicit, uncached error instead. Only a genuine empty id
        list or a genuine per-id not-found still yields `[]`/skip.
        """
        try:
            doc_ids = self._store.get_ids_by_timestamp_range(
                "conversations", ts_start, ts_end
            )
        except RetrievalError:
            raise
        except Exception as e:
            logger.warning("[MemoryExpander] get_ids_by_timestamp_range failed: %s", e)
            raise RetrievalError(source="timestamp_range", reason=type(e).__name__) from e
        if not doc_ids:
            return []

        matched = []
        for doc_id in doc_ids:
            # No per-id guard: one failed read aborts the whole range fetch
            # rather than silently returning a partial turn list.
            doc = self._store.get_by_id("conversations", doc_id)
            if not doc:
                continue
            doc_meta = doc.get("metadata") or {}
            content = doc.get("content", "")
            # Skip docs that fail hygiene checks
            if not self._passes_hygiene(content, doc_meta, "conversations"):
                continue
            matched.append(doc)

        matched.sort(key=lambda d: self._sort_key(d))
        return [self._doc_to_turn(d, is_anchor=False, collection="conversations") for d in matched]

    def _expand_temporal_window(
        self, anchor_doc: dict, memory_id: str, window: int, collection: str
    ) -> dict:
        """Standard expansion: chronological neighbors in the same collection."""
        all_docs = self._store.list_all(collection)
        all_docs.sort(key=lambda d: self._sort_key(d))

        anchor_idx = None
        for i, doc in enumerate(all_docs):
            if doc.get("id") == memory_id:
                anchor_idx = i
                break

        if anchor_idx is None:
            turn = self._doc_to_turn(anchor_doc, is_anchor=True, collection=collection)
            return {
                "anchor_id": memory_id,
                "collection": collection,
                "expansion_method": "timestamp_window",
                "turns": [turn],
                "total_in_collection": len(all_docs),
                "error": "Anchor found by ID but missing from list_all; returning anchor only",
            }

        lo = max(0, anchor_idx - window)
        hi = min(len(all_docs), anchor_idx + window + 1)
        window_docs = all_docs[lo:hi]

        turns = []
        for doc in window_docs:
            is_anchor = doc.get("id") == memory_id
            content = doc.get("content", "")
            metadata = doc.get("metadata") or {}
            # Skip non-anchor docs that fail hygiene checks
            if not is_anchor and not self._passes_hygiene(content, metadata, collection):
                continue
            turns.append(self._doc_to_turn(doc, is_anchor=is_anchor, collection=collection))

        return {
            "anchor_id": memory_id,
            "collection": collection,
            "expansion_method": "timestamp_window",
            "turns": turns,
            "total_in_collection": len(all_docs),
            "error": None,
        }

    def _find_doc_across_collections(
        self, memory_id: str
    ) -> Tuple[Optional[dict], Optional[str]]:
        """Try each expandable collection until the doc is found."""
        for coll_name in sorted(EXPANDABLE_COLLECTIONS):
            doc = self._store.get_by_id(coll_name, memory_id)
            if doc:
                return doc, coll_name
        return None, None

    @staticmethod
    def _sort_key(doc: dict) -> Tuple[str, str]:
        """Sort by (timestamp, doc_id) for deterministic ordering."""
        meta = doc.get("metadata") or {}
        ts = meta.get("timestamp", "")
        doc_id = doc.get("id") or ""
        return (ts, doc_id)

    # Collections with long-form documents that need higher char limits
    _LONG_FORM_COLLECTIONS = frozenset({"obsidian_notes", "reference_docs"})

    @staticmethod
    def _doc_to_turn(doc: dict, is_anchor: bool = False, collection: str = "") -> dict:
        """Convert a raw ChromaDB doc dict into a turn record."""
        meta = doc.get("metadata") or {}
        content = doc.get("content", "")
        if collection in MemoryExpander._LONG_FORM_COLLECTIONS:
            char_limit = cfg.EXPAND_ANCHOR_CHAR_LIMIT_LONG if is_anchor else cfg.EXPAND_CONTEXT_CHAR_LIMIT_LONG
        else:
            char_limit = cfg.EXPAND_ANCHOR_CHAR_LIMIT if is_anchor else cfg.EXPAND_CONTEXT_CHAR_LIMIT
        if len(content) > char_limit:
            content = content[:char_limit] + "..."
        return {
            "id": doc.get("id", ""),
            "timestamp": meta.get("timestamp", ""),
            "content": content,
            "is_anchor": is_anchor,
        }
