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
    - Caches results per ``(memory_id, window, collection)`` tuple.
    - Only expands collections that store timestamped turns/entries.

Public Interface:
    - MemoryExpander.expand(memory_id, window, collection) -> dict
    - MemoryExpander.clear_cache()

Dependencies:
    - memory.storage.multi_collection_chroma_store.MultiCollectionChromaStore
    - config.app_config (EXPAND_* constants)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import app_config as cfg

logger = logging.getLogger(__name__)

# Collections where temporal expansion is meaningful
EXPANDABLE_COLLECTIONS = frozenset({
    "conversations", "summaries", "reflections", "facts", "obsidian_notes",
})


class MemoryExpander:
    """Expand a single memory hit into its surrounding temporal window."""

    def __init__(self, chroma_store):
        self._store = chroma_store
        self._cache: Dict[Tuple[str, int, Optional[str]], dict] = {}

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
        if cache_key in self._cache:
            logger.debug("[MemoryExpander] Cache hit for %s", memory_id[:8])
            return self._cache[cache_key]

        result = self._do_expand(memory_id, window, collection)
        self._cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        """Drop cached expansions (call between ReAct sessions)."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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

        # --- check expandable ---
        if collection not in EXPANDABLE_COLLECTIONS:
            turn = self._doc_to_turn(anchor_doc, is_anchor=True)
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
        anchor_turn = self._doc_to_turn(anchor_doc, is_anchor=True)

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
                turns.append(self._doc_to_turn(doc, is_anchor=False))
        turns.sort(key=lambda t: (t.get("timestamp", ""), t.get("id", "")))
        return turns

    def _fetch_conversations_in_range(
        self, ts_start: str, ts_end: str, anchor_id: str
    ) -> List[dict]:
        """Fetch all conversations whose timestamp falls within [start, end]."""
        try:
            range_start = datetime.fromisoformat(ts_start)
            range_end = datetime.fromisoformat(ts_end)
        except (ValueError, TypeError):
            return []

        all_convos = self._store.list_all("conversations")
        matched = []
        for doc in all_convos:
            doc_meta = doc.get("metadata") or {}
            doc_ts_str = doc_meta.get("timestamp", "")
            if not doc_ts_str:
                continue
            try:
                doc_ts = datetime.fromisoformat(doc_ts_str)
            except (ValueError, TypeError):
                continue
            if range_start <= doc_ts <= range_end:
                matched.append(doc)

        matched.sort(key=lambda d: self._sort_key(d))
        return [self._doc_to_turn(d, is_anchor=False) for d in matched]

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
            turn = self._doc_to_turn(anchor_doc, is_anchor=True)
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
            turns.append(self._doc_to_turn(doc, is_anchor=is_anchor))

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

    @staticmethod
    def _doc_to_turn(doc: dict, is_anchor: bool = False) -> dict:
        """Convert a raw ChromaDB doc dict into a turn record."""
        meta = doc.get("metadata") or {}
        content = doc.get("content", "")
        char_limit = cfg.EXPAND_ANCHOR_CHAR_LIMIT if is_anchor else cfg.EXPAND_CONTEXT_CHAR_LIMIT
        if len(content) > char_limit:
            content = content[:char_limit] + "..."
        return {
            "id": doc.get("id", ""),
            "timestamp": meta.get("timestamp", ""),
            "content": content,
            "is_anchor": is_anchor,
        }
