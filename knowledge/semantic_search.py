# core/knowledge/semantic_search.py
"""
Semantic search with FAISS + SentenceTransformers, optimized for:
- one-time (lazy) loading of model, FAISS index, and metadata (thread-safe)
- optional offline mode for HF hubs
- graceful degradation if FAISS or metadata are missing
- predictable return schema compatible with existing callers

Public API:
    semantic_search_with_neighbors(query: str, k: int = 8) -> List[Dict[str, Any]]
        - Returns top-k results with fields:
          'text'/'content', 'source'/'namespace', 'similarity', 'timestamp', 'title'

This module is intentionally self-contained so it can be imported early
without heavy side-effects. Actual heavy resources are loaded on first use.
"""

from __future__ import annotations

import os
import time
import json
import threading
from typing import List, Dict, Any, Tuple

import numpy as np

# FAISS is optional; search will no-op if it's unavailable or files are missing
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from utils.logging_utils import get_logger
logger = get_logger("knowledge.semantic_search")

# ------------------------
# Configuration (env-tunable)
# ------------------------
EMBED_MODEL = os.getenv("SEM_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# HARDCODED PATHS - ignoring environment variables to avoid override issues
INDEX_PATH = "/run/media/lukeh/T9/wiki_data/vector_index_ivf.faiss"
META_PATH = "/run/media/lukeh/T9/wiki_data/metadata.parquet"

print(f"[DEBUG semantic_search.py] Using hardcoded INDEX_PATH: {INDEX_PATH}")
print(f"[DEBUG semantic_search.py] Using hardcoded META_PATH: {META_PATH}")

# Respect HF offline usage if you want to avoid network HEAD calls on boot
HF_OFFLINE  = os.getenv("HF_HUB_OFFLINE", "1") == "1"

# Small singletons to avoid repeated heavy loads across requests
_singleton_lock = threading.Lock()
_singleton: "SemanticSearchIndex | None" = None

# Guard to avoid spamming logs when index/meta are missing
_warned_missing = False


# ------------------------
# Helpers
# ------------------------
def _cuda_available() -> bool:
    """Cheap check for CUDA presence without importing torch at module import."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _load_embedder(name: str):
    """
    Load a SentenceTransformer with best-effort offline friendliness,
    and choose CUDA when available. Loading happens once per process.
    """
    # Avoid repeated remote HEADs; respect local cache/offline.
    # Try to use cached embedder first, fallback to loading directly
    try:
        from models.model_manager import ModelManager
        model = ModelManager._get_cached_embedder()
        logger.debug("Using cached embedder for semantic search")
        return model
    except Exception:
        # Fallback: load directly
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        if HF_OFFLINE:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from sentence_transformers import SentenceTransformer
        device = "cuda" if _cuda_available() else "cpu"
        model = SentenceTransformer(name, device=device)
        return model


# ------------------------
# Core index holder
# ------------------------
class SemanticSearchIndex:
    """
    Owns the embedder, FAISS index, and metadata.
    Loaded lazily (call load() or search() which calls load() if needed).
    """
    def __init__(self) -> None:
        self.embedder = None       # SentenceTransformer
        self.index = None          # faiss.Index
        self.meta = None           # pandas.DataFrame
        self.loaded = False

    def load(self) -> None:
        """Load model + FAISS + metadata once. Fast-return if already loaded."""
        global _warned_missing
        if self.loaded:
            return

        t0 = time.time()

        # DEBUG: Log the actual paths being checked
        logger.debug("[Semantic] Attempting to load: INDEX_PATH=%s, META_PATH=%s", INDEX_PATH, META_PATH)
        logger.debug("[Semantic] faiss=%s, index_exists=%s, meta_exists=%s",
                    faiss is not None, os.path.exists(INDEX_PATH), os.path.exists(META_PATH))

        # If FAISS or files missing, fail open (return empty on search)
        if not (faiss and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            if not _warned_missing:
                logger.error("[Semantic] Missing FAISS or metadata — fast-failing search "
                             "(index=%s, meta=%s)", INDEX_PATH, META_PATH)
                _warned_missing = True
            return

        try:
            # Embedder
            self.embedder = _load_embedder(EMBED_MODEL)

            # FAISS index
            self.index = faiss.read_index(INDEX_PATH)

            # Metadata (kept as a dataframe; if too big, use a sidecar JSON or sqlite)
            import pandas as pd  # local import to keep import-time cost low
            self.meta = pd.read_parquet(META_PATH)

            self.loaded = True
            logger.info("[Semantic] Loaded model=%s index=%s meta=%s in %.2fs",
                        EMBED_MODEL,
                        os.path.basename(INDEX_PATH),
                        os.path.basename(META_PATH),
                        time.time() - t0)
        except Exception as e:
            # Do not raise—degrade gracefully; callers get [] from search()
            logger.exception("[Semantic] Load failed: %s", e)

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode + normalize the query to float32 (shape: [1, dim]).
        SentenceTransformers can normalize internally, but we ensure dtype/shape.
        """
        vec = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        return vec  # already L2-normalized

    def _row_to_result(self, row, score: float) -> Dict[str, Any]:
        """
        Convert a metadata row into the expected result dict.
        Keeps back-compat with existing consumers.
        """
        # Prefer common text columns
        text = None
        for col in ("text", "content", "chunk_text", "passage"):
            if col in row.index and getattr(row, col) is not None:
                text = str(getattr(row, col))
                break
        if not text:
            # If no text field exists, skip the row.
            return {}

        # Determine a source-ish field (namespace/file/etc.)
        source = "unknown"
        for col in ("source", "namespace", "file", "document"):
            if col in row.index and getattr(row, col) is not None:
                source = str(getattr(row, col))
                break

        # Optional fields
        ts = getattr(row, "timestamp", "")
        title = getattr(row, "title", "")
        section = getattr(row, "section", "")
        section_level = getattr(row, "section_level", 0)
        chunk_index = getattr(row, "chunk_index", 0)

        return {
            "text": text,  # Return full text (no truncation)
            "content": text,  # Return full text (no truncation)
            "source": source,
            "namespace": source,
            "similarity": float(score),   # cosine similarity (IP on normalized vectors)
            "timestamp": ts,
            "title": title,
            "section": section,
            "section_level": section_level,
            "chunk_index": chunk_index,
        }

    def search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        """
        Top-k semantic search.
        - Returns [] if not loaded / resources missing
        - Keeps result shape compatible with previous implementation
        """
        if not query:
            return []

        if not self.loaded:
            self.load()
        if not self.loaded:
            # Still not ready (e.g., FAISS missing) -> no results
            return []

        # 1) Encode query once (normalized float32)
        q = self._encode_query(query)

        # 2) Search FAISS index (HNSW/IP, IVF/IP etc.)
        try:
            # FAISS returns (distances, indices); with normalized vectors + IP,
            # 'distances' are cosine similarities already in [-1, 1]
            D, I = self.index.search(q, int(max(1, k)))
        except Exception as e:
            logger.error("[Semantic] FAISS search error: %s", e, exc_info=True)
            return []

        # 3) Assemble rows from metadata
        rows: List[Dict[str, Any]] = []
        try:
            # iloc is faster than loc for int indices
            for idx, score in zip(I[0], D[0]):
                if idx < 0:
                    continue
                try:
                    row = self.meta.iloc[int(idx)]
                except Exception:
                    continue

                rec = self._row_to_result(row, score)
                if rec:
                    rows.append(rec)

            # Sort high->low cosine and cap to k
            rows.sort(key=lambda r: r["similarity"], reverse=True)
            return rows[:k]
        except Exception as e:
            logger.error("[Semantic] Result assembly error: %s", e, exc_info=True)
            return []


# ------------------------
# Module-level accessors
# ------------------------
def get_index() -> SemanticSearchIndex:
    """Return the process-wide singleton index holder."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = SemanticSearchIndex()
    return _singleton


# ------------------------
# Public API (kept stable)
# ------------------------
def semantic_search_with_neighbors(query: str, k: int = 8) -> List[Dict[str, Any]]:
    """
    Backwards-compatible wrapper that most of the codebase calls today.
    Usage remains:
        results = semantic_search_with_neighbors("your query", k=10)
    """
    return get_index().search(query, k=k)


# Optional: admin hook to force reload at runtime (if you update files on disk)
def reload_semantic_resources() -> None:
    """Force a full reload of embedder, FAISS index, and metadata."""
    global _singleton
    with _singleton_lock:
        _singleton = None  # next get_index() will rebuild
    logger.info("[Semantic] Resources scheduled for reload; will re-init on next query.")
"""
# knowledge/semantic_search.py

Module Contract
- Purpose: FAISS/embedding‑based semantic search across an offline corpus (e.g., Wikipedia parquet). Returns top‑k neighbors for a query with metadata.
- Inputs:
  - semantic_search_with_neighbors(query, k|top_k)
- Outputs:
  - List[dict] records with text/content/title/source/timestamp/similarity
- Side effects:
  - Loads FAISS index and metadata parquet; may cache index in memory.
"""
