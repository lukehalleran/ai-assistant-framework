# core/knowledge/semantic_search.py
"""
Semantic search with FAISS IVFPQ + SentenceTransformers, optimized for:
- one-time (lazy) loading of model, FAISS index, and row-group offset table (thread-safe)
- zero-copy metadata: parquet file read on-demand per query via row-group offset
  index — no DataFrame loaded into RAM. Footprint: FAISS index (~2.2 GB) + embedder (~0.4 GB)
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

_DATA_ROOT = os.getenv("WIKI_DATA_ROOT", "/run/media/lukeh/T9")
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join(_DATA_ROOT, "wiki_data", "vector_index_ivf.faiss"))
META_PATH = os.getenv("FAISS_META_PATH", os.path.join(_DATA_ROOT, "wiki_data", "metadata.parquet"))

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

    Memory optimization: NO metadata DataFrame is kept in RAM. The full parquet
    (40M rows, ~33 GB text column) is never loaded. Instead, a row-group offset
    index is built at load time (~0 MB) and metadata is read on-demand from the
    parquet file for just the ~8 rows returned by each FAISS search.
    Total footprint: FAISS index (~2.2 GB) + embedder (~0.4 GB).
    """
    # Columns needed for search result assembly
    _RESULT_COLS = ["text", "title", "section", "section_level",
                    "chunk_index", "source", "timestamp"]

    def __init__(self) -> None:
        self.embedder = None       # SentenceTransformer
        self.index = None          # faiss.Index
        self.meta = None           # legacy compat — kept as None
        self._pq_file = None       # pyarrow.parquet.ParquetFile for on-demand reads
        self._rg_offsets: list[int] = []  # cumulative row offsets per row group
        self._total_rows: int = 0
        self.loaded = False

    def load(self) -> None:
        """Load model + FAISS + row-group index once. Fast-return if already loaded."""
        global _warned_missing
        if self.loaded:
            return

        t0 = time.time()

        logger.debug("[Semantic] Attempting to load: INDEX_PATH=%s, META_PATH=%s", INDEX_PATH, META_PATH)
        logger.debug("[Semantic] faiss=%s, index_exists=%s, meta_exists=%s",
                    faiss is not None, os.path.exists(INDEX_PATH), os.path.exists(META_PATH))

        if not (faiss and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            if not _warned_missing:
                logger.error("[Semantic] Missing FAISS or metadata — fast-failing search "
                             "(index=%s, meta=%s)", INDEX_PATH, META_PATH)
                _warned_missing = True
            return

        try:
            import pyarrow.parquet as pq

            # Embedder
            self.embedder = _load_embedder(EMBED_MODEL)

            # FAISS index (~2.2 GB for IVFPQ)
            self.index = faiss.read_index(INDEX_PATH)

            # Parquet handle + row-group offset table (no data loaded into RAM)
            self._pq_file = pq.ParquetFile(META_PATH)
            meta = self._pq_file.metadata
            offset = 0
            self._rg_offsets = []
            for rg_idx in range(meta.num_row_groups):
                self._rg_offsets.append(offset)
                offset += meta.row_group(rg_idx).num_rows
            self._total_rows = offset

            self.loaded = True
            logger.info("[Semantic] Loaded model=%s index=%s rows=%d rg=%d in %.2fs "
                        "(zero-copy metadata — text read on demand)",
                        EMBED_MODEL,
                        os.path.basename(INDEX_PATH),
                        self._total_rows,
                        meta.num_row_groups,
                        time.time() - t0)
        except Exception as e:
            logger.exception("[Semantic] Load failed: %s", e)

    def _find_row_group(self, row_idx: int) -> tuple[int, int]:
        """Return (row_group_index, local_offset) for a global row index.

        Uses binary search on the precomputed offset table.
        """
        import bisect
        rg = bisect.bisect_right(self._rg_offsets, row_idx) - 1
        return rg, row_idx - self._rg_offsets[rg]

    def _read_rows(self, indices: list[int], columns: list[str] | None = None) -> dict[int, dict[str, Any]]:
        """Read specific columns from parquet for a small set of row indices.

        Groups indices by row group so each row group is read at most once.
        Returns {global_row_idx: {col: value, ...}}.
        """
        if not self._pq_file or not indices:
            return {}

        cols = columns or self._RESULT_COLS
        # Intersect with actual schema
        available = set(self._pq_file.schema_arrow.names)
        cols = [c for c in cols if c in available]

        # Group indices by row group
        rg_map: dict[int, list[tuple[int, int]]] = {}  # rg_idx -> [(global_idx, local_offset)]
        for idx in indices:
            rg, local = self._find_row_group(idx)
            rg_map.setdefault(rg, []).append((idx, local))

        result: dict[int, dict[str, Any]] = {}
        for rg_idx, pairs in rg_map.items():
            try:
                table = self._pq_file.read_row_group(rg_idx, columns=cols)
                for global_idx, local in pairs:
                    row_data: dict[str, Any] = {}
                    for col in cols:
                        val = table.column(col)[local].as_py()
                        row_data[col] = val
                    result[global_idx] = row_data
            except Exception as e:
                logger.warning("[Semantic] Failed reading row group %d: %s", rg_idx, e)

        return result

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

    @staticmethod
    def _row_to_result(row_data: dict[str, Any], score: float) -> Dict[str, Any]:
        """Convert a metadata dict (from parquet read) into the expected result dict."""
        # Text is required
        text = None
        for col in ("text", "content", "chunk_text", "passage"):
            val = row_data.get(col)
            if val is not None:
                text = str(val)
                break
        if not text:
            return {}

        source = "unknown"
        for col in ("source", "namespace", "file", "document"):
            val = row_data.get(col)
            if val is not None:
                source = str(val)
                break

        return {
            "text": text,
            "content": text,
            "source": source,
            "namespace": source,
            "similarity": float(score),
            "timestamp": row_data.get("timestamp", ""),
            "title": row_data.get("title", ""),
            "section": row_data.get("section", ""),
            "section_level": row_data.get("section_level", 0),
            "chunk_index": row_data.get("chunk_index", 0),
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

        # 3) Collect valid FAISS hits
        hits: list[tuple[int, float]] = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            if int(idx) < self._total_rows:
                hits.append((int(idx), float(score)))

        if not hits:
            return []

        # 4) Batch-read metadata + text for matched rows only (on-demand from parquet)
        row_data_map = self._read_rows([i for i, _ in hits])

        # 5) Assemble result dicts
        rows: List[Dict[str, Any]] = []
        try:
            for idx, score in hits:
                data = row_data_map.get(idx)
                if not data:
                    continue
                rec = self._row_to_result(data, score)
                if rec:
                    rows.append(rec)

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
