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
        self._metric = None        # faiss metric_type of the loaded index (L2 vs IP)
        self.meta = None           # legacy compat — kept as None
        self._pq_file = None       # pyarrow.parquet.ParquetFile for on-demand reads
        self._rg_offsets: list[int] = []  # cumulative row offsets per row group
        self._total_rows: int = 0
        self.loaded = False
        self.disabled_reason: str = ""
        self._load_lock = threading.Lock()

    def load(self) -> None:
        """Publish one fully initialized index across concurrent cold searches."""
        if self.loaded:
            return
        # get_index() protects singleton construction, not initialization.
        # Searches run in multiple worker threads and can overlap after an
        # async timeout; without this lock both load the multi-GB index and
        # mutate its metadata offsets concurrently.
        with self._load_lock:
            if not self.loaded:
                self._load_once()

    def _load_once(self) -> None:
        """Load resources while holding this instance's initialization lock."""
        global _warned_missing

        t0 = time.time()

        logger.debug("[Semantic] Attempting to load: INDEX_PATH=%s, META_PATH=%s", INDEX_PATH, META_PATH)
        logger.debug("[Semantic] faiss=%s, index_exists=%s, meta_exists=%s",
                    faiss is not None, os.path.exists(INDEX_PATH), os.path.exists(META_PATH))

        if not (faiss and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            # 2026-09-03 (Codex audit #6): an ABSENT external index is an
            # expected disabled state (the ~2 GB wiki FAISS index is optional
            # and often unmounted) — log it once at INFO so real errors stand
            # out. Only "faiss cannot be imported while the files exist" is an
            # actual fault and stays at ERROR.
            files_present = os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)
            if faiss is None and files_present:
                self.disabled_reason = "faiss import failed"
                if not _warned_missing:
                    logger.error("[Semantic] FAISS index files exist but the faiss module "
                                 "cannot be imported — semantic wiki search unavailable")
                    _warned_missing = True
            else:
                missing = [p for p in (INDEX_PATH, META_PATH) if not os.path.exists(p)]
                self.disabled_reason = f"external index not present: {missing}"
                if not _warned_missing:
                    logger.info("[Semantic] External FAISS index not present — semantic wiki "
                                "search DISABLED for this process (missing=%s). Mount the index "
                                "or ignore if intentional.", missing)
                    _warned_missing = True
            return

        try:
            import pyarrow.parquet as pq

            # Embedder
            self.embedder = _load_embedder(EMBED_MODEL)

            # FAISS index (~2.2 GB for IVFPQ)
            self.index = faiss.read_index(INDEX_PATH)
            # Capture the index metric so search() can normalize scores correctly.
            # The wiki index is built L2 (build_faiss_index.py), so raw scores are
            # SQUARED DISTANCES (smaller = closer), not similarities — see _to_similarity().
            self._metric = int(self.index.metric_type)
            _metric_name = ("IP" if self._metric == faiss.METRIC_INNER_PRODUCT
                            else "L2" if self._metric == faiss.METRIC_L2
                            else f"#{self._metric}")
            logger.info("[Semantic] Index metric_type=%s — scores normalized to cosine-like similarity",
                        _metric_name)

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

    def _to_similarity(self, raw_score: float) -> float:
        """Normalize a raw FAISS score into a cosine-like similarity in [-1, 1].

        Polarity depends on the index metric:
        - INNER_PRODUCT: already cosine for normalized vectors → pass through.
        - L2 (the wiki index, per build_faiss_index.py): FAISS returns a SQUARED
          distance where SMALLER means closer. For unit-normalized vectors
          ‖q-x‖² = 2 - 2·cos, so cos = 1 - d/2. If the stored doc vectors are not
          unit-normalized this is approximate, but it stays MONOTONIC (smaller
          distance → higher similarity), which is what the rest of the pipeline
          needs: it sorts descending and gates on a 0–1 "cosine" threshold.

        Without this, an L2 index makes the caller layer inverted — it surfaces
        the FARTHEST neighbors first and rejects the closest matches.
        """
        if faiss is None or self._metric is None:
            return raw_score
        if self._metric == faiss.METRIC_INNER_PRODUCT:
            return raw_score
        # METRIC_L2 (or any distance-like metric): squared distance → cosine.
        return max(-1.0, min(1.0, 1.0 - (raw_score / 2.0)))

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

        # 2) Search FAISS index. Raw score polarity depends on the index metric:
        #    IP → larger is closer (already cosine for normalized vectors);
        #    L2 → smaller squared-distance is closer. _to_similarity() below maps
        #    both into a cosine-like [-1, 1] similarity so the descending sort and
        #    0–1 thresholds downstream are metric-correct.
        try:
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
                hits.append((int(idx), self._to_similarity(float(score))))

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


def is_faiss_available() -> bool:
    """Check whether the FAISS index is loaded or loadable.

    Returns True only if the index file and metadata parquet both exist
    on disk and the faiss library is importable.  Does NOT trigger a
    full load — just checks prerequisites.
    """
    idx = get_index()
    if idx.loaded:
        return True
    # Not loaded yet — check if the files exist
    return bool(faiss and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH))


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
  - List[dict] records with text/content/title/source/timestamp/similarity.
    `similarity` is a cosine-like score in [-1, 1], normalized across index
    metrics by _to_similarity(): L2 squared-distance → 1 - d/2, IP passed
    through. Results are ordered most-similar-first. (The wiki index is L2;
    before this normalization the layer was inverted — see
    memory project_wiki_faiss_l2_metric_mismatch.)
- Side effects:
  - Loads FAISS index and metadata parquet; may cache index in memory.
"""
