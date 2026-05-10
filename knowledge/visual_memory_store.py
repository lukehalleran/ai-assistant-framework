"""
# knowledge/visual_memory_store.py

Module Contract
- Purpose: Dual storage layer for visual memories. Combines a ChromaDB collection
  (text-searchable captions via SentenceTransformer embeddings) with a FAISS FlatIP
  index (CLIP embeddings for cross-modal text→image search).
- Class: VisualMemoryStore(chroma_store, data_dir="data")
- Key methods:
  - add_image(image_path, clip_embedding, caption, source, entity_ids, media_type,
              image_hash) -> str: Store image metadata + CLIP vector. Returns doc_id.
  - search_by_clip(query_embedding, k=5) -> list[dict]: CLIP vector search.
  - search_by_text(query, k=5) -> list[dict]: ChromaDB text search on captions.
  - has_hash(image_hash) -> bool: Dedup check by SHA-256 image hash.
  - get_stats() -> dict: Collection stats (count, index size).
  - save() -> None: Persist FAISS index + JSON metadata to disk.
  - load() -> None: Load FAISS index + JSON metadata from disk. Idempotent.
- Dependencies:
  - faiss (faiss-cpu)
  - memory.storage.multi_collection_chroma_store.MultiCollectionChromaStore
  - config.app_config (VISUAL_MEMORY_INDEX_PATH, VISUAL_MEMORY_META_PATH,
    VISUAL_MEMORY_SIMILARITY_THRESHOLD)
- Side effects:
  - Writes data/clip_index.faiss and data/clip_metadata.json on save().
  - Adds documents to ChromaDB visual_memories collection.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logging_utils import get_logger

logger = get_logger("knowledge.visual_memory_store")

try:
    import faiss
except ImportError:
    faiss = None

COLLECTION_NAME = "visual_memories"


class VisualMemoryStore:
    """Dual storage: ChromaDB (text search) + FAISS FlatIP (CLIP vector search)."""

    def __init__(
        self,
        chroma_store=None,
        data_dir: str = "data",
        index_path: Optional[str] = None,
        meta_path: Optional[str] = None,
    ) -> None:
        try:
            from config.app_config import (
                VISUAL_MEMORY_INDEX_PATH,
                VISUAL_MEMORY_META_PATH,
                VISUAL_MEMORY_SIMILARITY_THRESHOLD,
            )
            self._index_path = index_path or VISUAL_MEMORY_INDEX_PATH
            self._meta_path = meta_path or VISUAL_MEMORY_META_PATH
            self._sim_threshold = VISUAL_MEMORY_SIMILARITY_THRESHOLD
        except ImportError:
            self._index_path = index_path or os.path.join(data_dir, "clip_index.faiss")
            self._meta_path = meta_path or os.path.join(data_dir, "clip_metadata.json")
            self._sim_threshold = 0.20

        self._chroma = chroma_store
        self._data_dir = data_dir

        # FAISS index (512-dim, inner product on L2-normalized vectors = cosine sim)
        self._index: Optional[Any] = None
        # Metadata keyed by FAISS row index
        self._metadata: List[Dict[str, Any]] = []
        # Hash set for dedup
        self._hash_set: set = set()
        self._loaded = False

    def load(self) -> None:
        """Load FAISS index + metadata from disk. Idempotent."""
        if self._loaded:
            return

        # Load FAISS index
        if faiss is not None and os.path.exists(self._index_path):
            try:
                self._index = faiss.read_index(self._index_path)
                logger.info(f"[VisualStore] Loaded FAISS index: {self._index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"[VisualStore] Failed to load FAISS index: {e}")
                self._index = None

        if self._index is None and faiss is not None:
            self._index = faiss.IndexFlatIP(512)

        # Load metadata
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path, "r") as f:
                    self._metadata = json.load(f)
                self._hash_set = {
                    m.get("image_hash", "") for m in self._metadata if m.get("image_hash")
                }
                logger.info(f"[VisualStore] Loaded metadata: {len(self._metadata)} entries")
            except Exception as e:
                logger.warning(f"[VisualStore] Failed to load metadata: {e}")
                self._metadata = []

        self._loaded = True

    def save(self) -> None:
        """Persist FAISS index + metadata to disk."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self._index_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self._meta_path) or ".", exist_ok=True)

        # Save FAISS index
        if self._index is not None and faiss is not None:
            try:
                faiss.write_index(self._index, self._index_path)
            except Exception as e:
                logger.warning(f"[VisualStore] Failed to save FAISS index: {e}")

        # Save metadata (atomic write)
        tmp_path = self._meta_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(self._metadata, f, indent=2)
            os.replace(tmp_path, self._meta_path)
        except Exception as e:
            logger.warning(f"[VisualStore] Failed to save metadata: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def add_image(
        self,
        image_path: str,
        clip_embedding: np.ndarray,
        caption: str,
        source: str = "upload",
        entity_ids: Optional[List[str]] = None,
        media_type: str = "image/png",
        image_hash: str = "",
    ) -> Optional[str]:
        """
        Store an image's CLIP embedding + metadata.

        Returns doc_id on success, None if duplicate or error.
        """
        if not self._loaded:
            self.load()

        # Dedup by hash
        if image_hash and image_hash in self._hash_set:
            logger.debug(f"[VisualStore] Skipping duplicate image (hash={image_hash[:12]}...)")
            return None

        doc_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        entity_ids = entity_ids or []

        # Add to FAISS index
        faiss_idx = -1
        if self._index is not None and faiss is not None:
            embedding = clip_embedding.reshape(1, -1).astype(np.float32)
            faiss_idx = self._index.ntotal
            self._index.add(embedding)

        # Build metadata entry
        meta_entry = {
            "doc_id": doc_id,
            "faiss_idx": faiss_idx,
            "image_path": image_path,
            "caption": caption,
            "source": source,
            "entity_ids": entity_ids,
            "media_type": media_type,
            "image_hash": image_hash,
            "timestamp": timestamp,
        }
        self._metadata.append(meta_entry)

        if image_hash:
            self._hash_set.add(image_hash)

        # Also store in ChromaDB for text search fallback
        if self._chroma is not None:
            try:
                flat_meta = {
                    "image_path": image_path,
                    "source": source,
                    "entity_ids": ",".join(entity_ids) if entity_ids else "",
                    "media_type": media_type,
                    "image_hash": image_hash,
                    "timestamp": timestamp,
                    "is_image": True,
                    "faiss_idx": faiss_idx,
                }
                self._chroma.add_to_collection(
                    COLLECTION_NAME, caption or f"Image: {os.path.basename(image_path)}", flat_meta
                )
            except Exception as e:
                logger.warning(f"[VisualStore] ChromaDB storage failed: {e}")

        # Auto-save after each add
        self.save()

        logger.info(
            f"[VisualStore] Added image: {os.path.basename(image_path)} "
            f"(source={source}, entities={len(entity_ids)}, faiss_idx={faiss_idx})"
        )
        return doc_id

    def search_by_clip(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search by CLIP vector similarity (inner product on L2-normalized vectors).

        Returns list of dicts with: doc_id, image_path, caption, source, entity_ids,
        media_type, timestamp, score.
        """
        if not self._loaded:
            self.load()

        if self._index is None or self._index.ntotal == 0:
            return []

        embedding = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(k, self._index.ntotal)

        distances, indices = self._index.search(embedding, k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            if score < self._sim_threshold:
                continue

            meta = self._metadata[idx]
            results.append({
                "doc_id": meta.get("doc_id", ""),
                "image_path": meta.get("image_path", ""),
                "caption": meta.get("caption", ""),
                "source": meta.get("source", ""),
                "entity_ids": meta.get("entity_ids", []),
                "media_type": meta.get("media_type", ""),
                "timestamp": meta.get("timestamp", ""),
                "score": float(score),
            })

        return results

    def search_by_text(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search by text query via ChromaDB (SentenceTransformer embeddings on captions)."""
        if self._chroma is None:
            return []

        try:
            raw = self._chroma.query_collection(COLLECTION_NAME, query, n_results=k)
            results = []
            for item in raw:
                meta = item.get("metadata", {})
                entity_str = meta.get("entity_ids", "")
                results.append({
                    "doc_id": item.get("id", ""),
                    "image_path": meta.get("image_path", ""),
                    "caption": item.get("content", ""),
                    "source": meta.get("source", ""),
                    "entity_ids": entity_str.split(",") if entity_str else [],
                    "media_type": meta.get("media_type", ""),
                    "timestamp": meta.get("timestamp", ""),
                    "score": item.get("relevance_score", 0.0),
                })
            return results
        except Exception as e:
            logger.warning(f"[VisualStore] ChromaDB text search failed: {e}")
            return []

    def has_hash(self, image_hash: str) -> bool:
        """Check if an image with this hash already exists."""
        if not self._loaded:
            self.load()
        return image_hash in self._hash_set

    def get_visual_entity_ids(self) -> set:
        """Return the set of entity IDs that have at least one stored image.

        Used for entity-gated retrieval: only run CLIP search when the query
        mentions an entity that has visual memories.
        """
        if not self._loaded:
            self.load()
        ids: set = set()
        for m in self._metadata:
            for eid in m.get("entity_ids", []):
                if eid:
                    ids.add(eid.lower())
        return ids

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        if not self._loaded:
            self.load()
        return {
            "total_images": len(self._metadata),
            "faiss_vectors": self._index.ntotal if self._index else 0,
            "unique_sources": len({m.get("source", "") for m in self._metadata}),
            "has_faiss": self._index is not None,
            "has_chroma": self._chroma is not None,
        }
