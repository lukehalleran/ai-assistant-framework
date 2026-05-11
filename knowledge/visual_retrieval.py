"""
# knowledge/visual_retrieval.py

Module Contract
- Purpose: High-level retrieval interface for visual memories. Combines CLIP vector
  search with ChromaDB text search, loads base64 image data from disk, and returns
  results ready for prompt injection (text section + multimodal images).
- Class: VisualRetriever(clip_manager, visual_store)
- Key methods:
  - retrieve_visual_memories(query, k=5, max_images=3, target_entities=None) -> dict:
    Async. Returns {"text_results": [...], "images": [...]}.
    text_results: captions + metadata for [VISUAL MEMORIES] prompt section.
    images: base64 image dicts matching existing note_images format for multimodal API.
    When target_entities is provided (set of lowercase entity IDs), results are
    hard-filtered to only include images tagged with at least one target entity.
    Falls back to unfiltered results if filtering leaves nothing.
- Dependencies:
  - knowledge.clip_manager.CLIPManager (text encoding)
  - knowledge.visual_memory_store.VisualMemoryStore (search)
  - config.app_config (VISUAL_MEMORY_MAX_IMAGES, VISUAL_MEMORY_SIMILARITY_THRESHOLD)
- Side effects:
  - Reads image files from disk to produce base64 data.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, List

from utils.logging_utils import get_logger

logger = get_logger("knowledge.visual_retrieval")

# Max image size for visual memory recall — kept small to control API costs.
# 3 images at 200KB each ≈ 600KB base64 ≈ ~800 tokens per image on multimodal APIs.
MAX_IMAGE_BYTES = 200_000


class VisualRetriever:
    """Retrieves visual memories via CLIP + ChromaDB hybrid search."""

    def __init__(self, clip_manager, visual_store) -> None:
        self._clip = clip_manager
        self._store = visual_store

    async def retrieve_visual_memories(
        self,
        query: str,
        k: int = 5,
        max_images: int = 3,
        target_entities: set[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Retrieve visual memories matching the query.

        Uses CLIP text→image search as primary signal, with ChromaDB text search
        as fallback/complement. Results are deduplicated and ranked by score.

        When target_entities is provided, results are hard-filtered to only include
        images tagged with at least one of the target entities. This prevents
        entity confusion when multiple entity names appear in the query
        (e.g. "I asked about Paczki but got Flapjack").

        Returns:
            {
                "text_results": [{"caption", "source", "score", "image_path", "entity_ids", "doc_id"}],
                "images": [{"note_index", "note_title", "filename", "media_type", "data"}],
            }
        """
        try:
            max_images = min(max_images, self._get_max_images())
        except Exception:
            pass

        # CLIP vector search (primary)
        clip_results = self._search_clip(query, k=k)

        # ChromaDB text search (fallback/complement)
        text_results = self._store.search_by_text(query, k=k)

        # Merge + deduplicate by image_path, boost entity matches
        merged = self._merge_results(clip_results, text_results, query=query)

        # Entity hard-filter: keep only images matching at least one target entity.
        # When CLIP can't rank the target above other images (e.g. 8 Flapjack
        # images outrank 2 Paczki images because CLIP just sees "cat"), fall
        # back to direct entity lookup from metadata.
        if target_entities and len(merged) > 0:
            filtered = [
                r for r in merged
                if {e.lower() for e in r.get("entity_ids", [])} & target_entities
            ]
            # CLIP missed the target — fetch directly by entity tag
            if not filtered:
                filtered = self._store.get_by_entity(target_entities)
                if filtered:
                    logger.info(
                        f"[VisualRetrieval] Entity direct lookup: {len(filtered)} results "
                        f"for {target_entities} (CLIP missed them)"
                    )
            if filtered:
                merged = filtered
                logger.info(
                    f"[VisualRetrieval] Entity filter: {len(filtered)} results "
                    f"match target entities {target_entities}"
                )
            else:
                # Target entity has no images at all
                merged = []
                logger.warning(
                    f"[VisualRetrieval] Entity filter: 0 results for {target_entities}"
                )

        # Build text_results for prompt section (all results)
        text_section = []
        for r in merged:
            text_section.append({
                "doc_id": r.get("doc_id", ""),
                "caption": r.get("caption", ""),
                "source": r.get("source", ""),
                "score": r.get("score", 0.0),
                "image_path": r.get("image_path", ""),
                "entity_ids": r.get("entity_ids", []),
            })

        # Load actual images for multimodal (capped at max_images)
        images = []
        for r in merged[:max_images]:
            img_dict = self._load_image_for_multimodal(r)
            if img_dict:
                images.append(img_dict)

        if text_section:
            logger.info(
                f"[VisualRetrieval] Query '{query[:50]}' → "
                f"{len(text_section)} results, {len(images)} images loaded"
            )

        return {"text_results": text_section, "images": images}

    def _search_clip(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """CLIP text→image search."""
        query_embedding = self._clip.encode_text(query)
        if query_embedding is None:
            return []
        return self._store.search_by_clip(query_embedding, k=k)

    def _merge_results(
        self,
        clip_results: List[Dict],
        text_results: List[Dict],
        query: str = "",
    ) -> List[Dict[str, Any]]:
        """Merge CLIP + text results, dedup by image_path, boost entity matches."""
        seen = {}

        for r in clip_results:
            path = r.get("image_path", "")
            if path and (path not in seen or r.get("score", 0) > seen[path].get("score", 0)):
                seen[path] = r

        for r in text_results:
            path = r.get("image_path", "")
            if path and path not in seen:
                seen[path] = r

        # Entity-aware scoring: boost results that match entities in the query
        if query:
            query_tokens = {t.lower() for t in query.split() if len(t) > 2}
            for path, r in seen.items():
                entity_ids = {e.lower() for e in r.get("entity_ids", [])}
                matched = query_tokens & entity_ids
                if matched:
                    # Strong boost for entity match — names matter more than visual similarity
                    r["score"] = r.get("score", 0) + 0.3 * len(matched)
                    # Also check caption for query entity mentions
                caption_lower = r.get("caption", "").lower()
                for token in query_tokens:
                    if token in caption_lower and token not in entity_ids:
                        r["score"] = r.get("score", 0) + 0.1

        # Sort by score descending
        merged = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)
        return merged

    def _load_image_for_multimodal(self, result: Dict) -> Dict[str, Any] | None:
        """Load image from disk as base64 for multimodal API injection.

        Always compresses to keep API costs low — target ~200KB per image.
        """
        image_path = result.get("image_path", "")
        if not image_path or not os.path.exists(image_path):
            return None

        try:
            # Always compress to control API token costs
            img_bytes = self._compress_image(image_path)
            if img_bytes is None:
                return None

            b64 = base64.b64encode(img_bytes).decode("utf-8")
            # Detect actual media type from bytes (compression outputs JPEG)
            if img_bytes[:3] == b'\xff\xd8\xff':
                media_type = "image/jpeg"
            elif img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                media_type = "image/png"
            else:
                media_type = result.get("media_type", "image/jpeg")
            filename = os.path.basename(image_path)
            caption = result.get("caption", "")

            return {
                "note_index": 0,
                "note_title": f"Visual Memory: {caption[:80]}" if caption else f"Visual Memory: {filename}",
                "note_section": "",
                "filename": filename,
                "media_type": media_type,
                "data": b64,
            }
        except Exception as e:
            logger.warning(f"[VisualRetrieval] Failed to load image {image_path}: {e}")
            return None

    @staticmethod
    def _compress_image(image_path: str, _img_bytes: bytes = None) -> bytes | None:
        """Compress image for multimodal API — targets ~200KB max.

        Resizes to max 512px on longest side and uses JPEG quality stepping.
        This keeps API costs reasonable (~800 tokens per image).
        """
        try:
            from PIL import Image
            import io

            img = Image.open(image_path).convert("RGB")

            # Resize to max 512px on longest side (enough detail for recognition)
            max_dim = 512
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            # Try decreasing quality until under budget
            for quality in [75, 50, 35, 20]:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                if buf.tell() <= MAX_IMAGE_BYTES:
                    return buf.getvalue()

            # Last resort — shrink further
            for max_d in [384, 256]:
                ratio = max_d / max(img.size)
                smaller = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
                buf = io.BytesIO()
                smaller.save(buf, format="JPEG", quality=40)
                if buf.tell() <= MAX_IMAGE_BYTES:
                    return buf.getvalue()

            return None
        except Exception:
            return None

    @staticmethod
    def _get_max_images() -> int:
        from config.app_config import VISUAL_MEMORY_MAX_IMAGES
        return VISUAL_MEMORY_MAX_IMAGES
