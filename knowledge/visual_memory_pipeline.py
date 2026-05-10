"""
# knowledge/visual_memory_pipeline.py

Module Contract
- Purpose: Orchestrates the full ingestion of images into visual memory:
  CLIP embedding → vision LLM captioning → entity tagging → storage.
- Class: VisualMemoryPipeline(clip_manager, visual_store, model_manager=None,
         entity_resolver=None)
- Key methods:
  - ingest_image(image_path, source, context_text, media_type) -> Optional[str]:
    Async. Ingest a single image. Returns doc_id or None.
  - ingest_batch(image_paths, source, context_texts) -> list[str]:
    Async. Batch ingest multiple images.
- Dependencies:
  - knowledge.clip_manager.CLIPManager (CLIP encoding)
  - knowledge.visual_memory_store.VisualMemoryStore (storage)
  - models.model_manager.ModelManager (vision LLM captioning, optional)
  - memory.entity_resolver.EntityResolver (entity tagging, optional)
  - memory.graph_utils.extract_graph_entities (entity extraction from text)
  - config.app_config (VISUAL_MEMORY_CAPTION_MODEL, VISUAL_MEMORY_CAPTION_TIMEOUT)
- Side effects:
  - Writes to FAISS index + ChromaDB via VisualMemoryStore
  - Makes LLM API call for captioning (optional, non-fatal)
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from typing import List, Optional

from utils.logging_utils import get_logger

logger = get_logger("knowledge.visual_memory_pipeline")

# Vision LLM captioning prompt
CAPTION_PROMPT = (
    "Describe this image in 2-3 sentences for a personal photo archive. "
    "Include any visible text, people, pets, objects, locations, and activities. "
    "Be specific about details that would help someone find this image later."
)


class VisualMemoryPipeline:
    """Orchestrates image ingestion: CLIP embed → caption → entity tag → store."""

    def __init__(
        self,
        clip_manager,
        visual_store,
        model_manager=None,
        entity_resolver=None,
    ) -> None:
        self._clip = clip_manager
        self._store = visual_store
        self._model_manager = model_manager
        self._entity_resolver = entity_resolver

    async def ingest_image(
        self,
        image_path: str,
        source: str = "upload",
        context_text: str = "",
        media_type: str = "",
    ) -> Optional[str]:
        """
        Ingest a single image into visual memory.

        Steps:
        1. Compute SHA-256 hash for dedup
        2. CLIP-encode the image
        3. Generate caption via vision LLM (optional)
        4. Extract entity tags from caption + context
        5. Store in VisualMemoryStore

        Returns doc_id on success, None on failure or duplicate.
        """
        if not os.path.exists(image_path):
            logger.warning(f"[VisualPipeline] Image not found: {image_path}")
            return None

        # 1. Compute hash for dedup
        image_hash = self._compute_hash(image_path)
        if self._store.has_hash(image_hash):
            logger.debug(f"[VisualPipeline] Duplicate image, skipping: {image_path}")
            return None

        # 2. CLIP encode
        clip_embedding = self._clip.encode_image_from_path(image_path)
        if clip_embedding is None:
            logger.warning(f"[VisualPipeline] CLIP encoding failed: {image_path}")
            return None

        # 3. Caption via vision LLM (optional, non-fatal)
        caption = await self._generate_caption(image_path, media_type)
        if not caption:
            # Fallback to filename-based caption
            caption = f"Image: {os.path.basename(image_path)}"

        # 4. Entity extraction from caption + context
        entity_ids = self._extract_entities(caption, context_text)

        # 5. Detect media type if not provided
        if not media_type:
            media_type = self._detect_media_type(image_path)

        # 6. Store
        doc_id = self._store.add_image(
            image_path=image_path,
            clip_embedding=clip_embedding,
            caption=caption,
            source=source,
            entity_ids=entity_ids,
            media_type=media_type,
            image_hash=image_hash,
        )

        if doc_id:
            logger.info(
                f"[VisualPipeline] Ingested {os.path.basename(image_path)} "
                f"(source={source}, entities={entity_ids})"
            )
        return doc_id

    async def ingest_batch(
        self,
        image_paths: List[str],
        source: str = "obsidian",
        context_texts: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Batch ingest multiple images.

        Returns list of doc_ids (empty strings for failures/dupes).
        """
        if context_texts is None:
            context_texts = [""] * len(image_paths)

        results = []
        for path, ctx in zip(image_paths, context_texts):
            try:
                doc_id = await self.ingest_image(path, source=source, context_text=ctx)
                results.append(doc_id or "")
            except Exception as e:
                logger.warning(f"[VisualPipeline] Batch ingest failed for {path}: {e}")
                results.append("")

        ingested = sum(1 for r in results if r)
        logger.info(f"[VisualPipeline] Batch complete: {ingested}/{len(image_paths)} ingested")
        return results

    async def _generate_caption(self, image_path: str, media_type: str = "") -> str:
        """Generate a caption via vision LLM. Returns empty string on failure."""
        if self._model_manager is None:
            return ""

        try:
            from config.app_config import VISUAL_MEMORY_CAPTION_TIMEOUT
            timeout = VISUAL_MEMORY_CAPTION_TIMEOUT
        except ImportError:
            timeout = 10.0

        try:
            import base64

            with open(image_path, "rb") as f:
                img_bytes = f.read()
            b64 = base64.b64encode(img_bytes).decode("utf-8")

            if not media_type:
                media_type = self._detect_media_type(image_path)

            images = [{"data": b64, "media_type": media_type, "filename": os.path.basename(image_path)}]

            # Use generate_async (supports images) and collect full response
            full_response = ""
            gen = await asyncio.wait_for(
                self._model_manager.generate_async(
                    CAPTION_PROMPT,
                    system_prompt="You are an image description assistant.",
                    max_tokens=200,
                    images=images,
                ),
                timeout=timeout,
            )
            async for chunk in gen:
                # chunk may be a string or a ChatCompletionChunk object
                if isinstance(chunk, str):
                    full_response += chunk
                elif hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        full_response += delta.content

            return full_response.strip()
        except asyncio.TimeoutError:
            logger.warning(f"[VisualPipeline] Caption timed out for {image_path}")
            return ""
        except Exception as e:
            logger.warning(f"[VisualPipeline] Caption generation failed: {e}")
            return ""

    def _extract_entities(self, caption: str, context_text: str = "") -> List[str]:
        """Extract entity IDs from caption and context text."""
        if self._entity_resolver is None:
            return []

        try:
            from memory.graph_utils import extract_graph_entities

            combined = f"{caption} {context_text}".strip()
            entities = extract_graph_entities(combined, self._entity_resolver)
            return list(entities)
        except Exception as e:
            logger.debug(f"[VisualPipeline] Entity extraction failed: {e}")
            return []

    @staticmethod
    def _compute_hash(image_path: str) -> str:
        """Compute SHA-256 hash of image file."""
        sha = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def _detect_media_type(image_path: str) -> str:
        """Detect media type from file extension."""
        ext = os.path.splitext(image_path)[1].lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
        }.get(ext, "image/png")
