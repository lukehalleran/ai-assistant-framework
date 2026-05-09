"""
# core/prompt/hygiene.py

Module Contract
- Purpose: Content hygiene operations for prompt building — deduplication, caps enforcement,
  cross-section dedup, semantic chunk stitching, and backfill for recent conversations.
- Class: ContentHygiene(memory_coordinator, context_gatherer)
- Key methods:
  - _hygiene_and_caps(context, stm_summary) -> Dict
    Deduplication and caps enforcement across all context sections.
    Cross-section dedup with backfill for recent_conversations.
    Stitches semantic chunks by title. Adds STM summary.
  - _backfill_recent_conversations(existing_items, seen_embeddings, seen_content,
      target_count, offset, embedder, similarity_threshold) -> List
    Fetches additional conversations from corpus until target count reached,
    deduplicating against existing items via string and embedding similarity.
- Dependencies:
  - .formatter._dedupe_keep_order, _sanitize_embedded_headers (text utilities)
  - PROMPT_MAX_PERSONAL_NOTES from builder module (caps)
  - memory_coordinator.corpus_manager (backfill source)
  - numpy (embedding-based dedup in backfill)
- Side effects:
  - Memory system queries for backfill
  - Modifies context dict in place (dedup, caps, section updates)
  - Logging of dedup actions and backfill progress
"""

import numpy as np
from typing import Dict, List, Optional, Any
from utils.logging_utils import get_logger
from .formatter import _dedupe_keep_order, _sanitize_embedded_headers

logger = get_logger("prompt_hygiene")


class ContentHygiene:
    """Content hygiene operations: dedup, caps, cross-section dedup, and backfill."""

    def __init__(self, memory_coordinator, context_gatherer):
        self.memory_coordinator = memory_coordinator
        self.context_gatherer = context_gatherer

    async def _hygiene_and_caps(self, context: Dict[str, Any], stm_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply deduplication and caps to all context sections.

        This ensures we don't have duplicate content and stay within
        reasonable limits for each content type.
        """
        # Import caps here to avoid circular import at module level
        from .builder import PROMPT_MAX_PERSONAL_NOTES

        # Debug: Log that we're starting dedup
        section_counts = {k: len(v) if isinstance(v, list) else 1 for k, v in context.items() if v}
        logger.info(f"[DEDUP START] Sections with content: {section_counts}")

        # Apply deduplication and caps to all sections
        sections_to_process = [
            "recent_conversations", "memories",
            "summaries", "recent_summaries", "semantic_summaries",
            "reflections", "recent_reflections", "semantic_reflections",
            "dreams", "semantic_chunks", "wiki"
        ]

        for section in sections_to_process:
            items = context.get(section, [])
            if not items:
                continue

            # Deduplicate
            if isinstance(items, list):
                # For memories and conversations, dedupe by content
                if section in ["recent_conversations", "memories"]:
                    original_count = len(items)
                    # Handle both content field (hybrid retriever) and query/response fields (corpus)
                    def dedup_key(x):
                        # Try content field first (from hybrid retriever)
                        content = x.get("content", "")
                        if content:
                            return content.strip().lower()
                        # Fallback to query/response
                        return str(x.get("response", "") + x.get("query", "")).strip().lower()

                    deduped = _dedupe_keep_order(items, key_fn=dedup_key)
                    logger.debug(f"ASSEMBLY DEDUP {section}: {original_count} -> {len(deduped)} items")
                else:
                    # For others, dedupe by string representation
                    deduped = _dedupe_keep_order(items)

                context[section] = deduped

        # Cross-section deduplication to catch content appearing in multiple sections
        # This is critical for avoiding duplicate ICE responses in conversations/memories
        # NOTE: We only dedup conversations/memories across each other, NOT summaries/reflections
        # because those need to stay in their dedicated sections with proper headers

        # String-based cross-section dedup (normalized first 500 chars).
        # Previously used embedding-based O(n^2) cosine similarity which added 300-500ms.
        # String dedup catches the vast majority of exact/near-exact duplicates at ~0 cost.
        seen_content = set()

        cross_dedup_sections = [
            "recent_conversations", "memories", "personal_notes"
        ]

        # Track target counts for backfilling
        target_counts = {
            "recent_conversations": 10,  # Target number of unique recent conversations
            "memories": 30,  # Target number of unique memories
            "personal_notes": PROMPT_MAX_PERSONAL_NOTES  # Target number of personal notes
        }

        for section in cross_dedup_sections:
            items = context.get(section, [])
            if not items or not isinstance(items, list):
                continue

            target_count = target_counts.get(section, len(items))
            original_count = len(items)

            deduplicated = []
            for item in items:
                # Extract content for dedup check
                if isinstance(item, dict):
                    content = item.get("content", "")
                    if not content:
                        response = item.get("response", "")
                        content = response if response else str(item.get("query", ""))
                else:
                    content = str(item)

                # Normalize content for comparison
                normalized = content.strip().lower()
                for prefix in ["user:", "daemon:", "luke,"]:
                    if normalized.startswith(prefix):
                        normalized = normalized[len(prefix):].strip()

                dedup_key = normalized[:500]
                if dedup_key and dedup_key not in seen_content:
                    seen_content.add(dedup_key)
                    deduplicated.append(item)
                else:
                    logger.debug(f"CROSS-SECTION DEDUP: Skipped duplicate in {section} (key: {dedup_key[:80]}...)")

            original_count = len(items)
            if len(deduplicated) < original_count:
                logger.info(f"CROSS-SECTION DEDUP {section}: {original_count} -> {len(deduplicated)} items (removed {original_count - len(deduplicated)} duplicates)")

            context[section] = deduplicated

            # Backfill if we're below target after deduplication
            if len(deduplicated) < target_count and section == "recent_conversations":
                logger.info(f"[BACKFILL] {section} has {len(deduplicated)}/{target_count} items, fetching more...")

                backfill_result = await self._backfill_recent_conversations(
                    existing_items=deduplicated,
                    seen_embeddings=[],
                    seen_content=seen_content,
                    target_count=target_count,
                    offset=original_count,
                    embedder=None,
                    similarity_threshold=0.90
                )

                context[section] = backfill_result

        # Stitch semantic chunks by title
        semantic_chunks = context.get("semantic_chunks", [])
        if semantic_chunks:
            # Group by title and stitch content
            chunks_by_title = {}
            for chunk in semantic_chunks:
                title = chunk.get("title", "")
                if title:
                    if title not in chunks_by_title:
                        chunks_by_title[title] = chunk.copy()
                    else:
                        # Combine content
                        existing = chunks_by_title[title]
                        existing_content = existing.get("content", "")
                        new_content = chunk.get("content", "")
                        combined = f"{existing_content}\n\n{new_content}"

                        # Apply length limit
                        if len(combined) <= 4000:  # SEM_STITCH_MAX_CHARS
                            existing["content"] = combined

            context["semantic_chunks"] = list(chunks_by_title.values())

        # Add STM summary if provided
        if stm_summary is not None:
            context["stm_summary"] = stm_summary
            logger.debug(f"Added STM summary to context: topic={stm_summary.get('topic')}")

        return context

    async def _backfill_recent_conversations(
        self,
        existing_items: List[Dict[str, Any]],
        seen_embeddings: List[tuple],
        seen_content: set,
        target_count: int,
        offset: int,
        embedder,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Backfill recent conversations to reach target count after deduplication.

        Fetches additional conversations from corpus and deduplicates them against
        existing items until we reach the target count or run out of conversations.

        Args:
            existing_items: Already deduplicated items
            seen_embeddings: List of (embedding, item) tuples for semantic dedup
            seen_content: Set of content keys for string-based dedup
            target_count: Target number of unique items
            offset: Starting offset in corpus
            embedder: Sentence embedder for semantic similarity
            similarity_threshold: Threshold for considering items duplicates

        Returns:
            List of deduplicated items (may be less than target_count if corpus exhausted)
        """
        deduplicated = existing_items.copy()
        batch_size = target_count - len(deduplicated)
        max_iterations = 10  # Safety limit
        iteration = 0

        logger.info(f"[BACKFILL] Starting with {len(deduplicated)} items, target={target_count}")

        while len(deduplicated) < target_count and iteration < max_iterations:
            iteration += 1

            # Fetch next batch from corpus
            try:
                if not self.memory_coordinator:
                    logger.warning("[BACKFILL] No memory_coordinator available")
                    break

                corpus_manager = getattr(self.memory_coordinator, 'corpus_manager', None)
                if not corpus_manager:
                    logger.warning("[BACKFILL] No corpus_manager in memory_coordinator")
                    break

                # Get more recent conversations from corpus
                all_recent = corpus_manager.get_recent_memories(
                    count=offset + batch_size
                )

                # Slice to get only the new batch
                if len(all_recent) <= offset:
                    logger.info(f"[BACKFILL] No more items in corpus (have {len(all_recent)}, offset={offset})")
                    break

                additional_items = all_recent[offset:offset + batch_size]

                if not additional_items:
                    logger.info(f"[BACKFILL] No more items available")
                    break

                logger.debug(f"[BACKFILL] Iteration {iteration}: fetched {len(additional_items)} items (offset={offset})")

                # Deduplicate new items against existing ones
                added_count = 0
                for item in additional_items:
                    # Extract content
                    if isinstance(item, dict):
                        content = item.get("content", "")
                        if not content:
                            response = item.get("response", "")
                            content = response if response else str(item.get("query", ""))
                    else:
                        content = str(item)

                    # Normalize
                    normalized = content.strip().lower()
                    for prefix in ["user:", "daemon:", "luke,"]:
                        if normalized.startswith(prefix):
                            normalized = normalized[len(prefix):].strip()

                    is_duplicate = False

                    # Check against existing deduplicated items
                    if embedder:
                        try:
                            item_embedding = embedder.encode(normalized[:512], convert_to_numpy=True)

                            for seen_emb, _ in seen_embeddings:
                                similarity = np.dot(item_embedding, seen_emb) / (
                                    np.linalg.norm(item_embedding) * np.linalg.norm(seen_emb) + 1e-8
                                )

                                if similarity >= similarity_threshold:
                                    is_duplicate = True
                                    logger.debug(f"[BACKFILL] Skipped duplicate (similarity={similarity:.3f})")
                                    break

                            if not is_duplicate:
                                seen_embeddings.append((item_embedding, item))
                                deduplicated.append(item)
                                added_count += 1

                                if len(deduplicated) >= target_count:
                                    break

                        except Exception as e:
                            logger.debug(f"[BACKFILL] Embedding failed: {e}")
                            # Fallback to string-based
                            dedup_key = normalized[:500]
                            if dedup_key and dedup_key not in seen_content:
                                seen_content.add(dedup_key)
                                deduplicated.append(item)
                                added_count += 1
                    else:
                        # String-based fallback
                        dedup_key = normalized[:500]
                        if dedup_key and dedup_key not in seen_content:
                            seen_content.add(dedup_key)
                            deduplicated.append(item)
                            added_count += 1

                    if len(deduplicated) >= target_count:
                        break

                logger.info(f"[BACKFILL] Iteration {iteration}: added {added_count} unique items, now have {len(deduplicated)}/{target_count}")

                # Update offset for next batch
                offset += len(additional_items)

                # If we didn't add any unique items, increase batch size
                if added_count == 0:
                    batch_size = min(batch_size * 2, 50)  # Double batch size up to 50
                else:
                    batch_size = target_count - len(deduplicated)

                if len(deduplicated) >= target_count:
                    break

            except Exception as e:
                logger.warning(f"[BACKFILL] Failed to fetch additional items: {e}")
                break

        logger.info(f"[BACKFILL] Complete: {len(deduplicated)}/{target_count} items after {iteration} iterations")
        return deduplicated
