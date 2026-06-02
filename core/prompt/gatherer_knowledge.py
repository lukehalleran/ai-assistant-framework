"""
# core/prompt/gatherer_knowledge.py

Mixin providing knowledge retrieval methods for ContextGatherer.

Methods:
  - get_personal_notes(query, limit, include_images, max_images_per_note) -> List[Dict]
  - get_reference_docs(query, limit) -> List[Dict]
  - get_user_uploads(query, limit) -> List[Dict]
  - get_git_commits(query, limit) -> List[Dict]
  - get_procedural_skills(query, limit) -> List[Dict]
  - get_proposed_features(query, limit) -> List[Dict]
  - get_graph_context(query, max_sentences) -> List[str]
  - get_unresolved_threads(max_results) -> List[Dict]
  - get_proactive_insights(query, max_insights) -> List[str]
  - get_visual_memories(query, max_images) -> Dict[str, Any]
  - get_codebase_changes(since_datetime) -> Dict
  - get_narrative_context() -> str
  - _get_wiki_content(query, limit) -> List[Dict]
  - _should_skip_wikipedia(query) -> bool
  - _get_wiki_snippet_cached(query) -> Optional[Dict]
  - _get_semantic_chunks(query, k, max_results) -> List[Dict]
  - _get_dreams(limit) -> List[Dict]

Visual memory retrieval (get_visual_memories) uses a multi-step entity-gated pipeline:
  A. Filter junk caption artifacts from stored entity IDs (_VISUAL_JUNK_IDS)
  B. Resolve query entities via extract_graph_entities() (alias-aware)
  C. Substring fallback for visual-only entities not in the knowledge graph
  D. Multi-entity disambiguation via visual-intent proximity (_VISUAL_INTENT_WORDS)
     or sentence-tail heuristic when no intent keywords present
  E. Pass target_entities to VisualRetriever for hard-filtering results

Depends on self.memory_coordinator, self.obsidian_manager, self.reference_docs_manager,
self.time_manager, self.memory_id_map, self.gate_system, self.token_manager,
self.model_manager (set by ContextGatherer.__init__).
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from core.wiki_util import get_wiki_snippet, clean_query
from knowledge.semantic_search import semantic_search_with_neighbors
from .formatter import _parse_bool

logger = logging.getLogger("prompt_context_gatherer")

# Configuration loading
try:
    from config.app_config import config as _APP_CFG
    _MEM_CFG = (_APP_CFG.get("memory") or {})
except (ImportError, AttributeError):
    _MEM_CFG = {}


# ---------------------------------------------------------------------------
# Visual memory entity resolution constants
# ---------------------------------------------------------------------------

# Caption artifacts that are not real entities — filtered at retrieval time.
_VISUAL_JUNK_IDS = frozenset({
    # Colors
    "black", "white", "gray", "grey", "brown", "orange", "blue", "red",
    "green", "yellow",
    # Descriptors
    "warm", "cold", "soft", "dark", "light",
    # Quantifiers
    "one", "two", "three", "several", "many", "few",
    # Hedges
    "possibly", "likely", "probably", "apparently",
    # Generic scene words
    "sections", "support", "area", "side", "part", "background",
    "image", "photo", "picture", "scene", "moment",
})

# Keywords indicating the user wants to *see* something.
_VISUAL_INTENT_WORDS = frozenset({
    "show", "see", "photo", "photos", "picture", "pictures",
    "image", "images", "look", "pic", "pics",
})


def _cfg_int(key: str, default_val: int) -> int:
    try:
        v = _MEM_CFG.get(key, default_val)
        return int(v) if v is not None else int(default_val)
    except (ValueError, TypeError):
        return int(default_val)


# Configuration constants
PROMPT_MAX_WIKI = _cfg_int("prompt_max_wiki", 3)
PROMPT_MAX_DREAMS = _cfg_int("prompt_max_dreams", 3)
PROMPT_MAX_SEMANTIC = _cfg_int("prompt_max_semantic", 10)

# Feature flags
DREAMS_ENABLED = _parse_bool(os.getenv("DREAMS_ENABLED", "1"))

# Semantic search configuration
SEM_K = int(os.getenv("SEM_K", "50"))
SEM_TIMEOUT_S = int(os.getenv("SEM_TIMEOUT_S", "8"))
SEM_STITCH_MAX_CHARS = int(os.getenv("SEM_STITCH_MAX_CHARS", "4000"))
try:
    from config.app_config import SEMANTIC_CHUNKS_GATE_THRESHOLD
except ImportError:
    SEMANTIC_CHUNKS_GATE_THRESHOLD = 0.35

# Wiki snippet caching (module-level)
_wiki_cache = {}  # Simple in-memory cache for wiki snippets
_WIKI_CACHE_MAX_SIZE = 100  # Maximum cache entries to prevent memory leaks


def _cleanup_wiki_cache():
    """Remove expired entries and enforce max size."""
    global _wiki_cache
    now = datetime.now()
    # Remove expired entries (older than 1 hour)
    _wiki_cache = {
        k: v for k, v in _wiki_cache.items()
        if now - v["timestamp"] < timedelta(hours=1)
    }
    # Enforce max size by removing oldest entries
    if len(_wiki_cache) > _WIKI_CACHE_MAX_SIZE:
        sorted_entries = sorted(_wiki_cache.items(), key=lambda x: x[1]["timestamp"])
        _wiki_cache = dict(sorted_entries[-_WIKI_CACHE_MAX_SIZE:])


class KnowledgeRetrievalMixin:
    """Mixin providing knowledge and document retrieval methods."""

    async def _get_wiki_snippet_cached(self, query: str) -> Optional[Dict[str, Any]]:
        """Get wiki snippet with caching."""
        # Periodic cleanup to prevent memory leaks
        if len(_wiki_cache) > _WIKI_CACHE_MAX_SIZE // 2:
            _cleanup_wiki_cache()

        cache_key = self._wiki_cache_key(query)

        # Check cache first
        if cache_key in _wiki_cache:
            cached_entry = _wiki_cache[cache_key]
            # Simple TTL: 1 hour
            if datetime.now() - cached_entry["timestamp"] < timedelta(hours=1):
                return cached_entry["data"]

        # Fetch from wiki
        try:
            snippet = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, get_wiki_snippet, query),
                timeout=5.0
            )
            if snippet:
                # Cache the result
                _wiki_cache[cache_key] = {
                    "data": snippet,
                    "timestamp": datetime.now()
                }
                return snippet
        except asyncio.TimeoutError:
            logger.warning(f"Wiki snippet timeout for: {query}")
        except Exception as e:
            logger.warning(f"Wiki snippet error for {query}: {e}")

        return None

    async def get_personal_notes(
        self,
        query: str,
        limit: int = 10,
        include_images: bool = False,
        max_images_per_note: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get relevant personal notes from Obsidian vault.

        Args:
            query: Search query for semantic retrieval
            limit: Maximum notes to return
            include_images: If True, load actual image data for multimodal models
            max_images_per_note: Maximum images to load per note chunk

        Returns:
            List of note dicts with content, metadata, relevance_score,
            and optionally 'image_data' containing base64-encoded images
        """
        manager = self.obsidian_manager
        if not manager:
            return []

        try:
            # Clean query: remove self-referential phrases that pollute search
            import re
            clean_query = re.sub(
                r'\b(from|in|check|look at|search|find in)?\s*(my|the)?\s*(notes?|vault|obsidian)\b',
                '', query, flags=re.IGNORECASE
            )
            # Clean up extra whitespace
            clean_query = ' '.join(clean_query.split()).strip()
            # Fall back to original if cleaning removed everything
            search_query = clean_query if clean_query else query
            logger.debug(f"[ContextGatherer] Personal notes query: '{query}' -> '{search_query}'")

            # Retrieve notes via semantic search (with optional image loading)
            logger.warning(f"[ContextGatherer] IMAGE DEBUG: Calling get_notes with include_images={include_images}")
            notes = await manager.get_notes(
                search_query,
                limit=limit,
                include_images=include_images,
                max_images_per_note=max_images_per_note
            )
            # Debug: check what we got back
            if notes:
                total_images = sum(len(n.get('image_data', [])) for n in notes)
                logger.warning(f"[ContextGatherer] IMAGE DEBUG: Got {len(notes)} notes with {total_images} total images")

            # Track note IDs for citations
            if notes:
                for idx, note in enumerate(notes[:limit], start=1):
                    note_id = f"NOTE_{idx}"
                    meta = note.get('metadata', {})
                    image_data = note.get('image_data', [])
                    self.memory_id_map[note_id] = {
                        'type': 'personal_note',
                        'timestamp': meta.get('timestamp', ''),
                        'content': note.get('content', '')[:500],
                        'relevance_score': note.get('relevance_score', 0.0),
                        'title': meta.get('title', ''),
                        'file_path': meta.get('file_path', ''),
                        'has_images': len(image_data) > 0,
                        'image_count': len(image_data),
                        'db_id': note.get('id', None),
                    }

                image_count = sum(len(n.get('image_data', [])) for n in notes)
                logger.debug(f"[ContextGatherer] Retrieved {len(notes)} personal notes with {image_count} images")

            return notes or []

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get personal notes: {e}")
            return []

    async def get_reference_docs(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get relevant reference documents from uploaded docs collection.
        Excludes user uploads (type='user_upload') which have their own section.

        Args:
            query: Search query for semantic retrieval
            limit: Maximum document chunks to return

        Returns:
            List of doc dicts with content, metadata, and relevance_score
        """
        manager = self.reference_docs_manager
        if not manager:
            return []

        try:
            # Retrieve documents via hybrid search (fetch extra to allow filtering)
            docs = await manager.get_documents(query, limit=limit * 2)

            # Filter OUT user uploads -- they appear in [USER UPLOADED ITEMS] instead
            docs = [d for d in docs if d.get('metadata', {}).get('type') != 'user_upload']
            docs = docs[:limit]

            # Track doc IDs for citations
            if docs:
                for idx, doc in enumerate(docs[:limit], start=1):
                    doc_id = f"REFDOC_{idx}"
                    meta = doc.get('metadata', {})
                    self.memory_id_map[doc_id] = {
                        'type': 'reference_doc',
                        'timestamp': meta.get('timestamp', ''),
                        'content': doc.get('content', '')[:500],
                        'relevance_score': doc.get('relevance_score', 0.0),
                        'title': meta.get('title', ''),
                        'section': meta.get('section', ''),
                        'file_path': meta.get('file_path', ''),
                        'db_id': doc.get('id', None),
                    }

                logger.debug(f"[ContextGatherer] Retrieved {len(docs)} reference docs for query")

            return docs or []

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get reference docs: {e}")
            return []

    async def get_user_uploads(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant user-uploaded items from reference_docs collection,
        filtered to only include entries with type='user_upload'.

        Args:
            query: Search query for semantic retrieval
            limit: Maximum uploads to return

        Returns:
            List of upload dicts with content, metadata, and relevance_score
        """
        manager = self.reference_docs_manager
        if not manager:
            return []

        try:
            # Fetch extra to allow filtering to user_upload type
            docs = await manager.get_documents(query, limit=limit * 2)

            # Filter to only user uploads
            uploads = [d for d in docs if d.get('metadata', {}).get('type') == 'user_upload']
            uploads = uploads[:limit]

            # Track for citations
            if uploads:
                for idx, upload in enumerate(uploads, start=1):
                    upload_id = f"UPLOAD_{idx}"
                    meta = upload.get('metadata', {})
                    self.memory_id_map[upload_id] = {
                        'type': 'user_upload',
                        'timestamp': meta.get('timestamp', ''),
                        'content': upload.get('content', '')[:500],
                        'relevance_score': upload.get('relevance_score', 0.0),
                        'title': meta.get('title', ''),
                        'is_image': meta.get('is_image', False),
                        'image_path': meta.get('image_path', ''),
                        'db_id': upload.get('id', None),
                    }

                logger.debug(f"[ContextGatherer] Retrieved {len(uploads)} user uploads for query")

            return uploads or []

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get user uploads: {e}")
            return []

    async def get_git_commits(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get relevant git commits from PROCEDURAL collection.

        Uses hybrid retrieval: 1/3 most recent commits + 2/3 semantically
        relevant commits, deduplicated by document ID.

        Args:
            query: Search query for semantic retrieval
            limit: Maximum commits to return

        Returns:
            List of commit dicts with content, metadata, and relevance_score
        """
        try:
            chroma = getattr(self.memory_coordinator, 'chroma_store', None)
            if not chroma or 'procedural' not in chroma.collections:
                return []

            from config.app_config import GIT_MEMORY_ENABLED
            if not GIT_MEMORY_ENABLED:
                return []

            # Hybrid split: 1/3 recent, 2/3 semantic
            recent_limit = max(limit // 3, 1)
            semantic_limit = limit - recent_limit

            # 1/3: Most recent commits (chronological)
            recent = chroma.get_recent('procedural', limit=recent_limit)

            # 2/3: Semantically relevant commits
            semantic = chroma.query_collection('procedural', query, n_results=semantic_limit + recent_limit)

            # Deduplicate: recent first, then fill with semantic
            seen_ids = set()
            merged = []

            for commit in recent:
                doc_id = commit.get('id')
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(commit)

            for commit in semantic:
                if len(merged) >= limit:
                    break
                doc_id = commit.get('id')
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(commit)

            # Track for citations
            for idx, commit in enumerate(merged, start=1):
                cid = f"COMMIT_{idx}"
                meta = commit.get('metadata', {})
                self.memory_id_map[cid] = {
                    'type': 'git_commit',
                    'timestamp': meta.get('timestamp', ''),
                    'content': commit.get('content', '')[:500],
                    'relevance_score': commit.get('relevance_score', 0.0),
                    'commit_hash': meta.get('commit_hash', ''),
                    'db_id': commit.get('id', None),
                }

            logger.debug(f"[ContextGatherer] Retrieved {len(merged)} git commits ({len(recent)} recent + {len(semantic)} semantic, deduped to {len(merged)})")
            return merged

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get git commits: {e}")
            return []

    async def get_proposed_features(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get relevant code proposals (proposed features) for prompt injection.

        Uses ProposalFilter for retrieval, dedup, gating, and ranking.
        Only returns results for project-related queries.

        Args:
            query: Search query for relevance filtering
            limit: Maximum proposals to return

        Returns:
            List of proposal dicts with content, metadata, and relevance_score
        """
        try:
            from config.app_config import CODE_PROPOSALS_PROMPT_ENABLED
            if not CODE_PROPOSALS_PROMPT_ENABLED:
                return []

            # Lazy-init ProposalFilter
            if not hasattr(self, '_proposal_filter'):
                self._proposal_filter = None

            if self._proposal_filter is None:
                from .proposal_filter import ProposalFilter
                chroma = getattr(self.memory_coordinator, 'chroma_store', None)
                self._proposal_filter = ProposalFilter(
                    chroma_store=chroma,
                    gate_system=self._gate_system,
                    model_manager=self.model_manager,
                )

            proposals = await self._proposal_filter.get_proposals(query, limit=limit)

            # Track for citations
            for idx, prop in enumerate(proposals[:limit], start=1):
                prop_id = f"PROPOSAL_{idx}"
                meta = prop.get('metadata', {})
                self.memory_id_map[prop_id] = {
                    'type': 'code_proposal',
                    'timestamp': str(meta.get('created_at', '')),
                    'content': prop.get('content', '')[:500],
                    'relevance_score': prop.get('relevance_score', 0.0),
                    'title': meta.get('title', ''),
                    'priority': meta.get('priority', 5),
                    'db_id': meta.get('proposal_id', None),
                }

            logger.info(f"[ContextGatherer] Retrieved {len(proposals)} proposed features")
            return proposals or []

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get proposed features: {e}", exc_info=True)
            return []

    async def get_procedural_skills(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant procedural skills (adaptive workflows) from procedural_skills collection.

        Uses hybrid retrieval via MemoryCoordinator.get_skills().

        Args:
            query: Search query for semantic retrieval
            limit: Maximum skills to return

        Returns:
            List of skill dicts with content, metadata, and relevance_score
        """
        try:
            from config.app_config import PROCEDURAL_SKILLS_ENABLED
            if not PROCEDURAL_SKILLS_ENABLED:
                return []

            if not hasattr(self.memory_coordinator, 'get_skills'):
                return []

            skills = await self.memory_coordinator.get_skills(query, limit=limit)

            # Track for citations
            for idx, skill in enumerate(skills[:limit], start=1):
                skill_id = f"SKILL_{idx}"
                meta = skill.get('metadata', {})
                self.memory_id_map[skill_id] = {
                    'type': 'procedural_skill',
                    'timestamp': meta.get('created_at', ''),
                    'content': meta.get('trigger', '')[:500],
                    'relevance_score': skill.get('relevance_score', 0.0),
                    'category': meta.get('category', ''),
                    'db_id': skill.get('id', None),
                }

            logger.debug(f"[ContextGatherer] Retrieved {len(skills)} procedural skills")
            return skills or []

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get procedural skills: {e}")
            return []

    async def get_graph_context(self, query: str, max_sentences: int = 12) -> List[str]:
        """Retrieve knowledge graph context for entities mentioned in the query.

        Extracts entities from the query, traverses their graph neighborhood,
        and returns natural language sentences describing relationships.

        Args:
            query: User query to extract entities from
            max_sentences: Maximum sentences to return

        Returns:
            List of natural language relationship sentences
        """
        try:
            from config.app_config import KNOWLEDGE_GRAPH_ENABLED, KNOWLEDGE_GRAPH_RETRIEVAL_DEPTH, ENABLE_GRAPH_ATTRIBUTION
            if not KNOWLEDGE_GRAPH_ENABLED:
                return []

            mc = self.memory_coordinator
            graph = getattr(mc, "graph_memory", None)
            resolver = getattr(mc, "entity_resolver", None)
            if not graph or not resolver or graph.node_count() == 0:
                return []

            # Extract entity mentions from query using the shared utility
            # (strips punctuation, filters stopwords/common words, skips wikidata concepts)
            from memory.graph_utils import extract_graph_entities
            seen_entities = extract_graph_entities(query, resolver, graph_memory=graph)

            sentences: list[str] = []
            for eid in seen_entities:
                ctx = graph.get_context_sentences(
                    eid, depth=KNOWLEDGE_GRAPH_RETRIEVAL_DEPTH,
                    max_sentences=max_sentences - len(sentences),
                    with_attribution=ENABLE_GRAPH_ATTRIBUTION,
                )

                # Track graph sentences in memory_id_map for citation
                for i, sentence in enumerate(ctx):
                    citation_id = f"GRAPH_REL_{len(self.memory_id_map) + 1}"
                    self.memory_id_map[citation_id] = {
                        "content": sentence,
                        "entity_id": eid,
                        "source_type": "graph_relationship",
                        "metadata": {"entity": eid}
                    }

                sentences.extend(ctx)
                if len(sentences) >= max_sentences:
                    break

            if sentences:
                logger.debug(
                    f"[ContextGatherer] Graph context: {len(sentences)} sentences "
                    f"for entities {seen_entities}"
                )
            return sentences[:max_sentences]

        except Exception as e:
            logger.warning(f"[ContextGatherer] Graph context retrieval failed: {e}")
            return []

    async def get_unresolved_threads(self, max_results: int = 3) -> List[Dict[str, Any]]:
        """Get top priority unresolved threads for session surfacing.

        Delegates to MemoryCoordinator.get_unresolved_threads().

        Args:
            max_results: Maximum threads to return

        Returns:
            List of thread dicts with topic, summary, thread_type, urgency, deadline_date
        """
        try:
            from config.app_config import THREAD_SURFACING_ENABLED
            if not THREAD_SURFACING_ENABLED:
                return []

            if not hasattr(self.memory_coordinator, 'get_unresolved_threads'):
                return []

            threads = self.memory_coordinator.get_unresolved_threads(max_results=max_results)
            logger.debug(f"[ContextGatherer] Retrieved {len(threads)} unresolved threads")
            return threads or []

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get unresolved threads: {e}")
            return []

    async def get_proactive_insights(self, query: str, max_insights: int = 2) -> List[str]:
        """Get cross-domain proactive insights from the knowledge graph.

        Delegates to MemoryCoordinator.context_surfacer.generate_insights().

        Args:
            query: Current user query
            max_insights: Maximum insights to return

        Returns:
            List of insight text strings for prompt injection
        """
        try:
            from config.app_config import PROACTIVE_SURFACING_ENABLED, ENABLE_INSIGHT_ATTRIBUTION
            if not PROACTIVE_SURFACING_ENABLED:
                return []

            surfacer = getattr(self.memory_coordinator, 'context_surfacer', None)
            if not surfacer:
                return []

            raw_insights = await surfacer.generate_insights(query, max_insights=max_insights)

            # Add attribution markers and track in memory_id_map for citation
            attributed_insights = []
            for i, insight_text in enumerate(raw_insights):
                # Add attribution marker if enabled
                if ENABLE_INSIGHT_ATTRIBUTION:
                    attributed_text = f"Analysis suggests: {insight_text}"
                else:
                    attributed_text = insight_text

                # Track in memory_id_map for citation system
                citation_id = f"AI_INSIGHT_{len(self.memory_id_map) + 1}"
                self.memory_id_map[citation_id] = {
                    "content": attributed_text,
                    "original_insight": insight_text,
                    "source_type": "ai_synthesis",
                    "query": query,
                    "metadata": {"insight_index": i, "query": query}
                }

                attributed_insights.append(attributed_text)

            logger.debug(f"[ContextGatherer] Retrieved {len(attributed_insights)} proactive insights")
            return attributed_insights

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get proactive insights: {e}")
            return []

    async def get_visual_memories(self, query: str, max_images: int = 3) -> Dict[str, Any]:
        """Retrieve CLIP-matched visual memories for the query.

        Entity-gated: only retrieves when the query mentions an entity that has
        stored images (e.g. "Whiskers" → cat photos). Prevents irrelevant images
        from being injected into unrelated conversations.

        Args:
            query: Current user query
            max_images: Maximum images to include for multimodal API

        Returns:
            Dict with text_results (captions for prompt section) and
            images (base64 dicts for multimodal injection).
        """
        empty = {"text_results": [], "images": []}
        try:
            from config.app_config import VISUAL_MEMORY_ENABLED
            if not VISUAL_MEMORY_ENABLED:
                return empty

            if not hasattr(self, '_visual_retriever') or self._visual_retriever is None:
                from knowledge.clip_manager import get_clip_manager
                from knowledge.visual_memory_store import VisualMemoryStore
                from knowledge.visual_retrieval import VisualRetriever

                clip = get_clip_manager()
                chroma = getattr(self.memory_coordinator, 'chroma_store', None)
                store = VisualMemoryStore(chroma_store=chroma)
                self._visual_retriever = VisualRetriever(clip, store)

            # Entity-gated retrieval: only search when query mentions an entity
            # that has stored images (associative recall, not broad CLIP matching).
            # If no images have entity tags yet, skip retrieval entirely to prevent
            # irrelevant images from leaking into unrelated conversations.
            visual_entities = self._visual_retriever._store.get_visual_entity_ids()
            if not visual_entities:
                logger.debug(
                    "[ContextGatherer] Visual memory skipped: no entities tagged on stored images"
                )
                return empty

            # --- Step A: Clean visual entity set (filter junk caption artifacts) ---
            clean_visual = {
                eid for eid in visual_entities
                if len(eid) >= 4 and eid not in _VISUAL_JUNK_IDS
            }
            if not clean_visual:
                logger.debug("[ContextGatherer] Visual memory skipped: all entity IDs are junk")
                return empty

            # --- Step B: Entity resolution via knowledge graph aliases ---
            mc = self.memory_coordinator
            graph = getattr(mc, "graph_memory", None)
            resolver = getattr(mc, "entity_resolver", None)

            if resolver:
                from memory.graph_utils import extract_graph_entities
                query_entities = extract_graph_entities(query, resolver, graph_memory=graph)
                matched_entities = query_entities & clean_visual
            else:
                matched_entities = set()

            # --- Step C: Substring fallback (for visual-only entities not in graph) ---
            if not matched_entities:
                query_lower = query.lower()
                query_words = set(query_lower.split())
                matched_entities = {
                    eid for eid in clean_visual
                    if eid in query_words or eid in query_lower
                }

            if not matched_entities:
                logger.debug(
                    f"[ContextGatherer] Visual memory skipped: no entity match "
                    f"(clean visual entities: {len(clean_visual)})"
                )
                return empty

            # --- Step D: Multi-entity disambiguation via visual-intent proximity ---
            if len(matched_entities) > 1:
                import re
                query_lower = query.lower()
                words = query_lower.split()

                # Find positions of visual-intent keywords
                intent_positions = [
                    i for i, w in enumerate(words)
                    if re.sub(r'[^\w]', '', w) in _VISUAL_INTENT_WORDS
                ]

                if intent_positions:
                    # Score each entity by minimum distance to any intent word
                    entity_scores: dict[str, int] = {}
                    for eid in matched_entities:
                        # Find word positions where entity or its alias appears
                        eid_positions = [
                            i for i, w in enumerate(words)
                            if eid in re.sub(r'[^\w]', '', w.lower())
                        ]
                        # Check aliases too (e.g. "Bella" in text resolved to "whiskers")
                        if not eid_positions and resolver:
                            for i, w in enumerate(words):
                                clean_w = re.sub(r'[^\w]', '', w.lower())
                                if len(clean_w) > 2 and resolver.resolve(clean_w) == eid:
                                    eid_positions.append(i)

                        if eid_positions:
                            entity_scores[eid] = min(
                                abs(ep - ip)
                                for ep in eid_positions
                                for ip in intent_positions
                            )
                        else:
                            entity_scores[eid] = 999

                    if entity_scores:
                        best_dist = min(entity_scores.values())
                        focused = {eid for eid, d in entity_scores.items() if d <= best_dist + 2}
                        if focused and focused != matched_entities:
                            logger.debug(
                                f"[ContextGatherer] Visual entity focus (intent proximity): "
                                f"{matched_entities} → {focused}"
                            )
                            matched_entities = focused
                else:
                    # No visual-intent words — fall back to sentence-tail heuristic
                    segments = re.split(r'[.!?]+', query_lower)
                    tail = " ".join(s.strip() for s in segments[-2:] if s.strip())
                    tail_entities = set()
                    for eid in matched_entities:
                        if eid in tail:
                            tail_entities.add(eid)
                        elif resolver:
                            for w in tail.split():
                                if resolver.resolve(re.sub(r'[^\w]', '', w)) == eid:
                                    tail_entities.add(eid)
                                    break
                    if tail_entities and tail_entities != matched_entities:
                        logger.debug(
                            f"[ContextGatherer] Visual entity focus (tail): "
                            f"{matched_entities} → {tail_entities}"
                        )
                        matched_entities = tail_entities

            logger.debug(
                f"[ContextGatherer] Visual memory entity matches: {matched_entities}"
            )

            result = await self._visual_retriever.retrieve_visual_memories(
                query, max_images=max_images, target_entities=matched_entities
            )

            # Track in memory_id_map for citation
            for i, tr in enumerate(result.get("text_results", []), start=1):
                cid = f"VISUAL_{i}"
                self.memory_id_map[cid] = {
                    "type": "visual_memory",
                    "content": tr.get("caption", "")[:500],
                    "image_path": tr.get("image_path", ""),
                    "relevance_score": tr.get("score", 0.0),
                    "db_id": tr.get("doc_id", ""),
                }

            return result

        except Exception as e:
            logger.warning(f"[ContextGatherer] Visual memory retrieval failed: {e}")
            return empty

    async def get_codebase_changes(self, since_datetime) -> Dict[str, Any]:
        """Detect codebase file changes since last session via git.

        Runs git log, git diff, and git status to identify committed and
        uncommitted changes, filtering by allowed extensions and excluding
        build artifacts.

        Args:
            since_datetime: datetime object or None. If None, returns empty.

        Returns:
            Dict with committed, uncommitted_modified, uncommitted_new,
            since_label keys. Empty dict on failure or when disabled.
        """
        try:
            from config.app_config import (
                SESSION_DIFF_ENABLED,
                SESSION_DIFF_MAX_COMMITTED,
                SESSION_DIFF_MAX_UNCOMMITTED,
                SESSION_DIFF_EXTENSIONS,
            )
            if not SESSION_DIFF_ENABLED or since_datetime is None:
                return {}

            import subprocess
            from datetime import datetime

            # Resolve the repo root
            repo_root = None
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    repo_root = result.stdout.strip()
            except Exception:
                return {}
            if not repo_root:
                return {}

            # Exclusion patterns for paths
            _EXCLUDE_PATTERNS = ("__pycache__", ".pyc", "venv/", "dist/", "build/", ".egg-info")

            def _ext_ok(path: str) -> bool:
                """Check if file extension is in the allowed list."""
                import os as _os
                _, ext = _os.path.splitext(path)
                return ext.lower() in SESSION_DIFF_EXTENSIONS

            def _path_ok(path: str) -> bool:
                """Check if path is not in exclusion patterns."""
                return not any(excl in path for excl in _EXCLUDE_PATTERNS)

            def _filter(paths: list) -> list:
                return [p for p in paths if _ext_ok(p) and _path_ok(p)]

            # 1) Committed changes since last session
            iso_since = since_datetime.isoformat() if hasattr(since_datetime, 'isoformat') else str(since_datetime)
            committed = []
            try:
                result = subprocess.run(
                    ["git", "log", f"--since={iso_since}", "--oneline", "--no-merges"],
                    capture_output=True, text=True, timeout=10, cwd=repo_root
                )
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split("\n")
                    committed = lines[:SESSION_DIFF_MAX_COMMITTED]
            except Exception as e:
                logger.debug(f"[ContextGatherer] git log failed: {e}")

            # 2) Uncommitted modified files
            uncommitted_modified = []
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only"],
                    capture_output=True, text=True, timeout=10, cwd=repo_root
                )
                if result.returncode == 0 and result.stdout.strip():
                    files = result.stdout.strip().split("\n")
                    uncommitted_modified = _filter(files)[:SESSION_DIFF_MAX_UNCOMMITTED]
            except Exception as e:
                logger.debug(f"[ContextGatherer] git diff failed: {e}")

            # 3) Untracked new files
            uncommitted_new = []
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True, text=True, timeout=10, cwd=repo_root
                )
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split("\n")
                    untracked = [line[3:].strip() for line in lines if line.startswith("??")]
                    uncommitted_new = _filter(untracked)[:SESSION_DIFF_MAX_UNCOMMITTED]
            except Exception as e:
                logger.debug(f"[ContextGatherer] git status failed: {e}")

            # Human-readable time delta
            since_label = "last session"
            try:
                if hasattr(since_datetime, 'timestamp'):
                    now = datetime.now()
                    delta = now - since_datetime
                    total_secs = int(delta.total_seconds())
                    if total_secs < 3600:
                        since_label = f"{total_secs // 60}m ago"
                    elif total_secs < 86400:
                        hours = total_secs // 3600
                        mins = (total_secs % 3600) // 60
                        since_label = f"{hours}h {mins}m ago"
                    else:
                        days = total_secs // 86400
                        hours = (total_secs % 86400) // 3600
                        since_label = f"{days}d {hours}h ago"
            except Exception:
                pass

            if not committed and not uncommitted_modified and not uncommitted_new:
                return {}

            return {
                "committed": committed,
                "uncommitted_modified": uncommitted_modified,
                "uncommitted_new": uncommitted_new,
                "since_label": since_label,
            }

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to get codebase changes: {e}")
            return {}

    def _should_skip_wikipedia(self, query: str) -> bool:
        """Determine if query is too simple/conversational for Wikipedia lookup."""
        if not query:
            return True

        query_lower = query.lower().strip()
        words = query_lower.split()

        # Skip very short queries
        if len(words) <= 2:
            return True

        # Skip common conversational patterns
        conversational_patterns = [
            'hello', 'hi', 'hey', 'thanks', 'thank you', 'ok', 'okay',
            'yes', 'no', 'lol', 'haha', 'good', 'great', 'nice', 'cool',
            'how are you', 'what\'s up', 'see you', 'bye', 'goodbye',
            'yeah', 'yep', 'nope', 'sure', 'alright', 'sounds good',
            'i think', 'i feel', 'i hope', 'i guess', 'i mean',
            'that\'s', 'it\'s', 'i\'m', 'i am', 'going to', 'gonna',
        ]

        if any(pattern in query_lower for pattern in conversational_patterns):
            return True

        # Skip if query is mostly short words (< 4 chars)
        long_words = [w for w in words if len(w) >= 4 and w.isalpha()]
        if len(long_words) < 2:
            return True

        return False

    async def _get_wiki_content(self, query: str, limit: int = PROMPT_MAX_WIKI) -> List[Dict[str, Any]]:
        """Get wiki content for query.

        Prefers the local wiki_knowledge ChromaDB collection (pre-embedded
        Wikipedia corpus) for fast, relevant semantic retrieval.  Falls back
        to live Wikipedia API if the collection is empty or unavailable.
        """
        if not query:
            return []

        # Smart skip for simple/conversational queries
        if self._should_skip_wikipedia(query):
            return []

        # --- Try local ChromaDB wiki_knowledge first ---
        chroma = getattr(self.memory_coordinator, 'chroma_store', None)
        if chroma:
            try:
                coll = chroma.collections.get('wiki_knowledge')
                if coll and coll.count() > 0:
                    results = chroma.query_collection(
                        'wiki_knowledge', query, n_results=limit
                    )
                    if results:
                        # Track wiki titles for session enrichment
                        from knowledge.wiki_tracker import WikiArticleTracker
                        for r in results:
                            t = r.get('metadata', {}).get('title', '')
                            if t:
                                WikiArticleTracker.get_instance().track(t, r.get('content', '')[:500])
                        return [
                            {
                                'content': r.get('content', ''),
                                'metadata': r.get('metadata', {}),
                                'relevance_score': r.get('relevance_score', 0.0),
                                'source': 'wiki_knowledge',
                            }
                            for r in results
                        ]
            except Exception as e:
                logger.debug(f"[ContextGatherer] wiki_knowledge query failed, falling back to API: {e}")

        # --- Fallback: live Wikipedia API ---
        try:
            search_terms = []
            words = query.lower().split()
            for word in words:
                if len(word) > 3 and word.isalpha():
                    search_terms.append(word)

            wiki_results = []
            for term in search_terms[:limit]:
                snippet = await self._get_wiki_snippet_cached(term)
                if snippet:
                    wiki_results.append(snippet)

            return wiki_results

        except Exception as e:
            logger.warning(f"Error getting wiki content: {e}")
            return []

    async def _get_semantic_chunks(self, query: str, k: int = SEM_K,
                                 max_results: int = PROMPT_MAX_SEMANTIC) -> List[Dict[str, Any]]:
        """Get semantic chunks using semantic search."""
        if not query:
            return []

        try:
            # Use semantic search with neighbors
            results = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    semantic_search_with_neighbors,
                    query,
                    k
                ),
                timeout=SEM_TIMEOUT_S
            )

            if not results:
                return []

            # Filter by similarity threshold to prevent irrelevant wiki content
            pre_gate = len(results)
            results = [r for r in results
                       if r.get("similarity", 0) >= SEMANTIC_CHUNKS_GATE_THRESHOLD]
            if pre_gate != len(results):
                logger.debug(f"Semantic chunks gate: {pre_gate} -> {len(results)} (threshold={SEMANTIC_CHUNKS_GATE_THRESHOLD})")
            if not results:
                return []

            # Track wiki titles for session enrichment
            from knowledge.wiki_tracker import WikiArticleTracker
            tracker = WikiArticleTracker.get_instance()
            for r in results:
                t = r.get("title", "")
                if t:
                    tracker.track(t, r.get("content", "")[:500])

            # Process and stitch results by title
            chunks_by_title = {}
            for result in results[:max_results * 2]:  # Get more to allow stitching
                title = result.get("title", "")
                content = result.get("content", "")

                if title and content:
                    if title not in chunks_by_title:
                        chunks_by_title[title] = {
                            "title": title,
                            "content": content,
                            "metadata": result.get("metadata", {})
                        }
                    else:
                        # Stitch content together
                        existing = chunks_by_title[title]
                        combined = existing["content"] + "\n\n" + content

                        # Check length limit
                        if len(combined) <= SEM_STITCH_MAX_CHARS:
                            existing["content"] = combined
                        else:
                            # Start new chunk with different title
                            title_alt = f"{title} (continued)"
                            chunks_by_title[title_alt] = {
                                "title": title_alt,
                                "content": content,
                                "metadata": result.get("metadata", {})
                            }

            # Return limited results
            chunks = list(chunks_by_title.values())
            return chunks[:max_results]

        except asyncio.TimeoutError:
            logger.warning(f"Semantic search timeout after {SEM_TIMEOUT_S}s")
        except Exception as e:
            logger.warning(f"Semantic search error: {e}")

        return []

    async def _get_dreams(self, limit: int = PROMPT_MAX_DREAMS) -> List[Dict[str, Any]]:
        """Get dreams if enabled."""
        if not DREAMS_ENABLED:
            return []

        try:
            if hasattr(self.memory_coordinator, 'get_dreams'):
                dreams = self.memory_coordinator.get_dreams(limit)
                return dreams or []
        except Exception as e:
            logger.warning(f"Error getting dreams: {e}")

        return []

    def get_narrative_context(self) -> str:
        """
        Retrieve the cached narrative context (temporal grounding).

        This reads the pre-synthesized narrative from the filesystem.
        The narrative is generated asynchronously during summary creation
        and cached to avoid per-query latency costs.

        Returns:
            The narrative context string, or empty string if not available.
        """
        try:
            from config.app_config import NARRATIVE_CONTEXT_ENABLED
            if not NARRATIVE_CONTEXT_ENABLED:
                return ""

            # Access corpus manager through memory coordinator
            if hasattr(self.memory_coordinator, 'corpus_manager'):
                corpus = self.memory_coordinator.corpus_manager
                if hasattr(corpus, 'get_narrative_context'):
                    narrative = corpus.get_narrative_context()
                    if narrative:
                        logger.debug(f"[ContextGatherer] Retrieved narrative context ({len(narrative)} chars)")
                    return narrative

            logger.debug("[ContextGatherer] Narrative context not available (no corpus manager)")
            return ""

        except Exception as e:
            logger.warning(f"[ContextGatherer] Failed to retrieve narrative context: {e}")
            return ""

    async def get_daemon_self_notes(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve daemon self-notes relevant to the query.

        Queries the daemon_self_notes ChromaDB collection, filtering out
        superseded notes. Results are used in the prompt's [DAEMON SELF-NOTES] section.

        Returns:
            List of note dicts with content, metadata, relevance_score
        """
        chroma = getattr(self, '_chroma_store', None) or (
            getattr(self.memory_coordinator, 'chroma_store', None)
            if self.memory_coordinator else None
        )
        if not chroma:
            return []

        try:
            results = chroma.query_collection(
                "daemon_self_notes",
                query_text=query,
                n_results=limit * 2,
            )
            if not results:
                return []

            # Filter out superseded notes
            filtered = []
            for item in results:
                meta = item.get("metadata", {}) or {}
                if meta.get("status") == "superseded":
                    continue
                filtered.append(item)

            return filtered[:limit]

        except Exception as e:
            logger.debug(f"[ContextGatherer] daemon_self_notes retrieval failed: {e}")
            return []

    async def get_google_calendar_events(self, max_events: int = 10) -> List[Dict[str, Any]]:
        """Fetch upcoming Google Calendar events for prompt injection.

        Returns list of event dicts (summary, start, end, all_day, location).
        All failures return [] silently — calendar is best-effort context.
        """
        try:
            from config.app_config import GOOGLE_CALENDAR_ENABLED
        except ImportError:
            return []

        if not GOOGLE_CALENDAR_ENABLED:
            return []

        try:
            from core.actions.google_calendar import fetch_upcoming_events
            events = await fetch_upcoming_events(
                max_events=max_events,
            )
            return events
        except Exception as e:
            logger.debug(f"[ContextGatherer] Google Calendar fetch failed: {e}")
            return []
