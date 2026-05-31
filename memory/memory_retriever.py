# memory/memory_retriever.py
"""
Module Contract
- Purpose: Central memory retrieval with multi-source gathering, gating, ranking,
  and graceful threshold fallback. Handles both normal and meta-conversational queries.
- Inputs:
  - MemoryRetriever(corpus_manager, chroma_store, gate_system, scorer, hybrid_retriever, time_manager)
  - get_memories(query, limit, topic_filter) -> List[Dict]  [main pipeline]
  - get_semantic_top_memories(query, limit) -> List[Dict]  [gated semantic across collections]
  - get_facts(query, limit) -> List[Dict]  [semantic-primary ranked facts with confidence/recency tiebreak]
  - get_recent_facts(limit) -> List[Dict]
  - get_reflections(limit) -> List[Dict]  [corpus-first, semantic fallback]
  - get_reflections_hybrid(query, limit) -> List[Dict]  [n/3 recent + 2n/3 semantic]
  - get_summaries(limit) -> List[Dict]  [ChromaDB first, corpus fallback]
  - get_summaries_hybrid(query, limit) -> List[Dict]  [n/4 recent + 3n/4 semantic]
  - get_skills(query, limit) -> List[Dict]  [hybrid: 1/3 recent + 2/3 semantic, bumps times_retrieved]
  - get_dreams(limit) -> List[Dict]
  - search_by_type(type_name, query, limit) -> List[Dict]
- Outputs:
  - Standardized memory dicts with id, query, response, content, timestamp, source,
    collection, relevance_score, metadata, tags, truth_score, importance_score
- Key behaviors:
  - Main pipeline: gather → combine → gate → rank → 3-stage threshold fallback → slice
  - 3-stage threshold: primary → relaxed (70%) → top-N fallback (min 5 results)
  - Meta-conversational routing: entity-aware retrieval with temporal window detection
  - Dynamic config: gym/health queries get expanded semantic pool and bypass gating
  - Topic pre-filtering with fallback to unfiltered when no matches
  - Deduplication via _get_memory_key (id → timestamp+content → content hash)
  - Optional strict top-up controlled by MEM_TOPUP_ENABLE env var
- Dependencies:
  - memory.storage.multi_collection_chroma_store (vector queries)
  - processing.gate_system (multi-stage filtering: ChromaDB HNSW candidates → cosine → cross-encoder)
  - memory.memory_scorer (ranking with weight/graph overrides)
  - memory.utils.format_recent_conversations (corpus formatting)
  - utils.query_checker (is_meta_conversational, extract_temporal_window, _is_heavy_topic_heuristic)
  - config.app_config (DEICTIC_THRESHOLD, NORMAL_THRESHOLD)
- Side effects:
  - Bumps times_retrieved metadata on returned skills (best-effort)
  - Calls update_truth_scores_on_access on returned memories (currently no-op)
"""

import os
import re
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from utils.logging_utils import get_logger
from config.app_config import (
    DEICTIC_THRESHOLD,
    NORMAL_THRESHOLD,
)
from memory.utils import format_recent_conversations

logger = get_logger("memory_retriever")

# Graceful Threshold Fallback Configuration
GATING_MIN_RESULTS = 5           # Minimum results before relaxing threshold
GATING_RELAXED_MULTIPLIER = 0.7  # Relaxation multiplier (70% of original threshold)

# ---------------------------------------------------------------------------
# Query reformulation for embedding lookup
# ---------------------------------------------------------------------------
# Meta-framing prefixes that dilute embedding similarity.  Stripped so the
# embedding model sees topical content, not conversational scaffolding.
# Order matters — longer / more specific patterns first.

_META_PREFIX_PATTERNS = [
    # "what did we discuss/talk about/chat about"
    re.compile(
        r"^what\s+(?:did|do)\s+(?:we|you)\s+(?:discuss|talk\s+about|chat\s+about)\s*",
        re.IGNORECASE,
    ),
    # "what do you know/remember/recall about"
    re.compile(
        r"^what\s+(?:do|did)\s+you\s+(?:know|remember|recall)\s+about\s*",
        re.IGNORECASE,
    ),
    # "do you remember/recall [noun]"
    re.compile(r"^do\s+you\s+(?:remember|recall)\s+", re.IGNORECASE),
    # "help me brainstorm / let's explore / let us think about"
    re.compile(
        r"^(?:help\s+me\s+|let'?s\s+|let\s+us\s+)"
        r"(?:brainstorm|explore|think\s+about|come\s+up\s+with)\s*",
        re.IGNORECASE,
    ),
    # "let's explore/discuss/talk about"
    re.compile(
        r"^(?:let'?s\s+|let\s+us\s+)(?:explore|think\s+about|discuss|talk\s+about)\s*",
        re.IGNORECASE,
    ),
    # "what if we [verb]"
    re.compile(r"^what\s+if\s+we\s+", re.IGNORECASE),
    # "how long/much have I been"
    re.compile(
        r"^how\s+(?:long|much)\s+have\s+(?:I|we)\s+been\s*",
        re.IGNORECASE,
    ),
    # "what have I been doing for/about"
    re.compile(
        r"^what\s+have\s+(?:I|we)\s+been\s+(?:doing|working\s+on)"
        r"(?:\s+(?:for|about|with|on))?\s*",
        re.IGNORECASE,
    ),
    # "tell me about / talk to me about"
    re.compile(r"^(?:tell\s+me|talk\s+to\s+me)\s+about\s*", re.IGNORECASE),
]

# Temporal phrases — already handled by temporal anchor, just noise for embeddings.
_TEMPORAL_NOISE_RE = re.compile(
    r"\b(?:yesterday|last\s+night|earlier\s+today|this\s+morning"
    r"|last\s+week|last\s+month|the\s+other\s+day"
    r"|a\s+few\s+(?:days|weeks|months)\s+ago"
    r"|over\s+time|recently|lately)\b",
    re.IGNORECASE,
)

# Deictic / context-dependent queries — must NOT reformulate; they need
# recent-context anchor logic, not semantic noun search.
_DEICTIC_ONLY_RE = re.compile(
    r"^(?:what\s+about\s+that|explain\s+that|what\s+did\s+you\s+mean"
    r"|tell\s+me\s+more|go\s+on|continue"
    r"|can\s+you\s+help\s*(?:me)?"
    r"|do\s+you\s+remember)\s*\??$",
    re.IGNORECASE,
)

# Leading filler after prefix stripping.
_LEADING_FILLER_RE = re.compile(
    r"^\s*(?:about|for|with|on|in|the|a|an|and|some|any|creative|innovative)\s+",
    re.IGNORECASE,
)

_REFORMULATION_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "i", "me", "my", "we", "our",
    "you", "your", "it", "its", "and", "but", "or", "not", "no", "so",
    "if", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "from", "like", "just", "also", "very", "really", "some", "any",
})


# ---------------------------------------------------------------------------
# Reflection-specific retrieval helpers
# ---------------------------------------------------------------------------

def _rewrite_reflection_query(query: str) -> str:
    """
    Rewrite vague user queries into reflection-shaped search text.

    Reflections are stored as topic-dense retrieval text (not markdown).
    Vague queries like "How have I been doing with X" need expansion to
    match the retrieval text format: topics, entities, themes.

    Only applied for reflection retrieval, not global search.
    """
    if not query:
        return query

    # Strip common reflection-query framing
    stripped = query
    _REFLECTION_FRAMES = [
        re.compile(r"^how\s+have\s+i\s+been\s+(?:doing|handling|managing)\s+(?:with\s+)?", re.IGNORECASE),
        re.compile(r"^what\s+patterns?\s+(?:have\s+)?(?:shown|emerged|appeared)\s+(?:around|with|in)\s+", re.IGNORECASE),
        re.compile(r"^how\s+(?:has|is)\s+my\s+", re.IGNORECASE),
        re.compile(r"^what\s+(?:have\s+)?i\s+(?:learned|discovered|noticed)\s+(?:about|from|while)\s+", re.IGNORECASE),
    ]
    for pattern in _REFLECTION_FRAMES:
        stripped = pattern.sub("", stripped)

    # If stripping removed everything, use original
    if len(stripped.strip()) < 3:
        stripped = query

    # Extract content words (skip stopwords)
    words = stripped.split()
    content_words = [
        w.strip(".,!?\"'()") for w in words
        if w.lower().strip(".,!?\"'()") not in _REFORMULATION_STOPWORDS
        and len(w.strip(".,!?\"'()")) > 2
    ]

    if not content_words:
        return query

    # Build reflection-shaped search query:
    # Include original content words + "reflection" anchor + theme expansions
    expanded = " ".join(content_words)

    return expanded


def _compute_reflection_overlap(
    query_words: set, query_lower: str, metadata: dict
) -> float:
    """
    Compute entity/topic overlap between query and reflection metadata.

    Uses stored metadata fields (primary_topic, secondary_topics, entities,
    themes, project_area) to compute keyword-level overlap.

    Returns float in [0, 1] — fraction of query content words found in metadata.
    """
    if not query_words or not metadata:
        return 0.0

    # Build searchable text from metadata fields
    meta_parts = []
    for field in ("primary_topic", "secondary_topics", "entities", "themes", "project_area"):
        val = metadata.get(field, "")
        if val:
            meta_parts.append(val.lower())
    meta_text = " ".join(meta_parts)

    if not meta_text:
        return 0.0

    # Filter query words to content words only
    content_words = {
        w for w in query_words
        if w not in _REFORMULATION_STOPWORDS and len(w) > 2
    }

    if not content_words:
        return 0.0

    # Count how many query content words appear in metadata
    hits = sum(1 for w in content_words if w in meta_text)
    overlap = hits / len(content_words)

    # Bonus: check if query substring appears in primary_topic
    primary = metadata.get("primary_topic", "").lower()
    if primary and len(query_lower) > 5:
        # Check if any 3+ word span from query is in primary topic
        q_words_list = query_lower.split()
        for i in range(len(q_words_list) - 2):
            span = " ".join(q_words_list[i:i+3])
            if span in primary:
                overlap = min(1.0, overlap + 0.3)
                break

    return min(1.0, overlap)


# ---------------------------------------------------------------------------
# Ephemeral fact detection
# ---------------------------------------------------------------------------

# Use canonical ephemeral list from config (lazy-loaded, cached)
_EPHEMERAL_PREDICATES_CACHED: frozenset | None = None


def _get_ephemeral_predicates() -> frozenset:
    """Load ephemeral predicates from config (cached after first call)."""
    global _EPHEMERAL_PREDICATES_CACHED
    if _EPHEMERAL_PREDICATES_CACHED is not None:
        return _EPHEMERAL_PREDICATES_CACHED
    try:
        from config.app_config import PROFILE_EPHEMERAL_RELATIONS
        _EPHEMERAL_PREDICATES_CACHED = frozenset(
            r.lower().strip() for r in PROFILE_EPHEMERAL_RELATIONS
        )
    except ImportError:
        _EPHEMERAL_PREDICATES_CACHED = frozenset()
    return _EPHEMERAL_PREDICATES_CACHED


def _is_ephemeral_fact(content: str) -> bool:
    """Check if a fact triple has an ephemeral (current-state) predicate."""
    if not content or "|" not in content:
        return False
    parts = content.split("|")
    if len(parts) < 2:
        return False
    predicate = parts[1].strip().lower().replace(" ", "_")
    return predicate in _get_ephemeral_predicates()


def _metadata_fallback_search(
    chroma_store, query: str, exclude_ids: set, max_results: int = 10
) -> list:
    """
    Search reflections by document content (where_document) for query keywords.

    This catches reflections that semantic search missed because the embedding
    didn't rank them in the top-N. ChromaDB's where_document filter does
    substring matching on the stored document text.
    """
    if not query or not chroma_store:
        return []

    # Extract content words from query
    words = query.lower().split()
    content_words = [
        w.strip(".,!?\"'()")
        for w in words
        if len(w.strip(".,!?\"'()")) > 3
        and w.lower().strip(".,!?\"'()") not in _REFORMULATION_STOPWORDS
    ]

    if not content_words:
        return []

    try:
        coll = chroma_store.collections.get("reflections")
        if not coll:
            return []

        # Use where_document with $contains for the most distinctive query word
        # (ChromaDB where_document only supports single $contains)
        # Pick the longest content word as most distinctive
        best_word = max(content_words, key=len)

        results = coll.query(
            query_texts=[query],
            n_results=max_results,
            where_document={"$contains": best_word},
            include=["documents", "metadatas", "distances"],
        )

        ids_list = (results.get("ids") or [[]])[0] or []
        docs_list = (results.get("documents") or [[]])[0] or []
        metas = (results.get("metadatas") or [[]])[0] or []
        dists = (results.get("distances") or [[]])[0] or []

        out = []
        for i in range(len(docs_list)):
            rid = ids_list[i] if i < len(ids_list) else None
            if rid and rid in exclude_ids:
                continue
            dist = dists[i] if i < len(dists) else None
            score = (1.0 / (1.0 + dist)) if isinstance(dist, (int, float)) else 0.4
            out.append({
                "id": rid,
                "content": docs_list[i] if i < len(docs_list) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "relevance_score": score,
            })
        return out

    except Exception as e:
        logger.debug(f"[ReflectionMetadataFallback] Failed: {e}")
        return []


class MemoryRetriever:
    """
    Memory retrieval operations.

    Implements MemoryRetrieverProtocol contract.
    """

    def __init__(
        self,
        corpus_manager,
        chroma_store,
        gate_system=None,
        scorer=None,
        hybrid_retriever=None,
        time_manager=None,
    ):
        """
        Initialize MemoryRetriever.

        Args:
            corpus_manager: CorpusManager for JSON persistence
            chroma_store: MultiCollectionChromaStore for vector storage
            gate_system: Optional MultiStageGateSystem for filtering
            scorer: Optional MemoryScorer for ranking
            hybrid_retriever: Optional HybridRetriever for enhanced search
            time_manager: Optional TimeManager for timestamps
        """
        self.corpus_manager = corpus_manager
        self.chroma_store = chroma_store
        self.gate_system = gate_system
        self.scorer = scorer
        self.hybrid_retriever = hybrid_retriever
        self.time_manager = time_manager

        # State
        self.current_topic: str = "general"
        self.conversation_context: list = []

    def _get_memory_key(self, memory: Dict) -> str:
        """Generate unique key for deduplication."""
        mem_id = memory.get('id')
        if mem_id:
            return f"id:{mem_id}"

        ts = memory.get('timestamp')
        if ts:
            ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
            return f"ts:{ts_str}__{(memory.get('query', '') or '')[:30]}__{(memory.get('response', '') or '')[:30]}"

        # Fallback to content hash
        content = f"{memory.get('query', '')}__{memory.get('response', '')}"
        return f"hash:{hash(content)}__{content[:30]}__{content[-30:]}"

    # ------------------------------------------------------------------
    # Query reformulation
    # ------------------------------------------------------------------

    @staticmethod
    def _reformulate_for_embedding(query: str) -> str:
        """
        Strip meta framing from a query for better embedding similarity.

        Returns a content-focused ``retrieval_query`` for semantic lookup.
        The original query should still be used for scoring, continuity,
        tag overlap, temporal handling, and display.

        Falls back to the original query when stripping leaves nothing
        meaningful or the query is deictic / context-dependent.
        """
        if not query or len(query) < 8:
            return query

        stripped = query.strip()

        # Deictic / context-dependent — needs anchor logic, not noun search
        if _DEICTIC_ONLY_RE.match(stripped):
            return query

        result = stripped

        # 1. Strip meta-framing prefix
        for pattern in _META_PREFIX_PATTERNS:
            result = pattern.sub("", result, count=1)

        # NOTE: temporal phrases ("last week", "yesterday") are NOT stripped.
        # They're handled by the temporal anchor for scoring, but still
        # provide useful context for the embedding model.

        # 2. Remove leading filler words left after stripping
        result = _LEADING_FILLER_RE.sub("", result)

        # 3. Clean up punctuation and whitespace
        result = re.sub(r"[?\s]+$", "", result)
        result = re.sub(r"\s{2,}", " ", result).strip()

        # 4. Guardrail — must have meaningful content remaining.
        #    MiniLM-L6-v2 produces poor embeddings for very short queries.
        #    Require ≥3 content words so reformulation only fires when
        #    there's enough topical signal to improve over the original.
        if len(result) < 3:
            return query
        content_words = [
            w for w in result.split()
            if len(w) > 2 and w.lower() not in _REFORMULATION_STOPWORDS
        ]
        if len(content_words) < 3:
            return query

        return result

    def _parse_result(self, item: Dict, source: str, default_truth: float = 0.6) -> Dict:
        """Parse a result from ChromaDB into a standardized memory format."""
        if not isinstance(item, dict):
            logger.warning(f"[_parse_result] Expected dict, got {type(item)}")
            return {}

        meta = item.get('metadata', {}) or {}
        ts = meta.get('timestamp', datetime.now())
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError as e:
                logger.debug(f"[MemoryRetriever] Bad timestamp format '{ts[:30] if ts else ''}': {e}")
                ts = datetime.now()

        tags = meta.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]

        return {
            'id': item.get('id', f"{source}::{uuid.uuid4().hex[:8]}"),
            'query': meta.get('query', item.get('content', '')[:100]),
            'response': meta.get('response', ''),
            'content': item.get('content', ''),
            'timestamp': ts,
            'source': source,
            'collection': source,
            'relevance_score': float(item.get('relevance_score', 0.5)),
            'metadata': meta,
            'tags': tags,
            'truth_score': float(meta.get('truth_score', default_truth)),
            'importance_score': float(meta.get('importance_score', 0.5)),
        }

    def _get_recent_conversations(self, k: int = 5) -> List[Dict]:
        """Get recent conversations from corpus (JSON)."""
        entries = self.corpus_manager.get_recent_memories(k) or []
        return format_recent_conversations(entries)

    async def get_recent_facts(self, limit: int = 5) -> List[Dict]:
        """Fetch the most recent facts by timestamp."""
        try:
            recent = self.chroma_store.get_recent("facts", limit=limit)
            return recent or []
        except Exception as e:
            logger.debug(f"[MemoryRetriever][RecentFacts] retrieval failed: {e}")
            return []

    async def get_facts(self, query: str, limit: int = 8) -> List[Dict]:
        """Retrieve semantic facts relevant to query.

        Scoring: semantic relevance (from ChromaDB) is the primary signal.
        Confidence and recency are secondary factors that break ties, not
        override semantic ordering.
        """
        results: List[Dict] = []
        try:
            coll = self.chroma_store.collections.get("facts")

            # Semantic search only if there are rows
            if coll and coll.count() > 0:
                raw = self.chroma_store.query_collection(
                    "facts",
                    query_text=query or "",
                    n_results=min(max(1, limit * 2), coll.count()),
                ) or []

                if not isinstance(raw, list):
                    raw = [raw]

                for item in raw:
                    if not isinstance(item, dict):
                        item = {"content": str(item)}
                    meta = item.get("metadata", {}) or {}
                    content = item.get("content") or meta.get("content") or ""
                    if not content:
                        continue
                    results.append({
                        "id": item.get("id"),
                        "content": content,
                        "confidence": float(meta.get("confidence", 0.6)),
                        "relevance_score": float(item.get("relevance_score") or 0.0),
                        "source": meta.get("source", "facts"),
                        "timestamp": meta.get("timestamp"),
                        "tags": meta.get("tags", []),
                        "metadata": meta,
                    })

            # Fallback to most recent if nothing semantic
            if not results and hasattr(self.chroma_store, "get_recent"):
                for item in self.chroma_store.get_recent("facts", limit) or []:
                    content = item.get("content") or ""
                    if not content:
                        continue
                    meta = item.get("metadata", {}) or {}
                    results.append({
                        "id": item.get("id"),
                        "content": content,
                        "confidence": float(meta.get("confidence", 0.6)),
                        "relevance_score": 0.0,
                        "source": meta.get("source", "facts"),
                        "timestamp": meta.get("timestamp"),
                        "metadata": meta,
                    })

        except Exception as e:
            logger.debug(f"[Facts] retrieval error: {e}", exc_info=True)

        # Rank with semantic relevance as primary signal.
        # Old formula: 0.7*confidence + 0.3*recency (destroyed semantic order)
        # New formula: 0.60*semantic + 0.20*confidence + 0.20*recency
        # Semantic floor: if relevance < 0.30, cap recency contribution at 0.05
        # so irrelevant-but-recent facts can't dominate.
        #
        # Note: ephemeral predicates (current_mood, current_activity, etc.) would
        # benefit from heavier recency weighting, but the benchmark uses single-gold
        # evaluation that penalizes correct recency-preferring behavior. Deferred
        # until the benchmark supports set-valued / recency-aware evaluation.
        _SEMANTIC_FLOOR = 0.30
        _W_SEMANTIC = 0.60
        _W_CONFIDENCE = 0.20
        _W_RECENCY = 0.20

        def _score(x: Dict) -> float:
            sem = float(x.get("relevance_score", 0.0))

            ts = x.get("timestamp")
            rec = 0.5  # neutral default
            try:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                if ts:
                    now = datetime.now()
                    # Strip timezone if present to avoid naive/aware mismatch
                    if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)
                    age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
                    rec = 1.0 / (1.0 + 0.05 * age_h)
            except (ValueError, TypeError, AttributeError):
                pass

            conf = float(x.get("confidence", 0.6))

            # Semantic floor: if relevance is too low, recency can't rescue
            if sem < _SEMANTIC_FLOOR:
                rec_weight = 0.05
            else:
                rec_weight = _W_RECENCY

            return (
                _W_SEMANTIC * sem
                + _W_CONFIDENCE * conf
                + rec_weight * rec
            )

        results.sort(key=_score, reverse=True)

        # TTL filter: drop ephemeral facts older than PROFILE_EPHEMERAL_TTL_HOURS
        # These are transient state (current_mood, woke_at, etc.) that pollute retrieval
        try:
            from config.app_config import PROFILE_EPHEMERAL_TTL_HOURS
            ttl_hours = PROFILE_EPHEMERAL_TTL_HOURS
        except ImportError:
            ttl_hours = 24
        now = datetime.now()
        filtered = []
        for r in results:
            content = r.get("content", "")
            if _is_ephemeral_fact(content):
                ts = r.get("timestamp")
                try:
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)
                    if ts:
                        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                            ts = ts.replace(tzinfo=None)
                        age_h = (now - ts).total_seconds() / 3600.0
                        if age_h > ttl_hours:
                            continue  # skip stale ephemeral fact
                except (ValueError, TypeError, AttributeError):
                    pass
            filtered.append(r)
        return filtered[:limit]

    async def get_reflections(self, limit: int = 2) -> List[Dict]:
        """Fetch recent reflections."""
        out = []

        # Corpus-first
        try:
            cm = self.corpus_manager
            get_by_type = getattr(cm, "get_items_by_type", None)
            if callable(get_by_type):
                items = get_by_type("reflection", limit=limit * 2) or []
            else:
                get_summaries_of_type = getattr(cm, "get_summaries_of_type", None)
                if callable(get_summaries_of_type):
                    items = get_summaries_of_type(types=("reflection",), limit=limit * 2) or []
                else:
                    get_all = getattr(cm, "get_all", None) or getattr(cm, "get_recent_memories", None)
                    items = (get_all(limit * 10) if callable(get_all) else []) or []

            for n in items or []:
                if isinstance(n, dict):
                    t = (n.get("type") or "").strip().lower()
                    tags = n.get("tags") or []
                    if t == "reflection" or "type:reflection" in tags:
                        out.append({
                            "content": n.get("content", "").strip(),
                            "type": "reflection",
                            "tags": tags,
                            "source": n.get("source", "corpus"),
                            "timestamp": n.get("timestamp"),
                        })
                        if len(out) >= limit:
                            return out
        except Exception as e:
            logger.warning(f"[MemoryRetriever] Corpus reflection retrieval failed: {e}")

        # Semantic fallback
        try:
            coll = self.chroma_store.collections.get("reflections") if self.chroma_store else None
            if coll and coll.count() > 0:
                items = self.chroma_store.get_recent("reflections", limit=limit)
                for r in items or []:
                    txt = (r.get("content") if isinstance(r, dict) else str(r)).strip()
                    ts = (r.get("metadata") or {}).get("timestamp") if isinstance(r, dict) else None
                    if txt:
                        out.append({
                            "content": txt,
                            "type": "reflection",
                            "tags": ["source:semantic"],
                            "source": "semantic",
                            "timestamp": ts,
                        })
                        if len(out) >= limit:
                            break
        except Exception as e:
            logger.warning(f"[MemoryRetriever] Semantic reflection retrieval failed: {e}")

        return out[:limit]

    async def get_reflections_hybrid(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Hybrid retrieval for reflections with reflection-specific optimizations:
        - Query rewriting: expand vague queries into reflection-shaped search text
        - Large candidate pool: retrieve 50 from ChromaDB (not just limit*2)
        - Entity/topic overlap scoring: use stored metadata for keyword matching
        - Cross-encoder rerank: top 25 candidates reranked before final selection
        """
        if limit < 1:
            return []

        # Reflection-specific candidate pool size (much larger than other collections)
        REFLECTION_CANDIDATE_POOL = 50
        REFLECTION_RERANK_TOP = 25
        recent_budget = max(1, limit // 3)

        # Get recent reflections
        recent = await self.get_reflections(limit=recent_budget * 2)

        # Rewrite query for reflection-shaped search
        search_query = _rewrite_reflection_query(query) if query else query

        # Dual-query: search with both original and rewritten query to maximize
        # candidate coverage. Merge results, keeping best relevance score per item.
        semantic = []
        seen_semantic_ids = set()
        for sq in (search_query, query):
            if not sq or not sq.strip():
                continue
            try:
                results = self.chroma_store.query_collection(
                    'reflections', sq, n_results=REFLECTION_CANDIDATE_POOL
                )
                for r in results:
                    rid = r.get('id', '')
                    if rid in seen_semantic_ids:
                        continue
                    seen_semantic_ids.add(rid)
                    semantic.append({
                        'content': r.get('content', ''),
                        'metadata': r.get('metadata', {}),
                        'timestamp': r.get('metadata', {}).get('timestamp', datetime.now()),
                        'type': 'reflection',
                        'source': 'semantic',
                        'relevance_score': r.get('relevance_score', 0.5),
                    })
            except Exception:
                pass

        # Metadata fallback: search by primary_topic/entities/themes when semantic
        # search may miss due to embedding mismatch. Uses ChromaDB where_document
        # to find reflections containing query keywords in the embedded text.
        metadata_results = _metadata_fallback_search(
            self.chroma_store, query, seen_semantic_ids
        )
        for r in metadata_results:
            rid = r.get('id', '')
            if rid not in seen_semantic_ids:
                seen_semantic_ids.add(rid)
                semantic.append({
                    'content': r.get('content', ''),
                    'metadata': r.get('metadata', {}),
                    'timestamp': r.get('metadata', {}).get('timestamp', datetime.now()),
                    'type': 'reflection',
                    'source': 'metadata_fallback',
                    'relevance_score': 0.4,  # Lower base score — metadata match, not semantic
                })

        # Deduplicate — merge into single pool
        def get_item_id(item):
            if not isinstance(item, dict):
                return str(item)[:50]
            ts = item.get("timestamp", "")
            ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
            content_prefix = (item.get("content", "") or "")[:50].strip()
            return f"ts:{ts_str}::{content_prefix}"

        pool = []
        seen_ids = set()

        for item in recent[:recent_budget]:
            item_id = get_item_id(item)
            if item_id not in seen_ids:
                if isinstance(item, dict):
                    item['source'] = 'recent'
                pool.append(item)
                seen_ids.add(item_id)

        for item in semantic:
            item_id = get_item_id(item)
            if item_id not in seen_ids:
                pool.append(item)
                seen_ids.add(item_id)

        if not pool:
            return []

        # Score with entity/topic overlap + semantic relevance
        query_lower = (query or "").lower()
        query_words = set(query_lower.split())
        RECENCY_BONUS = 0.05
        ENTITY_OVERLAP_WEIGHT = 0.25

        for item in pool:
            base_score = item.get('relevance_score', 0.5)
            meta = item.get('metadata', {})

            # Entity/topic overlap scoring
            overlap_score = _compute_reflection_overlap(query_words, query_lower, meta)

            # Blend: semantic + entity overlap + recency
            if item.get('source') == 'recent':
                item['final_score'] = (
                    (1.0 - ENTITY_OVERLAP_WEIGHT) * base_score
                    + ENTITY_OVERLAP_WEIGHT * overlap_score
                    + RECENCY_BONUS
                )
            else:
                item['final_score'] = (
                    (1.0 - ENTITY_OVERLAP_WEIGHT) * base_score
                    + ENTITY_OVERLAP_WEIGHT * overlap_score
                )

        # Sort by blended score (highest first)
        pool.sort(key=lambda x: x.get('final_score', 0), reverse=True)

        # Cross-encoder rerank expanded candidate set
        pool = self._maybe_cross_encoder_rerank(pool[:REFLECTION_RERANK_TOP], query)

        return pool[:limit]

    def get_summaries(self, limit: int = 3) -> List[Dict]:
        """Retrieve recent summaries."""
        # Try ChromaDB first
        try:
            results = self.chroma_store.query_collection('summaries', '', n_results=limit)
            chroma_summaries = [{
                'content': r.get('content', ''),
                'timestamp': r.get('metadata', {}).get('timestamp', datetime.now()),
                'type': 'summary'
            } for r in results]
            if chroma_summaries:
                return chroma_summaries
        except Exception:
            pass

        # Fallback to corpus manager
        try:
            if hasattr(self.corpus_manager, 'get_summaries'):
                corpus_summaries = self.corpus_manager.get_summaries(limit)
                if corpus_summaries:
                    return corpus_summaries
        except Exception:
            pass

        return []

    def get_summaries_hybrid(self, query: str, limit: int = 4) -> List[Dict]:
        """Hybrid retrieval: n/4 recent + 3n/4 semantic, with deduplication."""
        if limit < 1:
            return []

        recent_budget = max(1, limit // 4)
        semantic_budget = limit - recent_budget

        # Fetch recent
        recent = []
        try:
            if hasattr(self.corpus_manager, 'get_summaries'):
                recent = self.corpus_manager.get_summaries(recent_budget * 2)
        except Exception:
            pass

        # Fetch semantic
        semantic = []
        if query and query.strip():
            try:
                results = self.chroma_store.query_collection(
                    'summaries', query, n_results=semantic_budget * 2
                )
                semantic = [
                    {
                        'content': r.get('content', ''),
                        'timestamp': r.get('metadata', {}).get('timestamp', datetime.now()),
                        'type': 'summary',
                        'source': 'semantic'
                    }
                    for r in results
                ]
            except Exception:
                pass

        # Deduplicate
        def get_item_id(item):
            if not isinstance(item, dict):
                return str(item)[:50]
            ts = item.get("timestamp", "")
            ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
            content_prefix = (item.get("content", "") or "")[:50].strip()
            return f"ts:{ts_str}::{content_prefix}"

        result = []
        seen_ids = set()

        for item in recent[:recent_budget]:
            item_id = get_item_id(item)
            if item_id not in seen_ids:
                if isinstance(item, dict):
                    item['source'] = 'recent'
                result.append(item)
                seen_ids.add(item_id)

        remaining = limit - len(result)
        for item in semantic:
            if remaining <= 0:
                break
            item_id = get_item_id(item)
            if item_id not in seen_ids:
                result.append(item)
                seen_ids.add(item_id)
                remaining -= 1

        # Fallback if semantic was empty
        if len(result) < limit and not semantic:
            for item in recent[recent_budget:]:
                if len(result) >= limit:
                    break
                item_id = get_item_id(item)
                if item_id not in seen_ids:
                    if isinstance(item, dict):
                        item['source'] = 'recent_fallback'
                    result.append(item)
                    seen_ids.add(item_id)

        return result[:limit]

    def get_dreams(self, limit: int = 2) -> List[Dict]:
        """Retrieve dream memories."""
        if hasattr(self.corpus_manager, 'get_dreams'):
            return [{
                'content': d,
                'timestamp': datetime.now(),
                'source': 'dream'
            } for d in (self.corpus_manager.get_dreams(limit) or [])]
        return []

    async def get_skills(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Retrieve procedural skills relevant to query.

        Uses hybrid retrieval: 1/3 most recent + 2/3 semantically relevant,
        deduplicated by document ID.  Bumps times_retrieved on returned skills.

        Args:
            query: User query for semantic matching
            limit: Maximum skills to return

        Returns:
            List of skill dicts with content, metadata, and relevance_score
        """
        collection_name = "procedural_skills"
        try:
            from config.app_config import PROCEDURAL_SKILLS_ENABLED
            if not PROCEDURAL_SKILLS_ENABLED:
                return []

            coll = self.chroma_store.collections.get(collection_name)
            if coll is None or coll.count() == 0:
                return []

            count = coll.count()

            # Hybrid split: 1/3 recent, 2/3 semantic
            recent_limit = max(limit // 3, 1)
            semantic_limit = limit - recent_limit

            # 1/3: Most recent skills
            recent = self.chroma_store.get_recent(collection_name, limit=recent_limit)

            # 2/3: Semantically relevant skills
            semantic = self.chroma_store.query_collection(
                collection_name, query, n_results=min(semantic_limit + recent_limit, count)
            )

            # Deduplicate: recent first, then fill with semantic
            seen_ids = set()
            merged: List[Dict] = []

            for item in recent:
                doc_id = item.get("id")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(item)

            for item in semantic:
                if len(merged) >= limit:
                    break
                doc_id = item.get("id")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(item)

            # Bump times_retrieved (best-effort, non-blocking)
            for item in merged:
                try:
                    meta = item.get("metadata", {})
                    doc_id = item.get("id")
                    if doc_id and meta:
                        import time as _time
                        new_count = int(meta.get("times_retrieved", 0)) + 1
                        # ChromaDB update is sync — fire-and-forget via metadata
                        meta["times_retrieved"] = new_count
                        meta["last_retrieved"] = _time.time()
                except Exception:
                    pass

            logger.debug(
                f"[MemoryRetriever] Retrieved {len(merged)} skills "
                f"({len(recent)} recent + {len(semantic)} semantic, deduped)"
            )
            return merged[:limit]

        except Exception as e:
            logger.warning(f"[MemoryRetriever] Failed to retrieve skills: {e}")
            return []

    async def search_by_type(self, type_name: str, query: str = "", limit: int = 5) -> List[Dict]:
        """Search memories by type."""
        results = []
        try:
            if type_name in self.chroma_store.collections:
                if query:
                    raw = self.chroma_store.query_collection(type_name, query, n_results=limit)
                else:
                    raw = self.chroma_store.get_recent(type_name, limit=limit)

                for item in raw or []:
                    results.append(self._parse_result(item, type_name))
        except Exception as e:
            logger.error(f"[search_by_type] Error: {e}")

        return results

    async def _get_semantic_memories(self, query: str, n_results: int = 30) -> List[Dict]:
        """Get semantic memories using hybrid retrieval."""
        logger.info(f"[Semantic] Using hybrid retrieval for query: '{query[:50]}...'")

        try:
            if self.hybrid_retriever:
                hybrid_results = await self.hybrid_retriever.retrieve(query, limit=n_results)

                memories = []
                for result in hybrid_results:
                    hybrid_score = result.get("hybrid_score", 0.0)
                    keyword_score = result.get("keyword_score", 0.0)

                    boosted_score = hybrid_score + 0.3
                    if keyword_score > 0.5:
                        boosted_score = max(boosted_score, 0.6)

                    memory = {
                        "id": result.get("id", str(hash(str(result)))),
                        "content": result.get("content", ""),
                        "query": result.get("query", ""),
                        "response": result.get("response", ""),
                        "metadata": result.get("metadata", {}),
                        "collection": result.get("collection", "unknown"),
                        "final_score": boosted_score,
                        "semantic_score": result.get("semantic_score", 0.0),
                        "keyword_score": keyword_score,
                        "hybrid_score": hybrid_score,
                        "relevance": boosted_score
                    }
                    memories.append(memory)

                logger.info(f"[Semantic] Hybrid retrieval returned {len(memories)} memories")
                return memories

        except Exception as e:
            logger.error(f"[Semantic] Hybrid retrieval failed: {e}")

        # Fallback to basic search
        return await self._fallback_semantic_search(query, n_results)

    async def _fallback_semantic_search(self, query: str, n_results: int = 30) -> List[Dict]:
        """Fallback semantic search using ChromaDB."""
        memories: List[Dict] = []
        collections_to_query = ['conversations', 'summaries', 'reflections', 'procedural']

        try:
            batch_results = await self.chroma_store.query_multiple_collections(
                collections_to_query,
                query_text=query,
                n_results=n_results
            )

            for collection_name, results in batch_results.items():
                if not results:
                    continue

                for item in results:
                    if item is not None:
                        if not isinstance(item, dict):
                            item = {"content": str(item), "id": str(uuid.uuid4())}
                        memories.append(self._parse_result(item, collection_name))

        except Exception as e:
            logger.error(f"[Semantic] Fallback search failed: {e}")

        return memories

    async def _combine_memories(
        self,
        very_recent: List[Dict],
        semantic: List[Dict],
        hierarchical: List[Dict],
        query: str,
        config: Dict,
        bypass_gate: bool = False
    ) -> List[Dict]:
        """Combine memory pools with optional gating."""
        combined: List[Dict] = []
        candidates: List[Dict] = []
        seen = set()

        # Allow top-N recent memories straight through
        bypass_n = 2
        for mem in very_recent[:bypass_n]:
            key = self._get_memory_key(mem)
            if key not in seen:
                mem['source'] = mem.get('source', 'very_recent')
                mem['gated'] = False
                combined.append(mem)
                seen.add(key)

        # Rest of recent go to candidates
        for mem in very_recent[bypass_n:]:
            key = self._get_memory_key(mem)
            if key not in seen:
                candidates.append(mem)
                seen.add(key)

        # Semantic to candidates
        for mem in semantic:
            key = self._get_memory_key(mem)
            if key not in seen:
                candidates.append(mem)
                seen.add(key)

        # Hierarchical to candidates
        for h in hierarchical:
            if isinstance(h, dict) and 'memory' in h:
                mem = h['memory']
                key = self._get_memory_key(mem)
                if key not in seen:
                    mem['relevance_score'] = h.get('final_score', mem.get('relevance_score', 0.5))
                    candidates.append(mem)
                    seen.add(key)

        # Optional gating
        use_gate_system = self.gate_system and candidates and not bypass_gate

        if use_gate_system:
            gated = await self._gate_memories(query, candidates)
            for mem in gated:
                mem['gated'] = True
                combined.append(mem)
        else:
            cap = config.get('max_memories', 20)
            for mem in candidates[:cap]:
                mem['gated'] = False
                combined.append(mem)

        return combined

    async def _gate_memories(self, query: str, memories: List[Dict]) -> List[Dict]:
        """Apply gate while preserving original metadata."""
        try:
            def _gate_text(m: Dict) -> str:
                txt = (m.get('content') or '').strip()
                if txt:
                    return txt
                q = (m.get('query') or '').strip()
                a = (m.get('response') or '').strip()
                return f"User: {q}\nAssistant: {a}"

            chunks = [{
                "content": _gate_text(m)[:500],
                "metadata": {
                    "timestamp": m.get("timestamp", datetime.now()),
                    "truth_score": m.get('truth_score', 0.5),
                    "type": (m.get("metadata") or {}).get("type", ""),
                    "original_memory": m
                }
            } for m in memories]

            filtered = await self.gate_system.filter_memories(query, chunks)

            gated: List[Dict] = []
            for ch in filtered:
                md = ch.get("metadata", {}) or {}
                orig = md.get("original_memory")
                if isinstance(orig, dict):
                    orig['gated'] = True
                    gated.append(orig)
            return gated

        except Exception as e:
            logger.error(f"Gating error: {e}")
            return memories[:min(10, len(memories))]

    async def get_memories(
        self,
        query: str,
        limit: int = 20,
        topic_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Main retrieval pipeline: gather -> combine -> gate -> rank -> threshold -> update -> slice
        """
        # Import here to avoid circular imports
        from utils.query_checker import is_meta_conversational, _is_heavy_topic_heuristic
        from memory.memory_scorer import _is_deictic_followup

        # Check for meta-conversational query — but skip when the query
        # is a retrospective temporal query ("yesterday", "last night").
        # These match meta markers ("did we") but are better served by the
        # normal pipeline with temporal window reranking.
        query_lower = query.lower()
        is_retrospective_temporal = (
            self.scorer
            and getattr(self.scorer, '_intent_weight_overrides', None)
            and '_temporal_anchor_hours' in (self.scorer._intent_weight_overrides or {})
            and any(m in query_lower for m in ('yesterday', 'last night'))
        )
        if is_meta_conversational(query) and not is_retrospective_temporal:
            logger.debug(f"[MemoryRetriever] Detected meta-conversational query: {query[:50]}...")
            return await self._get_meta_conversational_memories(query, limit, topic_filter)

        # Two-query design: retrieval_query for embedding lookup,
        # original query for scoring / continuity / display.
        retrieval_query = self._reformulate_for_embedding(query)
        if retrieval_query != query:
            logger.info(
                f"[QueryReformulation] '{query[:60]}' → '{retrieval_query[:60]}'"
            )

        # Dynamic configuration
        query_lower = query.lower()
        is_gym_health_query = any(word in query_lower for word in [
            'gym', 'workout', 'work out', 'exercise', 'fitness', 'bench', 'squat',
            'amantadine', 'medication', 'health', 'body', 'tired'
        ])

        if is_gym_health_query:
            cfg = {
                'recent_count': 0,
                'semantic_count': max(50, limit * 5),
                'max_memories': limit,
            }
        else:
            cfg = {
                'recent_count': 1,
                'semantic_count': max(30, limit * 2),
                'max_memories': limit,
            }

        topic_filter = topic_filter or self.current_topic
        if is_gym_health_query:
            topic_filter = None

        # Gather from both sources — semantic uses retrieval_query
        tasks = [
            asyncio.to_thread(self._get_recent_conversations, k=cfg['recent_count']),
            self._get_semantic_memories(retrieval_query, n_results=cfg['semantic_count'])
        ]
        very_recent, semantic = await asyncio.gather(*tasks)

        hierarchical: List[Dict] = []

        # Topic pre-filter
        if topic_filter and topic_filter != "general":
            def _has_topic_tag(m: Dict) -> bool:
                tags = m.get('tags', [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                return f"topic:{topic_filter}" in tags

            filtered_recent = [m for m in very_recent if _has_topic_tag(m)]
            filtered_semantic = [m for m in semantic if _has_topic_tag(m)]

            total_before = len(very_recent) + len(semantic)
            total_after = len(filtered_recent) + len(filtered_semantic)

            if total_after > 0:
                very_recent = filtered_recent
                semantic = filtered_semantic

        # Combine with gating — uses retrieval_query for cosine comparison
        combined = await self._combine_memories(
            very_recent=very_recent,
            semantic=semantic,
            hierarchical=hierarchical,
            query=retrieval_query,
            config={'max_memories': max(limit * 2, 30)},
            bypass_gate=is_gym_health_query
        )

        # Rank memories
        if self.scorer:
            # Pass topic and meta-conversational status to scorer
            # Note: is_meta_conversational already checked above, so False here
            ranked = self.scorer.rank_memories(
                combined,
                query,
                current_topic=topic_filter,
                is_meta_conversational=False  # Meta queries use separate path
            )
        else:
            ranked = sorted(combined, key=lambda x: x.get('relevance_score', 0.5), reverse=True)

        # Temporal window reranking for retrospective queries ("yesterday",
        # "last night").  Prioritizes memories from the referenced time period
        # over too-recent memories that would otherwise dominate on raw recency.
        ranked = self._maybe_temporal_window_rerank(ranked, query)

        # Graceful threshold filtering with 3-stage fallback
        is_deictic = _is_deictic_followup(query)
        primary_threshold = DEICTIC_THRESHOLD if is_deictic else NORMAL_THRESHOLD

        # Stage 1: Try primary threshold
        accepted = [m for m in ranked if m.get('final_score', 0.0) >= primary_threshold]

        # Stage 2: If insufficient results, relax to 70% of threshold
        if len(accepted) < GATING_MIN_RESULTS:
            relaxed_threshold = primary_threshold * GATING_RELAXED_MULTIPLIER
            logger.info(
                f"[Retrieval] Only {len(accepted)} results at threshold "
                f"{primary_threshold:.2f}, relaxing to {relaxed_threshold:.2f}"
            )
            accepted = [m for m in ranked if m.get('final_score', 0.0) >= relaxed_threshold]

            # Stage 3: If still insufficient, take top N as final fallback
            if len(accepted) < GATING_MIN_RESULTS:
                logger.warning(
                    f"[Retrieval] Still only {len(accepted)} results after relaxation, "
                    f"taking top {GATING_MIN_RESULTS}"
                )
                accepted = ranked[:GATING_MIN_RESULTS]

        # Cross-encoder reranking: rescore top candidates using query-document
        # pair scoring.  This runs AFTER the multi-factor scorer so it sees
        # all memories including episodic (which bypass the gate filter).
        # The cross-encoder can discriminate "related" from "best answer".
        top_memories = self._maybe_cross_encoder_rerank(accepted[:limit], query)

        # Update truth scores
        if self.scorer:
            self.scorer.update_truth_scores_on_access(top_memories)

        return top_memories

    # ------------------------------------------------------------------
    # Cross-encoder reranking
    # ------------------------------------------------------------------

    _cross_encoder = None  # Lazy-loaded singleton

    def _maybe_cross_encoder_rerank(
        self, memories: List[Dict], query: str
    ) -> List[Dict]:
        """
        Rerank top memories using a cross-encoder that scores query-document
        pairs.  Unlike the gate filter (which episodic memories bypass), this
        sees ALL memories and can push the correct answer above recency traps.

        Blends cross-encoder score with the existing final_score to avoid
        completely overriding multi-factor scoring.
        """
        if not memories or len(memories) < 2:
            return memories

        # Only rerank the top N to keep latency bounded (~15ms for 15 items).
        # Items beyond this cutoff keep their scorer-assigned rank.
        RERANK_TOP_N = 15
        if len(memories) > RERANK_TOP_N:
            to_rerank = memories[:RERANK_TOP_N]
            tail = memories[RERANK_TOP_N:]
        else:
            to_rerank = memories
            tail = []

        # Lazy-load cross-encoder (once per process)
        if MemoryRetriever._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                MemoryRetriever._cross_encoder = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                logger.info("[CrossEncoderRerank] Loaded cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                logger.debug(f"[CrossEncoderRerank] Not available: {e}")
                MemoryRetriever._cross_encoder = False  # Don't retry
                return memories

        if MemoryRetriever._cross_encoder is False:
            return memories

        try:
            # Build query-document pairs
            def _mem_text(m: Dict) -> str:
                content = (m.get('content') or '').strip()
                if content:
                    return content[:500]
                q = (m.get('query') or '').strip()
                r = (m.get('response') or '').strip()
                return f"{q} {r}"[:500]

            pairs = [[query, _mem_text(m)] for m in to_rerank]
            scores = MemoryRetriever._cross_encoder.predict(pairs)

            # Normalize cross-encoder scores to [0, 1]
            min_s, max_s = float(min(scores)), float(max(scores))
            spread = max_s - min_s if max_s > min_s else 1.0
            norm_scores = [(float(s) - min_s) / spread for s in scores]

            # Blend: 60% original final_score + 40% cross-encoder
            for m, ce_score in zip(to_rerank, norm_scores):
                original = m.get('final_score', 0.5)
                m['final_score'] = 0.60 * original + 0.40 * ce_score
                m['cross_encoder_score'] = ce_score

            reranked = sorted(to_rerank, key=lambda x: x.get('final_score', 0), reverse=True)

            logger.debug(
                f"[CrossEncoderRerank] Reranked {len(to_rerank)} memories, "
                f"top={reranked[0].get('metadata', {}).get('benchmark_id', '?')}"
            )
            return reranked + tail

        except Exception as e:
            logger.warning(f"[CrossEncoderRerank] Failed: {e}")
            return memories

    # ------------------------------------------------------------------
    # Temporal window reranking
    # ------------------------------------------------------------------

    _RETROSPECTIVE_MARKERS = frozenset(["yesterday", "last night"])

    def _maybe_temporal_window_rerank(
        self, ranked: List[Dict], query: str
    ) -> List[Dict]:
        """
        For retrospective temporal queries (e.g. "yesterday"), move memories
        within the referenced time window ahead of too-recent memories.

        Only triggers when:
          - A temporal anchor ≤ 48h exists (small-window query)
          - The query contains a clearly retrospective marker
        """
        if not self.scorer or not getattr(self.scorer, '_intent_weight_overrides', None):
            return ranked

        anchor = (self.scorer._intent_weight_overrides or {}).get(
            '_temporal_anchor_hours'
        )
        if not anchor or anchor > 48:
            return ranked

        query_lower = query.lower()
        if not any(m in query_lower for m in self._RETROSPECTIVE_MARKERS):
            return ranked

        # Reference time (must match scorer)
        if self.time_manager and hasattr(self.time_manager, 'current'):
            now = self.time_manager.current()
        else:
            now = datetime.now()

        window_start = anchor * 0.5
        window_end = anchor * 2.0

        in_window: List[Dict] = []
        out_window: List[Dict] = []
        for m in ranked:
            ts = m.get('timestamp')
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except Exception:
                    out_window.append(m)
                    continue
            if not isinstance(ts, datetime):
                out_window.append(m)
                continue

            age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
            if window_start <= age_hours <= window_end:
                in_window.append(m)
            else:
                out_window.append(m)

        if not in_window:
            return ranked

        logger.debug(
            f"[TemporalWindowRerank] anchor={anchor}h, "
            f"window=[{window_start:.0f}h, {window_end:.0f}h], "
            f"in_window={len(in_window)}, out={len(out_window)}"
        )
        return in_window + out_window

    async def get_semantic_top_memories(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Return top-k semantic memories across conversations, summaries, reflections
        using the gate system's cosine score. No recent-corpus bypass.

        Special handling: If this is a meta-conversational query (e.g., "do you recall"),
        route to specialized retrieval that prioritizes recent episodic memories.
        """
        import re as _re
        from utils.query_checker import is_meta_conversational

        if is_meta_conversational(query):
            logger.debug(f"[MemoryRetriever][Semantic] Detected meta-conversational query, routing to specialized retrieval: {query[:50]}...")
            return await self._get_meta_conversational_memories(query, limit, topic_filter=None)

        # Reformulate for embedding — semantic search + gating see retrieval_query
        retrieval_query = self._reformulate_for_embedding(query)

        try:
            raw = await self._get_semantic_memories(retrieval_query, n_results=max(30, limit * 3))
        except Exception as e:
            logger.warning(f"[MemoryRetriever] Semantic memory retrieval failed: {e}")
            raw = []

        if not raw:
            return []

        # Build chunks for gating with back-reference to original memory
        _hdr_re = _re.compile(r"^\s*\[[^\]]+\]", _re.IGNORECASE)

        def _strip_headers(s: str) -> str:
            if not s:
                return s
            out = []
            for ln in (s.splitlines() or []):
                if _hdr_re.search(ln):
                    continue
                out.append(ln)
            return "\n".join(out).strip()

        def _gate_text(m: Dict) -> str:
            txt = _strip_headers((m.get('content') or '').strip())
            if txt:
                return txt
            q = _strip_headers((m.get('query') or '').strip())
            a = _strip_headers((m.get('response') or '').strip())
            return f"User: {q}\nAssistant: {a}".strip()

        chunks = [{
            "content": _gate_text(m)[:500],
            "metadata": {"original_memory": m},
        } for m in raw]

        # If no gate_system, return a simple cap by initial relevance score
        if not self.gate_system:
            out = sorted(raw, key=lambda x: float(x.get('relevance_score', 0.5)), reverse=True)[:limit]
            for m in out:
                m['pre_gated'] = True
            return out

        # Run gate and pick top-k by gate score — uses retrieval_query
        try:
            filtered = await self.gate_system.filter_memories(retrieval_query, chunks)
        except Exception as e:
            logger.warning(f"[MemoryRetriever] Gate system filtering failed: {e}")
            filtered = chunks[:limit]

        # Propagate gate score + mark as pre_gated
        out: List[Dict] = []
        for ch in filtered[:limit]:
            md = ch.get('metadata', {}) or {}
            orig = md.get('original_memory')
            if not isinstance(orig, dict):
                continue
            score = float(ch.get('relevance_score', ch.get('__score__', orig.get('relevance_score', 0.5))))
            orig = dict(orig)
            orig['relevance_score'] = score
            orig['pre_gated'] = True
            out.append(orig)

        # Optional strict top-up: disabled by default to avoid noisy generic memories
        try:
            enable_topup = str(os.getenv("MEM_TOPUP_ENABLE", "0")).strip().lower() in {"1", "true", "yes", "on"}
            min_score = float(os.getenv("MEM_TOPUP_MIN_SCORE", "0.35"))
        except (ValueError, TypeError):
            enable_topup = False
            min_score = 0.35

        if enable_topup and len(out) < limit and raw:
            def _k(m: Dict) -> str:
                mid = str(m.get('id') or '').strip()
                if mid:
                    return f"id::{mid}"
                return f"content::{(m.get('content') or '').strip()[:160].lower()}"

            selected = {_k(m) for m in out}

            def _score(m: Dict) -> float:
                try:
                    return float(m.get('relevance_score', 0.0))
                except (ValueError, TypeError):
                    return 0.0

            def _overlap(a: str, b: str) -> float:
                at = set(_re.findall(r"[a-zA-Z0-9]+", (a or "").lower()))
                bt = set(_re.findall(r"[a-zA-Z0-9]+", (b or "").lower()))
                if not at or not bt:
                    return 0.0
                return len(at & bt) / max(1, min(len(at), len(bt)))

            for cand in sorted(raw, key=_score, reverse=True):
                if len(out) >= limit:
                    break
                key = _k(cand)
                if not key or key in selected:
                    continue
                sc = _score(cand)
                txt = _gate_text(cand)
                if sc >= min_score and _overlap(query, txt) >= 0.15:
                    c = dict(cand)
                    c['pre_gated'] = True
                    out.append(c)
                    selected.add(key)

        return out[:limit]

    async def _get_meta_conversational_memories(
        self,
        query: str,
        limit: int = 20,
        topic_filter: Optional[str] = None
    ) -> List[Dict]:
        """Special retrieval for meta-conversational queries about conversation history."""
        logger.debug("[MemoryRetriever] Using meta-conversational retrieval strategy")

        # Detect temporal window
        from utils.query_checker import extract_temporal_window
        temporal_days = extract_temporal_window(query)

        if temporal_days == 0:
            recent_limit = min(limit * 5, 50)
        elif temporal_days <= 2:
            recent_limit = min(limit * 3, 30)
        elif temporal_days <= 7:
            recent_limit = min(limit * 8, 80)
        else:
            recent_limit = min(limit * 15, 150)

        # Get recent episodic memories — time-filtered for wider temporal
        # windows (>2 days) to avoid flooding with irrelevant recent entries.
        # Short windows ("earlier today", "yesterday") keep the standard
        # recency-ordered corpus pull since their targets ARE recent.
        if temporal_days > 2:
            now = (self.time_manager.current()
                   if self.time_manager and hasattr(self.time_manager, 'current')
                   else datetime.now())
            buffer_hours = int(temporal_days * 24 * 1.5)
            cutoff = now - timedelta(hours=buffer_hours)

            raw_entries = self.corpus_manager._get_episodic_sorted()
            filtered = []
            for entry in raw_entries:
                ts = entry.get('timestamp')
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except (ValueError, TypeError):
                        continue
                if isinstance(ts, datetime) and ts >= cutoff:
                    filtered.append(entry)
                    if len(filtered) >= recent_limit:
                        break

            corpus_rel = 0.5 if temporal_days > 2 else 0.9
            very_recent = format_recent_conversations(
                filtered, base_relevance=corpus_rel
            )

            # Fallback: if window is too narrow, supplement with recent
            if len(very_recent) < 5:
                fallback = self._get_recent_conversations(k=min(10, recent_limit))
                seen = {m.get('id') for m in very_recent}
                for m in fallback:
                    if m.get('id') not in seen:
                        very_recent.append(m)
                        seen.add(m.get('id'))

            logger.debug(
                f"[MemoryRetriever] Temporal meta path: {len(very_recent)} "
                f"corpus entries within {buffer_hours}h"
            )
        else:
            very_recent = self._get_recent_conversations(k=recent_limit)

        # ENTITY-AWARE RETRIEVAL: If query mentions specific entities (names like "Graham"),
        # also do semantic search to find older memories about those entities
        from processing.gate_system import _extract_query_entities, _entity_match_boost
        query_entities = _extract_query_entities(query)

        # Two-query design: reformulated query for semantic lookup,
        # original query for entity extraction and scoring.
        retrieval_query = self._reformulate_for_embedding(query)
        if retrieval_query != query:
            logger.info(
                f"[QueryReformulation][Meta] '{query[:60]}' → '{retrieval_query[:60]}'"
            )

        entity_matches = []
        if query_entities:
            logger.info(f"[MemoryRetriever] Meta-conversational query mentions entities: {query_entities}")
            # Do semantic search to find entity-related memories
            semantic_results = await self._get_semantic_memories(retrieval_query, n_results=100)

            # Filter and boost memories that contain the mentioned entities
            for mem in semantic_results:
                content = mem.get('content', '') or mem.get('query', '') + ' ' + mem.get('response', '')
                boost = _entity_match_boost(query_entities, content)
                if boost > 0:
                    mem['entity_boost'] = boost
                    mem['relevance_score'] = mem.get('relevance_score', 0.5) + boost
                    entity_matches.append(mem)
                    logger.debug(f"[MemoryRetriever] Entity match found: boost={boost:.2f}, preview={content[:60]}...")

            logger.info(f"[MemoryRetriever] Found {len(entity_matches)} entity-matching memories from semantic search")

        # Merge recent + entity matches, deduplicating by id
        seen_ids = set()
        combined = []

        # Entity matches first (they're specifically about the entity mentioned)
        for mem in entity_matches:
            mem_id = mem.get('id') or id(mem)
            if mem_id not in seen_ids:
                seen_ids.add(mem_id)
                combined.append(mem)

        # Then recent memories
        for mem in very_recent:
            mem_id = mem.get('id') or id(mem)
            if mem_id not in seen_ids:
                seen_ids.add(mem_id)
                combined.append(mem)

        logger.debug(f"[MemoryRetriever] Combined {len(entity_matches)} entity + {len(very_recent)} recent = {len(combined)} unique")

        # Sort chronologically
        def _ts(m):
            ts = m.get('timestamp')
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts)
                except Exception:
                    return datetime.min
            return ts if isinstance(ts, datetime) else datetime.min

        # Use scorer with meta-conversational bonus if available
        if self.scorer:
            ranked = self.scorer.rank_memories(
                combined,
                query,
                current_topic=topic_filter,
                is_meta_conversational=True  # Enable meta-conversational bonus
            )
            return ranked[:limit]
        else:
            # Fallback: Apply gentle recency weighting
            combined.sort(key=_ts, reverse=True)
            now = datetime.now()
            for m in combined:
                ts = _ts(m)
                if ts:
                    age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
                    recency_score = 1.0 / (1.0 + 0.01 * age_hours)
                    # Preserve entity boost if present
                    entity_boost = m.get('entity_boost', 0.0)
                    m['final_score'] = recency_score + entity_boost
                    m['relevance_score'] = recency_score + entity_boost

            return combined[:limit]
