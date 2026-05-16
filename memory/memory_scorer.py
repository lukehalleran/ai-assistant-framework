# memory/memory_scorer.py
"""
Module Contract
- Purpose: Unified memory scoring and ranking with intent-driven weight overrides,
  graph-boosted proximity scoring, evidence-based truth scoring, and staleness penalties.
- Inputs:
  - MemoryScorer(time_manager, conversation_context)
  - rank_memories(memories, current_query, current_topic, is_meta_conversational, weight_overrides) -> List[Dict]
  - calculate_truth_score(query, response) -> float
  - calculate_importance_score(content) -> float
  - apply_temporal_decay(memories) -> List[Dict]
  - update_truth_scores_on_access(memories) -> None [no-op, retained for API compat]
- Outputs:
  - Ranked memory list with final_score and optional debug dict per item
  - Individual truth/importance scores for new memories
- Key behaviors:
  - 12-step scoring pipeline: base relevance + collection boost, recency decay (active-day or hourly),
    evidence-based truth (TruthScorer), importance, continuity (token overlap + last-10m),
    structural alignment (numeric/op density), topic match, analogy penalty, anchor bonus
    (deictic follow-ups), meta-conversational bonus, graph proximity bonus, staleness penalty
  - Intent-driven weight overrides: instance attribute _intent_weight_overrides set/cleared by
    PromptBuilder to thread IntentClassifier overrides through deep call chains
  - Graph-boosted scoring: _graph_memory + _entity_resolver set by PromptBuilder; adds 0.05 per
    graph-connected entity mention (capped at GRAPH_SCORING_BOOST_CAP=0.15)
  - Temporal anchor: _temporal_anchor_hours weight key reshapes recency decay for TEMPORAL_RECALL
  - Staleness penalty: staleness_ratio from ClaimIndex, 2x multiplier at >=0.8, reflections at 60%
  - Size penalty for large docs lacking keyword relevance (>10KB threshold)
  - Deictic drift guardrail: 0.85x multiplier when continuity + anchor are both low
- Dependencies:
  - config.app_config (SCORE_WEIGHTS, RECENCY_DECAY_RATE, COLLECTION_BOOSTS, STALENESS_*, GRAPH_*)
  - memory.truth_scorer.TruthScorer (evidence-based truth at read time)
  - memory.graph_utils (extract_graph_entities, get_related_display_names) [optional]
  - utils.time_manager (active-day decay) [optional]
- Side effects:
  - Mutates memory dicts in-place (sets final_score, debug keys)
  - Debug dict only populated when logger level is DEBUG
"""

import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set

from utils.logging_utils import get_logger
from config.app_config import (
    RECENCY_DECAY_RATE,
    COLLECTION_BOOSTS,
    DEICTIC_ANCHOR_PENALTY,
    DEICTIC_CONTINUITY_MIN,
    SCORE_WEIGHTS,
    STALENESS_ENABLED,
    STALENESS_WEIGHT,
    STALENESS_MAX_PENALTY,
    STALENESS_STEEP_THRESHOLD,
    STALENESS_STEEP_MULTIPLIER,
    STALENESS_REFLECTION_WEIGHT_FACTOR,
)
from memory.truth_scorer import TruthScorer

logger = get_logger("memory_scorer")

# ---------------------------
# Size Penalty Configuration
# ---------------------------

LARGE_DOC_SIZE_THRESHOLD = 10000  # 10KB in bytes
LARGE_DOC_KEYWORD_THRESHOLD = 0.3  # Minimum keyword_score to avoid penalty
LARGE_DOC_BASE_PENALTY = -0.25    # Base penalty for large irrelevant docs

# ---------------------------
# Heuristics & token helpers
# ---------------------------

DEICTIC_HINTS = ("explain", "that", "it", "this", "again", "another way", "different way")

STOPWORDS = set("""
the a an to of in on for with and or if is are was were be been being by at from as this that it its
""".split())


def _is_deictic_followup(q: str) -> bool:
    """Check if query is a deictic follow-up (refers to previous context)."""
    ql = (q or "").lower()
    return any(h in ql for h in DEICTIC_HINTS)


def _stem(word: str) -> str:
    """Minimal suffix strip for overlap matching.

    Handles the common mismatches: anxious/anxiety, deployed/deployment,
    features/feature, etc.  Deliberately simple — full stemming (Porter)
    is too aggressive for code symbols like ``asyncio``.
    """
    if len(word) <= 4:
        return word
    for suffix in ("ment", "tion", "sion", "ness", "ious", "eous",
                   "ity", "ies", "ing", "ous", "ive", "ful",
                   "ure", "ated", "ting", "ted", "ed", "ly", "er", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def _salient_tokens(text: str, k: int = 12) -> Set[str]:
    """Extract most salient tokens from text (stemmed for overlap matching)."""
    toks = re.findall(r"[a-zA-Z0-9\+\-\*/=\^()]+", (text or "").lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    freq: Dict[str, int] = {}
    for t in toks:
        stemmed = _stem(t)
        freq[stemmed] = freq.get(stemmed, 0) + 1
    return {t for t, _ in sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))[:k]}


def _num_op_density(text: str) -> float:
    """Calculate numeric/operator density of text."""
    if not text:
        return 0.0
    nums = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    ops = len(re.findall(r"[\+\-\*/=\^]", text))
    toks = max(1, len(re.findall(r"\w+", text)))
    return (nums + ops) / toks


def _analogy_markers(text: str) -> int:
    """Count analogy markers in text."""
    if not text:
        return 0
    t = text.lower()
    markers = ["it's like", "its like", "imagine", "picture this", "as if", "like when", "metaphor", "analogy"]
    return sum(1 for m in markers if m in t)


def _build_anchor_tokens(conv: list, maxlen: int = 20) -> Set[str]:
    """Pull anchor tokens from the last exchange with better math handling."""
    anchors: Set[str] = set()
    if conv:
        last = conv[-1]
        blob = f"{last.get('query','')} {last.get('response','')}"
        math_patterns = [
            r"[a-zA-Z]\([a-zA-Z]\)",       # f(x)
            r"\d+[a-zA-Z]\^\d+",           # 7x^4
            r"\d+[a-zA-Z]\d*",             # 7x4, 9x
            r"[a-zA-Z]′\([a-zA-Z]\)",      # f'(x)
            r"\b\d+(?:\.\d+)?\b",          # numbers
            r"derivative|integral|function|equation|cdf|pdf|variance|expectation",
        ]
        for pattern in math_patterns:
            matches = re.findall(pattern, blob.lower())
            anchors.update(matches[:5])
        anchors |= _salient_tokens(blob, k=8)
    if len(anchors) > maxlen:
        anchors = set(list(anchors)[:maxlen])
    return anchors


class MemoryScorer:
    """
    Unified memory scoring and ranking.

    Implements MemoryScorerProtocol contract.
    """

    def __init__(self, time_manager=None, conversation_context=None):
        """
        Initialize MemoryScorer.

        Args:
            time_manager: Optional TimeManager for active day decay.
            conversation_context: Optional list/deque of recent conversation turns.
        """
        self.time_manager = time_manager
        self.conversation_context = conversation_context or []
        self.access_history: Dict[str, int] = {}
        # Intent-driven weight overrides — set by PromptBuilder before retrieval,
        # cleared after. Used as fallback when rank_memories() is called without
        # an explicit weight_overrides parameter (deep in the retrieval chain).
        self._intent_weight_overrides: Optional[Dict[str, float]] = None
        # Graph references — set by PromptBuilder before retrieval, cleared after.
        # Used for graph-boosted scoring (bonus for memories mentioning
        # entities connected to query entities in the knowledge graph).
        self._graph_memory = None
        self._entity_resolver = None

    def calculate_truth_score(self, query: str, response: str) -> float:
        """Calculate truth score based on response/continuity characteristics."""
        score = 0.5
        if len(response or "") > 200:
            score += 0.1
        if '?' in (query or ""):
            score += 0.1
        confirms = ('yes', 'correct', 'exactly', 'right', 'understood', 'makes sense', 'good point')
        if any(c in (response or "").lower() for c in confirms):
            score += 0.2
        # Continuity with previous response
        if self.conversation_context:
            last = self.conversation_context[-1]
            last_tokens = set((last.get('response', '') or '').lower().split()[:10])
            if any(t in last_tokens for t in (query or '').lower().split()):
                score += 0.15
        return min(score, 1.0)

    def calculate_importance_score(self, content: str) -> float:
        """Estimate importance for retention prioritization."""
        score = 0.5
        text = content or ""
        if len(text) > 200:
            score += 0.1
        if '?' in text:
            score += 0.1
        important_keywords = ['important', 'remember', 'note', 'key', 'critical', 'essential', 'todo', 'directive']
        if any(kw in text.lower() for kw in important_keywords):
            score += 0.2
        return min(score, 1.0)

    def update_truth_scores_on_access(self, memories: List[Dict]) -> None:
        """No-op: access-based truth boosting removed to prevent echo chamber.

        Truth scores are now managed by TruthScorer via evidence-based
        signals (confirmations, corrections, contradictions) and time decay.
        This method is retained for API compatibility only.
        """
        pass

    def _calculate_topic_match(self, memory: Dict, current_topic: Optional[str]) -> float:
        """
        Calculate topic alignment score (0.0-1.0).

        Returns:
            1.0 - Exact topic match
            0.5 - No topic info available (neutral)
            0.2 - Different topic (penalty)
        """
        # Extract topics from memory metadata
        memory_topics = (memory.get('metadata', {}) or {}).get('topics', [])

        # Also check tags field for topic tags
        tags = memory.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]

        # Extract topic: prefixed tags
        topic_tags = [t.replace('topic:', '') for t in tags if t.startswith('topic:')]
        memory_topics.extend(topic_tags)

        # Neutral if no topic info on either side
        if not memory_topics or not current_topic:
            return 0.5

        # Case-insensitive matching
        current_lower = current_topic.lower()
        memory_topics_lower = [t.lower() for t in memory_topics]

        if current_lower in memory_topics_lower:
            return 1.0  # Exact match

        return 0.2  # Different topic

    def _calculate_size_penalty(self, memory: Dict) -> float:
        """
        Penalize large documents that lack keyword relevance.

        Penalty scales with document size:
        - Under 10KB: no penalty (return 0.0)
        - Over 10KB with keyword_score > 0.3: no penalty (return 0.0)
        - Over 10KB with keyword_score <= 0.3: scaled penalty
          Example: 20KB doc = -0.25 × (20/10) = -0.50
                   95KB doc = -0.25 × (95/10) = -2.375 (capped at -1.0)

        Returns:
            0.0 or negative penalty value (capped at -1.0)
        """
        # Get content from either field
        content = memory.get('content', '') or memory.get('response', '')

        # Calculate size in bytes
        size_bytes = len(content.encode('utf-8')) if content else 0

        # Under threshold - no penalty
        if size_bytes < LARGE_DOC_SIZE_THRESHOLD:
            return 0.0

        # Large doc - check keyword relevance
        keyword_score = memory.get('keyword_score', 0.0)
        if keyword_score > LARGE_DOC_KEYWORD_THRESHOLD:
            return 0.0  # Has keyword relevance, no penalty

        # Calculate scaled penalty
        size_multiplier = size_bytes / LARGE_DOC_SIZE_THRESHOLD
        penalty = LARGE_DOC_BASE_PENALTY * size_multiplier

        # Cap penalty at -1.0 (prevents extreme penalties for very large docs)
        return max(-1.0, penalty)

    def rank_memories(
        self,
        memories: List[Dict],
        current_query: str,
        current_topic: Optional[str] = None,
        is_meta_conversational: bool = False,
        weight_overrides: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        Score each memory using:
          - base relevance (+ collection/source boost)
          - recency (configurable decay)
          - truth / importance
          - continuity (token overlap + last-10m)
          - structure alignment (numeric/op density)
          - topic match (alignment with current topic)
          - analogy penalty for mathy queries
          - anchor bonus (esp. for deictic follow-ups)
          - meta-conversational bonus (for history queries)
          - size penalty (for large irrelevant docs)
          - optional acceptance threshold (applied by caller after scoring)

        Args:
            weight_overrides: Optional dict of {weight_name: value} to override
                global SCORE_WEIGHTS for this call. Used by intent classifier
                to tune scoring per query intent.
        """
        if not memories:
            return []

        # Merge weight overrides (intent-driven) on top of global defaults.
        # Explicit parameter wins; instance attribute is the fallback
        # (set by PromptBuilder to thread overrides through deep call chains).
        effective_overrides = weight_overrides or self._intent_weight_overrides
        weights = dict(SCORE_WEIGHTS)
        if effective_overrides:
            weights.update(effective_overrides)

        # Pop temporal anchor (not a real scoring weight — used to reshape
        # the recency decay curve for TEMPORAL_RECALL queries).
        temporal_anchor = weights.pop('_temporal_anchor_hours', None)

        # Pop timeline mode flag — gives summaries/reflections a bonus
        # for progression queries like "how long" / "over time".
        timeline_mode = weights.pop('_timeline_mode', False)

        # Use time_manager's reference time when available (testability +
        # consistency between temporal-anchor and active-day-decay paths).
        if (self.time_manager is not None
                and hasattr(self.time_manager, 'current')):
            now = self.time_manager.current()
        else:
            now = datetime.now()
        last_10m = now - timedelta(minutes=10)

        is_deictic = _is_deictic_followup(current_query)
        anchors = _build_anchor_tokens(list(self.conversation_context))
        cq_salient = _salient_tokens(current_query, k=12)
        cq_density = _num_op_density(current_query)

        # Graph-boosted scoring: build set of related entity display names
        graph_related_names: Set[str] = set()
        graph_boost_enabled = False
        graph_boost_cap = 0.15
        try:
            from config.app_config import (
                KNOWLEDGE_GRAPH_ENABLED,
                GRAPH_SCORING_BOOST_ENABLED,
                GRAPH_SCORING_BOOST_CAP,
            )
            gm = self._graph_memory
            er = self._entity_resolver
            if (KNOWLEDGE_GRAPH_ENABLED and GRAPH_SCORING_BOOST_ENABLED
                    and gm and er and gm.node_count() > 0):
                from memory.graph_utils import extract_graph_entities, get_related_display_names
                query_entities = extract_graph_entities(current_query, er)
                if query_entities:
                    # depth=1 intentional: depth=2 in star topology connects
                    # everything through "user", making boost indiscriminate.
                    # Scoring boost activates when graph has lateral edges.
                    graph_related_names = get_related_display_names(
                        query_entities, gm, depth=1, skip_ids={"user"},
                    )
                    graph_boost_enabled = bool(graph_related_names)
                    graph_boost_cap = GRAPH_SCORING_BOOST_CAP
                    if graph_related_names:
                        logger.debug(
                            f"[Ranker] Graph boost: {len(graph_related_names)} related names "
                            f"from entities {query_entities}"
                        )
        except Exception as e:
            logger.debug(f"[Ranker] Graph boost setup failed: {e}")

        for m in memories:
            # 1) base relevance with collection/source boost
            rel = float(m.get('relevance_score', 0.5))
            collection_key = m.get('collection', m.get('source', ''))
            if collection_key in COLLECTION_BOOSTS:
                rel += COLLECTION_BOOSTS[collection_key]

            # Cap inflated corpus relevance for large temporal windows.
            # Corpus entries get a flat 0.9 regardless of query match,
            # drowning out semantically-scored ChromaDB results for
            # "last week" / "last month" queries.  Short windows
            # ("earlier today") benefit from the corpus recency bias.
            if temporal_anchor and temporal_anchor > 48 and m.get('source') == 'corpus':
                rel = min(rel, 0.5)

            # 2) recency with decay (using active days)
            ts = m.get('timestamp')
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except Exception:
                    ts = now
            elif not isinstance(ts, datetime):
                ts = now

            age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)

            if temporal_anchor and temporal_anchor > 0:
                # Two-regime temporal decay based on anchor size.
                #
                # Small anchor (<=48h, "today"/"yesterday"):
                #   In-window: flat plateau (1.0→0.85).
                #   Grace zone (up to +50% past anchor): gentle taper.
                #   Past grace: steep dropoff.
                #
                # Large anchor (>48h, "last week"/"last month"):
                #   Sqrt ramp from floor to 1.0 (faster rise, peaks at
                #   anchor).  Past-anchor: standard decay.
                if temporal_anchor <= 48:
                    grace_limit = temporal_anchor * 1.5
                    if age_hours <= temporal_anchor:
                        recency = 1.0 - (age_hours / temporal_anchor) * 0.15
                    elif age_hours <= grace_limit:
                        grace_frac = (age_hours - temporal_anchor) / (grace_limit - temporal_anchor)
                        recency = 0.85 - grace_frac * 0.15
                    else:
                        hours_past = age_hours - grace_limit
                        recency = 0.70 / (1.0 + RECENCY_DECAY_RATE * hours_past)
                else:
                    floor = max(0.60, 1.0 - (temporal_anchor / 500.0))
                    if age_hours <= temporal_anchor:
                        # Sqrt ramp: rises faster for early-window memories
                        frac = age_hours / temporal_anchor
                        recency = floor + (1.0 - floor) * (frac ** 0.5)
                    else:
                        hours_past = age_hours - temporal_anchor
                        recency = 1.0 / (1.0 + RECENCY_DECAY_RATE * hours_past)
            elif (self.time_manager is not None and
                  hasattr(self.time_manager, 'calculate_active_day_decay')):
                recency = self.time_manager.calculate_active_day_decay(ts, RECENCY_DECAY_RATE)
            else:
                # Fallback to original hourly decay
                recency = 1.0 / (1.0 + RECENCY_DECAY_RATE * age_hours)

            # 3) truth (evidence-based with time decay)
            md = m.get('metadata', {}) or {}
            truth = TruthScorer.compute_effective_truth(md)

            # 4) importance
            importance = float(m.get('importance_score', md.get('importance_score', 0.5)))

            # 5) continuity (overlap + recency)
            # Include content for non Q/A memories (summaries/reflections)
            blob = (m.get('query', '') + ' ' + m.get('response', '') + ' ' + m.get('content', '')).lower()
            m_raw_toks = re.findall(r"[a-zA-Z0-9\+\-\*/=\^()]+", blob)
            m_toks = {_stem(t) for t in m_raw_toks if len(t) > 1}
            continuity = 0.0
            if ts >= last_10m:
                continuity += 0.1
            if cq_salient:
                overlap = len(cq_salient & m_toks) / max(1, len(cq_salient))
                continuity += 0.3 * overlap

            # 5b) tag-keyword bonus: if query tokens match memory tags
            tags_raw = m.get('tags', (md.get('tags', '')))
            if tags_raw and cq_salient:
                if isinstance(tags_raw, str):
                    tag_set = {_stem(t.strip().lower()) for t in tags_raw.split(',') if t.strip()}
                else:
                    tag_set = {_stem(t.lower()) for t in tags_raw}
                tag_hits = len(cq_salient & tag_set)
                if tag_hits:
                    continuity += 0.15 * min(tag_hits, 3) / 3.0

            # 6) structural alignment
            m_density = _num_op_density(blob)
            density_alignment = 1.0 - min(1.0, abs(cq_density - m_density) * 3.0)
            structure = 0.15 * density_alignment

            # 7) penalties/bonuses
            penalty = 0.0
            if cq_density > 0.08 and _analogy_markers(blob) > 0 and "analogy" not in current_query.lower():
                penalty -= 0.1

            # NEW: Size penalty for large documents
            size_penalty = self._calculate_size_penalty(m)
            penalty += size_penalty

            anchor_bonus = 0.0
            if anchors:
                anchor_overlap = len(anchors & m_toks) / max(1, len(anchors))
                if is_deictic:
                    if anchor_overlap < 0.05:
                        penalty -= DEICTIC_ANCHOR_PENALTY
                    else:
                        anchor_bonus += 0.2 * anchor_overlap
                else:
                    anchor_bonus += 0.1 * anchor_overlap

            # 8) tone adjustment
            if any(t in blob for t in ("idiot", "stupid", "dumb", "toddler")):
                truth = max(0.0, truth - 0.2)

            # 9) topic match score
            topic_match = self._calculate_topic_match(m, current_topic)

            # 10) meta-conversational bonus
            meta_bonus = 0.0
            if is_meta_conversational:
                # Boost episodic memories for meta queries about conversation history
                mem_type = m.get('memory_type', '')
                collection = m.get('collection', '')
                if mem_type == 'EPISODIC' or collection == 'episodic':
                    meta_bonus = 0.15
                # Also boost summaries/reflections that capture conversation patterns
                elif mem_type == 'SUMMARY' or collection == 'summaries':
                    meta_bonus = 0.10
                elif mem_type == 'META' or collection == 'meta':
                    meta_bonus = 0.12

            # 11) graph proximity bonus
            graph_bonus = 0.0
            if graph_boost_enabled and graph_related_names:
                blob_lower = blob  # already lowered above
                matches = sum(
                    1 for name in graph_related_names
                    if name.lower() in blob_lower
                )
                if matches > 0:
                    graph_bonus = min(0.05 * matches, graph_boost_cap)

            # 12) staleness penalty (summaries/reflections with outdated claims)
            staleness_penalty = 0.0
            if STALENESS_ENABLED:
                staleness_ratio = float(
                    md.get('staleness_ratio', 0.0)
                    or m.get('staleness_ratio', 0.0)
                    or 0.0
                )
                if staleness_ratio > 0:
                    base_penalty = staleness_ratio * STALENESS_WEIGHT
                    if staleness_ratio >= STALENESS_STEEP_THRESHOLD:
                        base_penalty *= STALENESS_STEEP_MULTIPLIER
                    # Reduce penalty for reflections (behavioral patterns are more durable)
                    collection = m.get('collection', '')
                    if collection == 'reflections':
                        base_penalty *= STALENESS_REFLECTION_WEIGHT_FACTOR
                    staleness_penalty = -min(base_penalty, STALENESS_MAX_PENALTY)

            # Timeline bonus: summaries/reflections are more useful for
            # progression queries ("how long", "over time") because they
            # aggregate across multiple interactions.
            timeline_bonus = 0.0
            if timeline_mode:
                src = m.get('source', m.get('collection', ''))
                if src in ('summaries', 'reflections'):
                    timeline_bonus = 0.15

            m['final_score'] = (
                weights.get('relevance', 0.35) * rel +
                weights.get('recency', 0.25) * recency +
                weights.get('truth', 0.20) * truth +
                weights.get('importance', 0.05) * importance +
                weights.get('continuity', 0.10) * continuity +
                weights.get('topic_match', 0.00) * topic_match +
                structure +
                anchor_bonus +
                meta_bonus +
                graph_bonus +
                staleness_penalty +
                timeline_bonus +
                penalty
            )

            if logger.isEnabledFor(logging.DEBUG):
                m['debug'] = {
                    'rel': rel, 'recency': recency, 'truth': truth,
                    'importance': importance, 'continuity': continuity,
                    'topic_match': topic_match,
                    'structure': structure, 'anchor_bonus': anchor_bonus,
                    'meta_bonus': meta_bonus,
                    'graph_bonus': graph_bonus,
                    'staleness_penalty': staleness_penalty,
                    'size_penalty': size_penalty,
                    'penalty': penalty,
                }

            # extra guardrail for deictic drift
            if is_deictic and continuity < DEICTIC_CONTINUITY_MIN and anchor_bonus < 0.04:
                m['final_score'] *= 0.85

        memories.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)

        if logger.isEnabledFor(logging.DEBUG) and memories:
            logger.debug("\n[Ranker] Top 5 memories:")
            for i, mm in enumerate(memories[:5], 1):
                dbg = mm.get('debug', {})
                logger.debug(
                    f"  #{i}: score={mm.get('final_score', 0):.3f} "
                    f"(rel={dbg.get('rel', 0):.2f}, rec={dbg.get('recency', 0):.2f}, "
                    f"truth={dbg.get('truth', 0):.2f}, imp={dbg.get('importance', 0):.2f}, "
                    f"cont={dbg.get('continuity', 0):.2f}, topic={dbg.get('topic_match', 0):.2f}, "
                    f"struct={dbg.get('structure', 0):.2f}, anchor={dbg.get('anchor_bonus', 0):.2f}, "
                    f"meta={dbg.get('meta_bonus', 0):.2f}, graph={dbg.get('graph_bonus', 0):.2f}, "
                    f"stale={dbg.get('staleness_penalty', 0):.2f}, pen={dbg.get('penalty', 0):.2f}) "
                    f"Q: {mm.get('query', '')[:48]!r}"
                )

        return memories

    def apply_temporal_decay(self, memories: List[Dict]) -> List[Dict]:
        """Apply temporal decay to memory scores (simplified version)."""
        now = datetime.now()

        for mem_dict in memories:
            # Handle flat dictionary structure (both reflections and conversations)
            timestamp = mem_dict.get('timestamp') or mem_dict.get('metadata', {}).get('timestamp')
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except (ValueError, TypeError):
                    # If parsing fails, skip this memory
                    continue

            # Get decay rate with fallback
            decay_rate = mem_dict.get('metadata', {}).get('decay_rate', 0.01)

            # Get importance score with fallback
            importance_score = mem_dict.get('importance_score', 0.5)

            # Get truth score with fallback
            truth_score = mem_dict.get('truth_score', mem_dict.get('metadata', {}).get('truth_score', 0.5))

            # Use active day decay if time_manager supports it, otherwise fallback to hourly decay
            if (self.time_manager is not None and
                hasattr(self.time_manager, 'calculate_active_day_decay')):
                decay_factor = self.time_manager.calculate_active_day_decay(timestamp, decay_rate)
            else:
                # Fallback to original hourly decay (less aggressive)
                age_hours = (now - timestamp).total_seconds() / 3600.0
                decay_factor = 1.0 / (1.0 + decay_rate * (age_hours/168.0))  # Weekly instead of daily decay

            # Calculate final score
            mem_dict['final_score'] = (
                mem_dict.get('relevance_score', 0.0) *
                max(0.1, importance_score) *
                max(0.1, decay_factor) *
                (0.75 + 0.5*truth_score)
            )

        return memories
