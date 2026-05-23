"""
memory_scorer.py - Standalone memory/document scoring and ranking engine.

Module Contract
- Purpose: Score and rank memory documents (conversations, facts, summaries,
  reflections, etc.) using a multi-signal pipeline: relevance, recency decay,
  evidence-based truth scoring, importance, token continuity, structural
  alignment, topic matching, staleness penalties, and optional knowledge-graph
  proximity boosting. Single-file, drop-in, zero required external dependencies.
- Inputs:
  - MemoryScorer(time_fn, conversation_context, config): constructor
  - rank_memories(memories, query, topic, is_meta, weight_overrides) -> list[dict]
  - TruthScorer.compute_effective_truth(metadata) -> float
  - TruthScorer.apply_confirmation(score) -> float
  - TruthScorer.apply_correction(score) -> float
  - TruthScorer.apply_contradiction(score) -> float
  - TruthScorer.apply_time_decay(score, last_confirmed_at) -> float
  - TruthScorer.calculate_initial_score(source) -> float
  - calculate_truth_score(query, response) -> float
  - calculate_importance_score(content) -> float
  - apply_temporal_decay(memories) -> list[dict]
- Outputs:
  - Ranked memory list with final_score and optional debug dict per item
  - Individual truth/importance/decay scores
- Key behaviors:
  - 12-step scoring pipeline: base relevance + collection boost, recency decay
    (temporal-anchor or hourly fallback), evidence-based truth (TruthScorer),
    importance, continuity (stemmed token overlap + recency window), structural
    alignment (numeric/operator density), topic match, analogy penalty, anchor
    bonus (deictic follow-ups), meta-conversational bonus, graph proximity
    bonus (optional), staleness penalty
  - Two-regime temporal decay: small-anchor (<=48h) uses plateau+grace+dropoff;
    large-anchor (>48h) uses sqrt ramp
  - Intent-driven weight overrides via weight_overrides parameter or
    instance-level _intent_weight_overrides attribute
  - Graph-boosted scoring: optional, activated by setting graph_scorer
  - TruthScorer is stateless — all state lives in document metadata
  - Truth decay is read-only (computed at retrieval, not written back)
  - Confirmation resets the decay clock; corrections apply sharp penalty
  - Debug dict only populated when debug=True
- Dependencies: Python 3.9+ standard library only. No external packages required.

Usage:
    from memory_scorer import MemoryScorer, TruthScorer, ScorerConfig

    # 1. Basic usage with defaults
    scorer = MemoryScorer()
    ranked = scorer.rank_memories(memories, query="what is my cat's name")

    # 2. Custom config
    config = ScorerConfig(
        score_weights={"relevance": 0.40, "recency": 0.30, "truth": 0.15,
                       "importance": 0.05, "continuity": 0.10},
        recency_decay_rate=0.08,
        collection_boosts={"facts": 0.20, "summaries": 0.10},
    )
    scorer = MemoryScorer(config=config)
    ranked = scorer.rank_memories(memories, query="tell me about Flapjack")

    # 3. With conversation context (for continuity scoring)
    context = [{"query": "how is my cat?", "response": "Flapjack is doing well!"}]
    scorer = MemoryScorer(conversation_context=context)

    # 4. With intent-driven weight overrides
    ranked = scorer.rank_memories(
        memories, query="what happened last tuesday",
        weight_overrides={"recency": 0.40, "relevance": 0.25,
                          "_temporal_anchor_hours": 168},
    )

    # 5. With graph-boosted scoring (optional)
    scorer.graph_scorer = MyGraphScorer(graph, resolver)
    ranked = scorer.rank_memories(memories, query="my brother's dog")

    # 6. Truth scoring (standalone)
    truth = TruthScorer.compute_effective_truth({"truth_score": 0.8,
        "last_confirmed_at": "2025-01-15T10:00:00"})
    initial = TruthScorer.calculate_initial_score("user_stated")  # 0.8
    boosted = TruthScorer.apply_confirmation(0.7)                 # 0.85

    # 7. CLI usage
    #   python memory_scorer.py score '{"relevance_score": 0.8, "timestamp": "2025-06-01T12:00:00"}' --query "test"
    #   python memory_scorer.py truth '{"truth_score": 0.75, "last_confirmed_at": "2025-05-01T00:00:00"}'
    #   python memory_scorer.py demo

License: MIT
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

__all__ = [
    "MemoryScorer",
    "TruthScorer",
    "ScorerConfig",
    "GraphScorerProtocol",
]

logger = logging.getLogger("memory_scorer")


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ScorerConfig:
    """All tunable knobs for the scoring pipeline.

    Every field has a sensible default.  Override only what you need.
    """

    # --- Scoring weights (must sum to ~1.0 for interpretable scores) ---
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "relevance": 0.35,
        "recency": 0.25,
        "truth": 0.20,
        "importance": 0.05,
        "continuity": 0.10,
        "structure": 0.05,
    })

    # --- Recency decay ---
    recency_decay_rate: float = 0.05

    # --- Collection boosts (added to base relevance) ---
    collection_boosts: Dict[str, float] = field(default_factory=lambda: {
        "facts": 0.15,
        "summaries": 0.10,
        "conversations": 0.0,
        "semantic": 0.05,
        "wiki": 0.05,
        "daemon_self_notes": -0.05,
    })

    # --- Deictic follow-up handling ---
    deictic_anchor_penalty: float = 0.1
    deictic_continuity_min: float = 0.12

    # --- Truth scorer ---
    truth_enabled: bool = True
    truth_initial_score: float = 0.7
    truth_confirmed_boost: float = 0.15
    truth_correction_penalty: float = 0.25
    truth_contradiction_penalty: float = 0.15
    truth_decay_rate_per_week: float = 0.02
    truth_decay_floor: float = 0.3
    truth_source_scores: Dict[str, float] = field(default_factory=lambda: {
        "user_stated": 0.8,
        "corrected": 0.85,
        "llm_extracted": 0.7,
        "inferred": 0.5,
    })

    # --- Staleness ---
    staleness_enabled: bool = True
    staleness_weight: float = 0.15
    staleness_max_penalty: float = 0.4
    staleness_steep_threshold: float = 0.8
    staleness_steep_multiplier: float = 2.0
    staleness_reflection_weight_factor: float = 0.6

    # --- Size penalty ---
    large_doc_size_threshold: int = 10_000  # bytes
    large_doc_keyword_threshold: float = 0.3
    large_doc_base_penalty: float = -0.25

    # --- Debug ---
    debug: bool = False


# ============================================================================
# Graph Scorer Protocol (optional integration point)
# ============================================================================


@runtime_checkable
class GraphScorerProtocol(Protocol):
    """Optional protocol for knowledge-graph proximity scoring.

    Implement this to plug in your own graph backend.  The scorer calls
    ``get_related_names(query)`` once per ``rank_memories()`` invocation,
    then checks each memory's text for mentions of related entity names.
    """

    def get_related_names(self, query: str) -> Set[str]:
        """Return display names of entities related to the query.

        Args:
            query: The user's current query string.

        Returns:
            Set of lowercase display names that are graph-neighbors of
            entities mentioned in the query.
        """
        ...  # pragma: no cover


# ============================================================================
# Truth Scorer
# ============================================================================


class TruthScorer:
    """Stateless evidence-based truth scoring engine.

    Facts start at a source-dependent initial score, gain truth through user
    confirmations, lose truth through corrections/contradictions, and decay
    toward a floor when unconfirmed.  All state lives in document metadata —
    every method is a pure function.

    The config parameter is optional; when omitted, uses global defaults.
    """

    def __init__(self, config: Optional[ScorerConfig] = None) -> None:
        c = config or ScorerConfig()
        self._enabled = c.truth_enabled
        self._initial_score = c.truth_initial_score
        self._confirmed_boost = c.truth_confirmed_boost
        self._correction_penalty = c.truth_correction_penalty
        self._contradiction_penalty = c.truth_contradiction_penalty
        self._decay_rate = c.truth_decay_rate_per_week
        self._decay_floor = c.truth_decay_floor
        self._source_scores = dict(c.truth_source_scores)

    # ------------------------------------------------------------------
    # Initial scoring
    # ------------------------------------------------------------------

    def calculate_initial_score(self, source: str = "llm_extracted") -> float:
        """Return the initial truth score for a given fact source.

        Args:
            source: One of "user_stated", "corrected", "llm_extracted",
                    "inferred".

        Returns:
            Float initial score (falls back to truth_initial_score).
        """
        return float(self._source_scores.get(source, self._initial_score))

    # ------------------------------------------------------------------
    # Score adjustments
    # ------------------------------------------------------------------

    def apply_confirmation(self, current_score: float) -> float:
        """Boost truth when the user re-states or confirms a fact."""
        return min(1.0, current_score + self._confirmed_boost)

    def apply_correction(self, current_score: float) -> float:
        """Penalize truth when the user explicitly corrects a fact."""
        return max(0.0, current_score - self._correction_penalty)

    def apply_contradiction(self, current_score: float) -> float:
        """Mild penalty when a cross-collection contradiction is detected."""
        return max(0.0, current_score - self._contradiction_penalty)

    # ------------------------------------------------------------------
    # Time decay
    # ------------------------------------------------------------------

    def apply_time_decay(
        self,
        current_score: float,
        last_confirmed_at: Optional[datetime] = None,
        *,
        now: Optional[datetime] = None,
    ) -> float:
        """Decay truth toward the floor based on time since last confirmation.

        The decay is linear per week::

            decayed = current - (weeks_since_confirmed * decay_rate)
            clamped to [decay_floor, current]

        Args:
            current_score: The stored truth_score.
            last_confirmed_at: Timestamp of last confirmation/creation.
            now: Override for current time (for testing).

        Returns:
            Effective truth score after decay (read-only, not persisted).
        """
        if last_confirmed_at is None:
            return current_score

        now = now or datetime.now()
        try:
            if isinstance(last_confirmed_at, str):
                last_confirmed_at = datetime.fromisoformat(last_confirmed_at)
            elapsed_weeks = max(
                0.0, (now - last_confirmed_at).total_seconds() / (7 * 24 * 3600)
            )
        except (ValueError, TypeError):
            return current_score

        decayed = current_score - (elapsed_weeks * self._decay_rate)
        return max(self._decay_floor, min(current_score, decayed))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_effective_truth(self, metadata: dict) -> float:
        """Compute the effective truth score for a document at read time.

        Reads ``truth_score`` and ``last_confirmed_at`` from metadata,
        applies time decay, and returns the result.  If truth scoring is
        disabled or metadata lacks a truth_score, falls back to the
        legacy ``truth_score`` field or a default of 0.6.

        This is a read-only operation — the returned value should be used
        for ranking but NOT written back to storage (decay is transient).
        """
        if not self._enabled:
            return float(metadata.get("truth_score", 0.6))

        stored = float(metadata.get("truth_score", self._initial_score))
        last_confirmed = metadata.get("last_confirmed_at")

        if last_confirmed:
            return self.apply_time_decay(stored, last_confirmed)
        else:
            created = metadata.get("timestamp")
            if created:
                return self.apply_time_decay(stored, created)
            return stored


# ============================================================================
# Text helpers (stemming, tokenization, density)
# ============================================================================

DEICTIC_HINTS = (
    "explain", "that", "it", "this", "again", "another way", "different way",
)

STOPWORDS: Set[str] = set(
    "the a an to of in on for with and or if is are was were be been being "
    "by at from as this that it its".split()
)


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
    for suffix in (
        "ment", "tion", "sion", "ness", "ious", "eous",
        "ity", "ies", "ing", "ous", "ive", "ful",
        "ure", "ated", "ting", "ted", "ed", "ly", "er", "es", "s",
    ):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _salient_tokens(text: str, k: int = 12) -> Set[str]:
    """Extract most salient tokens from text (stemmed for overlap matching)."""
    toks = re.findall(r"[a-zA-Z0-9\+\-\*/=\^()]+", (text or "").lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    freq: Dict[str, int] = {}
    for t in toks:
        stemmed = _stem(t)
        freq[stemmed] = freq.get(stemmed, 0) + 1
    return {
        t
        for t, _ in sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))[:k]
    }


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
    markers = [
        "it's like", "its like", "imagine", "picture this",
        "as if", "like when", "metaphor", "analogy",
    ]
    return sum(1 for m in markers if m in t)


def _build_anchor_tokens(conv: list, maxlen: int = 20) -> Set[str]:
    """Pull anchor tokens from the last exchange with better math handling."""
    anchors: Set[str] = set()
    if conv:
        last = conv[-1]
        blob = f"{last.get('query', '')} {last.get('response', '')}"
        math_patterns = [
            r"[a-zA-Z]\([a-zA-Z]\)",
            r"\d+[a-zA-Z]\^\d+",
            r"\d+[a-zA-Z]\d*",
            r"[a-zA-Z]′\([a-zA-Z]\)",
            r"\b\d+(?:\.\d+)?\b",
            r"derivative|integral|function|equation|cdf|pdf|variance|expectation",
        ]
        for pattern in math_patterns:
            matches = re.findall(pattern, blob.lower())
            anchors.update(matches[:5])
        anchors |= _salient_tokens(blob, k=8)
    if len(anchors) > maxlen:
        anchors = set(list(anchors)[:maxlen])
    return anchors


# ============================================================================
# Memory Scorer
# ============================================================================


class MemoryScorer:
    """Unified memory scoring and ranking engine.

    Scores each memory document through a 12-step pipeline and returns them
    sorted by ``final_score`` (descending).

    Args:
        time_fn: Optional callable returning "now" as a datetime. Defaults
                 to ``datetime.now``.  Useful for testing or when you have
                 an application-level clock.
        conversation_context: Optional list of recent conversation turns
                              (dicts with ``query`` and ``response`` keys).
                              Used for continuity and anchor scoring.
        config: Optional ``ScorerConfig`` with all tunable knobs.

    Attributes:
        graph_scorer: Optional ``GraphScorerProtocol`` implementation for
                      knowledge-graph proximity boosting.  Set this after
                      construction if you have a graph backend.
        _intent_weight_overrides: Optional dict set by callers to thread
                                  intent-driven weight overrides through
                                  deep call chains (alternative to passing
                                  ``weight_overrides`` to ``rank_memories``).
    """

    def __init__(
        self,
        time_fn: Optional[Callable[[], datetime]] = None,
        conversation_context: Optional[list] = None,
        config: Optional[ScorerConfig] = None,
    ) -> None:
        self._time_fn = time_fn or datetime.now
        self.conversation_context: list = conversation_context or []
        self._config = config or ScorerConfig()
        self._truth_scorer = TruthScorer(self._config)

        # Optional integrations (set by caller)
        self.graph_scorer: Optional[GraphScorerProtocol] = None
        self._intent_weight_overrides: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Public convenience scorers
    # ------------------------------------------------------------------

    def calculate_truth_score(self, query: str, response: str) -> float:
        """Calculate truth score based on response/continuity characteristics."""
        score = 0.5
        if len(response or "") > 200:
            score += 0.1
        if "?" in (query or ""):
            score += 0.1
        confirms = (
            "yes", "correct", "exactly", "right", "understood",
            "makes sense", "good point",
        )
        if any(c in (response or "").lower() for c in confirms):
            score += 0.2
        if self.conversation_context:
            last = self.conversation_context[-1]
            last_tokens = set(
                (last.get("response", "") or "").lower().split()[:10]
            )
            if any(t in last_tokens for t in (query or "").lower().split()):
                score += 0.15
        return min(score, 1.0)

    def calculate_importance_score(self, content: str) -> float:
        """Estimate importance for retention prioritization."""
        score = 0.5
        text = content or ""
        if len(text) > 200:
            score += 0.1
        if "?" in text:
            score += 0.1
        important_keywords = [
            "important", "remember", "note", "key", "critical",
            "essential", "todo", "directive",
        ]
        if any(kw in text.lower() for kw in important_keywords):
            score += 0.2
        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Topic matching
    # ------------------------------------------------------------------

    def _calculate_topic_match(
        self, memory: Dict[str, Any], current_topic: Optional[str],
    ) -> float:
        """Calculate topic alignment score (0.0-1.0).

        Returns:
            1.0 - Exact topic match
            0.5 - No topic info available (neutral)
            0.2 - Different topic (penalty)
        """
        memory_topics: list = (memory.get("metadata", {}) or {}).get("topics", [])

        tags = memory.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        topic_tags = [t.replace("topic:", "") for t in tags if t.startswith("topic:")]
        memory_topics.extend(topic_tags)

        if not memory_topics or not current_topic:
            return 0.5

        current_lower = current_topic.lower()
        memory_topics_lower = [t.lower() for t in memory_topics]

        if current_lower in memory_topics_lower:
            return 1.0
        return 0.2

    # ------------------------------------------------------------------
    # Size penalty
    # ------------------------------------------------------------------

    def _calculate_size_penalty(self, memory: Dict[str, Any]) -> float:
        """Penalize large documents that lack keyword relevance.

        Penalty scales with document size relative to threshold.
        Returns 0.0 or a negative value (capped at -1.0).
        """
        content = memory.get("content", "") or memory.get("response", "")
        size_bytes = len(content.encode("utf-8")) if content else 0

        if size_bytes < self._config.large_doc_size_threshold:
            return 0.0

        keyword_score = memory.get("keyword_score", 0.0)
        if keyword_score > self._config.large_doc_keyword_threshold:
            return 0.0

        size_multiplier = size_bytes / self._config.large_doc_size_threshold
        penalty = self._config.large_doc_base_penalty * size_multiplier
        return max(-1.0, penalty)

    # ------------------------------------------------------------------
    # Main ranking pipeline
    # ------------------------------------------------------------------

    def rank_memories(
        self,
        memories: List[Dict[str, Any]],
        current_query: str,
        current_topic: Optional[str] = None,
        is_meta_conversational: bool = False,
        weight_overrides: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Score and rank memories through the 12-step pipeline.

        Args:
            memories: List of memory dicts.  Expected keys vary by source
                      but commonly include ``relevance_score``, ``timestamp``,
                      ``content``/``response``/``query``, ``metadata``,
                      ``collection``/``source``, ``tags``.
            current_query: The user's current query string.
            current_topic: Optional current conversation topic.
            is_meta_conversational: True for "what have we discussed" queries.
            weight_overrides: Optional per-call weight overrides. Special keys:
                ``_temporal_anchor_hours`` (float) reshapes recency decay;
                ``_timeline_mode`` (bool) boosts summaries/reflections.

        Returns:
            The same list, sorted by ``final_score`` descending.  Each dict
            gains a ``final_score`` key (and ``debug`` key when debug=True).
        """
        if not memories:
            return []

        cfg = self._config

        # Merge weight overrides on top of global defaults.
        effective_overrides = weight_overrides or self._intent_weight_overrides
        weights = dict(cfg.score_weights)
        if effective_overrides:
            weights.update(effective_overrides)

        # Pop special keys (not real scoring weights).
        temporal_anchor = weights.pop("_temporal_anchor_hours", None)
        timeline_mode = weights.pop("_timeline_mode", False)

        now = self._time_fn()
        last_10m = now - timedelta(minutes=10)

        is_deictic = _is_deictic_followup(current_query)
        anchors = _build_anchor_tokens(list(self.conversation_context))
        cq_salient = _salient_tokens(current_query, k=12)
        cq_density = _num_op_density(current_query)

        # Graph-boosted scoring (optional)
        graph_related_names: Set[str] = set()
        graph_boost_enabled = False
        graph_boost_cap = 0.15
        if self.graph_scorer is not None:
            try:
                graph_related_names = self.graph_scorer.get_related_names(
                    current_query
                )
                graph_boost_enabled = bool(graph_related_names)
            except Exception as exc:
                logger.debug(f"[Ranker] Graph boost failed: {exc}")

        for m in memories:
            # 1) base relevance with collection/source boost
            rel = float(m.get("relevance_score", 0.5))
            collection_key = m.get("collection", m.get("source", ""))
            if collection_key in cfg.collection_boosts:
                rel += cfg.collection_boosts[collection_key]

            # Cap inflated corpus relevance for large temporal windows.
            if (temporal_anchor and temporal_anchor > 48
                    and m.get("source") == "corpus"):
                rel = min(rel, 0.5)

            # 2) recency with decay
            ts = m.get("timestamp")
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
                if temporal_anchor <= 48:
                    grace_limit = temporal_anchor * 1.5
                    if age_hours <= temporal_anchor:
                        recency = 1.0 - (age_hours / temporal_anchor) * 0.15
                    elif age_hours <= grace_limit:
                        grace_frac = (age_hours - temporal_anchor) / (
                            grace_limit - temporal_anchor
                        )
                        recency = 0.85 - grace_frac * 0.15
                    else:
                        hours_past = age_hours - grace_limit
                        recency = 0.70 / (
                            1.0 + cfg.recency_decay_rate * hours_past
                        )
                else:
                    floor = max(0.60, 1.0 - (temporal_anchor / 500.0))
                    if age_hours <= temporal_anchor:
                        frac = age_hours / temporal_anchor
                        recency = floor + (1.0 - floor) * (frac ** 0.5)
                    else:
                        hours_past = age_hours - temporal_anchor
                        recency = 1.0 / (
                            1.0 + cfg.recency_decay_rate * hours_past
                        )
            else:
                recency = 1.0 / (1.0 + cfg.recency_decay_rate * age_hours)

            # 3) truth (evidence-based with time decay)
            md = m.get("metadata", {}) or {}
            truth = self._truth_scorer.compute_effective_truth(md)

            # 4) importance
            importance = float(
                m.get("importance_score", md.get("importance_score", 0.5))
            )

            # 5) continuity (overlap + recency)
            blob = (
                m.get("query", "") + " "
                + m.get("response", "") + " "
                + m.get("content", "")
            ).lower()
            m_raw_toks = re.findall(r"[a-zA-Z0-9\+\-\*/=\^()]+", blob)
            m_toks = {_stem(t) for t in m_raw_toks if len(t) > 1}
            continuity = 0.0
            if ts >= last_10m:
                continuity += 0.1
            if cq_salient:
                overlap = len(cq_salient & m_toks) / max(1, len(cq_salient))
                continuity += 0.3 * overlap

            # 5b) tag-keyword bonus
            tags_raw = m.get("tags", md.get("tags", ""))
            if tags_raw and cq_salient:
                if isinstance(tags_raw, str):
                    tag_set = {
                        _stem(t.strip().lower())
                        for t in tags_raw.split(",") if t.strip()
                    }
                else:
                    tag_set = {_stem(t.lower()) for t in tags_raw}
                tag_hits = len(cq_salient & tag_set)
                if tag_hits:
                    continuity += 0.15 * min(tag_hits, 3) / 3.0

            # 6) structural alignment
            m_density = _num_op_density(blob)
            density_alignment = 1.0 - min(
                1.0, abs(cq_density - m_density) * 3.0
            )
            structure = 0.15 * density_alignment

            # 7) penalties/bonuses
            penalty = 0.0
            if (cq_density > 0.08
                    and _analogy_markers(blob) > 0
                    and "analogy" not in current_query.lower()):
                penalty -= 0.1

            size_penalty = self._calculate_size_penalty(m)
            penalty += size_penalty

            anchor_bonus = 0.0
            if anchors:
                anchor_overlap = len(anchors & m_toks) / max(1, len(anchors))
                if is_deictic:
                    if anchor_overlap < 0.05:
                        penalty -= cfg.deictic_anchor_penalty
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
                mem_type = m.get("memory_type", "")
                collection = m.get("collection", "")
                if mem_type == "EPISODIC" or collection == "episodic":
                    meta_bonus = 0.15
                elif mem_type == "SUMMARY" or collection == "summaries":
                    meta_bonus = 0.10
                elif mem_type == "META" or collection == "meta":
                    meta_bonus = 0.12

            # 11) graph proximity bonus
            graph_bonus = 0.0
            if graph_boost_enabled and graph_related_names:
                matches = sum(
                    1 for name in graph_related_names
                    if name.lower() in blob
                )
                if matches > 0:
                    graph_bonus = min(0.05 * matches, graph_boost_cap)

            # 12) staleness penalty
            staleness_penalty = 0.0
            if cfg.staleness_enabled:
                staleness_ratio = float(
                    md.get("staleness_ratio", 0.0)
                    or m.get("staleness_ratio", 0.0)
                    or 0.0
                )
                if staleness_ratio > 0:
                    base_penalty = staleness_ratio * cfg.staleness_weight
                    if staleness_ratio >= cfg.staleness_steep_threshold:
                        base_penalty *= cfg.staleness_steep_multiplier
                    collection = m.get("collection", "")
                    if collection == "reflections":
                        base_penalty *= cfg.staleness_reflection_weight_factor
                    staleness_penalty = -min(
                        base_penalty, cfg.staleness_max_penalty
                    )

            # Timeline bonus
            timeline_bonus = 0.0
            if timeline_mode:
                src = m.get("source", m.get("collection", ""))
                if src in ("summaries", "reflections"):
                    timeline_bonus = 0.15

            m["final_score"] = (
                weights.get("relevance", 0.35) * rel
                + weights.get("recency", 0.25) * recency
                + weights.get("truth", 0.20) * truth
                + weights.get("importance", 0.05) * importance
                + weights.get("continuity", 0.10) * continuity
                + weights.get("topic_match", 0.00) * topic_match
                + structure
                + anchor_bonus
                + meta_bonus
                + graph_bonus
                + staleness_penalty
                + timeline_bonus
                + penalty
            )

            if cfg.debug or logger.isEnabledFor(logging.DEBUG):
                m["debug"] = {
                    "rel": rel,
                    "recency": recency,
                    "truth": truth,
                    "importance": importance,
                    "continuity": continuity,
                    "topic_match": topic_match,
                    "structure": structure,
                    "anchor_bonus": anchor_bonus,
                    "meta_bonus": meta_bonus,
                    "graph_bonus": graph_bonus,
                    "staleness_penalty": staleness_penalty,
                    "size_penalty": size_penalty,
                    "penalty": penalty,
                }

            # Extra guardrail for deictic drift
            if (is_deictic
                    and continuity < cfg.deictic_continuity_min
                    and anchor_bonus < 0.04):
                m["final_score"] *= 0.85

        memories.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

        if (cfg.debug or logger.isEnabledFor(logging.DEBUG)) and memories:
            logger.debug("\n[Ranker] Top 5 memories:")
            for i, mm in enumerate(memories[:5], 1):
                dbg = mm.get("debug", {})
                logger.debug(
                    f"  #{i}: score={mm.get('final_score', 0):.3f} "
                    f"(rel={dbg.get('rel', 0):.2f}, "
                    f"rec={dbg.get('recency', 0):.2f}, "
                    f"truth={dbg.get('truth', 0):.2f}, "
                    f"imp={dbg.get('importance', 0):.2f}, "
                    f"cont={dbg.get('continuity', 0):.2f}, "
                    f"topic={dbg.get('topic_match', 0):.2f}, "
                    f"struct={dbg.get('structure', 0):.2f}, "
                    f"anchor={dbg.get('anchor_bonus', 0):.2f}, "
                    f"meta={dbg.get('meta_bonus', 0):.2f}, "
                    f"graph={dbg.get('graph_bonus', 0):.2f}, "
                    f"stale={dbg.get('staleness_penalty', 0):.2f}, "
                    f"pen={dbg.get('penalty', 0):.2f}) "
                    f"Q: {mm.get('query', '')[:48]!r}"
                )

        return memories

    def apply_temporal_decay(
        self, memories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply temporal decay to memory scores (simplified version)."""
        now = self._time_fn()

        for mem_dict in memories:
            timestamp = (
                mem_dict.get("timestamp")
                or (mem_dict.get("metadata", {}) or {}).get("timestamp")
            )
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except (ValueError, TypeError):
                    continue

            decay_rate = (mem_dict.get("metadata", {}) or {}).get(
                "decay_rate", 0.01
            )
            importance_score = mem_dict.get("importance_score", 0.5)
            truth_score = mem_dict.get(
                "truth_score",
                (mem_dict.get("metadata", {}) or {}).get("truth_score", 0.5),
            )

            age_hours = (now - timestamp).total_seconds() / 3600.0
            decay_factor = 1.0 / (1.0 + decay_rate * (age_hours / 168.0))

            mem_dict["final_score"] = (
                mem_dict.get("relevance_score", 0.0)
                * max(0.1, importance_score)
                * max(0.1, decay_factor)
                * (0.75 + 0.5 * truth_score)
            )

        return memories


# ============================================================================
# CLI
# ============================================================================


def _cli_score(args: list) -> None:
    """Score a single memory document from JSON."""
    if not args:
        print("Usage: memory_scorer.py score '<json>' --query 'query text'")
        sys.exit(1)

    memory_json = args[0]
    query = "test query"
    for i, arg in enumerate(args[1:], 1):
        if arg == "--query" and i < len(args):
            query = args[i + 1]
            break

    try:
        memory = json.loads(memory_json)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        sys.exit(1)

    config = ScorerConfig(debug=True)
    scorer = MemoryScorer(config=config)
    result = scorer.rank_memories([memory], current_query=query)

    if result:
        m = result[0]
        print(f"final_score: {m['final_score']:.4f}")
        if "debug" in m:
            for k, v in m["debug"].items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def _cli_truth(args: list) -> None:
    """Compute effective truth from metadata JSON."""
    if not args:
        print("Usage: memory_scorer.py truth '<metadata_json>'")
        sys.exit(1)

    try:
        metadata = json.loads(args[0])
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        sys.exit(1)

    ts = TruthScorer()
    score = ts.compute_effective_truth(metadata)
    print(f"effective_truth: {score:.4f}")

    if "truth_score" in metadata:
        print(f"  stored:     {float(metadata['truth_score']):.4f}")
    if "last_confirmed_at" in metadata:
        print(f"  confirmed:  {metadata['last_confirmed_at']}")
    if "timestamp" in metadata:
        print(f"  created:    {metadata['timestamp']}")


def _cli_demo() -> None:
    """Run a demo scoring session."""
    print("=== Memory Scorer Demo ===\n")

    now = datetime.now()
    memories = [
        {
            "query": "what is my cat's name",
            "response": "Your cat's name is Flapjack",
            "relevance_score": 0.85,
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "collection": "facts",
            "metadata": {"truth_score": 0.9, "last_confirmed_at": now.isoformat()},
        },
        {
            "query": "how are you feeling today",
            "response": "I'm doing well, thanks for asking!",
            "relevance_score": 0.60,
            "timestamp": (now - timedelta(days=3)).isoformat(),
            "collection": "conversations",
            "metadata": {"truth_score": 0.5},
        },
        {
            "query": "what is my cat's name",
            "response": "I believe your cat is called Whiskers",
            "relevance_score": 0.80,
            "timestamp": (now - timedelta(days=30)).isoformat(),
            "collection": "facts",
            "metadata": {
                "truth_score": 0.4,
                "staleness_ratio": 0.9,
            },
        },
        {
            "content": "User has a cat named Flapjack. User enjoys cooking.",
            "relevance_score": 0.70,
            "timestamp": (now - timedelta(days=7)).isoformat(),
            "collection": "summaries",
            "metadata": {"truth_score": 0.8, "importance_score": 0.7},
        },
    ]

    config = ScorerConfig(debug=True)
    scorer = MemoryScorer(config=config)
    ranked = scorer.rank_memories(memories, current_query="what is my cat's name")

    for i, m in enumerate(ranked, 1):
        dbg = m.get("debug", {})
        q = m.get("query", m.get("content", ""))[:60]
        print(f"#{i} score={m['final_score']:.3f}  {q!r}")
        print(f"   rel={dbg.get('rel', 0):.2f}  rec={dbg.get('recency', 0):.2f}  "
              f"truth={dbg.get('truth', 0):.2f}  cont={dbg.get('continuity', 0):.2f}  "
              f"stale={dbg.get('staleness_penalty', 0):.2f}")
        print()

    print("--- Truth Scorer Demo ---\n")
    ts = TruthScorer()

    print(f"Initial (user_stated):  {ts.calculate_initial_score('user_stated'):.2f}")
    print(f"Initial (llm_extracted): {ts.calculate_initial_score('llm_extracted'):.2f}")
    print(f"Initial (inferred):     {ts.calculate_initial_score('inferred'):.2f}")
    print(f"After confirmation:     {ts.apply_confirmation(0.7):.2f}")
    print(f"After correction:       {ts.apply_correction(0.7):.2f}")
    print(f"After contradiction:    {ts.apply_contradiction(0.7):.2f}")
    print(f"Decay (4 weeks old):    {ts.apply_time_decay(0.8, now - timedelta(weeks=4), now=now):.2f}")


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: memory_scorer.py <command> [args]\n\n"
            "Commands:\n"
            "  score '<json>'  --query 'text'   Score a single memory\n"
            "  truth '<metadata_json>'          Compute effective truth\n"
            "  demo                             Run interactive demo"
        )
        sys.exit(1)

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "score":
        _cli_score(rest)
    elif cmd == "truth":
        _cli_truth(rest)
    elif cmd == "demo":
        _cli_demo()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
