# memory/memory_scorer.py
"""
Memory scoring and ranking module.

Implements the MemoryScorerProtocol contract for scoring/ranking memory items.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set

from utils.logging_utils import get_logger
from config.app_config import (
    RECENCY_DECAY_RATE,
    TRUTH_SCORE_UPDATE_RATE,
    TRUTH_SCORE_MAX,
    COLLECTION_BOOSTS,
    DEICTIC_ANCHOR_PENALTY,
    DEICTIC_CONTINUITY_MIN,
    SCORE_WEIGHTS,
)

logger = get_logger("memory_scorer")

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


def _salient_tokens(text: str, k: int = 12) -> Set[str]:
    """Extract most salient tokens from text."""
    toks = re.findall(r"[a-zA-Z0-9\+\-\*/=\^()]+", (text or "").lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    freq: Dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
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
        """Reinforce truth scores for accessed memories and stamp metadata."""
        for mem in memories:
            mem_id = mem.get('id')
            if mem_id:
                self.access_history[mem_id] = self.access_history.get(mem_id, 0) + 1

            current_truth = float(mem.get('truth_score', 0.5))
            new_truth = min(TRUTH_SCORE_MAX, current_truth + TRUTH_SCORE_UPDATE_RATE)
            mem['truth_score'] = new_truth

            md = mem.setdefault('metadata', {})
            md['truth_score'] = new_truth
            md['access_count'] = md.get('access_count', 0) + 1
            md['last_accessed'] = datetime.now().isoformat()

    def rank_memories(self, memories: List[Dict], current_query: str) -> List[Dict]:
        """
        Score each memory using:
          - base relevance (+ collection/source boost)
          - recency (configurable decay)
          - truth / importance
          - continuity (token overlap + last-10m)
          - structure alignment (numeric/op density)
          - analogy penalty for mathy queries
          - anchor bonus (esp. for deictic follow-ups)
          - optional acceptance threshold (applied by caller after scoring)
        """
        if not memories:
            return []

        now = datetime.now()
        last_10m = now - timedelta(minutes=10)

        is_deictic = _is_deictic_followup(current_query)
        anchors = _build_anchor_tokens(list(self.conversation_context))
        cq_salient = _salient_tokens(current_query, k=12)
        cq_density = _num_op_density(current_query)

        for m in memories:
            # 1) base relevance with collection/source boost
            rel = float(m.get('relevance_score', 0.5))
            collection_key = m.get('collection', m.get('source', ''))
            if collection_key in COLLECTION_BOOSTS:
                rel += COLLECTION_BOOSTS[collection_key]

            # 2) recency with decay (using active days)
            ts = m.get('timestamp')
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except Exception:
                    ts = now
            elif not isinstance(ts, datetime):
                ts = now

            # Use active day decay if time_manager supports it, otherwise fall back to hourly decay
            if (self.time_manager is not None and
                hasattr(self.time_manager, 'calculate_active_day_decay')):
                recency = self.time_manager.calculate_active_day_decay(ts, RECENCY_DECAY_RATE)
            else:
                # Fallback to original hourly decay
                age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
                recency = 1.0 / (1.0 + RECENCY_DECAY_RATE * age_hours)

            # 3) truth (access-aware)
            md = m.get('metadata', {}) or {}
            truth = float(m.get('truth_score', md.get('truth_score', 0.6)))
            access_count = int(md.get('access_count', 0))
            if access_count > 0:
                truth = min(TRUTH_SCORE_MAX, truth + (TRUTH_SCORE_UPDATE_RATE * access_count))

            # 4) importance
            importance = float(m.get('importance_score', md.get('importance_score', 0.5)))

            # 5) continuity (overlap + recency)
            # Include content for non Q/A memories (summaries/reflections)
            blob = (m.get('query', '') + ' ' + m.get('response', '') + ' ' + m.get('content', '')).lower()
            m_toks = set(re.findall(r"[a-zA-Z0-9\+\-\*/=\^()]+", blob))
            continuity = 0.0
            if ts >= last_10m:
                continuity += 0.1
            if cq_salient:
                overlap = len(cq_salient & m_toks) / max(1, len(cq_salient))
                continuity += 0.3 * overlap

            # 6) structural alignment
            m_density = _num_op_density(blob)
            density_alignment = 1.0 - min(1.0, abs(cq_density - m_density) * 3.0)
            structure = 0.15 * density_alignment

            # 7) penalties/bonuses
            penalty = 0.0
            if cq_density > 0.08 and _analogy_markers(blob) > 0 and "analogy" not in current_query.lower():
                penalty -= 0.1

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

            m['final_score'] = (
                SCORE_WEIGHTS.get('relevance', 0.35) * rel +
                SCORE_WEIGHTS.get('recency', 0.25) * recency +
                SCORE_WEIGHTS.get('truth', 0.20) * truth +
                SCORE_WEIGHTS.get('importance', 0.05) * importance +
                SCORE_WEIGHTS.get('continuity', 0.10) * continuity +
                structure +
                anchor_bonus +
                penalty
            )

            if logger.isEnabledFor(logging.DEBUG):
                m['debug'] = {
                    'rel': rel, 'recency': recency, 'truth': truth,
                    'importance': importance, 'continuity': continuity,
                    'structure': structure, 'anchor_bonus': anchor_bonus,
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
                    f"cont={dbg.get('continuity', 0):.2f}, struct={dbg.get('structure', 0):.2f}, "
                    f"anchor={dbg.get('anchor_bonus', 0):.2f}, pen={dbg.get('penalty', 0):.2f}) "
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
                    from dateutil import parser
                    timestamp = parser.parse(timestamp)
                except:
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
