"""
memory/valence.py

Module Contract
- Purpose: Lightweight, deterministic negative-affect scoring of memory text and
  a rank-preserving selector that caps mood-congruent recall during distress.
  Rationale: semantic nearest-neighbour retrieval on a distress-flavoured query
  returns distress-flavoured memories; with no counter-pressure the prompt
  saturates with negative context and the model synthesises the pattern into a
  hardened life-narrative — a rumination amplifier. This caps the negative
  fraction and backfills with lower-affect, still-relevant memories.
- Key public functions:
  - negative_affect_score(text) -> float in [0,1]
  - is_negative(text, threshold) -> bool
  - cap_negative_memories(memories, limit, max_negative_fraction,
        negative_threshold, content_key="content") -> (selected, n_displaced)
- No LLM, no I/O, no new storage: pure lexicon scan over existing memory text.
  Works on the existing corpus without a migration.
- Side effects: none.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

# Weighted negative-affect lexicon. Phrases first (matched as substrings), then
# single tokens (matched on word boundaries). Weights are deliberately coarse —
# this is a valence *signal*, not a classifier.
_NEG_PHRASES: Dict[str, float] = {
    "want to die": 3.0, "kill myself": 3.0, "end it all": 3.0, "better off dead": 3.0,
    "no point": 2.0, "no hope": 2.0, "given up": 2.0, "can't go on": 2.5,
    "hate myself": 2.0, "no one": 1.5, "nobody": 1.5, "always alone": 2.0,
    "will always be alone": 2.5, "never take me seriously": 2.0, "feel behind": 1.5,
    "falling apart": 2.0, "breaking down": 2.0, "can't cope": 2.0, "can't handle": 1.5,
    "so alone": 2.0, "completely alone": 2.0, "nothing matters": 2.5, "what's the point": 2.0,
    "dead friends": 2.0, "don't feel real": 2.0, "nothing feels real": 2.0,
    "feel like a failure": 2.0, "feel worthless": 2.0, "feel invisible": 2.0,
    "not good enough": 1.5, "everyone would be better": 2.5,
}

_NEG_TOKENS: Dict[str, float] = {
    "anxious": 1.0, "anxiety": 1.0, "scared": 1.0, "terrified": 1.5, "afraid": 1.0,
    "depressed": 1.5, "depression": 1.5, "hopeless": 1.5, "worthless": 1.5,
    "lonely": 1.5, "alone": 1.0, "isolated": 1.5, "empty": 1.0, "numb": 1.0,
    "despair": 2.0, "grief": 1.5, "grieving": 1.5, "miserable": 1.5, "suicidal": 3.0,
    "crying": 1.0, "sobbing": 1.5, "panic": 1.5, "dread": 1.5, "ashamed": 1.5,
    "shame": 1.5, "guilt": 1.0, "failure": 1.5, "failing": 1.0, "trapped": 1.5,
    "helpless": 1.5, "powerless": 1.5, "overwhelmed": 1.0, "exhausted": 0.5,
    "withdrawal": 0.5, "derealization": 1.0, "dissociating": 1.5, "unbearable": 2.0,
    "spiraling": 1.5, "broken": 1.0, "defeated": 1.5, "abandoned": 1.5,
}

# Positive/steadying markers — used only to rank backfill candidates (a clearly
# positive memory is preferred over a neutral one when displacing negatives).
_POS_TOKENS = {
    "better", "glad", "proud", "progress", "solid", "nice", "good", "great",
    "happy", "calm", "relief", "relieved", "hopeful", "excited", "win", "won",
    "accomplished", "grateful", "recharge", "rest", "enjoyed", "fun",
}

_WORD_RE = re.compile(r"[a-z']+")

# Soft normaliser: score saturates around this weighted-hit total → 1.0.
_SATURATION = 5.0


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def negative_affect_score(text: str) -> float:
    """
    Return a negative-affect score in [0, 1]. Deterministic, length-robust:
    weighted phrase + token hits, squashed through a soft saturation so a single
    strong phrase already registers but longer venting doesn't run away.
    """
    if not text:
        return 0.0
    low = text.lower()
    weight = 0.0
    for phrase, w in _NEG_PHRASES.items():
        if phrase in low:
            weight += w
    toks = set(_tokens(low))
    for tok, w in _NEG_TOKENS.items():
        if tok in toks:
            weight += w
    if weight <= 0.0:
        return 0.0
    # 1 - exp(-weight/S): 0→0, S→~0.63, 2S→~0.86, asymptotes at 1.
    return 1.0 - math.exp(-weight / _SATURATION)


def positive_affect_score(text: str) -> float:
    """Coarse positive signal in [0,1] for backfill preference ordering."""
    if not text:
        return 0.0
    toks = set(_tokens(text))
    hits = sum(1 for t in _POS_TOKENS if t in toks)
    return min(1.0, hits / 3.0)


def is_negative(text: str, threshold: float) -> bool:
    return negative_affect_score(text) >= threshold


def _content_of(mem: Any, content_key: str) -> str:
    if isinstance(mem, dict):
        val = mem.get(content_key)
        if not val:
            # Fall back to other common text fields.
            val = mem.get("content") or mem.get("text") or ""
            if not val and isinstance(mem.get("metadata"), dict):
                val = mem["metadata"].get("content", "")
        return str(val or "")
    return str(getattr(mem, content_key, "") or "")


def cap_negative_memories(
    memories: List[Any],
    limit: int,
    max_negative_fraction: float,
    negative_threshold: float,
    content_key: str = "content",
) -> Tuple[List[Any], int]:
    """
    Select up to `limit` memories, capping high-negative ones at
    `floor(limit * max_negative_fraction)` and backfilling the freed slots with
    the next lower-affect (positive preferred, then neutral) memories — all while
    preserving the original relevance ordering in the returned list.

    Returns (selected, n_displaced) where n_displaced is how many negative
    memories were pushed out by the cap (0 if none / feature inert). Caller
    should log n_displaced — a silent cap reads as "showed everything" when it
    didn't.
    """
    if not memories:
        return [], 0
    if limit <= 0:
        return [], 0

    n = len(memories)
    scored = []  # (orig_index, mem, neg_score, pos_score, is_neg)
    for i, mem in enumerate(memories):
        text = _content_of(mem, content_key)
        neg = negative_affect_score(text)
        pos = positive_affect_score(text)
        scored.append((i, mem, neg, pos, neg >= negative_threshold))

    total_negatives = sum(1 for s in scored if s[4])
    neg_budget = int(limit * max_negative_fraction)

    # If negatives already fit under the cap (or cap disabled), keep top `limit`
    # in original order — no behaviour change.
    if total_negatives <= neg_budget or max_negative_fraction >= 1.0:
        return [m for _, m, *_ in scored[:limit]], 0

    chosen_idx = set()
    neg_used = 0
    deferred_neg = []  # (orig_index, mem, pos_score)

    # First pass in rank order: take non-negatives freely, negatives up to budget.
    for i, mem, neg, pos, is_neg in scored:
        if len(chosen_idx) >= limit:
            break
        if is_neg:
            if neg_used < neg_budget:
                chosen_idx.add(i)
                neg_used += 1
            else:
                deferred_neg.append((i, mem, pos))
        else:
            chosen_idx.add(i)

    # Backfill any remaining slots. Prefer lower-affect material: the first pass
    # already used all available non-negatives (they're never deferred), so any
    # shortfall can only be filled from deferred negatives — take them by rank.
    if len(chosen_idx) < limit and deferred_neg:
        for i, mem, _pos in deferred_neg:
            if len(chosen_idx) >= limit:
                break
            chosen_idx.add(i)

    displaced = sum(1 for (i, _m, _p) in deferred_neg if i not in chosen_idx)
    selected = [mem for (i, mem, *_rest) in scored if i in chosen_idx]
    return selected[:limit], displaced
