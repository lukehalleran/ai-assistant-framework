"""
tests/unit/test_anti_amplification.py

Regression tests for the anti-amplification fixes: the debug session where a
distress spiral built from short/terse turns flatlined the tone signal at
CONVERSATIONAL, starving every downstream safety system.

Three mechanisms, tested deterministically (no embedder required):
1. Tone stickiness  — short non-casual turns during distress no longer reset to
   CONVERSATIONAL; explicit casual markers still decay.
2. Escalation       — sustained CONCERN (never spiking to ELEVATED) eventually
   reaches GROUNDING_PRESENCE, without touching consecutive_elevated_count.
3. Valence          — mood-congruent negative recall is capped + backfilled.
"""

import pytest

from utils.tone_detector import detect_crisis_level, CrisisLevel
from core.escalation_tracker import EscalationTracker, ResponseStrategy
from core.context_pipeline import ToneLevel
from memory.valence import (
    negative_affect_score,
    cap_negative_memories,
)


# ---------------------------------------------------------------------------
# 1. Tone stickiness (deterministic — no embedder; semantic path returns
#    CONVERSATIONAL without a model, so these exercise the fast-path gate + floor)
# ---------------------------------------------------------------------------

# The anti-flatline property: during distress, a short non-casual turn must
# stay at CONCERN-or-higher rather than resetting to CONVERSATIONAL. With the
# embedder present the escalation boost may push it past CONCERN (to MEDIUM/HIGH);
# without it the sticky floor holds it at CONCERN. Either way: not CONVERSATIONAL.
_DISTRESS = {CrisisLevel.CONCERN, CrisisLevel.MEDIUM, CrisisLevel.HIGH}


async def test_short_nonemotional_reply_during_distress_stays_elevated():
    # previous turn was CONCERN; a terse non-casual reply must NOT reset to
    # CONVERSATIONAL (this is the exact "im not sure" / "2nd" failure).
    for msg in ["im not sure", "2nd", "i dont feel like an adult often",
                "the people i dont have in my life"]:
        result = await detect_crisis_level(msg, previous_tone=CrisisLevel.CONCERN)
        assert result.level in _DISTRESS, f"{msg!r} collapsed to {result.level}"


async def test_explicit_casual_marker_decays_even_during_distress():
    # A genuine casual/ack marker is allowed to relax tone — real disengagement.
    for msg in ["ok", "lol", "sounds good", "thanks"]:
        result = await detect_crisis_level(msg, previous_tone=CrisisLevel.CONCERN)
        assert result.level == CrisisLevel.CONVERSATIONAL, f"{msg!r} did not decay"


async def test_heavy_history_prevents_fast_path_reset():
    heavy = [{"query": "I'm having a really hard time", "is_heavy_topic": True}]
    result = await detect_crisis_level("i still feel bad", conversation_history=heavy)
    assert result.level in _DISTRESS


async def test_cold_casual_short_message_still_fast_paths():
    result = await detect_crisis_level("hey", previous_tone=None)
    assert result.level == CrisisLevel.CONVERSATIONAL
    assert result.trigger == "short_casual"


async def test_keyword_crisis_unaffected_by_fast_path():
    # Keyword routing runs BEFORE the fast path; a single HIGH keyword scores 10
    # → MEDIUM by the harm-score routing (>=20 for HIGH). Point: not swallowed.
    result = await detect_crisis_level("i want to die")
    assert result.level in {CrisisLevel.HIGH, CrisisLevel.MEDIUM}


async def test_session_replay_does_not_flatline():
    """Replay the debug session's short-turn descent; distress must persist."""
    turns = [
        "i dont feel like an adult often",
        "im not sure",
        "emotionally i feel like no one will ever take me seriously",
        "the people i dont have in my life",
        "2nd",
    ]
    prev = CrisisLevel.CONCERN  # turn 1 ("...but I am anxious") had detected concern
    seen = []
    for msg in turns:
        res = await detect_crisis_level(msg, previous_tone=prev)
        seen.append(res.level)
        prev = res.level
    # The whole descent must not collapse to CONVERSATIONAL — that was the bug.
    assert all(lvl in _DISTRESS for lvl in seen), seen


# ---------------------------------------------------------------------------
# 2. Escalation tracker — sustained CONCERN reaches grounding
# ---------------------------------------------------------------------------

def test_sustained_concern_reaches_grounding():
    t = EscalationTracker(distress_threshold=5)
    strat = None
    for _ in range(5):
        strat = t.update(ToneLevel.CONCERN, "still feel behind")
    assert strat == ResponseStrategy.GROUNDING_PRESENCE
    # Regression guard: CONCERN must never count as elevated.
    assert t.consecutive_elevated_count == 0
    assert t.consecutive_distress_count == 5


def test_four_concern_turns_stay_validate():
    t = EscalationTracker(distress_threshold=5)
    strat = None
    for _ in range(4):
        strat = t.update(ToneLevel.CONCERN, "worried")
    assert strat == ResponseStrategy.VALIDATE_AND_SUGGEST


def test_conversational_resets_distress_counter():
    t = EscalationTracker(distress_threshold=5)
    for _ in range(4):
        t.update(ToneLevel.CONCERN, "worried")
    t.update(ToneLevel.CONVERSATIONAL, "ok cool")
    assert t.consecutive_distress_count == 0


def test_grounding_and_quiet_forbid_excavation():
    t = EscalationTracker(distress_threshold=5)
    for _ in range(5):
        t.update(ToneLevel.CONCERN, "still behind")
    instr = t.get_strategy_instructions()
    assert "probing question" in instr.lower()
    assert "downward" in instr.lower() or "further" in instr.lower()


# ---------------------------------------------------------------------------
# 3. Valence-aware retrieval
# ---------------------------------------------------------------------------

def test_negative_affect_scoring_orders_distress_above_neutral():
    neg = negative_affect_score("I feel so hopeless and alone, no one takes me seriously")
    neutral = negative_affect_score("I refactored the retrieval pipeline and ran the tests")
    pos = negative_affect_score("Glad that worked, solid progress on the project")
    assert neg > 0.4
    assert neutral == 0.0
    assert neg > pos


def test_cap_limits_and_backfills_negatives():
    # 8 negatives + 4 neutrals; limit 6, frac 0.5 → at most 3 negatives kept,
    # remaining slots backfilled with neutrals; order (index) preserved.
    mems = []
    for i in range(8):
        mems.append({"content": "I feel hopeless, worthless and completely alone", "id": f"n{i}"})
    for i in range(4):
        mems.append({"content": "Worked on the sim project and watched lectures", "id": f"x{i}"})
    selected, displaced = cap_negative_memories(
        mems, limit=6, max_negative_fraction=0.5, negative_threshold=0.30
    )
    assert len(selected) == 6
    n_neg = sum(1 for m in selected if negative_affect_score(m["content"]) >= 0.30)
    assert n_neg <= 3
    assert displaced > 0
    # neutrals (all 4) should have been pulled in to backfill
    assert sum(1 for m in selected if m["id"].startswith("x")) >= 3


def test_cap_is_inert_when_fraction_full():
    mems = [{"content": "hopeless and alone", "id": str(i)} for i in range(5)]
    selected, displaced = cap_negative_memories(
        mems, limit=3, max_negative_fraction=1.0, negative_threshold=0.30
    )
    assert displaced == 0
    assert [m["id"] for m in selected] == ["0", "1", "2"]


def test_cap_handles_empty():
    selected, displaced = cap_negative_memories([], limit=5, max_negative_fraction=0.5,
                                                negative_threshold=0.3)
    assert selected == [] and displaced == 0
