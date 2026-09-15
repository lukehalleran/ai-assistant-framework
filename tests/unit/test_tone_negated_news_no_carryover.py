"""tests/unit/test_tone_negated_news_no_carryover.py

T03 (2026-09-13, docs/execution/generalization/T03_design.md, owner-confirmed
"Hold, not reset"): a negated crisis phrase under news framing
("people are suffering but I don't want to die" — T02's Stage 0 tier,
trigger `observational_negated_crisis`) is CONCERN for THAT TURN ONLY. It
must not arm the sticky distress floor for a LATER turn through any of the
four verified carry-over paths (previous tone, restart seeding, heavy
history, the escalation tracker's distress counter) — but it must also NOT
clear genuine distress carried from an EARLIER turn. This is a "hold":
exactly like `distress_sticky_floor` is already held, not reset.

`model_manager=None` throughout: `detect_crisis_level`'s real Stage 2 uses
the locally cached, OFFLINE sentence-transformer embedder
(`ModelManager._get_cached_embedder()`, gated by `HF_HUB_OFFLINE=1` /
`TRANSFORMERS_OFFLINE=1`) — never a network or LLM-provider call. Tone-state
persistence goes through the conftest sandbox (autouse fixture repoints
`ContextPipeline._TONE_STATE_PATH` off `data/`); the `_bare_pipeline` seam
below additionally pins the per-instance path, matching
`test_tone_floor_self_latch.py` / `test_escalation_deescalation_fix.py`.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.context_pipeline import ContextPipeline, ToneLevel
from core.escalation_tracker import EscalationTracker
from utils.tone_detector import (
    OBSERVATIONAL_FIRST_PERSON_DISTRESS_TRIGGER,
    OBSERVATIONAL_NEGATED_CRISIS_TRIGGER,
    _heavy_row_is_distress_evidence,
)


NEG_NEWS = "people are suffering but I don't want to die"
MILD_NEWS = "people are suffering and I feel hopeless"
FOLLOWUPS = [
    "the weather was grey this morning",
    "I watched a documentary about whales last night",
    "my sister is visiting next week",
]


def _wrap(text: str, lead_words: int = 3) -> str:
    """Line-wrap the way a real chat client sends it (BC-64), the lead
    stopping BEFORE the subject/negation window T02 scopes its classifier
    over — same `lead_words=3` convention verified empirically in
    `test_tone_observational_mild_distress.py` for these exact fixtures
    ("people are suffering ...").
    """
    words = text.split(" ")
    lead = min(lead_words, max(len(words) - 1, 0))
    return " ".join(words[:lead]) + "\n  " + " ".join(words[lead:])


def _bare_pipeline(tmp_path):
    """`object.__new__` seam (test_tone_floor_self_latch.py,
    test_escalation_deescalation_fix.py): bypass `ContextPipeline.__init__`
    (which needs a real memory_system) and hand-set only the attributes
    `_detect_tone` reads or writes."""
    pipe = object.__new__(ContextPipeline)
    pipe._TONE_STATE_PATH = str(tmp_path / "tone_state.json")
    pipe._last_tone_level = None
    pipe._floor_chain = 0
    pipe.memory_system = None
    pipe.model_manager = None
    return pipe


def _row(text, heavy, minutes_ago=0):
    return {
        "query": text,
        "is_heavy_topic": heavy,
        "timestamp": (datetime.now() - timedelta(minutes=minutes_ago)).isoformat(),
    }


def _state(pipe):
    p = Path(pipe._TONE_STATE_PATH)
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------------------
# Real two-turn ContextPipeline._detect_tone
# ---------------------------------------------------------------------------

class TestNegatedNewsTurnItself:
    @pytest.mark.parametrize("wrapped", [False, True], ids=["clean", "wrapped"])
    async def test_turn1_concern_no_carryover_mutation(self, tmp_path, wrapped):
        pipe = _bare_pipeline(tmp_path)
        text = _wrap(NEG_NEWS) if wrapped else NEG_NEWS
        tone_level, ctx = await pipe._detect_tone(text, None)
        assert tone_level == ToneLevel.CONCERN
        assert ctx.tone_trigger == OBSERVATIONAL_NEGATED_CRISIS_TRIGGER
        # Hold: nothing about carry-over state changed for this turn.
        assert pipe._last_tone_level is None
        assert pipe._floor_chain == 0
        assert _state(pipe) is None  # _persist_tone was never called


class TestNoCarryoverAfterNegatedNews:
    """Verified path 1 (previous tone) AND path 3 (heavy history): a
    non-heavy history row isolates path 1 (turn 1 is in recent memories, so
    in-process stickiness would otherwise keep `_last_tone_level` alive);
    the heavy row additionally exercises path 3's read-time re-check."""

    @pytest.mark.parametrize("followup", FOLLOWUPS)
    async def test_neutral_followup_conversational_via_previous_tone_path(
        self, tmp_path, followup
    ):
        pipe = _bare_pipeline(tmp_path)
        await pipe._detect_tone(NEG_NEWS, None)
        history = [_row(NEG_NEWS, heavy=False)]
        tone_level, ctx = await pipe._detect_tone(followup, history)
        assert tone_level == ToneLevel.CONVERSATIONAL
        assert ctx.tone_trigger != "distress_sticky_floor"

    @pytest.mark.parametrize("followup", FOLLOWUPS)
    async def test_neutral_followup_conversational_via_heavy_history_path(
        self, tmp_path, followup
    ):
        pipe = _bare_pipeline(tmp_path)
        await pipe._detect_tone(NEG_NEWS, None)
        history = [_row(NEG_NEWS, heavy=True)]
        tone_level, ctx = await pipe._detect_tone(followup, history)
        assert tone_level == ToneLevel.CONVERSATIONAL
        assert ctx.tone_trigger != "distress_sticky_floor"


class TestControlMildDistressStillCarries:
    """T02's OTHER Tier B trigger, `observational_first_person_distress`,
    keeps NORMAL carry-over — only `observational_negated_crisis` is held."""

    @pytest.mark.parametrize("followup", FOLLOWUPS)
    async def test_neutral_followup_still_floors(self, tmp_path, followup):
        pipe = _bare_pipeline(tmp_path)
        tone1, ctx1 = await pipe._detect_tone(MILD_NEWS, None)
        assert tone1 == ToneLevel.CONCERN
        assert ctx1.tone_trigger == OBSERVATIONAL_FIRST_PERSON_DISTRESS_TRIGGER
        history = [_row(MILD_NEWS, heavy=False)]
        tone2, ctx2 = await pipe._detect_tone(followup, history)
        assert tone2 == ToneLevel.CONCERN
        assert ctx2.tone_trigger == "distress_sticky_floor"


class TestHoldNotReset:
    """Owner-confirmed 2026-09-13: the negated-news turn is TRANSPARENT to
    carry-over, not a reset — genuine CONCERN from an EARLIER turn survives
    a negated-news turn in between."""

    async def test_earlier_genuine_concern_survives_held_turn(self, tmp_path):
        pipe = _bare_pipeline(tmp_path)
        tone1, _ = await pipe._detect_tone(MILD_NEWS, None)  # N-1: organic
        assert tone1 == ToneLevel.CONCERN
        hist_n = [_row(MILD_NEWS, heavy=False)]
        tone2, ctx2 = await pipe._detect_tone(NEG_NEWS, hist_n)  # N: held
        assert tone2 == ToneLevel.CONCERN
        assert ctx2.tone_trigger == OBSERVATIONAL_NEGATED_CRISIS_TRIGGER
        hist_n1 = [_row(NEG_NEWS, heavy=False)]
        tone3, ctx3 = await pipe._detect_tone(FOLLOWUPS[0], hist_n1)  # N+1
        assert tone3 == ToneLevel.CONCERN
        assert ctx3.tone_trigger == "distress_sticky_floor"


# ---------------------------------------------------------------------------
# Restart seeding
# ---------------------------------------------------------------------------

class TestRestartSeeding:
    async def test_persisted_state_unchanged_by_held_turn(self, tmp_path):
        pipe = _bare_pipeline(tmp_path)
        await pipe._detect_tone(MILD_NEWS, None)  # organic CONCERN, persisted
        state_before = _state(pipe)
        assert state_before is not None
        await pipe._detect_tone(NEG_NEWS, None)  # held: no persist call
        assert _state(pipe) == state_before
        # A fresh pipeline (simulating a restart) seeds from that same state.
        pipe2 = _bare_pipeline(tmp_path)
        assert pipe2._load_persisted_tone() == state_before["level"]

    def test_loader_skips_negated_crisis_trigger_state(self, tmp_path):
        pipe = _bare_pipeline(tmp_path)
        Path(pipe._TONE_STATE_PATH).write_text(json.dumps({
            "level": "light_support",
            "trigger": OBSERVATIONAL_NEGATED_CRISIS_TRIGGER,
            "ts": datetime.now().isoformat(),
        }))
        assert pipe._load_persisted_tone() is None


# ---------------------------------------------------------------------------
# EscalationTracker distress counter
# ---------------------------------------------------------------------------

class TestEscalationTrackerHold:
    def test_negated_crisis_trigger_holds_the_count(self):
        tracker = EscalationTracker()
        tracker.update(ToneLevel.CONCERN, NEG_NEWS,
                        tone_trigger="harm_score: 4.0 (0H, 0M, 2C)")
        assert tracker.consecutive_distress_count == 1
        tracker.update(ToneLevel.CONCERN, NEG_NEWS,
                        tone_trigger=OBSERVATIONAL_NEGATED_CRISIS_TRIGGER)
        assert tracker.consecutive_distress_count == 1  # held, not incremented
        tracker.update(ToneLevel.CONCERN, NEG_NEWS,
                        tone_trigger=OBSERVATIONAL_NEGATED_CRISIS_TRIGGER)
        assert tracker.consecutive_distress_count == 1  # still held

    def test_first_person_distress_trigger_increments(self):
        tracker = EscalationTracker()
        tracker.update(ToneLevel.CONCERN, MILD_NEWS,
                        tone_trigger=OBSERVATIONAL_FIRST_PERSON_DISTRESS_TRIGGER)
        assert tracker.consecutive_distress_count == 1
        tracker.update(ToneLevel.CONCERN, MILD_NEWS,
                        tone_trigger=OBSERVATIONAL_FIRST_PERSON_DISTRESS_TRIGGER)
        assert tracker.consecutive_distress_count == 2

    def test_organic_trigger_increments(self):
        tracker = EscalationTracker()
        tracker.update(ToneLevel.CONCERN, "I feel awful",
                        tone_trigger="harm_score: 4.0 (0H, 0M, 2C)")
        assert tracker.consecutive_distress_count == 1

    def test_distress_sticky_floor_still_holds_unchanged(self):
        # Existing behavior, unmodified by this batch.
        tracker = EscalationTracker()
        tracker.update(ToneLevel.CONCERN, "ok sure", tone_trigger="distress_sticky_floor")
        assert tracker.consecutive_distress_count == 0


# ---------------------------------------------------------------------------
# Heavy-history row re-check
# ---------------------------------------------------------------------------

class TestHeavyRowReCheck:
    @pytest.mark.parametrize("wrapped", [False, True], ids=["clean", "wrapped"])
    def test_negated_news_row_not_distress_evidence(self, wrapped):
        text = _wrap(NEG_NEWS) if wrapped else NEG_NEWS
        assert _heavy_row_is_distress_evidence({"query": text}) is False

    @pytest.mark.parametrize("wrapped", [False, True], ids=["clean", "wrapped"])
    def test_mild_distress_row_is_distress_evidence(self, wrapped):
        text = _wrap(MILD_NEWS) if wrapped else MILD_NEWS
        assert _heavy_row_is_distress_evidence({"query": text}) is True

    def test_legacy_row_without_text_field_is_distress_evidence(self):
        # Unchanged fail-closed behavior for rows that predate this field.
        assert _heavy_row_is_distress_evidence({"is_heavy_topic": True}) is True

    @pytest.mark.parametrize("field", ["query", "user", "content"])
    def test_negated_news_row_false_regardless_of_field_name(self, field):
        assert _heavy_row_is_distress_evidence({field: NEG_NEWS}) is False
