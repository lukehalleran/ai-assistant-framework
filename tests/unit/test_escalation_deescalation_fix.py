"""De-escalation lag fixes from the 2026-08-28 evening thread audit.

Live session (17:09-17:20): turn 1 organic CONCERN, turns 2-5 all
trigger=distress_sticky_floor on jokey/casual messages. Three compounding
holes:

1. The TONE_FLOOR_CHAIN_MAX budget only withheld previous_tone —
   _session_in_distress fell through to the recent-heavy-history path and
   the floor re-fired anyway (4 chained floor turns, budget was 3).
   Fix: allow_sticky_floor threaded ContextPipeline → detect_crisis_level;
   the floor STAGE is disabled when the budget is exhausted.
2. Floor-produced CONCERN fed EscalationTracker.consecutive_distress_count
   like organic distress — the counter hit the threshold on manufactured
   tone and GROUNDING_PRESENCE fired on a joke turn.
   Fix: tone_trigger threaded orchestrator → tracker.update; floored turns
   HOLD the counter but never increment it.
3. The sustained-distress upgrade had no exit: the counter never resets
   while turns stay CONCERN, so GROUNDING latched indefinitely and
   GENTLE_REENGAGEMENT was unreachable for CONCERN-only sessions.
   Fix: distress_grounding_max budget → step down to GENTLE_REENGAGEMENT
   with the counter reset; calm exits get GENTLE for the de-escalation
   window (mirrors the elevated path).

Plus: grounding prefilter now matches "research on this stuff consistently
shows" (intervening words between subject and verb).
"""

import asyncio
import logging

import pytest

from core.context_pipeline import ContextResult, ContextPipeline, ToneLevel
from core.escalation_tracker import EscalationTracker, ResponseStrategy
from core.orchestrator import DaemonOrchestrator
from utils.tone_detector import CrisisLevel


# ---------------------------------------------------------------------------
# 1. Tracker: floored tone holds but never accumulates distress
# ---------------------------------------------------------------------------

class TestFlooredToneNeverAccumulates:
    def test_organic_concern_increments(self):
        t = EscalationTracker()
        t.update(ToneLevel.CONCERN, "I feel awful", tone_trigger="semantic")
        t.update(ToneLevel.CONCERN, "still bad", tone_trigger="keyword")
        assert t.consecutive_distress_count == 2

    def test_floored_concern_holds_without_incrementing(self):
        t = EscalationTracker()
        t.update(ToneLevel.CONCERN, "I feel awful", tone_trigger="semantic")
        t.update(ToneLevel.CONCERN, "lol ran into my therapist",
                 tone_trigger="distress_sticky_floor")
        t.update(ToneLevel.CONCERN, "makes sense",
                 tone_trigger="distress_sticky_floor")
        assert t.consecutive_distress_count == 1  # held, not grown

    def test_conversational_still_resets(self):
        t = EscalationTracker()
        t.update(ToneLevel.CONCERN, "bad day", tone_trigger="semantic")
        t.update(ToneLevel.CONVERSATIONAL, "anyway what's new",
                 tone_trigger="semantic")
        assert t.consecutive_distress_count == 0

    def test_missing_trigger_counts_as_organic(self):
        # Callers that don't thread the trigger keep the old behavior.
        t = EscalationTracker()
        t.update(ToneLevel.CONCERN, "I feel awful")
        assert t.consecutive_distress_count == 1

    def test_live_session_reproduction_no_grounding(self):
        """The 17:09-17:20 session: 1 organic CONCERN + 4 floored jokey turns
        must NOT reach GROUNDING_PRESENCE (it did, on the six-pack joke)."""
        t = EscalationTracker(distress_threshold=5)
        msgs = [
            ("Ugh fuck. Okay. Yeah the rest of today has been weird.",
             "borderline_backstop"),
            ("Lol I just ran into my therapist which is always awkward",
             "distress_sticky_floor"),
            ("Yeah it's like I don't want to be seen on days when I...",
             "distress_sticky_floor"),
            ("Makes sense. I don't want to be seen as if I am trying...",
             "distress_sticky_floor"),
            ("Maybe that's why you are helpful idk.",
             "distress_sticky_floor"),
        ]
        for msg, trig in msgs:
            strategy = t.update(ToneLevel.CONCERN, msg, tone_trigger=trig)
        assert strategy != ResponseStrategy.GROUNDING_PRESENCE
        assert t.consecutive_distress_count == 1


# ---------------------------------------------------------------------------
# 2. Tracker: distress-path grounding budget + GENTLE reachability
# ---------------------------------------------------------------------------

def _spiral_to_grounding(t, n=5):
    for i in range(n):
        strategy = t.update(ToneLevel.CONCERN, f"organic distress turn {i}",
                            tone_trigger="semantic")
    return strategy


class TestDistressGroundingStepDown:
    def test_sustained_organic_concern_still_grounds(self):
        t = EscalationTracker(distress_threshold=5)
        assert _spiral_to_grounding(t) == ResponseStrategy.GROUNDING_PRESENCE

    def test_grounding_steps_down_after_budget(self):
        t = EscalationTracker(distress_threshold=5, distress_grounding_max=3)
        _spiral_to_grounding(t)  # ground streak = 1
        s2 = t.update(ToneLevel.CONCERN, "still low", tone_trigger="semantic")
        s3 = t.update(ToneLevel.CONCERN, "yeah", tone_trigger="semantic")
        assert s2 == s3 == ResponseStrategy.GROUNDING_PRESENCE  # streak 2, 3
        s4 = t.update(ToneLevel.CONCERN, "mm", tone_trigger="semantic")
        assert s4 == ResponseStrategy.GENTLE_REENGAGEMENT
        # Fresh accumulation required to re-ground
        assert t.consecutive_distress_count == 0

    def test_regrounding_needs_fresh_sustained_run(self):
        t = EscalationTracker(distress_threshold=5, distress_grounding_max=3)
        _spiral_to_grounding(t)
        for _ in range(3):
            t.update(ToneLevel.CONCERN, "x", tone_trigger="semantic")
        # stepped down; the next few CONCERN turns accumulate below threshold
        strategies = [
            t.update(ToneLevel.CONCERN, "y", tone_trigger="semantic")
            for _ in range(4)
        ]
        assert ResponseStrategy.GROUNDING_PRESENCE not in strategies
        # 5th fresh organic turn re-grounds
        s = t.update(ToneLevel.CONCERN, "z", tone_trigger="semantic")
        assert s == ResponseStrategy.GROUNDING_PRESENCE

    def test_calm_exit_gets_gentle_reengagement(self):
        """CONCERN-only sessions previously snapped GROUNDING→VALIDATE with
        no GENTLE in between (GENTLE was unreachable without ELEVATED)."""
        t = EscalationTracker(distress_threshold=5, deescalation_window=2)
        _spiral_to_grounding(t)
        s = t.update(ToneLevel.CONVERSATIONAL, "haha okay fair",
                     tone_trigger="semantic")
        assert s == ResponseStrategy.GENTLE_REENGAGEMENT
        s = t.update(ToneLevel.CONVERSATIONAL, "anyway",
                     tone_trigger="semantic")
        assert s == ResponseStrategy.GENTLE_REENGAGEMENT
        s = t.update(ToneLevel.CONVERSATIONAL, "what's for dinner",
                     tone_trigger="semantic")
        assert s == ResponseStrategy.VALIDATE_AND_SUGGEST

    def test_perspective_shift_gets_full_engagement(self):
        # Same nuance as the elevated path: analytical mode wants engagement.
        t = EscalationTracker(distress_threshold=5)
        _spiral_to_grounding(t)
        s = t.update(ToneLevel.CONVERSATIONAL, "so why do I do this?",
                     need_type="PERSPECTIVE", tone_trigger="semantic")
        assert s == ResponseStrategy.VALIDATE_AND_SUGGEST

    def test_elevated_path_grounding_never_budgeted(self):
        """CRISIS/ELEVATED grounding must hold as long as tone stays elevated
        — the budget applies ONLY to the CONCERN slow-spiral upgrade."""
        t = EscalationTracker(escalation_threshold=3, distress_grounding_max=3)
        strategies = [
            t.update(ToneLevel.ELEVATED, f"elevated turn {i}",
                     tone_trigger="keyword")
            for i in range(8)
        ]
        assert ResponseStrategy.GENTLE_REENGAGEMENT not in strategies
        assert strategies[-1] == ResponseStrategy.GROUNDING_PRESENCE

    def test_reset_clears_new_state(self):
        t = EscalationTracker(distress_threshold=5)
        _spiral_to_grounding(t)
        t.reset()
        assert t._distress_ground_streak == 0
        assert t._post_ground_calm == 0
        assert t.consecutive_distress_count == 0


# ---------------------------------------------------------------------------
# 3. Orchestrator wiring: tone_trigger reaches the tracker
# ---------------------------------------------------------------------------

def _bare_orchestrator(tracker):
    orch = object.__new__(DaemonOrchestrator)
    orch.escalation_tracker = tracker
    orch.safety_canary = None
    orch.logger = logging.getLogger("test_deescalation_wiring")
    return orch


def _make_context(query, tone, emotional_context=None):
    return ContextResult(
        processed_query=query,
        original_query=query,
        tone_level=tone,
        tone_instructions="",
        emotional_context=emotional_context,
    )


def _stub_emotional_context(tone_trigger):
    from utils.emotional_context import EmotionalContext
    from utils.need_detector import NeedType
    return EmotionalContext(
        crisis_level=CrisisLevel.CONCERN,
        need_type=NeedType.NEUTRAL,
        tone_confidence=0.43,
        need_confidence=0.0,
        tone_trigger=tone_trigger,
        need_trigger="",
        explanation="stub",
    )


class TestOrchestratorTriggersTracker:
    def test_floor_trigger_threaded_to_tracker(self):
        """Drive the deployed _update_safety_trackers: floored turns must
        not advance the distress counter through the real wiring."""
        tracker = EscalationTracker()
        orch = _bare_orchestrator(tracker)
        orch._update_safety_trackers(_make_context(
            "I feel awful", ToneLevel.CONCERN,
            _stub_emotional_context("semantic")))
        orch._update_safety_trackers(_make_context(
            "lol ran into my therapist", ToneLevel.CONCERN,
            _stub_emotional_context("distress_sticky_floor")))
        assert tracker.consecutive_distress_count == 1

    def test_no_emotional_context_still_counts(self):
        tracker = EscalationTracker()
        orch = _bare_orchestrator(tracker)
        orch._update_safety_trackers(_make_context(
            "I feel awful", ToneLevel.CONCERN, None))
        assert tracker.consecutive_distress_count == 1


# ---------------------------------------------------------------------------
# 4. Floor budget binds the floor itself (not just previous_tone)
# ---------------------------------------------------------------------------

class TestFloorBudgetBindsFloor:
    HEAVY_HISTORY = None  # built per-test with fresh timestamps

    def _heavy_history(self):
        from datetime import datetime, timedelta
        now = datetime.now()
        return [{"query": "heavy stuff", "is_heavy_topic": True,
                 "timestamp": (now - timedelta(minutes=5)).isoformat()}]

    def _detect(self, allow_sticky_floor):
        from utils.tone_detector import detect_crisis_level
        # Non-casual, non-positive message that semantically scores
        # CONVERSATIONAL; session distress supplied via heavy history —
        # the exact path that defeated the previous_tone-only budget.
        return asyncio.run(detect_crisis_level(
            "makes sense yeah I guess that tracks for me here",
            conversation_history=self._heavy_history(),
            model_manager=None,
            previous_tone=None,
            allow_sticky_floor=allow_sticky_floor,
        ))

    def test_floor_fires_via_history_when_allowed(self):
        result = self._detect(allow_sticky_floor=True)
        assert result.trigger == "distress_sticky_floor"
        assert result.level == CrisisLevel.CONCERN

    def test_budget_exhausted_disables_floor_despite_history(self):
        result = self._detect(allow_sticky_floor=False)
        assert result.trigger != "distress_sticky_floor"
        assert result.level == CrisisLevel.CONVERSATIONAL

    def test_pipeline_threads_budget_flag(self, tmp_path, monkeypatch):
        """Runtime wiring (source-level pins missed the 08-27 dead attr)."""
        import utils.emotional_context as ec

        pipe = object.__new__(ContextPipeline)
        pipe._TONE_STATE_PATH = str(tmp_path / "tone_state.json")
        pipe._last_tone_level = CrisisLevel.CONCERN
        pipe.memory_system = None
        pipe.model_manager = None
        seen = {}

        async def fake_analyze(message, conversation_history=None,
                               model_manager=None, previous_tone=None,
                               allow_sticky_floor=True, **kwargs):
            seen["allow_sticky_floor"] = allow_sticky_floor
            return _stub_emotional_context("semantic")

        monkeypatch.setattr(ec, "analyze_emotional_context", fake_analyze)

        pipe._floor_chain = 0
        asyncio.run(pipe._detect_tone("some message", None))
        assert seen["allow_sticky_floor"] is True

        pipe._floor_chain = 3  # budget exhausted
        asyncio.run(pipe._detect_tone("some message", None))
        assert seen["allow_sticky_floor"] is False


# ---------------------------------------------------------------------------
# 5. Config wiring + prefilter adjacency
# ---------------------------------------------------------------------------

class TestConfigWiring:
    def test_app_config_constant(self):
        from config.app_config import ESCALATION_DISTRESS_GROUNDING_MAX
        assert ESCALATION_DISTRESS_GROUNDING_MAX == 3

    def test_schema_default_matches(self):
        from config.schema import EscalationSection
        assert EscalationSection().distress_grounding_max == 3


class TestPrefilterAdjacency:
    def test_intervening_words_fire(self):
        from core.grounding_check import has_checkable_claims
        assert has_checkable_claims(
            "The research on this stuff consistently shows early disclosure "
            "helps outcomes."
        ) is True

    def test_adjacent_form_still_fires(self):
        from core.grounding_check import has_checkable_claims
        assert has_checkable_claims("Studies show this works.") is True

    def test_presence_text_still_silent(self):
        from core.grounding_check import has_checkable_claims
        assert has_checkable_claims(
            "I hear you. This is really hard, and it makes sense you're tired."
        ) is False
