"""
Session-distress enum-substring latch + dead floor-chain wiring (2026-08-27).

Root cause of the "every session floors to CONCERN from turn 2" signature
(trigger=distress_sticky_floor, constant conf=0.43, LIGHT SUPPORT replies to
logistics questions — live 2026-07-21 → 2026-08-27):

  1. `_session_in_distress` tested `previous_tone` with SUBSTRING markers
     including bare "crisis" — and the pipeline passes a CrisisLevel enum,
     whose str() is "CrisisLevel.<NAME>". "crisis" ⊂ "crisislevel.…" for
     EVERY level, so a CONVERSATIONAL previous turn still marked the session
     as in-distress and the sticky floor latched on any non-casual message.
     Now: exact membership over the level's .value (class prefix stripped).

  2. The 08-22 floor-chain budget (TONE_FLOOR_CHAIN_MAX) never counted:
     `_detect_tone` read `emotional_ctx.trigger`, but EmotionalContext's
     field is `tone_trigger` — so `_floor_chain` stayed 0 forever AND
     tone_state.json persisted trigger="" (floor-produced levels became
     seedable across restarts through the legacy-trigger-less allowance).
     The prior test was source-level only ("_floor_chain" in src) and could
     not catch the dead attribute; these tests drive the deployed method.

  3. Gate cue gap: "I would like to confirm it's this Friday that's the drop
     date" had no "?", no interrogative opener, and no lookup cue — it read
     as vent-shaped and the tone-veto killed the agentic gate on a deadline
     question. confirm/verify/double-check are now info-seeking cues.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.context_pipeline import ContextPipeline
from utils.tone_detector import CrisisLevel, _session_in_distress


class TestSessionDistressExactMatch:
    def test_conversational_enum_is_not_distress(self):
        # the live bug: str(CrisisLevel.CONVERSATIONAL) contains "crisis"
        assert _session_in_distress(CrisisLevel.CONVERSATIONAL, None) is False

    @pytest.mark.parametrize("level", [
        CrisisLevel.CONCERN, CrisisLevel.MEDIUM, CrisisLevel.HIGH,
    ])
    def test_elevated_enums_are_distress(self, level):
        assert _session_in_distress(level, None) is True

    @pytest.mark.parametrize("encoding", [
        "CrisisLevel.CONCERN", "concern", "light_support", "MEDIUM",
        "elevated_support", "crisis_support", "crisis",
    ])
    def test_elevated_string_encodings_are_distress(self, encoding):
        assert _session_in_distress(encoding, None) is True

    @pytest.mark.parametrize("encoding", [
        "CrisisLevel.CONVERSATIONAL", "conversational", "CONVERSATIONAL",
    ])
    def test_conversational_string_encodings_are_not_distress(self, encoding):
        assert _session_in_distress(encoding, None) is False

    def test_none_falls_through_to_history(self):
        now = datetime.now()
        heavy = [{"query": "my pain is back again today", "timestamp": now.isoformat(),
                  "is_heavy_topic": True}]
        assert _session_in_distress(None, heavy) is True
        assert _session_in_distress(None, []) is False

    def test_conversational_previous_tone_still_checks_history(self):
        # exact-match must not short-circuit the heavy-history path
        now = datetime.now()
        heavy = [{"query": "my pain is back again today", "timestamp": now.isoformat(),
                  "is_heavy_topic": True}]
        assert _session_in_distress(CrisisLevel.CONVERSATIONAL, heavy) is True

    def test_live_turn_reproduction(self):
        # 2026-08-27 12:43 — turn 1 was CONVERSATIONAL "Hi", history all
        # non-heavy (yesterday's rows stale) → the floor must NOT latch
        now = datetime.now()
        recents = [
            {"query": "Hi", "timestamp": (now - timedelta(minutes=2)).isoformat(),
             "is_heavy_topic": False},
            {"query": "I can't stop imagining how that appointment...",
             "timestamp": (now - timedelta(hours=18)).isoformat(),
             "is_heavy_topic": False},
        ]
        assert _session_in_distress(CrisisLevel.CONVERSATIONAL, recents) is False


def _bare_pipeline(tmp_path):
    pipe = object.__new__(ContextPipeline)
    pipe._TONE_STATE_PATH = str(tmp_path / "tone_state.json")
    pipe._last_tone_level = None
    pipe._floor_chain = 0
    pipe.memory_system = None
    pipe.model_manager = None
    return pipe


def _stub_emotional_context(level, tone_trigger):
    from utils.emotional_context import EmotionalContext
    from utils.need_detector import NeedType
    return EmotionalContext(
        crisis_level=level,
        need_type=NeedType.PERSPECTIVE,
        tone_confidence=0.43,
        need_confidence=0.0,
        tone_trigger=tone_trigger,
        need_trigger="",
        explanation="stub",
    )


class TestFloorChainRuntimeWiring:
    """Drive the deployed _detect_tone — a source-level assertion could not
    catch the `.trigger` vs `tone_trigger` dead attribute."""

    def _run_turn(self, pipe, monkeypatch, tone_trigger,
                  level=CrisisLevel.CONCERN):
        import utils.emotional_context as ec

        async def fake_analyze(message, conversation_history=None,
                               model_manager=None, previous_tone=None,
                               **kwargs):
            return _stub_emotional_context(level, tone_trigger)

        monkeypatch.setattr(ec, "analyze_emotional_context", fake_analyze)
        return asyncio.run(pipe._detect_tone("some message", None))

    def test_floor_trigger_increments_chain(self, tmp_path, monkeypatch):
        pipe = _bare_pipeline(tmp_path)
        self._run_turn(pipe, monkeypatch, "distress_sticky_floor")
        assert pipe._floor_chain == 1
        self._run_turn(pipe, monkeypatch, "distress_sticky_floor")
        assert pipe._floor_chain == 2

    def test_organic_trigger_resets_chain(self, tmp_path, monkeypatch):
        pipe = _bare_pipeline(tmp_path)
        self._run_turn(pipe, monkeypatch, "distress_sticky_floor")
        self._run_turn(pipe, monkeypatch, "semantic")
        assert pipe._floor_chain == 0

    def test_floor_trigger_persisted_to_tone_state(self, tmp_path, monkeypatch):
        # trigger="" in the state file made floor levels seedable across
        # restarts (the legacy-trigger-less allowance)
        pipe = _bare_pipeline(tmp_path)
        self._run_turn(pipe, monkeypatch, "distress_sticky_floor")
        state = json.loads(Path(pipe._TONE_STATE_PATH).read_text())
        assert state["trigger"] == "distress_sticky_floor"

    def test_chain_budget_withholds_previous_tone(self, tmp_path, monkeypatch):
        import utils.emotional_context as ec
        pipe = _bare_pipeline(tmp_path)
        pipe._floor_chain = 3  # budget exhausted
        pipe._last_tone_level = CrisisLevel.CONCERN
        seen = {}

        async def fake_analyze(message, conversation_history=None,
                               model_manager=None, previous_tone=None,
                               **kwargs):
            seen["previous_tone"] = previous_tone
            return _stub_emotional_context(CrisisLevel.CONVERSATIONAL, "")

        monkeypatch.setattr(ec, "analyze_emotional_context", fake_analyze)
        asyncio.run(pipe._detect_tone("some message", None))
        assert seen["previous_tone"] is None


class TestConfirmInfoSeekingCues:
    def test_live_confirm_request_is_info_seeking(self):
        from core.agentic.gate import _is_info_seeking
        assert _is_info_seeking(
            "No. I was not able to. I would like to confirm it's this Friday "
            "that's the drop date (5 days into classes) vrs next."
        ) is True

    @pytest.mark.parametrize("q", [
        "Can you verify the withdrawal deadline",
        "double-check the appointment time for me",
        "we should double check whether that date is right",
        "check if the registrar closes at 4pm",
    ])
    def test_verification_shapes_are_info_seeking(self, q):
        from core.agentic.gate import _is_info_seeking
        assert _is_info_seeking(q) is True

    @pytest.mark.parametrize("q", [
        "I feel like I just can't do this. Realistically.",
        "Idk why but I feel like I am guilt sprialing.",
        "I fucking hate being misunderstood.",
    ])
    def test_vents_remain_non_info_seeking(self, q):
        from core.agentic.gate import _is_info_seeking
        assert _is_info_seeking(q) is False
