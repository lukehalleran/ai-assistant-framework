"""
Tone-deferred request clarify loop (2026-08-21).

Before this, a tone arm standing the gate down on a REQUEST was silent: the
model got a prompt with no tools and improvised ("I can't actually pull up
the logs"), and the user had no path to override short of rephrasing. Now:

  1. apply_intent_veto marks a TONE-arm veto of a request-shaped (non-vent)
     query with decision.deferred_request and arms a one-shot slot.
  2. handlers append a [DEFERRED REQUEST] note to the system prompt so the
     model acknowledges and offers instead of confabulating.
  3. A terse affirmation on the NEXT turn makes evaluate_agentic_gate re-run
     the ORIGINAL query veto-exempt (explicit user consent).

Scope guards: vent-shaped queries never get the offer (anti-excavation),
request-shaped queries never teach no_search, and the slot survives exactly
one turn.
"""

import inspect

import pytest

import core.agentic.gate as gate
from core.agentic.gate import (
    AgenticDecision,
    _is_request_shaped,
    apply_intent_veto,
    evaluate_agentic_gate,
)


def _decision(**kw):
    base = dict(
        should_trigger=True, modes=["web_search"], search_terms=[],
        matched_entities=set(), doc_gen_intent=None, self_note_intent=None,
        skip_initial_search=True, reason="triggered", veto_exempt=False,
    )
    base.update(kw)
    return AgenticDecision(**base)


@pytest.fixture(autouse=True)
def _clear_slot():
    gate._DEFERRED_REQUEST_SLOT.clear()
    yield
    gate._DEFERRED_REQUEST_SLOT.clear()


class TestRequestShaped:
    def test_imperatives(self):
        assert _is_request_shaped("check the docs for what changed")
        assert _is_request_shaped("pull up the logs from tuesday")
        assert _is_request_shaped("run the numbers on that")
        assert _is_request_shaped("please verify the fix landed")

    def test_second_person_asks(self):
        assert _is_request_shaped("can you check the docs")
        assert _is_request_shaped("hey could you summarize what changed")

    def test_discourse_marker_look_excluded(self):
        # "Look," is a discourse marker, not a command
        assert not _is_request_shaped("Look, I'm just really tired of this")

    def test_vents_and_statements_excluded(self):
        assert not _is_request_shaped("I am so unhappy with everything")
        assert not _is_request_shaped("It fixed check it out"[:8])  # "It fixed"
        assert not _is_request_shaped("the fixes on the car are done")
        assert not _is_request_shaped("")


class TestDeferralOnVeto:
    def test_acute_veto_of_request_defers(self):
        d = apply_intent_veto(
            _decision(), {"intent_type": "general", "confidence": 0.0},
            tone_level="MEDIUM", query="review the tuesday logs for what changed",
        )
        assert d.should_trigger is False
        assert d.deferred_request == "review the tuesday logs for what changed"
        assert gate._DEFERRED_REQUEST_SLOT.get("query") == d.deferred_request

    def test_emotional_support_veto_of_request_defers(self):
        # NOTE: "can you ..." asks are interrogative → info-seeking → they
        # ESCAPE the tone arms entirely (never vetoed, never deferred).
        # Imperatives without a lookup cue are the deferral's audience.
        d = apply_intent_veto(
            _decision(), {"intent_type": "emotional_support", "confidence": 0.60},
            tone_level="CONCERN", query="review the tuesday logs when you get a chance",
        )
        assert d.should_trigger is False
        assert d.deferred_request == "review the tuesday logs when you get a chance"

    def test_can_you_asks_escape_rather_than_defer(self):
        d = apply_intent_veto(
            _decision(), {"intent_type": "emotional_support", "confidence": 0.60},
            tone_level="CONCERN", query="can you review the tuesday logs",
        )
        assert d.should_trigger is True  # info-seeking escape — runs this turn
        assert d.deferred_request is None

    def test_vent_veto_never_defers(self):
        vent = "I am embarrassed for how I reacted earlier. I am so unhappy"
        d = apply_intent_veto(
            _decision(), {"intent_type": "general", "confidence": 0.0},
            tone_level="light_support", query=vent,
        )
        assert d.should_trigger is False
        assert d.deferred_request is None
        assert gate._DEFERRED_REQUEST_SLOT == {}

    def test_intent_only_veto_never_defers(self):
        # VETO_INTENTS arm is intent-driven, not tone-driven — no deferral.
        d = apply_intent_veto(
            _decision(), {"intent_type": "casual_social", "confidence": 0.90},
            tone_level="CONVERSATIONAL", query="check the docs for what changed",
        )
        if not d.should_trigger:  # only when casual_social is a veto intent
            assert d.deferred_request is None

    def test_request_shaped_never_teaches_no_search(self):
        from utils.adaptive_exemplars import get_store
        # request-shaped AND (contrived) first-person: must not become an anchor
        apply_intent_veto(
            _decision(), {"intent_type": "general", "confidence": 0.0},
            tone_level="MEDIUM", query="check on my meds schedule I am worried",
        )
        assert get_store().get_learned("web_search", "no_search") == []


class TestAffirmationRerun:
    @pytest.mark.asyncio
    async def test_affirmation_reruns_original_veto_exempt(self):
        apply_intent_veto(
            _decision(), {"intent_type": "general", "confidence": 0.0},
            tone_level="MEDIUM", query="please verify the fix landed",
        )
        assert gate._DEFERRED_REQUEST_SLOT  # armed
        redo = await evaluate_agentic_gate(user_text="yes go ahead")
        assert redo.should_trigger is True
        assert redo.veto_exempt is True
        assert "deferred-request affirmation" in redo.reason
        assert gate._DEFERRED_REQUEST_SLOT == {}  # consumed

    @pytest.mark.asyncio
    async def test_affirmed_rerun_survives_a_fresh_veto(self):
        apply_intent_veto(
            _decision(), {"intent_type": "general", "confidence": 0.0},
            tone_level="MEDIUM", query="please verify the fix landed",
        )
        redo = await evaluate_agentic_gate(user_text="sure")
        # tone is still elevated next turn — veto_exempt must hold
        after = apply_intent_veto(
            redo, {"intent_type": "general", "confidence": 0.0},
            tone_level="MEDIUM", query="sure",
        )
        assert after.should_trigger is True

    @pytest.mark.asyncio
    async def test_non_affirmation_drops_slot(self):
        apply_intent_veto(
            _decision(), {"intent_type": "general", "confidence": 0.0},
            tone_level="MEDIUM", query="run the numbers on my budget",
        )
        d = await evaluate_agentic_gate(
            user_text="anyway my mom got home from the trip yesterday"
        )
        assert "deferred-request" not in (d.reason or "")
        # slot consumed and discarded — a later affirmation does nothing
        d2 = await evaluate_agentic_gate(user_text="yes go ahead")
        assert "deferred-request" not in (d2.reason or "")

    @pytest.mark.asyncio
    async def test_no_slot_no_effect(self):
        d = await evaluate_agentic_gate(user_text="yes go ahead")
        assert "deferred-request" not in (d.reason or "")


class TestHandlersWiring:
    def test_system_prompt_note_injected_on_deferral(self):
        import gui.handlers as h
        src = inspect.getsource(h)
        assert "[DEFERRED REQUEST]" in src
        assert "deferred_request" in src
        # the note must instruct honesty, not silent skipping
        assert "Do NOT pretend" in src
