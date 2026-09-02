"""Regression tests for the 2026-09-01 19:24-19:43 stress-test misfires.

Five-turn red-team session (owner-run, deliberate): (1) a MEDIUM-tone vent
("I am VENTING. I am not taking any actions... emails in my head I will not
send") hit detect_action_intent and rode the veto-EXEMPT action arm into a
152s tools loop; (2) "their actions over time could lead..." matched the bare
'over time' temporal_recall token at 0.85 — above the veto floor, defeating
every tone veto AND teaching the exemplar store (which was already wholesale-
poisoned with 08-18 crisis vents taught pre-tone-gating); (3) the STM teacher
had taught terse acks ("good point") as intent prototypes; (4) the agentic
final synthesis truncated mid-list at 1915 chars — kimi-3's reasoning channel
consumed the 4096-token budget (2nd occurrence of the class).
"""
import pytest

LIVE_VENT = (
    "Ok. Rage. To be clear. I am VENTING. I am not taking any actions at any "
    "point related to this. I am furious. My psychiatrist hasn't responded to "
    "my email from more than a month ago the pain and about to take lorvatin "
    "and of course I had the reaction and could have died. I am writing emails "
    "in my head I will not send. About how especially now, they need to be "
    "careful."
)

LIVE_MORAL_VENT = (
    "I am actively choosing to do wrong here. If I believed their actions "
    "over time could lead to an uncharted flight to CAR the right thing to do "
    "is communicate. But because of societal forces, I cannot."
)


class TestActionDisavowalGuard:
    def test_live_vent_disavows(self):
        from core.agentic.gate import _ACTION_DISAVOWAL_RE
        assert _ACTION_DISAVOWAL_RE.search(LIVE_VENT)

    def test_genuine_requests_not_disavowed(self):
        from core.agentic.gate import _ACTION_DISAVOWAL_RE
        for q in (
            "email Morgan the update about registration",
            "can you send an email to my therapist summarizing the week",
            "put the HW due dates on my google calendar",
        ):
            assert not _ACTION_DISAVOWAL_RE.search(q), q

    def test_gate_does_not_force_tools_on_disavowed_vent(self):
        # Drive the deployed gate: the vent must not trigger the action arm.
        import asyncio
        from core.agentic.gate import evaluate_agentic_gate
        decision = asyncio.run(evaluate_agentic_gate(LIVE_VENT))
        assert not (decision.should_trigger and "tools" in (decision.modes or [])), (
            f"vent forced tools: {decision.reason}")


class TestTemporalRecallAnchoring:
    def _classify(self, q):
        from unittest.mock import patch
        import core.intent_classifier as intent_module

        # These are regex-routing regressions. Keep them deterministic and do
        # not load the semantic model or teach the owner's adaptive store.
        with patch.object(intent_module, "_semantic_intent", return_value=None), \
             patch.object(intent_module, "_learn_intent_exemplar"):
            return intent_module.IntentClassifier().classify(q)

    def test_moral_vent_not_temporal_recall(self):
        r = self._classify(LIVE_MORAL_VENT)
        assert r.intent.value != "temporal_recall", (r.intent, r.confidence)

    def test_third_party_over_time_not_matched(self):
        r = self._classify("their policies changed over time I guess")
        assert r.intent.value != "temporal_recall"

    def test_personal_recall_still_matches(self):
        assert self._classify("how has my sleep changed over time").confidence >= 0.85
        assert self._classify(
            "what is the trend in my email volume over time").confidence >= 0.85
        assert self._classify("what did we talk about last week").confidence >= 0.85


class TestStmTeacherShapeGuard:
    def test_ack_shapes_refused(self):
        from core.intent_classifier import _query_is_ack_shaped
        assert _query_is_ack_shaped("Oh yeah didn't think of that, good point")
        assert _query_is_ack_shaped("No they don't. And they are correct.")
        assert _query_is_ack_shaped("makes sense yeah")

    def test_substantive_query_allowed(self):
        from core.intent_classifier import _query_is_ack_shaped
        assert not _query_is_ack_shaped(
            "when did I last email the disability office about accommodations")


class TestSynthesisTokenBudget:
    def test_final_paths_use_raised_cap(self):
        # kimi-3's reasoning counts against max_tokens; 4096 truncated a
        # 1915-char answer mid-sentence twice (08-30, 09-01).
        src = open("core/agentic/controller.py").read()
        assert "max_tokens=4096," not in src
        assert src.count("max_tokens=8192,") >= 3
