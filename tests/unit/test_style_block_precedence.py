"""Tone-vs-intent style block precedence (2026-07-15).

Previously the profile style modifier ("WARM & SUPPORTIVE / prioritize
connection", from preferences.style="warm") and a confident intent style
block ("TECHNICAL HELP / skip reassurance, lead with the diagnosis") were
BOTH injected into the same system prompt, leaving the model to paper over
the contradiction. Precedence now: crisis tone > intent style > profile
style modifier.
"""

from unittest.mock import MagicMock

from core.tone_instructions import (
    get_tone_instructions,
    get_response_instructions,
    get_intent_style_instructions,
)
from utils.tone_detector import CrisisLevel
from utils.emotional_context import EmotionalContext
from utils.need_detector import NeedType


def _warm_profile():
    profile = MagicMock()
    profile.get_style_modifier.return_value = "\nSTYLE: WARM & SUPPORTIVE\n\nPrioritize connection over efficiency\n"
    return profile


def _ctx(level=CrisisLevel.CONVERSATIONAL, need=None):
    ctx = MagicMock(spec=EmotionalContext)
    ctx.crisis_level = level
    ctx.need_type = need or NeedType.NEUTRAL
    return ctx


class TestStyleModifierSuppression:
    def test_modifier_present_by_default(self):
        out = get_tone_instructions(CrisisLevel.CONVERSATIONAL, _warm_profile())
        assert "WARM & SUPPORTIVE" in out

    def test_modifier_suppressed_when_intent_owns_style(self):
        out = get_tone_instructions(
            CrisisLevel.CONVERSATIONAL, _warm_profile(), suppress_style_modifier=True
        )
        assert "WARM & SUPPORTIVE" not in out
        # The tone base instructions themselves must survive
        assert "RESPONSE MODE: CONVERSATIONAL" in out

    def test_response_instructions_passthrough(self):
        out = get_response_instructions(
            _ctx(), _warm_profile(), suppress_style_modifier=True
        )
        assert "WARM & SUPPORTIVE" not in out
        out_default = get_response_instructions(_ctx(), _warm_profile())
        assert "WARM & SUPPORTIVE" in out_default

    def test_high_crisis_never_has_modifier(self):
        out = get_tone_instructions(CrisisLevel.HIGH, _warm_profile())
        assert "WARM & SUPPORTIVE" not in out
        assert "CRISIS SUPPORT" in out


class TestCrisisSuppressesIntentStyle:
    """Crisis tone owns style entirely — the intent block must vanish."""

    def test_intent_block_suppressed_on_elevated_tone(self):
        for level in ("crisis_support", "elevated_support", "light_support"):
            assert get_intent_style_instructions("technical_help", 0.95, level) == ""

    def test_intent_block_present_when_conversational(self):
        out = get_intent_style_instructions("technical_help", 0.95, "conversational")
        assert "TECHNICAL HELP" in out


class TestOrchestratorWiring:
    def test_deployed_wrapper_forwards_suppression(self):
        """Drive THE deployed orchestrator method, not a re-derivation."""
        from core.orchestrator import DaemonOrchestrator

        orch = DaemonOrchestrator.__new__(DaemonOrchestrator)
        orch.user_profile = _warm_profile()

        with_style = orch._get_response_instructions(_ctx(), suppress_style_modifier=False)
        without_style = orch._get_response_instructions(_ctx(), suppress_style_modifier=True)
        assert "WARM & SUPPORTIVE" in with_style
        assert "WARM & SUPPORTIVE" not in without_style
