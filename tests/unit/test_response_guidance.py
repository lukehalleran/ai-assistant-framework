"""Generalized grounding layer (2026-09-06).

The first CONTEXTUAL_GROUNDING shipped ~440 incident-specific tokens (a drug
class, a neurotransmitter "reset", a drug/alcohol combination) INSIDE the cached
system-prompt prefix of every turn. Now: a short domain-neutral universal block
in the prefix + a decision-support block appended post-breakpoint only when
existing turn signals say so. Every check drives the deployed functions.
"""

import re
from pathlib import Path

import pytest

from core.prompt.section_instructions import conditional_instruction_tail
from core.response_guidance import (
    CONTEXTUAL_GROUNDING,
    DECISION_SUPPORT_GROUNDING,
    UNIVERSAL_GROUNDING,
    include_decision_support,
)

REPO = Path(__file__).resolve().parents[2]

# The exact words the first version hardcoded from one owner incident. A
# generic guard would itself be a keyword list, so this pins only the known
# leak plus a length cap — the cap is what stops the next incident's prose.
_INCIDENT_TOKENS = ("dopamine", "alcohol", "stimulant", "medication", "medicine",
                    "drug", "psychiatric", "tolerance", "vyvanse", "adhd")

Q1 = ("I took my stimulant at 10 AM today and I'm just resting this afternoon, "
      "feels good honestly even though I got nothing done")
Q2 = ("Does my history actually support scheduling occasional rest days off the "
      "medication? Weigh both sides.")
Q3 = ("Give me a detailed analysis in a table of what my record can establish "
      "about medication gaps.")


class TestWordingIsGeneric:
    @pytest.mark.parametrize("block", [UNIVERSAL_GROUNDING, DECISION_SUPPORT_GROUNDING])
    def test_no_incident_vocabulary(self, block):
        low = block.lower()
        leaked = [t for t in _INCIDENT_TOKENS if re.search(rf"\b{t}s?\b", low)]
        assert not leaked, f"owner-incident vocabulary in grounding text: {leaked}"

    def test_length_caps(self):
        # ~4 chars/token. Universal rides in EVERY cached prefix; decision block
        # rides uncached on decision turns only.
        assert len(UNIVERSAL_GROUNDING) <= 4 * 150, len(UNIVERSAL_GROUNDING)
        assert len(DECISION_SUPPORT_GROUNDING) <= 4 * 180, len(DECISION_SUPPORT_GROUNDING)

    def test_combined_constant_is_both(self):
        assert CONTEXTUAL_GROUNDING == UNIVERSAL_GROUNDING + DECISION_SUPPORT_GROUNDING

    def test_planner_and_notes_wording_generic(self):
        planner = (REPO / "core" / "response_planner.py").read_text().lower()
        assert "medication change" not in planner
        notes = (REPO / "core" / "prompt" / "section_instructions.py").read_text()
        assert "Obsidian Vault" not in notes


class TestIncludeDecisionSupport:
    def test_bare_self_report_excluded(self):
        assert include_decision_support(Q1) is False

    def test_question_and_analysis_request_included(self):
        assert include_decision_support(Q2) is True
        assert include_decision_support(Q3) is True

    def test_heavy_topic_and_tone_override_self_report(self):
        assert include_decision_support(Q1, is_heavy_topic=True) is True
        assert include_decision_support(Q1, tone_level="CONCERN") is True
        assert include_decision_support(Q1, tone_level="CrisisLevel.HIGH") is True

    def test_conversational_tone_does_not_override(self):
        assert include_decision_support(Q1, tone_level="CONVERSATIONAL") is False

    @pytest.mark.parametrize("text", ["", None, "ok", "lol"])
    def test_empty_and_small_talk_excluded(self, text):
        assert include_decision_support(text) is False
        assert include_decision_support("what's the weather", is_small_talk=True) is False

    def test_third_party_statement_excluded(self):
        assert include_decision_support("The president declared the strait closed") is False


class TestConditionalTailWiring:
    def _ctx(self, query, **kw):
        base = {"user_query": query, "tone_level": "CONVERSATIONAL"}
        base.update(kw)
        return base

    def test_self_report_gets_no_decision_block(self):
        tail = conditional_instruction_tail(self._ctx(Q1))
        assert "Decision support" not in tail

    def test_decision_turn_gets_block_after_sections(self):
        tail = conditional_instruction_tail(self._ctx(Q2, personal_notes=[{"x": 1}]))
        assert "Decision support" in tail
        assert tail.index("PERSONAL NOTES") < tail.index("Decision support")

    def test_decision_block_alone_when_no_sections(self):
        tail = conditional_instruction_tail(self._ctx(Q3))
        assert "Decision support" in tail
        assert "Context Section Guidance" not in tail

    def test_empty_ctx_still_injects_nothing(self):
        assert conditional_instruction_tail({}) == ""
        assert conditional_instruction_tail(None) == ""

    def test_tone_signal_reaches_gate(self):
        assert "Decision support" in conditional_instruction_tail(self._ctx(Q1, tone_level="CONCERN"))
        assert "Decision support" in conditional_instruction_tail(self._ctx(Q1, is_heavy_topic=True))


class TestCachePrefixPlacement:
    """The universal block is static, so it may live in the cached prefix; the
    decision block must NEVER precede PROMPT_CACHE_BREAKPOINT in the
    orchestrator (it would fork the cache every turn)."""

    def test_orchestrator_prefix_uses_universal_only(self):
        src = (REPO / "core" / "orchestrator.py").read_text()
        assert "UNIVERSAL_GROUNDING + PROMPT_CACHE_BREAKPOINT" in src
        assert "CONTEXTUAL_GROUNDING + PROMPT_CACHE_BREAKPOINT" not in src
        assert "DECISION_SUPPORT_GROUNDING + PROMPT_CACHE_BREAKPOINT" not in src

    def test_orchestrator_forwards_signals_before_tail(self):
        src = (REPO / "core" / "orchestrator.py").read_text()
        for key in ("prompt_ctx['is_heavy_topic']", "prompt_ctx['is_small_talk']", "prompt_ctx['user_query']"):
            assert key in src
            assert src.index(key) < src.index("_cond_tail = conditional_instruction_tail(prompt_ctx)")

    def test_synthesizer_gets_both_blocks(self):
        from core.insight.synthesizer import build_synthesis_prompts
        from core.insight.types import InsightIntent
        system, _ = build_synthesis_prompts(
            InsightIntent(kind="pattern_temporal", theme="t", raw_query=Q2, wants_document=False),
            [], None,
        )
        assert "Contextual grounding" in system and "Decision support" in system
