"""Unit tests for utils/trigger_match.py — the single chokepoint for
deterministic keyword-trigger matching + negation detection (2026-09-04).

Closes a recurring bug class: nine separate substring incidents were each
fixed one keyword list at a time, and negation was handled ad hoc in exactly
one module (knowledge/document_generator.py) after "Do not save a document"
fired document generation anyway. This module centralizes both concerns —
compile_keyword_matcher() (word-boundary/substring matching, moved from
core.agentic.gate) and is_negated()/find_hits() (5-token negation lookback,
moved from document_generator) — and this file both unit-tests the helper
directly and drives the DEPLOYED functions at every adopted call site to
prove a negated request no longer fires the corresponding arm while the
plain positive request still does.
"""
import pytest

from core.actions.registry import detect_action_intent
from core.actions.types import ActionType
from core.agentic.gate import evaluate_agentic_gate
from core.prompt.gatherer_knowledge import _query_wants_visual
from utils.trigger_match import (
    NEGATION_CUE_RE,
    compile_keyword_matcher,
    find_hits,
    has_non_negated_hit,
    is_negated,
)
from utils.web_search_trigger import should_search_heuristic


# ===========================================================================
# (a) compile_keyword_matcher — word-boundary vs substring
# ===========================================================================

class TestCompileKeywordMatcher:
    def test_bare_word_matches_at_left_boundary(self):
        m = compile_keyword_matcher(["solve"])
        assert m("please solve this") is True
        assert m("he solves the puzzle") is True  # left-boundary prefix hit

    def test_bare_word_does_not_match_mid_word(self):
        m = compile_keyword_matcher(["solve"])
        # No word boundary immediately before "solve" inside "unresolved".
        assert m("the incomplete task is unresolved") is False
        # "resolution" does not even contain "solve" as a raw substring.
        assert m("we reached a resolution yesterday") is False

    def test_phrase_keyword_keeps_substring_semantics(self):
        m = compile_keyword_matcher(["go to http"])
        assert m("please go to https://example.com") is True
        assert m("i will not go anywhere today") is False

    def test_gate_alias_is_the_same_function(self):
        """core.agentic.gate._compile_keyword_matcher must be an alias for
        this module's compile_keyword_matcher, not a re-implementation."""
        import core.agentic.gate as gate
        assert gate._compile_keyword_matcher is compile_keyword_matcher


# ===========================================================================
# (b) is_negated — cue forms, window, under-fire doctrine
# ===========================================================================

class TestIsNegated:
    def _pos(self, text: str, needle: str) -> int:
        idx = text.find(needle)
        assert idx != -1, f"{needle!r} not found in {text!r}"
        return idx

    @pytest.mark.parametrize("text", [
        "do not search this",
        "don't search this",
        "dont search this",
        "never search this",
        "no need to search this",
        "no need search this",
        "please continue without search this",
        "let's do this rather than search this",
        "instead of search this",
        "i am not going to search this",
        "won't search this",
        "wont search this",
        "skip search this",
        "avoid search this",
        "stop search this",
        "i am not asking to search this",
        "i am not asking you to search this",
    ])
    def test_recognized_cue_forms_negate(self, text):
        assert is_negated(text, self._pos(text, "search")) is True

    def test_cue_after_match_does_not_negate(self):
        text = "write a report on this, don't just answer in chat"
        assert is_negated(text, self._pos(text, "write")) is False

    def test_negation_within_window_suppresses(self):
        text = "please don't search for that"
        assert is_negated(text, self._pos(text, "search")) is True

    def test_negation_outside_window_does_not_suppress(self):
        # Six tokens separate "don't" from "search" — outside the 5-token
        # default lookback, so a real, later request survives.
        text = "i don't like mondays but i will search for the file"
        assert is_negated(text, self._pos(text, "search")) is False

    def test_bare_not_is_not_itself_a_cue(self):
        # "I'd rather not" contains "not" but NOT the literal "rather than"
        # cue form — under-fire doctrine: only the listed cue forms count.
        text = "i'd rather not search this randomly"
        assert is_negated(text, self._pos(text, "search")) is False

    def test_rather_than_is_a_recognized_cue(self):
        text = "let's do this rather than search randomly"
        assert is_negated(text, self._pos(text, "search")) is True

    def test_custom_window_tokens(self):
        text = "never mind that odd remark today just search for it"
        pos = self._pos(text, "search")
        # Default window (5) is too short to reach back to "never" here.
        assert is_negated(text, pos, window_tokens=5) is False
        assert is_negated(text, pos, window_tokens=10) is True


# ===========================================================================
# (c) find_hits / has_non_negated_hit
# ===========================================================================

class TestFindHits:
    def test_drops_negated_hit(self):
        m = compile_keyword_matcher(["search", "calculate"])
        hits = find_hits("please don't search for this", m)
        assert hits == []

    def test_keeps_non_negated_hit(self):
        m = compile_keyword_matcher(["search", "calculate"])
        hits = find_hits("please search for this", m)
        assert any(h.keyword == "search" for h in hits)

    def test_honor_negation_false_bypasses_filter(self):
        m = compile_keyword_matcher(["search"])
        hits = find_hits("please don't search for this", m, honor_negation=False)
        assert any(h.keyword == "search" for h in hits)

    def test_has_non_negated_hit_wrapper(self):
        m = compile_keyword_matcher(["search"])
        assert has_non_negated_hit("please don't search for this", m) is False
        assert has_non_negated_hit("please search for this", m) is True

    def test_negation_cue_re_is_case_insensitive(self):
        assert NEGATION_CUE_RE.search("Do NOT proceed") is not None


# ===========================================================================
# Adoption sites: negated/positive pairs driving DEPLOYED functions
# ===========================================================================

class TestGateWebSearchArm:
    """core/agentic/gate.py — WEB_SEARCH_KEYWORDS via _WEB_SEARCH_HIT."""

    @pytest.mark.asyncio
    async def test_negated_web_search_does_not_propose_web_mode(self):
        d = await evaluate_agentic_gate(
            "don't search the web for this, just tell me"
        )
        assert "web_search" not in d.modes

    @pytest.mark.asyncio
    async def test_plain_web_search_proposes_web_mode(self):
        d = await evaluate_agentic_gate("search the web for this")
        assert d.should_trigger is True
        assert "web_search" in d.modes


class TestGateComputationArm:
    """core/agentic/gate.py — COMPUTATION_KEYWORDS via _COMPUTATION_HIT."""

    @pytest.mark.asyncio
    async def test_negated_computation_does_not_propose_computation_mode(self):
        d = await evaluate_agentic_gate("I'm not asking you to calculate it")
        assert "computation" not in d.modes

    @pytest.mark.asyncio
    async def test_plain_computation_proposes_computation_mode(self):
        d = await evaluate_agentic_gate("calculate 5 plus 5 for me")
        assert d.should_trigger is True
        assert "computation" in d.modes


class TestWebSearchTriggerNoSearch:
    """utils/web_search_trigger.py — EXPLICIT_SEARCH_PHRASES deterministic arm."""

    def test_negated_lookup_phrase_no_search(self):
        decision = should_search_heuristic("no need to look it up")
        assert decision.should_search is False

    def test_negated_explicit_phrase_no_search(self):
        decision = should_search_heuristic(
            "don't search for that, I already know"
        )
        assert decision.should_search is False

    def test_plain_explicit_phrase_search(self):
        decision = should_search_heuristic("search for that please")
        assert decision.should_search is True


class TestActionRegistryCalendarArm:
    """core/actions/registry.py — detect_action_intent()."""

    def test_negated_calendar_request_returns_none(self):
        assert detect_action_intent("don't add that to my calendar") is None

    def test_plain_calendar_request_returns_calendar(self):
        assert (
            detect_action_intent("add that to my calendar")
            == ActionType.CALENDAR_CREATE_EVENT
        )


class TestVisualGate:
    """core/prompt/gatherer_knowledge.py — _query_wants_visual()."""

    def test_negated_photo_request_is_false(self):
        assert _query_wants_visual("don't show me the photos", None) is False

    def test_plain_photo_request_is_true(self):
        assert _query_wants_visual("show me the photos", None) is True
