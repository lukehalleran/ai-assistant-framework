"""Tests for Phase 6 objective checks."""

import pytest

from eval.checks import (
    CheckResult,
    ResponseCheckResults,
    _extract_names_from_response,
    _extract_profile_facts,
    aggregate_checks_by_section,
    check_citation_validity,
    check_filler,
    check_profile_grounding,
    check_response_length,
    check_thinking_leak,
    format_checks_report,
    run_all_checks,
)


# ---------------------------------------------------------------------------
# CheckResult tests
# ---------------------------------------------------------------------------

class TestCheckResult:
    def test_roundtrip(self):
        c = CheckResult(
            check_name="test_check",
            passed=True,
            score=0.9,
            details={"key": "value"},
        )
        d = c.to_dict()
        c2 = CheckResult.from_dict(d)
        assert c2.check_name == "test_check"
        assert c2.passed is True
        assert c2.score == 0.9


class TestResponseCheckResults:
    def test_roundtrip(self):
        r = ResponseCheckResults(
            snapshot_id="snap1",
            variant_id="__baseline__",
            strategy="baseline",
            query_text="test",
            sections_removed=[],
            checks=[
                CheckResult("len", True, 1.0),
                CheckResult("filler", False, 0.5),
            ],
        )
        d = r.to_dict()
        r2 = ResponseCheckResults.from_dict(d)
        assert len(r2.checks) == 2
        assert r2.checks[1].check_name == "filler"


# ---------------------------------------------------------------------------
# Response length tests
# ---------------------------------------------------------------------------

class TestResponseLength:
    def test_normal_response(self):
        r = check_response_length("This is a normal response.", "How are you?")
        assert r.passed is True

    def test_truncated(self):
        r = check_response_length("Hi", "Tell me about X")
        assert r.passed is False
        assert r.details["issue"] == "truncated"

    def test_empty(self):
        r = check_response_length("", "Hello")
        assert r.passed is False

    def test_verbose_casual(self):
        long_response = "This is a very long response. " * 50
        r = check_response_length(long_response, "hey man")
        assert r.passed is False
        assert r.details["issue"] == "verbose_casual"

    def test_long_for_complex_ok(self):
        long_response = "Detailed explanation. " * 100  # ~2200 chars
        r = check_response_length(long_response, "How does the memory system work?")
        assert r.passed is True

    def test_excessive(self):
        huge = "word " * 2000  # > 5000 chars
        r = check_response_length(huge, "How does X work?")
        assert r.passed is False
        assert r.details["issue"] == "excessive"


# ---------------------------------------------------------------------------
# Thinking leak tests
# ---------------------------------------------------------------------------

class TestThinkingLeak:
    def test_clean_response(self):
        r = check_thinking_leak("Your brother's name is Dillion.")
        assert r.passed is True

    def test_thinking_tags(self):
        r = check_thinking_leak("<thinking>Let me check</thinking>Your brother is Dillion.")
        assert r.passed is False

    def test_meta_reasoning(self):
        r = check_thinking_leak(
            "The user wants to know about their job. "
            "I should check the profile for employment info. "
            "Based on my context, they work at a bar."
        )
        assert r.passed is False

    def test_single_hit_ok(self):
        # One hit alone shouldn't trigger (could be natural language)
        r = check_thinking_leak("I should mention that Flapjack is doing well.")
        assert r.passed is True


# ---------------------------------------------------------------------------
# Profile grounding tests
# ---------------------------------------------------------------------------

class TestProfileGrounding:
    def test_grounded_name(self):
        prompt = "[USER PROFILE] n=5\nbrother_name=Dillion"
        response = "Your brother is named Dillion."
        r = check_profile_grounding(response, prompt)
        assert r.passed is True
        assert "Dillion" in r.details["grounded"]

    def test_ungrounded_name(self):
        prompt = "[USER PROFILE] n=5\nbrother_name=Dillion"
        response = "Your brother Marcus called yesterday."
        r = check_profile_grounding(response, prompt)
        assert r.passed is False
        assert "Marcus" in r.details["ungrounded"]

    def test_no_names(self):
        r = check_profile_grounding("The weather is nice.", "some prompt")
        assert r.passed is True

    def test_grounded_in_any_section(self):
        prompt = "[RECENT CONVERSATION]\nUser mentioned Oliver yesterday\n[USER PROFILE]\nboss=Oliver"
        response = "Oliver was your boss at the brewery."
        r = check_profile_grounding(response, prompt)
        assert r.passed is True


class TestExtractProfileFacts:
    def test_basic_extraction(self):
        prompt = """[USER PROFILE] n=5
Stored facts
User: name=Luke, age=33
identity: brother_name=Dillion [2026-01-14]; boss_name=Oliver

[ACTIVE FEATURES]"""
        facts = _extract_profile_facts(prompt)
        assert facts["name"] == "Luke"
        assert facts["brother_name"] == "Dillion"
        assert facts["boss_name"] == "Oliver"

    def test_no_profile_section(self):
        facts = _extract_profile_facts("[RECENT CONVERSATION]\nsome content")
        assert facts == {}


class TestExtractNames:
    def test_mid_sentence_names(self):
        names = _extract_names_from_response(
            "your brother Dillion called and your friend Auggie stopped by"
        )
        assert "Dillion" in names
        assert "Auggie" in names

    def test_no_names(self):
        names = _extract_names_from_response("the weather is nice today")
        assert names == []


# ---------------------------------------------------------------------------
# Citation validity tests
# ---------------------------------------------------------------------------

class TestCitationValidity:
    def test_no_citations(self):
        r = check_citation_validity("No citations here.", "some prompt")
        assert r.passed is True

    def test_valid_citation(self):
        prompt = "Some context [WEB_1] source here"
        response = "According to the report [WEB_1], things are fine."
        r = check_citation_validity(response, prompt)
        assert r.passed is True

    def test_invalid_citation(self):
        prompt = "Some context without that marker"
        response = "The data shows [WEB_99] something."
        r = check_citation_validity(response, prompt)
        assert r.passed is False
        assert "[WEB_99]" in r.details["invalid"]

    def test_mixed_citations(self):
        prompt = "Data [WEB_1] here and [MEM_2] there"
        response = "See [WEB_1] and [MEM_5] for details."
        r = check_citation_validity(response, prompt)
        assert r.passed is False
        assert r.details["valid_ratio"] == 0.5


# ---------------------------------------------------------------------------
# Filler detection tests
# ---------------------------------------------------------------------------

class TestFiller:
    def test_clean_response(self):
        r = check_filler("Your brother's name is Dillion.")
        assert r.passed is True

    def test_filler_at_end(self):
        r = check_filler(
            "Your brother is Dillion. Feel free to ask if you have any other questions!"
        )
        assert r.passed is False
        assert r.details["tail_filler"] > 0

    def test_multiple_fillers(self):
        r = check_filler(
            "Here's the info. Let me know if you'd like more details. "
            "I hope this helps! Feel free to reach out."
        )
        assert r.passed is False
        assert r.details["filler_count"] >= 2

    def test_no_filler(self):
        r = check_filler(
            "Flapjack is a black cat with golden eyes. He's been bullying "
            "the other cats lately and loves sitting under the porch."
        )
        assert r.passed is True


# ---------------------------------------------------------------------------
# run_all_checks tests
# ---------------------------------------------------------------------------

class TestRunAllChecks:
    def test_returns_all_checks(self):
        checks = run_all_checks(
            response_text="Your brother is Dillion.",
            prompt_text="[USER PROFILE]\nbrother_name=Dillion\n[ACTIVE FEATURES]\nstuff",
            query_text="What's my brother's name?",
        )
        check_names = {c.check_name for c in checks}
        assert "response_length" in check_names
        assert "thinking_leak" in check_names
        assert "profile_grounding" in check_names
        assert "citation_validity" in check_names
        assert "filler" in check_names
        assert len(checks) == 5


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_basic_aggregation(self):
        results = [
            ResponseCheckResults(
                snapshot_id="s1", variant_id="__baseline__",
                strategy="baseline", query_text="test",
                sections_removed=[],
                checks=[CheckResult("filler", True, 1.0)],
            ),
            ResponseCheckResults(
                snapshot_id="s1", variant_id="s1_LOO_memories",
                strategy="leave_one_out", query_text="test",
                sections_removed=["memories"],
                checks=[CheckResult("filler", False, 0.5)],
            ),
        ]
        stats = aggregate_checks_by_section(results)
        assert "memories" in stats
        assert stats["memories"]["filler"]["delta"] == pytest.approx(-0.5)

    def test_positive_delta(self):
        """Positive delta = removing section improved the check."""
        results = [
            ResponseCheckResults(
                snapshot_id="s1", variant_id="__baseline__",
                strategy="baseline", query_text="test",
                sections_removed=[],
                checks=[CheckResult("filler", False, 0.3)],
            ),
            ResponseCheckResults(
                snapshot_id="s1", variant_id="s1_LOO_user_profile",
                strategy="leave_one_out", query_text="test",
                sections_removed=["user_profile"],
                checks=[CheckResult("filler", True, 1.0)],
            ),
        ]
        stats = aggregate_checks_by_section(results)
        assert stats["user_profile"]["filler"]["delta"] == pytest.approx(0.7)

    def test_ignores_non_loo(self):
        results = [
            ResponseCheckResults(
                snapshot_id="s1", variant_id="__baseline__",
                strategy="baseline", query_text="test",
                sections_removed=[],
                checks=[CheckResult("filler", True, 1.0)],
            ),
            ResponseCheckResults(
                snapshot_id="s1", variant_id="s1_BUNDLE_x",
                strategy="bundle", query_text="test",
                sections_removed=["a", "b"],
                checks=[CheckResult("filler", False, 0.5)],
            ),
        ]
        stats = aggregate_checks_by_section(results)
        assert len(stats) == 0

    def test_format_report(self):
        results = [
            ResponseCheckResults(
                snapshot_id="s1", variant_id="__baseline__",
                strategy="baseline", query_text="test",
                sections_removed=[],
                checks=[CheckResult("filler", True, 1.0)],
            ),
            ResponseCheckResults(
                snapshot_id="s1", variant_id="s1_LOO_memories",
                strategy="leave_one_out", query_text="test",
                sections_removed=["memories"],
                checks=[CheckResult("filler", False, 0.5)],
            ),
        ]
        stats = aggregate_checks_by_section(results)
        report = format_checks_report(stats)
        assert "filler" in report
        assert "memories" in report
