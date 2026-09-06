"""STM "recall" mislabel -> deterministic new-data override (2026-09-06).

The STM prompt is deliberately biased toward "recall" when in doubt (its own
instruction: "Treating a recall as a new event is the more dangerous error.
Default to recall or unclear when in doubt"). That mislabels two live shapes:

  Q1: "I took my stimulant at 10 AM today and I'm just resting this
      afternoon, feels good honestly even though I got nothing done" —
      classified "recall" although the clock time / "today" state were new.
  Q3: "Give me a detailed analysis in a table of what my record can
      establish about medication gaps." — classified "recall" although it
      is a request, not a report of an event.

``core.stm_analyzer.new_data_override`` is a deterministic post-check that
runs (only) when the LLM verdict is still "recall" after the existing
novelty override (novel_named_entities) has had its chance — request shape
wins first, then a data-shaped token absent from the window, then a
present-tense self-report whose keyword vocabulary is mostly absent from
the window. On a hit, reference_type is demoted to "unclear" (never
"new_event") and the formatter renders a matching note instead of the
recall "do not count it as a separate occurrence" warning.

Every test drives the DEPLOYED functions end-to-end (STMAnalyzer.analyze
with a scripted fake model, PromptFormatter._assemble_prompt) — never a
re-derivation.
"""
import asyncio
import json
from unittest.mock import MagicMock

import pytest

from utils.query_checker import extract_data_tokens, has_present_state_report


# --------------------------------------------------------------------------
# extract_data_tokens / has_present_state_report — direct unit tests
# --------------------------------------------------------------------------

class TestExtractDataTokens:
    def test_clock_am_and_24h_canonicalize_equal(self):
        am = extract_data_tokens("I took it at 10 AM")
        h24 = extract_data_tokens("I took it at 10:00")
        assert am == ["10:00"]
        assert h24 == ["10:00"]
        assert am == h24

    def test_pm_time_canonicalizes(self):
        assert extract_data_tokens("call me at 10:30pm") == ["22:30"]

    def test_dose_extracted(self):
        assert "5mg" in extract_data_tokens("Take 5 mg of the medication")

    def test_day_count_extracted(self):
        assert "day8" in extract_data_tokens("today is day 8 of the streak")

    def test_iso_date_extracted(self):
        assert "2026-09-06" in extract_data_tokens("on 2026-09-06 I went")

    def test_empty_and_garbage_input(self):
        assert extract_data_tokens("") == []
        assert extract_data_tokens(None) == []
        assert extract_data_tokens("no data here at all") == []


class TestHasPresentStateReport:
    def test_q1_true(self):
        q1 = (
            "I took my stimulant at 10 AM today and I'm just resting this "
            "afternoon, feels good honestly even though I got nothing done"
        )
        assert has_present_state_report(q1) is True

    def test_past_tense_no_anchor_false(self):
        assert has_present_state_report("I went yesterday") is False

    def test_present_progressive_with_anchor_true(self):
        assert has_present_state_report("I'm reading this afternoon") is True

    def test_anchor_without_present_form_false(self):
        assert has_present_state_report("Today was a long day") is False

    def test_progressive_without_anchor_false(self):
        assert has_present_state_report("I'm resting") is False

    def test_empty_input_false(self):
        assert has_present_state_report("") is False
        assert has_present_state_report(None) is False


# --------------------------------------------------------------------------
# core.stm_analyzer.new_data_override — direct unit tests
# --------------------------------------------------------------------------

class TestNewDataOverrideDirect:
    def test_request_shape_wins(self):
        from core.stm_analyzer import new_data_override
        result = new_data_override(
            "Give me a detailed analysis in a table of what my record can "
            "establish about medication gaps.",
            window_text="",
        )
        assert result == {"reason": "request"}

    def test_question_is_request(self):
        from core.stm_analyzer import new_data_override
        result = new_data_override("Did I take it at 10?", window_text="")
        assert result["reason"] == "request"

    def test_command_is_request(self):
        from core.stm_analyzer import new_data_override
        result = new_data_override("summarize my week", window_text="")
        assert result["reason"] == "request"

    def test_novel_data_token_wins(self):
        from core.stm_analyzer import new_data_override
        result = new_data_override(
            "I took my stimulant at 10 AM today", window_text="yesterday was rough"
        )
        assert result["reason"] == "new_data"
        assert "10:00" in result["novel_data"]

    def test_data_already_in_window_no_override(self):
        from core.stm_analyzer import new_data_override
        result = new_data_override(
            "I took my stimulant at 10 AM today",
            window_text="User: I took my stimulant at 10 AM today\nDaemon: noted",
        )
        assert result == {}

    def test_present_state_report_majority_absent(self):
        from core.stm_analyzer import new_data_override
        result = new_data_override(
            "I'm feeling surprisingly settled and grounded right now",
            window_text="User: I had a rough panic attack yesterday\nDaemon: sorry to hear that",
        )
        assert result["reason"] == "new_data"

    def test_empty_input_no_override(self):
        from core.stm_analyzer import new_data_override
        assert new_data_override("", "") == {}
        assert new_data_override(None, None) == {}


# --------------------------------------------------------------------------
# End-to-end via STMAnalyzer.analyze (scripted "recall" model), mirroring
# tests/unit/test_sep03_followups_continuity.py's _RecallModel/_analyzer/_run.
# --------------------------------------------------------------------------

class _RecallModel:
    """Scripts the prompt's recall bias: every turn comes back as 'recall'."""

    def __init__(self):
        self.prompt = ""

    async def generate_once(self, prompt, **kwargs):
        self.prompt = prompt
        return json.dumps({
            "topic": "Medication check-in",
            "user_question": "User is restating their medication routine",
            "intent": "Share",
            "tone": "casual",
            "reference_type": "recall",
            "temporal_facts": [],
            "open_threads": [],
            "constraints": [],
        })


def _analyzer(notes_text: str = ""):
    from core.stm_analyzer import STMAnalyzer
    analyzer = STMAnalyzer(_RecallModel())
    analyzer._get_recent_daily_notes_text = lambda *a, **k: notes_text
    return analyzer


def _run(analyzer, query, memories=None, last_reply=None, graph_memory=None):
    return asyncio.run(analyzer.analyze(
        recent_memories=memories or [],
        user_query=query,
        last_assistant_response=last_reply,
        graph_memory=graph_memory,
    ))


_MEDS_WINDOW = [{
    "timestamp": "2026-09-05T09:00:00",
    "query": "I've been taking my stimulant most days this week",
    "response": "Good to hear it's been consistent.",
}]

Q1 = (
    "I took my stimulant at 10 AM today and I'm just resting this "
    "afternoon, feels good honestly even though I got nothing done"
)
Q3 = (
    "Give me a detailed analysis in a table of what my record can "
    "establish about medication gaps."
)


class TestSTMNewDataOverrideEndToEnd:
    def test_q1_demotes_with_new_data_reason(self):
        result = _run(_analyzer(), Q1, memories=_MEDS_WINDOW)
        assert result["reference_type"] == "unclear"
        assert result["new_data_override"] == "new_data"
        assert "novel_data" in result
        assert result["novel_data"]

    def test_q1_novel_data_absent_from_window(self):
        from core.stm_analyzer import _word_in_text
        window = (
            "User: I've been taking my stimulant most days this week\n"
            "Daemon: Good to hear it's been consistent."
        )
        result = _run(_analyzer(), Q1, memories=_MEDS_WINDOW)
        for tok in result["novel_data"]:
            assert not _word_in_text(tok, window)

    def test_q3_demotes_with_request_reason(self):
        result = _run(_analyzer(), Q3, memories=_MEDS_WINDOW)
        assert result["reference_type"] == "unclear"
        assert result["new_data_override"] == "request"
        assert "novel_data" not in result

    def test_pure_repeat_stays_recall(self):
        # The window contains the EXACT same sentence verbatim (a real
        # restatement) — every data token and every salient keyword is
        # already present, so neither ladder step fires.
        window = [{
            "timestamp": "2026-09-05T09:00:00",
            "query": Q1,
            "response": "Sounds like a good call.",
        }]
        result = _run(_analyzer(), Q1, memories=window)
        assert result["reference_type"] == "recall"
        assert "new_data_override" not in result

    def test_question_counter_example_is_request(self):
        result = _run(_analyzer(), "Did I take it at 10?", memories=_MEDS_WINDOW)
        assert result["reference_type"] == "unclear"
        assert result["new_data_override"] == "request"

    def test_command_counter_example_is_request(self):
        result = _run(_analyzer(), "summarize my week", memories=_MEDS_WINDOW)
        assert result["reference_type"] == "unclear"
        assert result["new_data_override"] == "request"

    def test_multiline_paste_already_in_window_unchanged(self):
        paste = (
            "I've been taking my stimulant most days this week\n"
            "and it's been going fine overall\n"
            "just wanted to check in"
        )
        window = [{
            "timestamp": "2026-09-05T09:00:00",
            "query": paste,
            "response": "Sounds consistent.",
        }]
        result = _run(_analyzer(), paste, memories=window)
        assert result["reference_type"] == "recall"
        assert "new_data_override" not in result

    def test_continuation_answer_override_still_wins_over_new_data(self):
        # is_continuation_answer fires first and sets "clarification"; the
        # new-data override only runs when reference_type is still "recall".
        prior = "Want me to log it as 10 AM, or the actual time you took it?"
        result = _run(_analyzer(), "10 AM please", last_reply=prior)
        assert result["reference_type"] == "clarification"
        assert "new_data_override" not in result

    def test_ordering_novelty_override_wins_when_it_fires(self):
        # A message with BOTH a novel proper noun and a novel data token:
        # the novelty override (entities) fires first and demotes to
        # "unclear" — the new-data override step is skipped (its window
        # computation is redundant but harmless; no second key set beyond
        # what novelty already set, since reference_type is no longer
        # "recall" when the new-data check runs).
        result = _run(
            _analyzer(), "I saw Jordan at 10 AM today",
            memories=_MEDS_WINDOW,
        )
        assert result["reference_type"] == "unclear"
        assert result.get("novelty_override") is True
        assert "new_data_override" not in result


# --------------------------------------------------------------------------
# Formatter rendering (STM region only)
# --------------------------------------------------------------------------

class TestFormatterNewDataNote:
    def _render(self, stm_summary):
        from core.prompt.formatter import PromptFormatter
        fmt = PromptFormatter(token_manager=MagicMock())
        return fmt._assemble_prompt(
            context={"stm_summary": stm_summary},
            user_input=Q1, directives="", system_prompt="",
        )

    def _summary(self, **extra):
        return {
            "topic": "Medication check-in", "user_question": "q", "intent": "i",
            "tone": "casual", "reference_type": "unclear", "temporal_facts": [],
            "open_threads": [], "constraints": [], **extra,
        }

    def test_new_data_note_rendered(self):
        out = self._render(self._summary(new_data_override="new_data", novel_data=["10:00"]))
        assert "Reference Type: unclear" in out
        assert (
            "Note: the current message carries details not present in the "
            "short-term window (10:00); treat those details as new "
            "information while verifying the underlying event against "
            "memory." in out
        )
        # The unclear branch's verify-first warning still renders.
        assert "Reference is ambiguous" in out

    def test_request_note_rendered(self):
        out = self._render(self._summary(new_data_override="request"))
        assert (
            "Note: the current message is a request for analysis or action, "
            "not a report of an event; do not treat it as restating or "
            "introducing an occurrence." in out
        )

    def test_no_note_without_override_key(self):
        out = self._render(self._summary())
        assert "carries details not present" not in out
        assert "is a request for analysis or action" not in out

    def test_no_note_with_empty_novel_data(self):
        out = self._render(self._summary(new_data_override="new_data", novel_data=[]))
        assert "carries details not present" not in out

    def test_recall_warning_absent_when_unclear(self):
        out = self._render(self._summary(new_data_override="request"))
        assert "Do NOT count it as a separate occurrence" not in out

    def test_both_notes_are_mutually_exclusive(self):
        # Sanity: only one of the two new-data notes renders per summary
        # since new_data_override carries a single reason string.
        out = self._render(self._summary(new_data_override="request", novel_data=["10:00"]))
        assert "is a request for analysis or action" in out
        assert "carries details not present" not in out
