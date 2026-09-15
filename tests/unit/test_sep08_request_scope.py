"""2026-09-08 homework-session audit, batch B2 (request scope + routing
precision). Fixes F2/F4/F6/N1/N2 from docs/HANDOFF_20260908_homework_session_audit.md.
Live shapes (synthetic reproductions of the recorded turns; no personal data):

- R2: a terse "wait, I found it" aside followed by a real request buried in
  a LATER clause — is_self_report/is_request_shaped only looked at the whole
  message's head/overall shape and missed it.
- R6: "read" used as an R-language function name, not a retrieval verb.
- R8: "ok next q please" — task navigation, not a casual acknowledgment.
- R16: a pasted R homework script whose comment line ("#read csv data file
  into data frame") and assignment line ("<- read.csv(...)") false-positived
  the Tier-1 file/document keyword and pattern matchers.
- R23/R40: statement-shaped turns (an R error status update; plans for the
  evening) that the LLM trigger mis-flagged as memory-search intent, with no
  independent recall/request signal to corroborate it.
- R28: a homework question ending in "?" immediately followed by "probably
  doing a bit more" — the old `.{0,100}` + DOTALL implicit-comparison regex
  crossed the "?" and misrouted the turn to insight-mode pattern_temporal.

All assertions drive the deployed functions directly (no re-derivation).
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.agentic.gate import _is_request_shaped, evaluate_agentic_gate
from core.insight.detector import detect_insight_request
from utils.query_checker import (
    is_casual_acknowledgment,
    is_request_shaped,
    is_self_report,
    is_task_navigation,
    request_clauses,
    strip_code_shaped_lines,
)

R2 = (
    "wait. lol i have it. must have installed and forgot. lauching now. "
    "please show me first question which is self contained, ie if i cant "
    "anwser 1 without doing 2 show me both please"
)
R6 = (
    "read not a function had been using read.delim but there not delimed, "
    "all being read into single col"
)
R8 = "ok next q please"
R16 = (
    "yeah its not working #####ABC HW 1 PT 1\n\n"
    "#QUESTION 1 \n#read csv data file into data frame\n"
    "used_car_data <- read.csv(\"UsedCars.csv\")\n"
)
R23 = (
    'hang on doing myself not reading yet \n> #QUESTION 3\n'
    '> SE<-model$coefficients[,"Std. Error"]\n'
    'Error in model$coefficients[, "Std. Error"] : \n  incorrect number of dimensions\n\n'
    '> #QUESTION 3\n> SE<-model$coefficients\n'
    'i have it read in as a named num now, but the str grabbing didnt worlk'
)
R28 = (
    "ok on 4 Determine the critical value (or cutoff) of the t-statistic for a "
    "β estimate to be considered as significant at 95% confidence level. "
    "You need to first determine the degree of freedom of your model (Hint: "
    "you can simply retrieve the value of df.residual from the regression "
    "result.) Then you need to find the corresponding percentile of the t "
    "distribution (with that degree of freedom). (Hint: use qt() function to "
    "find a certain percentile of a t distribution.) am i just finding the 5% "
    "perentile here given input of the t values? probably doing a bit more "
    "than that"
)
R40 = (
    "texted auggie so maybe hell get back to me. if not today thats okay, "
    "will probably have the beer i bought yesterday and forgot to drink and "
    "take Flappy in the backyared"
)


def _decision(**overrides):
    base = dict(
        should_search=False, search_terms=[],
        needs_memory_search=False, needs_knowledge_search=False,
        needs_document_generation=False, needs_pattern_analysis=False,
    )
    base.update(overrides)
    return MagicMock(**base)


async def _gate(text, decision):
    with patch("utils.web_search_trigger.analyze_for_web_search_llm",
               new_callable=AsyncMock, return_value=decision):
        return await evaluate_agentic_gate(text, model_manager=MagicMock())


# ---------------------------------------------------------------------------
# strip_code_shaped_lines / request_clauses — pure helpers
# ---------------------------------------------------------------------------

class TestStripCodeShapedLines:
    def test_removes_fenced_block(self):
        out = strip_code_shaped_lines("before\n```\nx = 1\nprint(x)\n```\nafter")
        assert "x = 1" not in out
        assert "before" in out and "after" in out

    def test_removes_comment_and_blockquote_and_shell_lines(self):
        text = "\n".join([
            "keep this line",
            "# a python comment",
            "> quoted paste",
            "$ ls -la",
            ">>> 1 + 1",
            "keep this too",
        ])
        out = strip_code_shaped_lines(text)
        assert "keep this line" in out
        assert "keep this too" in out
        for gone in ("a python comment", "quoted paste", "ls -la", "1 + 1"):
            assert gone not in out

    def test_removes_r_assignment_lines(self):
        out = strip_code_shaped_lines('used_car_data <- read.csv("UsedCars.csv")\nkeep')
        assert "read.csv" not in out
        assert "keep" in out

    def test_r16_comment_and_assignment_gone(self):
        out = strip_code_shaped_lines(R16)
        assert "read csv data file" not in out
        assert "read.csv" not in out
        assert "yeah its not working" in out


class TestRequestClauses:
    def test_r2_last_clause_is_the_request(self):
        clauses = request_clauses(R2)
        assert clauses[0] == "wait"
        assert clauses[-1].startswith("please show me")

    def test_quoted_email_imperative_is_excluded(self):
        raw = "I'm just resting today\nHi Sam,\nplease show me the report\nThanks"
        clauses = request_clauses(raw)
        assert clauses == ["I'm just resting today"]


# ---------------------------------------------------------------------------
# F2 (predicates): clause-level is_self_report
# ---------------------------------------------------------------------------

class TestSelfReportClauseLevel:
    def test_r2_is_no_longer_a_self_report(self):
        assert is_self_report(R2) is False

    def test_r2_whole_message_still_not_request_shaped(self):
        # The FABLE finding: is_request_shaped looks at the whole message's
        # head/shape and correctly stays False — the fix lives in
        # is_self_report's per-clause scan, not in is_request_shaped itself.
        assert is_request_shaped(R2) is False

    def test_quoted_imperative_does_not_flip_a_genuine_self_report(self):
        # The outer text ("I'm just resting today") is self-report-shaped on
        # its own; a pasted email's imperative, fully inside a detected
        # greeting/closing block, must not flip the verdict just because it
        # is physically present in the message.
        raw = "I'm just resting today\nHi Sam,\nplease show me the report\nThanks"
        assert is_self_report(raw) is True

    def test_forwarded_note_with_quoted_request_is_not_itself_a_report(self):
        # Documented actual behavior: the outer aside ("just forwarding this
        # fyi") carries no first-person self-report shape of its own, so the
        # whole message is False — via the ordinary no-first-person-verb
        # fallthrough, not because the quoted "please" was (incorrectly)
        # treated as the outer message's own request.
        raw = "just forwarding this fyi\nHi Sam,\nplease show me the report\nThanks"
        assert is_self_report(raw) is False

    @pytest.mark.parametrize("text", [
        "I took my stimulant at 10 AM today and I'm just resting this afternoon, "
        "feels good honestly even though I got nothing done",
        "im so tired today",
        "we finally moved the couch",
    ])
    def test_existing_self_reports_still_hold(self, text):
        assert is_self_report(text) is True


# ---------------------------------------------------------------------------
# F2 (predicates): is_task_navigation / is_casual_acknowledgment
# ---------------------------------------------------------------------------

class TestTaskNavigation:
    @pytest.mark.parametrize("text", [
        R8, "next question please", "Q3", "question 2", "part 1", "first q",
    ])
    def test_navigation_shapes(self, text):
        assert is_task_navigation(text) is True

    @pytest.mark.parametrize("text", [
        "don't show the next question", "that quick question aside",
        "hmm still broken", "yeah makes sense",
    ])
    def test_non_navigation_and_negated(self, text):
        assert is_task_navigation(text) is False


class TestCasualAcknowledgmentTaskNavGuard:
    def test_r8_is_not_a_casual_acknowledgment(self):
        assert is_casual_acknowledgment(R8) is False

    @pytest.mark.parametrize("text", [
        "ok", "ok cool", "hmm still broken",
        "got it sorry frustrated lol", "yeah makes sense",
    ])
    def test_existing_acknowledgments_still_hold(self, text):
        assert is_casual_acknowledgment(text) is True

    def test_please_anywhere_disqualifies(self):
        assert is_casual_acknowledgment("ok please") is False


# ---------------------------------------------------------------------------
# N1: gate._REQUEST_SHAPED_RE verb lookahead ("read" as an R function name)
# ---------------------------------------------------------------------------

class TestRequestShapedVerbLookahead:
    def test_r6_is_not_request_shaped(self):
        assert _is_request_shaped(R6) is False

    @pytest.mark.parametrize("text", [
        "read the file",
        "check it out now",
        "pull up the veto logic",
        "ok can you rerun it",
    ])
    def test_genuine_requests_still_match(self, text):
        assert _is_request_shaped(text) is True

    @pytest.mark.parametrize("text", [
        "read not a function",
        "read is broken again",
        "read.delim was the wrong call",
        "read <- function(x) x",
    ])
    def test_verb_followed_by_non_object_shapes_excluded(self, text):
        assert _is_request_shaped(text) is False


# ---------------------------------------------------------------------------
# N2: Tier-1 file arm ignores code-shaped lines + incidental mid-body hits
# ---------------------------------------------------------------------------

class TestFileIntentCodeStripping:
    def test_r16_does_not_route_to_file_tools(self):
        d = asyncio.run(_gate(R16, _decision()))
        assert "tools" not in (d.modes or [])

    def test_bare_code_comment_no_file_intent(self):
        text = "#read the file\nprint('hello world')\nx = 1"
        d = asyncio.run(_gate(text, _decision()))
        assert d.should_trigger is False

    def test_explicit_pull_up_request_with_pasted_script_still_routes(self):
        text = "can you pull up the notes doc? here is my script:\n" + R16
        d = asyncio.run(_gate(text, _decision()))
        assert "tools" in (d.modes or [])


# ---------------------------------------------------------------------------
# F6: LLM memory-search backstop requires an independent corroborating signal
# ---------------------------------------------------------------------------

class TestMemoryBackstopCorroboration:
    def test_r23_statement_shaped_error_update_suppressed(self):
        d = asyncio.run(_gate(R23, _decision(needs_memory_search=True)))
        assert d.should_trigger is False
        assert "memory" not in (d.modes or [])

    def test_r40_evening_plans_narration_suppressed(self):
        d = asyncio.run(_gate(R40, _decision(needs_memory_search=True)))
        assert d.should_trigger is False
        assert "memory" not in (d.modes or [])

    @pytest.mark.parametrize("text", [
        # Imperative recall requests carry no "?" and no retrieval verb — the
        # first cut of the generalized backstop suppressed all three (Fable
        # referee probe, 2026-09-08). The broader query_checker request shape
        # (imperative family) is an independent corroboration signal.
        "remind me about the dentist appointment I mentioned to you last week please",
        "describe my sister's job situation from what I told you",
        "give me a rundown of my medication changes over the summer",
    ])
    def test_imperative_recall_request_still_triggers(self, text):
        d = asyncio.run(_gate(text, _decision(needs_memory_search=True)))
        assert d.should_trigger is True
        assert "memory" in d.modes

    def test_recall_cue_still_triggers(self):
        # Long enough to clear the unrelated casual-skip word-count filter so
        # the LLM-verdict/corroboration branch under test actually runs.
        text = "I told you about my dentist before, the one who remembered my name"
        d = asyncio.run(_gate(text, _decision(needs_memory_search=True)))
        assert d.should_trigger is True
        assert "memory" in d.modes

    def test_explicit_recall_request_still_triggers(self):
        text = "remind me what I said about the dentist?"
        d = asyncio.run(_gate(text, _decision(needs_memory_search=True)))
        assert d.should_trigger is True
        assert "memory" in d.modes


# ---------------------------------------------------------------------------
# F4: _IMPLICIT_PERSONAL_COMPARISON_RE no longer crosses sentence boundaries
# ---------------------------------------------------------------------------

class TestPatternTemporalSentenceBoundary:
    def test_r28_does_not_route_to_pattern_temporal(self):
        intent = detect_insight_request(R28)
        assert intent is None or intent.kind != "pattern_temporal"

    @pytest.mark.parametrize("text", [
        "Has my sleep changed since I moved?",
        "Compare how I was before and after starting night shift",
        "Does my mood track with exercise?",
    ])
    def test_existing_pattern_temporal_shapes_still_route(self, text):
        intent = detect_insight_request(text)
        assert intent is not None
        assert intent.kind == "pattern_temporal"

    def test_more_with_longitudinal_qualifier_still_routes(self):
        intent = detect_insight_request(
            "I had more coffee than usual lately, has my sleep gotten worse?"
        )
        assert intent is not None
        assert intent.kind == "pattern_temporal"

    def test_bare_more_across_a_question_mark_does_not_route(self):
        intent = detect_insight_request(
            "am I finding the 5% percentile here? probably more than that"
        )
        assert intent is None or intent.kind != "pattern_temporal"
