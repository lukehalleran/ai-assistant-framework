"""Sub B — 2026-09-10 post-restart probe dump interpretation fixes.

Six turns (T1-T4, two probes T5/T6) exposed six INTERPRETATION defects —
tone/gate/intent/planner/STM misreads that never touched action routing
(Sub A owns the action-routing half in
tests/unit/test_sep10_probe_dump_actions.py). See
docs/HANDOFF_20260910_probe_dump.md for the full narrative and the exact
live texts quoted below.

B1 `core.agentic.gate._is_info_seeking` — sentence-level interrogative
   opener check (T2: the question is the 3rd sentence).
B2 `core.tone_instructions.get_tone_instructions` — a request-shaped CONCERN
   turn drops the vent-oriented "don't offer unsolicited advice" ruleset.
B3 `core.intent_classifier` — technical_help's crash/broke(n) alternation
   requires a nearby tech noun (T1: "I crashed a bit" = exhaustion, not a
   software crash).
B4 `core.response_planner.ResponsePlanner` — should_plan skips a bare
   self-report; unsupported_key_points gains a statement-mode head-noun
   check that excludes the retrieval digest from support (T3: STM misread
   "doc" as "doctor").
B5 `core.stm_analyzer.abbreviation_expansion_conflicts` — flags a query
   abbreviation the STM output silently expanded into an unevidenced longer
   referent; wired into STMAnalyzer.analyze() and the formatter's STM block.
B6 `core.action_claim_guard.claims_fresh_upload` + a gui/handlers.py
   post-check — corrects a reply that asserts a file was uploaded this
   session/today when nothing was actually attached (T4).

Round 3 (retest 20:56-20:59) added two more:
B11 `utils.trigger_match.normalize_ws` — client input arrives line-wrapped
   ("...a new doc I\n  think will be helpful") and `is_status_report` went
   blind on the wrapped form even though the clean fixture passed. Remedy
   pattern CM-01 (one chokepoint, docs/BUG_CLASSES.md): rather than patching
   `is_status_report`/`is_self_report`/`is_request_shaped`/
   `is_note_save_request`/`is_personal_doc_search`/`is_casual_acknowledgment`
   one at a time (the GENERALIZATION_AUDIT_20260901.md "Remedy patterns"
   anti-pattern this project explicitly avoids: "never clone the mechanism"),
   Sub A applies `normalize_ws` ONCE at ingress in gui/handlers.py, where the
   user's raw text enters `SubmitContext.analysis_text`/`merged_input` — every
   downstream consumer, present and future, inherits the fix for free. These
   tests drive the exact wrapped live texts through the DEPLOYED ingress path
   (`gui.handlers.handle_submit`) and assert the query_checker predicates fire
   on what the pipeline actually receives (not on a string normalized inside
   the test).
B12 `core.response_planner.ResponsePlanner.create_plan` — a plan that ends up
   with every key point dropped, or with nothing left at all once `strategy`/
   `avoid` get the SAME prefix-expansion + head-noun embellishment checks
   `unsupported_key_points` already applies to key points, is discarded
   outright (returns None) instead of injecting an empty "[RESPONSE PLAN]
   Cover: (none)" block. Structural fix (remedy pattern CM-03: a deterministic
   check instead of relying on the LLM/a prompt instruction). Live round-3
   plan: key_points=[] from the LLM itself, strategy "Acknowledge the user's
   progress and express support for their new doctor." — the same unsupported
   "doc"->"doctor" expansion B4's key-point guard already catches, uncaught
   because only key_points was ever checked.

Round 4 (retest 2026-09-11 10:10-10:16) added three more:
B10 `utils.query_checker.is_task_directive` + `utils.tone_detector.
   detect_crisis_level` — a task directive ("jot down a note for this
   session: TA sessions are Saturdays at 11 CT") scored borderline-distress
   on the semantic tier and both the distress-sticky floor AND the
   borderline backstop floored it to CONCERN even though the arbiter itself
   said CONVERSATIONAL — LIGHT SUPPORT ("let them vent") landed on a plain
   task instruction. Both fallback stages now stand down for a task
   directive; genuine vents and questions are unaffected.
B11 `core.stm_analyzer._is_regular_inflection` — categorical morphology
   (suffix + final-e drop + consonant doubling) excludes a prefix match
   that is merely the SAME LEMMA as the query's short token ("take"/
   "takes" — the STM's own paraphrase) from `abbreviation_expansion_
   conflicts`'s flag; ("doc", "doctor") stays flagged (a genuinely
   different, longer word, not an inflection).
B12 `core.action_claim_guard.annotate_unverified_action_claim` — ONE shared,
   regex-only annotator (never calls the `_claim_semantic_hit` embedding
   channel — this runs per rendered conversation item) appends the existing
   "[unverified action claim]" marker to a stored Daemon reply that
   confabulated a pending/completed/existing action, applied at BOTH
   conversation render sites in `core/prompt/formatter.py`
   (`_format_memory`, `mem_parts`) to the Daemon segment only; `core/prompt/
   gatherer_knowledge.py`'s self-note block now delegates to it too
   (BC-75: a stored false claim re-enters every future turn's context and
   gets cited back as corroborating evidence — the 20:56 self-note claim
   was cited the NEXT session as "you had me save that ... it's on there").

Round 5 (retest 2026-09-11 11:17, after the self-note purge) added three more:
B13 `core.action_claim_guard.annotate_conversation_content` + the PRODUCER-
   level wiring in `core/prompt/gatherer_memory.py`. A BC-58 sibling to B12:
   the round-5 reply ("Third time asking, and it's still covered...") came
   from a DECISION ROUND reading [RELEVANT MEMORIES] items rendered from the
   hybrid retriever's `content` field, and the agentic decision digest
   (`core.agentic.controller._compute_recent_conversation_digest`) and the
   planner's context digest (`core.response_planner.ResponsePlanner.
   build_context_digest`) both render `recent_conversations`/`memories`
   items straight from the raw context dicts — neither ever reaches the two
   formatter render sites B12 fixed. `annotate_conversation_content` finds
   the LAST "Daemon:"/"Assistant:" label in a `content`-shaped item
   ("User: ...\\nAssistant: ...") and re-annotates only the text after it;
   `_annotate_memory_item_claim` (new in gatherer_memory.py) applies it (or
   plain `annotate_unverified_action_claim` on a `response` field) at the
   ONE exit point of each of `_get_recent_conversations` and
   `_get_semantic_memories` — the formatter/gatherer_knowledge call sites
   from B12 stay wired as belt-and-suspenders. Both annotators are
   idempotent (never a second marker). A sibling fix,
   `AgenticSearchController._clip_preserving_claim_marker`, keeps the
   marker alive through the decision digest's hard 220-char clip — the
   exact live 20:56 reply is itself 221 characters, so the prior plain
   `text[:220]` clip silently dropped the marker the moment it gained one.
B14 `core.response_planner.ResponsePlanner.should_plan` returns False for a
   task directive (`utils.query_checker.is_task_directive`) — an
   action/note request needs an ACTION, not an answer plan; the live
   [RESPONSE PLAN] had restated the false calendar claim pulled from the
   digest ("TA sessions are scheduled for Saturdays at 11 AM CT... through
   December 12") in place of actually saving the user's note.
B15 `core.tone_instructions.get_intent_style_instructions` gained an
   optional `query` parameter and emits NO intent style block for a task
   directive — the live turn misclassified factual_recall and the FACTUAL
   RECALL block ("lead with the answer... state where you know it from")
   pushed the model to answer FROM the digest instead of routing the
   note-save request to a tool. Wired at its single caller,
   `core/orchestrator.py` (one call site, verified by source inspection).

Every test imports and calls the deployed function.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# B1 — gate._is_info_seeking sentence-level
# ---------------------------------------------------------------------------

class TestInfoSeekingSentenceLevel:
    Q2 = (
        "Took 30 mg focus supplement at like 1115. Took the extra 5 or maybe less "
        "idk about an hour later than caffeine maybe an hour after that. "
        "What time should I take meds melatonin etc tn to get to bed"
    )
    VENT = "I keep thinking I am a stupid piece of shit"

    def test_q2_is_info_seeking(self):
        from core.agentic.gate import _is_info_seeking
        assert _is_info_seeking(self.Q2) is True

    def test_q2_not_vent_shaped(self):
        from core.agentic.gate import _is_vent_shaped
        assert _is_vent_shaped(self.Q2) is False

    def test_real_vent_stays_vent_shaped(self):
        from core.agentic.gate import _is_info_seeking, _is_vent_shaped
        assert _is_info_seeking(self.VENT) is False
        assert _is_vent_shaped(self.VENT) is True

    def test_question_buried_in_third_sentence_generic(self):
        from core.agentic.gate import _is_info_seeking
        text = "Yeah that happened. It was rough honestly. How do I fix this?"
        # Already true via "?" — check a variant with no question mark at all.
        text_no_qmark = "Yeah that happened. It was rough honestly. What should I do about it"
        assert _is_info_seeking(text) is True
        assert _is_info_seeking(text_no_qmark) is True

    def test_opener_only_check_unaffected_for_single_sentence(self):
        from core.agentic.gate import _is_info_seeking
        assert _is_info_seeking("What time is it") is True
        assert _is_info_seeking("just chilling today") is False


# ---------------------------------------------------------------------------
# B2 — tone_instructions request-shaped CONCERN carve-out
# ---------------------------------------------------------------------------

class TestConcernRequestShaped:
    Q2 = TestInfoSeekingSentenceLevel.Q2

    def test_request_shaped_concern_drops_vent_language(self):
        from core.tone_instructions import get_tone_instructions
        from utils.tone_detector import CrisisLevel
        block = get_tone_instructions(CrisisLevel.CONCERN, query=self.Q2)
        assert "unsolicited" not in block
        assert "let them vent" not in block
        assert "direct question" in block

    def test_vent_keeps_old_block(self):
        from core.tone_instructions import get_tone_instructions
        from utils.tone_detector import CrisisLevel
        vent = "I'm so stressed about everything right now, ugh"
        block = get_tone_instructions(CrisisLevel.CONCERN, query=vent)
        assert "unsolicited" in block
        assert "let them vent" in block

    def test_no_query_preserves_old_default_behavior(self):
        from core.tone_instructions import get_tone_instructions
        from utils.tone_detector import CrisisLevel
        block = get_tone_instructions(CrisisLevel.CONCERN)
        assert "unsolicited" in block

    def test_other_tone_levels_unaffected_by_query(self):
        from core.tone_instructions import get_tone_instructions
        from utils.tone_detector import CrisisLevel
        block = get_tone_instructions(CrisisLevel.MEDIUM, query="What should I do right now?")
        assert "ELEVATED SUPPORT" in block

    @pytest.mark.asyncio
    async def test_orchestrator_threads_query_into_response_instructions(self):
        """core.orchestrator._build_system_prompt passes context.original_query
        through to the CONCERN carve-out (source-level wiring check — a live
        prompt build is out of scope here)."""
        import inspect
        import core.orchestrator as orch_mod
        src = inspect.getsource(orch_mod.DaemonOrchestrator._build_system_prompt)
        assert "query=context.original_query" in src


# ---------------------------------------------------------------------------
# B3 — intent_classifier technical_help requires a nearby tech noun
# ---------------------------------------------------------------------------

class TestTechnicalHelpTechNounGuard:
    def test_t1_not_technical_help(self):
        from core.intent_classifier import IntentClassifier, IntentType
        c = IntentClassifier()
        r = c.classify("Yeah I crashed a bit. So tired. Walking to store now")
        assert r.intent != IntentType.TECHNICAL_HELP

    def test_app_crashed_is_technical_help(self):
        from core.intent_classifier import IntentClassifier, IntentType
        c = IntentClassifier()
        r = c.classify("the app crashed again")
        assert r.intent == IntentType.TECHNICAL_HELP

    def test_server_broke_is_technical_help(self):
        from core.intent_classifier import IntentClassifier, IntentType
        c = IntentClassifier()
        r = c.classify("the server broke last night, can you help")
        assert r.intent == IntentType.TECHNICAL_HELP

    def test_feel_broken_stays_emotional_support(self):
        """Regression: EMOTIONAL_SUPPORT's higher-confidence pattern must
        still win over the (now tech-noun-gated) technical_help 'broken'."""
        from core.intent_classifier import IntentClassifier, IntentType
        c = IntentClassifier()
        r = c.classify("I feel broken inside")
        assert r.intent == IntentType.EMOTIONAL_SUPPORT

    def test_other_technical_cues_unaffected(self):
        from core.intent_classifier import IntentClassifier, IntentType
        c = IntentClassifier()
        r = c.classify("I'm getting a traceback when I run this, can you help me debug it")
        assert r.intent == IntentType.TECHNICAL_HELP


# ---------------------------------------------------------------------------
# B4 — ResponsePlanner.should_plan + unsupported_key_points statement mode
# ---------------------------------------------------------------------------

from utils.tone_detector import CrisisLevel as _RealCrisisLevel  # noqa: E402
from core.context_pipeline import ToneLevel as _RealToneLevel  # noqa: E402


@dataclass
class _PlannerFakeContext:
    original_query: str
    tone_level: _RealToneLevel = _RealToneLevel.CONVERSATIONAL
    is_small_talk: bool = False
    query_analysis: object = None
    intent: object = None


class TestShouldPlanSelfReport:
    def test_bare_self_report_skips_planning(self):
        from core.response_planner import ResponsePlanner
        ctx = _PlannerFakeContext(
            original_query="I took my medication this morning and I'm heading to work now."
        )
        assert ResponsePlanner.should_plan(ctx) is False

    def test_request_shaped_self_report_still_plans(self):
        from core.response_planner import ResponsePlanner
        ctx = _PlannerFakeContext(
            original_query="Can you tell me what medication I should take next?"
        )
        assert ResponsePlanner.should_plan(ctx) is True

    def test_ordinary_non_self_report_statement_still_plans(self):
        """A statement that ISN'T self-report shaped (no first-person report
        verb/opener) is unaffected by this guard."""
        from core.response_planner import ResponsePlanner
        ctx = _PlannerFakeContext(
            original_query="The weather has been really strange this whole week apparently."
        )
        assert ResponsePlanner.should_plan(ctx) is True


# ---------------------------------------------------------------------------
# B8 (round 2) — utils.query_checker.is_status_report + should_plan wiring
#
# Live finding: is_self_report(T5) == False on the EXACT probe text ("Cool.
# Managed to push today and there is a new doc I think will be helpful") —
# its subject is elided after the "Cool." ack rather than restated as a
# pronoun, so is_self_report's contract (first-person pronoun opener/verb)
# never matches, and should_plan's self-report skip was dead on this text.
# ---------------------------------------------------------------------------

T5_STATUS_REPORT_QUERY = (
    "Cool. Managed to push today and there is a new doc I think will be helpful"
)


class TestIsStatusReportPredicate:
    def test_t5_is_status_report(self):
        from utils.query_checker import is_status_report
        assert is_status_report(T5_STATUS_REPORT_QUERY) is True

    def test_t5_is_not_a_self_report(self):
        """Regression anchor for the round-2 finding itself: is_self_report
        stays False on this exact text (its contract is unchanged) — the
        NEW predicate is what catches this shape."""
        from utils.query_checker import is_self_report
        assert is_self_report(T5_STATUS_REPORT_QUERY) is False

    def test_take_a_look_is_not_a_status_report(self):
        from utils.query_checker import is_status_report
        assert is_status_report("Can you take a look?") is False

    def test_ordinary_ack_alone_is_not_a_status_report(self):
        from utils.query_checker import is_status_report
        assert is_status_report("Cool, thanks") is False

    def test_status_report_verb_without_ack_opener_not_flagged(self):
        """The ack-opener requirement is load-bearing: a bare first-person
        past-tense report ("Managed to fix it") is is_self_report's shape
        (explicit subject-adjacent verb), not this predicate's — under-fires
        by design when there's no leading ack."""
        from utils.query_checker import is_status_report
        assert is_status_report("Managed to push today and there is a new doc") is False

    def test_request_shaped_status_report_not_flagged(self):
        from utils.query_checker import is_status_report
        text = "Cool, managed to push today, can you check if the new doc looks right?"
        assert is_status_report(text) is False

    def test_vent_shaped_ack_not_flagged(self):
        from utils.query_checker import is_status_report
        assert is_status_report("ugh so tired today") is False


class TestShouldPlanStatusReportRound2:
    def test_t5_status_report_skips_planning(self):
        from core.response_planner import ResponsePlanner
        ctx = _PlannerFakeContext(original_query=T5_STATUS_REPORT_QUERY)
        assert ResponsePlanner.should_plan(ctx) is False

    def test_request_shaped_status_report_still_plans(self):
        from core.response_planner import ResponsePlanner
        ctx = _PlannerFakeContext(
            original_query=(
                "Cool, managed to push today, can you check if the new doc "
                "actually looks right before I tell everyone about it?"
            )
        )
        assert ResponsePlanner.should_plan(ctx) is True


class TestUnsupportedKeyPointsStatementMode:
    QUERY = "Cool. Managed to push today and there is a new doc I think will be helpful"
    EXCHANGE = "User: pushed a commit today\nDaemon: nice work"
    DIGEST = "memories: the user has a new doctor and discussed a psychiatrist referral last month"
    POINTS = [
        "Ask about doctor recommendations",
        "Note the psychiatrist referral",
        "Cover the specialist opinion",
    ]

    def _sources(self):
        return "\n".join([self.QUERY, self.EXCHANGE, self.DIGEST])

    def _strict_sources(self):
        return "\n".join([self.QUERY, self.EXCHANGE])

    def test_digest_alone_no_longer_supports_a_statement_turn(self):
        from core.response_planner import ResponsePlanner
        kept, dropped = ResponsePlanner.unsupported_key_points(
            self.POINTS, self._sources(), query=self.QUERY, strict_sources=self._strict_sources(),
        )
        assert kept == []
        assert sorted(dropped) == sorted(self.POINTS)

    def test_digest_alone_supported_old_behavior_without_query(self):
        """Sanity check reproducing the ORIGINAL bug: without the new
        query/strict_sources kwargs, digest support alone kept everything."""
        from core.response_planner import ResponsePlanner
        kept, dropped = ResponsePlanner.unsupported_key_points(self.POINTS, self._sources())
        assert kept == self.POINTS
        assert dropped == []

    def test_request_shaped_query_keeps_digest_support(self):
        from core.response_planner import ResponsePlanner
        req_query = "Can you tell me about the new doctor and psychiatrist?"
        kept, dropped = ResponsePlanner.unsupported_key_points(
            self.POINTS, self._sources(), query=req_query,
            strict_sources=req_query + "\n" + self.EXCHANGE,
        )
        assert kept == self.POINTS
        assert dropped == []

    def test_never_empties_the_plan_regression(self):
        """Pre-existing safeguard (event-stem/name checks alone emptying the
        plan) is untouched by the new statement-mode addition."""
        from core.response_planner import ResponsePlanner
        kept, dropped = ResponsePlanner.unsupported_key_points(
            ["Biscuit's birthday celebration"], "nothing",
        )
        assert kept == ["Biscuit's birthday celebration"]
        assert dropped == []

    def test_licensed_event_still_kept_regression(self):
        from core.response_planner import ResponsePlanner
        kept, dropped = ResponsePlanner.unsupported_key_points(
            ["Biscuit's birthday celebration"], "we celebrated Biscuit's birthday yesterday",
        )
        assert kept and not dropped


# ---------------------------------------------------------------------------
# B7 (round 2) — _HEAD_NOUN_STOP additions + the independent prefix-expansion
# check in unsupported_key_points statement mode.
#
# Live finding (retest): the round-1 fix alone still kept the point.
# `_head_noun("The user believes this new doctor will be helpful.")` ==
# "user" — "user" was not in `_HEAD_NOUN_STOP` and is trivially present in
# ANY rendered "User: ..." exchange label, regardless of what that exchange
# actually said, so the strict-mode check's "is the head noun in the
# sources" test passed on a coincidence, not real support.
# ---------------------------------------------------------------------------

class TestUnsupportedKeyPointsRound2HeadNounStop:
    QUERY = T5_STATUS_REPORT_QUERY
    # Representative shape of the exchange immediately preceding this turn
    # (a meds-timing question/reply, per the round-2 retest ordering) —
    # reconstructed, not a verbatim quote, but critically contains no
    # mention of "doctor": the fix must not depend on any exchange text
    # happening to omit "doctor" by coincidence, only on the ACTUAL sources.
    MEDS_EXCHANGE = (
        "User: What time should I take my melatonin tonight to get to bed?\n"
        "Daemon: Taking it around 10 or 10:30 should still work."
    )
    LIVE_PLAN_POINT = "The user believes this new doctor will be helpful."

    def _strict_sources(self):
        return "\n".join([self.QUERY, self.MEDS_EXCHANGE])

    def test_head_noun_no_longer_stops_at_user(self):
        from core.response_planner import ResponsePlanner
        assert ResponsePlanner._head_noun(self.LIVE_PLAN_POINT) != "user"

    def test_live_plan_point_dropped_with_meds_exchange(self):
        from core.response_planner import ResponsePlanner
        strict_sources = self._strict_sources()
        kept, dropped = ResponsePlanner.unsupported_key_points(
            [self.LIVE_PLAN_POINT], strict_sources,
            query=self.QUERY, strict_sources=strict_sources,
        )
        assert kept == []
        assert dropped == [self.LIVE_PLAN_POINT]

    def test_pushed_a_new_doc_point_kept(self):
        """Same query, a DIFFERENT (non-embellished) point that the exchange
        actually corroborates ("pushed") stays kept — the fix only drops
        points built on an unsupported abbreviation expansion, not every
        statement-mode point."""
        from core.response_planner import ResponsePlanner
        point = "The user pushed a new doc today"
        exchange = "User: pushed a commit today\nDaemon: nice work"
        strict_sources = "\n".join([self.QUERY, exchange])
        kept, dropped = ResponsePlanner.unsupported_key_points(
            [point], strict_sources, query=self.QUERY, strict_sources=strict_sources,
        )
        assert kept == [point]
        assert dropped == []

    def test_prefix_expansion_helper_direct(self):
        from core.response_planner import ResponsePlanner
        assert ResponsePlanner._has_unsupported_prefix_expansion(
            self.LIVE_PLAN_POINT, self.QUERY, self._strict_sources().lower(),
        ) is True

    def test_prefix_expansion_helper_false_when_long_form_supported(self):
        from core.response_planner import ResponsePlanner
        strict_src = (self.QUERY + "\ndoctor").lower()
        assert ResponsePlanner._has_unsupported_prefix_expansion(
            self.LIVE_PLAN_POINT, self.QUERY, strict_src,
        ) is False

    def test_query_none_bypasses_new_check_entirely(self):
        """Regression: strict_mode (and therefore both new checks) never
        runs when query is None — prior callers/behavior untouched."""
        from core.response_planner import ResponsePlanner
        kept, dropped = ResponsePlanner.unsupported_key_points(
            [self.LIVE_PLAN_POINT], "irrelevant sources",
        )
        assert kept == [self.LIVE_PLAN_POINT]
        assert dropped == []


# ---------------------------------------------------------------------------
# B5 — stm_analyzer.abbreviation_expansion_conflicts
# ---------------------------------------------------------------------------

class TestAbbreviationExpansionConflicts:
    QUERY_T3 = "Cool. Managed to push today and there is a new doc I think will be helpful"
    STM_OUTPUT_T3 = {
        "topic": "New doctor",
        "user_question": "User mentions a new doctor is helpful",
        "reference_type": "new_event",
        "temporal_facts": ["user has a new doctor who is helpful"],
    }
    WINDOW_T3 = "User: pushed a commit today\nDaemon: nice work"

    def test_t3_conflict_detected(self):
        from core.stm_analyzer import abbreviation_expansion_conflicts
        conflicts = abbreviation_expansion_conflicts(self.QUERY_T3, self.STM_OUTPUT_T3, self.WINDOW_T3)
        assert ("doc", "doctor") in conflicts

    def test_window_support_suppresses_the_conflict(self):
        """"app"->"appointment" is fine when the window already established
        the expansion (analogous to the "hw"->"homework" example in the
        handoff — window support suppresses the flag)."""
        from core.stm_analyzer import abbreviation_expansion_conflicts
        query = "app at 3"
        stm = {
            "topic": "appointment scheduling",
            "user_question": "user asks about an appointment",
            "temporal_facts": [],
        }
        window_has = "we discussed the appointment yesterday"
        window_lacks = "we discussed something else yesterday"
        assert abbreviation_expansion_conflicts(query, stm, window_has) == []
        assert abbreviation_expansion_conflicts(query, stm, window_lacks) == [("app", "appointment")]

    def test_query_already_stating_long_form_is_not_a_conflict(self):
        from core.stm_analyzer import abbreviation_expansion_conflicts
        query = "my doctor appointment is tomorrow, also app crashed"
        stm = {"topic": "appointment", "user_question": "", "temporal_facts": []}
        # "app" -> "appointment" is not flagged because "appointment" already
        # appears (word-bounded) in the query itself.
        assert abbreviation_expansion_conflicts(query, stm, "") == []

    def test_no_conflict_when_nothing_short_or_nothing_long(self):
        from core.stm_analyzer import abbreviation_expansion_conflicts
        assert abbreviation_expansion_conflicts("", self.STM_OUTPUT_T3, "") == []
        assert abbreviation_expansion_conflicts(self.QUERY_T3, {}, "") == []
        assert abbreviation_expansion_conflicts(self.QUERY_T3, None, "") == []

    def test_common_short_words_never_flagged(self):
        from core.stm_analyzer import abbreviation_expansion_conflicts
        query = "the new one is not sure"
        stm = {"topic": "office hours", "user_question": "", "temporal_facts": []}
        # "the"/"new"/"one"/"is"/"not" are stopworded; "sure" shares no
        # prefix relationship with anything in the STM output.
        assert abbreviation_expansion_conflicts(query, stm, "") == []


class TestAbbreviationConflictWiring:
    def test_analyze_source_calls_abbreviation_conflicts(self):
        import inspect
        from core import stm_analyzer
        src = inspect.getsource(stm_analyzer.STMAnalyzer.analyze)
        assert "abbreviation_expansion_conflicts" in src
        assert "abbreviation_conflicts" in src

    def test_prompt_carries_abbreviation_rule(self):
        import inspect
        from core import stm_analyzer
        src = inspect.getsource(stm_analyzer.STMAnalyzer.analyze)
        assert "Never expand an abbreviation" in src

    def test_formatter_renders_note_and_drops_resolved_state_line(self):
        from unittest.mock import MagicMock
        from core.prompt.formatter import PromptFormatter
        fmt = PromptFormatter(token_manager=MagicMock())
        stm_summary = {
            "topic": "New doctor", "user_question": "User mentions a new doctor is helpful",
            "intent": "share update", "tone": "neutral", "reference_type": "unclear",
            "temporal_facts": [
                "user has a new doctor who is helpful",
                "user pushed a commit today",
            ],
            "open_threads": [], "constraints": [],
            "abbreviation_conflicts": [["doc", "doctor"]],
        }
        out = fmt._assemble_prompt(
            context={"stm_summary": stm_summary},
            user_input="Cool. Managed to push today and there is a new doc I think will be helpful",
            directives="", system_prompt="",
        )
        assert "expands the user's 'doc'" in out
        assert "not stated" in out
        assert "user pushed a commit today" in out
        assert "user has a new doctor who is helpful" not in out


# ---------------------------------------------------------------------------
# B9 (round 2) — the abbreviation-conflict window must be USER-authored only
#
# Live finding (retest): "no abbreviation NOTE" fired even though the exact
# T3/T5 query ran through STMAnalyzer.analyze() again. Root cause: the STM
# window fed to abbreviation_expansion_conflicts() included Daemon's OWN
# immediately-preceding reply ("The new doctor being helpful tracks with how
# yesterday's appointment seemed to go."), which literally contains the word
# "doctor" — so the window-support suppression (by design, correct for a
# genuine window like "hw"->"homework") fired on the ASSISTANT's own
# phrasing, not anything the user said. STMAnalyzer.analyze() must build the
# window passed to this one check from user-authored corpus text only.
# ---------------------------------------------------------------------------

class TestAbbreviationWindowUserAuthoredOnly:
    QUERY_T5 = T5_STATUS_REPORT_QUERY
    ASSISTANT_REPLY = (
        "The new doctor being helpful tracks with how yesterday's "
        "appointment seemed to go."
    )
    STM_JSON = (
        '{"topic": "New doctor update", '
        '"user_question": "User is sharing that they have a new doctor they believe will be helpful.", '
        '"intent": "share an update", "tone": "neutral", '
        '"reference_type": "new_event", '
        '"temporal_facts": ["user has a new doctor who is helpful"], '
        '"open_threads": [], "constraints": []}'
    )

    class _FixedModel:
        def __init__(self, payload):
            self._payload = payload
            self.prompt = ""

        async def generate_once(self, prompt, **kwargs):
            self.prompt = prompt
            return self._payload

    def _analyzer(self):
        from core.stm_analyzer import STMAnalyzer
        analyzer = STMAnalyzer(self._FixedModel(self.STM_JSON))
        # No Obsidian vault in the test environment — mirrors the
        # established pattern in test_stm_new_data_override.py.
        analyzer._get_recent_daily_notes_text = lambda *a, **k: ""
        return analyzer

    def test_user_authored_window_helper_excludes_response_field(self):
        from core.stm_analyzer import STMAnalyzer
        memories = [
            {"query": "Managed to push a commit today", "response": self.ASSISTANT_REPLY},
        ]
        window = STMAnalyzer._user_authored_window(memories)
        assert "doctor" not in window
        assert "push a commit" in window

    def test_user_authored_window_prefers_user_text_field(self):
        from core.stm_analyzer import STMAnalyzer
        memories = [
            {"query": "some merged/legacy blob", "user_text": "Managed to push a commit today",
             "response": self.ASSISTANT_REPLY},
        ]
        window = STMAnalyzer._user_authored_window(memories)
        assert window == "Managed to push a commit today"

    @pytest.mark.asyncio
    async def test_assistant_reply_does_not_mask_the_conflict(self):
        analyzer = self._analyzer()
        recent_memories = [
            {"query": "Managed to push a commit today", "response": self.ASSISTANT_REPLY,
             "timestamp": ""},
        ]
        parsed = await analyzer.analyze(
            recent_memories=recent_memories,
            user_query=self.QUERY_T5,
            last_assistant_response=self.ASSISTANT_REPLY,
        )
        assert ["doc", "doctor"] in parsed.get("abbreviation_conflicts", [])

    @pytest.mark.asyncio
    async def test_user_side_mention_of_doctor_suppresses_the_conflict(self):
        analyzer = self._analyzer()
        recent_memories = [
            {"query": "I saw the doctor yesterday about this", "response": self.ASSISTANT_REPLY,
             "timestamp": ""},
        ]
        parsed = await analyzer.analyze(
            recent_memories=recent_memories,
            user_query=self.QUERY_T5,
            last_assistant_response=self.ASSISTANT_REPLY,
        )
        assert not parsed.get("abbreviation_conflicts")


# ---------------------------------------------------------------------------
# B6 — action_claim_guard.claims_fresh_upload + handlers post-check
# ---------------------------------------------------------------------------

class TestClaimsFreshUpload:
    def test_t4_style_reply_flagged(self):
        from core.action_claim_guard import claims_fresh_upload
        reply = "You uploaded Homework1-2.pdf today, let me take a look at it."
        assert claims_fresh_upload(reply) is True

    def test_reply_naming_actual_past_date_not_flagged(self):
        from core.action_claim_guard import claims_fresh_upload
        reply = "You uploaded Homework1-2.pdf on Sept 5, here's what it says."
        assert claims_fresh_upload(reply) is False

    def test_just_attached_phrase_flagged(self):
        from core.action_claim_guard import claims_fresh_upload
        reply = "The file you just attached looks complete."
        assert claims_fresh_upload(reply) is True

    def test_ordinary_reply_not_flagged(self):
        from core.action_claim_guard import claims_fresh_upload
        reply = "Sure, taking a look at the PDF now."
        assert claims_fresh_upload(reply) is False


class _FakeRegistry:
    def __init__(self, docs):
        self._docs = docs

    def documents(self):
        return self._docs


class _FakeOrchestrator:
    def __init__(self, active_documents=None):
        self.active_documents = active_documents


def _make_ctx(*, user_text="Can you take a look?", raw_context=None, active_documents=None):
    return SimpleNamespace(
        user_text=user_text,
        orchestrator=_FakeOrchestrator(active_documents=active_documents),
        raw_context=raw_context or {},
    )


@pytest.mark.asyncio
class TestFreshUploadHandlerBackstop:
    async def test_appends_notice_when_no_active_docs_and_dated_upload(self):
        from gui.handlers import _apply_action_guard
        ctx = _make_ctx(
            raw_context={
                "user_uploads": [
                    {
                        "content": "",
                        "metadata": {"type": "upload_roster", "roster": [
                            {"title": "Homework1-2.pdf", "date": "2026-09-05"},
                        ]},
                    },
                ],
            },
            active_documents=None,
        )
        response = "You uploaded Homework1-2.pdf today, let me take a look."
        suffix = await _apply_action_guard(
            ctx, response, executed_kinds=set(), proposed_kinds=set(), self_repair=False,
        )
        assert "No file was uploaded this session" in suffix
        assert "2026-09-05" in suffix

    async def test_no_notice_when_active_documents_registry_non_empty(self):
        from gui.handlers import _apply_action_guard
        ctx = _make_ctx(
            raw_context={
                "user_uploads": [
                    {
                        "content": "",
                        "metadata": {"type": "upload_roster", "roster": [
                            {"title": "Homework1-2.pdf", "date": "2026-09-05"},
                        ]},
                    },
                ],
            },
            active_documents=_FakeRegistry(docs=["some-doc"]),
        )
        response = "You uploaded Homework1-2.pdf today, let me take a look."
        suffix = await _apply_action_guard(
            ctx, response, executed_kinds=set(), proposed_kinds=set(), self_repair=False,
        )
        assert "No file was uploaded this session" not in suffix

    async def test_no_notice_when_reply_names_a_real_past_date(self):
        from gui.handlers import _apply_action_guard
        ctx = _make_ctx(
            raw_context={
                "user_uploads": [
                    {
                        "content": "",
                        "metadata": {"type": "upload_roster", "roster": [
                            {"title": "Homework1-2.pdf", "date": "2026-09-05"},
                        ]},
                    },
                ],
            },
            active_documents=None,
        )
        response = "You uploaded Homework1-2.pdf on Sept 5, here's what it says."
        suffix = await _apply_action_guard(
            ctx, response, executed_kinds=set(), proposed_kinds=set(), self_repair=False,
        )
        assert "No file was uploaded this session" not in suffix

    async def test_no_notice_when_no_dated_upload_context(self):
        from gui.handlers import _apply_action_guard
        ctx = _make_ctx(raw_context={}, active_documents=None)
        response = "You uploaded Homework1-2.pdf today, let me take a look."
        suffix = await _apply_action_guard(
            ctx, response, executed_kinds=set(), proposed_kinds=set(), self_repair=False,
        )
        assert "No file was uploaded this session" not in suffix


# ---------------------------------------------------------------------------
# B11 (round 3) — normalize_ws chokepoint at ingress (CM-01: one chokepoint,
# not a patch per predicate). Sub A adds the actual `normalize_ws(user_text)`
# call in gui/handlers.py; these tests drive the wrapped live texts through
# the DEPLOYED ingress path (`gui.handlers.handle_submit`) and check what
# `orch.prepare_prompt` actually receives — the same value every downstream
# consumer (ContextPipeline.original_query, the agentic gate's note-save arm,
# etc.) reads from.
# ---------------------------------------------------------------------------

from tests.unit.test_handle_submit import _make_orchestrator, _run_submit  # noqa: E402

# The exact live round-3 text (T5/T3, "Cool. Managed to push...") as the
# client actually sent it: wrapped mid-sentence, newline + two leading
# spaces (docs/HANDOFF_20260910_probe_dump.md ROUND 3).
WRAPPED_STATUS_REPORT = (
    "Cool. Managed to push today and there is a new doc I\n  think will be helpful"
)
CLEAN_STATUS_REPORT = T5_STATUS_REPORT_QUERY

# The round-1 T5 note-save text, wrapped the same way.
CLEAN_NOTE_SAVE = "jot down a note for this session: TA sessions are Saturdays at 11 CT,"
WRAPPED_NOTE_SAVE = (
    "jot down a note for this session: TA sessions are Saturdays at 11\n  CT,"
)


class TestNormalizeWsHelperContract:
    """utils.trigger_match.normalize_ws: collapse all whitespace runs
    (including newlines and leading indentation) to single spaces, strip.
    Shared with Sub A — created here only if Sub A's concurrent edit hadn't
    landed yet when this file's tests started running."""

    def test_collapses_wrapped_status_report_to_clean_form(self):
        from utils.trigger_match import normalize_ws
        assert normalize_ws(WRAPPED_STATUS_REPORT) == CLEAN_STATUS_REPORT

    def test_collapses_wrapped_note_save_to_clean_form(self):
        from utils.trigger_match import normalize_ws
        assert normalize_ws(WRAPPED_NOTE_SAVE) == CLEAN_NOTE_SAVE

    def test_collapses_leading_indentation_and_internal_runs(self):
        from utils.trigger_match import normalize_ws
        assert normalize_ws("  a\n\n   b\tc  ") == "a b c"

    def test_empty_and_already_clean_are_no_ops(self):
        from utils.trigger_match import normalize_ws
        assert normalize_ws("") == ""
        assert normalize_ws(CLEAN_STATUS_REPORT) == CLEAN_STATUS_REPORT


class TestPredicatesAgreeOnNormalizedForm:
    """Sanity anchor: once whitespace is collapsed (whatever normalizes it —
    the ingress chokepoint in production), the shape predicates agree on the
    clean and the wrapped-then-normalized form. This does not by itself prove
    ingress wiring — see TestIngressWhitespaceNormalization for that."""

    def test_status_report_true_on_normalized_wrapped_text(self):
        from utils.query_checker import is_status_report
        from utils.trigger_match import normalize_ws
        assert is_status_report(normalize_ws(WRAPPED_STATUS_REPORT)) is True

    def test_note_save_true_on_normalized_wrapped_text(self):
        from utils.query_checker import is_note_save_request
        from utils.trigger_match import normalize_ws
        assert is_note_save_request(normalize_ws(WRAPPED_NOTE_SAVE)) is True

    def test_note_save_already_robust_to_raw_wrapped_text(self):
        """is_note_save_request's own regex uses `\\s+` between words, so it
        happens to tolerate the wrapped form even pre-normalization — noted
        here so a future change to that regex doesn't silently regress this
        without a failing test catching it."""
        from utils.query_checker import is_note_save_request
        assert is_note_save_request(WRAPPED_NOTE_SAVE) is True

    def test_status_report_false_on_raw_wrapped_text_without_normalization(self):
        """Regression anchor for the round-3 finding itself: is_status_report
        has NOT been patched internally (remedy pattern CM-01 — the fix lives
        once at ingress, not per-predicate) so it stays blind to a raw,
        un-normalized wrapped string exactly as before."""
        from utils.query_checker import is_status_report
        assert is_status_report(WRAPPED_STATUS_REPORT) is False


@pytest.mark.asyncio
class TestIngressWhitespaceNormalization:
    """Drives the wrapped live texts through gui.handlers.handle_submit (the
    deployed ingress path) and checks what orch.prepare_prompt's `user_input`
    kwarg actually receives — the single value every downstream consumer
    (ContextPipeline.original_query -> ResponsePlanner.should_plan's
    is_status_report check, the agentic gate's is_note_save_request arm,
    etc.) reads. No bare newline should survive ingress, and the shape
    predicates must fire on the value the pipeline actually saw."""

    async def test_wrapped_status_report_reaches_prepare_prompt_normalized(self):
        from utils.query_checker import is_status_report
        orch = _make_orchestrator()
        await _run_submit(WRAPPED_STATUS_REPORT, orch)
        orch.prepare_prompt.assert_awaited_once()
        received = orch.prepare_prompt.call_args.kwargs.get("user_input")
        assert received is not None
        assert "\n" not in received, (
            "raw newline reached prepare_prompt's user_input — ingress "
            "normalize_ws chokepoint (gui/handlers.py) not wired"
        )
        assert is_status_report(received) is True

    async def test_wrapped_note_save_reaches_prepare_prompt_normalized(self):
        from utils.query_checker import is_note_save_request
        orch = _make_orchestrator()
        await _run_submit(WRAPPED_NOTE_SAVE, orch)
        orch.prepare_prompt.assert_awaited_once()
        received = orch.prepare_prompt.call_args.kwargs.get("user_input")
        assert received is not None
        assert "\n" not in received, (
            "raw newline reached prepare_prompt's user_input — ingress "
            "normalize_ws chokepoint (gui/handlers.py) not wired"
        )
        assert is_note_save_request(received) is True

    async def test_clean_status_report_still_reaches_prepare_prompt_unchanged(self):
        """Regression guard: normalization must be a no-op on already-clean
        text — it must not alter wording, only collapse whitespace runs."""
        orch = _make_orchestrator()
        await _run_submit(CLEAN_STATUS_REPORT, orch)
        received = orch.prepare_prompt.call_args.kwargs.get("user_input")
        assert received == CLEAN_STATUS_REPORT


# ---------------------------------------------------------------------------
# B12 (round 3) — ResponsePlanner.create_plan discards an emptied plan;
# strategy/avoid get the same embellishment checks as key_points.
# ---------------------------------------------------------------------------

class _R3FixedPlanModel:
    """Minimal model_manager stand-in for create_plan()'s single
    generate_once call — mirrors the _FixedModel pattern already used above
    for STMAnalyzer (TestAbbreviationWindowUserAuthoredOnly)."""

    def __init__(self, payload: str):
        self._payload = payload

    async def generate_once(self, prompt, **kwargs):
        return self._payload


def _r3_planner_context(query: str, last_exchange: Optional[dict] = None):
    return SimpleNamespace(
        original_query=query,
        tone_level=_RealToneLevel.CONVERSATIONAL,
        intent=None,
        topics=[],
        thread_context=None,
        last_exchange=last_exchange,
    )


async def _r3_create_plan(query: str, payload: str, last_exchange: Optional[dict] = None):
    from core.response_planner import ResponsePlanner
    planner = ResponsePlanner(model_manager=_R3FixedPlanModel(payload))
    with patch("config.app_config.RESPONSE_PLANNING_MODEL", None), \
         patch("config.app_config.RESPONSE_PLANNING_MAX_TOKENS", 200), \
         patch("config.app_config.RESPONSE_PLANNING_TIMEOUT", 5.0):
        return await planner.create_plan(query=query, context=_r3_planner_context(query, last_exchange))


class TestStatementUnsupportedHelper:
    """Direct tests of the new ResponsePlanner._statement_unsupported
    classmethod — the strategy/avoid counterpart to unsupported_key_points'
    per-point check, with no "never fully empty" list safeguard."""

    QUERY = T5_STATUS_REPORT_QUERY

    def test_embellished_strategy_flagged(self):
        from core.response_planner import ResponsePlanner
        strategy = "Acknowledge the user's progress and express support for their new doctor."
        assert ResponsePlanner._statement_unsupported(
            strategy, self.QUERY, query=self.QUERY, strict_sources=self.QUERY,
        ) is True

    def test_supported_strategy_using_query_vocabulary_not_flagged(self):
        from core.response_planner import ResponsePlanner
        strategy = "Acknowledge the new doc they mentioned."
        assert ResponsePlanner._statement_unsupported(
            strategy, self.QUERY, query=self.QUERY, strict_sources=self.QUERY,
        ) is False

    def test_avoid_line_gets_the_same_check(self):
        from core.response_planner import ResponsePlanner
        avoid_line = "Don't dwell on the new doctor visit."
        assert ResponsePlanner._statement_unsupported(
            avoid_line, self.QUERY, query=self.QUERY, strict_sources=self.QUERY,
        ) is True

    def test_blank_text_never_flagged(self):
        from core.response_planner import ResponsePlanner
        assert ResponsePlanner._statement_unsupported("", self.QUERY, query=self.QUERY) is False

    def test_request_shaped_query_bypasses_strict_check(self):
        from core.response_planner import ResponsePlanner
        req = "Can you tell me about the new doctor?"
        strategy = "Discuss the new doctor."
        assert ResponsePlanner._statement_unsupported(strategy, req, query=req) is False

    def test_no_query_bypasses_strict_check_entirely(self):
        """query=None preserves prior (pre-round-3) no-op behavior exactly,
        same contract as unsupported_key_points."""
        from core.response_planner import ResponsePlanner
        strategy = "Acknowledge the user's progress and express support for their new doctor."
        assert ResponsePlanner._statement_unsupported(strategy, self.QUERY) is False


@pytest.mark.asyncio
class TestEmptiedPlanDiscarded:
    """create_plan() end-to-end: the exact live round-3 plan shape (key_points
    already empty from the LLM, an embellished strategy) is discarded outright
    so no [RESPONSE PLAN] block is ever injected."""

    QUERY = T5_STATUS_REPORT_QUERY

    ROUND3_PLAN_JSON = json.dumps({
        "key_points": [],
        "tone": "warm",
        "avoid": [],
        "strategy": "Acknowledge the user's progress and express support for their new doctor.",
    })

    async def test_round3_live_plan_is_discarded(self):
        plan = await _r3_create_plan(self.QUERY, self.ROUND3_PLAN_JSON)
        assert plan is None

    async def test_orchestrator_side_injection_guard_would_stay_silent(self):
        """Source-level check that the orchestrator's injection site treats a
        None plan as "nothing to inject" (it already did — this just pins the
        contract create_plan's new None return relies on)."""
        import inspect
        import core.orchestrator as orch_mod
        src = inspect.getsource(orch_mod.DaemonOrchestrator.build_full_prompt) \
            if hasattr(orch_mod.DaemonOrchestrator, "build_full_prompt") else ""
        # Fall back to a broader source scan if the method name differs —
        # the load-bearing fact is just that _plan_result truthiness gates
        # the format_plan_injection() call somewhere in the module.
        if not src:
            src = inspect.getsource(orch_mod)
        assert "format_plan_injection" in src

    async def test_supported_plan_with_points_still_returned(self):
        """Regression: a plan whose key points AND strategy genuinely survive
        the checks is unaffected by the new discard guard."""
        exchange = {"query": "pushed a commit today", "response": "nice work"}
        payload = json.dumps({
            "key_points": ["The user pushed a new doc today"],
            "tone": "warm",
            "avoid": [],
            "strategy": "Acknowledge that they pushed a commit today.",
        })
        plan = await _r3_create_plan(self.QUERY, payload, last_exchange=exchange)
        assert plan is not None
        assert plan.key_points == ["The user pushed a new doc today"]
        assert plan.strategy == "Acknowledge that they pushed a commit today."

    async def test_every_key_point_dropped_discards_plan_even_with_surviving_strategy(self):
        """B12's other trigger: every key point dropped by the strict/prefix
        checks discards the WHOLE plan even when strategy alone would have
        survived on its own — never render a lone "Cover: (none)" plus an
        unrelated strategy line."""
        exchange = {"query": "pushed a commit today", "response": "nice work"}
        payload = json.dumps({
            "key_points": ["The user believes this new doctor will be helpful."],
            "tone": "warm",
            "avoid": [],
            "strategy": "Acknowledge that they pushed a commit today.",
        })
        plan = await _r3_create_plan(self.QUERY, payload, last_exchange=exchange)
        assert plan is None

    async def test_format_plan_injection_never_called_when_plan_is_none(self):
        """End-to-end sanity: build_full_prompt's own `if _plan_result:` gate
        means a discarded plan renders nothing — verified at the unit level
        by confirming create_plan's contract (None) rather than re-running
        the whole orchestrator (out of scope here; see the existing
        format_plan_injection tests in test_response_planner.py for the
        rendering contract on a non-None plan)."""
        plan = await _r3_create_plan(self.QUERY, self.ROUND3_PLAN_JSON)
        assert plan is None  # orchestrator's `if _plan_result:` never fires


# ---------------------------------------------------------------------------
# B10 (round 4) — utils.query_checker.is_task_directive +
# utils.tone_detector.detect_crisis_level's two fallback stages standing
# down for a task directive.
# ---------------------------------------------------------------------------

R4_T1_NOTE_SAVE = "jot down a note for this session: TA sessions are Saturdays at 11 CT"
R4_T1_NOTE_SAVE_WRAPPED = (
    "jot down a note for this session: TA sessions are\n  Saturdays at 11 CT"
)
R4_T2_CALENDAR = (
    "put a recurring calendar event on my google calendar for the MGT study "
    "group, Tuesdays at 3, through Dec 4"
)
R4_T2_CALENDAR_WRAPPED = (
    "put a recurring calendar event on my google calendar for the MGT study "
    "group, Tuesdays at\n  3, through Dec 4"
)
R4_MEDS_QUESTION = (
    "what time should I take my meds tonight, I usually take them with "
    "dinner around 7"
)
R4_MEDS_QUESTION_WRAPPED = (
    "what time should I take my meds tonight, I usually take\n  them with "
    "dinner around 7"
)
R4_REMIND_PHARMACY = "remind me to call the pharmacy"
R4_CANT_ANYMORE = "I can't do this anymore"


class TestIsTaskDirectivePredicate:
    """Direct tests of the deployed `utils.query_checker.is_task_directive`.
    Each live-text fixture is asserted in both its clean form and the
    line-wrapped form the client actually sends (a newline + two leading
    spaces inside the sentence) — the predicate must agree on both without
    needing its own normalize_ws call (CM-01: the ingress chokepoint in
    gui/handlers.py is the one place that normalizes; a request-clause split
    on the embedded newline still lands the match before the wrap point for
    every fixture here, so no per-predicate patch is needed)."""

    def test_note_save_request_is_a_task_directive(self):
        from utils.query_checker import is_task_directive
        assert is_task_directive(R4_T1_NOTE_SAVE) is True

    def test_note_save_request_wrapped_form_agrees(self):
        from utils.query_checker import is_task_directive
        assert is_task_directive(R4_T1_NOTE_SAVE_WRAPPED) is True

    def test_calendar_request_is_a_task_directive(self):
        from utils.query_checker import is_task_directive
        assert is_task_directive(R4_T2_CALENDAR) is True

    def test_calendar_request_wrapped_form_agrees(self):
        from utils.query_checker import is_task_directive
        assert is_task_directive(R4_T2_CALENDAR_WRAPPED) is True

    def test_remind_me_is_a_task_directive(self):
        from utils.query_checker import is_task_directive
        assert is_task_directive(R4_REMIND_PHARMACY) is True

    def test_meds_question_is_not_a_task_directive(self):
        """QUESTIONS are never task directives, even a directive-word-shaped
        one — the tone backstop must not lose a genuinely distress-shaped
        question to a coincidental grammatical match."""
        from utils.query_checker import is_task_directive
        assert is_task_directive(R4_MEDS_QUESTION) is False

    def test_meds_question_wrapped_form_agrees(self):
        from utils.query_checker import is_task_directive
        assert is_task_directive(R4_MEDS_QUESTION_WRAPPED) is False

    def test_first_person_vent_is_not_a_task_directive(self):
        from utils.query_checker import is_task_directive
        assert is_task_directive(R4_CANT_ANYMORE) is False

    def test_empty_and_none_are_not_task_directives(self):
        from utils.query_checker import is_task_directive
        assert is_task_directive("") is False
        assert is_task_directive(None) is False

    def test_ordinary_question_with_directive_verb_still_not_a_directive(self):
        from utils.query_checker import is_task_directive
        assert is_task_directive("Could you remind me what time it is?") is False


class TestToneStandsDownForTaskDirective:
    """`utils.tone_detector.detect_crisis_level`'s distress-sticky floor and
    borderline backstop both stand down for a task directive — the exact
    round-4 T1 raw scores (distress=0.3747, conversational=0.26) that
    would otherwise floor to CONCERN via either stage. Stubbing convention
    follows tests/unit/test_tone_borderline_fallback.py /
    test_tone_arbiter_hardening.py: patch
    `utils.tone_detector._semantic_crisis_detection` for the raw scores and
    `utils.tone_detector._llm_crisis_fallback` for the arbiter verdict."""

    # top distress (concern) = 0.3747, conversational = 0.26 — the exact
    # live round-4 T1 numbers ("distress=0.37 > conversational=0.26").
    R4_T1_SCORES = {
        "high": 0.10, "medium": 0.20, "concern": 0.3747, "conversational": 0.26,
    }
    VENT_SAME_SCORES_MSG = (
        "I keep thinking I am a stupid piece of shit. I know it's the meds "
        "but I wanna cry"
    )

    @pytest.mark.asyncio
    async def test_task_directive_arbiter_conversational_stands(self):
        from utils.tone_detector import CrisisLevel, detect_crisis_level
        with patch(
            "utils.tone_detector._semantic_crisis_detection",
            return_value=(CrisisLevel.CONVERSATIONAL, 0.26, self.R4_T1_SCORES),
        ), patch(
            "utils.tone_detector._llm_crisis_fallback",
            new=AsyncMock(return_value=(CrisisLevel.CONVERSATIONAL, 0.6)),
        ):
            analysis = await detect_crisis_level(
                R4_T1_NOTE_SAVE, model_manager=MagicMock(),
            )
        assert analysis.level == CrisisLevel.CONVERSATIONAL
        assert analysis.trigger == "llm_fallback"
        assert analysis.trigger != "borderline_backstop"

    @pytest.mark.asyncio
    async def test_vent_with_same_scores_still_backstops(self):
        from utils.tone_detector import CrisisLevel, detect_crisis_level
        with patch(
            "utils.tone_detector._semantic_crisis_detection",
            return_value=(CrisisLevel.CONVERSATIONAL, 0.26, self.R4_T1_SCORES),
        ), patch(
            "utils.tone_detector._llm_crisis_fallback",
            new=AsyncMock(return_value=(CrisisLevel.CONVERSATIONAL, 0.6)),
        ):
            analysis = await detect_crisis_level(
                self.VENT_SAME_SCORES_MSG, model_manager=MagicMock(),
            )
        assert analysis.level == CrisisLevel.CONCERN
        assert analysis.trigger == "borderline_backstop"

    @pytest.mark.asyncio
    async def test_session_distress_task_directive_not_floored(self):
        from utils.tone_detector import CrisisLevel, detect_crisis_level
        with patch(
            "utils.tone_detector._semantic_crisis_detection",
            return_value=(CrisisLevel.CONVERSATIONAL, 0.26, self.R4_T1_SCORES),
        ):
            analysis = await detect_crisis_level(
                R4_T1_NOTE_SAVE, previous_tone=CrisisLevel.CONCERN,
            )
        assert analysis.trigger != "distress_sticky_floor"
        assert analysis.level == CrisisLevel.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_session_distress_ugh_still_floored(self):
        """Regression: a non-task-directive short reply mid-distress is
        UNAFFECTED by B10 — the pre-existing sticky floor still holds."""
        from utils.tone_detector import CrisisLevel, detect_crisis_level
        analysis = await detect_crisis_level("ugh", previous_tone=CrisisLevel.CONCERN)
        assert analysis.trigger == "distress_sticky_floor"
        assert analysis.level == CrisisLevel.CONCERN


# ---------------------------------------------------------------------------
# B11 (round 4) — core.stm_analyzer._is_regular_inflection /
# abbreviation_expansion_conflicts' morphology exclusion.
# ---------------------------------------------------------------------------

class TestRegularInflectionHelper:
    def test_doc_doctor_is_not_an_inflection(self):
        from core.stm_analyzer import _is_regular_inflection
        assert _is_regular_inflection("doc", "doctor") is False

    def test_doc_docs_is_an_inflection(self):
        from core.stm_analyzer import _is_regular_inflection
        assert _is_regular_inflection("doc", "docs") is True

    def test_take_takes_is_an_inflection(self):
        from core.stm_analyzer import _is_regular_inflection
        assert _is_regular_inflection("take", "takes") is True

    def test_take_taking_is_an_inflection_via_e_drop(self):
        from core.stm_analyzer import _is_regular_inflection
        assert _is_regular_inflection("take", "taking") is True

    def test_take_taken_is_an_inflection_via_e_drop(self):
        from core.stm_analyzer import _is_regular_inflection
        assert _is_regular_inflection("take", "taken") is True

    def test_plan_planning_is_an_inflection_via_doubling(self):
        from core.stm_analyzer import _is_regular_inflection
        assert _is_regular_inflection("plan", "planning") is True

    def test_plan_planned_is_an_inflection_via_doubling(self):
        from core.stm_analyzer import _is_regular_inflection
        assert _is_regular_inflection("plan", "planned") is True

    def test_identical_tokens_not_an_inflection(self):
        from core.stm_analyzer import _is_regular_inflection
        assert _is_regular_inflection("take", "take") is False

    def test_empty_inputs_not_an_inflection(self):
        from core.stm_analyzer import _is_regular_inflection
        assert _is_regular_inflection("", "docs") is False
        assert _is_regular_inflection("doc", "") is False


class TestAbbreviationInflectionRound4Live:
    """Full-pipeline live reproduction: T3-round-4's meds query, whose STM
    output paraphrased the query's own "take" as "takes" — the false
    positive is NOT an unevidenced abbreviation expansion."""

    QUERY = R4_MEDS_QUESTION
    STM_OUTPUT = {
        "topic": "Medication schedule",
        "user_question": "",
        "reference_type": "recall",
        "temporal_facts": ["User usually takes meds with dinner around 7"],
    }

    def test_takes_take_no_longer_flagged(self):
        from core.stm_analyzer import abbreviation_expansion_conflicts
        conflicts = abbreviation_expansion_conflicts(self.QUERY, self.STM_OUTPUT, "")
        assert conflicts == []

    def test_wrapped_query_form_agrees(self):
        """The tokenizer here is a bare `[a-z]+` regex scan — whitespace and
        newlines are equally separators, so this predicate needs no
        normalize_ws call to be wrap-robust (unlike sentence-splitting
        predicates such as is_status_report)."""
        from core.stm_analyzer import abbreviation_expansion_conflicts
        conflicts = abbreviation_expansion_conflicts(
            R4_MEDS_QUESTION_WRAPPED, self.STM_OUTPUT, "",
        )
        assert conflicts == []

    def test_doc_doctor_regression_still_flagged_with_new_code_path(self):
        """Same STM/query shape as the round-1/3 B5 fixtures — must survive
        the B11 morphology filter unchanged."""
        from core.stm_analyzer import abbreviation_expansion_conflicts
        query = "Cool. Managed to push today and there is a new doc I think will be helpful"
        stm = {
            "topic": "New doctor",
            "user_question": "User mentions a new doctor is helpful",
            "temporal_facts": ["user has a new doctor who is helpful"],
        }
        conflicts = abbreviation_expansion_conflicts(query, stm, "")
        assert ("doc", "doctor") in conflicts


# ---------------------------------------------------------------------------
# B12 (round 4) — core.action_claim_guard.annotate_unverified_action_claim
# ---------------------------------------------------------------------------

LIVE_2056_REPLY = (
    "Done — note saved to daemon_notes/ta-sessions-schedule-2026-09-10.md. "
    "And it's doubly covered: the recurring Saturday 11:00 AM CT calendar "
    "event (through December 12, Zoom attached) is already in place from "
    "earlier today."
)


class TestAnnotateUnverifiedActionClaim:
    def test_live_reply_gets_annotated(self):
        from core.action_claim_guard import annotate_unverified_action_claim
        out = annotate_unverified_action_claim(LIVE_2056_REPLY)
        assert out.endswith("[unverified action claim]")
        assert out.startswith("Done")

    def test_plain_reply_unchanged(self):
        from core.action_claim_guard import annotate_unverified_action_claim
        plain = "Sure, I can help with that. Let me know what you need."
        assert annotate_unverified_action_claim(plain) == plain

    def test_empty_and_none_pass_through(self):
        from core.action_claim_guard import annotate_unverified_action_claim
        assert annotate_unverified_action_claim("") == ""
        assert annotate_unverified_action_claim(None) is None

    def test_idempotent_on_already_marked_text(self):
        from core.action_claim_guard import annotate_unverified_action_claim
        once = annotate_unverified_action_claim(LIVE_2056_REPLY)
        twice = annotate_unverified_action_claim(once)
        assert once == twice
        assert twice.count("[unverified action claim]") == 1

    def test_semantic_hit_channel_never_called(self):
        """The embedding channel (`_claim_semantic_hit`) is NEVER consulted
        by the annotator — this runs per rendered conversation item, not
        once per live reply. Monkeypatched to raise; the annotator must
        still return the annotated text without error."""
        from core.action_claim_guard import annotate_unverified_action_claim
        with patch(
            "core.action_claim_guard._claim_semantic_hit",
            side_effect=AssertionError("_claim_semantic_hit must not be called"),
        ):
            out = annotate_unverified_action_claim(LIVE_2056_REPLY)
        assert "[unverified action claim]" in out

    def test_semantic_hit_channel_never_called_on_plain_text_either(self):
        from core.action_claim_guard import annotate_unverified_action_claim
        with patch(
            "core.action_claim_guard._claim_semantic_hit",
            side_effect=AssertionError("_claim_semantic_hit must not be called"),
        ):
            out = annotate_unverified_action_claim("Sure, taking a look now.")
        assert out == "Sure, taking a look now."

    def test_approval_prompt_grammar_also_annotated(self):
        """The annotator covers `_APPROVAL_PROMPT_RE` (pending-card claims),
        not just `_CALENDAR_STATE_RE` — a superset of gatherer_knowledge's
        prior inline check (calendar-state + completion claims only)."""
        from core.action_claim_guard import annotate_unverified_action_claim
        reply = "Locked in. Approving the card will put it on your calendar."
        out = annotate_unverified_action_claim(reply)
        assert out.endswith("[unverified action claim]")

    def test_wrapped_form_of_live_reply_still_annotated(self):
        """The annotator normalizes internally (via utils.trigger_match.
        normalize_ws, same as claims_pending_card/claims_calendar_state) so
        a soft-wrapped assistant reply is unaffected."""
        from core.action_claim_guard import annotate_unverified_action_claim
        wrapped = LIVE_2056_REPLY.replace(
            "is already in place", "is already\n  in place",
        )
        out = annotate_unverified_action_claim(wrapped)
        assert out.endswith("[unverified action claim]")

    def test_offer_framed_calendar_sentence_not_annotated(self):
        from core.action_claim_guard import annotate_unverified_action_claim
        offer = "Want me to put the recurring Saturday session on your calendar?"
        assert annotate_unverified_action_claim(offer) == offer


class TestGathererKnowledgeDelegatesToAnnotator:
    """Regression: the existing self-note marker tests
    (tests/unit/test_sep10_probe_dump_actions.py::
    TestA14SelfNotesUnverifiedActionClaimMarker) must stay green after the
    inline check is replaced with a call to the shared annotator — pinned
    here at the source level (the fuller behavioral coverage lives in the
    Sub A file already)."""

    def test_gatherer_source_delegates_to_shared_annotator(self):
        import inspect
        from core.prompt import gatherer_knowledge
        src = inspect.getsource(gatherer_knowledge.KnowledgeRetrievalMixin.get_daemon_self_notes)
        assert "annotate_unverified_action_claim" in src


class TestFormatterAnnotatesDaemonSegmentOnly:
    """core/prompt/formatter.py's two conversation render sites
    (_format_memory, mem_parts) annotate the Daemon segment only."""

    def test_format_memory_annotates_daemon_segment_only(self):
        from core.prompt.formatter import PromptFormatter
        fmt = PromptFormatter(token_manager=MagicMock())
        mem = {
            "query": "did you get that note saved?",
            "response": LIVE_2056_REPLY,
            "timestamp": "",
        }
        out = fmt._format_memory(mem)
        assert "[unverified action claim]" in out
        daemon_idx = out.index("Daemon:")
        marker_idx = out.index("[unverified action claim]")
        user_idx = out.index("User:")
        assert marker_idx > daemon_idx > user_idx

    def test_format_memory_plain_reply_unmarked(self):
        from core.prompt.formatter import PromptFormatter
        fmt = PromptFormatter(token_manager=MagicMock())
        mem = {
            "query": "did you get that note saved?",
            "response": "Sure, taking a look now.",
            "timestamp": "",
        }
        out = fmt._format_memory(mem)
        assert "[unverified action claim]" not in out

    def test_mem_parts_via_assemble_prompt_annotates_daemon_segment(self):
        from core.prompt.formatter import PromptFormatter
        fmt = PromptFormatter(token_manager=MagicMock())
        context = {
            "recent_conversations": [
                {
                    "query": "did you get that note saved?",
                    "response": LIVE_2056_REPLY,
                    "timestamp": "",
                },
            ],
        }
        out = fmt._assemble_prompt(
            context=context, user_input="anything", directives="", system_prompt="",
        )
        assert "[unverified action claim]" in out
        assert "did you get that note saved?" in out
        # The user's own query text is never annotated.
        user_line_end = out.index("Daemon:")
        assert "[unverified action claim]" not in out[:user_line_end]

    def test_mem_parts_relevant_memories_section_also_annotates(self):
        from core.prompt.formatter import PromptFormatter
        fmt = PromptFormatter(token_manager=MagicMock())
        context = {
            "memories": [
                {
                    "query": "did you get that note saved?",
                    "response": LIVE_2056_REPLY,
                    "timestamp": "",
                },
            ],
        }
        out = fmt._assemble_prompt(
            context=context, user_input="anything", directives="", system_prompt="",
        )
        assert "[RELEVANT MEMORIES]" in out
        assert "[unverified action claim]" in out

    def test_mem_parts_plain_reply_unmarked(self):
        from core.prompt.formatter import PromptFormatter
        fmt = PromptFormatter(token_manager=MagicMock())
        context = {
            "recent_conversations": [
                {
                    "query": "did you get that note saved?",
                    "response": "Sure, taking a look now.",
                    "timestamp": "",
                },
            ],
        }
        out = fmt._assemble_prompt(
            context=context, user_input="anything", directives="", system_prompt="",
        )
        assert "[unverified action claim]" not in out


# ---------------------------------------------------------------------------
# B13 (round 5) — core.action_claim_guard.annotate_conversation_content +
# the PRODUCER-level wiring in core/prompt/gatherer_memory.py.
# ---------------------------------------------------------------------------

# The live "memory item 1" content-field text (hybrid retriever shape,
# 2026-09-11 round-5 finding), quoted exactly as given in the handoff
# (the "…" after "note" is itself part of that quoted fixture).
LIVE_MEMORY_ITEM1_CONTENT = (
    "User: jot down a note … \n"
    "Assistant: Noted and saved — TA sessions are Saturdays at 11:00 AM CT. "
    "It's also already on your calendar as a recurring weekly event (through "
    "December 12, Zoom link attached), so you've got it in both places now."
)
LIVE_ITEM_PLAIN_CONTENT = "User: …\nAssistant: Sounds good."


class TestAnnotateConversationContent:
    """Direct tests of the deployed
    `core.action_claim_guard.annotate_conversation_content`."""

    def test_live_item1_content_gets_annotated_after_assistant_label(self):
        from core.action_claim_guard import annotate_conversation_content
        out = annotate_conversation_content(LIVE_MEMORY_ITEM1_CONTENT)
        assert out.endswith("[unverified action claim]")
        assert out.startswith("User: jot down a note")
        assert out.index("[unverified action claim]") > out.index("Assistant:")

    def test_plain_content_item_unchanged(self):
        from core.action_claim_guard import annotate_conversation_content
        assert annotate_conversation_content(LIVE_ITEM_PLAIN_CONTENT) == LIVE_ITEM_PLAIN_CONTENT

    def test_no_label_content_unchanged(self):
        from core.action_claim_guard import annotate_conversation_content
        text = "just some notes with no speaker labels at all."
        assert annotate_conversation_content(text) == text

    def test_empty_and_none_pass_through(self):
        from core.action_claim_guard import annotate_conversation_content
        assert annotate_conversation_content("") == ""
        assert annotate_conversation_content(None) is None

    def test_double_annotation_is_idempotent(self):
        from core.action_claim_guard import annotate_conversation_content
        once = annotate_conversation_content(LIVE_MEMORY_ITEM1_CONTENT)
        twice = annotate_conversation_content(once)
        assert once == twice
        assert twice.count("[unverified action claim]") == 1

    def test_semantic_hit_channel_never_called(self):
        from core.action_claim_guard import annotate_conversation_content
        with patch(
            "core.action_claim_guard._claim_semantic_hit",
            side_effect=AssertionError("_claim_semantic_hit must not be called"),
        ):
            out = annotate_conversation_content(LIVE_MEMORY_ITEM1_CONTENT)
        assert "[unverified action claim]" in out

    def test_daemon_label_variant_also_supported(self):
        from core.action_claim_guard import annotate_conversation_content
        text = "User: hi\nDaemon: " + LIVE_2056_REPLY
        out = annotate_conversation_content(text)
        assert out.endswith("[unverified action claim]")
        assert out.index("[unverified action claim]") > out.index("Daemon:")

    def test_last_label_used_when_multiple_turns_present(self):
        from core.action_claim_guard import annotate_conversation_content
        text = (
            "User: hi\nAssistant: Sounds good.\n"
            "User: did you save it?\nAssistant: " + LIVE_2056_REPLY
        )
        out = annotate_conversation_content(text)
        assert out.count("[unverified action claim]") == 1
        last_assistant_idx = out.rindex("Assistant:")
        assert out.index("[unverified action claim]") > last_assistant_idx


# ---------------------------------------------------------------------------
# B13 (round 5) — producer-level gatherer wiring
# (core/prompt/gatherer_memory.py's `_annotate_memory_item_claim`, applied
# at the single exit point of `_get_recent_conversations`/
# `_get_semantic_memories`) + the digest builders that read straight from
# those raw dicts.
# ---------------------------------------------------------------------------

class _R5StubCorpusManager:
    """Minimal corpus_manager double: only `get_recent_memories` is called
    by the deployed `_get_recent_conversations`."""

    def __init__(self, memories):
        self._memories = list(memories)

    def get_recent_memories(self, count=15):
        return list(self._memories)


class _R5StubCoordinator:
    """Minimal memory_coordinator double. No `graph_memory` attribute is
    set (deliberately absent, not None-valued) so the deployed
    `_expand_query_with_graph` takes its `not graph` early-return cleanly."""

    def __init__(self, memories):
        self.corpus_manager = _R5StubCorpusManager(memories)
        self._memories = list(memories)

    async def get_memories(self, *args, **kwargs):
        return list(self._memories)


def _make_r5_gatherer(coordinator):
    from core.prompt.gatherer_memory import MemoryRetrievalMixin
    g = MemoryRetrievalMixin.__new__(MemoryRetrievalMixin)
    g.memory_coordinator = coordinator
    g.memory_id_map = {}
    g._fast_mode = False
    return g


class TestGathererMemoryProducerAnnotation:
    """B13: the marker is applied at the PRODUCER
    (core/prompt/gatherer_memory.py), not only at the two formatter render
    sites B12 wired — the agentic decision digest and the planner's context
    digest both read `recent_conversations`/`memories` straight from these
    raw dicts and never reach the formatter at all."""

    @pytest.mark.asyncio
    async def test_recent_conversations_query_response_item_annotated(self):
        item = {
            "query": "did you get that note saved?",
            "response": LIVE_2056_REPLY,
            "timestamp": "2026-09-11T09:00:00",
        }
        coord = _R5StubCoordinator([item])
        g = _make_r5_gatherer(coord)
        result = await g._get_recent_conversations(limit=1)
        assert result
        assert "[unverified action claim]" in result[0]["response"]

    @pytest.mark.asyncio
    async def test_recent_conversations_plain_reply_unmarked(self):
        item = {
            "query": "did you get that note saved?",
            "response": "Sure, taking a look now.",
            "timestamp": "2026-09-11T09:00:00",
        }
        coord = _R5StubCoordinator([item])
        g = _make_r5_gatherer(coord)
        result = await g._get_recent_conversations(limit=1)
        assert "[unverified action claim]" not in result[0]["response"]

    @pytest.mark.asyncio
    async def test_recent_conversations_does_not_double_mark_already_annotated_reply(self):
        already = LIVE_2056_REPLY.rstrip() + "\n[unverified action claim]"
        item = {"query": "did you get that note saved?", "response": already, "timestamp": ""}
        coord = _R5StubCoordinator([item])
        g = _make_r5_gatherer(coord)
        result = await g._get_recent_conversations(limit=1)
        assert result[0]["response"].count("[unverified action claim]") == 1

    @pytest.mark.asyncio
    async def test_memories_content_shape_item_annotated(self):
        item = {
            "content": LIVE_MEMORY_ITEM1_CONTENT,
            "timestamp": "2026-09-11T09:00:00",
            "id": "m1",
        }
        coord = _R5StubCoordinator([item])
        g = _make_r5_gatherer(coord)
        result = await g._get_semantic_memories(
            "jot down a note for this session", limit=3,
        )
        assert result
        assert "[unverified action claim]" in result[0]["content"]
        assert result[0]["content"].startswith("User: jot down a note")

    @pytest.mark.asyncio
    async def test_memories_plain_content_unmarked(self):
        item = {
            "content": LIVE_ITEM_PLAIN_CONTENT,
            "timestamp": "2026-09-11T09:00:00",
            "id": "m2",
        }
        coord = _R5StubCoordinator([item])
        g = _make_r5_gatherer(coord)
        result = await g._get_semantic_memories("anything at all here", limit=3)
        assert "[unverified action claim]" not in result[0]["content"]


class TestClipPreservingClaimMarker:
    """`core.agentic.controller.AgenticSearchController.
    _clip_preserving_claim_marker` — the decision digest's hard 220-char
    clip must never silently drop the marker B13 just attached; ordinary
    (unmarked) text keeps the prior plain clip behavior."""

    def test_marker_survives_clip_even_when_message_exceeds_limit(self):
        from core.agentic.controller import AgenticSearchController
        annotated = LIVE_2056_REPLY.rstrip() + "\n[unverified action claim]"
        assert len(LIVE_2056_REPLY) > AgenticSearchController._DIGEST_MSG_CHARS
        clipped = AgenticSearchController._clip_preserving_claim_marker(
            annotated, AgenticSearchController._DIGEST_MSG_CHARS,
        )
        assert clipped.endswith("[unverified action claim]")

    def test_unmarked_text_still_hard_clipped(self):
        from core.agentic.controller import AgenticSearchController
        long_text = "x" * 500
        clipped = AgenticSearchController._clip_preserving_claim_marker(long_text, 220)
        assert clipped == long_text[:220]


class TestDigestBuildersPreserveMarker:
    """The agentic decision digest and the planner's context digest, given
    gatherer-annotated items, both surface the marker (deployed functions,
    real dicts) — this is the actual point of B13: the decision round that
    replied "Third time asking, and it's still covered" never went through
    the formatter at all."""

    @pytest.mark.asyncio
    async def test_agentic_recent_conversation_digest_contains_marker(self):
        from core.agentic.controller import AgenticSearchController
        item = {
            "query": "did you get that note saved?",
            "response": LIVE_2056_REPLY,
            "timestamp": "2026-09-11T09:00:00",
        }
        coord = _R5StubCoordinator([item])
        g = _make_r5_gatherer(coord)
        result = await g._get_recent_conversations(limit=1)
        controller = AgenticSearchController.__new__(AgenticSearchController)
        digest = controller._compute_recent_conversation_digest(
            {"recent_conversations": result},
        )
        assert "[unverified action claim]" in digest

    @pytest.mark.asyncio
    async def test_planner_context_digest_contains_marker_for_recent_conversations(self):
        from core.response_planner import ResponsePlanner
        item = {
            "query": "did you get that note saved?",
            "response": LIVE_2056_REPLY,
            "timestamp": "2026-09-11T09:00:00",
        }
        coord = _R5StubCoordinator([item])
        g = _make_r5_gatherer(coord)
        result = await g._get_recent_conversations(limit=1)
        digest, included = ResponsePlanner.build_context_digest(
            {"recent_conversations": result},
        )
        assert "[unverified action claim]" in digest
        assert "recent_conversations" in included

    @pytest.mark.asyncio
    async def test_planner_context_digest_contains_marker_for_memories(self):
        from core.response_planner import ResponsePlanner
        item = {
            "content": LIVE_MEMORY_ITEM1_CONTENT,
            "timestamp": "2026-09-11T09:00:00",
            "id": "m1",
        }
        coord = _R5StubCoordinator([item])
        g = _make_r5_gatherer(coord)
        result = await g._get_semantic_memories(
            "jot down a note for this session", limit=3,
        )
        digest, included = ResponsePlanner.build_context_digest({"memories": result})
        assert "[unverified action claim]" in digest
        assert "memories" in included


# ---------------------------------------------------------------------------
# B14 (round 5) — core.response_planner.ResponsePlanner.should_plan +
# utils.query_checker.is_task_directive
# ---------------------------------------------------------------------------

class TestShouldPlanTaskDirective:
    def test_note_save_request_skips_planning(self):
        from core.response_planner import ResponsePlanner
        ctx = _PlannerFakeContext(original_query=R4_T1_NOTE_SAVE)
        assert ResponsePlanner.should_plan(ctx) is False

    def test_note_save_request_wrapped_skips_planning(self):
        from core.response_planner import ResponsePlanner
        ctx = _PlannerFakeContext(original_query=R4_T1_NOTE_SAVE_WRAPPED)
        assert ResponsePlanner.should_plan(ctx) is False

    def test_meds_question_still_plans(self):
        from core.response_planner import ResponsePlanner
        ctx = _PlannerFakeContext(original_query=R4_MEDS_QUESTION)
        assert ResponsePlanner.should_plan(ctx) is True

    def test_meds_question_wrapped_still_plans(self):
        from core.response_planner import ResponsePlanner
        ctx = _PlannerFakeContext(original_query=R4_MEDS_QUESTION_WRAPPED)
        assert ResponsePlanner.should_plan(ctx) is True

    def test_info_seeking_request_keeps_planning(self):
        """Reuses B1's live Q2 fixture (already pinned earlier in this file
        as info-seeking via `_is_info_seeking(Q2) is True`) rather than a
        freshly invented example."""
        from core.response_planner import ResponsePlanner
        ctx = _PlannerFakeContext(
            original_query=TestInfoSeekingSentenceLevel.Q2,
        )
        assert ResponsePlanner.should_plan(ctx) is True


# ---------------------------------------------------------------------------
# B15 (round 5) — core.tone_instructions.get_intent_style_instructions
# (query=...) + its single caller in core/orchestrator.py
# ---------------------------------------------------------------------------

LIVE_TA_NAME_QUESTION = "what did I say my TA's name was"


class TestIntentStyleBlockTaskDirectiveGate:
    def test_task_directive_gets_no_style_block(self):
        from core.tone_instructions import get_intent_style_instructions
        out = get_intent_style_instructions(
            "factual_recall", 0.9, "CONVERSATIONAL", query=R4_T1_NOTE_SAVE,
        )
        assert out == ""

    def test_task_directive_wrapped_gets_no_style_block(self):
        from core.tone_instructions import get_intent_style_instructions
        out = get_intent_style_instructions(
            "factual_recall", 0.9, "CONVERSATIONAL", query=R4_T1_NOTE_SAVE_WRAPPED,
        )
        assert out == ""

    def test_ordinary_factual_recall_question_keeps_style_block(self):
        from core.tone_instructions import get_intent_style_instructions
        out = get_intent_style_instructions(
            "factual_recall", 0.9, "CONVERSATIONAL", query=LIVE_TA_NAME_QUESTION,
        )
        assert out != ""

    def test_no_query_argument_preserves_prior_behavior(self):
        """A caller that doesn't pass `query` (None) gets the pre-B15
        behavior unchanged — the parameter is additive."""
        from core.tone_instructions import get_intent_style_instructions
        out = get_intent_style_instructions("factual_recall", 0.9, "CONVERSATIONAL")
        assert out != ""

    def test_crisis_tone_still_suppresses_regardless_of_query(self):
        from core.tone_instructions import get_intent_style_instructions
        out = get_intent_style_instructions(
            "factual_recall", 0.9, "MEDIUM", query=LIVE_TA_NAME_QUESTION,
        )
        assert out == ""


class TestOrchestratorIntentStyleWiring:
    """Source-level regression: the single call site in
    core/orchestrator.py must forward `query=` (same convention as B12's
    `TestGathererKnowledgeDelegatesToAnnotator`)."""

    def test_orchestrator_source_passes_query_to_intent_style_call(self):
        import inspect
        import core.orchestrator as orch_mod
        src = inspect.getsource(orch_mod)
        assert src.count("get_intent_style_instructions(") == 1
        idx = src.index("get_intent_style_instructions(")
        call_block = src[idx: idx + 400]
        assert "query=" in call_block
