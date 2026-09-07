"""B1-B4 (2026-09-07, Delegate B of
docs/HANDOFF_20260907_upload_reuse_contracts.md "Delegate B — reuse gate,
STM clock, intent regex").

Verified root causes these tests guard against (six-turn debug dump,
2026-09-07 01:29-10:06):

R3. `_decision_saw_admitted_evidence` (Phase A, A4) returned True whenever
    session.accumulated_context was non-empty — a turn-6 repo file-listing
    (file_list/file_grep tool output) counted as "evidence seen" while the
    base retrieval's memories/notes/uploads never reached the decision
    prompt, so `answer_call=decision_reuse` shipped a narration instead of
    using the admitted evidence. B1 splits admitted-evidence keys into
    background (never blocks reuse) vs. retrieval (blocks reuse whenever
    non-empty, regardless of accumulated_context) and adds the
    [USER UPLOADED ITEMS] context key ("user_uploads") to the retrieval set
    — it was missing from evidence tracking entirely.
R3(b). The turn-6 decision-round text itself ("I checked the uploads
    directory ... Can you tell me: 1. ... 2. ... 3. ...") passed every
    existing `_usable_decision_answer` guard (>=200 chars, ends in "?", no
    promissory verb, no loop-meta opener) and was reused as the final
    answer despite being a clarification list, not an answer. B2 adds a
    question-dominated guard.
R4. The STM prompt carried no clock. Turn 3 (08:31) `temporal_facts` said
    "Current time is approximately 1:31 AM", copied verbatim from Daemon's
    own 01:31 reply, and the planner repeated it. B3 adds an authoritative
    "Current time" line plus disambiguation rule 8 forbidding temporal_facts
    from restating a clock/date/elapsed-time figure from an ASSISTANT reply.
R5. `intent_classifier` CREATIVE_EXPLORATION regex matched bare `idea(s)?`;
    "give me an idea of what we're working with" -> creative_exploration
    0.75 -> CREATIVE style block on a document-lookup request. B4 narrows
    the regex.
"""
import re
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.agentic.controller import AgenticSearchController
from core.agentic.tools import ToolExecutor
from core.agentic.types import SearchDecision
from core.intent_classifier import IntentClassifier, IntentType
from core.stm_analyzer import STMAnalyzer

from tests.unit.test_agentic_decision_answer_reuse import LONG_ANSWER


# ---------------------------------------------------------------------------
# Shared helpers (mirrors tests/unit/test_evidence_transport.py)
# ---------------------------------------------------------------------------

def _mock_tool_executor():
    class _Exec:
        pass
    ex = _Exec()
    ex._current_web_source_map = {}
    ex._current_wiki_source_map = {}
    ex.get_tool_health = lambda: "All tools nominal."
    ex._merge_web_ids = types.MethodType(ToolExecutor._merge_web_ids, ex)
    return ex


@pytest.fixture
def controller():
    model_manager = MagicMock()
    model_manager.api_models = {}
    c = AgenticSearchController(model_manager=model_manager, web_search_manager=MagicMock())
    c._tool_executor = _mock_tool_executor()
    return c


@pytest.fixture
def classifier():
    return IntentClassifier()


def _fake_decision_factory(capture=None, answer=LONG_ANSWER, decisions=None):
    async def fake(*a, **k):
        if capture is not None:
            capture["prompt"] = k.get("prompt")
        if decisions is not None:
            return decisions
        return [SearchDecision(wants_answer=True, partial_response=answer)]
    return fake


async def _run(controller, **kwargs):
    events = []
    async for ev in controller.run_agentic_search(**kwargs):
        events.append(ev)
    text = "".join(e for e in events if isinstance(e, str))
    return events, text


def _stm_model_manager(json_payload=None):
    """A model_manager double whose generate_once returns a minimal valid
    STM JSON payload and captures the prompt it was called with."""
    payload = json_payload or (
        '{"topic": "t", "user_question": "q", "intent": "i", '
        '"tone": "neutral", "reference_type": "unclear", '
        '"temporal_facts": [], "open_threads": [], "constraints": []}'
    )
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=payload)
    return mm


# ===========================================================================
# B1 — reuse gate: retrieval evidence blocks, background evidence does not
# ===========================================================================

class TestB1ReuseGate:

    @pytest.mark.asyncio
    async def test_background_only_context_with_tool_text_allows_reuse(self, controller):
        """Old A4 rule looped over ALL admitted-evidence keys (background
        included) and would have blocked reuse here since user_profile was
        non-empty. B1: background keys never block. Also re-verifies A5's
        "real hash, not sentinel" receipt under a B1-compatible scenario —
        tests/unit/test_evidence_transport.py's former seeded-web version of
        this hash check now blocks reuse under the new rule (see that
        file's TestT9ReuseGate.test_seeded_web_evidence_now_blocks_reuse)."""
        controller._get_model_decision = _fake_decision_factory()

        await _run(
            controller,
            query="q", system_prompt="sys", model_name="test-model",
            initial_search_terms=[], skip_initial_search=True,
            initial_context={"user_profile": "Name: Test User\nAge: 30"},
        )

        session = controller._last_session
        assert session.answer_call == "decision_reuse"
        assert session.final_prompt_hash == session.decision_prompt_hash
        assert session.final_prompt_hash != "decision-answer-reuse"
        assert len(session.final_prompt_hash) == 16
        int(session.final_prompt_hash, 16)  # raises ValueError if not real hex

    @pytest.mark.asyncio
    async def test_memories_key_blocks_reuse_and_names_reason(self, controller):
        controller._get_model_decision = _fake_decision_factory()
        captured = {"called": False}

        async def fake_final(query, system_prompt, model_name, session, initial_context=None):
            captured["called"] = True
            yield "SYNTHESIZED."

        controller._generate_final_response = fake_final

        await _run(
            controller,
            query="q", system_prompt="sys", model_name="test-model",
            initial_search_terms=[], skip_initial_search=True,
            initial_context={"memories": [{"content": "m"}]},
        )

        session = controller._last_session
        assert captured["called"] is True
        assert session.answer_call == "final_synthesis"
        assert (
            session.reuse_skipped_reason
            == "decision prompt lacked admitted evidence: memories"
        )

    @pytest.mark.asyncio
    async def test_user_uploads_key_blocks_reuse(self, controller):
        """The [USER UPLOADED ITEMS] context key ("user_uploads" — confirmed
        in core/prompt/builder.py / token_manager.py / formatter.py, the
        section rendered as [USER UPLOADED ITEMS]) was entirely absent from
        evidence tracking before B1."""
        controller._get_model_decision = _fake_decision_factory()

        async def fake_final(query, system_prompt, model_name, session, initial_context=None):
            yield "SYNTHESIZED."

        controller._generate_final_response = fake_final

        await _run(
            controller,
            query="q", system_prompt="sys", model_name="test-model",
            initial_search_terms=[], skip_initial_search=True,
            initial_context={"user_uploads": [{"filename": "upload:Homework1.pdf"}]},
        )

        session = controller._last_session
        assert session.answer_call == "final_synthesis"
        assert (
            session.reuse_skipped_reason
            == "decision prompt lacked admitted evidence: user_uploads"
        )

    @pytest.mark.asyncio
    async def test_empty_initial_context_allows_reuse(self, controller):
        controller._get_model_decision = _fake_decision_factory()

        await _run(
            controller,
            query="q", system_prompt="sys", model_name="test-model",
            initial_search_terms=[], skip_initial_search=True,
            initial_context=None,
        )

        session = controller._last_session
        assert session.answer_call == "decision_reuse"

    def test_helper_functions_directly(self, controller):
        """Unit-level pin on the two new helpers, independent of the full
        loop plumbing above."""
        assert controller._decision_saw_admitted_evidence(
            session=None, initial_context={"user_profile": "x"}
        ) is True
        assert controller._decision_saw_admitted_evidence(
            session=None, initial_context={"memories": [{"content": "m"}]}
        ) is False
        assert controller._decision_saw_admitted_evidence(
            session=None, initial_context=None
        ) is True
        assert controller._decision_saw_admitted_evidence(
            session=None, initial_context={}
        ) is True
        assert controller._first_unmet_retrieval_key(
            {"memories": [{"content": "m"}]}
        ) == "memories"
        assert controller._first_unmet_retrieval_key(
            {"user_uploads": [{"filename": "x"}]}
        ) == "user_uploads"
        assert controller._first_unmet_retrieval_key({"user_profile": "x"}) is None
        assert controller._first_unmet_retrieval_key(None) is None


# ===========================================================================
# B2 — question-dominated decision answers are not reused
# ===========================================================================

class TestB2QuestionDominated:

    def test_turn6_shaped_question_list_rejected(self, controller):
        # The EXACT live turn-6 reply (2026-09-07 10:06) — six non-question
        # lines around the enumerated question list; a line-ratio test alone
        # lets it through (Fable referee).
        text = (
            "I checked the uploads directory and searched for assignment-related "
            "files, but I'm not seeing anything there. The file listing shows "
            "your project codebase, but no obvious \"user uploads\" folder with "
            "course materials.\n\n"
            "A few possibilities:\n"
            "- The upload might not have synced properly\n"
            "- It could be in a different location (Downloads, a course-specific folder, etc.)\n"
            "- The filename might not contain obvious keywords like \"assignment\" or \"hw1\"\n\n"
            "Can you tell me:\n"
            "1. What course this is for?\n"
            "2. Roughly when you uploaded it?\n"
            "3. Any part of the filename you remember?\n\n"
            "Or if you know the actual path where uploads land on your system, "
            "I can check there directly."
        )
        assert controller._usable_decision_answer(text) is None

    def test_short_question_list_rejected(self, controller):
        text = (
            "I checked the uploads directory but didn't find a file matching "
            "that description. Can you tell me:\n"
            "1. What is the exact filename or assignment number?\n"
            "2. When did you upload it?\n"
            "3. Which course is this for?"
        )
        assert controller._usable_decision_answer(text) is None

    def test_long_answer_with_one_trailing_question_unchanged(self, controller):
        text = LONG_ANSWER + " Does that help?"
        assert controller._usable_decision_answer(text) == text

    def test_existing_guards_untouched(self, controller):
        """B2 must not touch _PROMISSORY_RE / _LOOP_META_RE behavior."""
        assert controller._usable_decision_answer(LONG_ANSWER) == LONG_ANSWER
        promissory = "Let me check the available information. " + LONG_ANSWER
        assert controller._usable_decision_answer(promissory) is None


# ===========================================================================
# B3 — STM clock anchor
# ===========================================================================

class TestB3ClockAnchor:

    @pytest.mark.asyncio
    async def test_current_time_line_and_rule8_present(self):
        mm = _stm_model_manager()
        analyzer = STMAnalyzer(model_manager=mm)
        fixed_now = datetime(2026, 9, 7, 8, 31)

        result = await analyzer.analyze(
            recent_memories=[], user_query="what time is it", now=fixed_now,
        )

        assert result["topic"] == "t"
        prompt = mm.generate_once.call_args.kwargs["prompt"]
        assert (
            "Current time: Monday, 2026-09-07 08:31 "
            "(authoritative — the ONLY source for the present time)"
        ) in prompt
        assert (
            "temporal_facts must never restate a clock time, date, or "
            "elapsed-time figure taken from an earlier ASSISTANT reply"
        ) in prompt
        # The clock line renders directly above "Current user query:".
        assert re.search(
            r"Current time: Monday, 2026-09-07 08:31[^\n]*\nCurrent user query:",
            prompt,
        )

    @pytest.mark.asyncio
    async def test_default_now_does_not_crash(self):
        mm = _stm_model_manager()
        analyzer = STMAnalyzer(model_manager=mm)

        result = await analyzer.analyze(recent_memories=[], user_query="hello")

        assert result["topic"] == "t"
        prompt = mm.generate_once.call_args.kwargs["prompt"]
        assert "Current time:" in prompt
        assert "(authoritative — the ONLY source for the present time)" in prompt

    @pytest.mark.asyncio
    async def test_stale_assistant_time_precedes_rule8_text(self):
        mm = _stm_model_manager()
        analyzer = STMAnalyzer(model_manager=mm)
        stale_time_text = "yours currently reads 1:31 AM"

        await analyzer.analyze(
            recent_memories=[],
            user_query="what time is it now",
            last_assistant_response=f"Sure, {stale_time_text} if that helps.",
            now=datetime(2026, 9, 7, 8, 31),
        )

        prompt = mm.generate_once.call_args.kwargs["prompt"]
        assert stale_time_text in prompt
        rule8_idx = prompt.index(
            "temporal_facts must never restate a clock time"
        )
        stale_idx = prompt.index(stale_time_text)
        assert stale_idx < rule8_idx

    @pytest.mark.asyncio
    async def test_existing_caller_shape_unaffected(self):
        """context_pipeline.py's call site passes no `now` kwarg — confirm
        that positional/keyword call shape used there still works."""
        mm = _stm_model_manager()
        analyzer = STMAnalyzer(model_manager=mm)

        result = await analyzer.analyze(
            recent_memories=[],
            user_query="q",
            last_assistant_response=None,
            graph_memory=None,
        )
        assert result["topic"] == "t"


# ===========================================================================
# B4 — CREATIVE_EXPLORATION regex no longer fires on bare/negated "idea"
# ===========================================================================

class TestB4IntentRegexGuard:

    def test_idea_of_what_not_creative(self, classifier):
        r = classifier.classify("give me an idea of what we're working with")
        assert r.intent != IntentType.CREATIVE_EXPLORATION

    def test_no_idea_what_to_do_not_creative(self, classifier):
        r = classifier.classify("no idea what to do")
        assert r.intent != IntentType.CREATIVE_EXPLORATION

    def test_any_idea_why_not_creative(self, classifier):
        r = classifier.classify("any idea why")
        assert r.intent != IntentType.CREATIVE_EXPLORATION

    def test_have_an_idea_is_creative(self, classifier):
        r = classifier.classify("I have an idea. For later on.")
        assert r.intent == IntentType.CREATIVE_EXPLORATION

    def test_brainstorm_ideas_is_creative(self, classifier):
        r = classifier.classify("brainstorm some ideas")
        assert r.intent == IntentType.CREATIVE_EXPLORATION

    def test_what_if_is_creative(self, classifier):
        r = classifier.classify("what if we combined the two approaches")
        assert r.intent == IntentType.CREATIVE_EXPLORATION
