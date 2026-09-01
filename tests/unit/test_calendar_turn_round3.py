"""Regression coverage for the final 2026-08-29 calendar follow-up turns.

Live failures covered here:
- "Day of please thank you" was treated as a request for today's weekday.
- "create the calendar events" missed singular-only action detection and the gate.
- the fallback claimed Calendar access was unavailable despite healthy OAuth/scope.
- seven markers overflowed a five-proposal store while UI/API exposed one ID.
- the grounding verifier rewrote Saturday 2026-08-29 to impossible "Sunday".
- raw provider <|sep|> leaked into the conversation transcript.
"""

import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.actions.google_calendar_create import create_calendar_event
from core.actions.registry import (
    ACTION_SPECS,
    detect_action_intent,
    get_runtime_action_health,
)
from core.actions.types import ActionProposal, ActionType
from core.agentic.controller import AgenticSearchController
from core.agentic.gate import evaluate_agentic_gate
from core.agentic.protocols import NativeToolsHandler
from core.agentic.tools import ToolExecutor
from core.agentic.types import SearchDecision
from core.context_pipeline import ContextPipeline
from core.grounding_check import _parse_verdict, weekday_date_mismatches
from core.stm_analyzer import STMAnalyzer
from utils.query_checker import is_continuation_answer


PRIOR_CHOICE = (
    "Want me to queue up all 7 as calendar events — say, all-day entries on "
    "each due date, or evening reminders the day before?"
)

INITIAL_REQUEST = (
    "Please search for documents related to the MGT class I am enrolled in, "
    "catalog all dates and deadlines, and place each in the appropriate time "
    "slot on my Google calendar"
)


def _bare_controller():
    return object.__new__(AgenticSearchController)


class TestContinuationAndRouting:

    def test_day_of_is_an_answer_to_prior_choice(self):
        assert is_continuation_answer("Day of please thank you", PRIOR_CHOICE)

    @pytest.mark.asyncio
    async def test_topic_inherits_for_short_option_answer(self):
        topic_manager = MagicMock()
        topic_manager.last_topic = "Mgt Class Deadlines"
        pipeline = object.__new__(ContextPipeline)
        pipeline.topic_manager = topic_manager
        primary, topics = await pipeline._extract_topics(
            "Day of please thank you",
            last_exchange={"response": PRIOR_CHOICE},
        )
        assert primary == "Mgt Class Deadlines"
        assert topics == ["Mgt Class Deadlines"]
        topic_manager.get_primary_topic.assert_not_called()

    def test_plural_calendar_events_detected(self):
        assert detect_action_intent(
            "Yes please create the calendar events"
        ) == ActionType.CALENDAR_CREATE_EVENT

    @pytest.mark.asyncio
    async def test_plural_calendar_request_routes_to_tools(self):
        decision = await evaluate_agentic_gate(
            "Yes please create the calendar events",
            entity_resolver=None,
            model_manager=None,
            corpus_manager=None,
            intent_info=None,
        )
        assert decision.should_trigger is True
        assert decision.modes == ["tools"]
        assert decision.skip_initial_search is True
        assert decision.veto_exempt is True

    @pytest.mark.asyncio
    async def test_action_phrase_buried_in_long_paste_does_not_route(self):
        pasted = (
            "This is quoted material from a document. "
            + "background context " * 90
            + "The author wrote: please create a calendar event for the review."
        )
        decision = await evaluate_agentic_gate(
            pasted,
            entity_resolver=None,
            model_manager=None,
            corpus_manager=None,
            intent_info=None,
        )
        assert detect_action_intent(pasted) is None
        assert decision.should_trigger is False
        assert "tools" not in decision.modes

    @pytest.mark.asyncio
    async def test_long_head_anchored_action_request_still_routes(self):
        request = (
            "Please create a calendar event using all of this context: "
            + "course details " * 90
        )
        decision = await evaluate_agentic_gate(
            request,
            entity_resolver=None,
            model_manager=None,
            corpus_manager=None,
            intent_info=None,
        )
        assert decision.should_trigger is True
        assert decision.modes == ["tools"]


class _STMModel:
    def __init__(self):
        self.prompt = ""

    async def generate_once(self, prompt, **kwargs):
        self.prompt = prompt
        # Script the same bad semantic reading; deterministic post-processing
        # must still mark the current turn as an answer/clarification.
        return json.dumps({
            "topic": "Day inquiry",
            "user_question": "What day is today?",
            "intent": "Get today's date",
            "tone": "neutral",
            "reference_type": "new_event",
            "temporal_facts": [],
            "open_threads": [],
            "constraints": [],
        })


class TestSTMImmediateQuestion:

    @pytest.mark.asyncio
    async def test_last_assistant_question_is_injected_and_overrides_new_event(self):
        mm = _STMModel()
        analyzer = STMAnalyzer(mm)
        result = await analyzer.analyze(
            recent_memories=[],
            user_query="Day of please thank you",
            last_assistant_response=PRIOR_CHOICE,
        )
        assert PRIOR_CHOICE in mm.prompt
        assert "IMMEDIATELY PRECEDING" in mm.prompt
        assert result["reference_type"] == "clarification"
        assert "answering" in result["user_question"]
        assert "without asking for it again" in result["intent"]

    def test_memory_render_is_chronological_and_keeps_closing_question(self):
        analyzer = STMAnalyzer(_STMModel())
        newest = {
            "timestamp": "2026-08-29T16:35:00",
            "query": "Day of please thank you",
            "response": "Okay.",
        }
        older = {
            "timestamp": "2026-08-29T16:34:00",
            "query": "Create my events",
            "response": "x" * 900 + PRIOR_CHOICE,
        }
        rendered = analyzer._format_memories([newest, older])
        assert rendered.index("Create my events") < rendered.index("Day of please")
        assert "evening reminders the day before?" in rendered


class TestActionContextAndBatch:

    def test_action_context_keeps_dates_and_answered_preference(self):
        controller = _bare_controller()
        context = {
            "recent_conversations": [
                {
                    "timestamp": "2026-08-29T16:36:00",
                    "query": "Day of please thank you",
                    "response": "Okay.",
                },
                {
                    "timestamp": "2026-08-29T16:35:00",
                    "query": "Try the last query",
                    "response": (
                        "HW 1 Sep 13; HW 2 Sep 27; HW 3 Oct 11; HW 4 Oct 25; "
                        "HW 5 Nov 8; HW 6 Nov 22; HW 7 Dec 8. " + PRIOR_CHOICE
                    ),
                },
            ]
        }
        digest = controller._compute_action_context(context)
        assert "Sep 13" in digest and "Dec 8" in digest
        assert "Day of please thank you" in digest
        assert "Do not ask again" in digest

    def test_forced_prompt_retains_iteration_context(self):
        source = inspect.getsource(AgenticSearchController.run_agentic_search)
        assert '_round_prompt = iteration_prompt + "\\n\\n"' in source

    def test_xml_force_prompt_explains_all_day_and_et_timezone(self):
        prompt = AgenticSearchController._build_xml_action_force_prompt(
            "create the calendar events",
            ActionType.CALENDAR_CREATE_EVENT,
            ACTION_SPECS[ActionType.CALENDAR_CREATE_EVENT],
        )
        assert 'all_day="<true-or-false>"' in prompt
        assert "end is exclusive" in prompt
        assert "ET = America/New_York" in prompt
        assert "-04:00" in prompt

    def test_seven_calendar_calls_become_one_batch(self):
        decisions = [
            SearchDecision(
                wants_action=True,
                action_type="calendar_create_event",
                action_params={
                    "summary": f"MGT 6203 HW {i}",
                    "start_time": f"2026-09-{i:02d}T22:59:00-05:00",
                    "end_time": f"2026-09-{i:02d}T23:00:00-05:00",
                },
                action_reason="user asked",
            )
            for i in range(1, 8)
        ]
        result = AgenticSearchController._coalesce_action_decisions(decisions)
        assert len(result) == 1
        assert result[0].action_summary == "calendar_create_event: 7 events"
        assert len(result[0].action_params["events"]) == 7

    def test_native_batch_shape_is_accepted(self):
        events = [
            {
                "summary": "HW 1",
                "start_time": "2026-09-13T22:59:00-05:00",
                "end_time": "2026-09-13T23:00:00-05:00",
            },
            {
                "summary": "HW 2",
                "start_time": "2026-09-27T22:59:00-05:00",
                "end_time": "2026-09-27T23:00:00-05:00",
            },
        ]
        call = {
            "function": {
                "name": "propose_action",
                "arguments": json.dumps({
                    "action_type": "calendar_create_event",
                    "reason": "user asked",
                    "events": events,
                }),
            }
        }
        decision = NativeToolsHandler(actions_available=True)._parse_single_tool_call(call)
        assert decision is not None and decision.wants_action
        assert decision.action_params["events"] == events
        assert decision.action_summary == "calendar_create_event: 2 events"

    def test_calendar_spec_accepts_batch_without_fake_top_level_event(self):
        spec = ACTION_SPECS[ActionType.CALENDAR_CREATE_EVENT]
        assert spec.accepts_params({
            "events": [{
                "summary": "HW 1",
                "start_time": "2026-09-13T22:59:00-05:00",
                "end_time": "2026-09-13T23:00:00-05:00",
            }]
        })


class TestThreeTurnCalendarReplay:
    """A compact replay of the exact three-turn failure chain.

    This intentionally stops at a mocked Calendar API: it verifies the real
    routing/context/protocol/executor boundaries without creating user data.
    """

    @pytest.mark.asyncio
    async def test_initial_search_preference_and_approval_stay_one_flow(self):
        first = await evaluate_agentic_gate(
            INITIAL_REQUEST,
            entity_resolver=None,
            model_manager=None,
            corpus_manager=None,
            intent_info=None,
        )
        assert first.should_trigger is True
        assert "tools" in first.modes
        assert "web_search" not in first.modes

        topic_manager = MagicMock(last_topic="Mgt Class Deadlines")
        pipeline = object.__new__(ContextPipeline)
        pipeline.topic_manager = topic_manager
        topic, topics = await pipeline._extract_topics(
            "Day of please thank you",
            last_exchange={"response": PRIOR_CHOICE},
        )
        assert topic == "Mgt Class Deadlines"
        assert topics == ["Mgt Class Deadlines"]

        approval = "Yes please create the calendar events"
        second = await evaluate_agentic_gate(
            approval,
            entity_resolver=None,
            model_manager=None,
            corpus_manager=None,
            intent_info=None,
        )
        assert second.should_trigger is True
        assert second.modes == ["tools"]
        assert second.veto_exempt is True

        controller = _bare_controller()
        context = {
            "recent_conversations": [
                {"timestamp": "2026-08-29T16:35:00", "query": INITIAL_REQUEST,
                 "response": "HW 1 Sep 13; HW 2 Sep 27; HW 3 Oct 11; HW 4 Oct 25; "
                             "HW 5 Nov 8; HW 6 Nov 22; HW 7 Dec 8. " + PRIOR_CHOICE},
                {"timestamp": "2026-08-29T16:36:00", "query": "Day of please thank you",
                 "response": "Use all-day entries on each due date."},
            ]
        }
        digest = controller._compute_action_context(context)
        assert "Sep 13" in digest and "Dec 8" in digest
        assert "Use all-day entries" in digest

        events = [
            {"summary": f"MGT 6203 HW {i} due", "start_time": date,
             "end_time": end, "all_day": True}
            for i, (date, end) in enumerate([
                ("2026-09-13", "2026-09-14"),
                ("2026-09-27", "2026-09-28"),
                ("2026-10-11", "2026-10-12"),
                ("2026-10-25", "2026-10-26"),
                ("2026-11-08", "2026-11-09"),
                ("2026-11-22", "2026-11-23"),
                ("2026-12-08", "2026-12-09"),
            ], start=1)
        ]
        call = {"function": {"name": "propose_action", "arguments": json.dumps({
            "action_type": "calendar_create_event", "reason": "user asked",
            "events": events,
        })}}
        decision = NativeToolsHandler(actions_available=True)._parse_single_tool_call(call)
        batched = controller._coalesce_action_decisions([decision])
        assert len(batched) == 1
        assert len(batched[0].action_params["events"]) == 7

        response = MagicMock(status_code=200)
        response.json.return_value = {"id": "event", "htmlLink": ""}
        client = AsyncMock()
        client.post.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=TestBatchExecution._auth()), \
             patch("httpx.AsyncClient", return_value=client):
            result = await create_calendar_event(ActionProposal(
                action_type=ActionType.CALENDAR_CREATE_EVENT,
                params=batched[0].action_params,
                summary=batched[0].action_summary,
            ))
        assert result.success is True
        assert client.post.await_count == 7


class TestBatchExecution:

    @staticmethod
    def _auth():
        auth = MagicMock()
        auth.is_authenticated = True
        auth.has_scope.return_value = True
        auth.get_credentials.return_value = MagicMock(token="token")
        return auth

    @staticmethod
    def _proposal(count=7):
        events = [
            {
                "summary": f"HW {i}",
                "start_time": f"2026-09-{i:02d}T22:59:00-05:00",
                "end_time": f"2026-09-{i:02d}T23:00:00-05:00",
            }
            for i in range(1, count + 1)
        ]
        return ActionProposal(
            action_type=ActionType.CALENDAR_CREATE_EVENT,
            params={"events": events},
            summary=f"calendar_create_event: {count} events",
        )

    @pytest.mark.asyncio
    async def test_one_approved_proposal_creates_all_seven(self):
        responses = []
        for i in range(7):
            response = MagicMock(status_code=200)
            response.json.return_value = {"id": f"e{i}", "htmlLink": ""}
            responses.append(response)
        client = AsyncMock()
        client.post.side_effect = responses
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=self._auth()), \
             patch("httpx.AsyncClient", return_value=client):
            result = await create_calendar_event(self._proposal())
        assert result.success is True
        assert client.post.await_count == 7
        assert "all 7 calendar events" in result.message

    @pytest.mark.asyncio
    async def test_invalid_later_event_prevents_any_partial_creation(self):
        proposal = self._proposal(2)
        proposal.params["events"][1].pop("end_time")
        client = AsyncMock()
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=self._auth()), \
             patch("httpx.AsyncClient", return_value=client):
            result = await create_calendar_event(proposal)
        assert result.success is False
        assert "event 2" in result.message.lower()
        client.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_day_event_uses_google_date_fields(self):
        proposal = ActionProposal(
            action_type=ActionType.CALENDAR_CREATE_EVENT,
            params={
                "summary": "MGT 6203 HW 1 due",
                "start_time": "2026-09-13",
                "end_time": "2026-09-14",
                "all_day": True,
            },
            summary="calendar_create_event: MGT 6203 HW 1 due",
        )
        response = MagicMock(status_code=200)
        response.json.return_value = {"id": "e1", "htmlLink": ""}
        client = AsyncMock()
        client.post.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=self._auth()), \
             patch("httpx.AsyncClient", return_value=client):
            result = await create_calendar_event(proposal)
        assert result.success is True
        body = client.post.await_args.kwargs["json"]
        assert body["start"] == {"date": "2026-09-13"}
        assert body["end"] == {"date": "2026-09-14"}
        assert "dateTime" not in body["start"]
        assert "timeZone" not in body["start"]

    @pytest.mark.asyncio
    async def test_batch_inherits_top_level_all_day(self):
        proposal = ActionProposal(
            action_type=ActionType.CALENDAR_CREATE_EVENT,
            params={
                "all_day": True,
                "events": [{
                    "summary": "MGT 6203 HW 1 due",
                    "start_time": "2026-09-13",
                    "end_time": "2026-09-14",
                }],
            },
            summary="calendar_create_event: 1 event",
        )
        response = MagicMock(status_code=200)
        response.json.return_value = {"id": "e1", "htmlLink": ""}
        client = AsyncMock()
        client.post.return_value = response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=self._auth()), \
             patch("httpx.AsyncClient", return_value=client):
            result = await create_calendar_event(proposal)
        assert result.success is True
        body = client.post.await_args.kwargs["json"]
        assert body["start"] == {"date": "2026-09-13"}
        assert body["end"] == {"date": "2026-09-14"}

    @pytest.mark.asyncio
    async def test_date_only_values_without_all_day_are_rejected(self):
        proposal = ActionProposal(
            action_type=ActionType.CALENDAR_CREATE_EVENT,
            params={
                "summary": "Ambiguous event",
                "start_time": "2026-09-13",
                "end_time": "2026-09-14",
            },
            summary="calendar_create_event: Ambiguous event",
        )
        client = AsyncMock()
        with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=self._auth()), \
             patch("httpx.AsyncClient", return_value=client):
            result = await create_calendar_event(proposal)
        assert result.success is False
        assert "invalid ISO 8601" in result.message
        client.post.assert_not_awaited()


class TestTruthfulStatusAndGrounding:

    def test_runtime_health_reports_calendar_write_scope(self):
        auth = MagicMock()
        auth.is_authenticated = True
        # Explicit since audit F15: a bare MagicMock auto-attr is truthy and
        # would read as an expired-unrefreshable token.
        auth.token_expired_no_refresh = False
        auth.has_scope.return_value = True
        with patch("config.app_config.INTERNET_ACTIONS_ENABLED", True), \
             patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("core.actions.google_auth.get_google_auth", return_value=auth):
            health = get_runtime_action_health()
        assert "calendar_create_event backend: AVAILABLE" in health
        assert "write scope granted" in health

    def test_enhanced_fallback_gets_authoritative_action_status(self):
        source = inspect.getsource(__import__("gui.handlers", fromlist=["_run_enhanced"])._run_enhanced)
        assert "get_runtime_action_health" in source
        assert "APPLICATION ACTION STATUS" in source
        assert "Never contradict an AVAILABLE backend" in source

    def test_agentic_tool_status_uses_same_authoritative_health(self):
        executor = object.__new__(ToolExecutor)
        for attr in (
            "web_search_manager", "chroma_store", "wolfram_manager",
            "file_access_manager", "git_stats_manager", "memory_expander",
            "github_manager",
        ):
            setattr(executor, attr, None)
        expected = (
            "propose_action: AVAILABLE (calendar_create_event)\n"
            "calendar_create_event backend: AVAILABLE "
            "(OAuth token present; calendar.events write scope granted)"
        )
        with patch("knowledge.semantic_search.is_faiss_available", return_value=False), \
             patch("core.actions.registry.get_runtime_action_health", return_value=expected):
            health = executor.get_tool_health()
        assert expected in health

    def test_impossible_weekday_correction_is_demoted(self):
        raw = json.dumps({
            "false_claim_present": True,
            "claim": "August 29, 2026 is Saturday",
            "why_false": "The correct weekday is Sunday.",
            "confidence": 1.0,
            "correction": "August 29, 2026 falls on a Sunday.",
        })
        verdict = _parse_verdict(raw)
        assert verdict is not None
        assert verdict.false_claim_present is False
        assert weekday_date_mismatches("Sunday, August 29, 2026")
        assert not weekday_date_mismatches("Saturday, August 29, 2026")

    @pytest.mark.asyncio
    async def test_grounding_verifier_always_receives_runtime_clock(self):
        import gui.handlers as handlers

        clean = json.dumps({
            "false_claim_present": False,
            "claim": "",
            "why_false": "",
            "confidence": 0.0,
            "correction": "",
        })
        mm = SimpleNamespace(generate_once=AsyncMock(return_value=clean))
        ctx = SimpleNamespace(
            user_text="What day is it?",
            telemetry={},
            orchestrator=SimpleNamespace(model_manager=mm),
            raw_context={"tone_level": "CONVERSATIONAL"},
        )
        await handlers._apply_grounding_check(
            ctx,
            "Today is Saturday, August 29, 2026, and the deadline is later this year.",
        )
        prompt = mm.generate_once.await_args.args[0]
        assert "AUTHORITATIVE RUNTIME CLOCK" in prompt
        # The clock block is built from the REAL runtime clock — assert
        # today's weekday/date, not the date the test was written (this
        # assert was red on every run after 2026-08-29).
        from datetime import datetime as _dt
        assert _dt.now().strftime("%A, %Y-%m-%d") in prompt


@pytest.mark.asyncio
async def test_background_conversation_log_uses_sanitized_response():
    import gui.handlers as handlers

    memory = SimpleNamespace(store_interaction=AsyncMock(return_value="mem-1"))
    orchestrator = SimpleNamespace(
        memory_system=memory,
        current_topic="Calendar Events",
    )
    conversation_logger = MagicMock()
    await handlers._background_store_interaction(
        orchestrator=orchestrator,
        merged_input="create events",
        response_to_store="Clean response",
        tags=["topic:Calendar Events"],
        user_text="create events",
        final_output="<|sep|>Clean response",
        personality="default",
        file_names=[],
        conversation_logger=conversation_logger,
    )
    assert conversation_logger.log_interaction.call_args.kwargs[
        "assistant_response"
    ] == "Clean response"
