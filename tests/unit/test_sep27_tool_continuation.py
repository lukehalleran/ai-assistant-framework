"""Tool-thread continuation (2026-09-27, session-audit plan E2).

Evidence (PLAN_20260927_session_audit_fixes.md, section "E2"):
  1. "Aidvantage. Search that" (prior turn ran email_search 60d) -> web
     search: the email Tier-1 arm needs an email noun in the CURRENT
     message; Tier 4 has no email output. class: BC-58, BC-15.
  2. "Yes please" after a reply ending "Want me to run one more Gmail
     search for 'X' ...?" -> nothing: the continuation override only sets
     needs_tools when request-shaped, and _prior_turn_offer_action covers
     WRITE actions only. class: BC-58, BC-74.
  3. "No it's for sure gmail maybe it was more than 2 weeks ago" -> no
     re-run: statement-shaped, _is_info_seeking false. class: BC-04, BC-58.

Fix: core/agentic/tool_thread.py records the READ tools the agentic
controller dispatched each turn; core/agentic/gate.py's
_prior_tool_followup() recognizes a same-thread follow-up on the NEXT turn
and routes it back to tools (AgenticDecision.tool_continuation); the
controller prepends a [TOOL CONTINUATION] note to the FIRST decision round.

Per the plan doctrine: neutral fixture words are used throughout (never the
live vendor/servicer names) — "gmail"/"outlook"/"email"/"inbox" are
pre-existing generic, categorized terms already used by the deployed
email-cue regex (gate.py Tier 1), not owner vocabulary. Some live phrasings
(e.g. "Aidvantage. Search that") don't literally satisfy the SAME arm once
reworded with a neutral entity name (the request-shape regex is head-
anchored), so fixtures below are reshaped to exercise each arm's actual
mechanism rather than copying the incident text verbatim.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from core.actions.types import ActionType
from core.agentic import tool_thread
from core.agentic.controller import (
    AgenticSearchController,
    _build_tool_continuation_prompt,
    _describe_tool_call,
    _read_tool_call_record,
)
from core.agentic.gate import (
    TOOL_CONTINUATION_MAX_AGE_S,
    AgenticDecision,
    _carries_prior_tool_cue,
    _is_prior_tool_narration,
    _prior_tool_followup,
    evaluate_agentic_gate,
)
from core.agentic.types import SearchDecision


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class _FakeCorpus:
    """Minimal corpus_manager stand-in — get_recent_memories(n) returns the
    most recent N entries, newest first (matches the real contract)."""

    def __init__(self, entries):
        self._entries = entries

    def get_recent_memories(self, n=1):
        return self._entries[:n]


def _entry(response, response_mode="agentic-search", query="q", timestamp=None):
    e = {"query": query, "response": response, "response_mode": response_mode}
    if timestamp is not None:
        e["timestamp"] = timestamp
    return e


@pytest.fixture(autouse=True)
def _reset_tool_thread():
    tool_thread.reset()
    yield
    tool_thread.reset()


def _gate(user_text, corpus):
    return asyncio.run(evaluate_agentic_gate(
        user_text=user_text, entity_resolver=None, model_manager=None,
        corpus_manager=corpus, intent_info=None,
    ))


# ---------------------------------------------------------------------------
# core/agentic/tool_thread.py — the in-process slot
# ---------------------------------------------------------------------------

class TestToolThreadSlot:
    def test_record_then_recent_round_trips(self):
        calls = [{"tool": "email_search", "args": {"query": "loan", "window_days": 60}}]
        tool_thread.record_read_tool_calls(calls)
        assert tool_thread.recent_read_tool_calls(TOOL_CONTINUATION_MAX_AGE_S) == calls

    def test_empty_list_is_a_deliberate_clear(self):
        tool_thread.record_read_tool_calls([{"tool": "web_search", "args": {"query": "x"}}])
        tool_thread.record_read_tool_calls([])
        assert tool_thread.recent_read_tool_calls(TOOL_CONTINUATION_MAX_AGE_S) == []

    def test_never_recorded_is_empty(self):
        assert tool_thread.recent_read_tool_calls(TOOL_CONTINUATION_MAX_AGE_S) == []

    def test_stale_record_is_not_returned(self):
        tool_thread.record_read_tool_calls([{"tool": "email_search", "args": {"query": "loan"}}])
        tool_thread._STATE["ts"] -= (TOOL_CONTINUATION_MAX_AGE_S + 1)
        assert tool_thread.recent_read_tool_calls(TOOL_CONTINUATION_MAX_AGE_S) == []

    def test_reset_clears(self):
        tool_thread.record_read_tool_calls([{"tool": "email_search", "args": {}}])
        tool_thread.reset()
        assert tool_thread.recent_read_tool_calls(TOOL_CONTINUATION_MAX_AGE_S) == []


# ---------------------------------------------------------------------------
# controller._read_tool_call_record — READ-vs-WRITE classification
# ---------------------------------------------------------------------------

class TestReadToolCallRecord:
    def test_email_search(self):
        d = SearchDecision(wants_email_search=True, email_query="loan", email_window_days=60)
        assert _read_tool_call_record(d, "_dispatch_email_search") == {
            "tool": "email_search", "args": {"query": "loan", "window_days": 60},
        }

    def test_web_search(self):
        d = SearchDecision(wants_search=True, search_query="acme corp news")
        assert _read_tool_call_record(d, "_dispatch_web_search") == {
            "tool": "web_search", "args": {"query": "acme corp news"},
        }

    def test_search_memory(self):
        d = SearchDecision(wants_memory_search=True, memory_query="q", memory_collection="facts")
        assert _read_tool_call_record(d, "_dispatch_memory_search") == {
            "tool": "search_memory", "args": {"query": "q", "collection": "facts"},
        }

    def test_file_read(self):
        d = SearchDecision(wants_file_read=True, file_read_path="docs/x.md")
        assert _read_tool_call_record(d, "_dispatch_file_read") == {
            "tool": "file_read", "args": {"path": "docs/x.md"},
        }

    def test_propose_action_never_recorded(self):
        d = SearchDecision(wants_action=True, action_type="send_email")
        assert _read_tool_call_record(d, "_dispatch_action_proposal") is None

    def test_create_daemon_note_never_recorded(self):
        d = SearchDecision(wants_create_daemon_note=True, daemon_note_title="t")
        assert _read_tool_call_record(d, "_dispatch_create_daemon_note") is None

    def test_unrecognized_decision_is_none(self):
        assert _read_tool_call_record(SearchDecision(), "_dispatch_wolfram") is None


class TestBuildToolContinuationPrompt:
    def test_renders_prior_calls_and_offer(self):
        text = _build_tool_continuation_prompt({
            "prior_calls": [{"tool": "email_search", "args": {"query": "loan", "window_days": 60}}],
            "accepted_offer": 'Want me to check further back?',
        })
        assert "[TOOL CONTINUATION]" in text
        assert "email_search(query='loan', window_days=60)" in text
        assert "Want me to check further back?" in text
        assert "carry it out now" in text

    def test_renders_without_offer(self):
        text = _build_tool_continuation_prompt({
            "prior_calls": [{"tool": "web_search", "args": {"query": "x"}}],
            "accepted_offer": None,
        })
        assert "carry it out now" not in text
        assert "run the relevant tool again" in text

    def test_describe_tool_call_no_args(self):
        assert _describe_tool_call({"tool": "search_memory", "args": {}}) == "search_memory()"


# ---------------------------------------------------------------------------
# controller._dispatch_single_inner — the single recording chokepoint
# ---------------------------------------------------------------------------

class TestDispatchSingleInnerRecords:
    @pytest.fixture
    def controller(self):
        manager = MagicMock()
        manager.api_models = {}
        return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())

    @pytest.mark.asyncio
    async def test_read_tool_dispatch_is_recorded(self, controller):
        controller._read_tool_calls_this_turn = []

        async def fake_memory_search(decision, round_number):
            return "ok"
        controller._dispatch_memory_search = fake_memory_search

        decision = SearchDecision(
            wants_memory_search=True, memory_query="loan status", memory_collection="facts",
        )
        result = await controller._dispatch_single_inner(decision, 1, None, None, None)
        assert result == "ok"
        assert controller._read_tool_calls_this_turn == [
            {"tool": "search_memory", "args": {"query": "loan status", "collection": "facts"}}
        ]

    @pytest.mark.asyncio
    async def test_propose_action_is_never_recorded(self, controller):
        controller._read_tool_calls_this_turn = []

        async def fake_action(decision, round_number):
            return "proposed"
        controller._dispatch_action_proposal = fake_action

        decision = SearchDecision(wants_action=True, action_type="send_email")
        await controller._dispatch_single_inner(decision, 1, None, None, None)
        assert controller._read_tool_calls_this_turn == []

    @pytest.mark.asyncio
    async def test_missing_attribute_does_not_raise(self):
        """A controller built via __new__ (bypassing __init__/run_agentic_search
        — the exact shape test_tool_wiring_parity.py drives) has no turn in
        progress; the hasattr guard must not crash dispatch."""
        controller = AgenticSearchController.__new__(AgenticSearchController)

        async def fake_memory_search(decision, round_number):
            return "ok"
        controller._dispatch_memory_search = fake_memory_search

        decision = SearchDecision(
            wants_memory_search=True, memory_query="q", memory_collection="facts",
        )
        result = await controller._dispatch_single_inner(decision, 1, None, None, None)
        assert result == "ok"
        assert controller._read_tool_calls_this_turn == [
            {"tool": "search_memory", "args": {"query": "q", "collection": "facts"}}
        ]


# ---------------------------------------------------------------------------
# gate._prior_tool_followup — the three continuation shapes
# ---------------------------------------------------------------------------

class TestPriorToolFollowup:
    def test_shape_b_terse_request_shaped_after_email_search(self):
        """Arm (b): a terse (<=12 words), request-shaped follow-up after a
        recorded email_search — the reworded live shape #1 ('search that
        <name>' rather than '<name>. search that', since _is_request_shaped
        is head-anchored on the deployed regex)."""
        tool_thread.record_read_tool_calls(
            [{"tool": "email_search", "args": {"query": "loan", "window_days": 60}}]
        )
        corpus = _FakeCorpus([_entry("I searched your inbox for the last 60 days and found nothing.")])
        result = _prior_tool_followup("Search that, it's Acme Corp", corpus)
        assert result == {
            "prior_calls": [{"tool": "email_search", "args": {"query": "loan", "window_days": 60}}],
            "accepted_offer": None,
        }

    def test_shape_b_live_order_term_then_imperative(self):
        """The LIVE order: the new term first, the imperative as its own
        sentence ("<Name>. Search that") — request shape is judged per
        sentence, so the head-anchored check still sees "Search that"."""
        tool_thread.record_read_tool_calls(
            [{"tool": "email_search", "args": {"query": "loan", "window_days": 60}}]
        )
        corpus = _FakeCorpus([_entry("I searched your inbox for the last 60 days and found nothing.")])
        result = _prior_tool_followup("Acmecorp. Search that", corpus)
        assert result is not None and result["prior_calls"][0]["tool"] == "email_search"

    def test_shape_c_cue_noun_after_email_search(self):
        """Arm (c): the current message names the un-negated email cue noun
        (a pre-existing, categorized term — not owner vocabulary) — the
        reworded live shape #3."""
        tool_thread.record_read_tool_calls(
            [{"tool": "email_search", "args": {"query": "loan", "window_days": 14}}]
        )
        corpus = _FakeCorpus([_entry("I searched your inbox for the last 14 days and found nothing.")])
        result = _prior_tool_followup(
            "No it's for sure gmail, maybe it was more than 2 weeks ago", corpus,
        )
        assert result == {
            "prior_calls": [{"tool": "email_search", "args": {"query": "loan", "window_days": 14}}],
            "accepted_offer": None,
        }

    def test_shape_a_affirmation_of_offer_with_recorded_calls(self):
        """Arm (a), primary path: prior_calls non-empty, prior reply ends on
        a question, current text affirms it."""
        tool_thread.record_read_tool_calls(
            [{"tool": "email_search", "args": {"query": "loan", "window_days": 30}}]
        )
        offer = "Found nothing in the last 30 days. Want me to check further back?"
        corpus = _FakeCorpus([_entry(offer)])
        result = _prior_tool_followup("Yes please", corpus)
        assert result["accepted_offer"] == "Want me to check further back?"
        assert result["prior_calls"] == [{"tool": "email_search", "args": {"query": "loan", "window_days": 30}}]

    def test_shape_a_extension_empty_calls_agentic_prior_email_offer(self):
        """Arm (a), extension: NO trackable read-tool record survived (Round
        1's own web search bypasses the dispatch chokepoint), but the prior
        turn's response_mode says agentic, and the offer question itself
        carries the email cue — the live 16:18 case (evidence #2)."""
        # tool_thread slot is empty (autouse fixture reset it; nothing recorded)
        offer_reply = (
            "I found nothing there. Want me to run one more Gmail search for "
            '"acme corp" specifically before you go digging in Outlook?'
        )
        corpus = _FakeCorpus([_entry(offer_reply, response_mode="agentic-search")])
        result = _prior_tool_followup("Yes please", corpus)
        assert result["prior_calls"] == []
        assert result["accepted_offer"].endswith("digging in Outlook?")

    def test_shape_a_extension_needs_agentic_prior_mode(self):
        """Same offer question, but the prior turn was NOT agentic — must
        not fire (nothing to continue)."""
        offer_reply = (
            "Want me to run one more Gmail search for \"acme corp\" before "
            "you go digging in Outlook?"
        )
        corpus = _FakeCorpus([_entry(offer_reply, response_mode="enhanced")])
        assert _prior_tool_followup("Yes please", corpus) is None

    def test_negative_no_prior_calls_non_agentic_prior(self):
        """'yes please' with no prior read calls and a non-agentic prior ->
        unchanged."""
        corpus = _FakeCorpus([_entry("Doing well, anything else?", response_mode="enhanced")])
        assert _prior_tool_followup("Yes please", corpus) is None

    def test_negative_self_narration_does_not_fire_c(self):
        """The message is the user narrating their OWN action, not
        continuing the search thread — the narration guard on arm (c)."""
        tool_thread.record_read_tool_calls(
            [{"tool": "email_search", "args": {"query": "loan"}}]
        )
        corpus = _FakeCorpus([_entry("I searched your inbox and found nothing.")])
        assert _prior_tool_followup("I email them every week", corpus) is None

    def test_negative_prior_calls_older_than_max_age(self):
        tool_thread.record_read_tool_calls(
            [{"tool": "email_search", "args": {"query": "loan"}}]
        )
        tool_thread._STATE["ts"] -= (TOOL_CONTINUATION_MAX_AGE_S + 1)
        corpus = _FakeCorpus([_entry("I searched your inbox and found nothing.")])
        assert _prior_tool_followup("Search that, it's Acme Corp", corpus) is None

    def test_negative_extension_older_than_max_age(self):
        """The (a)-extension path is ALSO bounded by recency of the prior
        turn itself, via the corpus entry's own timestamp."""
        offer_reply = (
            "Want me to run one more Gmail search for \"acme corp\" before "
            "you go digging in Outlook?"
        )
        stale_ts = (datetime.now() - timedelta(seconds=TOOL_CONTINUATION_MAX_AGE_S + 500)).isoformat()
        corpus = _FakeCorpus([_entry(offer_reply, response_mode="agentic-search", timestamp=stale_ts)])
        assert _prior_tool_followup("Yes please", corpus) is None

    def test_no_corpus_manager_is_safe(self):
        assert _prior_tool_followup("Search that", None) is None

    def test_no_user_text_is_safe(self):
        assert _prior_tool_followup("", _FakeCorpus([_entry("x?")])) is None


class TestCarriesPriorToolCueAndNarration:
    def test_cue_present_and_not_negated(self):
        calls = [{"tool": "email_search", "args": {}}]
        assert _carries_prior_tool_cue("check my gmail again", calls) is True

    def test_negated_cue_does_not_count(self):
        calls = [{"tool": "email_search", "args": {}}]
        assert _carries_prior_tool_cue("don't check gmail for this", calls) is False

    def test_no_email_search_in_prior_calls(self):
        calls = [{"tool": "web_search", "args": {}}]
        assert _carries_prior_tool_cue("check my gmail again", calls) is False

    def test_narration_guard_blocks_direct_subject_adjacent_form(self):
        assert _is_prior_tool_narration("I email them every week", "I email them every week".index("email")) is True

    def test_non_narration_form_is_false(self):
        text = "check my gmail again"
        assert _is_prior_tool_narration(text, text.index("gmail")) is False


# ---------------------------------------------------------------------------
# evaluate_agentic_gate — full end-to-end routing + write-action precedence
# ---------------------------------------------------------------------------

class TestGateToolContinuationEndToEnd:
    @pytest.fixture(autouse=True)
    def _no_pending_cards(self):
        from core.agentic.tools import ToolExecutor
        store = ToolExecutor._get_pending_actions_store()
        with patch.object(store, "get_all_pending", return_value=[]):
            yield

    def test_shape_b_routes_to_tools_with_continuation(self):
        tool_thread.record_read_tool_calls(
            [{"tool": "email_search", "args": {"query": "loan", "window_days": 60}}]
        )
        corpus = _FakeCorpus([_entry("I searched your inbox for the last 60 days and found nothing.")])
        d = _gate("Search that, it's Acme Corp", corpus)
        assert d.should_trigger
        assert d.modes == ["tools"]
        assert d.veto_exempt
        assert d.skip_initial_search
        assert d.tool_continuation is not None
        assert d.tool_continuation["prior_calls"][0]["tool"] == "email_search"

    def test_shape_c_routes_to_tools_with_continuation(self):
        tool_thread.record_read_tool_calls(
            [{"tool": "email_search", "args": {"query": "loan", "window_days": 14}}]
        )
        corpus = _FakeCorpus([_entry("I searched your inbox for the last 14 days and found nothing.")])
        d = _gate("No it's for sure gmail, maybe it was more than 2 weeks ago", corpus)
        assert d.should_trigger
        assert d.tool_continuation is not None

    def test_shape_a_extension_routes_to_tools_with_continuation(self):
        offer_reply = (
            "I found nothing there. Want me to run one more Gmail search for "
            '"acme corp" specifically before you go digging in Outlook?'
        )
        corpus = _FakeCorpus([_entry(offer_reply, response_mode="agentic-search")])
        d = _gate("Yes please", corpus)
        assert d.should_trigger
        assert d.modes == ["tools"]
        assert d.tool_continuation is not None
        assert d.tool_continuation["accepted_offer"].endswith("digging in Outlook?")

    def test_write_action_offer_keeps_precedence(self):
        """A pending WRITE-action offer (calendar create) still wins over a
        stale read-tool thread — _prior_turn_offer_action is checked FIRST
        and returns before _prior_tool_followup ever runs."""
        tool_thread.record_read_tool_calls(
            [{"tool": "email_search", "args": {"query": "loan"}}]
        )
        offer = "Want me to create the follow-up event now?"
        corpus = _FakeCorpus([_entry(offer)])
        d = _gate("please create", corpus)
        assert d.forced_action == ActionType.CALENDAR_CREATE_EVENT.value
        assert d.tool_continuation is None

    def test_negative_case_leaves_tiers_alone(self):
        corpus = _FakeCorpus([_entry("Doing well, anything else?", response_mode="enhanced")])
        d = _gate("Yes please", corpus)
        assert d.tool_continuation is None


# ---------------------------------------------------------------------------
# AgenticDecision.tool_continuation field
# ---------------------------------------------------------------------------

class TestAgenticDecisionField:
    def test_defaults_to_none(self):
        assert AgenticDecision(should_trigger=False).tool_continuation is None

    def test_accepts_a_dict(self):
        d = AgenticDecision(should_trigger=True, tool_continuation={"prior_calls": [], "accepted_offer": None})
        assert d.tool_continuation == {"prior_calls": [], "accepted_offer": None}


# ---------------------------------------------------------------------------
# Controller integration: tool_continuation threads to the first decision
# round's prompt, and this turn's read-tool dispatches are recorded at loop
# end regardless of how the loop exits.
# ---------------------------------------------------------------------------

class TestControllerIntegration:
    @pytest.fixture
    def controller(self):
        manager = MagicMock()
        manager.api_models = {}
        return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())

    @pytest.mark.asyncio
    async def test_continuation_note_in_first_round_and_recorded_at_loop_end(self, controller, monkeypatch):
        captured_system_prompts = []
        calls = {"n": 0}

        async def fake_decision(prompt, system_prompt, model_name, handler, session,
                                 tool_choice="auto", tools_override=None, forced_action_type=None):
            calls["n"] += 1
            captured_system_prompts.append(system_prompt)
            if calls["n"] == 1:
                return [SearchDecision(
                    wants_memory_search=True, memory_query="loan status", memory_collection="facts",
                )]
            return [SearchDecision(is_done=True)]

        async def fake_execute_memory_search(query, collection):
            return "no relevant memories"

        async def fake_final(query, system_prompt, model_name, session, initial_context=None):
            yield "done"

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_execute_memory_search", fake_execute_memory_search)
        monkeypatch.setattr(controller, "_generate_final_response", fake_final)

        recorded = {}

        def fake_record(calls_list):
            recorded["calls"] = list(calls_list)
        monkeypatch.setattr(tool_thread, "record_read_tool_calls", fake_record)

        out = []
        async for ev in controller.run_agentic_search(
            query="Search that, it's Acme Corp",
            system_prompt="sys",
            model_name="glm-5.2",
            initial_search_terms=[],
            skip_initial_search=True,
            tool_continuation={
                "prior_calls": [{"tool": "email_search", "args": {"query": "loan", "window_days": 60}}],
                "accepted_offer": None,
            },
        ):
            out.append(ev)

        assert captured_system_prompts, "no decision round ran"
        assert "[TOOL CONTINUATION]" in captured_system_prompts[0]
        assert "email_search(query='loan', window_days=60)" in captured_system_prompts[0]
        # Not re-injected on the second (is_done) round.
        assert "[TOOL CONTINUATION]" not in captured_system_prompts[-1] or len(captured_system_prompts) == 1

        assert recorded["calls"] == [
            {"tool": "search_memory", "args": {"query": "loan status", "collection": "facts"}}
        ]

    @pytest.mark.asyncio
    async def test_no_continuation_means_no_block_and_empty_record(self, controller, monkeypatch):
        captured_system_prompts = []

        async def fake_decision(prompt, system_prompt, model_name, handler, session,
                                 tool_choice="auto", tools_override=None, forced_action_type=None):
            captured_system_prompts.append(system_prompt)
            return [SearchDecision(is_done=True)]

        async def fake_final(query, system_prompt, model_name, session, initial_context=None):
            yield "done"

        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_generate_final_response", fake_final)

        recorded = {}

        def fake_record(calls_list):
            recorded["calls"] = list(calls_list)
        monkeypatch.setattr(tool_thread, "record_read_tool_calls", fake_record)

        out = []
        async for ev in controller.run_agentic_search(
            query="anything else going on",
            system_prompt="sys",
            model_name="glm-5.2",
            initial_search_terms=[],
            skip_initial_search=True,
        ):
            out.append(ev)

        assert all("[TOOL CONTINUATION]" not in sp for sp in captured_system_prompts)
        assert recorded["calls"] == []
