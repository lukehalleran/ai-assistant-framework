"""Forced read-tool round + gate text-shape fixes (2026-09-27, plan F/X1).

Live probe evidence (PLAN_20260927_followup_tool_calls.md, "Live evidence"):
  1. "[test]Navient. Search that[/test]": gate correctly recognized a
     tool-thread continuation, but the [TOOL CONTINUATION] note is
     PROMPT-ONLY — a misbehaving model narrated tool intent in prose
     instead of calling/emitting anything, and a raw
     "<email_search>Navient</email_search>" marker shipped as the reply.
     class: BC-46, BC-49, BC-44.
  2. "[test]Don't search yet, just ask me whether I want you to search my
     email...[/test]": the Tier-1 email arm fired anyway — its negation
     check only looked BACK from the EMAIL NOUN's own position; "don't"
     scoped the REQUEST VERB ("search"), many tokens earlier, and was never
     seen. class: BC-02.
  3. "[test]yes please[/test]": `is_offer_affirmation` is False on the
     literal enveloped text (True on "yes please" alone), and
     `_last_offer_question` required the PRIOR reply's own final sentence
     to be the question — a live reply had a claim-y tail AFTER the real
     offer ("...before you go digging in Outlook? Running that now.").
     class: BC-58, BC-04, BC-48, BC-44.

Fix (this batch): gate._prior_tool_followup now resolves and returns
"target_tool" (gate._continuation_target_tool); the controller's round loop
FORCES that tool on round 1 (native tool_choice/tools_override via
controller._native_tool_definition, or an XML marker directive via
controller._build_xml_tool_force_prompt) — never together with a forced
WRITE action in the same round; gate._negated_request_verb_present closes
the negated-request-verb gap in the Tier-1 email arm and
gate._carries_prior_tool_cue; gate._last_offer_question now finds the last
question-sentence ANYWHERE in the reply; gate._envelope_inner_text judges
affirmation/request shape on the text inside a "[test]...[/test]" probe
envelope (a sibling batch's utils.test_envelope.inner_text, imported
lazily — this file stubs it via sys.modules so the mechanism is exercised
deterministically regardless of that module's landing order in this shared
tree).

Every test below is a "misbehaving model" fixture in the sense the plan
means: the fakes here MUST fail to demonstrate the fixed behavior on the
pristine base (no target_tool key at all, no forcing machinery, the old
narrower negation/offer-question/affirmation logic) — see each test's
docstring for the specific base-failure mode, and the batch handoff for the
cross-clone verification run.
"""
from __future__ import annotations

import asyncio
import sys
import types
import re
from unittest.mock import MagicMock

import pytest

from core.actions.types import ActionType
from core.agentic import tool_thread
from core.agentic.controller import (
    AgenticSearchController,
    _build_xml_tool_force_prompt,
    _native_tool_definition,
    _xml_tool_doc_block,
)
from core.agentic.gate import (
    TOOL_CONTINUATION_MAX_AGE_S,
    _carries_prior_tool_cue,
    _continuation_target_tool,
    _envelope_inner_text,
    _last_offer_question,
    _negated_request_verb_present,
    _prior_tool_followup,
    evaluate_agentic_gate,
)
from core.agentic.protocols import NativeToolsHandler
from core.agentic.types import SearchDecision, SearchProtocol


# ---------------------------------------------------------------------------
# Shared fixtures/helpers (mirrors test_sep27_tool_continuation.py)
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


def _gate(user_text, corpus=None):
    return asyncio.run(evaluate_agentic_gate(
        user_text=user_text, entity_resolver=None, model_manager=None,
        corpus_manager=corpus, intent_info=None,
    ))


@pytest.fixture
def _no_pending_cards():
    from unittest.mock import patch
    from core.agentic.tools import ToolExecutor
    store = ToolExecutor._get_pending_actions_store()
    with patch.object(store, "get_all_pending", return_value=[]):
        yield


@pytest.fixture
def _stub_test_envelope(monkeypatch):
    """A minimal, in-memory stand-in for the sibling batch's
    utils.test_envelope module (X2 creates the real file concurrently in
    this same tree — X1 must not). Recognizes the whole-message inline form
    the live probes used: "[test]...[/test]". Stubbing via sys.modules
    (never writing the file) makes the envelope-aware code path exercised
    deterministically regardless of landing order. Since the referee hoisted
    gate's import to module level (X2 landed), gate uses the real
    utils.test_envelope; the stub remains harmless for these tests."""
    mod = types.ModuleType("utils.test_envelope")
    envelope_re = re.compile(r"^\[test\](.*)\[/test\]$", re.IGNORECASE | re.DOTALL)

    def inner_text(text):
        m = envelope_re.match((text or "").strip())
        return m.group(1).strip() if m else text

    mod.inner_text = inner_text
    monkeypatch.setitem(sys.modules, "utils.test_envelope", mod)
    yield


# ---------------------------------------------------------------------------
# gate._envelope_inner_text — lazy import + graceful fallback
# ---------------------------------------------------------------------------

class TestEnvelopeInnerText:
    def test_uses_stubbed_module_when_present(self, _stub_test_envelope):
        assert _envelope_inner_text("[test]yes please[/test]") == "yes please"


# ---------------------------------------------------------------------------
# gate._negated_request_verb_present — item 5 (evidence #2, BC-02)
# ---------------------------------------------------------------------------

class TestNegatedRequestVerbPresent:
    def test_negated_verb_far_from_the_noun_is_caught(self):
        """The live probe #2 shape: 'don't' scopes 'search', eleven tokens
        before the email noun — the OLD Tier-1 check only looked back from
        the noun's own position and never saw it. Base has no
        `_negated_request_verb_present` at all (this call raises
        NameError/ImportError there)."""
        assert _negated_request_verb_present(
            "Don't search yet, just ask me whether I want you to search "
            "my email about the loan"
        ) is True

    def test_unnegated_verb_is_false(self):
        assert _negated_request_verb_present("check my gmail again") is False

    def test_envelope_tag_word_never_coincidentally_counts(self):
        """The literal word "test" inside "[test]"/"[/test]" tags is one of
        the reused request verbs — envelope-stripping first (this function
        judges the ENVELOPE-INNER text) keeps that coincidence from ever
        mattering."""
        assert _negated_request_verb_present("[test]yes please[/test]") is False


# ---------------------------------------------------------------------------
# Tier-1 email arm — the SAME gap, exercised end-to-end (item 5, evidence #2)
# ---------------------------------------------------------------------------

class TestTier1EmailArmNegatedVerb:
    def test_carries_prior_tool_cue_false_on_negated_request_verb(self):
        """Acceptance (d) at the _carries_prior_tool_cue layer: base returns
        True here (the email noun itself isn't negated at its own position;
        only the request verb, far earlier, is) — this is the exact
        BEFORE/AFTER flip this batch makes."""
        calls = [{"tool": "email_search", "args": {}}]
        assert _carries_prior_tool_cue(
            "Don't search yet, just ask me whether I want you to search my email",
            calls,
        ) is False

    @pytest.mark.usefixtures("_no_pending_cards")
    def test_gate_does_not_route_to_tools_on_the_live_shape(self):
        """Acceptance (d), full gate: base fires the Tier-1 email arm (or,
        via _prior_tool_followup's arm (c)/_carries_prior_tool_cue, a
        continuation) on this exact live shape — must FAIL on base since
        this negation gap is precisely what this batch closes."""
        d = _gate(
            "Don't search yet, just ask me whether I want you to search my "
            "email about the loan"
        )
        assert not (d.should_trigger and "tools" in (d.modes or [])), (
            f"email arm fired: {d.reason}"
        )


# ---------------------------------------------------------------------------
# gate._last_offer_question — item 3 (evidence #3)
# ---------------------------------------------------------------------------

class TestLastOfferQuestion:
    def test_finds_the_offer_when_it_is_not_the_final_sentence(self):
        """Acceptance (b): base requires the WHOLE reply to end in "?" —
        this reply's true offer question is followed by an unbacked-claim
        tail ("Running that now."), so base's `_last_offer_question`
        returns None here. Fails on base."""
        reply = (
            "I found nothing there. Want me to run one more Gmail search "
            'for "acme corp" specifically before you go digging in '
            "Outlook? Running that now."
        )
        assert _last_offer_question(reply) == (
            'Want me to run one more Gmail search for "acme corp" '
            "specifically before you go digging in Outlook?"
        )

    def test_still_finds_a_true_final_question(self):
        assert _last_offer_question("Found nothing. Want me to check further back?") == (
            "Want me to check further back?"
        )

    def test_no_question_sentence_is_none(self):
        assert _last_offer_question("Found nothing there. Running that now.") is None

    def test_empty_is_none(self):
        assert _last_offer_question("") is None


# ---------------------------------------------------------------------------
# gate._continuation_target_tool — item 1
# ---------------------------------------------------------------------------

class TestContinuationTargetTool:
    def test_email_search_in_prior_calls_wins(self):
        calls = [
            {"tool": "web_search", "args": {}},
            {"tool": "email_search", "args": {}},
        ]
        assert _continuation_target_tool(calls) == "email_search"

    def test_empty_prior_calls_with_email_cue_offer(self):
        assert _continuation_target_tool(
            [], "Want me to run one more Gmail search before you go digging in Outlook?"
        ) == "email_search"

    def test_empty_prior_calls_with_negated_email_cue_offer_is_unknown(self):
        assert _continuation_target_tool(
            [], "I won't touch your inbox unless you ask — anything else?"
        ) is None

    def test_falls_back_to_last_prior_call_tool(self):
        calls = [{"tool": "web_search", "args": {}}, {"tool": "search_memory", "args": {}}]
        assert _continuation_target_tool(calls) == "search_memory"

    def test_nothing_known_is_none(self):
        assert _continuation_target_tool([]) is None


# ---------------------------------------------------------------------------
# gate._prior_tool_followup — target_tool on every return + envelope judging
# ---------------------------------------------------------------------------

class TestPriorToolFollowupTargetToolAndEnvelope:
    def test_shape_a_affirmation_carries_target_tool(self):
        tool_thread.record_read_tool_calls(
            [{"tool": "email_search", "args": {"query": "loan", "window_days": 30}}]
        )
        offer = "Found nothing in the last 30 days. Want me to check further back?"
        corpus = _FakeCorpus([_entry(offer)])
        result = _prior_tool_followup("Yes please", corpus)
        assert result["target_tool"] == "email_search"

    def test_shape_b_request_shaped_carries_target_tool(self):
        tool_thread.record_read_tool_calls(
            [{"tool": "email_search", "args": {"query": "loan", "window_days": 60}}]
        )
        corpus = _FakeCorpus([_entry("I searched your inbox for the last 60 days and found nothing.")])
        result = _prior_tool_followup("Search that, it's Acme Corp", corpus)
        assert result["target_tool"] == "email_search"

    def test_extension_shape_carries_target_tool(self):
        offer_reply = (
            "I found nothing there. Want me to run one more Gmail search for "
            '"acme corp" specifically before you go digging in Outlook?'
        )
        corpus = _FakeCorpus([_entry(offer_reply, response_mode="agentic-search")])
        result = _prior_tool_followup("Yes please", corpus)
        assert result["target_tool"] == "email_search"

    def test_acceptance_c_enveloped_affirmation_fires(self, _stub_test_envelope):
        """Acceptance (c): base's is_offer_affirmation("[test]yes
        please[/test]") is False (verified against the deployed function;
        bare "yes please" is True) — _prior_tool_followup must therefore
        return None on base for this exact input. Fails on base."""
        tool_thread.record_read_tool_calls(
            [{"tool": "email_search", "args": {"query": "loan", "window_days": 30}}]
        )
        offer = "Found nothing in the last 30 days. Want me to check further back?"
        corpus = _FakeCorpus([_entry(offer)])
        result = _prior_tool_followup("[test]yes please[/test]", corpus)
        assert result is not None
        assert result["accepted_offer"] == "Want me to check further back?"
        assert result["target_tool"] == "email_search"


# ---------------------------------------------------------------------------
# controller._native_tool_definition / XML doc-block helpers
# ---------------------------------------------------------------------------

class TestNativeToolDefinition:
    def test_finds_email_search_by_name(self):
        handler = NativeToolsHandler(email_search_available=True)
        d = _native_tool_definition(handler, "email_search")
        assert d is not None
        assert d["function"]["name"] == "email_search"

    def test_alias_resolves_search_stackexchange(self):
        handler = NativeToolsHandler()
        d = _native_tool_definition(handler, "stackexchange")
        assert d is not None
        assert d["function"]["name"] == "search_stackexchange"

    def test_unavailable_tool_is_none(self):
        handler = NativeToolsHandler(email_search_available=False)
        assert _native_tool_definition(handler, "email_search") is None

    def test_unknown_tool_is_none(self):
        handler = NativeToolsHandler()
        assert _native_tool_definition(handler, "not_a_real_tool") is None


class TestXmlToolForcePrompt:
    def test_email_search_doc_block_found(self):
        block = _xml_tool_doc_block("email_search")
        assert block is not None
        assert "<email_search" in block

    def test_force_prompt_carries_directive(self):
        text = _build_xml_tool_force_prompt("email_search")
        assert text is not None
        assert "[TOOL EXECUTION DIRECTIVE]" in text
        assert "<email_search" in text
        assert "no prose, markers only" in text

    def test_tool_with_no_xml_doc_is_none(self):
        assert _build_xml_tool_force_prompt("recall_image") is None

    def test_unknown_target_tool_is_none(self):
        assert _build_xml_tool_force_prompt("not_a_real_tool") is None


# ---------------------------------------------------------------------------
# End-to-end controller: forced round actually dispatches (acceptance a, e)
# ---------------------------------------------------------------------------

class TestForcedReadToolRoundEndToEnd:
    @pytest.fixture
    def controller(self):
        manager = MagicMock()
        manager.api_models = {}
        return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())

    @pytest.mark.asyncio
    async def test_acceptance_a_native_force_makes_the_tool_execute(self, controller, monkeypatch):
        """Acceptance (a): a misbehaving model NARRATES (no tool call) on
        round 1 unless tool_choice/tools_override forces email_search. Base
        has no target_tool at all — round 1 always sees tool_choice="auto",
        so this fake model always narrates and _execute_email_search is
        NEVER called there. Fails on base."""
        captured: dict = {}
        calls = {"n": 0}

        async def fake_decision(prompt, system_prompt, model_name, handler, session,
                                 tool_choice="auto", tools_override=None, forced_action_type=None):
            calls["n"] += 1
            if calls["n"] == 1:
                captured["tool_choice"] = tool_choice
                captured["tools_override"] = tools_override
                if (isinstance(tool_choice, dict)
                        and tool_choice.get("function", {}).get("name") == "email_search"):
                    return [SearchDecision(
                        wants_email_search=True, email_query="Acme Corp", email_window_days=None,
                    )]
                # Misbehaving: narrates tool intent in prose, calls nothing.
                return [SearchDecision(wants_answer=True)]
            return [SearchDecision(is_done=True)]

        executed = {"n": 0}

        async def fake_execute_email_search(query, window_days):
            executed["n"] += 1
            executed["query"] = query
            return "no matches"

        async def fake_final(query, system_prompt, model_name, session, initial_context=None):
            yield "done"

        monkeypatch.setattr(controller, "detect_protocol", lambda model_name: SearchProtocol.NATIVE_TOOLS)
        monkeypatch.setattr(controller, "_email_search_is_available", lambda: True)
        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller._tool_executor, "_execute_email_search", fake_execute_email_search)
        monkeypatch.setattr(controller, "_generate_final_response", fake_final)

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
                "target_tool": "email_search",
            },
        ):
            out.append(ev)

        assert captured.get("tool_choice") == {"type": "function", "function": {"name": "email_search"}}
        assert captured.get("tools_override") is not None
        assert len(captured["tools_override"]) == 1
        assert captured["tools_override"][0]["function"]["name"] == "email_search"
        assert executed["n"] == 1, "the misbehaving model never actually ran email_search"
        assert executed["query"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_acceptance_e_write_action_force_wins(self, controller, monkeypatch):
        """Acceptance (e): an explicit write-action request (detect_action_intent
        hits on the query itself) must keep precedence over a SIMULTANEOUSLY
        set tool_continuation target_tool — the two are never forced together
        in one round. This exercises code this batch adds; on base there is
        no read-tool forcing to conflict with at all, so this specific
        precedence guard doesn't exist there (nothing to assert against)."""
        captured: dict = {}
        calls = {"n": 0}

        async def fake_decision(prompt, system_prompt, model_name, handler, session,
                                 tool_choice="auto", tools_override=None, forced_action_type=None):
            calls["n"] += 1
            if calls["n"] == 1:
                # Round 1 is what this test cares about (write-force
                # precedence). Return is_done without actually proposing
                # anything — the loop's own forced-round-failure retry then
                # runs round 2 unforced; captured stays pinned to round 1.
                captured["tool_choice"] = tool_choice
                captured["tools_override"] = tools_override
            return [SearchDecision(is_done=True)]

        async def fake_final(query, system_prompt, model_name, session, initial_context=None):
            yield "done"

        monkeypatch.setattr(controller, "detect_protocol", lambda model_name: SearchProtocol.NATIVE_TOOLS)
        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_generate_final_response", fake_final)

        out = []
        async for ev in controller.run_agentic_search(
            query="send an email to Morgan about the update",
            system_prompt="sys",
            model_name="glm-5.2",
            initial_search_terms=[],
            skip_initial_search=True,
            tool_continuation={
                "prior_calls": [{"tool": "email_search", "args": {"query": "x"}}],
                "accepted_offer": None,
                "target_tool": "email_search",
            },
        ):
            out.append(ev)

        assert captured.get("tool_choice") == {"type": "function", "function": {"name": "propose_action"}}
        names = captured.get("tools_override") or []
        assert names and names[0]["function"]["name"] == "propose_action"

    @pytest.mark.asyncio
    async def test_no_target_tool_leaves_round_unforced(self, controller, monkeypatch):
        """Regression: a continuation with target_tool=None (unknown) must
        leave tool_choice untouched — prompt-only, exactly as before this
        batch — never crash trying to force nothing."""
        captured: dict = {}

        async def fake_decision(prompt, system_prompt, model_name, handler, session,
                                 tool_choice="auto", tools_override=None, forced_action_type=None):
            captured["tool_choice"] = tool_choice
            captured["tools_override"] = tools_override
            return [SearchDecision(is_done=True)]

        async def fake_final(query, system_prompt, model_name, session, initial_context=None):
            yield "done"

        monkeypatch.setattr(controller, "detect_protocol", lambda model_name: SearchProtocol.NATIVE_TOOLS)
        monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
        monkeypatch.setattr(controller, "_generate_final_response", fake_final)

        out = []
        async for ev in controller.run_agentic_search(
            query="anything else going on",
            system_prompt="sys",
            model_name="glm-5.2",
            initial_search_terms=[],
            skip_initial_search=True,
            tool_continuation={"prior_calls": [], "accepted_offer": None, "target_tool": None},
        ):
            out.append(ev)

        assert captured.get("tool_choice") == "auto"
        assert captured.get("tools_override") is None
