"""F12 (2026-09-09, docs/HANDOFF_20260909_independent_bug_audit.md /
docs/PLAN_20260909_audit_repairs.md): a forced decision round proposes the
WRONG action_type, gets rejected, retries blind, and the final reply narrates
a card that was never created.

Live incident: gate `Tier 1: explicit write action detected
(calendar_delete_event)` -> controller "forcing propose_action on first
decision round" -> protocol `propose_action rejected: action_type=
'calendar_create_event' is unknown or missing required fields` -> "Forced
action round produced no action marker -- retrying once" -> "Model ready to
answer (implicit)" -> final reply "Queued the deletion: ... Confirm and it's
off" with NO pending card and no NO_CARD_NOTICE.

Root cause found while implementing this fix: PROPOSE_ACTION_TOOL_DEFINITION
(core/agentic/types.py)'s action_type enum never included
calendar_update_event/calendar_delete_event at all -- a native-tools model
forced to delete had no valid way to express it and substituted the only
calendar option it could see (calendar_create_event).

Three deployed-function fixes exercised here:
(a) core.actions.registry.resolve_forced_action -- the single acceptance /
    coercion / rejection decision both protocol handlers call. Coercion is
    scoped to forced rounds ONLY.
(b) core.actions.registry.build_forced_tool_schema + the controller's forced
    native-tools round -- a per-round tool schema scoped to exactly the
    required type, and a retry prompt that carries the prior rejection
    reason (core.agentic.controller.AgenticSearchController.
    _build_xml_action_force_prompt for XML; the native inline prompt).
(c) core.action_claim_guard.claims_pending_card -- catches the exact
    "Queued the ... Confirm and it's off" narration shape so the no-card
    backstop fires when a forced round produces no proposal.
"""

import json
from types import SimpleNamespace

import pytest

from core.actions.registry import (
    ACTION_SPECS,
    build_forced_tool_schema,
    resolve_forced_action,
)
from core.actions.types import ActionType
from core.action_claim_guard import claims_pending_card
from core.agentic.controller import AgenticSearchController
from core.agentic.protocols import NativeToolsHandler, XMLMarkerHandler
from core.agentic.types import AgenticSearchSession, SearchProtocol


def _bare_controller() -> AgenticSearchController:
    """Controller instance without running __init__ (matches the pattern in
    test_calendar_turn_round2.py / test_calendar_turn_round3.py -- the
    methods under test only touch regex class attrs and stashed state)."""
    return object.__new__(AgenticSearchController)


# ===========================================================================
# (a) core.actions.registry.resolve_forced_action -- the shared decision
# ===========================================================================

class TestResolveForcedAction:
    def test_accepted_as_is_outside_a_forced_round(self):
        # A normal (non-forced) propose_action call behaves exactly as
        # before -- no forced_action_type means no coercion is attempted.
        resolved_type, params, reason = resolve_forced_action(
            "calendar_create_event",
            {"summary": "HW 1", "start_time": "2026-09-13T23:59:00",
             "end_time": "2026-09-13T23:59:59"},
            forced_action_type=None,
        )
        assert resolved_type == "calendar_create_event"
        assert params["summary"] == "HW 1"
        assert reason is None

    def test_wrong_type_coerced_when_params_fit_the_forced_spec(self):
        # The exact live shape: model proposes CREATE with delete-shaped
        # params (summary+date, no start/end) while calendar_delete_event
        # was required -- those params DO satisfy delete's required fields.
        resolved_type, params, reason = resolve_forced_action(
            "calendar_create_event",
            {"summary": "Psychiatrist appointment", "date": "2026-09-11",
             "reason": "user asked"},
            forced_action_type="calendar_delete_event",
        )
        assert resolved_type == "calendar_delete_event"
        assert params["summary"] == "Psychiatrist appointment"
        assert params["date"] == "2026-09-11"
        assert reason is None

    def test_valid_sibling_type_in_forced_round_is_never_accepted_as_is(self):
        # Referee case: a WELL-FORMED create (own spec satisfied) during a
        # forced DELETE round must not create the event the user asked to
        # delete. Its params do not fit delete (no `date`), so it is rejected
        # with a reason naming the required type.
        resolved_type, params, reason = resolve_forced_action(
            "calendar_create_event",
            {"summary": "Synthetic meeting", "start_time": "2026-09-13T15:00:00",
             "end_time": "2026-09-13T15:30:00"},
            forced_action_type="calendar_delete_event",
        )
        assert resolved_type is None and params is None
        assert "calendar_delete_event" in reason
        # The same proposal is accepted untouched when the forced type
        # matches — the tightening only bites on a MISMATCH.
        resolved_type, params, reason = resolve_forced_action(
            "calendar_create_event",
            {"summary": "Synthetic meeting", "start_time": "2026-09-13T15:00:00",
             "end_time": "2026-09-13T15:30:00"},
            forced_action_type="calendar_create_event",
        )
        assert resolved_type == "calendar_create_event" and reason is None

    def test_wrong_type_rejected_when_params_fit_neither_spec(self):
        resolved_type, params, reason = resolve_forced_action(
            "calendar_create_event",
            {"summary": "Psychiatrist appointment"},  # no date, no start/end
            forced_action_type="calendar_delete_event",
        )
        assert resolved_type is None
        assert params is None
        assert reason is not None
        assert "calendar_delete_event" in reason
        assert "date" in reason  # names the missing required field

    def test_never_coerces_outside_a_forced_round(self):
        # Same mismatched shape as above, but forced_action_type=None: must
        # reject on the CLAIMED type's own terms, never silently substitute.
        resolved_type, params, reason = resolve_forced_action(
            "calendar_create_event",
            {"summary": "Psychiatrist appointment", "date": "2026-09-11"},
            forced_action_type=None,
        )
        assert resolved_type is None
        assert "calendar_create_event" in reason

    def test_matching_type_not_reported_as_a_coercion(self):
        resolved_type, params, reason = resolve_forced_action(
            "calendar_delete_event",
            {"summary": "X", "date": "2026-09-11"},
            forced_action_type="calendar_delete_event",
        )
        assert resolved_type == "calendar_delete_event"
        assert reason is None

    def test_unregistered_action_type_reports_unrecognized(self):
        resolved_type, params, reason = resolve_forced_action(
            "delete_everything", {}, forced_action_type=None,
        )
        assert resolved_type is None
        assert "not a recognized action" in reason

    def test_backfillable_type_still_accepted_without_coercion(self):
        # github_create_issue has a backfill -- accepted even with no
        # required fields present, exactly like the pre-F12 behavior.
        resolved_type, params, reason = resolve_forced_action(
            "github_create_issue", {}, forced_action_type=None,
        )
        assert resolved_type == "github_create_issue"
        assert reason is None


# ===========================================================================
# (b) build_forced_tool_schema -- per-round scoped native tool definition
# ===========================================================================

class TestBuildForcedToolSchema:
    def test_calendar_delete_event_schema_has_only_that_type(self):
        schema = build_forced_tool_schema(ActionType.CALENDAR_DELETE_EVENT)
        props = schema["function"]["parameters"]["properties"]
        assert props["action_type"]["enum"] == ["calendar_delete_event"]
        for field in ("summary", "date", "event_id", "calendar_id"):
            assert field in props
        # CREATE-only fields must not leak into the delete schema.
        assert "start_time" not in props
        assert "end_time" not in props

    def test_calendar_update_event_schema_has_new_star_fields(self):
        schema = build_forced_tool_schema(ActionType.CALENDAR_UPDATE_EVENT)
        props = schema["function"]["parameters"]["properties"]
        assert props["action_type"]["enum"] == ["calendar_update_event"]
        for field in ("summary", "date", "new_summary", "new_start_time",
                      "new_end_time", "new_description", "new_location"):
            assert field in props

    def test_non_calendar_type_schema_scoped_too(self):
        schema = build_forced_tool_schema(ActionType.SEND_EMAIL)
        props = schema["function"]["parameters"]["properties"]
        assert props["action_type"]["enum"] == ["send_email"]
        assert "message" in props and "recipient" in props
        assert "start_time" not in props

    def test_unregistered_type_returns_none(self):
        class _Fake:
            value = "not_a_real_type"
        assert build_forced_tool_schema(_Fake()) is None

    def test_generic_tool_definition_now_advertises_update_and_delete(self):
        # The root cause: the generic (unscoped) tool's enum never included
        # these two -- a spontaneous (non-forced) native-tools proposal for
        # either was structurally impossible before this fix.
        from core.agentic.types import PROPOSE_ACTION_TOOL_DEFINITION
        enum_vals = set(
            PROPOSE_ACTION_TOOL_DEFINITION["function"]["parameters"]
            ["properties"]["action_type"]["enum"]
        )
        assert "calendar_update_event" in enum_vals
        assert "calendar_delete_event" in enum_vals
        props = PROPOSE_ACTION_TOOL_DEFINITION["function"]["parameters"]["properties"]
        for field in ("date", "event_id", "new_start_time", "new_end_time"):
            assert field in props


# ===========================================================================
# (c) NativeToolsHandler -- the exact parsing path the live incident hit
# ===========================================================================

def _tool_call(action_type, **params):
    args = {"action_type": action_type, "reason": "user asked", **params}
    return SimpleNamespace(function=SimpleNamespace(name="propose_action", arguments=json.dumps(args)))


class TestNativeToolsCoercion:
    def test_wrong_type_coerced_in_a_forced_round(self):
        handler = NativeToolsHandler(actions_available=True)
        tc = _tool_call("calendar_create_event", summary="Psychiatrist appointment", date="2026-09-11")
        decision = handler._parse_single_tool_call(tc, forced_action_type="calendar_delete_event")
        assert decision is not None
        assert decision.wants_action is True
        assert decision.action_type == "calendar_delete_event"
        assert decision.action_params["summary"] == "Psychiatrist appointment"
        assert decision.action_params["date"] == "2026-09-11"
        assert decision.action_reject_reason is None

    def test_wrong_type_rejected_with_reason_when_params_fit_neither(self):
        handler = NativeToolsHandler(actions_available=True)
        tc = _tool_call("calendar_create_event", summary="Psychiatrist appointment")
        decision = handler._parse_single_tool_call(tc, forced_action_type="calendar_delete_event")
        assert decision is not None  # surfaced, not silently dropped
        assert decision.wants_action is False
        assert decision.action_reject_reason is not None
        assert "calendar_delete_event" in decision.action_reject_reason

    def test_rejected_call_dropped_silently_outside_a_forced_round(self):
        # Same malformed call, but no round is forcing anything -- must
        # match the ORIGINAL behavior exactly (return None, not a decision).
        handler = NativeToolsHandler(actions_available=True)
        tc = _tool_call("calendar_create_event", summary="Psychiatrist appointment")
        decision = handler._parse_single_tool_call(tc, forced_action_type=None)
        assert decision is None

    def test_correctly_typed_forced_proposal_is_unchanged(self):
        # Regression: a correctly-typed proposal in a forced round must not
        # be altered by the new coercion machinery.
        handler = NativeToolsHandler(actions_available=True)
        tc = _tool_call("calendar_delete_event", summary="Psychiatrist appointment", date="2026-09-11")
        decision = handler._parse_single_tool_call(tc, forced_action_type="calendar_delete_event")
        assert decision.action_type == "calendar_delete_event"
        assert decision.wants_action is True
        assert decision.action_reject_reason is None

    def test_parse_response_end_to_end_coerces(self):
        # Drives the full parse_response entry point (not just the single-
        # call helper), matching how _get_model_decision calls it.
        handler = NativeToolsHandler(actions_available=True)
        response = SimpleNamespace(tool_calls=[
            _tool_call("calendar_create_event", summary="Psychiatrist appointment", date="2026-09-11"),
        ])
        decisions = handler.parse_response(response, forced_action_type="calendar_delete_event")
        assert len(decisions) == 1
        assert decisions[0].action_type == "calendar_delete_event"
        assert decisions[0].wants_action is True


# ===========================================================================
# XMLMarkerHandler -- the equivalent path for local/XML-protocol models
# ===========================================================================

class TestXmlMarkerCoercion:
    def test_wrong_type_coerced_in_a_forced_round(self):
        handler = XMLMarkerHandler()
        text = ('<action type="calendar_create_event" summary="Psychiatrist appointment" '
                'date="2026-09-11" reason="user asked"></action>')
        decisions = handler.parse_response(text, forced_action_type="calendar_delete_event")
        acted = [d for d in decisions if d.wants_action]
        assert len(acted) == 1
        assert acted[0].action_type == "calendar_delete_event"
        assert acted[0].action_params["summary"] == "Psychiatrist appointment"
        assert acted[0].action_params["date"] == "2026-09-11"

    def test_wrong_type_rejected_with_reason(self):
        handler = XMLMarkerHandler()
        text = ('<action type="calendar_create_event" summary="Psychiatrist appointment" '
                'reason="user asked"></action>')
        decisions = handler.parse_response(text, forced_action_type="calendar_delete_event")
        assert not any(d.wants_action for d in decisions)
        rejected = [d for d in decisions if d.action_reject_reason]
        assert len(rejected) == 1
        assert "calendar_delete_event" in rejected[0].action_reject_reason

    def test_correctly_typed_forced_marker_is_unchanged(self):
        handler = XMLMarkerHandler()
        text = ('<action type="calendar_delete_event" summary="Psychiatrist appointment" '
                'date="2026-09-11" reason="user asked"></action>')
        decisions = handler.parse_response(text, forced_action_type="calendar_delete_event")
        acted = [d for d in decisions if d.wants_action]
        assert len(acted) == 1
        assert acted[0].action_type == "calendar_delete_event"

    def test_propose_action_marker_wrong_type_also_coerces(self):
        handler = XMLMarkerHandler()
        text = ('<propose_action type="calendar_create_event" summary="Psychiatrist appointment" '
                'date="2026-09-11" reason="user asked"></propose_action>')
        decisions = handler.parse_response(text, forced_action_type="calendar_delete_event")
        acted = [d for d in decisions if d.wants_action]
        assert len(acted) == 1
        assert acted[0].action_type == "calendar_delete_event"

    def test_unforced_rejection_stays_silent_like_before(self):
        # No forced_action_type -- regression against the pre-existing
        # TestXmlActionValidation coverage in test_calendar_update_delete.py.
        handler = XMLMarkerHandler()
        text = ('<action type="calendar_create_event" summary="Psychiatrist appointment"'
                ' reason="user asked">Corrected appointment slot</action>')
        decisions = handler.parse_response(text)
        assert not any(d.wants_action for d in decisions)
        assert not any(getattr(d, "action_reject_reason", None) for d in decisions)


# ===========================================================================
# Force-prompt builders carry the required type + prior rejection reason
# ===========================================================================

class TestForcePromptCarriesReason:
    def test_xml_force_prompt_names_type_and_reason(self):
        spec = ACTION_SPECS[ActionType.CALENDAR_DELETE_EVENT]
        prompt = AgenticSearchController._build_xml_action_force_prompt(
            "delete the psychiatrist appointment", ActionType.CALENDAR_DELETE_EVENT, spec,
            reject_reason="proposed action_type='calendar_create_event' does not match "
                           "the required 'calendar_delete_event'",
        )
        assert 'type="calendar_delete_event"' in prompt
        assert "REJECTED" in prompt
        assert "calendar_create_event" in prompt  # names what was wrongly tried

    def test_xml_force_prompt_with_no_reason_is_unchanged_shape(self):
        spec = ACTION_SPECS[ActionType.CALENDAR_CREATE_EVENT]
        prompt = AgenticSearchController._build_xml_action_force_prompt(
            "place my deadlines on my calendar", ActionType.CALENDAR_CREATE_EVENT, spec)
        assert "REJECTED" not in prompt
        assert '<action type="calendar_create_event"' in prompt


# ===========================================================================
# Full controller round: drives the deployed _get_model_decision with a
# scripted native-tools model response (create-typed proposal), then a
# genuinely empty round -- matching the brief's "answers create-typed
# proposal first, then nothing" scenario.
# ===========================================================================

class TestControllerForcedRound:
    @pytest.mark.asyncio
    async def test_get_model_decision_coerces_wrong_type(self):
        controller = _bare_controller()

        async def _fake_generate_with_tools(**kwargs):
            return SimpleNamespace(tool_calls=[
                _tool_call("calendar_create_event", summary="Psychiatrist appointment",
                           date="2026-09-11"),
            ])

        controller._generate_with_tools = _fake_generate_with_tools
        handler = NativeToolsHandler(actions_available=True)
        session = AgenticSearchSession(query="delete the psychiatrist appointment",
                                        protocol=SearchProtocol.NATIVE_TOOLS)

        decisions = await controller._get_model_decision(
            prompt="p", system_prompt="s", model_name="m", handler=handler, session=session,
            tool_choice={"type": "function", "function": {"name": "propose_action"}},
            tools_override=[handler.propose_action_tool],
            forced_action_type="calendar_delete_event",
        )

        assert len(decisions) == 1
        assert decisions[0].wants_action is True
        assert decisions[0].action_type == "calendar_delete_event"

    @pytest.mark.asyncio
    async def test_get_model_decision_second_round_produces_nothing(self):
        # The "then nothing" half of the scripted scenario: a round where
        # the model returns no tool_calls and no usable text produces the
        # generic "ready to answer" decision, carrying no reject reason
        # (there was nothing to reject) -- the controller's own retry-once
        # gate (not this helper) is what stops a THIRD attempt.
        controller = _bare_controller()

        async def _fake_generate_with_tools(**kwargs):
            return SimpleNamespace(tool_calls=None, content="")

        controller._generate_with_tools = _fake_generate_with_tools
        handler = NativeToolsHandler(actions_available=True)
        session = AgenticSearchSession(query="delete the psychiatrist appointment",
                                        protocol=SearchProtocol.NATIVE_TOOLS)

        decisions = await controller._get_model_decision(
            prompt="p", system_prompt="s", model_name="m", handler=handler, session=session,
            tool_choice={"type": "function", "function": {"name": "propose_action"}},
            tools_override=[handler.propose_action_tool],
            forced_action_type="calendar_delete_event",
        )
        assert len(decisions) == 1
        assert decisions[0].wants_answer is True
        assert not getattr(decisions[0], "action_reject_reason", None)

    @pytest.mark.asyncio
    async def test_never_coerces_when_this_round_is_not_forcing(self):
        # A mid-loop round where _forced_action is still set on the session
        # conceptually but THIS round isn't forcing (forced_action_type is
        # None) must never coerce a spontaneous, differently-typed proposal.
        controller = _bare_controller()

        async def _fake_generate_with_tools(**kwargs):
            return SimpleNamespace(tool_calls=[
                _tool_call("send_email", recipient="a@b.com", message="hi"),
            ])

        controller._generate_with_tools = _fake_generate_with_tools
        handler = NativeToolsHandler(actions_available=True)
        session = AgenticSearchSession(query="anything", protocol=SearchProtocol.NATIVE_TOOLS)

        decisions = await controller._get_model_decision(
            prompt="p", system_prompt="s", model_name="m", handler=handler, session=session,
            forced_action_type=None,
        )
        assert decisions[0].action_type == "send_email"
        assert decisions[0].wants_action is True


# ===========================================================================
# (c) claims_pending_card -- the no-card backstop's trigger shapes
# ===========================================================================

class TestClaimsPendingCardQueueConfirmShapes:
    LIVE_SHAPED = (
        "Queued the deletion: Psychiatrist appointment on 2026-09-11. "
        "Confirm and it's off your calendar."
    )

    def test_live_reconstructed_sentence_claims_a_card(self):
        assert claims_pending_card(self.LIVE_SHAPED)

    def test_queued_the_clause_alone_claims_a_card(self):
        assert claims_pending_card("Queued the deletion for you.")

    def test_confirm_and_it_clause_alone_claims_a_card(self):
        assert claims_pending_card("Confirm and it's removed from your calendar.")

    def test_waiting_for_your_confirmation_claims_a_card(self):
        assert claims_pending_card("Waiting for your confirmation to proceed.")

    @pytest.mark.parametrize("text", [
        "I confirm that the file exists.",
        "The queue is empty right now.",
        "Once you confirm, I'll queue it.",
        "Want me to queue it so you can approve it?",
    ])
    def test_ordinary_prose_never_matches(self, text):
        assert not claims_pending_card(text)
