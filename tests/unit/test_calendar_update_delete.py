"""calendar_update_event + calendar_delete_event (2026-09-01).

New registry-driven write actions so the user can edit/remove existing
events, not just create them. Resolution doctrine: target identified by
summary + date (or verified event_id); EXACTLY one live match required —
zero or many fails honestly and touches nothing (delete is irreversible).
Intent patterns require an explicit calendar/event anchor: "reschedule
appointment with new psychiatrist" (live 2026-09-01 12:33 turn) is a life
task, never a forced action loop.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.actions.registry import ACTION_SPECS, detect_action_intent
from core.actions.types import ActionProposal, ActionResult, ActionType


def _proposal(action_type, **params):
    return ActionProposal(action_type=action_type, params=params, summary="t")


def _resp(status=200, payload=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    r.text = text
    return r


def _client(list_payload=None, get_resp=None, patch_resp=None, delete_resp=None):
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=get_resp or _resp(payload=list_payload or {}))
    client.patch = AsyncMock(return_value=patch_resp or _resp(payload={}))
    client.delete = AsyncMock(return_value=delete_resp or _resp(status=204))
    return client


def _auth_ok():
    auth = MagicMock()
    auth.is_authenticated = True
    auth.token_expired_no_refresh = False
    auth.has_scope.return_value = True
    auth.get_credentials.return_value = SimpleNamespace(token="tok")
    return auth


def _run(coro):
    return asyncio.run(coro)


_EVENT = {"id": "ev1", "summary": "Psychiatrist appointment (rescheduled)",
          "start": {"dateTime": "2026-09-09T12:00:00-05:00"},
          "end": {"dateTime": "2026-09-09T13:00:00-05:00"}}
_OTHER = {"id": "ev2", "summary": "Dentist",
          "start": {"dateTime": "2026-09-09T09:00:00-05:00"}}


class TestIntentRouting:
    def test_delete_with_calendar_anchor(self):
        assert detect_action_intent(
            "Remove the psychiatrist event from my calendar"
        ) == ActionType.CALENDAR_DELETE_EVENT

    def test_update_with_event_anchor(self):
        assert detect_action_intent(
            "Move the psychiatrist event on the 9th to 2pm"
        ) == ActionType.CALENDAR_UPDATE_EVENT

    def test_live_reschedule_life_task_not_hijacked(self):
        # The exact 2026-09-01 12:33 phrasing — rescheduling WITH the office.
        assert detect_action_intent(
            "I need to finish figuring out that form stuff and reschedule "
            "appointment with new psychiatrist"
        ) is None

    def test_cancel_narration_not_hijacked(self):
        assert detect_action_intent("I had to cancel my appointment yesterday") is None

    def test_create_still_routes_create(self):
        assert detect_action_intent(
            "let's make calendar event for Wednesday at 1"
        ) == ActionType.CALENDAR_CREATE_EVENT


class TestDeleteExecutor:
    def _delete(self, client, **params):
        from core.actions.google_calendar_modify import delete_calendar_event
        prop = _proposal(ActionType.CALENDAR_DELETE_EVENT, **params)
        with patch("core.actions.google_auth.get_google_auth", return_value=_auth_ok()), \
             patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("httpx.AsyncClient", return_value=client):
            return _run(delete_calendar_event(prop))

    def test_exact_match_deleted(self):
        client = _client(list_payload={"items": [_EVENT, _OTHER]})
        res = self._delete(client, summary="Psychiatrist appointment (rescheduled)",
                           date="2026-09-09")
        assert res.success
        assert "Deleted event: Psychiatrist appointment (rescheduled)" in res.message
        client.delete.assert_awaited_once()
        assert "/events/ev1" in client.delete.await_args.args[0]

    def test_zero_matches_fails_honestly(self):
        client = _client(list_payload={"items": [_OTHER]})
        res = self._delete(client, summary="Psychiatrist appointment",
                           date="2026-09-09")
        assert not res.success
        assert "No event titled" in res.message
        client.delete.assert_not_awaited()

    def test_multiple_matches_refuses(self):
        twin = dict(_EVENT, id="ev3", summary="Psychiatrist appointment (old)")
        near = dict(_EVENT, id="ev4", summary="Psychiatrist appointment (new)")
        client = _client(list_payload={"items": [twin, near]})
        res = self._delete(client, summary="Psychiatrist appointment",
                           date="2026-09-09")
        assert not res.success
        assert "refusing to guess" in res.message
        client.delete.assert_not_awaited()

    def test_exact_title_beats_containment(self):
        exact = dict(_EVENT, id="ev5", summary="Dentist")
        contains = dict(_EVENT, id="ev6", summary="Dentist follow-up")
        client = _client(list_payload={"items": [exact, contains]})
        res = self._delete(client, summary="Dentist", date="2026-09-09")
        assert res.success
        assert "/events/ev5" in client.delete.await_args.args[0]

    def test_wrong_day_event_not_matched(self):
        other_day = dict(_EVENT, start={"dateTime": "2026-09-10T12:00:00-05:00"})
        client = _client(list_payload={"items": [other_day]})
        res = self._delete(client, summary="Psychiatrist appointment (rescheduled)",
                           date="2026-09-09")
        assert not res.success
        client.delete.assert_not_awaited()


class TestUpdateExecutor:
    def _update(self, client, **params):
        from core.actions.google_calendar_modify import update_calendar_event
        prop = _proposal(ActionType.CALENDAR_UPDATE_EVENT, **params)
        with patch("core.actions.google_auth.get_google_auth", return_value=_auth_ok()), \
             patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("httpx.AsyncClient", return_value=client):
            return _run(update_calendar_event(prop))

    def test_time_move_patches_start_and_end(self):
        client = _client(
            list_payload={"items": [_EVENT]},
            patch_resp=_resp(payload={"summary": _EVENT["summary"],
                                      "start": {"dateTime": "2026-09-09T13:00:00-05:00"},
                                      "htmlLink": "http://cal/x"}))
        res = self._update(client, summary="Psychiatrist appointment (rescheduled)",
                           date="2026-09-09",
                           new_start_time="2026-09-09T13:00:00",
                           new_end_time="2026-09-09T14:00:00")
        assert res.success, res.message
        body = client.patch.await_args.kwargs["json"]
        assert body["start"]["dateTime"] == "2026-09-09T13:00:00"
        assert body["end"]["dateTime"] == "2026-09-09T14:00:00"
        assert body["start"]["timeZone"] == "America/Chicago"  # local default

    def test_lone_start_time_rejected(self):
        client = _client(list_payload={"items": [_EVENT]})
        res = self._update(client, summary=_EVENT["summary"], date="2026-09-09",
                           new_start_time="2026-09-09T13:00:00")
        assert not res.success
        assert "together" in res.message
        client.patch.assert_not_awaited()

    def test_no_change_fields_rejected(self):
        client = _client(list_payload={"items": [_EVENT]})
        res = self._update(client, summary=_EVENT["summary"], date="2026-09-09")
        assert not res.success
        assert "No change fields" in res.message
        client.patch.assert_not_awaited()

    def test_event_id_path_verified_by_get(self):
        client = _client(get_resp=_resp(payload=_EVENT),
                         patch_resp=_resp(payload=_EVENT))
        res = self._update(client, summary="x", date="2026-09-09",
                           event_id="ev1", new_summary="Renamed")
        assert res.success
        assert "/events/ev1" in client.get.await_args.args[0]


class TestSpecWiring:
    def test_forward_params_carry_all_fields(self):
        upd = ACTION_SPECS[ActionType.CALENDAR_UPDATE_EVENT]
        for f in ("summary", "date", "event_id", "new_start_time", "new_end_time"):
            assert f in upd.forward_params
        dele = ACTION_SPECS[ActionType.CALENDAR_DELETE_EVENT]
        for f in ("summary", "date", "event_id"):
            assert f in dele.forward_params

    def test_xml_action_marker_end_to_end(self):
        from core.agentic.protocols import XMLMarkerHandler
        h = XMLMarkerHandler()
        d = h.parse_response(
            '<action type="calendar_delete_event" summary="Psychiatrist appointment"'
            ' date="2026-09-09" reason="user asked"></action>')
        assert d and d[0].wants_action
        assert d[0].action_type == "calendar_delete_event"
        assert d[0].action_params["summary"] == "Psychiatrist appointment"
        assert d[0].action_params["date"] == "2026-09-09"

    def test_gate_action_arm_routes_delete(self):
        # Same registry the agentic gate consults (detect_action_intent) —
        # the test phrasing Luke will use routes to a forced tool loop.
        assert detect_action_intent(
            "Please remove the psychiatrist appointment event from my Google calendar"
        ) == ActionType.CALENDAR_DELETE_EVENT


class TestProposalCard:
    def test_delete_card_names_the_target(self):
        from gui.handlers import _format_action_proposal_card
        prop = _proposal(ActionType.CALENDAR_DELETE_EVENT,
                         summary="Psychiatrist appointment (rescheduled)",
                         date="2026-09-09")
        card = _format_action_proposal_card(prop)
        assert "calendar_delete_event" in card
        assert "Psychiatrist appointment (rescheduled)" in card
        assert "2026-09-09" in card

    def test_update_card_shows_changes(self):
        from gui.handlers import _format_action_proposal_card
        prop = _proposal(ActionType.CALENDAR_UPDATE_EVENT,
                         summary="Psychiatrist appointment", date="2026-09-09",
                         new_start_time="2026-09-09T13:00:00",
                         new_end_time="2026-09-09T14:00:00")
        card = _format_action_proposal_card(prop)
        assert "calendar_update_event" in card
        assert "start_time → 2026-09-09T13:00:00" in card


class TestXmlActionValidation:
    """2026-09-01 live: an XML calendar create WITHOUT start/end times entered
    the store, rendered a 'time missing' card, and failed only after the user
    approved it. Incomplete markers now drop at parse (forced-retry re-asks)."""

    def _parse(self, text):
        from core.agentic.protocols import XMLMarkerHandler
        return XMLMarkerHandler().parse_response(text)

    def test_create_without_times_dropped(self):
        d = self._parse(
            '<action type="calendar_create_event" summary="Psychiatrist appointment"'
            ' reason="user asked">Corrected appointment slot</action>')
        assert not any(x.wants_action for x in d)

    def test_create_with_times_kept(self):
        d = self._parse(
            '<action type="calendar_create_event" summary="Psychiatrist appointment"'
            ' start_time="2026-09-09T13:00:00-05:00" end_time="2026-09-09T14:00:00-05:00">'
            'note</action>')
        assert any(x.wants_action for x in d)

    def test_delete_without_date_dropped(self):
        d = self._parse('<action type="calendar_delete_event" summary="X">why</action>')
        assert not any(x.wants_action for x in d)

    def test_github_issue_missing_subject_kept_backfill(self):
        # github_create_issue has a backfill — incomplete is allowed through.
        d = self._parse('<action type="github_create_issue">just the body</action>')
        assert any(x.wants_action for x in d)

    def test_telegram_body_message_satisfies_required(self):
        d = self._parse('<action type="send_telegram" recipient="@luke">hello</action>')
        assert any(x.wants_action for x in d)

    def test_propose_action_incomplete_dropped(self):
        d = self._parse(
            '<propose_action type="calendar_create_event" summary="X">'
            'no times</propose_action>')
        assert not any(x.wants_action for x in d)


class TestApprovalChaining:
    """2026-09-01 live: a delete+create turn orphaned the older proposal —
    newest-only card, no way to approve the second. Now every decision
    surfaces the next still-pending proposal."""

    def _store_with_two(self, tmp_path):
        from core.actions.types import PendingActionsStore
        store = PendingActionsStore(path=str(tmp_path / "p.json"), persist=False)
        older = _proposal(ActionType.CALENDAR_DELETE_EVENT,
                          summary="Psychiatrist appointment", date="2026-09-08")
        older.summary = "calendar_delete_event: Psychiatrist appointment on 2026-09-08"
        newer = _proposal(ActionType.CALENDAR_CREATE_EVENT,
                          summary="Psychiatrist appointment",
                          start_time="2026-09-09T13:00:00-05:00",
                          end_time="2026-09-09T14:00:00-05:00")
        newer.summary = "calendar_create_event: Psychiatrist appointment"
        import datetime as _dt
        newer.proposed_at = older.proposed_at + _dt.timedelta(seconds=1)
        assert store.propose(older) and store.propose(newer)
        return store, older, newer

    def test_get_all_pending_oldest_first(self, tmp_path):
        store, older, newer = self._store_with_two(tmp_path)
        ids = [p.action_id for p in store.get_all_pending()]
        assert ids == [older.action_id, newer.action_id]

    @pytest.mark.asyncio
    async def test_execute_surfaces_next_pending(self, tmp_path):
        from gui.handlers import execute_pending_action_core
        store, older, newer = self._store_with_two(tmp_path)
        result = MagicMock(success=True, message="Calendar event created")
        registry = MagicMock()
        registry.execute = AsyncMock(return_value=result)
        with patch("core.agentic.tools.ToolExecutor._get_pending_actions_store",
                   return_value=store), \
             patch("core.actions.audit.ActionAuditLog", return_value=MagicMock()), \
             patch("core.actions.executors.ActionExecutorRegistry",
                   return_value=registry):
            outcome = await execute_pending_action_core(newer.action_id)
        assert outcome.status == "executed"
        assert outcome.next_action_id == older.action_id
        assert "still pending" in outcome.message
        assert "calendar_delete_event" in outcome.message  # its card rendered

    @pytest.mark.asyncio
    async def test_no_chain_when_nothing_left(self, tmp_path):
        from gui.handlers import execute_pending_action_core
        from core.actions.types import PendingActionsStore
        store = PendingActionsStore(path=str(tmp_path / "p.json"), persist=False)
        only = _proposal(ActionType.CALENDAR_CREATE_EVENT,
                         summary="X", start_time="2026-09-09T13:00:00",
                         end_time="2026-09-09T14:00:00")
        assert store.propose(only)
        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MagicMock(success=True, message="ok"))
        with patch("core.agentic.tools.ToolExecutor._get_pending_actions_store",
                   return_value=store), \
             patch("core.actions.audit.ActionAuditLog", return_value=MagicMock()), \
             patch("core.actions.executors.ActionExecutorRegistry",
                   return_value=registry):
            outcome = await execute_pending_action_core(only.action_id)
        assert outcome.next_action_id is None
        assert "still pending" not in outcome.message


class TestCalendarLookaheadWiring:
    def test_gatherer_passes_configured_lookahead(self, monkeypatch):
        # The YAML knob existed but was never forwarded — the 7-day default
        # hid the Sep 9 event from the Sep 1 turn that was editing it.
        import asyncio as _aio
        import config.app_config as cfg
        import core.prompt.gatherer_knowledge as gk
        import core.actions.google_calendar as gcal
        monkeypatch.setattr(cfg, "GOOGLE_CALENDAR_ENABLED", True, raising=False)
        monkeypatch.setattr(cfg, "GOOGLE_CALENDAR_LOOKAHEAD_DAYS", 14, raising=False)
        seen = {}

        async def fake_fetch(max_events=10, lookahead_days=7):
            seen["lookahead"] = lookahead_days
            return []

        monkeypatch.setattr(gcal, "fetch_upcoming_events", fake_fetch)
        stub = SimpleNamespace()
        _aio.run(
            gk.KnowledgeRetrievalMixin.get_google_calendar_events(stub, max_events=5))
        assert seen.get("lookahead") == 14


class TestWallClockDoctrine:
    """2026-09-01 live: three consecutive turns emitted -04:00 Eastern offsets
    for a Central user; a "noon -> 1 PM" update PATCHed 13:00-04:00 = the same
    noon-Central instant (no-op). Prompt teaching did not hold — the executor
    now strips UTC offsets so wall time + the IANA timeZone field govern."""

    def test_wall_clock_time_strips_offsets(self):
        from core.actions.google_calendar_create import wall_clock_time as w
        assert w("2026-09-09T13:00:00-04:00") == "2026-09-09T13:00:00"
        assert w("2026-09-09T13:00:00Z") == "2026-09-09T13:00:00"
        assert w("2026-09-09T13:00:00+05:30") == "2026-09-09T13:00:00"
        assert w("2026-09-09T13:00:00") == "2026-09-09T13:00:00"
        assert w("2026-09-09") == "2026-09-09"  # date-only untouched

    def test_update_patch_strips_model_offset(self):
        # The exact live shape: model supplies -04:00, user is Central.
        client = _client(
            list_payload={"items": [_EVENT]},
            patch_resp=_resp(payload={"summary": _EVENT["summary"],
                                      "start": {"dateTime": "2026-09-09T13:00:00"},
                                      "htmlLink": "http://cal/x"}))
        from core.actions.google_calendar_modify import update_calendar_event
        prop = _proposal(ActionType.CALENDAR_UPDATE_EVENT,
                         summary="Psychiatrist appointment (rescheduled)",
                         date="2026-09-09",
                         new_start_time="2026-09-09T13:00:00-04:00",
                         new_end_time="2026-09-09T14:00:00-04:00")
        with patch("core.actions.google_auth.get_google_auth", return_value=_auth_ok()), \
             patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
             patch("httpx.AsyncClient", return_value=client):
            res = _run(update_calendar_event(prop))
        assert res.success, res.message
        body = client.patch.await_args.kwargs["json"]
        assert body["start"] == {"dateTime": "2026-09-09T13:00:00",
                                 "timeZone": "America/Chicago"}
        assert body["end"] == {"dateTime": "2026-09-09T14:00:00",
                               "timeZone": "America/Chicago"}

    def test_vocabulary_teaches_offsetless_local_times(self):
        import inspect
        import core.agentic.types as t
        src = inspect.getsource(t)
        assert "WITHOUT a UTC offset" in src
        assert "calendar_update_event, calendar_delete_event" in src


class TestChangelessUpdateRejectedAtParse:
    """2026-09-01 live round 3: the force prompt's required-only example
    taught summary+date, the model emitted a changeless update marker, the
    card rendered, and the executor could only fail after approval. The spec
    now carries an accepts_check (>=1 new_* field) that the XML parse-time
    validation enforces, and the force prompt SHOWS the change fields."""

    def test_changeless_update_marker_dropped(self):
        from core.agentic.protocols import XMLMarkerHandler
        d = XMLMarkerHandler().parse_response(
            '<action type="calendar_update_event" summary="Psychiatrist appointment'
            ' (rescheduled)" date="2026-09-09" reason="user asked">move to 1</action>')
        assert not any(x.wants_action for x in d)

    def test_update_with_changes_kept(self):
        from core.agentic.protocols import XMLMarkerHandler
        d = XMLMarkerHandler().parse_response(
            '<action type="calendar_update_event" summary="Psychiatrist appointment"'
            ' date="2026-09-09" new_start_time="2026-09-09T13:00:00"'
            ' new_end_time="2026-09-09T14:00:00" reason="user asked"></action>')
        assert any(x.wants_action for x in d)

    def test_force_prompt_shows_change_fields(self):
        from unittest.mock import MagicMock
        from core.agentic.controller import AgenticSearchController
        from core.actions.registry import ACTION_SPECS
        mm = MagicMock(); mm.api_models = {}
        c = AgenticSearchController(model_manager=mm, web_search_manager=MagicMock())
        spec = ACTION_SPECS[ActionType.CALENDAR_UPDATE_EVENT]
        prompt = c._build_xml_action_force_prompt(
            "Move the psychiatrist event on the 9th from noon to 1 PM",
            ActionType.CALENDAR_UPDATE_EVENT, spec)
        assert "new_start_time" in prompt and "new_end_time" in prompt
        assert "WITHOUT a UTC offset" in prompt
        assert "At least one new_* field" in prompt

    def test_delete_force_prompt_minimal(self):
        from unittest.mock import MagicMock
        from core.agentic.controller import AgenticSearchController
        from core.actions.registry import ACTION_SPECS
        mm = MagicMock(); mm.api_models = {}
        c = AgenticSearchController(model_manager=mm, web_search_manager=MagicMock())
        spec = ACTION_SPECS[ActionType.CALENDAR_DELETE_EVENT]
        prompt = c._build_xml_action_force_prompt(
            "Remove the psychiatrist event on the 9th",
            ActionType.CALENDAR_DELETE_EVENT, spec)
        assert "identify the EXISTING event to" in prompt
