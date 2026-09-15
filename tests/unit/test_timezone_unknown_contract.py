"""Contract tests for A03b-1 (F02 timezone half / G02-T03 / G08-T08).

An unknown user timezone must never silently become a guessed zone (Central,
UTC, or anything else) — BC-59 (owner-identity/locale hardcoding), BC-47
(a failure/unknown collapsed into a valid-looking result) and BC-46 (the
prompt's own instruction losing to a code default). Deployed functions, a
fake calendar service, no network, synthetic zones only.

Complements tests/unit/test_timezone_resolver.py (resolver internals) and
tests/unit/test_calendar_create.py (executor plumbing already covered
there). conftest.py's autouse `_sandbox_timezone_resolver` fixture pins
utils.timezone_resolver.get_user_timezone to "America/Chicago" for every
test; `_unknown()`/`_known()` below further monkeypatch it for the duration
of one `with` block and are restored afterward, same as any nested patch.
"""
import asyncio
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import utils.timezone_resolver as tzmod
from core.actions.google_calendar_create import UNKNOWN_TIMEZONE_MESSAGE, create_calendar_event
from core.actions.google_calendar_modify import update_calendar_event
from core.actions.registry import _current_wall_clock, calendar_timezone_unknown, resolve_forced_action
from core.actions.types import ActionProposal


def _unknown():
    """Override the resolver to None (override/profile/system all unknown)."""
    return patch.object(tzmod, "get_user_timezone", lambda: None)


def _known(zone="America/Denver"):
    return patch.object(tzmod, "get_user_timezone", lambda: zone)


def _async_client():
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _created_resp():
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"id": "e1", "htmlLink": ""}
    return resp


def _auth_ok():
    auth = MagicMock(is_authenticated=True)
    auth.has_scope.return_value = True
    auth.get_credentials.return_value = MagicMock(token="tok")
    return auth


def _run(coro):
    return asyncio.run(coro)


@contextmanager
def _executor_env(client, zone_ctx):
    """Shared prereq patches (config/auth/httpx) + a resolver zone context."""
    with patch("config.app_config.GOOGLE_CALENDAR_ENABLED", True), \
         patch("core.actions.google_auth.get_google_auth", return_value=_auth_ok()), \
         patch("httpx.AsyncClient", return_value=client), zone_ctx:
        yield


class TestCreateExecutorRefusal:
    """create_calendar_event refuses with ZERO service calls when unknown."""

    def _proposal(self, **params):
        return ActionProposal(action_type="calendar_create_event", params=params, summary="t")

    def test_refuses_before_any_call(self):
        client = _async_client()
        proposal = self._proposal(summary="Standup", start_time="2026-05-28T14:00:00",
                                   end_time="2026-05-28T15:00:00")
        with _executor_env(client, _unknown()):
            result = _run(create_calendar_event(proposal))
        assert result.success is False
        assert result.message == UNKNOWN_TIMEZONE_MESSAGE
        client.get.assert_not_awaited()
        client.post.assert_not_awaited()

    def test_control_explicit_zone_succeeds_when_resolver_unknown(self):
        client = _async_client()
        client.post.return_value = _created_resp()
        client.get.return_value = MagicMock(status_code=500)  # dup-check unavailable, tolerated
        proposal = self._proposal(summary="Standup", start_time="2026-05-28T14:00:00",
                                   end_time="2026-05-28T15:00:00", time_zone="America/Denver")
        with _executor_env(client, _unknown()):
            result = _run(create_calendar_event(proposal))
        assert result.success is True, result.message
        body = client.post.call_args.kwargs["json"]
        assert body["start"]["timeZone"] == "America/Denver"

    def test_explicit_alias_zone_is_honored_when_resolver_unknown(self):
        """Parent review R1: an explicitly stated zone label maps through the
        resolver's alias table instead of being dropped."""
        client = _async_client()
        client.post.return_value = _created_resp()
        client.get.return_value = MagicMock(status_code=500)
        proposal = self._proposal(summary="Standup", start_time="2026-05-28T14:00:00",
                                   end_time="2026-05-28T15:00:00", time_zone="Eastern")
        with _executor_env(client, _unknown()):
            result = _run(create_calendar_event(proposal))
        assert result.success is True, result.message
        body = client.post.call_args.kwargs["json"]
        assert body["start"]["timeZone"] == "America/New_York"

    def test_unrecognized_explicit_zone_refuses_instead_of_using_user_zone(self):
        """Parent review R1 (BC-47): a stated but unrecognized zone must never
        be silently replaced by the user's own zone (that would schedule the
        event at the wrong wall-clock time); it refuses before any call."""
        client = _async_client()
        proposal = self._proposal(summary="Standup", start_time="2026-05-28T14:00:00",
                                   end_time="2026-05-28T15:00:00", time_zone="Mars/Olympus")
        with _executor_env(client, _known("America/Denver")):
            result = _run(create_calendar_event(proposal))
        assert result.success is False
        client.get.assert_not_awaited()
        client.post.assert_not_awaited()

    def test_control_all_day_succeeds_when_resolver_unknown(self):
        client = _async_client()
        client.post.return_value = _created_resp()
        client.get.return_value = MagicMock(status_code=500)
        proposal = self._proposal(summary="Trip", start_time="2026-05-28",
                                   end_time="2026-05-29", all_day=True)
        with _executor_env(client, _unknown()):
            result = _run(create_calendar_event(proposal))
        assert result.success is True, result.message

    def test_batch_one_zoneless_event_refuses_whole_proposal(self):
        client = _async_client()
        events = [
            {"summary": "A", "start_time": "2026-05-28T14:00:00",
             "end_time": "2026-05-28T15:00:00", "time_zone": "America/Denver"},
            {"summary": "B", "start_time": "2026-05-29T14:00:00",
             "end_time": "2026-05-29T15:00:00"},  # no explicit zone
        ]
        proposal = self._proposal(events=events)
        with _executor_env(client, _unknown()):
            result = _run(create_calendar_event(proposal))
        assert result.success is False
        client.get.assert_not_awaited()
        client.post.assert_not_awaited()

    def test_denver_profile_flows_to_aware_payload_end_to_end(self):
        """A non-Central zone (profile/override) passes through into an
        aware payload with the correct wall-clock dateTime."""
        client = _async_client()
        client.post.return_value = _created_resp()
        client.get.return_value = MagicMock(status_code=500)
        proposal = self._proposal(summary="Standup", start_time="2026-05-28T14:00:00",
                                   end_time="2026-05-28T15:00:00")
        with _executor_env(client, _known("America/Denver")):
            result = _run(create_calendar_event(proposal))
        assert result.success is True, result.message
        body = client.post.call_args.kwargs["json"]
        assert body["start"] == {"dateTime": "2026-05-28T14:00:00", "timeZone": "America/Denver"}
        assert body["end"] == {"dateTime": "2026-05-28T15:00:00", "timeZone": "America/Denver"}


class TestUpdateExecutorRefusal:
    """update_calendar_event refuses with ZERO service calls when unknown."""

    def _proposal(self, **params):
        return ActionProposal(action_type="calendar_update_event", params=params, summary="t")

    def test_refuses_before_any_call(self):
        client = _async_client()
        proposal = self._proposal(summary="Standup", date="2026-05-28",
                                   new_start_time="2026-05-28T14:00:00",
                                   new_end_time="2026-05-28T15:00:00")
        with _executor_env(client, _unknown()):
            result = _run(update_calendar_event(proposal))
        assert result.success is False
        assert result.message == UNKNOWN_TIMEZONE_MESSAGE
        client.get.assert_not_awaited()
        client.patch.assert_not_awaited()

    def test_control_explicit_zone_succeeds_when_resolver_unknown(self):
        client = _async_client()
        client.get.return_value = MagicMock(status_code=200, json=lambda: {
            "items": [{"id": "ev1", "summary": "Standup",
                       "start": {"dateTime": "2026-05-28T14:00:00-05:00"}}]})
        client.patch.return_value = MagicMock(status_code=200, json=lambda: {"summary": "Standup"})
        proposal = self._proposal(summary="Standup", date="2026-05-28",
                                   new_start_time="2026-05-28T15:00:00",
                                   new_end_time="2026-05-28T16:00:00",
                                   time_zone="America/Denver")
        with _executor_env(client, _unknown()):
            result = _run(update_calendar_event(proposal))
        assert result.success is True, result.message
        body = client.patch.call_args.kwargs["json"]
        assert body["start"]["timeZone"] == "America/Denver"

    # The all-day control is exercised once, on create (above) — modify's
    # all_day branch is the same trivial date-only passthrough and never
    # touches timezone resolution at all.


class TestResolveForcedActionCalendarBranch:
    """Proposal-time (card-minting) rejection, beside _shape_bad."""

    def test_rejects_with_timezone_reason_when_unknown(self):
        base = {"summary": "Standup", "start_time": "2026-05-28T14:00:00",
                "end_time": "2026-05-28T15:00:00"}
        with _unknown():
            resolved_type, resolved_params, reason = resolve_forced_action(
                "calendar_create_event", base)
            assert resolved_type is None and resolved_params is None
            assert "timezone" in (reason or "").lower()
            # Paired non-triggering control: an explicit zone still passes.
            resolved_type, _, reason = resolve_forced_action(
                "calendar_create_event", {**base, "time_zone": "America/Denver"})
            assert reason is None
            assert resolved_type == "calendar_create_event"

    def test_exempt_for_all_day_and_non_calendar_payloads(self):
        with _unknown():
            assert calendar_timezone_unknown({
                "summary": "Trip", "start_time": "2026-05-28",
                "end_time": "2026-05-29", "all_day": True,
            }) is False
            assert calendar_timezone_unknown({"recipient": "a@b.com", "message": "hi"}) is False


class TestCurrentWallClock:
    """The None branch is explicit, not exception-driven."""

    def test_none_branch_returns_naive_datetime(self):
        with _unknown():
            now = _current_wall_clock()
        assert isinstance(now, datetime)
        assert now.tzinfo is None
        # Paired non-triggering control, same call: a known zone is aware.
        with _known("America/Denver"):
            assert _current_wall_clock().tzinfo is not None
