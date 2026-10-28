"""Behavioural siblings: both routes deliver ONE ordered revision pipeline and hand
the personal-claim task to storage.

Runs the DEPLOYED ``handle_submit`` -> ``_run_enhanced`` / ``_run_agentic_search``
(helpers_handler_turn). The grounding and personal-claim passes are replaced at their
delivery seams with recording fakes, so the tests assert what the routes DO with them:
the personal check audits the GROUNDED body, the action-guard suffix stays detached
from both checks, the revised body is what the user reads AND what storage receives,
and the deferred personal-claim task rides to ``_dispatch_storage``.

Incident class: BC-63 (source-text test proves spelling only), BC-45. Lane SRCH, 2026-10-27.
Source tests kept: test_grounding_wiring::{test_enhanced_path_wired,test_agentic_path_wired},
test_handlers_delivery_revision_sequence::test_agentic_and_enhanced_sites_use_one_sequential_revision_pipeline,
test_personal_claim_delivery::test_both_routes_and_the_ingress_capture_are_wired.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

import gui.handlers as handlers
from tests.unit.helpers_handler_turn import make_gate_decision, run_handler_turn

pytestmark = pytest.mark.asyncio

QUERY = "When is the sample meeting this week?"
CLEAN = "The sample meeting is at three in the afternoon."
WRAPPED = "The sample meeting is\n   at three in the afternoon."
GUARD = "\n\n> guard-notice-sentinel"
ROUTES = ["enhanced", "agentic"]


def _route_kwargs(route, body):
    if route == "agentic":
        return dict(gate=make_gate_decision(should_trigger=True, modes=["tools"]),
                    agentic_items=[body])
    return dict(stream_chunks=[body])


def _install_fakes(monkeypatch):
    seen = {"grounding": [], "personal": []}

    async def grounding(ctx, body, source_material=""):
        seen["grounding"].append(body)
        return body.replace("three", "four"), ""

    async def personal(ctx, body):
        seen["personal"].append(body)
        return body.replace("meeting", "session")

    monkeypatch.setattr(handlers, "_apply_grounding_check_for_delivery", grounding)
    monkeypatch.setattr(handlers, "_apply_personal_claim_check_for_delivery", personal)
    return seen


@pytest.mark.parametrize("body", [CLEAN, WRAPPED], ids=["clean", "wrapped"])
@pytest.mark.parametrize("route", ROUTES)
async def test_personal_check_audits_the_grounded_body_and_the_guard_suffix_stays_out(
        monkeypatch, route, body):
    seen = _install_fakes(monkeypatch)
    turn = await run_handler_turn(monkeypatch, QUERY, action_guard_suffix=GUARD,
                                  **_route_kwargs(route, body))

    revised = body.replace("three", "four").replace("meeting", "session")
    assert seen["grounding"] == [body]                       # clean body, no guard notice
    assert seen["personal"] == [body.replace("three", "four")]   # sequential: sees the correction
    assert turn.final["content"] == revised + GUARD          # notice reattached once, after both
    assert turn.storage_calls[0].stored_text.startswith(revised)
    assert GUARD.strip() in turn.storage_calls[0].stored_text
    assert turn.debug["mode"] == ("agentic-search" if route == "agentic" else "enhanced")


@pytest.mark.parametrize("route", ROUTES)
async def test_no_revision_delivers_the_original_body(monkeypatch, route):
    async def keep(ctx, body, source_material=""):
        return None, ""

    async def keep_personal(ctx, body):
        return None

    monkeypatch.setattr(handlers, "_apply_grounding_check_for_delivery", keep)
    monkeypatch.setattr(handlers, "_apply_personal_claim_check_for_delivery", keep_personal)
    turn = await run_handler_turn(monkeypatch, QUERY, **_route_kwargs(route, CLEAN))
    assert turn.final["content"] == CLEAN
    assert turn.storage_calls[0].stored_text == CLEAN


@pytest.mark.parametrize("route", ROUTES)
async def test_a_failing_check_is_fail_open(monkeypatch, route):
    async def boom(ctx, body, source_material=""):
        raise RuntimeError("verifier down")

    async def personal(ctx, body):
        return body.replace("meeting", "session")

    monkeypatch.setattr(handlers, "_apply_grounding_check_for_delivery", boom)
    monkeypatch.setattr(handlers, "_apply_personal_claim_check_for_delivery", personal)
    turn = await run_handler_turn(monkeypatch, QUERY, **_route_kwargs(route, CLEAN))
    assert turn.final["content"] == CLEAN.replace("meeting", "session")


# ── the personal-claim task rides to storage ─────────────────────────────────

@pytest.mark.parametrize("route", ROUTES)
async def test_log_only_personal_claim_task_is_handed_to_storage(monkeypatch, route):
    monkeypatch.setattr(handlers.personal_claim_check, "audit_personal_claims",
                        AsyncMock(return_value=None))
    turn = await run_handler_turn(monkeypatch, QUERY, personal_claim_mode="log_only",
                                  **_route_kwargs(route, CLEAN))
    ctx = turn.ctx
    assert ctx.personal_claim_mode == "log_only"
    assert ctx.personal_claim_pending == CLEAN               # the audited body was captured
    task = turn.storage_calls[0].kwargs["personal_claim_task"]
    assert task is not None and task is ctx.personal_claim_task
    await asyncio.gather(task, return_exceptions=True)
    assert ctx.telemetry["personal_claim_status"] == "failed"   # invalid_response from the stub


@pytest.mark.parametrize("route", ROUTES)
async def test_personal_claim_off_hands_storage_no_task(monkeypatch, route):
    turn = await run_handler_turn(monkeypatch, QUERY, personal_claim_mode="off",
                                  **_route_kwargs(route, CLEAN))
    assert turn.ctx.personal_claim_mode == "off"
    assert turn.storage_calls[0].kwargs["personal_claim_task"] is None


@pytest.mark.parametrize("route", ROUTES)
async def test_correct_mode_withholds_the_draft_and_delivers_the_omission(monkeypatch, route):
    revised = "The sample meeting is this week."
    check = AsyncMock(return_value=({"status": "checked", "delivery": "omitted"}, revised))
    monkeypatch.setattr(handlers, "_apply_personal_claim_check", check)
    turn = await run_handler_turn(monkeypatch, QUERY, personal_claim_mode="correct",
                                  **_route_kwargs(route, CLEAN))
    assert turn.ctx.personal_claim_mode == "correct"        # captured once at ingress
    check.assert_awaited()
    assert check.await_args.kwargs["mode"] == "correct"
    assert turn.final["content"] == revised
    assert all("three in the afternoon" not in c for c in turn.contents)  # draft never on the wire
    assert turn.storage_calls[0].stored_text == revised
