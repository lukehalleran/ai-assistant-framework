"""Behavioural siblings for the handler source-text tests on the enhanced route.

Each test runs the DEPLOYED ``handle_submit`` -> ``_run_enhanced`` (via
helpers_handler_turn) and asserts what the model stream actually received / what
the turn recorded, instead of grepping handlers.py for the name.

Incident class: BC-63 (source-text test proves spelling only). Lane SRCH, 2026-10-27.
Source tests kept: test_calendar_turn_round3 (action status), test_deferred_request_clarify,
test_insight_mode_handler (offer note), test_agentic_gate_trigger_source (telemetry key),
test_section_outcome_receipts (two callers).
"""
import pytest

from tests.unit.helpers_handler_turn import make_gate_decision, run_handler_turn

pytestmark = pytest.mark.asyncio

QUERY = "What did we decide about the sample project last week?"
WRAPPED = "What did we decide about\n   the sample project\n   last week?"


@pytest.mark.parametrize("text", [QUERY, WRAPPED], ids=["clean", "wrapped"])
async def test_enhanced_stream_gets_authoritative_action_status(monkeypatch, text):
    import core.actions.registry as registry
    monkeypatch.setattr(registry, "get_runtime_action_health",
                        lambda: "calendar_create_event backend: AVAILABLE (sentinel-health)")
    turn = await run_handler_turn(monkeypatch, text)
    system = turn.stream_calls[0]["system_prompt"]
    assert system.startswith("Test system prompt")
    assert "[APPLICATION ACTION STATUS — AUTHORITATIVE]" in system
    assert "calendar_create_event backend: AVAILABLE (sentinel-health)" in system
    assert "Never contradict an AVAILABLE backend" in system


async def test_action_status_failure_does_not_block_the_turn(monkeypatch):
    import core.actions.registry as registry

    def _boom():
        raise RuntimeError("registry unavailable")
    monkeypatch.setattr(registry, "get_runtime_action_health", _boom)
    turn = await run_handler_turn(monkeypatch, QUERY)
    assert "APPLICATION ACTION STATUS" not in turn.stream_calls[0]["system_prompt"]
    assert turn.final["content"] == "Hello world"


@pytest.mark.parametrize("text", [QUERY, WRAPPED], ids=["clean", "wrapped"])
async def test_deferred_request_note_reaches_the_model_stream(monkeypatch, text):
    gate = make_gate_decision(deferred_request="look up the sample thing for me")
    turn = await run_handler_turn(monkeypatch, text, gate=gate)
    system = turn.stream_calls[0]["system_prompt"]
    assert "[DEFERRED REQUEST]" in system
    assert "look up the sample thing for me" in system
    assert "Do NOT pretend you did it" in system
    assert turn.ctx.telemetry["gate_deferred_request"] is True


async def test_no_deferred_request_means_no_note(monkeypatch):
    turn = await run_handler_turn(monkeypatch, QUERY, gate=make_gate_decision())
    assert "[DEFERRED REQUEST]" not in turn.stream_calls[0]["system_prompt"]
    assert "gate_deferred_request" not in turn.ctx.telemetry


@pytest.mark.parametrize("text", [QUERY, WRAPPED], ids=["clean", "wrapped"])
async def test_insight_offer_note_is_injected_only_when_armed(monkeypatch, text):
    armed = await run_handler_turn(monkeypatch, text, gate=make_gate_decision(), offer_armed=True)
    system = armed.stream_calls[0]["system_prompt"]
    assert "[INSIGHT OFFER]" in system
    assert "Do not push" in system
    assert armed.ctx.telemetry["insight_offer_armed"] is True

    quiet = await run_handler_turn(monkeypatch, text, gate=make_gate_decision(), offer_armed=False)
    assert "[INSIGHT OFFER]" not in quiet.stream_calls[0]["system_prompt"]
    assert "insight_offer_armed" not in quiet.ctx.telemetry


async def test_gate_trigger_source_lands_in_turn_telemetry_and_debug_record(monkeypatch):
    gate = make_gate_decision(should_trigger=False, trigger_source="tier1_memory",
                              reason="untriggered: sample")
    turn = await run_handler_turn(monkeypatch, QUERY, gate=gate)
    assert turn.ctx.telemetry["gate_trigger_source"] == "tier1_memory"
    assert turn.debug["gate_trigger_source"] == "tier1_memory"
    assert turn.debug["gate_reason"]


async def test_no_gate_means_empty_trigger_source_in_the_debug_record(monkeypatch):
    turn = await run_handler_turn(monkeypatch, QUERY)  # agentic disabled: gate never runs
    assert turn.gate_calls == []
    assert "gate_trigger_source" not in turn.ctx.telemetry
    assert turn.debug["gate_trigger_source"] == ""


async def test_enhanced_debug_record_forwards_the_orchestrators_section_outcomes(monkeypatch):
    outcomes = {"memories": "ok", "wiki": "timeout-sentinel"}
    turn = await run_handler_turn(monkeypatch, QUERY,
                                  orch_attrs={"_last_section_outcomes": outcomes})
    assert turn.debug["section_outcomes"] == outcomes
