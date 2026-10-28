"""Smoke test of the handler-turn driver itself (helpers_handler_turn.py).

Incident class: BC-63 (source-text test proves spelling only), lane SRCH, 2026-10-27.
"""
import pytest

from tests.unit.helpers_handler_turn import make_gate_decision, run_handler_turn

pytestmark = pytest.mark.asyncio


async def test_enhanced_turn_runs_through_the_deployed_handler(monkeypatch):
    turn = await run_handler_turn(monkeypatch, "How is the sample weather today?",
                                  stream_chunks=["Fine.", " Sunny."])
    assert turn.final["content"] == "Fine. Sunny."
    assert turn.debug["mode"] == "enhanced"
    assert turn.stream_calls[0]["system_prompt"].startswith("Test system prompt")
    assert turn.storage_calls[0].stored_text == "Fine. Sunny."
    assert turn.events == ["dispatch_storage", "track_storage", "write_telemetry"]


async def test_agentic_turn_runs_through_the_deployed_handler(monkeypatch):
    turn = await run_handler_turn(
        monkeypatch, "calculate fibonacci 10 for the sample table",
        gate=make_gate_decision(should_trigger=True, modes=["tools"]),
        agentic_items=["The result is fifty-five."],
    )
    assert turn.debug["mode"] == "agentic-search"
    assert "fifty-five" in turn.final["content"]
    assert turn.storage_calls[0].kwargs["prompt_only_suffix"] == ""
