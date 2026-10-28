"""Behavioural siblings for the stream-artifact / watchdog source-text tests.

Every display path of the deployed ``handle_submit`` (enhanced, agentic, duel, raw)
must hand the user a reply without the kimi-3 lone-'e' artifact, the enhanced route
must never put an empty thinking shell or ``<|sep|>`` on screen, and a degenerate
stream must abort before storage while a long legitimate stream is delivered.

Incident class: BC-63 (source-text test proves spelling only). Lane SRCH, 2026-10-27.
Source tests kept: test_stream_artifacts::{test_all_display_paths_call_artifact_strip,
test_handlers_wire_shell_strip_and_display_strip, test_watchdog_shape_check_present_no_wall_clock,
test_watchdog_returns_before_storage}.
"""
import pytest

from tests.unit.helpers_handler_turn import make_gate_decision, run_handler_turn
from tests.unit.test_handle_submit import _make_orchestrator

pytestmark = pytest.mark.asyncio

QUERY = "Did they ever respond to the sample email thread?"
CLEAN = "Did they ever respond, by the way?"
WRAPPED = "Did they ever\n   respond, by the way?"
DUEL = {"gens": ["model-a", "model-b"], "sels": ["judge"]}


def _agentic_gate():
    return make_gate_decision(should_trigger=True, modes=["tools"])


@pytest.mark.parametrize("body", [CLEAN, WRAPPED], ids=["clean", "wrapped"])
async def test_enhanced_display_and_storage_drop_the_lone_e_artifact(monkeypatch, body):
    turn = await run_handler_turn(monkeypatch, QUERY, stream_chunks=[body + "e"])
    assert turn.final["content"] == body
    assert turn.storage_calls[0].stored_text == body


@pytest.mark.parametrize("body", [CLEAN, WRAPPED], ids=["clean", "wrapped"])
async def test_agentic_display_debug_and_storage_drop_the_lone_e_artifact(monkeypatch, body):
    turn = await run_handler_turn(monkeypatch, QUERY, gate=_agentic_gate(),
                                  agentic_items=[body + "e"])
    assert turn.debug["mode"] == "agentic-search"
    assert turn.final["content"] == body
    assert turn.debug["response"] == body
    assert turn.storage_calls[0].stored_text == body


@pytest.mark.parametrize("body", [CLEAN, WRAPPED], ids=["clean", "wrapped"])
async def test_duel_display_and_storage_drop_the_lone_e_artifact(monkeypatch, body):
    result = {"answer": body + "e", "thinking_a": "", "thinking_b": "",
              "model_a": "model-a", "model_b": "model-b", "winner": "A",
              "scores": {"A": 8.0, "B": 6.0}}
    turn = await run_handler_turn(monkeypatch, QUERY, duel={**DUEL, "result": result})
    assert turn.debug["mode"] == "best-of-duel"
    assert turn.final["content"] == body
    assert turn.storage_calls[0].stored_text == body


@pytest.mark.parametrize("body", [CLEAN, WRAPPED], ids=["clean", "wrapped"])
async def test_raw_display_drops_the_lone_e_artifact(monkeypatch, body):
    orch = _make_orchestrator(raw_result=(body + "e", {}))
    turn = await run_handler_turn(monkeypatch, QUERY, orchestrator=orch, use_raw_gpt=True)
    assert turn.debug["mode"] == "raw"
    assert turn.final["content"] == body


async def test_enhanced_never_shows_the_empty_shell_or_sep_token(monkeypatch):
    chunks = ["<thinking>", "</thinking>", "<|sep|>Nice", " — pushed."]
    turn = await run_handler_turn(monkeypatch, QUERY, stream_chunks=chunks)
    for content in turn.contents:
        assert "<thinking>" not in content and "</thinking>" not in content
        assert "<|sep|>" not in content
    assert turn.final["content"] == "Nice — pushed."
    assert turn.storage_calls[0].stored_text == "Nice — pushed."


DEGENERATE_LINE = "the loop repeats this exact line again and again\n"
DEGENERATE = [DEGENERATE_LINE * 10] * 6
LEGIT = [
    " ".join(f"Sentence {i + 10 * j} covers topic {(i + 10 * j) * 7} in its own detail."
             for i in range(10)) + " "
    for j in range(8)
]


async def test_enhanced_degenerate_stream_shows_the_abort_notice_not_the_garbage(monkeypatch):
    turn = await run_handler_turn(monkeypatch, QUERY, stream_chunks=DEGENERATE)
    assert "repetitive and incoherent" in turn.final["content"]
    assert DEGENERATE_LINE.strip() not in turn.final["content"]
    assert turn.debug is None  # no debug record for an aborted generation


async def test_enhanced_degenerate_stream_aborts_before_storage(monkeypatch):
    turn = await run_handler_turn(monkeypatch, QUERY, stream_chunks=DEGENERATE)
    assert turn.storage_calls == []
    assert turn.events == []


async def test_enhanced_long_legitimate_stream_is_not_aborted(monkeypatch):
    turn = await run_handler_turn(monkeypatch, QUERY, stream_chunks=LEGIT)
    body = "".join(LEGIT)
    assert len(body) > 2000
    assert "repetitive" not in turn.final["content"]
    assert turn.final["content"].strip() == body.strip()
    assert len(turn.storage_calls) == 1
