"""Behavioural siblings for the handle_submit wiring/ordering source-text tests.

Runs the DEPLOYED dispatcher (helpers_handler_turn) and asserts the order and the
arguments the source tests inferred from text positions: the failed-action retry
runs before the agentic gate, insight mode owns the turn ahead of document
generation, storage tracking precedes turn telemetry at every dispatch site, and
the prompt-only attachment suffix reaches storage at every site.

Incident class: BC-63 (source-text test proves spelling only). Lane SRCH, 2026-10-27.
Source tests kept: test_handlers_calendar_offer_continuation::test_dispatcher_wires_retry_before_the_gate,
test_insight_mode_handler::test_insight_branch_precedes_doc_gen,
test_background_storage_receipts::test_track_storage_task_precedes_write_turn_telemetry_at_every_site,
test_handlers_navigation_not_stored_as_user_text::test_every_dispatch_site_passes_the_suffix.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import gui.handlers as handlers
from tests.unit.helpers_handler_turn import (
    doc_attachment_result,
    make_gate_decision,
    run_handler_turn,
)
from tests.unit.test_handlers_calendar_offer_continuation import RETRY_B, _failed_store

pytestmark = pytest.mark.asyncio

QUERY = "What did we decide about the sample project last week?"
ORDERED = ["dispatch_storage", "track_storage", "write_telemetry"]
DUEL = {"gens": ["model-a", "model-b"], "sels": ["judge"]}


def _agentic_gate():
    return make_gate_decision(should_trigger=True, modes=["tools"])


# ── storage tracking precedes turn telemetry, at every dispatch site ─────────

async def test_enhanced_site_tracks_storage_before_writing_telemetry(monkeypatch):
    turn = await run_handler_turn(monkeypatch, QUERY)
    assert turn.events == ORDERED
    assert turn.telemetry_modes == ["enhanced"]


async def test_agentic_site_tracks_storage_before_writing_telemetry(monkeypatch):
    turn = await run_handler_turn(monkeypatch, QUERY, gate=_agentic_gate(),
                                  agentic_items=["Done."])
    assert turn.events == ORDERED
    assert turn.telemetry_modes == ["agentic-search"]


async def test_duel_site_tracks_storage_before_writing_telemetry(monkeypatch):
    turn = await run_handler_turn(monkeypatch, QUERY, duel=DUEL)
    assert turn.events == ORDERED
    assert turn.telemetry_modes == ["best-of-duel"]


# ── the prompt-only suffix is handed to storage at every site ───────────────

SUFFIX_QUERY = "Please use Housing.csv for the sample analysis of notes.txt"
SUFFIX_WRAPPED = "Please use Housing.csv for\n   the sample analysis\n   of notes.txt"


def _attachment_turn_kwargs(text):
    return dict(
        files=[SimpleNamespace(name="/tmp/notes.txt")],
        files_result=doc_attachment_result(text),
    )


@pytest.mark.parametrize("text", [SUFFIX_QUERY, SUFFIX_WRAPPED], ids=["clean", "wrapped"])
@pytest.mark.parametrize("route", ["enhanced", "agentic", "duel"])
async def test_every_route_hands_the_prompt_only_suffix_to_storage(monkeypatch, route, text):
    kwargs = _attachment_turn_kwargs(text)
    if route == "agentic":
        kwargs.update(gate=_agentic_gate(), agentic_items=["Done."])
    elif route == "duel":
        kwargs.update(duel=DUEL)
    turn = await run_handler_turn(monkeypatch, text, **kwargs)

    assert len(turn.storage_calls) == 1
    call = turn.storage_calls[0]
    suffix = call.kwargs["prompt_only_suffix"]
    assert suffix.startswith("\n\n[ATTACHMENT NOTE]")
    assert call.merged_input.endswith(suffix)           # the prompt carried it
    # what the real storage boundary keeps: the user's text + attachment, not the note
    kept = handlers._strip_prompt_only_suffix(call.merged_input, suffix)
    assert "[ATTACHMENT NOTE]" not in kept
    assert kept.endswith("Sample body.")


async def test_a_turn_without_notes_hands_storage_an_empty_suffix(monkeypatch):
    turn = await run_handler_turn(monkeypatch, QUERY)
    assert turn.storage_calls[0].kwargs["prompt_only_suffix"] == ""


# ── failed-action retry runs BEFORE the agentic gate ─────────────────────────

async def test_retry_of_a_failed_action_is_served_before_the_gate_runs(monkeypatch):
    from core.agentic.tools import ToolExecutor
    store, failed = _failed_store()
    with patch.object(ToolExecutor, "_get_pending_actions_store", return_value=store), \
         patch("config.app_config.INTERNET_ACTIONS_ENABLED", True):
        turn = await run_handler_turn(monkeypatch, RETRY_B, gate=_agentic_gate())

    assert turn.gate_calls == []                    # the gate was never evaluated
    assert turn.stream_calls == []                  # no model stream either
    orch = turn.orchestrator
    orch.prepare_prompt.assert_not_awaited()
    card = next(c for c in turn.chunks if c.get("pending_action_id"))
    assert card["pending_action_id"] != failed.action_id
    assert store.get(card["pending_action_id"]).status == "pending"


async def test_without_a_failed_action_the_same_words_reach_the_gate(monkeypatch):
    from core.actions.types import PendingActionsStore
    from core.agentic.tools import ToolExecutor
    empty = PendingActionsStore(persist=False)
    with patch.object(ToolExecutor, "_get_pending_actions_store", return_value=empty), \
         patch("config.app_config.INTERNET_ACTIONS_ENABLED", True):
        turn = await run_handler_turn(monkeypatch, RETRY_B, gate=_agentic_gate(),
                                      agentic_items=["Done."])
    assert len(turn.gate_calls) == 1
    assert turn.debug["mode"] == "agentic-search"


# ── insight mode owns the turn ahead of document generation ─────────────────

INSIGHT = {"kind": "theme_sweep", "theme": "sleep", "wants_document": False, "raw_query": "q"}
DOCGEN = {"doc_type": "report", "topic": "sleep"}


def _two_mode_gate():
    return make_gate_decision(should_trigger=True, modes=["insight"],
                              insight_intent=INSIGHT, doc_gen_intent=DOCGEN)


def _fake_mode(name, calls, *, handles):
    async def _run(ctx):
        calls.append(name)
        yield {"role": "assistant", "content": f"{name} answer"}
        ctx.handled = handles
    return _run


async def test_insight_runs_first_and_doc_generation_never_runs_once_handled(monkeypatch):
    calls = []
    monkeypatch.setattr(handlers, "_run_insight_mode", _fake_mode("insight", calls, handles=True))
    monkeypatch.setattr(handlers, "_run_doc_generation", _fake_mode("docgen", calls, handles=True))
    turn = await run_handler_turn(monkeypatch, QUERY, gate=_two_mode_gate())
    assert calls == ["insight"]
    assert turn.final["content"] == "insight answer"
    assert turn.stream_calls == []


async def test_doc_generation_runs_after_an_insight_fall_through(monkeypatch):
    calls = []
    monkeypatch.setattr(handlers, "_run_insight_mode", _fake_mode("insight", calls, handles=False))
    monkeypatch.setattr(handlers, "_run_doc_generation", _fake_mode("docgen", calls, handles=True))
    turn = await run_handler_turn(monkeypatch, QUERY, gate=_two_mode_gate())
    assert calls == ["insight", "docgen"]
    assert turn.final["content"] == "docgen answer"
