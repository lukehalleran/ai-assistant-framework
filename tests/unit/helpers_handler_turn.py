"""Turn driver for the DEPLOYED ``gui.handlers.handle_submit`` (BC-63).

Purpose: a test that greps the source of ``_run_enhanced`` /
``_run_agentic_search`` / ``handle_submit`` for a name proves only that the name
is still spelled there. This helper runs the real dispatcher and the real mode
body for one turn, with the model, stream, storage and tool seams replaced by
recording fakes, so a test can assert the behaviour the name stands for (the
system prompt the model stream received, what storage was handed, the order of
storage tracking and telemetry, the telemetry the turn recorded, the text the
user was shown).

Seams faked (everything between them is the deployed code):
  * ``handlers.file_processor`` / ``get_conversation_logger``
  * ``orchestrator.prepare_prompt``              -> a fixed (prompt, system, raw_context)
  * ``response_generator.generate_streaming_response`` -> scripted chunks, recording
    the prompt / system prompt / model it received
  * ``orchestrator.agentic_controller.run_agentic_search`` -> scripted items
  * ``core.agentic.gate.evaluate_agentic_gate`` -> a REAL ``AgenticDecision``
    (``apply_intent_veto`` is the identity so the scripted decision is what runs)
  * ``handlers._dispatch_storage`` -> recording (no Chroma, no corpus)
  * ``handlers._write_turn_telemetry`` -> recording (the telemetry dict the turn
    accumulated is read from ``turn.ctx.telemetry``)
  * ``handlers._apply_action_guard`` -> returns no suffix unless the test says so

Usage::

    turn = await run_handler_turn(
        monkeypatch, "How is the weather in the sample town today?",
        stream_chunks=["Fine.", " Sunny."],
        gate=make_gate_decision(trigger_source="tier1_memory"),
    )
    turn.final["content"]
    turn.stream_calls[0]["system_prompt"]
    turn.ctx.telemetry["gate_trigger_source"]
    turn.events                      # ["track_storage", "write_telemetry"]
    turn.storage_calls[0].kwargs["prompt_only_suffix"]

The orchestrator is the ``_make_orchestrator`` mock from test_handle_submit.py;
pass ``orchestrator=`` to reuse one, and ``orch_attrs={...}`` to set attributes
on it (e.g. ``_last_section_outcomes``).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from config import app_config
import core.agentic.gate as gate_mod
import gui.handlers as handlers
from core.agentic.gate import AgenticDecision
from tests.unit.test_handle_submit import (
    _FakeProcessedFilesResult,
    _make_file_processor_mock,
    _make_orchestrator,
)


@dataclass
class StorageCall:
    args: tuple
    kwargs: dict

    @property
    def stored_text(self) -> str:
        """The assistant text handed to persistence (3rd positional arg)."""
        return self.args[2]

    @property
    def merged_input(self) -> str:
        return self.args[1]


@dataclass
class HandlerTurn:
    chunks: List[Any]
    ctx: Any
    orchestrator: Any
    storage_calls: List[StorageCall] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    stream_calls: List[Dict[str, Any]] = field(default_factory=list)
    gate_calls: List[Dict[str, Any]] = field(default_factory=list)
    telemetry_modes: List[str] = field(default_factory=list)

    @property
    def final(self) -> Optional[dict]:
        """The last non-progress assistant chunk (what the user ends up reading)."""
        for chunk in reversed(self.chunks):
            if isinstance(chunk, dict) and chunk.get("content") and not chunk.get("is_progress"):
                return chunk
        return None

    @property
    def debug(self) -> Optional[dict]:
        for chunk in reversed(self.chunks):
            if isinstance(chunk, dict) and chunk.get("debug"):
                return chunk["debug"]
        return None

    @property
    def contents(self) -> List[str]:
        return [c["content"] for c in self.chunks
                if isinstance(c, dict) and isinstance(c.get("content"), str)]


def make_gate_decision(**fields) -> AgenticDecision:
    """A real ``AgenticDecision``; defaults to 'did not trigger' (enhanced route)."""
    fields.setdefault("should_trigger", False)
    return AgenticDecision(**fields)


def stream_script(chunks) -> Callable:
    """``generate_streaming_response`` replacement yielding ``chunks`` in order."""
    async def _gen(*args, **kwargs):
        for chunk in chunks:
            yield chunk
    return _gen


async def run_handler_turn(
    monkeypatch,
    user_text: str,
    *,
    stream_chunks=("Hello", " world"),
    agentic_items: Optional[list] = None,
    gate: Optional[AgenticDecision] = None,
    agentic_enabled: Optional[bool] = None,
    orchestrator=None,
    orch_attrs: Optional[dict] = None,
    prepare_result: tuple = ("Test prompt", "Test system prompt", {}),
    history: Optional[list] = None,
    files: Optional[list] = None,
    files_result=None,
    use_raw_gpt: bool = False,
    duel: Optional[dict] = None,
    grounding_mode: str = "off",
    personal_claim_mode: str = "off",
    offer_armed: bool = False,
    action_guard_suffix: str = "",
    fast_mode: bool = False,
) -> HandlerTurn:
    """Run one turn through the deployed ``handlers.handle_submit``; see module doc."""
    gate_enabled = gate is not None if agentic_enabled is None else agentic_enabled
    orch = orchestrator or _make_orchestrator(
        agentic_enabled=gate_enabled,
        duel_enabled=bool(duel),
        duel_gens=(duel or {}).get("gens"),
        duel_sels=(duel or {}).get("sels"),
        streaming_chunks=list(stream_chunks),
        agentic_items=agentic_items,
        prepare_result=prepare_result,
    )
    if duel and "result" in duel:
        orch.response_generator.generate_duel_and_judge = AsyncMock(return_value=duel["result"])
    for name, value in (orch_attrs or {}).items():
        setattr(orch, name, value)

    turn = HandlerTurn(chunks=[], ctx=None, orchestrator=orch)

    # --- model stream: scripted + recording --------------------------------
    scripted = stream_script(stream_chunks)

    async def _recording_stream(*args, **kwargs):
        turn.stream_calls.append({"args": args, **kwargs})
        async for chunk in scripted(*args, **kwargs):
            yield chunk

    orch.response_generator.generate_streaming_response = _recording_stream

    # --- ingress / logging seams -------------------------------------------
    monkeypatch.setattr(
        handlers, "file_processor",
        _make_file_processor_mock(user_text) if files_result is None else _fp_returning(files_result),
    )
    monkeypatch.setattr(handlers, "get_conversation_logger", MagicMock())
    monkeypatch.setattr(handlers, "_persist_uploads", AsyncMock(return_value=None))
    monkeypatch.setattr(handlers, "_apply_action_guard", AsyncMock(return_value=action_guard_suffix))

    # --- modes captured at ingress ------------------------------------------
    monkeypatch.setattr(app_config, "GROUNDING_CHECK_ENABLED", grounding_mode != "off")
    monkeypatch.setattr(app_config, "GROUNDING_MODE", grounding_mode if grounding_mode != "off" else "log_only")
    monkeypatch.setattr(app_config, "PERSONAL_CLAIM_CHECK_ENABLED", personal_claim_mode != "off")
    monkeypatch.setattr(
        app_config, "PERSONAL_CLAIM_MODE",
        personal_claim_mode if personal_claim_mode != "off" else "log_only",
    )

    # --- gate ----------------------------------------------------------------
    if gate is not None:
        async def _gate(**kwargs):
            turn.gate_calls.append(kwargs)
            return gate
        monkeypatch.setattr(gate_mod, "evaluate_agentic_gate", _gate)
    else:
        async def _gate_must_not_run(**kwargs):  # pragma: no cover - guard
            turn.gate_calls.append(kwargs)
            return make_gate_decision()
        monkeypatch.setattr(gate_mod, "evaluate_agentic_gate", _gate_must_not_run)
    monkeypatch.setattr(gate_mod, "apply_intent_veto", lambda decision, *a, **kw: decision)
    monkeypatch.setattr(gate_mod, "maybe_arm_insight_offer", lambda *a, **kw: offer_armed)

    # --- storage + telemetry: recording --------------------------------------
    def _dispatch(*args, **kwargs):
        turn.storage_calls.append(StorageCall(args=args, kwargs=kwargs))
        turn.events.append("dispatch_storage")
        return None

    real_track = handlers._track_storage_task

    def _track(ctx, value):
        turn.events.append("track_storage")
        return real_track(ctx, value)

    def _telemetry(ctx, mode, *args, **kwargs):
        turn.events.append("write_telemetry")
        turn.telemetry_modes.append(mode)

    monkeypatch.setattr(handlers, "_dispatch_storage", _dispatch)
    monkeypatch.setattr(handlers, "_track_storage_task", _track)
    monkeypatch.setattr(handlers, "_write_turn_telemetry", _telemetry)

    # --- expose the turn's SubmitContext ---------------------------------------
    real_ctx_cls = handlers.SubmitContext

    def _ctx_factory(*args, **kwargs):
        ctx = real_ctx_cls(*args, **kwargs)
        turn.ctx = ctx
        return ctx

    monkeypatch.setattr(handlers, "SubmitContext", _ctx_factory)

    gen = handlers.handle_submit(
        user_text=user_text,
        files=files,
        history=history if history is not None else [],
        use_raw_gpt=use_raw_gpt,
        orchestrator=orch,
        fast_mode=fast_mode,
    )
    async for chunk in gen:
        turn.chunks.append(chunk)
    # let fire-and-forget callbacks scheduled by the turn settle
    await asyncio.sleep(0)
    return turn


def _fp_returning(files_result):
    mock = MagicMock()
    mock.process_files_structured = AsyncMock(return_value=files_result)
    return mock


def files_result_with_text(text: str) -> _FakeProcessedFilesResult:
    return _FakeProcessedFilesResult(text_content=text)


def doc_attachment_result(user_text: str, filename: str = "notes.txt", body: str = "Sample body."):
    """A files_result carrying one text attachment (drives the dispatcher's prompt-only suffix)."""
    doc = SimpleNamespace(
        filename=filename, extension=".txt", content_text=body, error="",
    )
    return _FakeProcessedFilesResult(text_content=f"{user_text}\n\n{body}", documents=[doc])
