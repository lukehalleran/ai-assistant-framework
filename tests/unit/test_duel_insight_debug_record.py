"""
F13c-2b: duel (_run_duel) and insight-assembly (_run_insight_mode) must set
`ctx.debug_record` to the exact dict object yielded in their final chunk,
BEFORE that yield -- the same object api/chat_service.py appends to
session.debug_records. Without it, F13c-1's `_track_storage_task` done-
callback (which labels `ctx.debug_record["storage_failed"]` only when that
attribute is already a dict) can never reach the delivered record on these
two paths, so a background memory-save failure on a duel or insight turn
never shows up downstream (F13c-2a's SPA read included).

Fixtures are local, minimal copies of the test_insight_mode_handler.py
`_ctx`/`stages` shapes (FIXTURE RULE) plus a smallest-possible duel fixture.
No real models, stores, Gradio, or API server -- orchestrator is a MagicMock;
`_dispatch_storage` is replaced with a callable returning a real
asyncio.Task so the real F13c-1 done-callback runs against a real Task.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import gui.handlers as handlers
from core.insight.types import EvidenceItem, FacetPlan, FacetQuery


# --------------------------------------------------------------------------
# shared: drain an async generator, then (if a real background storage task
# was tracked) await it and yield once more so its add_done_callback -- which
# asyncio schedules via call_soon once the task settles -- has actually run
# before the test inspects ctx.debug_record.
# --------------------------------------------------------------------------
def _drain(agen, ctx):
    async def _go():
        chunks = []
        async for c in agen:
            chunks.append(c)
        task = getattr(ctx, "storage_task", None)
        if task is not None:
            await task
            await asyncio.sleep(0)
        return chunks
    return asyncio.run(_go())


async def _ok_storage_result():
    return None


async def _failing_storage_result():
    return "store_interaction: RuntimeError"


def _fake_dispatch(coro_fn):
    """Matches _dispatch_storage's call signature; ignores the real args and
    returns a REAL asyncio.Task wrapping coro_fn -- no store, no orchestrator
    call, just something for _track_storage_task's real done-callback to
    attach to and later read .result() from."""
    def _dispatch(*a, **k):
        return asyncio.create_task(coro_fn())
    return _dispatch


# ---------------------------------------------------------------- duel ----

_DUEL_RESULT = {
    "answer": "Duel answer text.",
    "thinking_a": "reasoning a", "thinking_b": "reasoning b",
    "model_a": "gen1", "model_b": "gen2", "winner": "gen1",
    "scores": {"gen1": 0.9, "gen2": 0.5},
}


def _duel_ctx():
    orchestrator = MagicMock()
    orchestrator.enable_citations = False  # gate: avoid the citations branch
    orchestrator.memory_system.session_id = "sess-duel"
    orchestrator.model_manager.get_active_model_name = MagicMock(return_value="active-model")
    orchestrator.response_generator.generate_duel_and_judge = AsyncMock(
        return_value=_DUEL_RESULT,
    )
    ctx = handlers.SubmitContext(
        user_text="duel question", files=None, history=[], use_raw_gpt=False,
        orchestrator=orchestrator, personality=None, fast_mode=False,
        conversation_logger=None, file_names=[], merged_input="duel question",
        files_result=None,
    )
    ctx.full_prompt = "full prompt for duel"
    ctx.system_prompt = "system prompt for duel"
    return ctx


def _run_duel(monkeypatch, dispatch):
    monkeypatch.setattr(handlers, "_dispatch_storage", dispatch)
    ctx = _duel_ctx()
    chunks = _drain(
        handlers._run_duel(
            ctx, ("gen1", "gen2"), ("judge",),
            {"best_of_latency_budget_s": 0},  # gate: skip the wait_for wrap
        ),
        ctx,
    )
    return ctx, chunks


class TestDuelDebugRecord:
    def test_final_chunk_debug_is_ctx_debug_record(self, monkeypatch):
        ctx, chunks = _run_duel(monkeypatch, lambda *a, **k: None)
        assert ctx.handled is True
        assert ctx.storage_dispatched is True
        final = chunks[-1]
        assert "debug" in final
        assert final["debug"] is ctx.debug_record
        # never call _capture_delivery on this path (brief: behaviour change)
        assert "wall_elapsed_s" not in ctx.debug_record
        assert not any(k.startswith("grounding_") for k in ctx.debug_record)

    def test_storage_failure_labels_the_delivered_record(self, monkeypatch):
        ctx, chunks = _run_duel(monkeypatch, _fake_dispatch(_failing_storage_result))
        final = chunks[-1]
        assert final["debug"] is ctx.debug_record
        assert ctx.debug_record["storage_failed"] == "store_interaction: RuntimeError"

    def test_storage_success_control_no_key(self, monkeypatch):
        ctx, chunks = _run_duel(monkeypatch, _fake_dispatch(_ok_storage_result))
        final = chunks[-1]
        assert "storage_failed" not in final["debug"]
        assert "storage_failed" not in ctx.debug_record

    def test_chunk_and_record_shape_unchanged_by_storage_outcome(self, monkeypatch):
        ok_ctx, ok_chunks = _run_duel(monkeypatch, _fake_dispatch(_ok_storage_result))
        fail_ctx, fail_chunks = _run_duel(monkeypatch, _fake_dispatch(_failing_storage_result))
        assert ok_chunks[-1]["content"] == fail_chunks[-1]["content"]
        assert set(fail_ctx.debug_record) - {"storage_failed"} == set(ok_ctx.debug_record)


# -------------------------------------------------------------- insight ---

def _insight_ctx(intent_dict):
    orchestrator = MagicMock()
    orchestrator.model_manager.get_active_model_name = MagicMock(return_value="kimi-3")
    orchestrator.memory_system.chroma_store = MagicMock()  # gate: chroma_store required
    ctx = handlers.SubmitContext(
        user_text=intent_dict.get("raw_query", "q"),
        files=None, history=[], use_raw_gpt=False,
        orchestrator=orchestrator, personality=None, fast_mode=False,
        conversation_logger=None, file_names=[], merged_input="q",
        files_result=None,
    )
    ctx.raw_context = {}
    ctx.gate_decision = MagicMock()
    ctx.gate_decision.insight_intent = intent_dict
    return ctx


@pytest.fixture
def insight_stages(monkeypatch):
    """Patch the three stage functions theme_sweep actually reaches (no
    assessor -- see test_insight_mode_handler.py's
    test_theme_sweep_owns_turn_with_provenance precedent)."""
    import core.insight.facets as facets_mod
    import core.insight.sweep as sweep_mod
    import core.insight.synthesizer as synth_mod

    plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q")], claims=["c"])
    monkeypatch.setattr(facets_mod, "decompose", AsyncMock(return_value=plan))
    monkeypatch.setattr(sweep_mod, "run_sweep", AsyncMock(return_value=[
        EvidenceItem(doc_id="d1", text="evidence one", date="2026-08-18",
                     collection="conversations", speaker="user"),
    ]))

    async def synthesize_stream(intent, evidence, assessment, **kw):
        yield "insight synthesis text"

    monkeypatch.setattr(synth_mod, "synthesize_stream", synthesize_stream)


def _run_insight(monkeypatch, insight_stages, dispatch):
    monkeypatch.setattr(handlers, "_dispatch_storage", dispatch)
    ctx = _insight_ctx({"kind": "theme_sweep", "theme": "sleep",
                        "wants_document": False, "raw_query": "gather it"})
    chunks = _drain(handlers._run_insight_mode(ctx), ctx)
    return ctx, chunks


class TestInsightDebugRecord:
    def test_final_chunk_debug_is_ctx_debug_record(self, monkeypatch, insight_stages):
        ctx, chunks = _run_insight(monkeypatch, insight_stages, lambda *a, **k: None)
        assert ctx.handled is True
        final = chunks[-1]
        assert "debug" in final
        assert final["debug"] is ctx.debug_record
        assert "wall_elapsed_s" not in ctx.debug_record
        assert not any(k.startswith("grounding_") for k in ctx.debug_record)

    def test_storage_failure_labels_the_delivered_record(self, monkeypatch, insight_stages):
        ctx, chunks = _run_insight(
            monkeypatch, insight_stages, _fake_dispatch(_failing_storage_result),
        )
        final = chunks[-1]
        assert final["debug"] is ctx.debug_record
        assert ctx.debug_record["storage_failed"] == "store_interaction: RuntimeError"

    def test_storage_success_control_no_key(self, monkeypatch, insight_stages):
        ctx, chunks = _run_insight(
            monkeypatch, insight_stages, _fake_dispatch(_ok_storage_result),
        )
        final = chunks[-1]
        assert "storage_failed" not in final["debug"]
        assert "storage_failed" not in ctx.debug_record

    def test_chunk_and_record_shape_unchanged_by_storage_outcome(self, monkeypatch, insight_stages):
        ok_ctx, ok_chunks = _run_insight(
            monkeypatch, insight_stages, _fake_dispatch(_ok_storage_result),
        )
        fail_ctx, fail_chunks = _run_insight(
            monkeypatch, insight_stages, _fake_dispatch(_failing_storage_result),
        )
        assert ok_chunks[-1]["content"] == fail_chunks[-1]["content"]
        assert set(fail_ctx.debug_record) - {"storage_failed"} == set(ok_ctx.debug_record)
