"""F13c-1 -- a background memory-save failure reaches the delivered debug
record and the turn record (design: failure_outcome_design.md, amendment
"F13 split and receipt carriers"). Precedents: _start_background_grounding.
finished (mutates ctx.debug_record in place) and core.orchestrator.
_hook_turn_telemetry (defers the turn row via add_done_callback).

Covers: _background_store_interaction's return value; the new
_track_storage_task(ctx, value) helper; _write_turn_telemetry's telemetry=/
telemetry_task= combined waiter; _hook_turn_telemetry.write_completed's
storage_*-key copy; end to end through the deployed handle_submit.

Fakes only; real asyncio.Task objects where the contract needs them; turn
telemetry goes to tmp_path (autouse sandbox + explicit patch here).
"""
import asyncio
import json
import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gui.handlers as handlers
from utils.retrieval_outcome import StoreWriteError

MARKER = "F13C1MARKER_9d2b7e51"


def _failing_store(source="store_interaction", reason="RuntimeError"):
    """AsyncMock store_interaction: raises StoreWriteError chained from a
    plain exception carrying MARKER (F13a/F13b pattern) -- a fix that logs
    or records {e} / the chained cause instead of the label would leak it."""
    async def _raise(*_a, **_kw):
        try:
            raise RuntimeError(MARKER)
        except RuntimeError as inner:
            raise StoreWriteError(source=source, reason=reason) from inner
    return AsyncMock(side_effect=_raise)


async def _label_coro(value):
    return value


async def _raise_marker():
    raise RuntimeError(MARKER)


async def _settle(task):
    """Await a task to completion (success, exception or cancellation), then
    yield a few times so its add_done_callback chain (scheduled via
    call_soon -- including a wrapping asyncio.gather's OWN internal
    bookkeeping callback, which needs an extra hop or two beyond a plain
    task's single callback) has actually run."""
    try:
        await task
    except BaseException:
        pass
    for _ in range(5):
        await asyncio.sleep(0)
    return task


def _patch_telemetry_path(monkeypatch, tmp_path):
    import config.app_config as config
    monkeypatch.setattr(config, "TURN_TELEMETRY_ENABLED", True)
    path = tmp_path / "turns.jsonl"
    monkeypatch.setattr(config, "TURN_TELEMETRY_PATH", str(path))
    return path


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# 1. _background_store_interaction return value
# ---------------------------------------------------------------------------

class TestBackgroundStoreInteractionReturnValue:
    @pytest.mark.asyncio
    async def test_failure_returns_label_and_logs_once(self, caplog):
        memory = SimpleNamespace(store_interaction=_failing_store())
        orchestrator = SimpleNamespace(memory_system=memory, current_topic="general")
        conv_logger = MagicMock()
        with caplog.at_level(logging.ERROR, logger="gradio_gui"):
            result = await handlers._background_store_interaction(
                orchestrator=orchestrator, merged_input="hi", response_to_store="r",
                tags=["t"], user_text="hi", final_output="hi", personality="default",
                file_names=[], conversation_logger=conv_logger,
            )
        assert result == "store_interaction: RuntimeError"
        assert MARKER not in result
        assert MARKER not in caplog.text
        conv_logger.log_interaction.assert_called_once()
        assert conv_logger.log_interaction.call_args.kwargs["metadata"]["storage_failed"] == result

    @pytest.mark.asyncio
    async def test_success_returns_none(self):
        memory = SimpleNamespace(store_interaction=AsyncMock(return_value="mem-1"))
        orchestrator = SimpleNamespace(memory_system=memory, current_topic="general")
        conv_logger = MagicMock()
        result = await handlers._background_store_interaction(
            orchestrator=orchestrator, merged_input="hi", response_to_store="r",
            tags=[], user_text="hi", final_output="hi", personality="default",
            file_names=[], conversation_logger=conv_logger,
        )
        assert result is None
        assert "storage_failed" not in conv_logger.log_interaction.call_args.kwargs["metadata"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("memory_system,conv_logger", [
        (None, MagicMock()),
        (SimpleNamespace(store_interaction=AsyncMock(return_value="mem-1")), None),
    ], ids=["memory_system_none", "conversation_logger_none"])
    async def test_outer_except_path_returns_none(self, memory_system, conv_logger):
        orchestrator = SimpleNamespace(memory_system=memory_system, current_topic="general")
        result = await handlers._background_store_interaction(
            orchestrator=orchestrator, merged_input="hi", response_to_store="r",
            tags=[], user_text="hi", final_output="hi", personality="default",
            file_names=[], conversation_logger=conv_logger,
        )
        assert result is None


# ---------------------------------------------------------------------------
# 2. _track_storage_task helper
# ---------------------------------------------------------------------------

class TestTrackStorageTaskHelper:
    @pytest.mark.asyncio
    async def test_label_sets_telemetry_and_same_debug_record_object(self):
        ctx = SimpleNamespace(telemetry={}, debug_record={})
        task = asyncio.create_task(_label_coro("store_interaction: RuntimeError"))
        handlers._track_storage_task(ctx, task)
        assert ctx.storage_task is task
        await _settle(task)
        assert ctx.telemetry["storage_failed"] == "store_interaction: RuntimeError"
        assert ctx.debug_record["storage_failed"] == "store_interaction: RuntimeError"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("make_task", [
        lambda: asyncio.create_task(_label_coro(None)),
        lambda: asyncio.create_task(_raise_marker()),
    ], ids=["none_result", "raised"])
    async def test_no_key_on_none_or_raise(self, make_task):
        ctx = SimpleNamespace(telemetry={}, debug_record={})
        task = make_task()
        handlers._track_storage_task(ctx, task)
        await _settle(task)  # must not raise, even for the "raised" case
        assert "storage_failed" not in ctx.telemetry
        assert "storage_failed" not in ctx.debug_record

    @pytest.mark.asyncio
    async def test_cancelled_task_no_key_no_exception(self):
        ctx = SimpleNamespace(telemetry={}, debug_record={})
        async def _never():
            await asyncio.sleep(10)
        task = asyncio.create_task(_never())
        handlers._track_storage_task(ctx, task)
        task.cancel()
        await _settle(task)
        assert "storage_failed" not in ctx.telemetry
        assert "storage_failed" not in ctx.debug_record

    @pytest.mark.parametrize("value", [MagicMock(), None, lambda: None])
    def test_non_task_value_sets_no_storage_task_attribute(self, value):
        ctx = SimpleNamespace(telemetry={})
        handlers._track_storage_task(ctx, value)
        assert not hasattr(ctx, "storage_task")

    @pytest.mark.asyncio
    async def test_ctx_without_telemetry_or_debug_record_no_exception(self):
        ctx = SimpleNamespace()
        task = asyncio.create_task(_label_coro("store_interaction: RuntimeError"))
        handlers._track_storage_task(ctx, task)
        await _settle(task)  # must not raise
        assert ctx.storage_task is task


# ---------------------------------------------------------------------------
# 3. _write_turn_telemetry deferral + deployed _hook_turn_telemetry
# ---------------------------------------------------------------------------

class TestWriteTurnTelemetryDeferral:
    def test_telemetry_dict_identity_preserved(self, monkeypatch):
        """An empty ctx.telemetry dict is the SAME object the hook reads --
        `telemetry or {}` would swap in a new dict here, and a later
        done-callback write to ctx.telemetry would never reach the row."""
        import core.orchestrator as orch_mod
        captured = {}
        monkeypatch.setattr(
            orch_mod, "run_post_response_hooks",
            lambda hctx: captured.__setitem__("t", hctx.telemetry),
        )
        ctx = SimpleNamespace(telemetry={}, orchestrator=None, user_text="q")
        handlers._write_turn_telemetry(ctx, "enhanced", "s", "m", 0)
        assert captured["t"] is ctx.telemetry

    @pytest.mark.asyncio
    async def test_storage_only_hooks_run_immediately_row_deferred(self, monkeypatch, tmp_path):
        import core.orchestrator as orch_mod
        path = _patch_telemetry_path(monkeypatch, tmp_path)
        calls = []
        spied = [
            (n, (lambda hctx, _n=n, _f=f: (calls.append(_n), _f(hctx))[1]))
            for n, f in orch_mod.POST_RESPONSE_HOOKS
        ]
        monkeypatch.setattr(orch_mod, "POST_RESPONSE_HOOKS", spied)

        release = asyncio.Event()

        async def _slow():
            await release.wait()
            return "store_interaction: RuntimeError"

        task = asyncio.create_task(_slow())
        ctx = SimpleNamespace(telemetry={}, orchestrator=None, user_text="q")
        handlers._track_storage_task(ctx, task)

        handlers._write_turn_telemetry(ctx, "enhanced", "s", "m", 5, response_text="hi")
        assert {n for n, _ in spied} == set(calls)  # every hook ran immediately
        assert _rows(path) == []  # row not yet written

        release.set()
        await _settle(task)
        rows = _rows(path)
        assert len(rows) == 1
        assert rows[0]["storage_failed"] == "store_interaction: RuntimeError"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("storage_first", [True, False])
    async def test_combined_grounding_and_storage_either_order(self, monkeypatch, tmp_path, storage_first):
        path = _patch_telemetry_path(monkeypatch, tmp_path)
        ev_a, ev_b = asyncio.Event(), asyncio.Event()
        ctx = SimpleNamespace(telemetry={}, orchestrator=None, user_text="q")

        async def _grounding():
            await (ev_a if storage_first else ev_b).wait()
            ctx.telemetry["grounding_status"] = "complete"

        async def _storage():
            await (ev_b if storage_first else ev_a).wait()
            return "store_interaction: RuntimeError"

        ctx.grounding_task = asyncio.create_task(_grounding())
        storage_task = asyncio.create_task(_storage())
        handlers._track_storage_task(ctx, storage_task)
        handlers._write_turn_telemetry(ctx, "enhanced", "s", "m", 5, response_text="hi")
        assert _rows(path) == []

        ev_a.set()
        await asyncio.sleep(0)
        ev_b.set()
        await _settle(ctx.grounding_task)
        await _settle(storage_task)
        rows = _rows(path)
        assert len(rows) == 1  # exactly one row -- no double-write either way
        assert rows[0]["grounding_status"] == "complete"
        assert rows[0]["storage_failed"] == "store_interaction: RuntimeError"

    def test_magicmock_ctx_truthy_attrs_do_not_abort_the_hooks(self, monkeypatch):
        """Fixture gate (COMPLIANCE): a MagicMock ctx auto-vivifies
        `.grounding_task`/`.storage_task` as truthy non-Task mocks --
        asyncio.gather would reject them, and an uncaught TypeError there
        would silently skip EVERY post-response hook (test_adaptive_adopters
        .py's TestCitationTeachesSearchWorthy shape, which regressed once
        during this batch's development)."""
        import core.orchestrator as orch_mod
        calls = []
        spied = [(n, (lambda hctx, _n=n, _f=f: (calls.append(_n), _f(hctx))[1]))
                 for n, f in orch_mod.POST_RESPONSE_HOOKS]
        monkeypatch.setattr(orch_mod, "POST_RESPONSE_HOOKS", spied)
        ctx = MagicMock()
        ctx.telemetry = {}
        handlers._write_turn_telemetry(ctx, "enhanced", "s", "m", 0)
        assert set(calls) == {n for n, _ in spied}  # every hook still ran

    def test_no_tasks_immediate_row(self, monkeypatch, tmp_path):
        path = _patch_telemetry_path(monkeypatch, tmp_path)
        ctx = SimpleNamespace(telemetry={}, orchestrator=None, user_text="q")
        handlers._write_turn_telemetry(ctx, "enhanced", "s", "m", 0)
        assert len(_rows(path)) == 1

    @pytest.mark.asyncio
    async def test_grounding_only_deferred_no_storage_today_behaviour(self, monkeypatch, tmp_path):
        path = _patch_telemetry_path(monkeypatch, tmp_path)
        ctx = SimpleNamespace(telemetry={}, orchestrator=None, user_text="q")

        async def _grounding():
            await asyncio.sleep(0)
            ctx.telemetry["grounding_status"] = "complete"

        ctx.grounding_task = asyncio.create_task(_grounding())
        handlers._write_turn_telemetry(ctx, "enhanced", "s", "m", 0)
        assert _rows(path) == []
        await _settle(ctx.grounding_task)
        rows = _rows(path)
        assert len(rows) == 1
        assert rows[0]["grounding_status"] == "complete"


# ---------------------------------------------------------------------------
# 4. End to end through the deployed handle_submit (enhanced route) --
# exercises the real core.orchestrator._hook_turn_telemetry.write_completed
# key filter (ownership item 4) at the same time, alongside
# TestWriteTurnTelemetryDeferral above (both call the real, unmocked
# run_post_response_hooks -> _hook_turn_telemetry).
# ---------------------------------------------------------------------------

class TestEndToEndHandleSubmit:
    @pytest.mark.asyncio
    async def test_storage_failure_reaches_turn_row_and_delivered_debug_record(self, monkeypatch, tmp_path):
        import config.app_config as config
        from tests.unit.test_handle_submit import _debug_record, _final_content, _make_orchestrator, _run_submit
        path = _patch_telemetry_path(monkeypatch, tmp_path)
        monkeypatch.setattr(config, "GROUNDING_CHECK_ENABLED", False)

        ok_orch = _make_orchestrator(streaming_chunks=["Hello", " world"])
        ok_results = await _run_submit("hi", ok_orch)
        await handlers.wait_for_pending_storage(timeout=2)

        bad_orch = _make_orchestrator(streaming_chunks=["Hello", " world"])
        bad_orch.memory_system.store_interaction = _failing_store()
        bad_results = await _run_submit("hi", bad_orch)
        bad_debug = _debug_record(bad_results)
        assert "storage_failed" not in bad_debug  # not yet -- dispatched, not settled
        await handlers.wait_for_pending_storage(timeout=2)

        ok_debug = _debug_record(ok_results)
        assert "storage_failed" not in ok_debug
        assert _final_content(bad_results) == _final_content(ok_results)  # no user-visible change
        assert bad_debug["storage_failed"] == "store_interaction: RuntimeError"  # updated in place
        assert MARKER not in json.dumps(bad_debug)

        rows = _rows(path)
        assert len(rows) == 2
        assert sum(1 for r in rows if r.get("storage_failed") == "store_interaction: RuntimeError") == 1
        assert sum(1 for r in rows if "storage_failed" in r) == 1


# ---------------------------------------------------------------------------
# Structural guard: the helper runs before _write_turn_telemetry at every
# _dispatch_storage call site.
# ---------------------------------------------------------------------------

class TestCallSiteOrder:
    def test_track_storage_task_precedes_write_turn_telemetry_at_every_site(self):
        import inspect
        src = inspect.getsource(handlers)
        def_start = src.index("def _dispatch_storage(")
        def_call_pos = def_start + len("def ")
        call_positions = [
            m.start() for m in re.finditer(r"_dispatch_storage\(", src)
            # Exclude the def itself and the module-docstring's bare mention
            # "_dispatch_storage(), ..." (empty parens -- not a real call).
            if m.start() != def_call_pos and src[m.end():m.end() + 1] != ")"
        ]
        assert len(call_positions) == 4, "expected exactly four _dispatch_storage call sites"
        # The window for each site runs to the NEXT dispatch call (or the end
        # of the module), not a pinned character count: a fixed 1200-char
        # window went stale on 2026-09-15 when the agentic site gained the
        # personal-claim kwarg lines (same drift class as the ordered-slice
        # guard's pinned line numbers, 09-05). The ordering guarantee is what
        # this test pins, not the size of the code between the calls.
        bounds = call_positions + [len(src)]
        for pos, nxt in zip(call_positions, bounds[1:]):
            window = src[pos:nxt]
            track_idx = window.find("_track_storage_task(ctx")
            write_idx = window.find("_write_turn_telemetry(")
            assert track_idx != -1, f"no _track_storage_task call found near offset {pos}"
            assert write_idx != -1, f"no _write_turn_telemetry call found near offset {pos}"
            assert track_idx < write_idx, f"_track_storage_task must precede _write_turn_telemetry near {pos}"
            assert write_idx < 4000, f"_write_turn_telemetry is {write_idx} chars past the dispatch near {pos} -- is it still the same site?"
