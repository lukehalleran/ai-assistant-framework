"""store_interaction -> user-claim ledger hook (memory/memory_storage.py).

Incident 2026-10-20 (restatement drift). Classes: BC-75, BC-52, BC-38 (the
authored/attachment boundary: a merged attachment blob is never the user's
words), BC-58 (every store path goes through store_interaction).

Drives the deployed ``MemoryStorage.store_interaction`` / ``MemoryCoordinator``
forwarding / ``gui.handlers`` receipt plumbing with the real writer replaced by
a recording coroutine. Latency: the background storage task awaits the bounded
write; every other caller gets a detached task and never waits.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import memory.claim_ledger_writer as writer
from config import app_config
from memory.memory_storage import MemoryStorage

BLOB = "Here is where I am at. I am only 6 days behind.\n\n[attached transcript]\nProfessor: week 7 covers 40 pages."
AUTHORED = "Here is where I am at. I am only 6 days behind."


def make_storage(model_manager="MM"):
    chroma = MagicMock()
    chroma.add_conversation_memory.return_value = "doc-123"
    return MemoryStorage(corpus_manager=MagicMock(), chroma_store=chroma, fact_extractor=MagicMock(),
                         model_manager=model_manager)


@pytest.fixture()
def enabled(monkeypatch):
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_ENABLED", True)


@pytest.fixture()
def recorder(monkeypatch):
    calls = []

    async def fake_record_turn(user_text, said_at, turn_id, model_manager, *, receipt=None):
        calls.append({"user_text": user_text, "said_at": said_at, "turn_id": turn_id,
                      "model_manager": model_manager, "receipt": receipt})
        out = {"claim_ledger_written": 2, "claim_ledger_status": "ok"}
        if receipt is not None:
            receipt.update(out)
        return out

    monkeypatch.setattr(writer, "record_turn", fake_record_turn)
    return calls


@pytest.mark.asyncio
async def test_hook_gets_the_user_authored_text_not_the_merged_attachment_blob(enabled, recorder):
    ms = make_storage()
    await ms.store_interaction(query=BLOB, response="Noted.", user_text=AUTHORED, ledger_receipt={})
    (call,) = recorder
    assert call["user_text"] == AUTHORED and "Professor" not in call["user_text"]
    assert call["turn_id"] == "doc-123" and call["model_manager"] == "MM"


@pytest.mark.asyncio
async def test_an_attachment_only_turn_passes_empty_authored_text_never_the_blob(enabled, recorder):
    await make_storage().store_interaction(query=BLOB, response="Noted.", user_text="", ledger_receipt={})
    assert recorder[0]["user_text"] == ""


@pytest.mark.asyncio
async def test_without_user_text_the_query_is_the_authored_text_like_the_fact_path(enabled, recorder):
    await make_storage().store_interaction(query=AUTHORED, response="Noted.", ledger_receipt={})
    assert recorder[0]["user_text"] == AUTHORED


@pytest.mark.asyncio
async def test_flag_off_never_calls_the_writer(recorder):
    assert app_config.CLAIM_LEDGER_ENABLED is False
    ms = make_storage()
    assert await ms.store_interaction(query=AUTHORED, response="Noted.", ledger_receipt={}) == "doc-123"
    assert recorder == []


@pytest.mark.asyncio
async def test_a_skipped_store_never_reaches_the_writer(enabled, recorder):
    ms = make_storage()
    assert await ms.store_interaction(query=AUTHORED, response="[API Error] boom") is None
    assert recorder == []


@pytest.mark.asyncio
async def test_hook_runs_after_the_turn_is_persisted(enabled, monkeypatch):
    ms = make_storage()
    order = []
    ms.chroma_store.add_conversation_memory.side_effect = lambda *a, **k: order.append("chroma") or "doc-1"

    async def record(*a, **k):
        order.append("ledger")
        return {}

    monkeypatch.setattr(writer, "record_turn", record)
    await ms.store_interaction(query=AUTHORED, response="Noted.", ledger_receipt={})
    assert order == ["chroma", "ledger"]


@pytest.mark.asyncio
async def test_background_caller_with_a_receipt_awaits_the_write_and_gets_the_keys(enabled, recorder):
    receipt = {}
    await make_storage().store_interaction(query=AUTHORED, response="Noted.", ledger_receipt=receipt)
    assert receipt == {"claim_ledger_written": 2, "claim_ledger_status": "ok"}


@pytest.mark.asyncio
async def test_inline_caller_never_waits_for_a_slow_ledger_write(enabled, monkeypatch):
    started, release, finished = asyncio.Event(), asyncio.Event(), []

    async def slow(*a, **k):
        started.set()
        await release.wait()
        finished.append(True)
        return {}

    monkeypatch.setattr(writer, "record_turn", slow)
    result = await asyncio.wait_for(make_storage().store_interaction(query=AUTHORED, response="Noted."), 2)
    assert result == "doc-123" and finished == []            # returned while the write is still running
    await asyncio.wait_for(started.wait(), 2)                # ... and the detached task did start
    release.set()
    await asyncio.sleep(0.05)
    assert finished == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [None, {}])
async def test_a_raising_writer_never_breaks_storage(enabled, monkeypatch, receipt):
    async def boom(*a, **k):
        raise RuntimeError("ledger exploded")

    monkeypatch.setattr(writer, "record_turn", boom)
    kwargs = {} if receipt is None else {"ledger_receipt": receipt}
    assert await make_storage().store_interaction(query=AUTHORED, response="Noted.", **kwargs) == "doc-123"
    await asyncio.sleep(0.02)          # let a detached task finish and be reaped


# --- coordinator forwarding -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_coordinator_forwards_the_receipt_and_omits_it_when_absent():
    from memory.memory_coordinator import MemoryCoordinator
    coord = object.__new__(MemoryCoordinator)
    seen = []

    async def fake_store(query, response, tags, **kwargs):
        seen.append(kwargs)
        return "id-1"

    coord._storage = SimpleNamespace(store_interaction=fake_store, conversation_context=[],
                                     interactions_since_consolidation=0)
    coord.current_topic, coord.conversation_context, coord.thread_store = "general", [], None
    await coord.store_interaction("q", "r", user_text="u", ledger_receipt={"x": 1})
    await coord.store_interaction("q", "r", user_text="u")
    assert seen[0]["ledger_receipt"] == {"x": 1} and "ledger_receipt" not in seen[1]


# --- handlers: receipt plumbing -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_storage_passes_a_receipt_only_when_enabled(monkeypatch):
    import gui.handlers as handlers
    captured = []

    async def fake_background(**kwargs):
        captured.append(kwargs)
        kwargs.get("ledger_receipt", {}).update({"claim_ledger_written": 1, "claim_ledger_status": "ok"}) \
            if "ledger_receipt" in kwargs else None
        return None

    monkeypatch.setattr(handlers, "_background_store_interaction", fake_background)
    orch = SimpleNamespace(current_topic="general", memory_system=None)

    def dispatch():
        return handlers._dispatch_storage(orch, "q", "r", "q", "r", "p", [], MagicMock(), "s", {}, "enhanced")

    off = dispatch()
    await off
    assert "ledger_receipt" not in captured[0] and not hasattr(off, "ledger_receipt")
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_ENABLED", True)
    on = dispatch()
    await on
    assert captured[1]["ledger_receipt"] is on.ledger_receipt
    assert on.ledger_receipt == {"claim_ledger_written": 1, "claim_ledger_status": "ok"}


@pytest.mark.asyncio
async def test_track_storage_task_copies_the_ledger_receipt_into_the_turn_row():
    import gui.handlers as handlers

    async def storage():
        return None

    task = asyncio.create_task(storage())
    task.ledger_receipt = {"claim_ledger_written": 3, "claim_ledger_status": "ok"}
    ctx = SimpleNamespace(telemetry={"intent": "x"}, debug_record={})
    handlers._track_storage_task(ctx, task)
    await task
    await asyncio.sleep(0)
    assert ctx.telemetry == {"intent": "x", "claim_ledger_written": 3, "claim_ledger_status": "ok"}
    assert "claim_ledger_status" not in ctx.debug_record        # the share-surface debug record is untouched


@pytest.mark.asyncio
async def test_track_storage_task_without_a_receipt_changes_nothing():
    import gui.handlers as handlers

    async def storage():
        return None

    task = asyncio.create_task(storage())
    ctx = SimpleNamespace(telemetry={"intent": "x"}, debug_record={})
    handlers._track_storage_task(ctx, task)
    await task
    await asyncio.sleep(0)
    assert ctx.telemetry == {"intent": "x"}


# --- startup: a corrupt store aborts through the coordinator ------------------------------------------------

def test_coordinator_startup_with_a_corrupt_ledger_raises_the_startup_abort_error(monkeypatch):
    import memory.claim_ledger_store as store
    from memory.memory_coordinator import MemoryCoordinator
    from utils.safe_json import CorruptStoreError
    monkeypatch.setattr(app_config, "CLAIM_LEDGER_ENABLED", True)
    open(store.store_path(), "w").write("{ broken")
    with pytest.raises(CorruptStoreError):
        MemoryCoordinator(corpus_manager=MagicMock(), chroma_store=MagicMock())


def test_coordinator_startup_with_the_ledger_off_never_touches_the_store(monkeypatch):
    import memory.claim_ledger_store as store
    from memory.memory_coordinator import MemoryCoordinator
    open(store.store_path(), "w").write("{ broken")
    MemoryCoordinator(corpus_manager=MagicMock(), chroma_store=MagicMock())     # no raise: flag off
