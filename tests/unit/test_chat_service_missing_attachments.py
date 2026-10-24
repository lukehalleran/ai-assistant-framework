"""A requested attachment that cannot be served refuses the turn (audit A08).

Incident date 2026-10-23, class BC-47 / BC-58: ``resolve_uploads`` used to log
and skip unknown ids / vanished temp files, and ``submit_stream`` ran the turn
as text-only with no notice. Drives the deployed ``submit_stream`` with a
recording fake ``handle_submit`` (the model-facing seam only).
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from api.chat_service import submit_stream
from api.schemas import ChatRequest
from api.state import AppState


def _state():
    orch = MagicMock()
    return AppState(orchestrator=orch)


def _recorder(calls):
    async def fake(**kwargs):
        calls.append(kwargs)
        yield {"content": "ok", "debug": {"response": "ok"}, "pending_action_id": None}
    return fake


async def _run(state, req, calls):
    with patch("gui.handlers.handle_submit", _recorder(calls)):
        return [e async for e in submit_stream(req, state)]


def _file(tmp_path, name):
    p = tmp_path / name
    p.write_text("body")
    return str(p)


@pytest.mark.asyncio
async def test_unknown_id_refuses_turn(tmp_path):
    state, calls = _state(), []
    events = await _run(state, ChatRequest(text="summarize", file_ids=["gone"]), calls)
    assert calls == []
    assert state.session.history == []
    assert [e.event for e in events] == ["error"]
    assert "please re-attach" in events[0].data["message"]
    assert events[0].data["missing"][0]["reason"] == "unknown_id"


@pytest.mark.asyncio
async def test_vanished_temp_file_names_attachment(tmp_path):
    state, calls = _state(), []
    path = _file(tmp_path, "notes.txt")
    fid = state.register_upload(path, "notes.txt", 4)
    os.remove(path)
    events = await _run(state, ChatRequest(text="summarize", file_ids=[fid]), calls)
    assert calls == [] and state.session.history == []
    assert [e.event for e in events] == ["error"]
    assert "notes.txt" in events[0].data["message"]
    assert events[0].data["missing"][0]["reason"] == "file_missing"


@pytest.mark.asyncio
async def test_partial_batch_refuses_whole_turn(tmp_path):
    state, calls = _state(), []
    ok = state.register_upload(_file(tmp_path, "a.txt"), "a.txt", 4)
    bad_path = _file(tmp_path, "b.txt")
    bad = state.register_upload(bad_path, "b.txt", 4)
    os.remove(bad_path)
    events = await _run(state, ChatRequest(text="compare", file_ids=[ok, bad]), calls)
    assert calls == [] and state.session.history == []
    assert [e.event for e in events] == ["error"]
    msg = events[0].data["message"]
    assert "b.txt" in msg and "a.txt" not in msg


@pytest.mark.asyncio
async def test_no_file_ids_unchanged():
    state, calls = _state(), []
    events = await _run(state, ChatRequest(text="hello"), calls)
    assert len(calls) == 1 and calls[0]["files"] is None
    assert events[-1].event == "complete"
    assert [m["role"] for m in state.session.history] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_all_present_passes_files(tmp_path):
    state, calls = _state(), []
    fid = state.register_upload(_file(tmp_path, "a.txt"), "a.txt", 4)
    events = await _run(state, ChatRequest(text="read", file_ids=[fid]), calls)
    assert len(calls[0]["files"]) == 1
    assert events[-1].event == "complete"
