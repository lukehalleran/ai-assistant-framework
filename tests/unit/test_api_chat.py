"""Tests for the FastAPI chat surface: POST /api/chat SSE stream + session routes.

Drives THE deployed pipeline (gui.handlers.handle_submit) through the API
adapter with the same mock-orchestrator factory as test_handle_submit.py —
no re-derived stand-in for the streaming contract.
"""

import json

import pytest
import httpx
from unittest.mock import MagicMock, patch

from api.app import create_app
from tests.unit.helpers_orchestrator import (
    _make_file_processor_mock,
    _make_orchestrator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sse(body: str):
    """Parse an SSE body into [(event, payload_dict_or_None), ...]. Skips comments."""
    events = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
        if event:
            events.append((event, data))
    return events


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _make_app(orch):
    return create_app(orch, start_background=False)


async def _post_chat(app, text, **kwargs):
    async with _client(app) as client:
        resp = await client.post("/api/chat", json={"text": text, **kwargs})
        body = (await resp.aread()).decode()
    return resp, _parse_sse(body)


def _patched_pipeline(user_text):
    """The same base patches test_handle_submit.py applies around handle_submit."""
    return [
        patch("gui.handlers.file_processor", _make_file_processor_mock(user_text)),
        patch("gui.handlers.get_conversation_logger", return_value=MagicMock()),
    ]


# ---------------------------------------------------------------------------
# Streaming contract
# ---------------------------------------------------------------------------

class TestChatStream:
    @pytest.mark.asyncio
    async def test_stream_event_ordering_and_cumulative_content(self):
        orch = _make_orchestrator(streaming_chunks=["Hello", " world"])
        app = _make_app(orch)

        for p in _patched_pipeline("How are you?"):
            p.start()
        try:
            resp, events = await _post_chat(app, "How are you?")
        finally:
            patch.stopall()

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        names = [e for e, _ in events]
        assert names[-1] == "complete", f"stream must end with complete, got {names}"
        assert "error" not in names

        # message events carry CUMULATIVE content (replace-render semantics)
        message_payloads = [d["content"] for e, d in events if e == "message"]
        if len(message_payloads) >= 2:
            assert message_payloads[0] in message_payloads[-1]

        complete = dict(events)["complete"]
        assert "Hello" in complete["content"] and "world" in complete["content"]
        assert "debug" in complete
        assert complete["debug"]["mode"] == "enhanced"

    @pytest.mark.asyncio
    async def test_session_history_updated_after_turn(self):
        orch = _make_orchestrator(streaming_chunks=["Answer."])
        app = _make_app(orch)

        for p in _patched_pipeline("Question?"):
            p.start()
        try:
            await _post_chat(app, "Question?")
        finally:
            patch.stopall()

        async with _client(app) as client:
            resp = await client.get("/api/session")
        state = resp.json()
        assert [m["role"] for m in state["history"]] == ["user", "assistant"]
        assert state["history"][0]["content"] == "Question?"
        assert "Answer." in state["history"][1]["content"]

    @pytest.mark.asyncio
    async def test_error_event_on_pipeline_exception(self):
        orch = _make_orchestrator()
        app = _make_app(orch)

        async def _boom(*args, **kwargs):
            raise RuntimeError("pipeline exploded")
            yield  # pragma: no cover — makes this an async generator factory

        with patch("gui.handlers.handle_submit", _boom):
            resp, events = await _post_chat(app, "hi")

        names = [e for e, _ in events]
        assert "error" in names
        error = dict(events)["error"]
        assert "pipeline exploded" in error["message"]
        # Session must not be left with a dangling user message and no reply
        session = app.state.daemon.session
        assert session.history[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_conflict_when_stream_in_flight(self):
        orch = _make_orchestrator()
        app = _make_app(orch)
        session = app.state.daemon.session

        await session.stream_lock.acquire()
        try:
            async with _client(app) as client:
                resp = await client.post("/api/chat", json={"text": "hi"})
            assert resp.status_code == 409
        finally:
            session.stream_lock.release()

    @pytest.mark.asyncio
    async def test_pending_action_id_surfaces_in_complete(self):
        orch = _make_orchestrator(streaming_chunks=["Done."])
        app = _make_app(orch)

        async def _with_action(*args, **kwargs):
            yield {"role": "assistant", "content": "Done.",
                   "debug": {"mode": "enhanced"}, "pending_action_id": "act-42"}

        with patch("gui.handlers.handle_submit", _with_action):
            resp, events = await _post_chat(app, "send that email")

        complete = dict(events)["complete"]
        assert complete["pending_action_id"] == "act-42"
        assert app.state.daemon.session.pending_action_id == "act-42"

    @pytest.mark.asyncio
    async def test_enable_citations_flag_reaches_orchestrator(self):
        orch = _make_orchestrator(streaming_chunks=["ok"])
        app = _make_app(orch)

        for p in _patched_pipeline("q"):
            p.start()
        try:
            await _post_chat(app, "q", enable_citations=True)
        finally:
            patch.stopall()

        assert orch.enable_citations is True

    @pytest.mark.asyncio
    async def test_file_ids_resolve_to_shims_for_pipeline(self, tmp_path):
        orch = _make_orchestrator(streaming_chunks=["ok"])
        app = _make_app(orch)

        upload = tmp_path / "notes.txt"
        upload.write_text("hello")
        file_id = app.state.daemon.register_upload(str(upload), "notes.txt", 5)

        fp = _make_file_processor_mock("q")
        patches = [
            patch("gui.handlers.file_processor", fp),
            patch("gui.handlers.get_conversation_logger", return_value=MagicMock()),
        ]
        for p in patches:
            p.start()
        try:
            async with _client(app) as client:
                resp = await client.post("/api/chat", json={"text": "q", "file_ids": [file_id]})
                await resp.aread()
        finally:
            patch.stopall()

        # The pipeline's FileProcessor received our shim (with .name = temp path)
        args = fp.process_files_structured.await_args
        files_arg = args.args[1] if len(args.args) > 1 else args.kwargs.get("files")
        assert files_arg and files_arg[0].name == str(upload)

    @pytest.mark.asyncio
    async def test_progress_and_thinking_chunks_map_to_events(self):
        orch = _make_orchestrator()
        app = _make_app(orch)

        async def _staged(*args, **kwargs):
            yield {"role": "assistant", "content": "🔍 Searching...", "is_progress": True}
            yield {"role": "assistant", "content": "", "is_thinking": True}
            yield {"role": "assistant", "content": "Final answer"}
            yield {"role": "assistant", "content": "Final answer", "debug": {"mode": "enhanced"}}

        with patch("gui.handlers.handle_submit", _staged):
            resp, events = await _post_chat(app, "hi")

        names = [e for e, _ in events]
        assert "progress" in names
        assert "thinking" in names
        assert names[-1] == "complete"
        assert dict(events)["complete"]["content"] == "Final answer"


# ---------------------------------------------------------------------------
# Session routes
# ---------------------------------------------------------------------------

class TestSessionRoutes:
    @pytest.mark.asyncio
    async def test_clear_session_is_ui_state_only(self):
        """DELETE /api/session must never touch stored memory (repo safety rule)."""
        orch = _make_orchestrator()
        app = _make_app(orch)
        session = app.state.daemon.session
        session.history.append({"role": "user", "content": "x"})
        session.pending_action_id = "a1"

        async with _client(app) as client:
            resp = await client.delete("/api/session")

        assert resp.status_code == 204
        assert session.history == []
        assert session.pending_action_id is None
        # No deletion API on the memory system was called
        orch.memory_system.store_interaction.assert_not_awaited()
        assert not any(
            "delete" in str(c) for c in orch.memory_system.chroma_store.mock_calls
        ), "session clear must not call any chroma delete API"
