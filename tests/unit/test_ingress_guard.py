"""Ingress guard — duplicate-submit dedupe + client-error artifact strip.

Regression for the 2026-08-28 retrospective: the SPA "⚠️ Failed to fetch"
resend path double-processed turns for weeks (x2/x3 telemetry records for the
same heavy query on crisis days since ≥08-18, duplicate stored corpus entries,
and one stored query permanently carrying the error text). The guard is
IN-FLIGHT-ONLY by design — a resend after the first attempt finished may be
the REAL turn (live 08-28: 14:42 attempt died mid-stream, 14:46 resend
produced the actual reply), so completed-window dedupe is deliberately absent.
"""
import asyncio

import pytest

import gui.handlers as handlers
from gui.handlers import (
    _inflight_key,
    _strip_client_error_artifacts,
    _INFLIGHT_SUBMITS,
)


@pytest.fixture(autouse=True)
def _clean_inflight():
    _INFLIGHT_SUBMITS.clear()
    yield
    _INFLIGHT_SUBMITS.clear()


class TestClientErrorStrip:
    def test_line_anchored_artifact_removed(self):
        out = _strip_client_error_artifacts("my email text\n⚠️ Failed to fetch\n\nUgh")
        assert "Failed to fetch" not in out
        assert "my email text" in out and "Ugh" in out

    def test_plain_variant_removed(self):
        out = _strip_client_error_artifacts("hello\nFailed to fetch\nworld")
        assert "Failed to fetch" not in out

    def test_mid_sentence_mention_kept(self):
        text = "the request failed to fetch my data, can you retry"
        assert _strip_client_error_artifacts(text) == text

    def test_no_artifact_untouched(self):
        assert _strip_client_error_artifacts("ordinary message") == "ordinary message"
        assert _strip_client_error_artifacts("") == ""


class TestInflightDedupe:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @pytest.mark.asyncio
    async def test_duplicate_while_inflight_rejected(self, monkeypatch):
        """Second identical submit while the first is mid-stream gets the
        duplicate notice and never reaches the inner dispatcher."""
        entered = []
        release = asyncio.Event()

        async def fake_inner(user_text, *a, **kw):
            entered.append(user_text)
            await release.wait()
            yield {"role": "assistant", "content": "real answer"}

        monkeypatch.setattr(handlers, "_handle_submit_inner", fake_inner)
        text = "I rely on things making sense and being predictable."

        async def first():
            chunks = []
            async for c in handlers.handle_submit(text, None, [], False, object()):
                chunks.append(c)
            return chunks

        t1 = asyncio.create_task(first())
        await asyncio.sleep(0.05)  # let first register + block

        dup_chunks = []
        async for c in handlers.handle_submit(text, None, [], False, object()):
            dup_chunks.append(c)
        assert len(dup_chunks) == 1
        assert "duplicate" in dup_chunks[0]["content"].lower()
        assert entered == [text]  # inner ran exactly once

        release.set()
        first_chunks = await t1
        assert first_chunks[0]["content"] == "real answer"

    @pytest.mark.asyncio
    async def test_resend_after_completion_processes_normally(self, monkeypatch):
        """IN-FLIGHT-ONLY: once the first turn finishes, the same text runs
        again (the resend may be the real turn)."""
        entered = []

        async def fake_inner(user_text, *a, **kw):
            entered.append(user_text)
            yield {"role": "assistant", "content": "answer"}

        monkeypatch.setattr(handlers, "_handle_submit_inner", fake_inner)
        text = "What a fucking disaster. I am so confused right now."
        for _ in range(2):
            async for _c in handlers.handle_submit(text, None, [], False, object()):
                pass
        assert len(entered) == 2

    @pytest.mark.asyncio
    async def test_short_messages_exempt(self, monkeypatch):
        """Short repeats ('ugh', 'hello') are legit conversation, never deduped."""
        entered = []
        release = asyncio.Event()

        async def fake_inner(user_text, *a, **kw):
            entered.append(user_text)
            await release.wait()
            yield {"role": "assistant", "content": "ok"}

        monkeypatch.setattr(handlers, "_handle_submit_inner", fake_inner)

        t1 = asyncio.create_task(
            handlers.handle_submit("ugh", None, [], False, object()).__anext__())
        await asyncio.sleep(0.05)
        t2 = asyncio.create_task(
            handlers.handle_submit("ugh", None, [], False, object()).__anext__())
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.gather(t1, t2)
        assert len(entered) == 2

    @pytest.mark.asyncio
    async def test_inflight_entry_cleared_on_inner_exception(self, monkeypatch):
        """A crashed turn must not leave a stale in-flight entry blocking the
        retry (the retry after a failure is the expected recovery path)."""
        calls = []

        async def fake_inner(user_text, *a, **kw):
            calls.append(user_text)
            if len(calls) == 1:
                raise RuntimeError("stream died")
            yield {"role": "assistant", "content": "second attempt works"}

        monkeypatch.setattr(handlers, "_handle_submit_inner", fake_inner)
        text = "long enough message to be subject to the dedupe guard"

        with pytest.raises(RuntimeError):
            async for _c in handlers.handle_submit(text, None, [], False, object()):
                pass
        assert not _INFLIGHT_SUBMITS, "finally must clear the registration"

        chunks = []
        async for c in handlers.handle_submit(text, None, [], False, object()):
            chunks.append(c)
        assert chunks[0]["content"] == "second attempt works"

    @pytest.mark.asyncio
    async def test_artifact_stripped_before_inner(self, monkeypatch):
        """The inner dispatcher (and therefore storage) must never see the
        client error text."""
        seen = []

        async def fake_inner(user_text, *a, **kw):
            seen.append(user_text)
            yield {"role": "assistant", "content": "ok"}

        monkeypatch.setattr(handlers, "_handle_submit_inner", fake_inner)
        async for _c in handlers.handle_submit(
                "here is my long resent message\n⚠️ Failed to fetch\nUgh",
                None, [], False, object()):
            pass
        assert seen and "Failed to fetch" not in seen[0]

    def test_key_includes_files(self):
        assert _inflight_key("same text", ["a.txt"]) != _inflight_key("same text", ["b.txt"])
        assert _inflight_key("Same   Text", []) == _inflight_key("same text", [])
