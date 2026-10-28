"""Behavioural sibling for ``TestWiring.test_handle_submit_marks_at_ingress``
in tests/unit/test_pacing_metrics_fix.py (BC-63: it grepped
``_handle_submit_inner``'s source for ``mark_query_time()`` and for the string
order against ``_prepare_submit_context``).

Incident date: 2026-07-25 ([TIME CONTEXT] showed "53 m" for a 2-minute gap).
Bug classes: BC-63, BC-62. THE deployed ``_handle_submit_inner`` is driven with
a recording time manager; the prompt-preparation prelude is replaced with a
recorder that notes whether the turn had already been marked when prompt
building began, then stops the turn.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import gui.handlers as handlers


class _Stop(Exception):
    pass


class _TimeManager:
    def __init__(self, log):
        self.log = log

    def mark_query_time(self):
        self.log.append("mark")


def _orchestrator(log):
    orch = MagicMock()
    orch.time_manager = _TimeManager(log)
    orch.active_documents = None
    return orch


async def _run(monkeypatch, user_text, log):
    async def fake_prepare(ctx):
        log.append("prepare")
        raise _Stop()
        yield  # pragma: no cover - makes this an async generator

    async def fake_files(text, files):
        return SimpleNamespace(text_content=text, documents=[], images=[])

    monkeypatch.setattr(handlers, "_prepare_submit_context", fake_prepare)
    monkeypatch.setattr(handlers.file_processor, "process_files_structured", fake_files)
    gen = handlers._handle_submit_inner(
        user_text, [], [], False, _orchestrator(log))
    try:
        async for _ in gen:
            pass
    except _Stop:
        pass
    except Exception as e:  # surfaced so a failing setup is not read as a pass
        log.append(f"error:{type(e).__name__}:{e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "how is the build going",
    "    how is the build\n    going\n   ",
])
async def test_turn_is_marked_before_prompt_preparation(monkeypatch, text):
    log = []
    await _run(monkeypatch, text, log)
    assert "prepare" in log, log
    assert log.index("mark") < log.index("prepare"), log
    assert log.count("mark") == 1


@pytest.mark.asyncio
async def test_empty_input_does_not_mark(monkeypatch):
    log = []
    await _run(monkeypatch, "   ", log)
    assert "mark" not in log
