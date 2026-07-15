"""Tests for the per-turn progress bus (utils/turn_progress.py) and its
end-to-end wiring: events emitted inside prepare_prompt surface as is_progress
chunks from THE deployed handle_submit."""

import asyncio

import pytest
from unittest.mock import MagicMock, patch

from utils import turn_progress
from tests.unit.helpers_orchestrator import (
    _collect,
    _make_file_processor_mock,
    _make_orchestrator,
)


class TestBus:
    def teardown_method(self):
        turn_progress.end_turn()

    def test_emit_without_turn_is_noop(self):
        turn_progress.end_turn()
        turn_progress.emit("nobody listening")  # must not raise

    def test_begin_emit_drain(self):
        q = turn_progress.begin_turn()
        turn_progress.emit("step 1")
        turn_progress.emit("step 2")
        assert turn_progress.drain(q) == ["step 1", "step 2"]
        assert turn_progress.drain(q) == []

    def test_end_turn_detaches_queue(self):
        q = turn_progress.begin_turn()
        turn_progress.end_turn()
        turn_progress.emit("after end")
        assert turn_progress.drain(q) == []

    def test_full_queue_drops_silently(self):
        q = turn_progress.begin_turn()
        for i in range(500):
            turn_progress.emit(f"e{i}")  # beyond maxsize; must not raise
        assert len(turn_progress.drain(q)) <= 500

    def test_empty_text_ignored(self):
        q = turn_progress.begin_turn()
        turn_progress.emit("")
        assert turn_progress.drain(q) == []


class TestPipelineEventsSurfaceInStream:
    @pytest.mark.asyncio
    async def test_prepare_prompt_emits_reach_ui(self):
        from gui.handlers import handle_submit

        orch = _make_orchestrator(streaming_chunks=["done"])

        async def _emitting_prepare(*args, **kwargs):
            # Simulates the builder publishing during context gathering
            turn_progress.emit("📥 memory retrieval ✓ 1.2s · 12 hits")
            await asyncio.sleep(0.5)  # let one handler poll cycle run
            turn_progress.emit("🧱 Context retrieved (1.2s) — gating, dedup, token budget…")
            return ("Test prompt", "Test system prompt", {})

        orch.prepare_prompt = _emitting_prepare

        patches = [
            patch("gui.handlers.file_processor", _make_file_processor_mock("q")),
            patch("gui.handlers.get_conversation_logger", return_value=MagicMock()),
        ]
        for p in patches:
            p.start()
        try:
            results = await _collect(handle_submit(
                user_text="q", files=None, history=[], use_raw_gpt=False,
                orchestrator=orch,
            ))
        finally:
            patch.stopall()

        progress_texts = [r.get("content", "") for r in results if r.get("is_progress")]
        assert any("memory retrieval" in t for t in progress_texts), progress_texts
        assert any("gating, dedup" in t for t in progress_texts), progress_texts
        # Generation marker fires before streaming
        assert any("Generating response" in t for t in progress_texts), progress_texts
        # Bus torn down after the turn
        turn_progress.emit("late event")  # no active queue — must not raise
