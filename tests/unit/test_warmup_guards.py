"""
Guard smoke test for gui.launch._run_model_warmup.

Every warmup step must be independently guarded — warmup can never take down
startup, even with a completely broken orchestrator. Extended 2026-08-02 with
tone/need exemplar warm and the end-to-end memory-retrieval warm pass (turn-1
memories task ran 8-17s cold vs 3-5s warm; exemplars were computed inside the
user's first message).
"""

from unittest.mock import MagicMock, patch


class _InlineThread:
    """Run the warmup body synchronously so the test observes any exception."""

    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


class TestWarmupGuards:
    def test_broken_orchestrator_never_raises(self):
        from gui.launch import _run_model_warmup

        orch = MagicMock()
        # Force every attribute access down the failure path
        orch.memory_system = None
        orch.model_manager = None
        with patch("threading.Thread", _InlineThread):
            _run_model_warmup(orch)  # must not raise

    def test_retrieval_warm_uses_memory_system(self):
        from gui.launch import _run_model_warmup

        async def _fake_get_memories(query, limit=3):
            _fake_get_memories.called = (query, limit)
            return []

        orch = MagicMock()
        orch.memory_system.get_memories = _fake_get_memories
        with patch("threading.Thread", _InlineThread):
            _run_model_warmup(orch)
        assert getattr(_fake_get_memories, "called", None) is not None
