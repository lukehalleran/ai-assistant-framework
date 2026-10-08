"""BC-10: main._gather_session_state read ``conversation_logger.buffer`` -- an
attribute ConversationLogger has never defined -- so session_convos was always
[] by accident. The dead read is gone; the contract is pinned here."""
from types import SimpleNamespace

import main


class _ExplodingLogger:
    def __getattr__(self, name):
        raise AssertionError(f"conversation_logger.{name} must not be read")


def test_gather_session_state_does_not_touch_conversation_logger():
    orch = SimpleNamespace(
        conversation_logger=_ExplodingLogger(),
        prompt_builder=SimpleNamespace(_last_summaries=["s1", "s2"]),
    )
    assert main._gather_session_state(orch) == ([], ["s1", "s2"])


def test_gather_session_state_without_prompt_builder_summaries():
    orch = SimpleNamespace(conversation_logger=_ExplodingLogger(),
                           prompt_builder=SimpleNamespace(_last_summaries=None))
    assert main._gather_session_state(orch) == ([], [])
