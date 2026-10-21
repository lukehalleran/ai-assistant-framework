"""Profile-fact embeds run off the event loop (BC-41, incident 2026-10-20).

LoopStall 268 ms at ``user_profile._embed_cached`` (bge encode of profile facts)
reached from the async ``get_user_profile_context``.  Drives the deployed async
caller with a real UserProfile-shaped host; the sync seam records its thread and
the SENSITIVE_OPEN context var must still be visible on the worker thread.
"""
import threading
from types import SimpleNamespace

import pytest

from core.prompt.gatherer_memory import MemoryRetrievalMixin
from memory.user_profile import SENSITIVE_OPEN


class _Profile:
    def __init__(self):
        self.thread = None
        self.sensitive_open = "unset"

    def get_context_injection(self, max_tokens, query, facts_per_category):
        self.thread = threading.get_ident()
        self.sensitive_open = SENSITIVE_OPEN.get()
        return "User: Rowan"


def _host(profile):
    g = MemoryRetrievalMixin.__new__(MemoryRetrievalMixin)
    g.memory_coordinator = SimpleNamespace()
    g.memory_id_map = {}
    g.user_profile = profile
    g._distress_active = True
    return g


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["what do you know about me", "  \n  what do you know about me\n "])
async def test_profile_injection_runs_off_loop_and_keeps_context(query):
    profile = _Profile()
    out = await _host(profile).get_user_profile_context(query)
    assert out == "User: Rowan"
    assert profile.thread is not None and profile.thread != threading.get_ident()
    assert profile.sensitive_open is True  # contextvar crossed the thread hop
