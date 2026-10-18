"""get_visual_memories loads the visual store off the event loop.

Incident 2026-10-17: `[LoopStall] event loop blocked 2980 ms` at
visual_memory_store.load (faiss.read_index) via get_visual_entity_ids. Class: BC-41.
"""
import threading
from types import SimpleNamespace

import pytest

from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin


class _G(KnowledgeRetrievalMixin):
    def __init__(self):
        self.memory_coordinator = None
        self.memory_id_map = {}


@pytest.mark.asyncio
async def test_visual_entity_ids_read_runs_off_loop(monkeypatch):
    monkeypatch.setattr("config.app_config.VISUAL_MEMORY_ENABLED", True)
    seen = {}

    def fake_ids():
        seen["t"] = threading.get_ident()
        return set()

    g = _G()
    g._visual_retriever = SimpleNamespace(_store=SimpleNamespace(get_visual_entity_ids=fake_ids))
    loop_thread = threading.get_ident()
    out = await g.get_visual_memories("show me the photos of the cat")
    assert out == {"text_results": [], "images": []}
    assert seen["t"] != loop_thread
