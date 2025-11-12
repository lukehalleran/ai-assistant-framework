"""
Internal-flow tests for MemoryCoordinator that avoid heavy dependencies.

Covers:
- _combine_memories bypass of top-N recents and gate application
- _rank_memories scoring path on simple candidates
- run_shutdown_reflection end-to-end with stub LLM and store
"""

import asyncio
import types
import pytest

from memory.corpus_manager import CorpusManager
from memory.memory_coordinator import MemoryCoordinator


class FakeStore:
    def __init__(self):
        self.collections = {}

    def create_collection(self, name: str):
        self.collections[name] = types.SimpleNamespace()
        return self.collections[name]

    def add_to_collection(self, name: str, text: str, md: dict):
        # no-op, but ensures method exists
        self.collections.setdefault(name, types.SimpleNamespace())
        return "ok"


class StubGate:
    async def filter_memories(self, _q, chunks):
        # return first two chunks as gated winners
        return list(chunks)[:2]


class StubLLM:
    def __init__(self, text: str = "Reflection text"):
        self._text = text

    def get_active_model_name(self):
        return "stub"

    def switch_model(self, *_):
        return None

    async def generate_once(self, *_args, **_kwargs):
        return self._text


@pytest.mark.asyncio
async def test_combine_and_rank_paths(tmp_path):
    cm = CorpusManager(corpus_file=str(tmp_path / "c.json"))
    mc = MemoryCoordinator(corpus_manager=cm, chroma_store=FakeStore(), gate_system=StubGate())

    very_recent = [
        {"id": "r1", "content": "recent 1"},
        {"id": "r2", "content": "recent 2"},
        {"id": "r3", "content": "recent 3"},
    ]
    semantic = [
        {"id": "s1", "content": "sem 1", "relevance_score": 0.8},
        {"id": "s2", "content": "sem 2", "relevance_score": 0.2},
    ]
    out = await mc._combine_memories(
        very_recent=very_recent,
        semantic=semantic,
        hierarchical=[],
        query="q",
        config={"max_memories": 5},
    )
    # Should include at least one bypassed recent and some gated entries
    assert len(out) >= 1
    assert any(m.get("gated") in (True, False) for m in out)

    ranked = mc._rank_memories(out, "q")
    assert isinstance(ranked, list) and len(ranked) == len(out)


@pytest.mark.asyncio
async def test_run_shutdown_reflection(tmp_path):
    cm = CorpusManager(corpus_file=str(tmp_path / "c.json"))
    mc = MemoryCoordinator(corpus_manager=cm, chroma_store=FakeStore(), gate_system=None, model_manager=StubLLM())

    # Provide enough exchanges to satisfy minimum requirement
    conv = [
        {"query": f"q{i}", "response": f"a{i}"} for i in range(6)
    ]
    ok = await mc.run_shutdown_reflection(session_conversations=conv, session_summaries=None)
    assert ok is True
