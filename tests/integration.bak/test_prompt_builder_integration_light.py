"""
Lightweight integration tests for UnifiedPromptBuilder using stubbed dependencies.

Avoids heavy model/chroma/wiki by:
- Stubbing ModelManager and GateSystem
- Providing a minimal MemoryCoordinator stub
- Monkeypatching wiki fetch to a fast local stub
"""

import asyncio
import os
import types

import pytest


class StubModelManager:
    def __init__(self):
        self.api_models = {}

    def get_active_model_name(self):
        return "stub"

    def switch_model(self, *_args, **_kwargs):
        return None


class StubCorpus:
    def get_recent_memories(self, n=5):
        return [
            {"query": "Hello?", "response": "Hi!", "timestamp": "2024-01-01T00:00:00"},
            {"query": "How are you?", "response": "Good", "timestamp": "2024-01-01T00:01:00"},
        ][:n]


class StubMemoryCoordinator:
    def __init__(self):
        self.corpus_manager = StubCorpus()
        self.current_topic = "general"

    async def get_semantic_top_memories(self, query: str, limit: int = 10):
        return [
            {"content": "Semantic: something related", "relevance": 0.9},
            {"query": query, "response": "related answer", "relevance": 0.8},
        ][:limit]

    async def get_memories(self, query: str, limit: int = 10):
        return [{"query": query, "response": "fallback mem"}]

    async def get_recent_facts(self, limit: int = 3):
        return [{"content": "Recency fact", "confidence": 0.8}]

    async def get_facts(self, query: str, limit: int = 5):
        return [{"content": f"Fact about {query}", "confidence": 0.9}]

    def get_summaries(self, limit: int = 3):
        return [{"content": "Summary A"}, {"content": "Summary B"}][:limit]

    async def get_reflections_hybrid(self, query: str, limit: int = 3):
        return [{"content": "Reflection", "source": "recent"}]


class StubGateSystem:
    async def filter_memories(self, _query, orig):
        return list(orig)

    async def filter_wiki_content(self, _query, wiki_raw):
        return True, (wiki_raw or "Filtered wiki stub")

    async def filter_semantic_chunks(self, _query, orig):
        return list(orig)

    async def cosine_filter_summaries(self, _query, sums):
        return list(sums)


@pytest.mark.asyncio
async def test_build_prompt_with_stubs(monkeypatch):
    # Force fast paths
    os.environ["DREAMS_ENABLED"] = "0"

    # Monkeypatch wiki function used inside core.prompt
    import core.prompt as prompt_mod
    monkeypatch.setattr(prompt_mod, "get_wiki_snippet", lambda q: "Stub wiki snippet", raising=True)

    # Construct builder with stubs
    from core.prompt import UnifiedPromptBuilder

    builder = UnifiedPromptBuilder(
        model_manager=StubModelManager(),
        memory_coordinator=StubMemoryCoordinator(),
        gate_system=StubGateSystem(),
        tokenizer_manager=None,
        wiki_manager=None,
        topic_manager=None,
    )

    ctx = await builder.build_prompt(user_input="Test query")

    assert isinstance(ctx, dict)
    # Core sections present
    assert "memories" in ctx
    assert "facts" in ctx
    assert "summaries" in ctx
    assert "recent_conversations" in ctx
    # Wiki populated via stub
    assert ctx.get("wiki") == "Stub wiki snippet" or ctx.get("wiki_snippet") == "Stub wiki snippet"

