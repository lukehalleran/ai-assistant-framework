"""
Integration tests for UnifiedPromptBuilder summaries and reflections helpers.

Covers:
- _get_summaries with FORCE (LLM) and micro fallback
- _get_reflections_hybrid_filtered path with gate_system present
"""

import asyncio
import types
import pytest


class AsyncModelManager:
    def __init__(self, text: str = "LLM forced summary"):
        # Expose an alias so _llm_summarize_recent will proceed
        self.api_models = {"gpt-4o-mini": object()}
        self._text = text
        self._active = "gpt-4o-mini"

    def get_active_model_name(self):
        return self._active

    def switch_model(self, name: str):
        self._active = name

    async def generate_once(self, _prompt: str, max_tokens: int = 160):
        # Immediate return for speed; respects await in code
        return self._text


class EmptyLLMManager(AsyncModelManager):
    async def generate_once(self, _prompt: str, max_tokens: int = 160):
        return ""


class StubCorpus:
    def __init__(self, summaries=None, recents=None):
        self._sums = summaries or []
        self._recents = recents or [
            {"query": "Q1", "response": "A1"},
            {"query": "Q2", "response": "A2"},
        ]

    def get_summaries(self, n=5):
        return self._sums[:n]

    def add_summary(self, _):
        # no-op for tests
        pass

    def get_recent_memories(self, n=10):
        return self._recents[:n]


class StubMemoryCoordinator:
    def __init__(self, summaries=None, recents=None):
        self.corpus_manager = StubCorpus(summaries=summaries, recents=recents)

    async def get_reflections_hybrid(self, query: str, limit: int = 3):
        return [
            {"content": "R about cats", "score": 0.9},
            {"content": "R about dogs", "score": 0.4},
            {"content": "R general", "score": 0.2},
        ][:limit]


class StubGateSystem:
    async def cosine_filter_summaries(self, query: str, items, threshold=None, source_type=None):
        # keep only items that mention query token if provided
        q = (query or "").lower()
        if not q:
            return list(items)
        out = []
        for it in items:
            txt = (it.get("content") if isinstance(it, dict) else str(it)).lower()
            if any(tok in txt for tok in q.split()):
                out.append(it)
        return out or list(items)[:1]


@pytest.mark.asyncio
async def test_get_summaries_forced_then_stored_filtering():
    from core.prompt import UnifiedPromptBuilder

    # Include a placeholder-like stored summary that should be filtered out
    stored = [
        {"content": "Q: prior chat dump"},
        {"content": "A real stored summary"},
    ]
    builder = UnifiedPromptBuilder(
        model_manager=AsyncModelManager("LLM forced summary"),
        memory_coordinator=StubMemoryCoordinator(summaries=stored),
        gate_system=StubGateSystem(),
        tokenizer_manager=types.SimpleNamespace(count_tokens=lambda s, m: len((s or "").split())),
    )

    # Force LLM path for this call
    builder.force_llm_summaries = True
    out = await builder._get_summaries(count=3)

    assert isinstance(out, list) and len(out) >= 1
    # First item should be from LLM forced path
    assert out[0]["source"].startswith("llm")
    assert "LLM forced summary" in out[0]["content"]
    # Stored placeholder filtered out; real stored may appear
    assert all("q:" not in (o["content"].lower()) for o in out)


@pytest.mark.asyncio
async def test_get_summaries_llm_fallback_when_no_stored():
    from core.prompt import UnifiedPromptBuilder

    builder = UnifiedPromptBuilder(
        model_manager=AsyncModelManager("LLM fallback summary"),
        memory_coordinator=StubMemoryCoordinator(summaries=[]),
        gate_system=None,
        tokenizer_manager=types.SimpleNamespace(count_tokens=lambda s, m: len((s or "").split())),
    )

    # Not forced; should take LLM fallback path since no stored summaries
    out = await builder._get_summaries(count=1)
    assert isinstance(out, list) and len(out) == 1
    assert out[0]["source"].startswith("llm_fallback")


@pytest.mark.asyncio
async def test_get_summaries_micro_fallback_when_llm_empty():
    from core.prompt import UnifiedPromptBuilder

    builder = UnifiedPromptBuilder(
        model_manager=EmptyLLMManager(),
        memory_coordinator=StubMemoryCoordinator(summaries=[]),
        gate_system=None,
        tokenizer_manager=types.SimpleNamespace(count_tokens=lambda s, m: len((s or "").split())),
    )

    out = await builder._get_summaries(count=1)
    assert isinstance(out, list) and len(out) == 1
    assert out[0]["source"].startswith("fallback_micro")


@pytest.mark.asyncio
async def test_reflections_hybrid_filtered():
    from core.prompt import UnifiedPromptBuilder

    builder = UnifiedPromptBuilder(
        model_manager=AsyncModelManager(),
        memory_coordinator=StubMemoryCoordinator(),
        gate_system=StubGateSystem(),
        tokenizer_manager=types.SimpleNamespace(count_tokens=lambda s, m: len((s or "").split())),
    )

    out = await builder._get_reflections_hybrid_filtered("cats topic", count=3)
    assert isinstance(out, list)
    # Expect at least one item mentioning "cats" due to filter
    assert any("cats" in (i.get("content", "").lower()) for i in out)
