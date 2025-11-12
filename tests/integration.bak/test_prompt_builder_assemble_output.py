"""
Integration tests targeting prompt assembly and helper branches.

Covers:
- _assemble_prompt with rich context (timestamps, sections)
- _get_recent_conversations topic-filtered ordering
- _middle_out compression helper
- _strip_prompt_artifacts cleanup
"""

import asyncio
from datetime import datetime
import types
import pytest


class StubModelManager:
    def get_active_model_name(self):
        return "stub"


@pytest.mark.asyncio
async def test_assemble_prompt_sections_rendered(monkeypatch):
    from core.prompt import UnifiedPromptBuilder

    builder = UnifiedPromptBuilder(
        model_manager=StubModelManager(),
        memory_coordinator=types.SimpleNamespace(corpus_manager=types.SimpleNamespace()),
        gate_system=None,
        tokenizer_manager=types.SimpleNamespace(count_tokens=lambda s, m: len((s or "").split())),
    )

    ctx = {
        "time_context": "UTC now",
        "fresh_facts": [
            {"content": "F1", "metadata": {"source": "msg"}},
        ],
        "recent_conversations": [
            {"query": "Hi", "response": "Hello", "timestamp": datetime.now()},
            {"query": "When?", "response": "Now", "timestamp": datetime.now().isoformat()},
        ],
        "semantic_chunks": [
            {"text": "Chunk about cats"},
            {"content": "Chunk about dogs"},
        ],
        "memories": [
            {"query": "Q1", "response": "A1", "timestamp": datetime.now().isoformat()},
            {"content": "Note memory", "timestamp": datetime.now()},
        ],
        "facts": [{"content": "fact one"}],
        "summaries": [{"content": "sum one", "timestamp": datetime.now()}],
        "reflections": [{"content": "reflect"}],
        "dreams": [{"content": "dream A"}],
        "wiki": "Background knowledge here",
        "raw_user_input": "Tell me things",
    }

    out = builder._assemble_prompt(
        user_input="Tell me things",
        context=ctx,
        system_prompt="ignored",
        directives_file="",
    )
    assert isinstance(out, str) and "[TIME CONTEXT]" in out and "[BACKGROUND KNOWLEDGE]" in out


@pytest.mark.asyncio
async def test_get_recent_conversations_topic_filter(monkeypatch):
    from core.prompt import UnifiedPromptBuilder

    corpus = types.SimpleNamespace(
        get_recent_memories=lambda n: [
            {"query": "q1", "response": "a1", "tags": ["topic:general"]},
            {"query": "q2", "response": "a2", "tags": ["topic:tech"]},
            {"query": "q3", "response": "a3", "tags": []},
        ]
    )

    builder = UnifiedPromptBuilder(
        model_manager=StubModelManager(),
        memory_coordinator=types.SimpleNamespace(corpus_manager=corpus, current_topic="tech"),
        gate_system=None,
        tokenizer_manager=types.SimpleNamespace(count_tokens=lambda s, m: len((s or "").split())),
    )

    out = await builder._get_recent_conversations(2, topic="tech")
    assert len(out) == 2
    # First should be the on-topic one
    assert out[0]["response"] == "a2"


def test_middle_out_and_strip_artifacts(monkeypatch):
    from core.prompt import UnifiedPromptBuilder, _strip_prompt_artifacts

    builder = UnifiedPromptBuilder(
        model_manager=StubModelManager(),
        memory_coordinator=types.SimpleNamespace(corpus_manager=types.SimpleNamespace()),
        gate_system=None,
        tokenizer_manager=types.SimpleNamespace(count_tokens=lambda s, m: len((s or "").split())),
    )

    # Force middle-out by simulating high prompt usage
    builder._prompt_token_usage = 50000
    long = " ".join(["word"] * 200)
    compressed = builder._middle_out(long, max_tokens=20, force=False)
    assert len(compressed.split()) <= 40  # head+tail approx

    noisy = """
    [RECENT CONVERSATIONS]
    This should be removed

    Normal text should remain.
    """.strip()
    cleaned = _strip_prompt_artifacts(noisy)
    assert "RECENT CONVERSATIONS" not in cleaned
    assert "Normal text" in cleaned
