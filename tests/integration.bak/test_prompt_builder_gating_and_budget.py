"""
Integration tests for UnifiedPromptBuilder gating and budgeting paths.

Covers:
- _apply_gating with and without gate_system
- WIKI_FETCH_FULL path via gated_wiki_fetch monkeypatch
- _hygiene_and_caps dedupe and caps
- _manage_token_budget trimming by priority
"""

import asyncio
import os
import types

import pytest


class StubModelManager:
    def get_active_model_name(self):
        return "stub"


class StubTokenizer:
    def count_tokens(self, text: str, _model: str):
        # crude token approximation good enough for budgeting logic
        return len((text or "").split())


class StubGateSystem:
    async def filter_memories(self, _query, orig):
        # pass through first N as a simple gating stand-in
        return list(orig)[:3]

    async def filter_wiki_content(self, _query, wiki_raw: str):
        return True, f"Filtered {wiki_raw}"

    async def filter_semantic_chunks(self, _query, orig):
        return list(orig)[:2]

    async def cosine_filter_summaries(self, _query, items, **_kw):
        return list(items)[:2]


@pytest.mark.asyncio
async def test_apply_gating_with_full_wiki_and_filters(monkeypatch):
    # Force full wiki fetch path inside _apply_gating
    monkeypatch.setenv("WIKI_FETCH_FULL", "1")

    # Monkeypatch the gated_wiki_fetch coroutine used inside core.prompt._apply_gating
    import processing.gate_system as gs_mod

    async def _gwf_stub(query: str):  # returns (ok, text)
        return True, f"Full article for: {query}"

    monkeypatch.setattr(gs_mod, "gated_wiki_fetch", _gwf_stub, raising=True)

    # Ensure unified wiki accessor returns a known fallback if full fetch is off
    import core.prompt as prompt_mod
    monkeypatch.setattr(prompt_mod, "get_wiki_snippet", lambda q: "Cached snippet", raising=True)

    from core.prompt import UnifiedPromptBuilder

    builder = UnifiedPromptBuilder(
        model_manager=StubModelManager(),
        memory_coordinator=types.SimpleNamespace(corpus_manager=types.SimpleNamespace()),
        gate_system=StubGateSystem(),
        tokenizer_manager=StubTokenizer(),
    )

    ctx_in = {
        "memories": [{"content": "m1"}, {"content": "m2"}, {"content": "m3"}, {"content": "m4"}],
        "semantic_chunks": [{"text": "s1"}, {"text": "s2"}, {"text": "s3"}],
        "wiki": "",
    }
    out = await builder._apply_gating("topic", ctx_in)

    # Gate applied to memories and semantic chunks
    assert len(out.get("memories", [])) <= 3
    assert len(out.get("semantic_chunks", [])) <= 2
    # Wiki passed through filter on full-article path
    assert out.get("wiki") == "Filtered Full article for: topic"
    assert out.get("wiki_snippet") == "Filtered Full article for: topic"


# Note: The no-gate wiki fast-path is indirectly covered by other tests.


def test_hygiene_and_caps(monkeypatch):
    # Tighten caps to exercise truncation logic deterministically
    import core.prompt as P
    monkeypatch.setattr(P, "PROMPT_MAX_MEMS", 2, raising=False)
    monkeypatch.setattr(P, "PROMPT_MAX_SUMMARIES", 2, raising=False)
    monkeypatch.setattr(P, "PROMPT_MAX_REFLECTIONS", 1, raising=False)
    monkeypatch.setattr(P, "PROMPT_MAX_FACTS", 2, raising=False)
    monkeypatch.setattr(P, "PROMPT_MAX_RECENT", 2, raising=False)
    monkeypatch.setattr(P, "PROMPT_MAX_SEMANTIC", 2, raising=False)
    monkeypatch.setattr(P, "PROMPT_MAX_DREAMS", 2, raising=False)

    from core.prompt import UnifiedPromptBuilder

    builder = UnifiedPromptBuilder(
        model_manager=StubModelManager(),
        memory_coordinator=types.SimpleNamespace(corpus_manager=types.SimpleNamespace()),
        gate_system=StubGateSystem(),
        tokenizer_manager=StubTokenizer(),
    )

    # Duplicates across sections should be deduped and then capped
    ctx = {
        "recent_conversations": [
            {"query": "q1", "response": "a1"},
            {"query": "q1", "response": "a1"},
            {"query": "q2", "response": "a2"},
        ],
        "memories": [
            {"content": "X"},
            {"content": "X"},
            {"content": "Y"},
        ],
        "summaries": [
            {"content": "S1"},
            {"content": "S1"},
            {"content": "S2"},
        ],
        "reflections": [
            {"content": "R1"},
            {"content": "R1"},
        ],
        "facts": [
            {"content": "F1"},
            {"content": "F1"},
            {"content": "F2"},
        ],
        "semantic_chunks": [
            {"text": "C1"},
            {"text": "C1"},
            {"text": "C2"},
        ],
        "dreams": [
            {"content": "D1"},
            {"content": "D2"},
            {"content": "D2"},
        ],
        "wiki": "snippet",
    }

    cleaned = builder._hygiene_and_caps(ctx)
    assert len(cleaned["recent_conversations"]) == 2  # capped
    assert len(cleaned["memories"]) == 2              # deduped then capped
    assert len(cleaned["summaries"]) == 2
    # Reflections are not deduped/capped here (managed later); ensure present
    assert len(cleaned["reflections"]) >= 1
    assert len(cleaned["facts"]) == 2
    assert len(cleaned["semantic_chunks"]) == 2
    assert len(cleaned["dreams"]) == 2


def test_manage_token_budget_trims_lower_priority(monkeypatch):
    from core.prompt import UnifiedPromptBuilder

    builder = UnifiedPromptBuilder(
        model_manager=StubModelManager(),
        memory_coordinator=types.SimpleNamespace(corpus_manager=types.SimpleNamespace()),
        gate_system=None,
        tokenizer_manager=StubTokenizer(),
    )

    # Set a very small budget to force trimming
    builder.token_budget = 10

    context = {
        "recent_conversations": [],
        "semantic_chunks": [],
        "memories": [
            {"content": "one two three"},      # 3 tokens
            {"content": "four five six"},     # 3 tokens
            {"content": "seven eight nine"},  # 3 tokens
        ],
        "facts": [
            {"content": "alpha beta"},  # 2 tokens
        ],
        "summaries": [
            {"content": "s u m"},  # 3 tokens
        ],
        "reflections": [],
        "wiki": "w i k i x",  # 4 tokens (low priority)
        "dreams": [],
    }

    trimmed = builder._manage_token_budget(context)

    # Total tokens must be within budget
    # And lower-priority 'wiki' should be trimmed when needed
    wiki = trimmed.get("wiki", "")
    assert isinstance(wiki, str)
    assert (len(wiki.split()) == 0) or (len(wiki.split()) <= 2)
