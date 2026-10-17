"""B4 items 4+5 acceptance: a user's own fact survives pastes and a recall
question ("where do I work?") gets it as USER-AUTHORED evidence through the
deployed gatherer -> builder -> token budget -> formatter (BC-46, BC-75).

Original batch label: oct12_b4 (renamed 2026-10-16; old path in tests/RENAMES.tsv)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.prompt.formatter import PromptFormatter
from core.prompt.gatherer_memory import (
    MemoryRetrievalMixin, is_user_fact_question, select_user_stated_candidates,
)
from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits

FACT = "I just started a new job at Northwind as a data analyst"
QUERY = "where do I work?"
PASTE = "lorem ipsum dolor " * 200


class _Model:
    """Deterministic stand-in for the store's bge embedder: the fact turn is
    near the question, everything else is orthogonal (real-model cosines are
    in the handoff MEASUREMENTS)."""
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        out = []
        for t in texts:
            if t.startswith("where do I"):
                out.append([1.0, 0.0])
            elif "Northwind" in t:
                out.append([0.6, 0.8])
            else:
                out.append([0.0, 1.0])
        return np.array(out, dtype=float)


class _Corpus:
    def __init__(self, newest_first):
        self._items = newest_first

    def get_recent_memories(self, count=3):
        return list(self._items[:count])


def _turn(i, q):
    return {"query": q, "response": f"assistant reply {i}", "timestamp": f"2026-10-11T22:{59 - i:02d}:00"}


def _gatherer(items):
    g = MemoryRetrievalMixin.__new__(MemoryRetrievalMixin)
    g.memory_coordinator = SimpleNamespace(
        corpus_manager=_Corpus(items), chroma_store=SimpleNamespace(get_st_model=lambda: _Model()))
    g.memory_id_map = {}
    g.user_profile = None
    return g


def test_structural_user_fact_question():
    assert is_user_fact_question("where do I work?")
    assert is_user_fact_question("when is my dentist appointment?")
    assert not is_user_fact_question("what is the capital of France?")
    assert not is_user_fact_question("I work at Northwind")


def test_candidates_are_user_statements_only():
    entries = [_turn(0, "where do I work?"), _turn(1, "what can you do?"), _turn(2, PASTE),
               _turn(3, "ok"), _turn(4, FACT), _turn(5, FACT)]
    got = select_user_stated_candidates(entries, QUERY)
    assert [t for t, _ in got] == [FACT]


@pytest.mark.asyncio
async def test_non_recall_intent_returns_nothing():
    g = _gatherer([_turn(0, FACT)])
    assert await g.get_user_stated_evidence(QUERY, "general") == []
    assert await g.get_user_stated_evidence("tell me about Rome", "factual_recall") == []


@pytest.mark.asyncio
async def test_fact_outranks_noise_and_is_user_authored():
    items = [_turn(0, "random chatter here today"), _turn(1, FACT), _turn(2, "more chatter about nothing")]
    got = await _gatherer(items).get_user_stated_evidence(QUERY, "factual_recall")
    assert got[0]["query"] == FACT and got[0]["similarity"] > got[-1]["similarity"] - 1e-9


@pytest.mark.asyncio
async def test_replay_fact_pastes_recall_reaches_prompt_as_users_words(monkeypatch):
    # newest first: 6 chatter turns (push the fact out of a 2-turn window), then the fact
    chatter = [_turn(i, f"chatter number {i} about the weekend") for i in range(6)]
    items = chatter + [_turn(10, FACT)]
    g = _gatherer(items)
    builder = full_builder(monkeypatch, [], budget=4000)
    builder.context_gatherer._get_recent_conversations = g._get_recent_conversations
    builder.context_gatherer.get_user_stated_evidence = g.get_user_stated_evidence
    limits = retrieval_limits()
    result = await builder.build_prompt(QUERY, retrieval_overrides=limits, intent_type="factual_recall")
    stated = result["user_stated_evidence"]
    assert [u["query"] for u in stated][:1] == [FACT]

    fmt = PromptFormatter(token_manager=MagicMock(count_tokens=MagicMock(return_value=10)), time_manager=None)
    fmt._feature_inventory_cache = None
    prompt = fmt._assemble_prompt(result, QUERY)
    assert "[THE USER'S OWN STATEMENTS" in prompt
    assert f'User said: "{FACT}"' in prompt
    assert "never describe them as your guess" in prompt
    if "[RELEVANT MEMORIES]" in prompt:
        assert prompt.index("[THE USER'S OWN STATEMENTS") < prompt.index("[RELEVANT MEMORIES]")


@pytest.mark.asyncio
async def test_other_intents_never_run_the_channel(monkeypatch):
    g = _gatherer([_turn(1, FACT)])
    builder = full_builder(monkeypatch, [], budget=4000)
    builder.context_gatherer._get_recent_conversations = g._get_recent_conversations
    called = []

    async def _spy(*a, **k):
        called.append(1)
        return []
    builder.context_gatherer.get_user_stated_evidence = _spy
    await builder.build_prompt(QUERY, retrieval_overrides=retrieval_limits(), intent_type="general")
    assert called == []


@pytest.mark.asyncio
async def test_failure_is_a_typed_outcome_not_a_silent_empty():
    from utils.retrieval_outcome import outcome_status

    class _Boom:
        def get_recent_memories(self, count=3):
            raise RuntimeError("store down")

    g = _gatherer([])
    g.memory_coordinator.corpus_manager = _Boom()
    got = await g.get_user_stated_evidence(QUERY, "factual_recall")
    assert got == [] and outcome_status(got) == ("failed", "RuntimeError")
