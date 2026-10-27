"""2026-10-26 (lane R2 referee follow-up): the facts-collection half of the
sensitive-fact gate was inert because nothing set the open/closed turn state around
`get_facts`. The two LIVE routes by which a facts-collection record reaches the
model are driven here through their deployed call sites:
  (a) ContextPipeline.build -> _check_heavy_topics -> memory_system.get_facts
      (heavy-topic fresh facts, rendered by the builder as fresh_facts);
  (b) the agentic search_memory tool, ToolExecutor._execute_memory_search("facts").
plus the proactive-insight edge render (ContextSurfacer cluster sentences), which is
always a closed turn. Insight-mode sweep (core/insight/sweep.py) is deliberately
ungated: an explicit insight request is an open turn by definition.

Incident date: 2026-10-10 (gate in profile only). class: BC-46, BC-58.
Fixture facts are invented (SYNTH_*); closed turn -> withheld, open turn
(elevated tone / history ask) -> admitted.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from core.agentic.formatters import AgenticFormatter
from core.agentic.tools import ToolExecutor
from core.context_pipeline import ContextPipeline, ToneLevel
from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge, GraphNode
from memory.memory_retriever import MemoryRetriever

from tests.unit.test_memory_retriever_sensitive_fact_parity import (  # noqa: F401  (fixtures + fakes)
    HISTORY, HURT, LOANS, VALUE, _FakeStore, _fact_rows, fake_embedder,
)

LONG_LOANS = LOANS + " and also compare the interest rates across the different servicers I have"
LONG_HISTORY = HISTORY + " please go through everything you have stored about me from the past year"


def _wrapped(text):
    return "\n".join("   " + w for w in text.split(". "))


# ---- (a) heavy-topic fresh facts through ContextPipeline.build --------------

class _MemorySystem:
    def __init__(self):
        self._retr = MemoryRetriever(corpus_manager=MagicMock(), chroma_store=_FakeStore(_fact_rows()))

    async def get_facts(self, query, limit=8):
        return await self._retr.get_facts(query, limit=limit)


def _pipeline(monkeypatch, tone):
    with patch.object(ContextPipeline, "_load_persisted_tone", return_value=None):
        cp = ContextPipeline(model_manager=None, topic_manager=None, memory_system=_MemorySystem(),
                             config={"enable_query_rewrite": False})

    async def topics(user_input, last_exchange=None):
        return "general", ["general"]

    async def detect_tone(user_input, history):
        return tone, None

    async def thread():
        return None

    monkeypatch.setattr(cp, "_extract_topics", topics)
    monkeypatch.setattr(cp, "_detect_tone", detect_tone)
    monkeypatch.setattr(cp, "_get_identity_context", lambda: ("", None))
    monkeypatch.setattr(cp, "_get_thread_context", thread)
    monkeypatch.setattr(cp, "_get_tone_instructions", lambda level: "")
    monkeypatch.setattr(cp, "_intent_classifier", None)
    monkeypatch.setattr(cp, "_should_run_stm", lambda history: False)
    heavy = SimpleNamespace(is_heavy_topic=True, is_question=False, is_command=False, token_count=30)
    monkeypatch.setattr("utils.query_checker.analyze_query", lambda q, mm=None: heavy)

    async def heavy_async(q, mm=None):
        return heavy

    monkeypatch.setattr("utils.query_checker.analyze_query_async", heavy_async)
    return cp


def _fresh_facts(cp, query):
    result = asyncio.run(cp.build(query))
    return " ".join(r["content"] for r in result.extracted_facts)


@pytest.mark.parametrize("wrap", [lambda t: t, _wrapped], ids=["clean", "wrapped"])
class TestHeavyTopicFreshFacts:
    def test_closed_turn_withholds_sensitive_fact(self, monkeypatch, wrap):
        text = _fresh_facts(_pipeline(monkeypatch, ToneLevel.CONVERSATIONAL), wrap(LONG_LOANS))
        assert VALUE not in text and "loan_servicer" in text

    def test_elevated_tone_admits_it(self, monkeypatch, wrap):
        text = _fresh_facts(_pipeline(monkeypatch, ToneLevel.CONCERN), wrap(LONG_LOANS))
        assert VALUE in text

    def test_history_ask_admits_it(self, monkeypatch, wrap):
        text = _fresh_facts(_pipeline(monkeypatch, ToneLevel.CONVERSATIONAL), wrap(LONG_HISTORY))
        assert VALUE in text


# ---- (b) agentic search_memory(facts) ---------------------------------------

def _executor(user_text):
    ex = ToolExecutor.__new__(ToolExecutor)
    store = MagicMock()
    store.query_collection = lambda **kw: _fact_rows()
    ex.chroma_store = store
    ex.formatter = AgenticFormatter()
    ex._current_wiki_source_map = {}
    ex._fetch_prov = {"current": user_text, "user_texts": [], "session": None}
    return ex


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [lambda t: t, _wrapped], ids=["clean", "wrapped"])
class TestAgenticMemorySearchFacts:
    async def test_closed_turn_withholds(self, wrap):
        out = await _executor(wrap(LOANS))._execute_memory_search("what do I owe", "facts")
        assert VALUE not in out and "loan_servicer" in out

    async def test_history_ask_admits(self, wrap):
        out = await _executor(wrap(HISTORY))._execute_memory_search("anything", "facts")
        assert VALUE in out

    async def test_topical_user_text_admits(self, wrap):
        out = await _executor(wrap(HURT))._execute_memory_search("anything", "facts")
        assert VALUE in out

    async def test_model_authored_query_cannot_launder(self, wrap):
        out = await _executor(wrap(LOANS))._execute_memory_search("self harm history", "facts")
        assert VALUE not in out

    async def test_other_collections_untouched(self, wrap):
        ex = _executor(wrap(LOANS))
        ex.chroma_store.query_collection = lambda **kw: [
            {"id": "x", "content": f"user | self_harm | {VALUE}", "metadata": {}, "relevance_score": 0.5}]
        assert VALUE in await ex._execute_memory_search("q", "reflections")


# ---- proactive-insight edge render ------------------------------------------

def test_proactive_insight_cluster_sentences_never_carry_sensitive_edge(tmp_path):
    from memory.context_surfacer import ContextSurfacer

    gm = GraphMemory(persist_path=str(tmp_path / "g.json"))
    for nid in ("user", "kiln", VALUE):
        gm.add_entity(GraphNode(entity_id=nid, display_name=nid, entity_type="other"))
    now = datetime.now()
    for rel, tgt in (("works_on", "kiln"), ("self_harm", VALUE)):
        gm.add_relation(GraphEdge(source_id="user", relation=rel, target_id=tgt, weight=1.0,
                                  first_seen=now, last_seen=now))
    cs = ContextSurfacer(gm, None, None)
    clusters = cs._classify_user_edges()
    sentences = " ".join(s for c in clusters.values() for s in c.fact_sentences).lower()
    assert "kiln" in sentences
    assert VALUE not in sentences and "self harm" not in sentences
