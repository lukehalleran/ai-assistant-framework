"""2026-10-10 gated sensitive facts in [USER PROFILE] only; 2026-10-26 (lane R2)
the same fact still reached the prompt through the `facts` collection
(MemoryRetriever.get_facts/get_recent_facts) and through graph edges
(get_graph_context -> GraphMemory.get_context_sentences). One shared gate
(`memory_retriever.gate_sensitive_records`, delegating to the profile's own
`hold_back_sensitive_facts`) now serves all three; this file pins PARITY: the same
sensitive fact is admitted/refused identically by profile, facts and graph.

Incident date: 2026-10-10 (BACKLOG item 1 of the oct26b R2 brief). class: BC-46, BC-58.

Every path is the DEPLOYED one: MemoryRetrievalMixin.get_user_profile_context over a
real UserProfile, a real MemoryRetriever over a fake Chroma store, and the real
MemoryKnowledgeMixin.get_graph_context over a real GraphMemory. Fixture values are
invented (SYNTH_*); the embedder is a deterministic concept fake.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

import memory.user_profile as up
from core.prompt.gatherer_knowledge import KnowledgeRetrievalMixin
from core.prompt.gatherer_memory import MemoryRetrievalMixin
from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge, GraphNode
from memory.memory_retriever import MemoryRetriever
from memory.user_profile import SENSITIVE_OPEN, UserProfile

LOANS = "can you help me figure out my student loans repayment plan"
HURT = "I keep thinking about how I hurt myself back then"
HISTORY = "what do you know about me?"
VALUE = "synth_crisis_disclosure_2019"

_CONCEPTS = [
    {"loan", "loans", "student", "repayment", "plan", "payment", "debt", "balance", "servicer"},
    {"self", "harm", "hurt", "myself", "suicide", "crisis", "disclosure"},
    {"gym", "squat", "bench"},
]


class FakeEmbedder:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kw):
        out = []
        for t in texts:
            words = {w.strip(".,:;'\"?!").lower() for w in t.replace("_", " ").split()}
            v = np.array([float(len(words & c)) for c in _CONCEPTS] + [0.01])
            out.append(v / np.linalg.norm(v))
        return np.array(out)


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    up._fact_emb_cache.clear()
    up._query_emb_cache.clear()
    monkeypatch.setattr(up, "_profile_embedder", lambda: FakeEmbedder())
    yield
    up._fact_emb_cache.clear()
    up._query_emb_cache.clear()


# ---- the three deployed read paths ---------------------------------------

async def _via_profile(tmp_path, query, distress):
    p = UserProfile(str(tmp_path / "profile.json"))
    base = datetime.now() - timedelta(hours=2)  # clock-safe: only relative offsets order the fixture facts; no date assertion
    assert p.add_fact("loan_servicer", "synthetic servicer", 0.9, "fixture", timestamp=base)
    assert p.add_fact("self_harm", VALUE, 0.9, "fixture", timestamp=base + timedelta(minutes=2))
    g = MemoryRetrievalMixin.__new__(MemoryRetrievalMixin)
    g.memory_id_map = {}
    g.user_profile = p
    g._distress_active = distress
    out = await g.get_user_profile_context(query, max_tokens=1000)
    return VALUE in out, "loan_servicer" in out


class _FakeColl:
    def __init__(self, n):
        self._n = n

    def count(self):
        return self._n


class _FakeStore:
    def __init__(self, rows):
        self._rows = rows
        self.collections = {"facts": _FakeColl(len(rows))}

    def query_collection(self, name, query_text, n_results):
        return self._rows

    def get_recent(self, name, limit=5):
        return self._rows[:limit]


def _fact_rows():
    ts = datetime.now().isoformat()
    meta = {"timestamp": ts, "confidence": 0.8}
    return [
        {"id": "a", "content": f"user | self_harm | {VALUE}", "metadata": dict(meta), "relevance_score": 0.8},
        {"id": "b", "content": "user | loan_servicer | synthetic servicer", "metadata": dict(meta),
         "relevance_score": 0.8},
    ]


def _open_expr(query, distress):
    from core.prompt.gatherer_memory import is_profile_history_ask
    return bool(distress) or is_profile_history_ask(query)


async def _via_facts(query, distress, recent=False):
    retr = MemoryRetriever(corpus_manager=MagicMock(), chroma_store=_FakeStore(_fact_rows()))
    token = SENSITIVE_OPEN.set(_open_expr(query, distress))  # what the gatherer sets per turn
    try:
        rows = await (retr.get_recent_facts(10) if recent else retr.get_facts(query, limit=10))
    finally:
        SENSITIVE_OPEN.reset(token)
    text = " ".join(r["content"] for r in rows)
    return VALUE in text, "loan_servicer" in text


async def _via_graph(tmp_path, query, distress, monkeypatch):
    gm = GraphMemory(persist_path=str(tmp_path / "g.json"))
    for nid, name in (("user", "User"), (VALUE, "Disclosure"), ("servicer", "Servicer")):
        gm.add_entity(GraphNode(entity_id=nid, display_name=name, entity_type="other"))
    now = datetime.now()
    for rel, tgt in (("self_harm", VALUE), ("uses_loan_servicer", "servicer")):
        gm.add_relation(GraphEdge(source_id="user", relation=rel, target_id=tgt, weight=1.0,
                                  first_seen=now, last_seen=now))
    import core.prompt.gatherer_knowledge as mod
    monkeypatch.setattr(mod.graph_utils, "extract_graph_entities", lambda q, r, graph_memory=None: {"user"})
    monkeypatch.setattr("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True)
    host = KnowledgeRetrievalMixin.__new__(KnowledgeRetrievalMixin)
    host.memory_id_map = {}
    host._distress_active = distress
    host.memory_coordinator = type("MC", (), {"graph_memory": gm, "entity_resolver": object()})()
    out = await host.get_graph_context(query, max_sentences=10)
    text = " ".join(out).lower()
    return "disclosure" in text or "self harm" in text, "servicer" in text


CASES = [
    # (label, query, distress, sensitive fact admitted?)
    ("unrelated_turn_refused", LOANS, False, False),
    ("elevated_tone_admitted", LOANS, True, True),
    ("topical_query_admitted", HURT, False, True),
    ("history_ask_admitted", HISTORY, False, True),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("label,query,distress,admitted", CASES, ids=[c[0] for c in CASES])
async def test_same_sensitive_fact_same_verdict_on_all_paths(tmp_path, monkeypatch, label, query, distress, admitted):
    profile = await _via_profile(tmp_path, query, distress)
    facts = await _via_facts(query, distress)
    graph = await _via_graph(tmp_path, query, distress, monkeypatch)
    assert profile[0] is admitted, f"profile {label}"
    assert facts[0] is admitted, f"facts {label}"
    assert graph[0] is admitted, f"graph {label}"
    # the non-sensitive fact is never collateral damage on any path
    assert facts[1] is True and graph[1] is True and profile[1] is True


@pytest.mark.asyncio
async def test_recent_facts_refused_on_gated_turn_admitted_on_open_turn():
    assert (await _via_facts(LOANS, False, recent=True)) == (False, True)
    assert (await _via_facts(LOANS, True, recent=True)) == (True, True)


@pytest.mark.asyncio
async def test_ungated_caller_is_unchanged():
    """No SENSITIVE_OPEN set (shutdown reflection, CLI): everything passes — the profile's own contract."""
    retr = MemoryRetriever(corpus_manager=MagicMock(), chroma_store=_FakeStore(_fact_rows()))
    assert SENSITIVE_OPEN.get() is None
    rows = await retr.get_facts(LOANS, limit=10)
    assert VALUE in " ".join(r["content"] for r in rows)


@pytest.mark.asyncio
async def test_facts_gate_is_the_profile_predicate_not_a_copy(monkeypatch):
    """Changing the profile's predicate changes the facts verdict: one function, not two."""
    retr = MemoryRetriever(corpus_manager=MagicMock(), chroma_store=_FakeStore(_fact_rows()))
    monkeypatch.setattr(up.relation_classifier, "is_sensitive_relation", lambda rel: False)
    token = SENSITIVE_OPEN.set(False)
    try:
        rows = await retr.get_facts(LOANS, limit=10)
    finally:
        SENSITIVE_OPEN.reset(token)
    assert VALUE in " ".join(r["content"] for r in rows)
