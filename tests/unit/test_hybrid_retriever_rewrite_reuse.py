"""HybridRetriever: topic reuse + additive rewrite candidates, bag unchanged.

2026-10-25, owner decision query-rewrite ON; referee redesign. Classes: BC-31
(two routes to the same purpose), BC-82 (live config), BC-72 (the serial ~1 s
call in the memories task is TopicManager's topic LLM call, not a rewrite),
BC-06 (the gate keeps scoring the raw query).

Contracts, driven through THE deployed ``HybridRetriever.retrieve``,
``utils.query_rewriter.rewrite_query`` and ``MemoryRetriever.get_memories``
(only the TopicManager LLM, the chroma search and the gate seams are doubles):
  * the local keyword/synonym expansion ALWAYS runs and is identical with or
    without a hint; the enable_query_rewrite switch does not remove it;
  * the pipeline's Stage-1 topics replace the TopicManager LLM call; no hint
    means the retriever makes its own call as before;
  * the LLM rewrite sentence is an ADDITIONAL candidate query merged (deduped
    by id) after the bag pool; bag candidates are never displaced or altered;
  * switch off: no additive query.
"""
import asyncio
from datetime import datetime
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

import memory.hybrid_retriever as hr
import utils.query_rewriter as qr
from gui.settings_core import apply_streaming
from memory.corpus_manager import CorpusManager
from memory.memory_retriever import MemoryRetriever
from utils.query_rewriter import set_turn_hint

RAW = "what did Rowan say about the trip to the coast last spring"
RAW_WRAPPED = "  what did Rowan say about\n  the trip to the coast last spring\n"
TOPIC = "Coastal Trip"


@pytest.fixture(autouse=True)
def _clear_hint():
    set_turn_hint(None)
    yield
    set_turn_hint(None)


@pytest.fixture
def topic_llm(monkeypatch):
    """Counting TopicManager double = the ~1 s topic LLM round-trip."""
    calls = []

    class FakeTopicManager:
        def __init__(self, *a, **kw):
            calls.append("init")

        def get_primary_topic(self, query):
            calls.append(query)
            return TOPIC

    monkeypatch.setattr(qr.topic_manager, "TopicManager", FakeTopicManager)
    return calls


def _doc(i, score=0.5):
    return {"id": f"d{i}", "content": f"User: m{i}\nAssistant: a{i}",
            "query": f"m{i}", "response": f"a{i}", "metadata": {},
            "collection": "conversations", "semantic_score": score}


def _retriever(bag_pool, extra_pool=None):
    """Real HybridRetriever; _semantic_search double answers per query text."""
    r = hr.HybridRetriever(chroma_store=MagicMock())
    seen = []

    async def routed(query, n_results=90):
        seen.append((query, n_results))
        if query.startswith("LLM REWRITE"):
            return [dict(d) for d in (extra_pool or [])]
        return [dict(d) for d in bag_pool]

    r._semantic_search = routed
    r.seen = seen
    return r


@pytest.mark.parametrize("q", [RAW, RAW_WRAPPED], ids=["clean", "wrapped"])
class TestTopicReuseAndBagUnchanged:
    def test_pipeline_topics_replace_the_topic_llm_call(self, topic_llm, q):
        r = _retriever([_doc(1)])
        asyncio.run(r.retrieve(q, limit=5, topics=[TOPIC]))
        assert topic_llm == [], "TopicManager LLM was called despite the hint"

    def test_without_hint_the_retriever_makes_its_own_topic_call(self, topic_llm, q):
        r = _retriever([_doc(1)])
        asyncio.run(r.retrieve(q, limit=5))
        assert topic_llm.count("init") == 1

    def test_bag_query_identical_with_and_without_hint(self, topic_llm, q):
        r1 = _retriever([_doc(1)])
        asyncio.run(r1.retrieve(q, limit=5))
        r2 = _retriever([_doc(1)])
        asyncio.run(r2.retrieve(q, limit=5, topics=[TOPIC]))
        r3 = _retriever([_doc(1)])
        asyncio.run(r3.retrieve(q, limit=5, topics=[TOPIC], rewritten_query="LLM REWRITE x"))
        assert r1.seen[0] == r2.seen[0] == r3.seen[0]
        assert r1.seen[0][0] == qr.rewrite_query(q, topics=[TOPIC])


@pytest.mark.parametrize("q", [RAW, RAW_WRAPPED], ids=["clean", "wrapped"])
class TestAdditiveRewriteCandidates:
    def test_extra_candidates_merge_after_bag_deduped_and_bag_untouched(self, topic_llm, q):
        bag = [_doc(1, 0.9), _doc(2, 0.8)]
        extra = [_doc(2, 0.99), _doc(3, 0.7)]  # d2 duplicate with a different score
        r = _retriever(bag, extra)
        res = asyncio.run(r.retrieve(q, limit=10, topics=[TOPIC],
                                     rewritten_query="LLM REWRITE of the trip"))
        by_id = {m["id"]: m for m in res}
        assert set(by_id) == {"d1", "d2", "d3"}, "additive candidate d3 missing"
        assert len(res) == 3, "duplicate id was not deduped"
        assert by_id["d2"]["semantic_score"] == 0.8, "bag candidate's score was replaced"
        assert by_id["d1"]["semantic_score"] == 0.9

    def test_each_query_keeps_its_own_per_collection_cap(self, topic_llm, q):
        r = _retriever([_doc(1)], [_doc(2)])
        asyncio.run(r.retrieve(q, limit=5, topics=[TOPIC],
                               rewritten_query="LLM REWRITE of the trip"))
        assert [n for _, n in r.seen] == [15, 15]
        assert r.seen[1][0] == "LLM REWRITE of the trip"

    def test_switch_off_means_no_additive_query_but_bag_and_topics_stay(
            self, monkeypatch, topic_llm, q):
        import config.app_config as cfg
        monkeypatch.setattr(cfg, "ENABLE_QUERY_REWRITE", False)
        r = _retriever([_doc(1)], [_doc(2)])
        res = asyncio.run(r.retrieve(q, limit=5, topics=[TOPIC],
                                     rewritten_query="LLM REWRITE of the trip"))
        assert len(r.seen) == 1, "additive query ran with the switch off"
        assert [m["id"] for m in res] == ["d1"]
        assert topic_llm == [], "topic reuse is not a rewrite; it stays on"
        assert r.seen[0][0] == qr.rewrite_query(q, topics=[TOPIC]), "local expansion must stay"

    def test_no_rewrite_means_single_search(self, topic_llm, q):
        r = _retriever([_doc(1)], [_doc(2)])
        asyncio.run(r.retrieve(q, limit=5, topics=[TOPIC]))
        assert len(r.seen) == 1


class TestSettingsCheckboxReachesTheRunningRetriever:
    def test_same_instance_before_after(self, topic_llm):
        r = _retriever([_doc(1)], [_doc(2)])
        orch = NS(config={"features": {}}, context_pipeline=None, prompt_builder=None)

        def apply(disable):
            return apply_streaming(
                orch, disable_best_of=True, disable_query_rewrite=disable,
                best_of_latency_budget_s=0,
                save=lambda fn: (True, None),
            )

        def run():
            before = len(r.seen)
            res = asyncio.run(r.retrieve(RAW, limit=5, topics=[TOPIC],
                                         rewritten_query="LLM REWRITE of the trip"))
            return len(r.seen) - before, {m["id"] for m in res}

        assert run() == (2, {"d1", "d2"}), "enabled: additive query runs"
        assert apply(True)["ok"]
        assert run() == (1, {"d1"}), "disabled: the SAME retriever drops the additive query"
        assert apply(False)["ok"]
        assert run() == (2, {"d1", "d2"}), "re-enabled: additive query again"


class TestMemoryRetrieverHandOff:
    """get_memories hands the turn hint to candidate generation ONLY; the
    gate's query stays the (reformulated) raw one."""

    @pytest.fixture
    def retriever(self, tmp_path):
        cm = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
        return MemoryRetriever(corpus_manager=cm, chroma_store=None)

    @pytest.mark.parametrize("q", [RAW, RAW_WRAPPED], ids=["clean", "wrapped"])
    def test_hint_reaches_candidate_generation_not_the_gate(self, retriever, q):
        # set_turn_hint and get_memories must share one task context
        seen = asyncio.run(self._in_one_task(retriever, q, "LLM REWRITE of the trip question"))
        assert seen["kw"] == {"rewritten_query": "LLM REWRITE of the trip question",
                              "topics": [TOPIC]}
        assert "LLM REWRITE" not in seen["gate_q"], "gate must score the raw query"
        assert "LLM REWRITE" not in seen["semantic_q"]

    def test_topics_only_hint_passes_topics_only(self, retriever):
        seen = asyncio.run(self._in_one_task(retriever, RAW, None))
        assert seen["kw"] == {"topics": [TOPIC]}

    def test_unrelated_query_in_same_turn_gets_no_hint(self, retriever):
        seen = asyncio.run(self._in_one_task(
            retriever, RAW, "LLM REWRITE of the trip question",
            query="how is my sleep schedule going this month"))
        assert seen["kw"] == {}

    async def _in_one_task(self, retriever, raw, rewritten, query=None):
        set_turn_hint(raw, rewritten, [TOPIC])
        seen = {}

        async def fake_semantic(q, n_results=0, **kw):
            seen["semantic_q"], seen["kw"] = q, kw
            return [{"id": "m1", "content": "User: trip\nAssistant: ok",
                     "relevance_score": 0.8, "timestamp": datetime(2026, 8, 1)}]

        async def fake_combine(**kwargs):
            seen["gate_q"] = kwargs.get("query")
            return list(kwargs.get("semantic") or [])

        async def fake_rerank(mems, q):
            return mems

        retriever._get_semantic_memories = fake_semantic
        retriever._combine_memories = fake_combine
        retriever._maybe_cross_encoder_rerank = fake_rerank
        await retriever.get_memories(query or (raw.strip() + " coast Tamsin"), limit=5)
        return seen
