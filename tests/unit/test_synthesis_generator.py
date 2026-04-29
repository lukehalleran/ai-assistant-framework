"""
Tests for knowledge/synthesis_generator.py

Tests the cross-store sampling generator with mocked ChromaDB (facts) and FAISS (wiki).
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge.synthesis_generator import SynthesisGenerator
from knowledge.synthesis_models import SynthesisCandidate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fact_item(doc_id, subject, predicate, obj, content=""):
    """Build a fake ChromaDB query result for a fact."""
    return {
        "id": doc_id,
        "content": content or f"{subject} {predicate} {obj}",
        "metadata": {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
        },
        "relevance_score": 0.8,
        "collection": "facts",
        "rank": 1,
    }


def _make_wiki_item(doc_id, title, content):
    """Build a fake ChromaDB query result for a wiki article (legacy helper for non-FAISS tests)."""
    return {
        "id": doc_id,
        "content": content,
        "metadata": {"title": title, "source": "wikipedia"},
        "relevance_score": 0.7,
        "collection": "wiki_knowledge",
        "rank": 1,
    }


def _make_faiss_wiki_result(title, content, similarity=0.7):
    """Build a FAISS semantic_search_with_neighbors result dict."""
    return {
        "text": content,
        "content": content,
        "title": title,
        "source": "wikipedia",
        "namespace": "wikipedia",
        "similarity": similarity,
        "timestamp": "",
        "section": "",
        "section_level": 0,
        "chunk_index": 0,
    }


# FAISS wiki pool (returned by mocked semantic_search_with_neighbors)
_FAISS_WIKI_POOL = [
    _make_faiss_wiki_result("Quantum Entanglement", "Quantum entanglement is a phenomenon where particles become correlated"),
    _make_faiss_wiki_result("Treaty of Westphalia", "The Peace of Westphalia established the principle of state sovereignty"),
    _make_faiss_wiki_result("Lotka-Volterra Equations", "The Lotka-Volterra equations describe predator-prey population dynamics"),
    _make_faiss_wiki_result("Byzantine Fault Tolerance", "Byzantine fault tolerance allows distributed systems to reach consensus"),
    _make_faiss_wiki_result("Baroque Counterpoint", "Baroque counterpoint involves independent melodic voices constrained by harmony"),
    _make_faiss_wiki_result("Mycelial Networks", "Mycorrhizal networks distribute nutrients between trees in forest ecosystems"),
    _make_faiss_wiki_result("Fermentation Chemistry", "Fermentation converts sugars to acids, gases, or alcohol using microorganisms"),
    _make_faiss_wiki_result("TCP Congestion Control", "TCP slow start probes network capacity by exponentially increasing window size"),
]


def _mock_faiss_search(query, k=8):
    """Mock FAISS semantic_search_with_neighbors returning shuffled wiki pool."""
    import random
    sample = random.sample(_FAISS_WIKI_POOL, min(k, len(_FAISS_WIKI_POOL)))
    return sample


@pytest.fixture
def mock_store():
    """Mock MultiCollectionChromaStore that returns diverse facts (wiki is via FAISS)."""
    store = MagicMock()

    facts_pool = [
        _make_fact_item("f1", "user", "works_at", "Acme Brewery", "User works at Acme Brewery"),
        _make_fact_item("f2", "user", "has_cat", "Paczki", "User has a cat named Paczki"),
        _make_fact_item("f3", "user", "lives_in", "Portland", "User lives in Portland"),
        _make_fact_item("f4", "user", "bench_press_max", "225lbs", "User bench press max is 225lbs"),
        _make_fact_item("f5", "user", "studies", "actuarial science", "User studies actuarial science"),
        _make_fact_item("f6", "user", "hobby", "sourdough baking", "User enjoys sourdough baking"),
        _make_fact_item("f7", "user", "goal", "run a marathon", "User wants to run a marathon"),
        _make_fact_item("f8", "user", "brother_name", "Jake", "User's brother is named Jake"),
    ]

    def query_side_effect(collection_name, query_text, n_results=5):
        if collection_name == "facts":
            import random
            sample = random.sample(facts_pool, min(n_results, len(facts_pool)))
            return sample
        return []

    store.query_collection = MagicMock(side_effect=query_side_effect)
    store.get_collection_stats = MagicMock(return_value={
        "facts": {"count": len(facts_pool)},
    })
    return store


@pytest.fixture
def mock_model_manager():
    """Mock ModelManager that returns plausible bridge articulations."""
    mm = MagicMock()

    async def generate_once_side_effect(prompt, model_name=None, system_prompt="",
                                        max_tokens=150, temperature=0.7, **kwargs):
        # Return a plausible connection claim
        return (
            "Both share a feedback mechanism where incremental adjustments "
            "converge on an optimal state through iterative refinement."
        )

    mm.generate_once = AsyncMock(side_effect=generate_once_side_effect)
    return mm


@pytest.fixture
def mock_graph_memory():
    """Mock GraphMemory with shortest_path support."""
    gm = MagicMock()
    gm.node_count.return_value = 50
    gm.edge_count.return_value = 80
    gm.shortest_path.return_value = ["a", "user", "b"]  # typical star-graph path
    return gm


@pytest.fixture
def mock_entity_resolver():
    er = MagicMock()
    er.resolve.return_value = None  # no alias resolution by default
    return er


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSynthesisGenerator:

    @pytest.mark.asyncio
    @patch("knowledge.semantic_search.semantic_search_with_neighbors", side_effect=_mock_faiss_search)
    async def test_generate_produces_valid_candidates(self, mock_faiss, mock_store, mock_model_manager):
        """Generator should produce SynthesisCandidate objects."""
        gen = SynthesisGenerator(
            chroma_store=mock_store,
            model_manager=mock_model_manager,
        )
        with patch("config.app_config.SYNTHESIS_GENERATOR_ENABLED", True), \
             patch("config.app_config.SYNTHESIS_GENERATOR_LLM_CONCURRENCY", 3), \
             patch("config.app_config.SYNTHESIS_GENERATOR_MIN_GRAPH_NODES", 20):
            candidates = await gen.generate_candidates(count=3)

        assert len(candidates) > 0
        for c in candidates:
            assert isinstance(c, SynthesisCandidate)
            assert c.concept_a
            assert c.concept_b
            assert c.connection_claim
            assert len(c.source_domains) >= 1
            assert c.endpoint_distance > 0

    @pytest.mark.asyncio
    @patch("knowledge.semantic_search.semantic_search_with_neighbors", side_effect=_mock_faiss_search)
    async def test_generate_respects_count(self, mock_faiss, mock_store, mock_model_manager):
        """Should not return more candidates than requested."""
        gen = SynthesisGenerator(
            chroma_store=mock_store,
            model_manager=mock_model_manager,
        )
        with patch("config.app_config.SYNTHESIS_GENERATOR_ENABLED", True), \
             patch("config.app_config.SYNTHESIS_GENERATOR_LLM_CONCURRENCY", 3), \
             patch("config.app_config.SYNTHESIS_GENERATOR_MIN_GRAPH_NODES", 20):
            candidates = await gen.generate_candidates(count=2)

        assert len(candidates) <= 2

    @pytest.mark.asyncio
    async def test_generate_disabled_returns_empty(self, mock_store, mock_model_manager):
        """When disabled, should return empty list."""
        gen = SynthesisGenerator(
            chroma_store=mock_store,
            model_manager=mock_model_manager,
        )
        with patch("config.app_config.SYNTHESIS_GENERATOR_ENABLED", False), \
             patch("config.app_config.SYNTHESIS_GENERATOR_LLM_CONCURRENCY", 3), \
             patch("config.app_config.SYNTHESIS_GENERATOR_MIN_GRAPH_NODES", 20):
            candidates = await gen.generate_candidates(count=5)

        assert candidates == []

    @pytest.mark.asyncio
    async def test_sparse_graph_skips(self, mock_store, mock_model_manager, mock_graph_memory):
        """Should skip generation when graph is too sparse."""
        mock_graph_memory.node_count.return_value = 5  # below threshold
        gen = SynthesisGenerator(
            chroma_store=mock_store,
            model_manager=mock_model_manager,
            graph_memory=mock_graph_memory,
        )
        with patch("config.app_config.SYNTHESIS_GENERATOR_ENABLED", True), \
             patch("config.app_config.SYNTHESIS_GENERATOR_LLM_CONCURRENCY", 3), \
             patch("config.app_config.SYNTHESIS_GENERATOR_MIN_GRAPH_NODES", 20):
            candidates = await gen.generate_candidates(count=5)

        assert candidates == []

    @pytest.mark.asyncio
    @patch("knowledge.semantic_search.semantic_search_with_neighbors", side_effect=_mock_faiss_search)
    async def test_no_connection_skipped(self, mock_faiss, mock_store, mock_model_manager):
        """LLM returning NO_CONNECTION should result in no candidate."""
        mock_model_manager.generate_once = AsyncMock(return_value="NO_CONNECTION")

        gen = SynthesisGenerator(
            chroma_store=mock_store,
            model_manager=mock_model_manager,
        )
        with patch("config.app_config.SYNTHESIS_GENERATOR_ENABLED", True), \
             patch("config.app_config.SYNTHESIS_GENERATOR_LLM_CONCURRENCY", 3), \
             patch("config.app_config.SYNTHESIS_GENERATOR_MIN_GRAPH_NODES", 20):
            candidates = await gen.generate_candidates(count=3)

        assert candidates == []

    @pytest.mark.asyncio
    @patch("knowledge.semantic_search.semantic_search_with_neighbors", side_effect=_mock_faiss_search)
    async def test_llm_error_handled_gracefully(self, mock_faiss, mock_store, mock_model_manager):
        """LLM errors should not crash, just reduce candidate count."""
        mock_model_manager.generate_once = AsyncMock(
            side_effect=RuntimeError("API timeout")
        )

        gen = SynthesisGenerator(
            chroma_store=mock_store,
            model_manager=mock_model_manager,
        )
        with patch("config.app_config.SYNTHESIS_GENERATOR_ENABLED", True), \
             patch("config.app_config.SYNTHESIS_GENERATOR_LLM_CONCURRENCY", 3), \
             patch("config.app_config.SYNTHESIS_GENERATOR_MIN_GRAPH_NODES", 20):
            candidates = await gen.generate_candidates(count=3)

        # Should not raise, just return empty
        assert candidates == []

    @pytest.mark.asyncio
    @patch("knowledge.semantic_search.semantic_search_with_neighbors", return_value=[])
    async def test_empty_stores_returns_empty(self, mock_faiss, mock_model_manager):
        """Empty facts + empty FAISS should return no candidates."""
        store = MagicMock()
        store.query_collection = MagicMock(return_value=[])

        gen = SynthesisGenerator(
            chroma_store=store,
            model_manager=mock_model_manager,
        )
        with patch("config.app_config.SYNTHESIS_GENERATOR_ENABLED", True), \
             patch("config.app_config.SYNTHESIS_GENERATOR_LLM_CONCURRENCY", 3), \
             patch("config.app_config.SYNTHESIS_GENERATOR_MIN_GRAPH_NODES", 20):
            candidates = await gen.generate_candidates(count=3)

        assert candidates == []

    def test_extract_concept_name_personal_fact(self, mock_store, mock_model_manager):
        """Should extract concept name from fact metadata."""
        gen = SynthesisGenerator(chroma_store=mock_store, model_manager=mock_model_manager)

        item = _make_fact_item("f1", "user", "has_cat", "Paczki")
        name = gen._extract_concept_name(item, source="personal")
        assert name == "Paczki"

    def test_extract_concept_name_user_subject(self, mock_store, mock_model_manager):
        """When object is 'user', should use subject instead."""
        gen = SynthesisGenerator(chroma_store=mock_store, model_manager=mock_model_manager)

        item = _make_fact_item("f1", "Daemon", "created_by", "user")
        name = gen._extract_concept_name(item, source="personal")
        assert name == "Daemon"

    def test_extract_concept_name_wiki(self, mock_store, mock_model_manager):
        """Should extract title from wiki metadata."""
        gen = SynthesisGenerator(chroma_store=mock_store, model_manager=mock_model_manager)

        item = _make_wiki_item("w1", "Quantum Entanglement", "Some content")
        name = gen._extract_concept_name(item, source="wiki")
        assert name == "Quantum Entanglement"

    def test_classify_domain_personal(self, mock_store, mock_model_manager):
        """Should classify personal facts using categorize_relation."""
        gen = SynthesisGenerator(chroma_store=mock_store, model_manager=mock_model_manager)

        item = _make_fact_item("f1", "user", "bench_press_max", "225")
        domain = gen._classify_domain(item)
        assert domain == "fitness"

    def test_classify_domain_wiki(self, mock_store, mock_model_manager):
        """Should classify wiki articles by keyword heuristics."""
        gen = SynthesisGenerator(chroma_store=mock_store, model_manager=mock_model_manager)

        item = _make_wiki_item("w1", "Quantum Entanglement", "Quantum physics describes atom and particle behavior")
        item["collection"] = "wiki_knowledge"
        domain = gen._classify_wiki_domain(item)
        assert domain == "science"

    def test_endpoint_distance_with_graph(self, mock_store, mock_model_manager, mock_graph_memory, mock_entity_resolver):
        """Should compute distance from graph shortest path."""
        # Make entity resolver return the name itself, and graph contain any node
        mock_entity_resolver.resolve.side_effect = lambda x: x
        mock_graph_memory.graph.__contains__ = MagicMock(return_value=True)
        gen = SynthesisGenerator(
            chroma_store=mock_store,
            model_manager=mock_model_manager,
            graph_memory=mock_graph_memory,
            entity_resolver=mock_entity_resolver,
        )
        # shortest_path returns ["a", "user", "b"] — length 3
        dist = gen._compute_endpoint_distance("a", "b")
        assert 0.15 <= dist <= 1.0
        assert abs(dist - 0.5) < 0.01  # 3/6 = 0.5

    def test_endpoint_distance_without_graph(self, mock_store, mock_model_manager):
        """Should return default mid-range distance without graph."""
        gen = SynthesisGenerator(
            chroma_store=mock_store,
            model_manager=mock_model_manager,
        )
        dist = gen._compute_endpoint_distance("anything", "else")
        assert dist == 0.55

    def test_form_pairs_deduplicates(self, mock_store, mock_model_manager):
        """Should not produce duplicate (concept_a, concept_b) pairs."""
        gen = SynthesisGenerator(chroma_store=mock_store, model_manager=mock_model_manager)

        personal = [
            _make_fact_item("f1", "user", "has_cat", "Paczki"),
            _make_fact_item("f2", "user", "has_cat", "Paczki"),  # duplicate concept
        ]
        wiki = [_make_wiki_item("w1", "Quantum Entanglement", "Physics content")]

        pairs = gen._form_pairs(personal, wiki, max_pairs=10)
        # Should deduplicate — only one pair with Paczki + Quantum Entanglement
        pair_keys = [frozenset([
            gen._extract_concept_name(p, "personal").lower(),
            gen._extract_concept_name(w, "wiki").lower(),
        ]) for p, w in pairs]
        assert len(pair_keys) == len(set(pair_keys))

    def test_get_sampling_stats(self, mock_store, mock_model_manager, mock_graph_memory):
        """Should return collection and graph counts."""
        mock_index = MagicMock()
        mock_index.loaded = True
        mock_index._total_rows = 41_000_000

        gen = SynthesisGenerator(
            chroma_store=mock_store,
            model_manager=mock_model_manager,
            graph_memory=mock_graph_memory,
        )
        with patch("knowledge.semantic_search.get_index", return_value=mock_index):
            stats = gen.get_sampling_stats()
        assert stats["facts_count"] == 8
        assert stats["wiki_count"] == 41_000_000
        assert stats["wiki_source"] == "faiss"
        assert stats["graph_nodes"] == 50

    @pytest.mark.asyncio
    @patch("knowledge.semantic_search.semantic_search_with_neighbors", side_effect=_mock_faiss_search)
    async def test_short_response_skipped(self, mock_faiss, mock_store, mock_model_manager):
        """Very short LLM responses (< 5 words) should be skipped."""
        mock_model_manager.generate_once = AsyncMock(return_value="They connect somehow.")

        gen = SynthesisGenerator(
            chroma_store=mock_store,
            model_manager=mock_model_manager,
        )
        with patch("config.app_config.SYNTHESIS_GENERATOR_ENABLED", True), \
             patch("config.app_config.SYNTHESIS_GENERATOR_LLM_CONCURRENCY", 3), \
             patch("config.app_config.SYNTHESIS_GENERATOR_MIN_GRAPH_NODES", 20):
            candidates = await gen.generate_candidates(count=3)

        # "They connect somehow." is only 3 words — should be skipped
        assert candidates == []

    @pytest.mark.asyncio
    @patch("knowledge.semantic_search.semantic_search_with_neighbors", side_effect=_mock_faiss_search)
    async def test_graph_used_for_distance(self, mock_faiss, mock_store, mock_model_manager, mock_graph_memory, mock_entity_resolver):
        """When graph is available, candidates should use graph-based distance."""
        # Make entity resolver return the name itself, and graph contain any node
        mock_entity_resolver.resolve.side_effect = lambda x: x
        mock_graph_memory.graph.__contains__ = MagicMock(return_value=True)
        gen = SynthesisGenerator(
            chroma_store=mock_store,
            model_manager=mock_model_manager,
            graph_memory=mock_graph_memory,
            entity_resolver=mock_entity_resolver,
        )
        with patch("config.app_config.SYNTHESIS_GENERATOR_ENABLED", True), \
             patch("config.app_config.SYNTHESIS_GENERATOR_LLM_CONCURRENCY", 3), \
             patch("config.app_config.SYNTHESIS_GENERATOR_MIN_GRAPH_NODES", 20):
            candidates = await gen.generate_candidates(count=2)

        # All candidates should have graph-computed distance (path of length 3 -> 0.5)
        for c in candidates:
            assert c.endpoint_distance != 0.55  # not the default
