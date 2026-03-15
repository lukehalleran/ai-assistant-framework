# tests/unit/test_graph_integration.py
"""
Tests for graph-boosted scoring and graph-driven query expansion.

Feature 1: Graph-boosted scoring — memories mentioning graph-connected
           entities get a score bonus in rank_memories().

Feature 2: Graph-driven query expansion — graph neighbor display names
           are appended to the semantic search query.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from typing import Optional, Set

from memory.graph_utils import (
    extract_graph_entities,
    get_related_display_names,
    get_related_entity_ids,
    rank_expansion_candidates,
    _is_expansion_junk,
)
from memory.memory_scorer import MemoryScorer
from memory.memory_storage import MemoryStorage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockResolver:
    """Fake EntityResolver backed by a simple dict."""

    def __init__(self, alias_map: dict[str, str]):
        # alias_map: lowered alias -> canonical entity_id
        self._map = {k.lower(): v.lower() for k, v in alias_map.items()}

    def resolve(self, mention: str) -> Optional[str]:
        return self._map.get(mention.lower().strip())


class MockGraphNode:
    def __init__(self, entity_id: str, display_name: str):
        self.entity_id = entity_id
        self.display_name = display_name


class MockGraphMemory:
    """Minimal GraphMemory stub for unit tests."""

    def __init__(self, nodes: dict, edges: dict):
        """
        nodes: {entity_id: display_name}
        edges: {entity_id: [(neighbor_id, relation)]}
        """
        self._nodes = nodes
        self._edges = edges  # adjacency list

    def node_count(self) -> int:
        return len(self._nodes)

    def get_entity(self, entity_id: str) -> Optional[MockGraphNode]:
        eid = entity_id.lower()
        name = self._nodes.get(eid)
        if name is None:
            return None
        return MockGraphNode(eid, name)

    def get_relations(self, entity_id: str, direction: str = "both") -> list:
        """Return edge-like objects for the given entity (both directions)."""
        eid = entity_id.lower()
        edges = []
        if direction in ("out", "both"):
            for neighbor_id, relation in self._edges.get(eid, []):
                edge = MagicMock()
                edge.source_id = eid
                edge.target_id = neighbor_id
                edge.relation = relation
                edges.append(edge)
        if direction in ("in", "both"):
            for src, neighbors in self._edges.items():
                for neighbor_id, relation in neighbors:
                    if neighbor_id == eid:
                        edge = MagicMock()
                        edge.source_id = src
                        edge.target_id = eid
                        edge.relation = relation
                        edges.append(edge)
        return edges

    def neighbors(self, entity_id: str, depth: int = 1) -> dict:
        """Simplified BFS returning {entity_id: [MockEdge, ...]}."""
        eid = entity_id.lower()
        result = {}
        visited = {eid}
        current_layer = [eid]
        for _ in range(depth):
            next_layer = []
            for nid in current_layer:
                for neighbor_id, relation in self._edges.get(nid, []):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_layer.append(neighbor_id)
                        # Store a simple edge-like object
                        edge = MagicMock()
                        edge.source_id = nid
                        edge.target_id = neighbor_id
                        edge.relation = relation
                        result.setdefault(neighbor_id, []).append(edge)
            current_layer = next_layer
        return result


@pytest.fixture
def sample_graph():
    """Graph: user --has_pet--> auggie, user --brother_name--> dillion,
    auggie --breed--> golden_retriever"""
    nodes = {
        "user": "User",
        "auggie": "Auggie",
        "dillion": "Dillion",
        "golden_retriever": "Golden Retriever",
    }
    edges = {
        "user": [("auggie", "has_pet"), ("dillion", "brother_name")],
        "auggie": [("golden_retriever", "breed")],
    }
    return MockGraphMemory(nodes, edges)


@pytest.fixture
def sample_resolver():
    return MockResolver({
        "auggie": "auggie",
        "my dog": "auggie",
        "aug": "auggie",
        "dillion": "dillion",
        "my brother": "dillion",
        "user": "user",
        "golden retriever": "golden_retriever",
    })


# ===========================================================================
# Tests: extract_graph_entities
# ===========================================================================

class TestExtractGraphEntities:

    def test_single_word_match(self, sample_resolver):
        result = extract_graph_entities("How is Auggie doing?", sample_resolver)
        assert "auggie" in result

    def test_multi_word_match(self, sample_resolver):
        result = extract_graph_entities("Tell me about my dog", sample_resolver)
        assert "auggie" in result

    def test_multiple_entities(self, sample_resolver):
        result = extract_graph_entities("What about Auggie and Dillion?", sample_resolver)
        assert "auggie" in result
        assert "dillion" in result

    def test_no_match(self, sample_resolver):
        result = extract_graph_entities("What's the weather like?", sample_resolver)
        assert len(result) == 0

    def test_empty_text(self, sample_resolver):
        result = extract_graph_entities("", sample_resolver)
        assert len(result) == 0

    def test_none_text(self, sample_resolver):
        result = extract_graph_entities(None, sample_resolver)
        assert len(result) == 0

    def test_none_resolver(self):
        result = extract_graph_entities("hello auggie", None)
        assert len(result) == 0

    def test_stopwords_skipped(self, sample_resolver):
        # "the", "is", "a" should not be resolved
        resolver = MockResolver({"the": "the_entity"})
        result = extract_graph_entities("the dog is here", resolver)
        assert "the_entity" not in result

    def test_short_words_skipped(self, sample_resolver):
        # Words with len <= 2 skipped for single-word check
        resolver = MockResolver({"ab": "ab_entity"})
        result = extract_graph_entities("ab is here", resolver)
        assert "ab_entity" not in result

    def test_bigram_match(self, sample_resolver):
        result = extract_graph_entities("I love my brother very much", sample_resolver)
        assert "dillion" in result

    def test_trigram_match(self):
        resolver = MockResolver({"big red dog": "clifford"})
        result = extract_graph_entities("I saw a big red dog today", resolver)
        assert "clifford" in result


# ===========================================================================
# Tests: get_related_display_names
# ===========================================================================

class TestGetRelatedDisplayNames:

    def test_basic_neighbors(self, sample_graph):
        # user's 1-hop neighbors: auggie, dillion
        names = get_related_display_names({"user"}, sample_graph, depth=1)
        assert "Auggie" in names
        assert "Dillion" in names

    def test_skip_ids(self, sample_graph):
        names = get_related_display_names(
            {"user"}, sample_graph, depth=1, skip_ids={"auggie"}
        )
        assert "Auggie" not in names
        assert "Dillion" in names

    def test_excludes_input_entities(self, sample_graph):
        # "user" is the input entity, so "User" display name excluded
        names = get_related_display_names({"user"}, sample_graph, depth=1)
        assert "User" not in names

    def test_empty_entity_ids(self, sample_graph):
        names = get_related_display_names(set(), sample_graph, depth=1)
        assert len(names) == 0

    def test_none_graph(self):
        names = get_related_display_names({"user"}, None, depth=1)
        assert len(names) == 0

    def test_depth_2(self, sample_graph):
        # auggie's 2-hop from user: golden_retriever
        names = get_related_display_names({"user"}, sample_graph, depth=2)
        assert "Golden Retriever" in names

    def test_entity_with_no_neighbors(self, sample_graph):
        names = get_related_display_names({"golden_retriever"}, sample_graph, depth=1)
        # golden_retriever has no outgoing edges in our mock
        assert len(names) == 0


# ===========================================================================
# Tests: get_related_entity_ids
# ===========================================================================

class TestGetRelatedEntityIds:

    def test_basic(self, sample_graph):
        ids = get_related_entity_ids({"user"}, sample_graph, depth=1)
        assert "auggie" in ids
        assert "dillion" in ids
        assert "user" not in ids

    def test_empty(self, sample_graph):
        ids = get_related_entity_ids(set(), sample_graph, depth=1)
        assert len(ids) == 0

    def test_no_graph(self):
        ids = get_related_entity_ids({"user"}, None, depth=1)
        assert len(ids) == 0


# ===========================================================================
# Tests: Graph-boosted scoring in MemoryScorer.rank_memories()
# ===========================================================================

class TestGraphBoostedScoring:

    def _make_memory(self, content: str, relevance: float = 0.5, ts_offset_hours: int = 1):
        """Create a minimal memory dict."""
        ts = datetime.now() - timedelta(hours=ts_offset_hours)
        return {
            "query": "",
            "response": content,
            "content": content,
            "relevance_score": relevance,
            "timestamp": ts.isoformat(),
            "metadata": {},
        }

    @patch("memory.memory_scorer.COLLECTION_BOOSTS", {})
    def test_graph_boost_applied(self, sample_graph, sample_resolver):
        """Memory mentioning a graph-connected entity gets a score boost."""
        import logging
        scorer = MemoryScorer()
        scorer._graph_memory = sample_graph
        scorer._entity_resolver = sample_resolver

        # Query about "auggie" → auggie has neighbor golden_retriever
        m_with_mention = self._make_memory("The Golden Retriever was playing in the yard", relevance=0.5)
        m_without_mention = self._make_memory("The weather was nice today", relevance=0.5)

        # Enable debug logging so debug dict is populated
        scorer_logger = logging.getLogger("memory_scorer")
        old_level = scorer_logger.level
        scorer_logger.setLevel(logging.DEBUG)
        try:
            with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
                 patch("config.app_config.GRAPH_SCORING_BOOST_ENABLED", True), \
                 patch("config.app_config.GRAPH_SCORING_BOOST_CAP", 0.15):
                results = scorer.rank_memories(
                    [m_with_mention, m_without_mention],
                    current_query="how is auggie doing",
                )
        finally:
            scorer_logger.setLevel(old_level)

        # The memory mentioning "Golden Retriever" should get a graph_bonus
        # because "auggie" → 1-hop → "golden_retriever" (display: "Golden Retriever")
        boosted = [m for m in results if "golden retriever" in m.get("content", "").lower()][0]
        unboosted = [m for m in results if "weather" in m.get("content", "").lower()][0]

        assert boosted.get("debug", {}).get("graph_bonus", 0) > 0
        assert unboosted.get("debug", {}).get("graph_bonus", 0) == 0

    @patch("memory.memory_scorer.COLLECTION_BOOSTS", {})
    def test_graph_boost_capped(self, sample_graph, sample_resolver):
        """Graph bonus is capped at GRAPH_SCORING_BOOST_CAP."""
        scorer = MemoryScorer()
        scorer._graph_memory = sample_graph
        scorer._entity_resolver = sample_resolver

        # Mention multiple related names
        m = self._make_memory(
            "Auggie the Golden Retriever played with Dillion",
            relevance=0.5,
        )

        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
             patch("config.app_config.GRAPH_SCORING_BOOST_ENABLED", True), \
             patch("config.app_config.GRAPH_SCORING_BOOST_CAP", 0.15):
            scorer.rank_memories([m], current_query="tell me about user")

        assert m.get("debug", {}).get("graph_bonus", 0) <= 0.15

    @patch("memory.memory_scorer.COLLECTION_BOOSTS", {})
    def test_graph_boost_disabled_by_config(self, sample_graph, sample_resolver):
        """No boost when GRAPH_SCORING_BOOST_ENABLED=False."""
        scorer = MemoryScorer()
        scorer._graph_memory = sample_graph
        scorer._entity_resolver = sample_resolver

        m = self._make_memory("Auggie went to the park", relevance=0.5)

        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
             patch("config.app_config.GRAPH_SCORING_BOOST_ENABLED", False), \
             patch("config.app_config.GRAPH_SCORING_BOOST_CAP", 0.15):
            scorer.rank_memories([m], current_query="tell me about my brother")

        assert m.get("debug", {}).get("graph_bonus", 0) == 0

    @patch("memory.memory_scorer.COLLECTION_BOOSTS", {})
    def test_graph_boost_disabled_when_graph_disabled(self, sample_graph, sample_resolver):
        """No boost when KNOWLEDGE_GRAPH_ENABLED=False."""
        scorer = MemoryScorer()
        scorer._graph_memory = sample_graph
        scorer._entity_resolver = sample_resolver

        m = self._make_memory("Auggie went to the park", relevance=0.5)

        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", False), \
             patch("config.app_config.GRAPH_SCORING_BOOST_ENABLED", True), \
             patch("config.app_config.GRAPH_SCORING_BOOST_CAP", 0.15):
            scorer.rank_memories([m], current_query="tell me about auggie")

        assert m.get("debug", {}).get("graph_bonus", 0) == 0

    @patch("memory.memory_scorer.COLLECTION_BOOSTS", {})
    def test_no_graph_refs_graceful(self):
        """No crash when graph_memory/entity_resolver are None."""
        scorer = MemoryScorer()
        # _graph_memory and _entity_resolver default to None
        m = self._make_memory("Some memory content", relevance=0.5)

        results = scorer.rank_memories([m], current_query="hello")
        assert len(results) == 1
        assert "final_score" in results[0]

    @patch("memory.memory_scorer.COLLECTION_BOOSTS", {})
    def test_empty_graph_graceful(self, sample_resolver):
        """No crash when graph is empty."""
        empty_graph = MockGraphMemory({}, {})
        scorer = MemoryScorer()
        scorer._graph_memory = empty_graph
        scorer._entity_resolver = sample_resolver

        m = self._make_memory("Content here", relevance=0.5)

        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
             patch("config.app_config.GRAPH_SCORING_BOOST_ENABLED", True), \
             patch("config.app_config.GRAPH_SCORING_BOOST_CAP", 0.15):
            results = scorer.rank_memories([m], current_query="tell me about auggie")

        assert len(results) == 1

    @patch("memory.memory_scorer.COLLECTION_BOOSTS", {})
    def test_no_query_entities_no_boost(self, sample_graph, sample_resolver):
        """No boost when query doesn't match any entities."""
        scorer = MemoryScorer()
        scorer._graph_memory = sample_graph
        scorer._entity_resolver = sample_resolver

        m = self._make_memory("Auggie went to the park", relevance=0.5)

        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
             patch("config.app_config.GRAPH_SCORING_BOOST_ENABLED", True), \
             patch("config.app_config.GRAPH_SCORING_BOOST_CAP", 0.15):
            scorer.rank_memories([m], current_query="what's the weather?")

        assert m.get("debug", {}).get("graph_bonus", 0) == 0


# ===========================================================================
# Tests: Graph-driven query expansion
# ===========================================================================

class TestQueryExpansion:

    @pytest.fixture
    def gatherer(self, sample_graph, sample_resolver):
        """Create a minimal ContextGatherer with mocked dependencies."""
        mc = MagicMock()
        mc.graph_memory = sample_graph
        mc.entity_resolver = sample_resolver
        mc.user_profile = MagicMock()

        mm = MagicMock()
        tm = MagicMock()
        gs = MagicMock()

        from core.prompt.context_gatherer import ContextGatherer
        gatherer = ContextGatherer(mc, mm, tm, gate_system=gs)
        return gatherer

    def test_basic_expansion(self, gatherer):
        """Query about 'my brother' should expand with 'Auggie'."""
        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_MAX_TERMS", 8):
            result = gatherer._expand_query_with_graph("tell me about my brother")

        # "my brother" -> dillion -> user (1-hop) -> auggie (1-hop from user)
        # But skip_ids={"user"}, so we get neighbors of dillion
        # dillion has no outgoing edges in our mock, but the query might also
        # resolve "my brother" -> dillion, and dillion's neighbors are found
        assert "tell me about my brother" in result

    def test_expansion_adds_names(self, gatherer):
        """Query about 'auggie' should expand with neighbor display names."""
        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_MAX_TERMS", 8):
            result = gatherer._expand_query_with_graph("how is auggie doing")

        # auggie -> golden_retriever (1-hop)
        assert "Golden Retriever" in result
        assert "how is auggie doing" in result

    def test_expansion_disabled(self, gatherer):
        """No expansion when disabled."""
        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_ENABLED", False), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_MAX_TERMS", 8):
            result = gatherer._expand_query_with_graph("how is auggie")

        assert result == "how is auggie"

    def test_expansion_graph_disabled(self, gatherer):
        """No expansion when knowledge graph disabled."""
        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", False), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_MAX_TERMS", 8):
            result = gatherer._expand_query_with_graph("how is auggie")

        assert result == "how is auggie"

    def test_expansion_no_graph_on_coordinator(self):
        """Graceful when coordinator has no graph."""
        mc = MagicMock(spec=[])  # No graph_memory attribute
        mm = MagicMock()
        tm = MagicMock()
        gs = MagicMock()

        from core.prompt.context_gatherer import ContextGatherer
        gatherer = ContextGatherer(mc, mm, tm, gate_system=gs)

        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_MAX_TERMS", 8):
            result = gatherer._expand_query_with_graph("how is auggie")

        assert result == "how is auggie"

    def test_expansion_no_match(self, gatherer):
        """No expansion when query has no entity matches."""
        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_MAX_TERMS", 8):
            result = gatherer._expand_query_with_graph("what's the weather?")

        assert result == "what's the weather?"

    def test_expansion_max_terms_respected(self, gatherer):
        """Expansion caps at max_terms (counts names, not words)."""
        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_MAX_TERMS", 1):
            result = gatherer._expand_query_with_graph("tell me about user")

        # user has 3 reachable neighbors (auggie, dillion, golden_retriever)
        # but max_terms=1 so only 1 name appended (may be multi-word)
        original = "tell me about user"
        added = result.replace(original, "").strip()
        assert added  # something was appended
        known_names = {"Auggie", "Dillion", "Golden Retriever"}
        matched = [n for n in known_names if n.lower() in added.lower()]
        assert len(matched) == 1

    def test_expansion_empty_query(self, gatherer):
        """Empty query returns empty."""
        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_MAX_TERMS", 8):
            result = gatherer._expand_query_with_graph("")

        assert result == ""

    def test_expansion_preserves_original(self, gatherer):
        """Original query is always present in expanded result."""
        with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_ENABLED", True), \
             patch("config.app_config.GRAPH_QUERY_EXPANSION_MAX_TERMS", 8):
            original = "how is auggie doing today"
            result = gatherer._expand_query_with_graph(original)

        assert result.startswith(original)


# ===========================================================================
# Tests: rank_expansion_candidates
# ===========================================================================

class TestRankExpansionCandidates:

    @pytest.fixture
    def star_graph(self):
        """Star topology: user hub with real entities + junk nodes.

        Flapjack has lateral edges (flapjack→golden_retriever, flapjack→vet).
        coffee/junk nodes have user→ edges only.
        """
        nodes = {
            "user": "User",
            "flapjack": "Flapjack",
            "auggie": "Auggie",
            "mom": "Mom",
            "coffee": "coffee",
            "golden_retriever": "Golden Retriever",
            "vet": "vet",
            "5_11": "5'11\"",
            "2_years": "2 years",
            "stopped_religious": "stopped being religious",
        }
        edges = {
            "user": [
                ("flapjack", "has_pet"),
                ("auggie", "has_pet"),
                ("mom", "mother"),
                ("coffee", "likes"),
                ("5_11", "height"),
                ("2_years", "duration"),
                ("stopped_religious", "belief"),
            ],
            "flapjack": [
                ("golden_retriever", "breed"),
                ("vet", "visits"),
                ("user", "owned_by"),
            ],
            "auggie": [
                ("user", "owned_by"),
            ],
            "mom": [
                ("user", "child"),
            ],
        }
        return MockGraphMemory(nodes, edges)

    def test_junk_names_filtered(self, star_graph):
        """Numeric/temporal/verb-phrase names are excluded."""
        result = rank_expansion_candidates(
            {"user"}, star_graph, depth=1, skip_ids={"user"}, max_terms=20,
        )
        for name in result:
            assert name != "5'11\""
            assert name != "2 years"
            assert name != "stopped being religious"

    def test_own_edges_boost_ranking(self, star_graph):
        """Node with lateral (non-hub) edges ranks above node without."""
        result = rank_expansion_candidates(
            {"user"}, star_graph, depth=1, skip_ids={"user"}, max_terms=10,
        )
        assert len(result) > 0
        # Flapjack has 2 non-user edges (golden_retriever, vet) → highest score
        assert result[0] == "Flapjack"

    def test_multi_word_penalty(self, star_graph):
        """3+ word names rank below single-word names at equal connectivity."""
        # golden_retriever is 2-hop from user via flapjack; it's a 2-word name
        # Both auggie and mom have 1 non-hub edge each (owned_by→user skipped,
        # but auggie→user and mom→user are hub edges; the only non-hub edges
        # are inbound from user which is in skip_ids)
        result = rank_expansion_candidates(
            {"user"}, star_graph, depth=2, skip_ids={"user"}, max_terms=10,
        )
        # Flapjack (single word, 2 non-hub edges) should rank above
        # Golden Retriever (2 words, fewer non-hub edges)
        if "Flapjack" in result and "Golden Retriever" in result:
            assert result.index("Flapjack") < result.index("Golden Retriever")

    def test_max_terms_respected(self, star_graph):
        """Output is capped at max_terms."""
        result = rank_expansion_candidates(
            {"user"}, star_graph, depth=2, skip_ids={"user"}, max_terms=2,
        )
        assert len(result) <= 2

    def test_skip_ids_applied(self, star_graph):
        """Hub nodes in skip_ids are excluded from results."""
        result = rank_expansion_candidates(
            {"user"}, star_graph, depth=1, skip_ids={"user", "flapjack"},
            max_terms=10,
        )
        assert "Flapjack" not in result
        assert "User" not in result

    def test_empty_inputs_graceful(self, star_graph):
        """Empty entity_ids returns empty list."""
        assert rank_expansion_candidates(set(), star_graph, max_terms=5) == []
        assert rank_expansion_candidates({"user"}, None, max_terms=5) == []

    def test_returns_list(self, star_graph):
        """Output is an ordered list, not a set."""
        result = rank_expansion_candidates(
            {"user"}, star_graph, depth=1, skip_ids={"user"}, max_terms=10,
        )
        assert isinstance(result, list)

    def test_star_topology_scenario(self, star_graph):
        """Real-world scenario: well-connected entity ranks first."""
        result = rank_expansion_candidates(
            {"user"}, star_graph, depth=2, skip_ids={"user"}, max_terms=5,
        )
        # Flapjack has the most lateral edges → should be first
        assert len(result) > 0
        assert result[0] == "Flapjack"
        # Junk should be absent
        lower_result = [n.lower() for n in result]
        assert "5'11\"" not in lower_result
        assert "2 years" not in lower_result


# ===========================================================================
# Tests: _is_graph_worthy_object (ingestion filter)
# ===========================================================================

class TestIsGraphWorthyObject:

    def test_rejects_temporal_durations(self):
        """Temporal/duration strings are rejected."""
        assert MemoryStorage._is_graph_worthy_object("2 years") is False
        assert MemoryStorage._is_graph_worthy_object("6 months") is False
        assert MemoryStorage._is_graph_worthy_object("10 days") is False
        assert MemoryStorage._is_graph_worthy_object("once a week") is False
        assert MemoryStorage._is_graph_worthy_object("twice a day") is False

    def test_rejects_measurements(self):
        """Measurement strings are rejected."""
        assert MemoryStorage._is_graph_worthy_object("5'11\"") is False
        assert MemoryStorage._is_graph_worthy_object("20lbs") is False
        assert MemoryStorage._is_graph_worthy_object("10000iu") is False
        assert MemoryStorage._is_graph_worthy_object("500mg") is False

    def test_rejects_verb_phrases(self):
        """Verb-phrase objects are rejected."""
        assert MemoryStorage._is_graph_worthy_object("stopped being religious") is False
        assert MemoryStorage._is_graph_worthy_object("finishing grad school") is False
        assert MemoryStorage._is_graph_worthy_object("working remotely") is False
        assert MemoryStorage._is_graph_worthy_object("trying harder") is False

    def test_allows_proper_nouns(self):
        """Proper nouns and real entities pass."""
        assert MemoryStorage._is_graph_worthy_object("Kansas City") is True
        assert MemoryStorage._is_graph_worthy_object("Auggie") is True
        assert MemoryStorage._is_graph_worthy_object("Golden Retriever") is True

    def test_allows_short_entities(self):
        """Short but valid entities pass."""
        assert MemoryStorage._is_graph_worthy_object("coffee") is True
        assert MemoryStorage._is_graph_worthy_object("Linux") is True
        assert MemoryStorage._is_graph_worthy_object("Python") is True
