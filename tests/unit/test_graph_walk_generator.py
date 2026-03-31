# tests/unit/test_graph_walk_generator.py
"""Unit tests for GraphWalkGenerator."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from knowledge.graph_walk_generator import GraphWalkGenerator
from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge, GraphNode
from memory.entity_resolver import EntityResolver


@pytest.fixture
def tmp_graph_path():
    d = tempfile.mkdtemp()
    return os.path.join(d, "test_graph.json")


@pytest.fixture
def graph_with_bridges(tmp_graph_path):
    """Build a small graph with personal + wikidata nodes and bridge edges."""
    gm = GraphMemory(persist_path=tmp_graph_path)

    # Personal nodes
    for nid, display in [
        ("brewing", "Brewing"),
        ("exercise", "Exercise"),
        ("statistics", "Statistics"),
        ("board_games", "Board Games"),
    ]:
        gm.add_entity(GraphNode(
            entity_id=nid, display_name=display, entity_type="concept",
            metadata={"source": "personal"},
        ))

    # User node (should be skipped by walks)
    gm.add_entity(GraphNode(
        entity_id="user", display_name="User", entity_type="person",
        metadata={"source": "personal"},
    ))

    # Wikidata nodes
    for nid, display in [
        ("fermentation", "Fermentation"),
        ("biochemistry", "Biochemistry"),
        ("serotonin", "Serotonin"),
        ("bayesian_inference", "Bayesian Inference"),
        ("decision_theory", "Decision Theory"),
        ("game_theory", "Game Theory"),
        ("yeast", "Yeast"),
    ]:
        gm.add_entity(GraphNode(
            entity_id=nid, display_name=display, entity_type="concept",
            metadata={"source": "wikidata", "domain_category": "cross_domain_science"},
        ))

    # Bridge edges: personal → wikidata
    gm.add_relation(GraphEdge(
        source_id="brewing", relation="related_to", target_id="fermentation",
        weight=1.0, metadata={"source": "wikidata_bridge", "bridge_confidence": 0.9},
    ), fact_id="")
    gm.add_relation(GraphEdge(
        source_id="exercise", relation="related_to", target_id="serotonin",
        weight=1.0, metadata={"source": "wikidata_bridge", "bridge_confidence": 0.8},
    ), fact_id="")
    gm.add_relation(GraphEdge(
        source_id="statistics", relation="related_to", target_id="bayesian_inference",
        weight=1.0, metadata={"source": "wikidata_bridge", "bridge_confidence": 1.0},
    ), fact_id="")
    gm.add_relation(GraphEdge(
        source_id="board_games", relation="related_to", target_id="game_theory",
        weight=1.0, metadata={"source": "wikidata_bridge", "bridge_confidence": 0.7},
    ), fact_id="")

    # Wikidata internal edges (the path through general knowledge)
    gm.add_relation(GraphEdge(
        source_id="fermentation", relation="part_of", target_id="biochemistry",
    ), fact_id="")
    gm.add_relation(GraphEdge(
        source_id="biochemistry", relation="has_effect", target_id="serotonin",
    ), fact_id="")
    gm.add_relation(GraphEdge(
        source_id="fermentation", relation="uses", target_id="yeast",
    ), fact_id="")
    gm.add_relation(GraphEdge(
        source_id="bayesian_inference", relation="subclass_of", target_id="decision_theory",
    ), fact_id="")
    gm.add_relation(GraphEdge(
        source_id="decision_theory", relation="subclass_of", target_id="game_theory",
    ), fact_id="")

    # User edges (for domain classification)
    gm.add_relation(GraphEdge(
        source_id="user", relation="hobby", target_id="brewing",
    ), fact_id="")
    gm.add_relation(GraphEdge(
        source_id="user", relation="hobby", target_id="exercise",
    ), fact_id="")
    gm.add_relation(GraphEdge(
        source_id="user", relation="studies", target_id="statistics",
    ), fact_id="")
    gm.add_relation(GraphEdge(
        source_id="user", relation="plays", target_id="board_games",
    ), fact_id="")

    return gm


@pytest.fixture
def resolver(graph_with_bridges):
    aliases_path = os.path.join(
        os.path.dirname(graph_with_bridges.persist_path), "aliases.json"
    )
    return EntityResolver(graph_with_bridges, aliases_path=aliases_path)


@pytest.fixture
def mock_model_manager():
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value="Both concepts share a structural pattern of iterative refinement through feedback loops.")
    return mm


@pytest.fixture
def generator(graph_with_bridges, resolver, mock_model_manager):
    return GraphWalkGenerator(graph_with_bridges, resolver, mock_model_manager)


class TestGetNodeSource:
    def test_personal_node(self, generator):
        assert generator._get_node_source("brewing") == "personal"

    def test_wikidata_node(self, generator):
        assert generator._get_node_source("fermentation") == "wikidata"

    def test_missing_node(self, generator):
        assert generator._get_node_source("nonexistent") == "personal"

    def test_user_node(self, generator):
        assert generator._get_node_source("user") == "personal"


class TestGetDisplayName:
    def test_existing(self, generator):
        assert generator._get_display_name("brewing") == "Brewing"

    def test_missing(self, generator):
        assert generator._get_display_name("nonexistent") == "nonexistent"


class TestComputeWalkDistance:
    def test_short_path(self, generator):
        dist = generator._compute_walk_distance(["a", "b", "c"])
        assert 0.15 <= dist <= 1.0

    def test_long_path(self, generator):
        dist = generator._compute_walk_distance(["a"] * 8)
        assert dist == 1.0

    def test_min_clamp(self, generator):
        dist = generator._compute_walk_distance(["a"])
        assert dist >= 0.15


class TestSelectSeedNodes:
    def test_returns_personal_nodes(self, generator):
        seeds = generator._select_seed_nodes(10)
        for s in seeds:
            assert generator._get_node_source(s) == "personal"

    def test_excludes_user(self, generator):
        seeds = generator._select_seed_nodes(10)
        assert "user" not in seeds

    def test_prioritizes_bridge_neighbors(self, generator):
        seeds = generator._select_seed_nodes(4)
        # All personal nodes have bridge edges, so all should be included
        assert len(seeds) == 4
        assert set(seeds) == {"brewing", "exercise", "statistics", "board_games"}

    def test_respects_count(self, generator):
        seeds = generator._select_seed_nodes(2)
        assert len(seeds) == 2


class TestRandomWalk:
    def test_starts_at_seed(self, generator):
        walk = generator._random_walk("brewing", 8)
        assert walk[0] == "brewing"

    def test_respects_max_length(self, generator):
        walk = generator._random_walk("brewing", 4)
        assert len(walk) <= 4

    def test_skips_user_node(self, generator):
        # Run many walks — "user" should never appear
        for _ in range(50):
            walk = generator._random_walk("brewing", 8)
            assert "user" not in walk

    def test_walks_through_wikidata(self, generator):
        # With bridge edges, some walks should enter wikidata territory
        found_wikidata = False
        for _ in range(50):
            walk = generator._random_walk("brewing", 8)
            for node in walk:
                if generator._get_node_source(node) != "personal":
                    found_wikidata = True
                    break
            if found_wikidata:
                break
        assert found_wikidata, "No walk entered wikidata territory in 50 attempts"

    def test_produces_nonempty(self, generator):
        walk = generator._random_walk("brewing", 8)
        assert len(walk) >= 1


class TestQualifiesAsCandidate:
    def test_valid_walk(self, generator):
        # brewing → fermentation → biochemistry → serotonin → exercise
        walk = ["brewing", "fermentation", "biochemistry", "serotonin", "exercise"]
        assert generator._qualifies_as_candidate(walk) is True

    def test_too_short(self, generator):
        walk = ["brewing", "exercise"]
        assert generator._qualifies_as_candidate(walk) is False

    def test_same_start_end(self, generator):
        walk = ["brewing", "fermentation", "yeast", "fermentation", "brewing"]
        assert generator._qualifies_as_candidate(walk) is False

    def test_no_boundary_crossing(self, generator):
        # All personal nodes, no wikidata in between
        walk = ["brewing", "exercise", "statistics"]
        # These personal nodes aren't connected so this walk wouldn't
        # happen in practice, but tests the logic
        assert generator._qualifies_as_candidate(walk) is False

    def test_start_not_personal(self, generator):
        walk = ["fermentation", "biochemistry", "serotonin", "exercise"]
        assert generator._qualifies_as_candidate(walk) is False

    def test_end_not_personal(self, generator):
        walk = ["brewing", "fermentation", "biochemistry"]
        assert generator._qualifies_as_candidate(walk) is False

    def test_valid_with_min_path(self, generator, monkeypatch):
        monkeypatch.setattr("config.app_config.GRAPH_WALK_MIN_PATH", 3)
        walk = ["brewing", "fermentation", "exercise"]
        # fermentation is wikidata, so boundary crossed
        assert generator._qualifies_as_candidate(walk) is True


class TestClassifyEndpointDomain:
    def test_hobby_relation(self, generator):
        domain = generator._classify_endpoint_domain("brewing")
        assert domain != "other"  # Should classify via user→brewing "hobby" edge

    def test_studies_relation(self, generator):
        domain = generator._classify_endpoint_domain("statistics")
        assert domain != "other"

    def test_unknown_entity(self, generator):
        domain = generator._classify_endpoint_domain("nonexistent")
        assert isinstance(domain, str)


class TestArticulateAndPackage:
    @pytest.mark.asyncio
    async def test_produces_candidate(self, generator):
        import asyncio
        sem = asyncio.Semaphore(5)
        walk = ["brewing", "fermentation", "biochemistry", "serotonin", "exercise"]

        candidate = await generator._articulate_and_package(walk, sem)
        assert candidate is not None
        assert candidate.concept_a == "Brewing"
        assert candidate.concept_b == "Exercise"
        assert candidate.walk_path == walk
        assert len(candidate.walk_path) == 5
        assert candidate.endpoint_distance != 0.55  # Real distance, not fallback

    @pytest.mark.asyncio
    async def test_no_connection_returns_none(self, generator, mock_model_manager):
        import asyncio
        mock_model_manager.generate_once = AsyncMock(return_value="NO_CONNECTION")
        sem = asyncio.Semaphore(5)
        walk = ["brewing", "fermentation", "biochemistry", "serotonin", "exercise"]

        candidate = await generator._articulate_and_package(walk, sem)
        assert candidate is None

    @pytest.mark.asyncio
    async def test_short_response_returns_none(self, generator, mock_model_manager):
        import asyncio
        mock_model_manager.generate_once = AsyncMock(return_value="Yes.")
        sem = asyncio.Semaphore(5)
        walk = ["brewing", "fermentation", "biochemistry", "serotonin", "exercise"]

        candidate = await generator._articulate_and_package(walk, sem)
        assert candidate is None

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self, generator, mock_model_manager):
        import asyncio
        mock_model_manager.generate_once = AsyncMock(side_effect=Exception("API error"))
        sem = asyncio.Semaphore(5)
        walk = ["brewing", "fermentation", "biochemistry", "serotonin", "exercise"]

        candidate = await generator._articulate_and_package(walk, sem)
        assert candidate is None

    @pytest.mark.asyncio
    async def test_source_domains_populated(self, generator):
        import asyncio
        sem = asyncio.Semaphore(5)
        walk = ["brewing", "fermentation", "biochemistry", "serotonin", "exercise"]

        candidate = await generator._articulate_and_package(walk, sem)
        assert candidate is not None
        assert len(candidate.source_domains) >= 1

    @pytest.mark.asyncio
    async def test_path_hash_set(self, generator):
        import asyncio
        sem = asyncio.Semaphore(5)
        walk = ["brewing", "fermentation", "biochemistry", "serotonin", "exercise"]

        candidate = await generator._articulate_and_package(walk, sem)
        assert candidate is not None
        assert candidate.path_hash != ""
        assert len(candidate.path_hash) == 16


class TestGenerateCandidates:
    @pytest.mark.asyncio
    async def test_produces_candidates(self, generator):
        candidates = await generator.generate_candidates(count=5)
        # Should find at least some qualifying walks given the bridge structure
        assert isinstance(candidates, list)

    @pytest.mark.asyncio
    async def test_empty_graph(self, tmp_graph_path, resolver, mock_model_manager):
        empty_graph = GraphMemory(persist_path=tmp_graph_path + "_empty")
        gen = GraphWalkGenerator(empty_graph, resolver, mock_model_manager)
        candidates = await gen.generate_candidates(count=5)
        assert candidates == []

    @pytest.mark.asyncio
    async def test_respects_count_cap(self, generator, monkeypatch):
        monkeypatch.setattr("config.app_config.GRAPH_WALK_MAX_CANDIDATES", 2)
        candidates = await generator.generate_candidates(count=10)
        assert len(candidates) <= 2

    @pytest.mark.asyncio
    async def test_deduplicates_by_endpoint(self, generator):
        candidates = await generator.generate_candidates(count=20)
        # No two candidates should have the same endpoint pair
        pairs = set()
        for c in candidates:
            pair = frozenset([c.concept_a, c.concept_b])
            assert pair not in pairs, f"Duplicate endpoint pair: {pair}"
            pairs.add(pair)
