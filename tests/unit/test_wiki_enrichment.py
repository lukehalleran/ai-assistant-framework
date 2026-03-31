# tests/unit/test_wiki_enrichment.py
"""Unit tests for WikiGraphEnricher."""

import os
import tempfile

import pytest
import pytest_asyncio

from knowledge.wiki_enrichment import WikiGraphEnricher, _slugify
from knowledge.wiki_tracker import WikiArticleTracker
from memory.graph_memory import GraphMemory
from memory.graph_models import GraphNode
from memory.entity_resolver import EntityResolver


@pytest.fixture
def tmp_graph_path():
    d = tempfile.mkdtemp()
    return os.path.join(d, "test_graph.json")


@pytest.fixture
def graph(tmp_graph_path):
    gm = GraphMemory(persist_path=tmp_graph_path)
    # Pre-populate with personal entities
    gm.add_entity(GraphNode(
        entity_id="serotonin",
        display_name="Serotonin",
        entity_type="concept",
        metadata={"source": "personal"},
    ))
    gm.add_entity(GraphNode(
        entity_id="exercise",
        display_name="Exercise",
        entity_type="concept",
        metadata={"source": "personal"},
    ))
    gm.add_entity(GraphNode(
        entity_id="brewing",
        display_name="Brewing",
        entity_type="concept",
        metadata={"source": "personal"},
    ))
    return gm


@pytest.fixture
def resolver(graph):
    aliases_path = os.path.join(os.path.dirname(graph.persist_path), "aliases.json")
    er = EntityResolver(graph, aliases_path=aliases_path)
    return er


@pytest.fixture(autouse=True)
def reset_tracker():
    WikiArticleTracker._instance = None
    yield
    if WikiArticleTracker._instance:
        WikiArticleTracker._instance.clear()
    WikiArticleTracker._instance = None


class TestSlugify:
    def test_basic(self):
        assert _slugify("Quantum Mechanics") == "quantum_mechanics"

    def test_special_chars(self):
        assert _slugify("Serotonin (neurotransmitter)") == "serotonin_neurotransmitter"

    def test_extra_spaces(self):
        assert _slugify("  Beer  Brewing  ") == "beer_brewing"


class TestTitleExistsInGraph:
    def test_exact_slug_match(self, graph, resolver):
        enricher = WikiGraphEnricher(graph, resolver)
        assert enricher._title_exists_in_graph("Serotonin") is True

    def test_no_match(self, graph, resolver):
        enricher = WikiGraphEnricher(graph, resolver)
        assert enricher._title_exists_in_graph("Quantum Mechanics") is False

    def test_alias_match(self, graph, resolver):
        resolver.learn_alias("brain chemical", "serotonin", context="test")
        enricher = WikiGraphEnricher(graph, resolver)
        assert enricher._title_exists_in_graph("brain chemical") is True


class TestCreateWikiEntity:
    def test_creates_node(self, graph, resolver):
        enricher = WikiGraphEnricher(graph, resolver)
        eid = enricher._create_wiki_entity("Dopamine", "Dopamine is a neurotransmitter...")
        assert eid == "dopamine"
        assert graph.graph.has_node("dopamine")
        data = graph.graph.nodes["dopamine"]
        assert data["metadata"]["source"] == "wiki_retrieved"
        assert "text_preview" in data["metadata"]

    def test_registers_alias(self, graph, resolver):
        enricher = WikiGraphEnricher(graph, resolver)
        enricher._create_wiki_entity("Dopamine", "text")
        assert resolver.resolve("dopamine") == "dopamine"


class TestLinkToExistingEntities:
    def test_links_to_mentioned_entity(self, graph, resolver):
        enricher = WikiGraphEnricher(graph, resolver)
        enricher._create_wiki_entity("Brain Chemistry", "text")
        # Text that mentions "serotonin" which is an existing entity
        edges = enricher._link_to_existing_entities(
            "brain_chemistry",
            "Serotonin levels in the brain affect mood and exercise patterns",
        )
        assert edges >= 1  # Should find "serotonin" and possibly "exercise"

    def test_does_not_self_link(self, graph, resolver):
        enricher = WikiGraphEnricher(graph, resolver)
        eid = enricher._create_wiki_entity("Serotonin Research", "text")
        # Even though "serotonin" is in the text, it shouldn't link to itself
        # (well, serotonin_research != serotonin, so this tests the discard logic)
        edges = enricher._link_to_existing_entities(
            eid,
            "Serotonin research is important",
        )
        # Should link to existing "serotonin" node
        assert edges >= 1

    def test_does_not_link_to_user(self, graph, resolver):
        # Add "user" node
        graph.add_entity(GraphNode(
            entity_id="user",
            display_name="User",
            entity_type="person",
        ))
        enricher = WikiGraphEnricher(graph, resolver)
        enricher._create_wiki_entity("Test Article", "text")
        edges = enricher._link_to_existing_entities(
            "test_article", "The user should exercise more"
        )
        # Should NOT link to "user" (excluded by enricher)
        for edge in graph._edge_index.values():
            if edge.source_id == "test_article":
                assert edge.target_id != "user"

    def test_edge_properties(self, graph, resolver):
        enricher = WikiGraphEnricher(graph, resolver)
        enricher._create_wiki_entity("Mood Research", "text")
        enricher._link_to_existing_entities(
            "mood_research",
            "Serotonin affects mood",
            relation="mentioned_alongside",
            weight=0.5,
        )
        # Find the edge
        for edge in graph._edge_index.values():
            if edge.source_id == "mood_research" and edge.target_id == "serotonin":
                assert edge.relation == "mentioned_alongside"
                assert edge.weight == 0.5
                assert edge.metadata.get("source") == "wiki_enrichment"
                break
        else:
            pytest.fail("Expected edge not found")


class TestEnrichFromSession:
    @pytest.mark.asyncio
    async def test_empty_tracker(self, graph, resolver):
        enricher = WikiGraphEnricher(graph, resolver)
        result = await enricher.enrich_from_session()
        assert result["articles_added"] == 0
        assert result["relations_added"] == 0

    @pytest.mark.asyncio
    async def test_adds_new_articles(self, graph, resolver):
        tracker = WikiArticleTracker.get_instance()
        tracker.track("Dopamine", "Dopamine is a neurotransmitter involved in serotonin pathways. " * 10)
        tracker.track("Fermentation", "Fermentation is used in brewing beer and other processes. " * 10)

        enricher = WikiGraphEnricher(graph, resolver)
        result = await enricher.enrich_from_session()

        assert result["articles_added"] == 2
        assert graph.graph.has_node("dopamine")
        assert graph.graph.has_node("fermentation")
        # Tracker should be cleared after enrichment
        assert tracker.count == 0

    @pytest.mark.asyncio
    async def test_skips_existing_articles(self, graph, resolver):
        tracker = WikiArticleTracker.get_instance()
        # "Serotonin" already exists in graph
        tracker.track("Serotonin", "Serotonin is a neurotransmitter" * 20)

        enricher = WikiGraphEnricher(graph, resolver)
        result = await enricher.enrich_from_session()

        assert result["articles_added"] == 0
        assert result["skipped"] == 1

    @pytest.mark.asyncio
    async def test_skips_short_text(self, graph, resolver):
        tracker = WikiArticleTracker.get_instance()
        tracker.track("Short Article", "Too short")

        enricher = WikiGraphEnricher(graph, resolver)
        result = await enricher.enrich_from_session()

        assert result["articles_added"] == 0
        assert result["skipped"] == 1

    @pytest.mark.asyncio
    async def test_respects_session_cap(self, graph, resolver, monkeypatch):
        monkeypatch.setattr("config.app_config.WIKI_ENRICHMENT_MAX_PER_SESSION", 2)

        tracker = WikiArticleTracker.get_instance()
        for i in range(5):
            tracker.track(f"Article {i}", f"{'Content ' * 50} number {i}")

        enricher = WikiGraphEnricher(graph, resolver)
        result = await enricher.enrich_from_session()

        assert result["articles_added"] == 2

    @pytest.mark.asyncio
    async def test_creates_bridge_edges(self, graph, resolver):
        tracker = WikiArticleTracker.get_instance()
        tracker.track(
            "Neurotransmitter Release",
            "Serotonin release during exercise is a key mechanism in mood regulation. " * 10
        )

        enricher = WikiGraphEnricher(graph, resolver)
        result = await enricher.enrich_from_session()

        assert result["articles_added"] == 1
        assert result["relations_added"] >= 1

        # Verify the new node has wiki_retrieved source
        node_data = graph.graph.nodes.get("neurotransmitter_release", {})
        assert node_data.get("metadata", {}).get("source") == "wiki_retrieved"

        # Verify bridge edges exist (wiki_retrieved → personal)
        assert graph.count_bridge_edges() >= 1
