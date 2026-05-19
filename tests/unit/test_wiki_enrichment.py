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
    from memory.graph_models import GraphEdge
    gm = GraphMemory(persist_path=tmp_graph_path)
    # Pre-populate with personal entities
    gm.add_entity(GraphNode(
        entity_id="user",
        display_name="User",
        entity_type="person",
    ))
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
    # Add conversation edges so these count as conversation entities
    # (wiki enrichment only links entities that have non-wiki edges)
    for eid in ("serotonin", "exercise", "brewing"):
        gm.add_relation(GraphEdge(
            source_id="user", relation="discussed", target_id=eid,
            weight=1.0, truth_score=0.8,
            metadata={"source": "conversation"},
        ), fact_id="")
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

    def test_does_not_link_non_conversation_wiki_node(self, graph, resolver):
        """A wiki-created node not in conversation graph doesn't get linked."""
        enricher = WikiGraphEnricher(graph, resolver)
        eid = enricher._create_wiki_entity("Serotonin Research", "text")
        # serotonin_research is a wiki node, not a conversation entity
        # Even though text mentions "serotonin", the wiki node shouldn't
        # get edges because it's not in the conversation graph.
        edges = enricher._link_to_existing_entities(
            eid,
            "Serotonin research is important",
        )
        # Only one conversation entity mentioned (serotonin) — need 2+ for bridges
        assert edges == 0

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
        """Bridge edges between conversation entities have correct properties."""
        enricher = WikiGraphEnricher(graph, resolver)
        convo_entities = enricher._get_conversation_entities()
        # Text mentions serotonin + exercise (both conversation entities)
        edges = enricher._link_conversation_entities(
            None,
            "Serotonin levels improve with exercise and physical activity",
            convo_entities,
            relation="mentioned_alongside",
            weight=0.5,
        )
        assert edges >= 1
        # Find a bridge edge
        for edge in graph._edge_index.values():
            if edge.relation == "mentioned_alongside" and edge.metadata.get("source") == "wiki_enrichment":
                assert edge.weight == 0.5
                break
        else:
            pytest.fail("Expected wiki_enrichment edge not found")


class TestEnrichFromSession:
    @pytest.mark.asyncio
    async def test_empty_tracker(self, graph, resolver):
        enricher = WikiGraphEnricher(graph, resolver)
        result = await enricher.enrich_from_session()
        assert result["articles_added"] == 0
        assert result["relations_added"] == 0

    @pytest.mark.asyncio
    async def test_adds_new_articles_only_if_conversation_entity(self, graph, resolver):
        """Wiki articles only create nodes if title matches a conversation entity."""
        tracker = WikiArticleTracker.get_instance()
        # "Dopamine" is NOT a conversation entity — should NOT create a node
        tracker.track("Dopamine", "Dopamine is a neurotransmitter involved in serotonin pathways. " * 10)
        # "Fermentation" is NOT a conversation entity — should NOT create a node
        tracker.track("Fermentation", "Fermentation is used in brewing beer and other processes. " * 10)
        # "Serotonin" IS a conversation entity but already exists — should skip
        tracker.track("Serotonin", "Serotonin is a key neurotransmitter for mood. " * 10)

        enricher = WikiGraphEnricher(graph, resolver)
        result = await enricher.enrich_from_session()

        # No new nodes (dopamine/fermentation not in convo, serotonin already exists)
        assert result["articles_added"] == 0
        assert not graph.graph.has_node("dopamine")
        assert not graph.graph.has_node("fermentation")
        # But bridge edges may have been created between conversation entities
        # mentioned in the article text (serotonin ↔ exercise from the dopamine article)
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
        """Session cap limits articles_added for titles matching conversation entities."""
        from memory.graph_models import GraphEdge
        monkeypatch.setattr("config.app_config.WIKI_ENRICHMENT_MAX_PER_SESSION", 2)

        # Add conversation entities that match article titles
        for i in range(5):
            eid = f"article_{i}"
            graph.add_entity(GraphNode(entity_id=eid, display_name=f"Article {i}", entity_type="concept"))
            graph.add_relation(GraphEdge(
                source_id="user", relation="discussed", target_id=eid,
                weight=1.0, truth_score=0.8, metadata={"source": "conversation"},
            ), fact_id="")

        tracker = WikiArticleTracker.get_instance()
        for i in range(5):
            tracker.track(f"Article {i}", f"{'Content ' * 50} about serotonin and exercise number {i}")

        enricher = WikiGraphEnricher(graph, resolver)
        result = await enricher.enrich_from_session()

        # Cap should limit to 2 even though 5 match
        assert result["articles_added"] <= 2

    @pytest.mark.asyncio
    async def test_creates_bridge_edges_between_conversation_entities(self, graph, resolver):
        """Wiki article mentioning multiple conversation entities creates bridges between them."""
        tracker = WikiArticleTracker.get_instance()
        # Article title is NOT a conversation entity, but text mentions serotonin + exercise
        tracker.track(
            "Neurotransmitter Release",
            "Serotonin release during exercise is a key mechanism in mood regulation. " * 10
        )

        enricher = WikiGraphEnricher(graph, resolver)
        result = await enricher.enrich_from_session()

        # No new node (title not in convo entities), but bridge edges created
        assert result["articles_added"] == 0
        assert result["relations_added"] >= 1

        # Verify bridge edge exists between conversation entities
        has_bridge = False
        for edge in graph._edge_index.values():
            if edge.metadata.get("source") == "wiki_enrichment":
                has_bridge = True
                break
        assert has_bridge, "Expected at least one wiki_enrichment bridge edge"
