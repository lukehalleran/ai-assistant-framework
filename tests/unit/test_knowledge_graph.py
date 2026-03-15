# tests/unit/test_knowledge_graph.py
"""
Comprehensive tests for the knowledge graph system:
- GraphNode / GraphEdge models
- GraphMemory (CRUD, alias, traversal, persistence, weight strengthening)
- EntityResolver (exact match, alias, possessive learning, relation normalization)
- Integration hook (_ingest_fact_to_graph)
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from memory.graph_models import GraphNode, GraphEdge
from memory.graph_memory import GraphMemory
from memory.entity_resolver import (
    EntityResolver,
    normalize_relation,
    extract_possessive_aliases,
    _normalize_id,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def tmp_graph_path(tmp_path):
    return str(tmp_path / "test_graph.json")


@pytest.fixture
def graph(tmp_graph_path):
    return GraphMemory(persist_path=tmp_graph_path)


@pytest.fixture
def tmp_aliases_path(tmp_path):
    return str(tmp_path / "test_aliases.json")


@pytest.fixture
def resolver(graph, tmp_aliases_path):
    return EntityResolver(graph_memory=graph, aliases_path=tmp_aliases_path)


# =====================================================================
# GraphNode Model Tests
# =====================================================================

class TestGraphNode:
    def test_create_node(self):
        node = GraphNode(entity_id="spain", display_name="Spain", entity_type="place")
        assert node.entity_id == "spain"
        assert node.display_name == "Spain"
        assert node.entity_type == "place"
        assert node.mention_count == 0
        assert node.aliases == []

    def test_to_dict_roundtrip(self):
        node = GraphNode(
            entity_id="daemon",
            display_name="Daemon",
            entity_type="project",
            aliases=["the project", "my ai"],
            first_seen=datetime(2026, 1, 1),
            last_seen=datetime(2026, 3, 14),
            mention_count=42,
            metadata={"version": "1.0"},
        )
        d = node.to_dict()
        restored = GraphNode.from_dict("daemon", d)
        assert restored.entity_id == "daemon"
        assert restored.display_name == "Daemon"
        assert restored.entity_type == "project"
        assert restored.mention_count == 42
        assert "the project" in restored.aliases
        assert restored.metadata == {"version": "1.0"}

    def test_to_dict_no_timestamps(self):
        node = GraphNode(entity_id="test", display_name="Test")
        d = node.to_dict()
        assert "first_seen" not in d
        assert "last_seen" not in d

    def test_defaults(self):
        node = GraphNode(entity_id="x", display_name="X")
        assert node.entity_type == "other"
        assert node.metadata == {}
        assert node.first_seen is None


# =====================================================================
# GraphEdge Model Tests
# =====================================================================

class TestGraphEdge:
    def test_create_edge(self):
        edge = GraphEdge(source_id="user", relation="lives_in", target_id="spain")
        assert edge.source_id == "user"
        assert edge.relation == "lives_in"
        assert edge.target_id == "spain"
        assert edge.weight == 1.0
        assert edge.truth_score == 1.0

    def test_edge_key(self):
        edge = GraphEdge(source_id="a", relation="r", target_id="b")
        assert edge.edge_key() == "a|r|b"

    def test_to_natural_language(self):
        edge = GraphEdge(source_id="user", relation="lives_in", target_id="spain")
        text = edge.to_natural_language("Luke", "Spain")
        assert text == "Luke lives in Spain"

    def test_to_natural_language_defaults(self):
        edge = GraphEdge(source_id="user", relation="works_on", target_id="daemon")
        text = edge.to_natural_language()
        assert text == "user works on daemon"

    def test_to_dict_roundtrip(self):
        edge = GraphEdge(
            source_id="user",
            relation="created",
            target_id="daemon",
            weight=5.0,
            truth_score=0.9,
            first_seen=datetime(2026, 1, 1),
            last_seen=datetime(2026, 3, 14),
            source_fact_ids=["fact_1", "fact_2"],
            metadata={"context": "initial"},
        )
        d = edge.to_dict()
        restored = GraphEdge.from_dict(d)
        assert restored.source_id == "user"
        assert restored.relation == "created"
        assert restored.weight == 5.0
        assert restored.truth_score == 0.9
        assert "fact_1" in restored.source_fact_ids


# =====================================================================
# GraphMemory CRUD Tests
# =====================================================================

class TestGraphMemoryCRUD:
    def test_add_entity(self, graph):
        node = GraphNode(entity_id="Spain", display_name="Spain", entity_type="place")
        eid = graph.add_entity(node)
        assert eid == "spain"  # lowercased
        assert graph.node_count() == 1

    def test_add_entity_with_aliases(self, graph):
        node = GraphNode(
            entity_id="daemon",
            display_name="Daemon",
            aliases=["the project", "my AI"],
        )
        graph.add_entity(node)
        assert graph.resolve_entity("the project") == "daemon"
        assert graph.resolve_entity("my ai") == "daemon"
        assert graph.resolve_entity("daemon") == "daemon"

    def test_add_entity_updates_existing(self, graph):
        node1 = GraphNode(entity_id="spain", display_name="Spain", entity_type="place")
        graph.add_entity(node1)
        # Add again with new alias
        node2 = GraphNode(entity_id="spain", display_name="Spain", aliases=["espana"])
        graph.add_entity(node2)
        assert graph.node_count() == 1
        entity = graph.get_entity("spain")
        assert entity.mention_count == 2  # incremented
        assert "espana" in entity.aliases

    def test_get_entity_not_found(self, graph):
        assert graph.get_entity("nonexistent") is None

    def test_add_relation(self, graph):
        edge = GraphEdge(source_id="user", relation="lives_in", target_id="spain")
        graph.add_relation(edge)
        assert graph.edge_count() == 1
        # Auto-created stubs
        assert graph.node_count() == 2

    def test_add_relation_creates_stub_nodes(self, graph):
        edge = GraphEdge(source_id="alice", relation="works_at", target_id="acme")
        graph.add_relation(edge)
        alice = graph.get_entity("alice")
        assert alice is not None
        assert alice.display_name == "alice"  # stub uses raw id

    def test_duplicate_edge_strengthens_weight(self, graph):
        edge1 = GraphEdge(source_id="user", relation="likes", target_id="python")
        graph.add_relation(edge1)
        assert graph.edge_count() == 1

        edge2 = GraphEdge(source_id="user", relation="likes", target_id="python")
        graph.add_relation(edge2)
        # Still one edge, but weight increased
        assert graph.edge_count() == 1
        edges = graph.get_relations("user", direction="out")
        assert len(edges) == 1
        assert edges[0].weight == 2.0

    def test_duplicate_edge_accumulates_fact_ids(self, graph):
        edge = GraphEdge(source_id="a", relation="r", target_id="b")
        graph.add_relation(edge, fact_id="f1")
        graph.add_relation(edge, fact_id="f2")
        edges = graph.get_relations("a", direction="out")
        assert "f1" in edges[0].source_fact_ids
        assert "f2" in edges[0].source_fact_ids

    def test_get_relations_both(self, graph):
        graph.add_relation(GraphEdge(source_id="a", relation="r1", target_id="b"))
        graph.add_relation(GraphEdge(source_id="c", relation="r2", target_id="a"))
        edges = graph.get_relations("a", direction="both")
        assert len(edges) == 2

    def test_get_relations_out_only(self, graph):
        graph.add_relation(GraphEdge(source_id="a", relation="r1", target_id="b"))
        graph.add_relation(GraphEdge(source_id="c", relation="r2", target_id="a"))
        edges = graph.get_relations("a", direction="out")
        assert len(edges) == 1
        assert edges[0].target_id == "b"

    def test_get_relations_in_only(self, graph):
        graph.add_relation(GraphEdge(source_id="a", relation="r1", target_id="b"))
        graph.add_relation(GraphEdge(source_id="c", relation="r2", target_id="a"))
        edges = graph.get_relations("a", direction="in")
        assert len(edges) == 1
        assert edges[0].source_id == "c"

    def test_get_relations_empty(self, graph):
        assert graph.get_relations("nonexistent") == []


# =====================================================================
# GraphMemory Alias Tests
# =====================================================================

class TestGraphMemoryAlias:
    def test_resolve_entity(self, graph):
        node = GraphNode(entity_id="daemon", display_name="Daemon", aliases=["the project"])
        graph.add_entity(node)
        assert graph.resolve_entity("the project") == "daemon"
        assert graph.resolve_entity("DAEMON") == "daemon"
        assert graph.resolve_entity("unknown") is None

    def test_register_alias(self, graph):
        node = GraphNode(entity_id="flapjack", display_name="Flapjack")
        graph.add_entity(node)
        graph.register_alias("the cat", "flapjack")
        assert graph.resolve_entity("the cat") == "flapjack"

    def test_register_alias_stored_on_node(self, graph):
        node = GraphNode(entity_id="flapjack", display_name="Flapjack")
        graph.add_entity(node)
        graph.register_alias("lil guy", "flapjack")
        entity = graph.get_entity("flapjack")
        assert "lil guy" in entity.aliases

    def test_case_insensitive_alias(self, graph):
        node = GraphNode(entity_id="spain", display_name="Spain", aliases=["España"])
        graph.add_entity(node)
        assert graph.resolve_entity("españa") == "spain"
        assert graph.resolve_entity("ESPAÑA") == "spain"


# =====================================================================
# GraphMemory Traversal Tests
# =====================================================================

class TestGraphMemoryTraversal:
    def _build_graph(self, graph):
        """Build: user -> spain, user -> daemon, daemon -> python, spain -> europe"""
        graph.add_relation(GraphEdge(source_id="user", relation="wants_to_move_to", target_id="spain"))
        graph.add_relation(GraphEdge(source_id="user", relation="created", target_id="daemon"))
        graph.add_relation(GraphEdge(source_id="daemon", relation="uses", target_id="python"))
        graph.add_relation(GraphEdge(source_id="spain", relation="part_of", target_id="europe"))

    def test_neighbors_depth_1(self, graph):
        self._build_graph(graph)
        nbrs = graph.neighbors("user", depth=1)
        # Should include user, spain, daemon (direct neighbors)
        assert "user" in nbrs
        assert "spain" in nbrs
        assert "daemon" in nbrs
        # Should NOT include python or europe (depth > 1)
        assert "python" not in nbrs
        assert "europe" not in nbrs

    def test_neighbors_depth_2(self, graph):
        self._build_graph(graph)
        nbrs = graph.neighbors("user", depth=2)
        # Now python and europe should be reachable
        assert "python" in nbrs
        assert "europe" in nbrs

    def test_subgraph_around(self, graph):
        self._build_graph(graph)
        edges = graph.subgraph_around("user", depth=1)
        # depth=1 visits user + spain + daemon, so we see all their edges:
        # user->spain, user->daemon, spain->europe, daemon->python
        assert len(edges) == 4
        relations = {e.relation for e in edges}
        assert "wants_to_move_to" in relations
        assert "created" in relations
        assert "part_of" in relations
        assert "uses" in relations

    def test_subgraph_deduplicates(self, graph):
        self._build_graph(graph)
        edges = graph.subgraph_around("user", depth=2)
        edge_keys = [e.edge_key() for e in edges]
        assert len(edge_keys) == len(set(edge_keys))

    def test_shortest_path(self, graph):
        self._build_graph(graph)
        path = graph.shortest_path("user", "python")
        assert path == ["user", "daemon", "python"]

    def test_shortest_path_no_path(self, graph):
        graph.add_entity(GraphNode(entity_id="isolated", display_name="Isolated"))
        graph.add_relation(GraphEdge(source_id="a", relation="r", target_id="b"))
        assert graph.shortest_path("isolated", "a") == []

    def test_context_sentences(self, graph):
        self._build_graph(graph)
        sentences = graph.get_context_sentences("user", depth=1, max_sentences=10)
        # depth=1 visits user + spain + daemon = 4 edges total
        assert len(sentences) == 4
        # Check they're readable
        for s in sentences:
            assert isinstance(s, str)
            assert len(s) > 5

    def test_context_sentences_sorted_by_weight(self, graph):
        graph.add_relation(GraphEdge(source_id="user", relation="likes", target_id="a"))
        # Strengthen this edge
        for _ in range(5):
            graph.add_relation(GraphEdge(source_id="user", relation="likes", target_id="a"))
        graph.add_relation(GraphEdge(source_id="user", relation="knows", target_id="b"))

        sentences = graph.get_context_sentences("user", depth=1)
        # First sentence should be the stronger edge (likes a, weight=6)
        assert "likes" in sentences[0]

    def test_neighbors_nonexistent(self, graph):
        assert graph.neighbors("nonexistent") == {}

    def test_most_connected(self, graph):
        self._build_graph(graph)
        top = graph.most_connected(n=2)
        # user has most connections (2 out edges)
        assert top[0][0] == "user"


# =====================================================================
# GraphMemory Persistence Tests
# =====================================================================

class TestGraphMemoryPersistence:
    def test_save_load_roundtrip(self, tmp_graph_path):
        g1 = GraphMemory(persist_path=tmp_graph_path)
        node = GraphNode(entity_id="Spain", display_name="Spain", entity_type="place", aliases=["espana"])
        g1.add_entity(node)
        g1.add_relation(GraphEdge(source_id="user", relation="lives_in", target_id="spain"))
        g1.save()

        g2 = GraphMemory(persist_path=tmp_graph_path)
        assert g2.node_count() == 2  # user + spain
        assert g2.edge_count() == 1
        assert g2.resolve_entity("espana") == "spain"
        edges = g2.get_relations("user", direction="out")
        assert len(edges) == 1
        assert edges[0].relation == "lives_in"

    def test_save_only_when_dirty(self, tmp_graph_path):
        g = GraphMemory(persist_path=tmp_graph_path)
        g.save()
        # File shouldn't exist since nothing was added
        assert not os.path.exists(tmp_graph_path)

    def test_auto_save_threshold(self, tmp_graph_path):
        g = GraphMemory(persist_path=tmp_graph_path)
        g._auto_save_threshold = 3
        g.add_entity(GraphNode(entity_id="a", display_name="A"))
        g.add_entity(GraphNode(entity_id="b", display_name="B"))
        # Not yet saved
        assert g._dirty is True
        g.add_entity(GraphNode(entity_id="c", display_name="C"))
        # Should have auto-saved at threshold
        assert g._dirty is False
        assert os.path.exists(tmp_graph_path)

    def test_load_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        g = GraphMemory(persist_path=path)
        assert g.node_count() == 0

    def test_load_corrupted_file(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not valid json")
        g = GraphMemory(persist_path=path)
        assert g.node_count() == 0

    def test_persistence_preserves_edge_weights(self, tmp_graph_path):
        g1 = GraphMemory(persist_path=tmp_graph_path)
        edge = GraphEdge(source_id="a", relation="r", target_id="b")
        g1.add_relation(edge)
        g1.add_relation(edge)  # weight -> 2.0
        g1.save()

        g2 = GraphMemory(persist_path=tmp_graph_path)
        edges = g2.get_relations("a", direction="out")
        assert edges[0].weight == 2.0

    def test_json_readable(self, tmp_graph_path):
        g = GraphMemory(persist_path=tmp_graph_path)
        g.add_entity(GraphNode(entity_id="test", display_name="Test"))
        g.save()
        with open(tmp_graph_path) as f:
            data = json.load(f)
        assert "nodes" in data
        assert "edges" in data
        assert "test" in data["nodes"]


# =====================================================================
# Relation Normalization Tests
# =====================================================================

class TestRelationNormalization:
    def test_exact_canonical(self):
        assert normalize_relation("lives_in") == "lives_in"

    def test_synonym_mapping(self):
        assert normalize_relation("resides in") == "lives_in"
        assert normalize_relation("Located In") == "lives_in"
        assert normalize_relation("building") == "works_on"
        assert normalize_relation("employed at") == "works_at"

    def test_passthrough(self):
        assert normalize_relation("favorite_color") == "favorite_color"
        assert normalize_relation("unknown relation") == "unknown_relation"

    def test_spaces_replaced(self):
        assert normalize_relation("some  weird  relation") == "some_weird_relation"

    def test_case_insensitive(self):
        assert normalize_relation("LIVES IN") == "lives_in"
        assert normalize_relation("Works On") == "works_on"


# =====================================================================
# EntityResolver Tests
# =====================================================================

class TestEntityResolver:
    def test_resolve_known_entity(self, graph, resolver):
        graph.add_entity(GraphNode(entity_id="spain", display_name="Spain"))
        assert resolver.resolve("spain") == "spain"
        assert resolver.resolve("Spain") == "spain"

    def test_resolve_unknown(self, resolver):
        assert resolver.resolve("nonexistent") is None

    def test_resolve_or_create_existing(self, graph, resolver):
        graph.add_entity(GraphNode(entity_id="spain", display_name="Spain"))
        eid = resolver.resolve_or_create("spain")
        assert eid == "spain"
        assert graph.node_count() == 1  # no new node

    def test_resolve_or_create_new(self, graph, resolver):
        eid = resolver.resolve_or_create("Python", entity_type="concept", display_name="Python")
        assert eid == "python"
        assert graph.node_count() == 1
        entity = graph.get_entity("python")
        assert entity.display_name == "Python"
        assert entity.entity_type == "concept"

    def test_learn_alias(self, graph, resolver):
        graph.add_entity(GraphNode(entity_id="flapjack", display_name="Flapjack", entity_type="pet"))
        resolver.learn_alias("the cat", "flapjack")
        assert resolver.resolve("the cat") == "flapjack"

    def test_learn_possessive_alias(self, graph, resolver):
        graph.add_entity(GraphNode(entity_id="flapjack", display_name="Flapjack"))
        resolver.learn_alias("flapjack", "flapjack", context="I love my cat Flapjack")
        assert resolver.resolve("my cat") == "flapjack"

    def test_load_external_aliases(self, graph, tmp_aliases_path):
        # Write external aliases file
        aliases = {"spain": ["espana", "the country"]}
        with open(tmp_aliases_path, "w") as f:
            json.dump(aliases, f)

        graph.add_entity(GraphNode(entity_id="spain", display_name="Spain"))
        resolver = EntityResolver(graph_memory=graph, aliases_path=tmp_aliases_path)
        assert resolver.resolve("espana") == "spain"
        assert resolver.resolve("the country") == "spain"

    def test_save_external_aliases(self, graph, tmp_aliases_path):
        resolver = EntityResolver(graph_memory=graph, aliases_path=tmp_aliases_path)
        graph.add_entity(GraphNode(entity_id="daemon", display_name="Daemon", aliases=["the project"]))
        resolver.save_external_aliases()
        assert os.path.exists(tmp_aliases_path)
        with open(tmp_aliases_path) as f:
            data = json.load(f)
        assert "the project" in data.get("daemon", [])

    def test_normalize_id(self):
        assert _normalize_id("Georgia Tech") == "georgia_tech"
        assert _normalize_id("My Cat") == "my_cat"
        assert _normalize_id("  hello world  ") == "hello_world"
        assert _normalize_id("It's complicated!") == "its_complicated"


# =====================================================================
# Possessive Alias Extraction Tests
# =====================================================================

class TestPossessiveAliases:
    def test_extract_my_cat(self):
        aliases = extract_possessive_aliases("I love my cat Flapjack")
        assert len(aliases) == 1
        assert aliases[0] == ("my cat", "pet")

    def test_extract_my_boss(self):
        aliases = extract_possessive_aliases("my boss told me to do it")
        assert len(aliases) == 1
        assert aliases[0] == ("my boss", "person")

    def test_extract_my_project(self):
        aliases = extract_possessive_aliases("I'm working on my project")
        assert len(aliases) == 1
        assert aliases[0] == ("my project", "project")

    def test_no_possessives(self):
        aliases = extract_possessive_aliases("the weather is nice today")
        assert len(aliases) == 0

    def test_multiple_possessives(self):
        aliases = extract_possessive_aliases("my cat and my dog are friends")
        assert len(aliases) == 2


# =====================================================================
# GraphMemory Stats Tests
# =====================================================================

class TestGraphMemoryStats:
    def test_empty_stats(self, graph):
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
        assert graph.most_connected() == []

    def test_stats_after_additions(self, graph):
        graph.add_relation(GraphEdge(source_id="a", relation="r1", target_id="b"))
        graph.add_relation(GraphEdge(source_id="a", relation="r2", target_id="c"))
        assert graph.node_count() == 3
        assert graph.edge_count() == 2


# =====================================================================
# Integration: _ingest_fact_to_graph
# =====================================================================

class TestIngestFactToGraph:
    def test_ingest_creates_nodes_and_edge(self, graph, resolver):
        from memory.memory_storage import MemoryStorage
        storage = MemoryStorage(
            corpus_manager=MagicMock(),
            chroma_store=MagicMock(),
            fact_extractor=MagicMock(),
            graph_memory=graph,
            entity_resolver=resolver,
        )

        with patch("memory.memory_storage._get_graph_enabled", return_value=True), \
             patch("config.app_config.KNOWLEDGE_GRAPH_MIN_CONFIDENCE", 0.5):
            storage._ingest_fact_to_graph(
                subj="user", rel="lives in", obj="Spain",
                fact_id="fact_1", entity_type="", confidence=0.8,
            )

        assert graph.node_count() == 2
        assert graph.edge_count() == 1
        edges = graph.get_relations("user", direction="out")
        assert edges[0].relation == "lives_in"  # normalized

    def test_ingest_below_confidence_skipped(self, graph, resolver):
        from memory.memory_storage import MemoryStorage
        storage = MemoryStorage(
            corpus_manager=MagicMock(),
            chroma_store=MagicMock(),
            fact_extractor=MagicMock(),
            graph_memory=graph,
            entity_resolver=resolver,
        )

        with patch("memory.memory_storage._get_graph_enabled", return_value=True), \
             patch("config.app_config.KNOWLEDGE_GRAPH_MIN_CONFIDENCE", 0.7):
            storage._ingest_fact_to_graph(
                subj="user", rel="likes", obj="coffee",
                confidence=0.4,
            )

        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_ingest_empty_fields_skipped(self, graph, resolver):
        from memory.memory_storage import MemoryStorage
        storage = MemoryStorage(
            corpus_manager=MagicMock(),
            chroma_store=MagicMock(),
            fact_extractor=MagicMock(),
            graph_memory=graph,
            entity_resolver=resolver,
        )

        with patch("memory.memory_storage._get_graph_enabled", return_value=True):
            storage._ingest_fact_to_graph(subj="", rel="", obj="", confidence=0.8)

        assert graph.node_count() == 0

    def test_ingest_user_subject_type(self, graph, resolver):
        from memory.memory_storage import MemoryStorage
        storage = MemoryStorage(
            corpus_manager=MagicMock(),
            chroma_store=MagicMock(),
            fact_extractor=MagicMock(),
            graph_memory=graph,
            entity_resolver=resolver,
        )

        with patch("memory.memory_storage._get_graph_enabled", return_value=True), \
             patch("config.app_config.KNOWLEDGE_GRAPH_MIN_CONFIDENCE", 0.5):
            storage._ingest_fact_to_graph(
                subj="user", rel="owns", obj="Flapjack",
                entity_type="", confidence=0.8,
            )

        user_node = graph.get_entity("user")
        assert user_node is not None
        assert user_node.entity_type == "person"


# =====================================================================
# Context Sentences for Prompt Tests
# =====================================================================

class TestContextSentences:
    def test_max_sentences_limit(self, graph):
        # Create many edges
        for i in range(20):
            graph.add_relation(GraphEdge(source_id="hub", relation=f"rel_{i}", target_id=f"node_{i}"))
        sentences = graph.get_context_sentences("hub", depth=1, max_sentences=5)
        assert len(sentences) == 5

    def test_empty_graph_returns_empty(self, graph):
        sentences = graph.get_context_sentences("nonexistent")
        assert sentences == []

    def test_sentences_are_strings(self, graph):
        graph.add_entity(GraphNode(entity_id="user", display_name="Luke"))
        graph.add_entity(GraphNode(entity_id="spain", display_name="Spain"))
        graph.add_relation(GraphEdge(source_id="user", relation="wants_to_move_to", target_id="spain"))
        sentences = graph.get_context_sentences("user")
        assert len(sentences) == 1
        assert "Luke" in sentences[0]
        assert "Spain" in sentences[0]
        assert "wants to move to" in sentences[0]
