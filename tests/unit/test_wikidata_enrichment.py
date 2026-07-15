# tests/unit/test_wikidata_enrichment.py
"""
Tests for knowledge/wikidata_enrichment.py — anchored Wikidata typed-edge
enrichment. Uses a synthetic cache dict injected via the `cache` param and a
REAL GraphMemory + EntityResolver (no mocked graph — dead wiring can't hide).
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from memory.graph_memory import GraphMemory
from memory.graph_models import GraphNode
from memory.entity_resolver import EntityResolver
from knowledge.wikidata_enrichment import WikidataGraphEnricher


@pytest.fixture
def graph(tmp_path):
    return GraphMemory(persist_path=str(tmp_path / "graph.json"))


@pytest.fixture
def resolver(graph, tmp_path):
    return EntityResolver(graph_memory=graph, aliases_path=str(tmp_path / "aliases.json"))


def _personal(graph, entity_id, display_name=None, etype="concept", aliases=None):
    graph.add_entity(GraphNode(
        entity_id=entity_id,
        display_name=display_name or entity_id,
        entity_type=etype,
        aliases=aliases or [],
        metadata={"source": "conversation"},
    ))


CACHE = {
    "entities": {
        "Q1": {"qid": "Q1", "label": "creatine", "description": "supplement",
               "aliases": ["creatine monohydrate"], "domain_category": "fitness_exercise"},
        "Q2": {"qid": "Q2", "label": "dietary supplement", "description": "",
               "aliases": [], "domain_category": "health_medical"},
        "Q3": {"qid": "Q3", "label": "skeletal muscle", "description": "",
               "aliases": [], "domain_category": "health_medical"},
        "Q4": {"qid": "Q4", "label": "python", "description": "programming language",
               "aliases": ["python programming language"], "domain_category": "computer_science"},
        "Q5": {"qid": "Q5", "label": "programming language", "description": "",
               "aliases": [], "domain_category": "computer_science"},
    },
    "relations": [
        {"source_qid": "Q1", "property_id": "P31", "relation_label": "instance_of",
         "target_qid": "Q2"},
        {"source_qid": "Q1", "property_id": "P2283", "relation_label": "uses",
         "target_qid": "Q3"},
        # Non-whitelisted relation — must be ignored
        {"source_qid": "Q1", "property_id": "P999", "relation_label": "said_to_be_same_as",
         "target_qid": "Q3"},
        {"source_qid": "Q4", "property_id": "P31", "relation_label": "instance_of",
         "target_qid": "Q5"},
        {"source_qid": "Q4", "property_id": "P361", "relation_label": "part_of",
         "target_qid": "Q5"},
    ],
}

_CFG = "config.app_config"


def _run(graph, resolver, cache=CACHE, **overrides):
    defaults = {
        "WIKIDATA_ENRICHMENT_RELATION_WHITELIST":
            ["instance_of", "subclass_of", "part_of", "has_part", "uses"],
        "WIKIDATA_ENRICHMENT_MAX_EDGES_PER_ENTITY": 5,
        "WIKIDATA_ENRICHMENT_MAX_NEW_NODES": 25,
        "WIKIDATA_ENRICHMENT_MAX_EDGES_PER_RUN": 50,
    }
    defaults.update(overrides)
    patches = [patch(f"{_CFG}.{k}", v) for k, v in defaults.items()]
    for p in patches:
        p.start()
    try:
        enricher = WikidataGraphEnricher(graph, resolver, cache=cache)
        return enricher.enrich()
    finally:
        for p in patches:
            p.stop()


class TestMatching:
    def test_personal_entity_gets_typed_edges(self, graph, resolver):
        _personal(graph, "creatine")
        stats = _run(graph, resolver)

        assert stats["matched"] == 1
        assert stats["edges_added"] == 2  # instance_of + uses; same_as ignored
        rels = {e.relation for e in graph.get_relations("creatine", direction="out")}
        assert rels == {"instance_of", "uses"}
        assert graph.graph.has_node("dietary_supplement")
        assert graph.graph.has_node("skeletal_muscle")

    def test_match_via_alias(self, graph, resolver):
        _personal(graph, "my_supplement", display_name="Creatine Monohydrate")
        stats = _run(graph, resolver)
        assert stats["matched"] == 1

    def test_unmatched_entity_untouched(self, graph, resolver):
        _personal(graph, "biscuit", etype="animal")
        stats = _run(graph, resolver)
        assert stats["matched"] == 0
        assert stats["edges_added"] == 0
        assert graph.node_count() == 1

    def test_user_node_never_enriched(self, graph, resolver):
        # Even if "user" somehow matched a cache label, it must be skipped
        graph.add_entity(GraphNode(entity_id="user", display_name="python",
                                   entity_type="person"))
        stats = _run(graph, resolver)
        assert stats["matched"] == 0

    def test_wiki_and_wikidata_nodes_not_expanded(self, graph, resolver):
        """1-hop only: nodes created BY enrichment (or wiki retrieval) are
        never used as anchors, even across runs."""
        graph.add_entity(GraphNode(
            entity_id="python", display_name="python", entity_type="concept",
            metadata={"source": "wiki_retrieved"},
        ))
        stats = _run(graph, resolver)
        assert stats["matched"] == 0
        assert stats["edges_added"] == 0


class TestIdempotency:
    def test_second_run_skips_enriched(self, graph, resolver):
        _personal(graph, "creatine")
        first = _run(graph, resolver)
        assert first["matched"] == 1
        nodes_after_first = graph.node_count()
        edges_after_first = graph.edge_count()

        second = _run(graph, resolver)
        assert second["matched"] == 0
        assert second["skipped_existing"] == 1
        assert graph.node_count() == nodes_after_first
        assert graph.edge_count() == edges_after_first

    def test_qid_stamped_on_personal_node(self, graph, resolver):
        _personal(graph, "creatine")
        _run(graph, resolver)
        node = graph.get_entity("creatine")
        assert node.metadata.get("wikidata_qid") == "Q1"
        # Source marker preserved — still a personal node
        assert node.metadata.get("source") == "conversation"


class TestCaps:
    def test_per_entity_edge_cap(self, graph, resolver):
        _personal(graph, "creatine")
        stats = _run(graph, resolver, WIKIDATA_ENRICHMENT_MAX_EDGES_PER_ENTITY=1)
        assert stats["edges_added"] == 1

    def test_per_run_edge_cap(self, graph, resolver):
        _personal(graph, "creatine")
        _personal(graph, "python")
        stats = _run(graph, resolver, WIKIDATA_ENRICHMENT_MAX_EDGES_PER_RUN=2)
        assert stats["edges_added"] == 2

    def test_new_node_cap_blocks_creation(self, graph, resolver):
        _personal(graph, "creatine")
        stats = _run(graph, resolver, WIKIDATA_ENRICHMENT_MAX_NEW_NODES=0)
        assert stats["nodes_created"] == 0
        assert stats["edges_added"] == 0  # counterparts couldn't be created

    def test_node_cap_still_links_existing_counterparts(self, graph, resolver):
        # Counterpart already in graph -> edge added even with node cap 0
        graph.add_entity(GraphNode(
            entity_id="dietary_supplement", display_name="dietary supplement",
            entity_type="concept", metadata={"source": "wikidata_enrichment"},
        ))
        _personal(graph, "creatine")
        stats = _run(graph, resolver, WIKIDATA_ENRICHMENT_MAX_NEW_NODES=0)
        assert stats["edges_added"] == 1
        edges = graph.get_relations("creatine", direction="out")
        assert edges[0].target_id == "dietary_supplement"


class TestCounterpartNodes:
    def test_created_node_metadata(self, graph, resolver):
        _personal(graph, "creatine")
        _run(graph, resolver)
        node = graph.get_entity("dietary_supplement")
        assert node.metadata["source"] == "wikidata_enrichment"
        assert node.metadata["wikidata_qid"] == "Q2"
        assert node.entity_type == "concept"

    def test_edge_direction_preserved(self, graph, resolver):
        """creatine instance_of dietary_supplement — not the reverse."""
        _personal(graph, "creatine")
        _run(graph, resolver)
        out = graph.get_relations("creatine", direction="out")
        assert {e.target_id for e in out} == {"dietary_supplement", "skeletal_muscle"}
        inc = graph.get_relations("creatine", direction="in")
        assert inc == []

    def test_reverse_compositional_kept_taxonomic_skipped(self, graph, resolver):
        """Personal entity on the TARGET side: compositional relations
        (part_of) come in; taxonomic ones (instance_of) are skipped —
        reverse instance_of enumerates category members (junk fan-in)."""
        _personal(graph, "programming_language", display_name="programming language")
        stats = _run(graph, resolver)
        assert stats["matched"] == 1
        inc = graph.get_relations("programming_language", direction="in")
        assert len(inc) == 1
        assert inc[0].source_id == "python"
        assert inc[0].relation == "part_of"  # instance_of reverse skipped


class TestDegenerateInputs:
    def test_missing_cache(self, graph, resolver, tmp_path):
        _personal(graph, "creatine")
        enricher = WikidataGraphEnricher(
            graph, resolver, cache_path=str(tmp_path / "nope.json"))
        stats = enricher.enrich()
        assert stats == {"matched": 0, "edges_added": 0, "nodes_created": 0,
                         "skipped_existing": 0}

    def test_empty_cache(self, graph, resolver):
        _personal(graph, "creatine")
        stats = _run(graph, resolver, cache={"entities": {}, "relations": []})
        assert stats["matched"] == 0
