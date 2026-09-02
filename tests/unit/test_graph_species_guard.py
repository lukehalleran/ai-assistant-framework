"""Species-contradiction guard for graph edge ingestion (2026-08-28).

2026-08-18 incident: the shutdown LLM extractor invented ``user | has_dog |
Daisy`` (and ``user | has_dog | Biscuit``) from an excerpt that never
mentions a dog ("I have been playing with Biscuit and Daisy for hours a
day"). Both graph nodes already carried curated ``species: cat`` metadata
and older ``has_cat`` edges — and the wrong ``has_dog`` edges fed a junk
proactive insight.

Doctrine: curated node metadata is ground truth; a single extraction naming
a conflicting species is an extraction error, not new information. The
guard fires ONLY when the node explicitly declares a species (under-fires
by design) and blocks the EDGE only — fact storage is unaffected.

- ``relation_species_conflict`` lives in memory/graph_utils.py.
- The call site is ``memory_storage._ingest_fact_to_graph`` (edge path).
"""

import pytest

from memory.graph_utils import relation_species_conflict


class TestRelationSpeciesConflict:
    """Pure-helper behavior."""

    def test_live_reproduction_has_dog_vs_cat_node(self):
        # exact live metadata shape — descriptive value, not a bare word
        assert relation_species_conflict(
            "has_dog", {"species": "cat, black, big golden eyes"}) is True

    def test_matching_species_passes(self):
        assert relation_species_conflict(
            "has_cat", {"species": "cat, black, big golden eyes"}) is False
        assert relation_species_conflict("has_dog", {"species": "dog"}) is False

    def test_no_metadata_never_blocks(self):
        assert relation_species_conflict("has_dog", None) is False
        assert relation_species_conflict("has_dog", {}) is False

    def test_no_species_key_never_blocks(self):
        assert relation_species_conflict(
            "has_dog", {"color": "black", "born": "2019"}) is False

    def test_non_species_relations_never_block(self):
        md = {"species": "cat"}
        for rel in ("likes", "has_appointment", "plays_with", "cares_for",
                    "talked_about", "has"):
            assert relation_species_conflict(rel, md) is False, rel

    def test_juvenile_forms_collapse_to_adult_species(self):
        assert relation_species_conflict(
            "adopted_kitten", {"species": "dog"}) is True
        assert relation_species_conflict(
            "adopted_kitten", {"species": "cat"}) is False
        assert relation_species_conflict(
            "has_puppy", {"species": "cat"}) is True

    def test_plural_relation_form(self):
        assert relation_species_conflict("has_dogs", {"species": "cat"}) is True
        assert relation_species_conflict("has_cats", {"species": "cat"}) is False

    def test_species_word_must_be_token_bounded_in_relation(self):
        # "cat" ⊂ "catalog" must NOT fire — the 'solve'⊂"resolution" lesson
        assert relation_species_conflict(
            "has_catalog", {"species": "dog"}) is False
        assert relation_species_conflict(
            "dogmatic_about", {"species": "cat"}) is False

    def test_declared_value_containment_is_word_bounded(self):
        # declared "caterpillar" must not satisfy a claimed "cat" —
        # containment against the descriptive value is word-bounded too
        assert relation_species_conflict(
            "has_cat", {"species": "caterpillar"}) is True

    def test_multi_word_species(self):
        assert relation_species_conflict(
            "has_guinea_pig", {"species": "cat"}) is True
        assert relation_species_conflict(
            "has_guinea_pig", {"species": "guinea pig"}) is False


@pytest.fixture
def storage(tmp_path):
    """Deployed MemoryStorage._ingest_fact_to_graph against a real (cold,
    tmp-path) GraphMemory + EntityResolver — no chroma, no LLM."""
    from memory.graph_memory import GraphMemory
    from memory.entity_resolver import EntityResolver
    from memory.memory_storage import MemoryStorage

    gm = GraphMemory(persist_path=str(tmp_path / "kg.json"))
    resolver = EntityResolver(gm, aliases_path=str(tmp_path / "aliases.json"))
    st = object.__new__(MemoryStorage)
    st.graph_memory = gm
    st.entity_resolver = resolver
    return st


def _seed_cat_node(storage, entity_id, display):
    from memory.graph_models import GraphNode
    storage.graph_memory.add_entity(GraphNode(
        entity_id=entity_id, display_name=display, entity_type="pet",
        metadata={"species": "cat, black, big golden eyes"},
    ))


class TestIngestSpeciesGuard:
    """Drive the deployed _ingest_fact_to_graph (the call site, not just
    the helper — dead-wiring lesson)."""

    def test_live_case_wrong_species_edge_blocked(self, storage):
        _seed_cat_node(storage, "daisy", "Daisy")
        storage._ingest_fact_to_graph(
            "user", "has_dog", "Daisy",
            fact_id="fact_test", confidence=0.9)
        edges = storage.graph_memory.get_relations("user")
        assert not any(e.relation == "has_dog" and e.target_id == "daisy"
                       for e in edges)

    def test_matching_species_edge_allowed(self, storage):
        _seed_cat_node(storage, "daisy", "Daisy")
        storage._ingest_fact_to_graph(
            "user", "has_cat", "Daisy",
            fact_id="fact_test", confidence=0.9)
        edges = storage.graph_memory.get_relations("user")
        assert any(e.relation == "has_cat" and e.target_id == "daisy"
                   for e in edges)

    def test_node_without_species_metadata_unaffected(self, storage):
        from memory.graph_models import GraphNode
        storage.graph_memory.add_entity(GraphNode(
            entity_id="biscuit", display_name="Biscuit",
            entity_type="pet"))
        storage._ingest_fact_to_graph(
            "user", "has_dog", "Biscuit",
            fact_id="fact_test", confidence=0.9)
        edges = storage.graph_memory.get_relations("user")
        assert any(e.relation == "has_dog" and e.target_id == "biscuit"
                   for e in edges)

    def test_non_species_relation_unaffected(self, storage):
        _seed_cat_node(storage, "daisy", "Daisy")
        storage._ingest_fact_to_graph(
            "user", "plays_with", "Daisy",
            fact_id="fact_test", confidence=0.9)
        edges = storage.graph_memory.get_relations("user")
        assert any(e.target_id == "daisy" for e in edges)

    def test_fresh_object_node_never_blocked(self, storage):
        # brand-new node has no metadata — first-mention edges always land
        storage._ingest_fact_to_graph(
            "user", "has_dog", "Rex",
            fact_id="fact_test", confidence=0.9)
        edges = storage.graph_memory.get_relations("user")
        assert any(e.relation == "has_dog" for e in edges)
