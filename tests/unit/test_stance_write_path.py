"""
Write-time stance wiring (Phase B2, 2026-08-23).

Covers:
  * llm_fact_extractor._normalize_triple: stance field read tolerantly, the
    DETERMINISTIC classifier overriding on lexicon hits, referent scoping.
  * fact_extractor._clean_triple: evaluative pronoun subjects re-scope to a
    user-owned subject instead of being dropped as stop-subjects; the scoped
    subject NEVER binds to a named entity.
  * memory_storage._ingest_fact_to_graph: stance/capture_tone ride into
    GraphEdge.metadata; role subjects become verbatim entity_type="role" nodes
    bypassing the alias resolver even when a possessive alias points at a
    registered person (the casey/she incident class).
  * shutdown_processor._capture_tone_for_triple: corpus-join tone mapping.
"""

import pytest

from memory.entity_resolver import EntityResolver
from memory.fact_extractor import _clean_triple
from memory.graph_memory import GraphMemory
from memory.graph_models import GraphNode
from memory.llm_fact_extractor import _normalize_triple
from memory.memory_storage import MemoryStorage
from memory.shutdown_processor import ShutdownProcessor


class TestLLMNormalizeStance:
    def test_appraisal_overrides_llm_objective(self):
        t = _normalize_triple({"subject": "casey", "relation": "is",
                               "object": "evil", "stance": "objective"})
        assert t is not None
        assert t["stance"] == "appraisal"  # deterministic lexicon hit wins

    def test_llm_stance_fills_gap_when_deterministic_objective(self):
        t = _normalize_triple({"subject": "user", "relation": "lives_in",
                               "object": "chicago", "stance": "reported"})
        assert t["stance"] == "reported"

    def test_invalid_llm_stance_ignored(self):
        t = _normalize_triple({"subject": "user", "relation": "lives_in",
                               "object": "chicago", "stance": "vibes"})
        assert t["stance"] == "objective"

    def test_missing_stance_defaults_deterministic(self):
        t = _normalize_triple({"subject": "user", "relation": "lives_in",
                               "object": "chicago"})
        assert t["stance"] == "objective"

    def test_pronoun_subject_evaluative_rescopes(self):
        t = _normalize_triple({"subject": "she", "relation": "is",
                               "object": "abusive"})
        assert t is not None
        assert t["subject"] == "user's unnamed referent"
        assert t["stance"] == "appraisal"
        assert t["fact_scope"] == "entity"

    def test_named_subject_not_rescoped(self):
        t = _normalize_triple({"subject": "casey", "relation": "is",
                               "object": "evil"})
        assert t["subject"] == "casey"


class TestRegexCleanTripleScoping:
    def test_evaluative_pronoun_subject_preserved_user_scoped(self):
        out = _clean_triple("she", "is", "abusive")
        assert out is not None
        assert out[0] == "user's unnamed referent"

    def test_non_evaluative_pronoun_subject_still_dropped(self):
        assert _clean_triple("she", "works at", "deloitte") is None

    def test_role_subject_evaluative_rescopes(self):
        out = _clean_triple("my last partner", "was", "toxic")
        assert out is not None
        assert out[0] == "user's last partner"

    def test_named_subject_untouched(self):
        out = _clean_triple("jordan", "works_at", "deloitte")
        assert out == ("jordan", "works_at", "deloitte")


@pytest.fixture
def graph_env(tmp_path):
    gm = GraphMemory(persist_path=str(tmp_path / "graph.json"))
    resolver = EntityResolver(gm, aliases_path="")
    # Register casey WITH the possessive alias the fuzzy-bind hazard needs.
    gm.add_entity(GraphNode(entity_id="casey", display_name="Casey",
                            entity_type="person", aliases=["user's last partner"]))
    ms = MemoryStorage.__new__(MemoryStorage)
    ms.graph_memory = gm
    ms.entity_resolver = resolver
    return ms, gm, resolver


class TestGraphIngestStance:
    def test_edge_carries_stance_metadata(self, graph_env):
        ms, gm, resolver = graph_env
        gm.add_entity(GraphNode(entity_id="evil", display_name="evil"))
        ms._ingest_fact_to_graph(
            subj="casey", rel="is", obj="evil", fact_id="f1",
            confidence=0.9, stance="appraisal", capture_tone="elevated",
        )
        edges = gm.get_relations("casey")
        edge = next(e for e in edges if e.target_id == "evil")
        assert edge.metadata.get("stance") == "appraisal"
        assert edge.metadata.get("capture_tone") == "elevated"

    def test_role_subject_never_binds_to_registered_person(self, graph_env):
        ms, gm, resolver = graph_env
        # sanity: the resolver WOULD bind this alias to casey
        assert resolver.resolve("user's last partner") == "casey"
        gm.add_entity(GraphNode(entity_id="abusive", display_name="abusive"))
        ms._ingest_fact_to_graph(
            subj="user's last partner", rel="is", obj="abusive",
            fact_id="f2", confidence=0.9, stance="appraisal",
            capture_tone="elevated",
        )
        # the appraisal edge hangs off a verbatim role node, NOT casey
        casey_edges = gm.get_relations("casey")
        assert not any(e.target_id == "abusive" for e in casey_edges)
        role_node = gm.get_entity("user_s_last_partner")
        assert role_node is not None
        assert role_node.entity_type == "role"
        role_edges = gm.get_relations("user_s_last_partner")
        assert any(e.target_id == "abusive" for e in role_edges)

    def test_role_subject_metadata_branch_also_bypasses_resolver(self, graph_env):
        ms, gm, resolver = graph_env
        # non-graph-worthy object → node-metadata branch
        ms._ingest_fact_to_graph(
            subj="user's last partner", rel="communication_style",
            obj="stopped answering messages for weeks at a time",
            fact_id="f3", confidence=0.9, stance="appraisal",
        )
        casey = gm.get_entity("casey")
        assert "communication_style" not in (casey.metadata or {})


class TestCaptureToneJoin:
    def _items(self):
        return [
            {"query": "casual chat about the gym", "response": "",
             "is_heavy_topic": False},
            {"query": "casey was evil to me", "response": "that sounds heavy",
             "is_heavy_topic": True},
        ]

    def test_matched_heavy_entry_elevated(self):
        tone = ShutdownProcessor._capture_tone_for_triple(
            {"subject": "casey", "relation": "is", "object": "evil"},
            self._items(),
        )
        assert tone == "elevated"

    def test_matched_light_entry_non_elevated(self):
        tone = ShutdownProcessor._capture_tone_for_triple(
            {"subject": "user", "relation": "attends", "object": "gym"},
            self._items(),
        )
        assert tone == "non_elevated"

    def test_unmatched_unknown(self):
        tone = ShutdownProcessor._capture_tone_for_triple(
            {"subject": "user", "relation": "lives_in", "object": "chicago"},
            self._items(),
        )
        assert tone == "unknown"

    def test_missing_flag_unknown(self):
        tone = ShutdownProcessor._capture_tone_for_triple(
            {"subject": "casey", "relation": "is", "object": "evil"},
            [{"query": "casey was evil to me", "response": ""}],
        )
        assert tone == "unknown"
