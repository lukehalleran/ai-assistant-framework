"""Tests for Proactive Context Surfacing.

Covers:
- Pydantic model construction (DomainEntity, DomainCluster, CrossDomainCandidate, ProactiveInsight)
- Novelty key determinism
- SurfacingHistory (recently_shown, cooldown, JSON roundtrip, cleanup)
- Domain classification (_classify_user_edges, override precedence)
- Active domain identification (entity resolution, keyword fallback, multi-domain)
- Bridge candidate selection (scoring, novelty filter, max cap)
- LLM response parsing (JSON parsing, null handling, malformed JSON graceful)
- Full pipeline (generate_insights, sparse graph no-op, session cache, graceful failure)
"""
import json
import os
import pytest
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from memory.surfacing_models import (
    DomainEntity,
    DomainCluster,
    CrossDomainCandidate,
    ProactiveInsight,
)
from memory.surfacing_history import SurfacingHistory
from memory.context_surfacer import ContextSurfacer


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_edge(source_id="user", target_id="vyvanse", relation="medication", weight=1.0):
    """Create a mock GraphEdge-like object."""
    edge = MagicMock()
    edge.source_id = source_id
    edge.target_id = target_id
    edge.relation = relation
    edge.weight = weight
    return edge


def _make_node(entity_id, display_name=None, entity_type="other"):
    """Create a mock GraphNode-like object."""
    node = MagicMock()
    node.entity_id = entity_id
    node.display_name = display_name or entity_id.title()
    node.entity_type = entity_type
    return node


def _make_graph_memory(edges=None, nodes=None, n_nodes=50, n_edges=30):
    """Create a mock GraphMemory with configurable edges and nodes."""
    gm = MagicMock()
    gm.node_count.return_value = n_nodes
    gm.edge_count.return_value = n_edges

    edges = edges or []
    gm.get_relations.return_value = edges

    nodes = nodes or {}
    def _get_entity(eid):
        return nodes.get(eid)
    gm.get_entity.side_effect = _get_entity

    gm.get_context_sentences.return_value = ["User takes Vyvanse", "User works at Sturdy Shelter"]
    return gm


def _make_resolver(mapping=None):
    """Create a mock EntityResolver."""
    resolver = MagicMock()
    mapping = mapping or {}
    def _resolve(text):
        return mapping.get(text.lower())
    resolver.resolve.side_effect = _resolve
    return resolver


# ---------------------------------------------------------------------------
# TestSurfacingModels
# ---------------------------------------------------------------------------

class TestSurfacingModels:
    def test_domain_entity_construction(self):
        e = DomainEntity(
            entity_id="vyvanse",
            display_name="Vyvanse",
            domain="health",
            relation="medication",
            edge_weight=2.0,
        )
        assert e.entity_id == "vyvanse"
        assert e.domain == "health"
        assert e.edge_weight == 2.0

    def test_domain_entity_defaults(self):
        e = DomainEntity(
            entity_id="x", display_name="X", domain="health", relation="condition"
        )
        assert e.edge_weight == 1.0

    def test_domain_cluster_construction(self):
        c = DomainCluster(domain="career")
        assert c.domain == "career"
        assert c.entities == []
        assert c.fact_sentences == []

    def test_domain_cluster_with_entities(self):
        e = DomainEntity(entity_id="a", display_name="A", domain="career", relation="works_at")
        c = DomainCluster(domain="career", entities=[e])
        assert len(c.entities) == 1

    def test_cross_domain_candidate(self):
        ac = DomainCluster(domain="career")
        bc = DomainCluster(domain="health")
        cand = CrossDomainCandidate(
            active_domain="career",
            bridged_domain="health",
            active_cluster=ac,
            bridged_cluster=bc,
            relevance_score=0.75,
        )
        assert cand.active_domain == "career"
        assert cand.relevance_score == 0.75

    def test_proactive_insight_construction(self):
        now = datetime.now()
        i = ProactiveInsight(
            insight_text="Work stress connects to sleep.",
            active_domain="career",
            bridged_domain="health",
            entity_ids=["sturdy_shelter", "vyvanse"],
            confidence=0.8,
            generated_at=now,
        )
        assert i.insight_text == "Work stress connects to sleep."
        assert i.confidence == 0.8

    def test_novelty_key_determinism(self):
        i1 = ProactiveInsight(
            insight_text="A",
            active_domain="health",
            bridged_domain="career",
            entity_ids=["vyvanse", "sturdy_shelter"],
        )
        i2 = ProactiveInsight(
            insight_text="B",
            active_domain="career",
            bridged_domain="health",
            entity_ids=["sturdy_shelter", "vyvanse"],
        )
        assert i1.novelty_key() == i2.novelty_key()

    def test_novelty_key_different_entities(self):
        i1 = ProactiveInsight(
            insight_text="A",
            active_domain="health",
            bridged_domain="career",
            entity_ids=["vyvanse"],
        )
        i2 = ProactiveInsight(
            insight_text="A",
            active_domain="health",
            bridged_domain="career",
            entity_ids=["ibuprofen"],
        )
        assert i1.novelty_key() != i2.novelty_key()


# ---------------------------------------------------------------------------
# TestSurfacingHistory
# ---------------------------------------------------------------------------

class TestSurfacingHistory:
    def test_not_recently_shown_when_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            h = SurfacingHistory(persist_path=path)
            assert not h.was_recently_shown("career|health|vyvanse")
        finally:
            os.unlink(path)

    def test_recently_shown_after_record(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            h = SurfacingHistory(persist_path=path)
            h.record_surfaced("key1")
            assert h.was_recently_shown("key1", cooldown_hours=72)
        finally:
            os.unlink(path)

    def test_not_recently_shown_after_cooldown(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            h = SurfacingHistory(persist_path=path)
            # Manually insert an old entry
            h._entries["old_key"] = {
                "last_surfaced": (datetime.now() - timedelta(hours=100)).isoformat(),
                "count": 1,
            }
            assert not h.was_recently_shown("old_key", cooldown_hours=72)
        finally:
            os.unlink(path)

    def test_json_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            h = SurfacingHistory(persist_path=path)
            h.record_surfaced("key1")
            h.record_surfaced("key2")

            # Reload
            h2 = SurfacingHistory(persist_path=path)
            assert h2.was_recently_shown("key1")
            assert h2.was_recently_shown("key2")
        finally:
            os.unlink(path)

    def test_record_increments_count(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            h = SurfacingHistory(persist_path=path)
            h.record_surfaced("key1")
            h.record_surfaced("key1")
            assert h._entries["key1"]["count"] == 2
        finally:
            os.unlink(path)

    def test_cleanup_old(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            h = SurfacingHistory(persist_path=path)
            h._entries["old"] = {
                "last_surfaced": (datetime.now() - timedelta(days=60)).isoformat(),
                "count": 1,
            }
            h._entries["recent"] = {
                "last_surfaced": datetime.now().isoformat(),
                "count": 1,
            }
            removed = h.cleanup_old(max_age_days=30)
            assert removed == 1
            assert "old" not in h._entries
            assert "recent" in h._entries
        finally:
            os.unlink(path)

    def test_cleanup_removes_invalid_entries(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            h = SurfacingHistory(persist_path=path)
            h._entries["bad"] = {"last_surfaced": "not-a-date", "count": 1}
            removed = h.cleanup_old(max_age_days=30)
            assert removed == 1
            assert "bad" not in h._entries
        finally:
            os.unlink(path)

    def test_load_nonexistent_file(self):
        h = SurfacingHistory(persist_path="/tmp/nonexistent_surfacing_test.json")
        assert h._entries == {}

    def test_load_corrupt_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("{bad json")
            path = f.name
        try:
            h = SurfacingHistory(persist_path=path)
            assert h._entries == {}
        finally:
            os.unlink(path)

    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "history.json")
            h = SurfacingHistory(persist_path=path)
            h.record_surfaced("key1")
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# TestClassifyUserEdges
# ---------------------------------------------------------------------------

class TestClassifyUserEdges:
    def _make_surfacer(self, edges, nodes):
        gm = _make_graph_memory(n_nodes=50, n_edges=30)
        gm.get_relations.return_value = edges
        gm.get_entity.side_effect = lambda eid: nodes.get(eid)
        gm.get_context_sentences.return_value = ["fact sentence"]
        resolver = _make_resolver()
        mm = MagicMock()
        return ContextSurfacer(graph_memory=gm, entity_resolver=resolver, model_manager=mm)

    def test_basic_classification(self):
        edges = [
            _make_edge(target_id="vyvanse", relation="medication"),
            _make_edge(target_id="sturdy_shelter", relation="works_at"),
        ]
        nodes = {
            "vyvanse": _make_node("vyvanse", "Vyvanse"),
            "sturdy_shelter": _make_node("sturdy_shelter", "Sturdy Shelter"),
        }
        s = self._make_surfacer(edges, nodes)
        clusters = s._classify_user_edges()
        assert "health" in clusters
        assert "career" in clusters

    def test_override_takes_precedence(self):
        """pet relation should map to hobbies, not default."""
        edges = [_make_edge(target_id="flapjack", relation="pet")]
        nodes = {"flapjack": _make_node("flapjack", "Flapjack")}
        s = self._make_surfacer(edges, nodes)
        clusters = s._classify_user_edges()
        assert "hobbies" in clusters
        ent_ids = [e.entity_id for e in clusters["hobbies"].entities]
        assert "flapjack" in ent_ids

    def test_sibling_override(self):
        edges = [_make_edge(target_id="auggie", relation="sibling_of")]
        nodes = {"auggie": _make_node("auggie", "Auggie")}
        s = self._make_surfacer(edges, nodes)
        clusters = s._classify_user_edges()
        assert "relationships" in clusters

    def test_caching_by_edge_count(self):
        edges = [_make_edge(target_id="vyvanse", relation="medication")]
        nodes = {"vyvanse": _make_node("vyvanse", "Vyvanse")}
        s = self._make_surfacer(edges, nodes)
        c1 = s._classify_user_edges()
        c2 = s._classify_user_edges()
        # Should be the same object (cached)
        assert c1 is c2

    def test_cache_invalidation_on_edge_count_change(self):
        edges = [_make_edge(target_id="vyvanse", relation="medication")]
        nodes = {"vyvanse": _make_node("vyvanse", "Vyvanse")}
        s = self._make_surfacer(edges, nodes)
        c1 = s._classify_user_edges()
        # Simulate edge count change
        s._graph_memory.edge_count.return_value = 999
        c2 = s._classify_user_edges()
        assert c1 is not c2

    def test_skips_self_edges(self):
        edges = [_make_edge(target_id="user", relation="self_ref")]
        nodes = {}
        s = self._make_surfacer(edges, nodes)
        clusters = s._classify_user_edges()
        assert len(clusters) == 0

    def test_multiple_entities_same_domain(self):
        edges = [
            _make_edge(target_id="vyvanse", relation="medication"),
            _make_edge(target_id="long_covid", relation="condition"),
        ]
        nodes = {
            "vyvanse": _make_node("vyvanse", "Vyvanse"),
            "long_covid": _make_node("long_covid", "Long Covid"),
        }
        s = self._make_surfacer(edges, nodes)
        clusters = s._classify_user_edges()
        assert len(clusters["health"].entities) == 2

    def test_heuristic_fallback_for_unknown_relation(self):
        """categorize_relation should handle unknown relations via heuristics."""
        edges = [_make_edge(target_id="5k", relation="fitness_goal")]
        nodes = {"5k": _make_node("5k", "5K Race")}
        s = self._make_surfacer(edges, nodes)
        clusters = s._classify_user_edges()
        assert "fitness" in clusters


# ---------------------------------------------------------------------------
# TestIdentifyActiveDomains
# ---------------------------------------------------------------------------

class TestIdentifyActiveDomains:
    def _make_surfacer_with_clusters(self, resolver_mapping=None):
        gm = _make_graph_memory()
        resolver = _make_resolver(resolver_mapping or {})
        mm = MagicMock()
        s = ContextSurfacer(graph_memory=gm, entity_resolver=resolver, model_manager=mm)
        return s

    def test_entity_resolution_finds_domain(self):
        s = self._make_surfacer_with_clusters({"vyvanse": "vyvanse"})
        clusters = {
            "health": DomainCluster(
                domain="health",
                entities=[DomainEntity(entity_id="vyvanse", display_name="Vyvanse",
                                      domain="health", relation="medication")],
            ),
            "career": DomainCluster(domain="career", entities=[]),
        }
        active = s._identify_active_domains("How is my Vyvanse working?", clusters)
        assert "health" in active

    def test_keyword_fallback(self):
        s = self._make_surfacer_with_clusters()
        clusters = {
            "career": DomainCluster(
                domain="career",
                entities=[DomainEntity(entity_id="sturdy_shelter", display_name="Sturdy Shelter",
                                      domain="career", relation="works_at")],
            ),
        }
        active = s._identify_active_domains("I'm stressed about work lately", clusters)
        assert "career" in active

    def test_multi_domain_detection(self):
        s = self._make_surfacer_with_clusters({"vyvanse": "vyvanse"})
        clusters = {
            "health": DomainCluster(
                domain="health",
                entities=[DomainEntity(entity_id="vyvanse", display_name="Vyvanse",
                                      domain="health", relation="medication")],
            ),
            "career": DomainCluster(
                domain="career",
                entities=[DomainEntity(entity_id="sturdy_shelter", display_name="Sturdy Shelter",
                                      domain="career", relation="works_at")],
            ),
        }
        active = s._identify_active_domains("My Vyvanse and work stress", clusters)
        assert "health" in active
        assert "career" in active

    def test_no_domain_detected(self):
        s = self._make_surfacer_with_clusters()
        clusters = {
            "career": DomainCluster(domain="career", entities=[]),
        }
        active = s._identify_active_domains("hey what's up", clusters)
        assert len(active) == 0

    def test_domain_not_in_clusters_ignored(self):
        s = self._make_surfacer_with_clusters()
        clusters = {}  # No clusters at all
        active = s._identify_active_domains("work stress", clusters)
        assert len(active) == 0

    def test_empty_query(self):
        s = self._make_surfacer_with_clusters()
        clusters = {"career": DomainCluster(domain="career", entities=[])}
        active = s._identify_active_domains("", clusters)
        assert len(active) == 0

    def test_health_keywords(self):
        s = self._make_surfacer_with_clusters()
        clusters = {
            "health": DomainCluster(
                domain="health",
                entities=[DomainEntity(entity_id="x", display_name="X",
                                      domain="health", relation="condition")],
            ),
        }
        active = s._identify_active_domains("My anxiety has been bad", clusters)
        assert "health" in active


# ---------------------------------------------------------------------------
# TestSelectBridgeCandidates
# ---------------------------------------------------------------------------

class TestSelectBridgeCandidates:
    def _make_surfacer(self):
        gm = _make_graph_memory()
        gm.get_relations.return_value = []  # No lateral edges by default
        resolver = _make_resolver()
        mm = MagicMock()
        s = ContextSurfacer(graph_memory=gm, entity_resolver=resolver, model_manager=mm)
        # Inject a no-op history to avoid file I/O
        history = MagicMock()
        history.was_recently_shown.return_value = False
        s._history = history
        return s

    def test_selects_non_active_domains(self):
        s = self._make_surfacer()
        clusters = {
            "career": DomainCluster(
                domain="career",
                entities=[DomainEntity(entity_id="a", display_name="A",
                                      domain="career", relation="works_at")],
            ),
            "health": DomainCluster(
                domain="health",
                entities=[DomainEntity(entity_id="b", display_name="B",
                                      domain="health", relation="medication")],
            ),
        }
        cands = s._select_bridge_candidates({"career"}, clusters)
        assert len(cands) == 1
        assert cands[0].bridged_domain == "health"

    def test_filters_recently_shown(self):
        s = self._make_surfacer()
        s._history.was_recently_shown.return_value = True
        clusters = {
            "career": DomainCluster(
                domain="career",
                entities=[DomainEntity(entity_id="a", display_name="A",
                                      domain="career", relation="works_at")],
            ),
            "health": DomainCluster(
                domain="health",
                entities=[DomainEntity(entity_id="b", display_name="B",
                                      domain="health", relation="medication")],
            ),
        }
        cands = s._select_bridge_candidates({"career"}, clusters)
        assert len(cands) == 0

    def test_max_cap(self):
        s = self._make_surfacer()
        clusters = {
            "career": DomainCluster(
                domain="career",
                entities=[DomainEntity(entity_id="a", display_name="A",
                                      domain="career", relation="works_at")],
            ),
        }
        # Add many bridgeable domains
        for i in range(10):
            domain = f"domain_{i}"
            clusters[domain] = DomainCluster(
                domain=domain,
                entities=[DomainEntity(entity_id=f"e{i}", display_name=f"E{i}",
                                      domain=domain, relation="rel")],
            )
        cands = s._select_bridge_candidates({"career"}, clusters, max_candidates=3)
        assert len(cands) <= 3

    def test_sorts_by_relevance(self):
        s = self._make_surfacer()
        clusters = {
            "career": DomainCluster(
                domain="career",
                entities=[DomainEntity(entity_id="a", display_name="A",
                                      domain="career", relation="works_at")],
            ),
            "health": DomainCluster(
                domain="health",
                entities=[
                    DomainEntity(entity_id=f"h{i}", display_name=f"H{i}",
                                domain="health", relation="condition")
                    for i in range(5)  # More entities = higher score
                ],
            ),
            "hobbies": DomainCluster(
                domain="hobbies",
                entities=[DomainEntity(entity_id="p", display_name="P",
                                      domain="hobbies", relation="pet")],
            ),
        }
        cands = s._select_bridge_candidates({"career"}, clusters)
        # Health should rank higher (more entities)
        assert cands[0].bridged_domain == "health"

    def test_skips_empty_bridged_clusters(self):
        s = self._make_surfacer()
        clusters = {
            "career": DomainCluster(
                domain="career",
                entities=[DomainEntity(entity_id="a", display_name="A",
                                      domain="career", relation="works_at")],
            ),
            "health": DomainCluster(domain="health", entities=[]),  # Empty
        }
        cands = s._select_bridge_candidates({"career"}, clusters)
        assert len(cands) == 0

    def test_skips_active_domains_as_bridge(self):
        s = self._make_surfacer()
        clusters = {
            "career": DomainCluster(
                domain="career",
                entities=[DomainEntity(entity_id="a", display_name="A",
                                      domain="career", relation="works_at")],
            ),
            "health": DomainCluster(
                domain="health",
                entities=[DomainEntity(entity_id="b", display_name="B",
                                      domain="health", relation="medication")],
            ),
        }
        # Both are active — no bridges possible
        cands = s._select_bridge_candidates({"career", "health"}, clusters)
        assert len(cands) == 0

    def test_lateral_edges_boost_score(self):
        """Domains with lateral (non-user) edges should score higher."""
        gm = _make_graph_memory()
        # Return lateral edges for health entities
        def side_effect_rels(eid, direction="both"):
            if eid in ("h0", "h1"):
                return [
                    _make_edge(source_id=eid, target_id="related", relation="affects"),
                    _make_edge(source_id=eid, target_id="other", relation="causes"),
                ]
            return []
        gm.get_relations.side_effect = side_effect_rels

        resolver = _make_resolver()
        mm = MagicMock()
        s = ContextSurfacer(graph_memory=gm, entity_resolver=resolver, model_manager=mm)
        s._history = MagicMock()
        s._history.was_recently_shown.return_value = False

        clusters = {
            "career": DomainCluster(
                domain="career",
                entities=[DomainEntity(entity_id="a", display_name="A",
                                      domain="career", relation="works_at")],
            ),
            "health": DomainCluster(
                domain="health",
                entities=[
                    DomainEntity(entity_id="h0", display_name="H0",
                                domain="health", relation="condition"),
                    DomainEntity(entity_id="h1", display_name="H1",
                                domain="health", relation="medication"),
                ],
            ),
            "hobbies": DomainCluster(
                domain="hobbies",
                entities=[DomainEntity(entity_id="p", display_name="P",
                                      domain="hobbies", relation="pet")],
            ),
        }
        cands = s._select_bridge_candidates({"career"}, clusters)
        # Health should be first (has lateral edges)
        assert cands[0].bridged_domain == "health"


# ---------------------------------------------------------------------------
# TestSynthesizeInsight
# ---------------------------------------------------------------------------

class TestSynthesizeInsight:
    def _make_surfacer(self, llm_response=""):
        gm = _make_graph_memory()
        resolver = _make_resolver()
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value=llm_response)
        return ContextSurfacer(graph_memory=gm, entity_resolver=resolver, model_manager=mm)

    def _make_candidate(self, active="career", bridged="health"):
        ac = DomainCluster(domain=active, fact_sentences=["User works at brewery"])
        bc = DomainCluster(domain=bridged, fact_sentences=["User takes Vyvanse"])
        return CrossDomainCandidate(
            active_domain=active,
            bridged_domain=bridged,
            active_cluster=ac,
            bridged_cluster=bc,
        )

    @pytest.mark.asyncio
    async def test_valid_json_response(self):
        response = json.dumps([
            {"insight": "Sleep disruption from brewery connects to Vyvanse timing.",
             "confidence": 0.8, "entity_ids": ["sturdy_shelter", "vyvanse"]}
        ])
        s = self._make_surfacer(response)
        cand = self._make_candidate()
        insights = await s._synthesize_insights_batch([cand], max_insights=2)
        assert len(insights) == 1
        assert "Vyvanse" in insights[0].insight_text
        assert insights[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_null_entries_skipped(self):
        response = json.dumps([
            None,
            {"insight": "Valid insight.", "confidence": 0.6, "entity_ids": ["a"]}
        ])
        s = self._make_surfacer(response)
        cands = [self._make_candidate("career", "health"), self._make_candidate("career", "hobbies")]
        insights = await s._synthesize_insights_batch(cands, max_insights=2)
        assert len(insights) == 1
        assert insights[0].insight_text == "Valid insight."

    @pytest.mark.asyncio
    async def test_malformed_json_graceful(self):
        s = self._make_surfacer("This is not JSON at all")
        cand = self._make_candidate()
        insights = await s._synthesize_insights_batch([cand])
        assert insights == []

    @pytest.mark.asyncio
    async def test_empty_response(self):
        s = self._make_surfacer("")
        cand = self._make_candidate()
        insights = await s._synthesize_insights_batch([cand])
        assert insights == []

    @pytest.mark.asyncio
    async def test_max_insights_cap(self):
        response = json.dumps([
            {"insight": "Insight 1.", "confidence": 0.9, "entity_ids": ["a"]},
            {"insight": "Insight 2.", "confidence": 0.8, "entity_ids": ["b"]},
            {"insight": "Insight 3.", "confidence": 0.7, "entity_ids": ["c"]},
        ])
        s = self._make_surfacer(response)
        cands = [self._make_candidate() for _ in range(3)]
        insights = await s._synthesize_insights_batch(cands, max_insights=2)
        assert len(insights) == 2

    @pytest.mark.asyncio
    async def test_json_with_surrounding_text(self):
        response = 'Here are the insights:\n[{"insight": "Connection found.", "confidence": 0.7, "entity_ids": []}]\nDone.'
        s = self._make_surfacer(response)
        cand = self._make_candidate()
        insights = await s._synthesize_insights_batch([cand])
        assert len(insights) == 1

    @pytest.mark.asyncio
    async def test_llm_exception_graceful(self):
        gm = _make_graph_memory()
        resolver = _make_resolver()
        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=Exception("API error"))
        s = ContextSurfacer(graph_memory=gm, entity_resolver=resolver, model_manager=mm)
        cand = self._make_candidate()
        insights = await s._synthesize_insights_batch([cand])
        assert insights == []

    @pytest.mark.asyncio
    async def test_no_model_manager(self):
        gm = _make_graph_memory()
        resolver = _make_resolver()
        s = ContextSurfacer(graph_memory=gm, entity_resolver=resolver, model_manager=None)
        cand = self._make_candidate()
        insights = await s._synthesize_insights_batch([cand])
        assert insights == []


# ---------------------------------------------------------------------------
# TestGenerateInsights (full pipeline)
# ---------------------------------------------------------------------------

class TestGenerateInsights:
    def _make_full_surfacer(self, llm_response="[]", edges=None, nodes=None,
                            n_nodes=50, n_edges=30, resolver_mapping=None):
        edges = edges or []
        nodes = nodes or {}
        gm = _make_graph_memory(edges, nodes, n_nodes, n_edges)
        resolver = _make_resolver(resolver_mapping or {})
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value=llm_response)
        s = ContextSurfacer(graph_memory=gm, entity_resolver=resolver, model_manager=mm)
        # Inject no-op history
        history = MagicMock()
        history.was_recently_shown.return_value = False
        history.record_surfaced = MagicMock()
        s._history = history
        return s

    @pytest.mark.asyncio
    async def test_sparse_graph_returns_empty(self):
        s = self._make_full_surfacer(n_nodes=5, n_edges=3)
        result = await s.generate_insights("work stress")
        assert result == []

    @pytest.mark.asyncio
    async def test_no_graph_returns_empty(self):
        mm = MagicMock()
        s = ContextSurfacer(graph_memory=None, entity_resolver=None, model_manager=mm)
        result = await s.generate_insights("work stress")
        assert result == []

    @pytest.mark.asyncio
    async def test_session_cache(self):
        edges = [
            _make_edge(target_id="vyvanse", relation="medication"),
            _make_edge(target_id="sturdy_shelter", relation="works_at"),
        ]
        nodes = {
            "vyvanse": _make_node("vyvanse", "Vyvanse"),
            "sturdy_shelter": _make_node("sturdy_shelter", "Sturdy Shelter"),
        }
        response = json.dumps([
            {"insight": "Connection found.", "confidence": 0.8, "entity_ids": ["vyvanse"]}
        ])
        s = self._make_full_surfacer(
            llm_response=response,
            edges=edges,
            nodes=nodes,
        )
        result1 = await s.generate_insights("work stress")
        result2 = await s.generate_insights("work stress again")
        # Should be cached — LLM called only once
        assert s._model_manager.generate_once.call_count == 1
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        s = self._make_full_surfacer()
        with patch("config.app_config.PROACTIVE_SURFACING_ENABLED", False):
            result = await s.generate_insights("work stress")
        assert result == []

    @pytest.mark.asyncio
    async def test_full_pipeline_success(self):
        edges = [
            _make_edge(target_id="vyvanse", relation="medication"),
            _make_edge(target_id="sturdy_shelter", relation="works_at"),
        ]
        nodes = {
            "vyvanse": _make_node("vyvanse", "Vyvanse"),
            "sturdy_shelter": _make_node("sturdy_shelter", "Sturdy Shelter"),
        }
        response = json.dumps([
            {"insight": "Brewery shifts disrupt Vyvanse timing.", "confidence": 0.85,
             "entity_ids": ["sturdy_shelter", "vyvanse"]}
        ])
        s = self._make_full_surfacer(
            llm_response=response,
            edges=edges,
            nodes=nodes,
        )
        result = await s.generate_insights("work stress")
        assert len(result) == 1
        assert "Vyvanse" in result[0]

    @pytest.mark.asyncio
    async def test_fewer_than_2_domains_returns_empty(self):
        """If all entities belong to one domain, no bridges possible."""
        edges = [
            _make_edge(target_id="vyvanse", relation="medication"),
            _make_edge(target_id="long_covid", relation="condition"),
        ]
        nodes = {
            "vyvanse": _make_node("vyvanse", "Vyvanse"),
            "long_covid": _make_node("long_covid", "Long Covid"),
        }
        s = self._make_full_surfacer(edges=edges, nodes=nodes)
        result = await s.generate_insights("health stuff")
        assert result == []

    @pytest.mark.asyncio
    async def test_graceful_llm_failure(self):
        edges = [
            _make_edge(target_id="vyvanse", relation="medication"),
            _make_edge(target_id="sturdy_shelter", relation="works_at"),
        ]
        nodes = {
            "vyvanse": _make_node("vyvanse", "Vyvanse"),
            "sturdy_shelter": _make_node("sturdy_shelter", "Sturdy Shelter"),
        }
        gm = _make_graph_memory(edges, nodes)
        resolver = _make_resolver()
        mm = MagicMock()
        mm.generate_once = AsyncMock(side_effect=Exception("API down"))
        s = ContextSurfacer(graph_memory=gm, entity_resolver=resolver, model_manager=mm)
        s._history = MagicMock()
        s._history.was_recently_shown.return_value = False
        result = await s.generate_insights("work stress")
        assert result == []
