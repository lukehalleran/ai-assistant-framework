"""2026-09-03 follow-ups — knowledge-graph read model.

Two defects from the cat-fetch session review:

* The NetworkX DiGraph holds ONE edge per (source, target) pair, and reads
  rebuilt the relation key from that single nx attribute — so a second relation
  on the same pair was invisible (146 of 982 live edges; ``has_dog`` hid the
  older ``mom_pet`` on a cat). ``_edge_index`` plus the adjacency maps are now
  the source of truth for every relation-level read; nx stays the topology
  index.
* ``get_context_sentences`` fanned out THROUGH the ``user`` star hub (degree
  770) at depth 2 and rendered structural ``mentioned_alongside`` edges, so a
  pet seed produced "Python mentioned alongside name". The prompt path now uses
  an opt-in hub barrier, skips structural relations, and orders seed-incident
  edges first.
"""
from datetime import datetime

from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge, GraphNode


def _gm(tmp_path):
    return GraphMemory(persist_path=str(tmp_path / "g.json"))


def _node(gm, eid, etype="other", mentions=1, **md):
    gm.add_entity(GraphNode(entity_id=eid, display_name=eid.title(), entity_type=etype,
                            mention_count=mentions, metadata=md))


def _edge(gm, src, rel, tgt, weight=1.0):
    now = datetime.now()
    gm.add_relation(GraphEdge(source_id=src, relation=rel, target_id=tgt, weight=weight,
                              first_seen=now, last_seen=now))


# ── multi-relation pairs ──────────────────────────────────────────────────
class TestMultiRelationPairs:
    def test_two_relations_same_pair_both_returned(self, tmp_path):
        gm = _gm(tmp_path)
        _node(gm, "user", "person"); _node(gm, "mochi", "animal", 3, species="cat")
        _edge(gm, "user", "mom_pet", "mochi")
        _edge(gm, "user", "has_dog", "mochi")
        rels = {e.relation for e in gm.get_relations("user", direction="out")}
        assert rels == {"mom_pet", "has_dog"}
        assert {e.relation for e in gm.get_relations("mochi", direction="in")} == {"mom_pet", "has_dog"}
        assert {e.relation for e in gm.neighbors("user", depth=1)["user"]} == {"mom_pet", "has_dog"}
        assert {e.relation for e in gm.subgraph_around("user", depth=1)} == {"mom_pet", "has_dog"}
        assert gm.edge_count() == 2
        assert gm.graph.number_of_edges() == 1  # topology index keeps one pair

    def test_context_sentences_render_the_surviving_relation(self, tmp_path):
        gm = _gm(tmp_path)
        _node(gm, "user", "person"); _node(gm, "mochi", "animal", 3, species="cat")
        _edge(gm, "user", "mom_pet", "mochi")
        _edge(gm, "user", "has_dog", "mochi")   # species conflict → suppressed at read
        text = " ".join(gm.get_context_sentences("user", depth=1)).lower()
        assert "mom pet mochi" in text
        assert "dog" not in text

    def test_strengthen_does_not_touch_other_relation(self, tmp_path):
        gm = _gm(tmp_path)
        _node(gm, "user", "person"); _node(gm, "mochi", "animal", 3)
        _edge(gm, "user", "mom_pet", "mochi")     # first relation owns the nx pair
        _edge(gm, "user", "likes", "mochi")
        for _ in range(3):
            _edge(gm, "user", "likes", "mochi")
        assert gm._edge_index["user|mom_pet|mochi"].weight == 1.0
        assert gm._edge_index["user|likes|mochi"].weight == 4.0
        nx_attrs = gm.graph["user"]["mochi"]
        assert nx_attrs["relation"] == "mom_pet" and nx_attrs["weight"] == 1.0

    def test_save_load_round_trip_preserves_all_relations(self, tmp_path):
        gm = _gm(tmp_path)
        _node(gm, "user", "person"); _node(gm, "biscuit", "animal", 5)
        for rel in ("pet", "has_cat", "misses"):
            _edge(gm, "user", rel, "biscuit")
        gm.save()
        fresh = GraphMemory(persist_path=str(tmp_path / "g.json")); fresh.load()
        assert fresh.edge_count() == 3
        assert {e.relation for e in fresh.get_relations("user")} == {"pet", "has_cat", "misses"}
        assert fresh._adj_count == len(fresh._edge_index)

    def test_remove_entity_clears_adjacency(self, tmp_path):
        gm = _gm(tmp_path)
        _node(gm, "user", "person"); _node(gm, "daisy", "animal", 2); _node(gm, "bean", "animal", 2)
        _edge(gm, "user", "pet", "daisy"); _edge(gm, "user", "misses", "daisy"); _edge(gm, "daisy", "sibling", "bean")
        gm.remove_entity("daisy")
        for maps in (gm._out_keys, gm._in_keys):
            for keys in maps.values():
                assert not any("daisy" in k.split("|") for k in keys)
        assert gm._adj_count == len(gm._edge_index) == 0
        assert gm.get_relations("user") == []

    def test_rebuild_after_direct_index_mutation(self, tmp_path):
        gm = _gm(tmp_path)
        _node(gm, "user", "person"); _node(gm, "mochi", "animal", 3)
        _edge(gm, "user", "pet", "mochi")
        e = gm._edge_index.pop("user|pet|mochi")
        e2 = GraphEdge(source_id="user", relation="has_cat", target_id="mochi", weight=e.weight,
                       first_seen=e.first_seen, last_seen=e.last_seen)
        gm._edge_index["user|has_cat|mochi"] = e2   # count-neutral swap
        gm.rebuild_edge_indexes()
        assert [x.relation for x in gm.get_relations("user")] == ["has_cat"]
        gm._edge_index["user|likes|mochi"] = GraphEdge(source_id="user", relation="likes", target_id="mochi")
        # count drift is self-healing on the next read
        assert {x.relation for x in gm.get_relations("user")} == {"has_cat", "likes"}

    def test_get_relations_insertion_ordered(self, tmp_path):
        gm = _gm(tmp_path)
        _node(gm, "casey", "person"); _node(gm, "evil", "other")
        _edge(gm, "casey", "is", "evil"); _edge(gm, "casey", "lives_in", "evil")
        assert [e.relation for e in gm.get_relations("casey")] == ["is", "lives_in"]


# ── prompt-path hub barrier + ordering ────────────────────────────────────
def _pet_world(tmp_path):
    gm = _gm(tmp_path)
    _node(gm, "user", "person", 0)
    _node(gm, "biscuit", "animal", 24, species="cat")
    _node(gm, "mochi", "animal", 12, species="cat")
    _node(gm, "python", "other", 20); _node(gm, "gym", "other", 23); _node(gm, "name", "other", 1)
    _node(gm, "huge", "other", 1)
    for tgt in ("biscuit", "mochi", "python", "gym", "name"):
        _edge(gm, "user", "has", tgt)
    _edge(gm, "user", "goes_to", "gym", weight=6.0)
    _edge(gm, "biscuit", "seems_best_buds_with", "mochi")
    _edge(gm, "biscuit", "size", "huge")
    _edge(gm, "python", "mentioned_alongside", "name", weight=2.5)
    return gm


class TestGraphContextBarrier:
    def test_user_hub_not_expanded_through_at_depth_2(self, tmp_path):
        gm = _pet_world(tmp_path)
        # mochi's neighbourhood (2 incident edges + biscuit's size edge) fills
        # three slots before any reached-through hub edge can appear.
        text = " ".join(gm.get_context_sentences("mochi", depth=2, max_sentences=3)).lower()
        assert "python" not in text and "gym" not in text
        assert "mochi" in text
        # the hub's own neighbours (python/gym) are never TRAVERSED: nothing
        # beyond user's direct edges is reachable
        reach = gm.neighbors("mochi", depth=2, hub_barrier=True)
        assert "python" not in reach and "gym" not in reach

    def test_hub_edges_rank_after_the_seed_neighbourhood(self, tmp_path):
        gm = _pet_world(tmp_path)
        sents = [x.lower() for x in gm.get_context_sentences("mochi", depth=2, max_sentences=40)]
        # every reached-through hub edge (touches user, not the seed) renders
        # after every sentence of the pet neighbourhood
        hub = [i for i, x in enumerate(sents) if x.startswith("user ") and "mochi" not in x]
        neighbourhood = [i for i, x in enumerate(sents) if not x.startswith("user ")]
        assert hub and neighbourhood
        assert max(neighbourhood) < min(hub)

    def test_seed_user_still_expands(self, tmp_path):
        gm = _pet_world(tmp_path)
        text = " ".join(gm.get_context_sentences("user", depth=1, max_sentences=20)).lower()
        assert "gym" in text and "biscuit" in text

    def test_degree_hub_barrier(self, tmp_path):
        gm = _gm(tmp_path)
        for n in ("a", "hub", "x", "y", "z", "far"):
            _node(gm, n)
        _edge(gm, "a", "r", "hub")
        for t in ("x", "y", "z"):
            _edge(gm, "hub", "r", t)
        _edge(gm, "x", "r", "far")
        reach_default = set(gm.neighbors("a", depth=3))
        reach_barrier = set(gm.neighbors("a", depth=3, hub_barrier=True, hub_degree=4))
        assert "far" in reach_default
        assert "hub" in reach_barrier and "far" not in reach_barrier

    def test_mentioned_alongside_never_rendered_or_traversed(self, tmp_path):
        gm = _pet_world(tmp_path)
        text = " ".join(gm.get_context_sentences("python", depth=2, max_sentences=20)).lower()
        assert "mentioned alongside" not in text
        nb = gm.neighbors("python", depth=1, skip_relations=frozenset({"mentioned_alongside"}))
        assert all(e.relation != "mentioned_alongside" for e in nb.get("python", []))

    def test_far_low_mention_edges_sorted_last(self, tmp_path):
        gm = _pet_world(tmp_path)
        sents = [x.lower() for x in gm.get_context_sentences("mochi", depth=2, max_sentences=40)]
        i_buds = next(i for i, x in enumerate(sents) if "best buds" in x)      # seed-incident
        i_huge = next(i for i, x in enumerate(sents) if "huge" in x)           # far node mentioned once
        assert i_buds < i_huge

    def test_neighbors_default_unchanged(self, tmp_path):
        gm = _pet_world(tmp_path)
        assert "gym" in gm.neighbors("mochi", depth=2)          # plain BFS still reaches through user
        assert "gym" not in gm.neighbors("mochi", depth=2, hub_barrier=True)
