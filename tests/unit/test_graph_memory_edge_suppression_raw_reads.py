"""edge_is_suppressed reads raw node attributes, never builds GraphNode models.

2026-10-17, class BC-41 (sync work on the event loop; the cost was per-edge
pydantic construction in a hub subgraph: 4688 ms live at graph_memory.py:638,
759 ms in GraphNode.from_dict).  Verdicts must be unchanged for a quarantined
node, a quarantined edge, a deictic endpoint, a species conflict and a clean
edge, and ``GraphNode.from_dict`` must be called ZERO times during
``edge_is_suppressed`` and during the edge filter of ``get_context_sentences``
(the final display-name lookups, at most max_sentences, still use get_entity).
"""
from datetime import datetime

import pytest

from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge, GraphNode


def _edge(src, rel, tgt, **md):
    now = datetime.now()
    return GraphEdge(source_id=src, relation=rel, target_id=tgt, weight=1.0,
                     first_seen=now, last_seen=now, metadata=md)


@pytest.fixture
def gm(tmp_path):
    g = GraphMemory(persist_path=str(tmp_path / "g.json"))
    for nid, name, typ, md in (
        ("user", "User", "person", {}),
        ("tamsin", "Tamsin", "person", {}),
        ("hidden", "Hidden", "other", {"curation_quarantined": True}),
        ("rowan", "Rowan", "animal", {"species": "cat"}),
        ("today", "Today", "other", {}),
        ("kiln", "Kiln", "project", {}),
    ):
        g.add_entity(GraphNode(entity_id=nid, display_name=name, entity_type=typ, metadata=md))
    return g


def _count_from_dict(monkeypatch):
    calls = {"n": 0}
    orig = GraphNode.from_dict.__func__

    def counting(cls, entity_id, data):
        calls["n"] += 1
        return orig(cls, entity_id, data)

    monkeypatch.setattr(GraphNode, "from_dict", classmethod(counting))
    return calls


CASES = [
    ("quarantined_node", ("user", "knows", "hidden", {}), True),
    ("quarantined_edge", ("user", "knows", "tamsin", {"curation_quarantined": True}), True),
    ("deictic", ("user", "feels_better", "today", {}), True),
    ("species_conflict", ("user", "has_dog", "rowan", {}), True),
    ("clean", ("user", "works_on", "kiln", {}), False),
]


@pytest.mark.parametrize("name,spec,expected", CASES, ids=[c[0] for c in CASES])
def test_verdicts_unchanged_and_no_model_construction(gm, monkeypatch, name, spec, expected):
    src, rel, tgt, md = spec
    edge = _edge(src, rel, tgt, **md)
    calls = _count_from_dict(monkeypatch)
    assert gm.edge_is_suppressed(edge) is expected
    assert calls["n"] == 0


def test_mixed_case_ids_normalize_like_get_entity(gm):
    assert gm.edge_is_suppressed(_edge("user", "knows", " HIDDEN ")) is True


def test_context_sentence_filter_builds_no_models_for_edges(gm, monkeypatch):
    for tgt in ("tamsin", "hidden", "kiln", "rowan"):
        rel = "has_dog" if tgt == "rowan" else "knows"
        gm.add_relation(_edge("user", rel, tgt))
    calls = _count_from_dict(monkeypatch)
    out = gm.get_context_sentences("user", depth=1, max_sentences=2)
    assert len(out) == 2
    # only the <= max_sentences display-name lookups (2 per sentence) may build models
    assert calls["n"] <= 2 * 2
    joined = " ".join(gm.get_context_sentences("user", depth=1, max_sentences=10))
    assert "Hidden" not in joined and "Rowan" not in joined
    assert "Kiln" in joined and "Tamsin" in joined
