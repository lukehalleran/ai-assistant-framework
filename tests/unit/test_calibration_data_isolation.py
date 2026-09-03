"""Synthetic calibration data must never enter — or silently stay in — live stores.

2026-09-02 audit: 48 `source=test_calibration` facts (D&D, running, a brewery
job…) were live in the owner's facts collection and knowledge graph.
"""

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load("generate_test_facts")
purge = _load("purge_calibration_facts")


# --- generate_test_facts.py: sandbox only ---------------------------------

def test_generator_requires_sandbox_dir():
    with pytest.raises(gen.LiveStoreRefused):
        gen.resolve_sandbox_paths(None)
    with pytest.raises(gen.LiveStoreRefused):
        gen.run(dry_run=False, sandbox_dir=None)


def test_generator_paths_live_under_sandbox(tmp_path):
    paths = gen.resolve_sandbox_paths(tmp_path)
    for p in paths.values():
        assert tmp_path.resolve() in Path(p).parents


def test_generator_refuses_live_store_paths(tmp_path):
    live_chroma = tmp_path / "live" / "chroma"
    live_graph = tmp_path / "live" / "knowledge_graph.json"
    live_aliases = tmp_path / "live" / "entity_aliases.json"
    # Sandbox rooted AT the live data dir -> refused (graph path coincides).
    with pytest.raises(gen.LiveStoreRefused):
        gen.refuse_live_paths(
            gen.resolve_sandbox_paths(tmp_path / "live"),
            live_chroma=live_chroma, live_graph=live_graph, live_aliases=live_aliases,
        )
    # Sandbox INSIDE the live chroma tree -> refused.
    with pytest.raises(gen.LiveStoreRefused):
        gen.refuse_live_paths(
            gen.resolve_sandbox_paths(live_chroma / "sub"),
            live_chroma=live_chroma, live_graph=live_graph, live_aliases=live_aliases,
        )
    # A genuinely separate directory is fine.
    ok = gen.refuse_live_paths(
        gen.resolve_sandbox_paths(tmp_path / "elsewhere"),
        live_chroma=live_chroma, live_graph=live_graph, live_aliases=live_aliases,
    )
    assert ok["chroma"] == (tmp_path / "elsewhere" / "chroma").resolve()


def test_dry_run_never_touches_stores(capsys):
    gen.run(dry_run=True, sandbox_dir=None)  # no sandbox needed for a report
    out = capsys.readouterr().out
    assert "Dry run: True" in out


# --- purge_calibration_facts.py: selection is by the synthetic marker only ---

_DOCS = [
    {"id": "f1", "content": "user | plays | Dungeons and Dragons", "metadata": {"source": "test_calibration"}},
    {"id": "f2", "content": "user | hobby | running", "metadata": {"source": "test_calibration"}},
    {"id": "f3", "content": "user | hobby | running", "metadata": {"source": "llm_shutdown"}},
    {"id": "f4", "content": "user | likes | Python", "metadata": {}},
]
_GRAPH = {
    "nodes": {"user": {}, "dungeons_and_dragons": {}, "running": {}, "python": {}, "sam": {}},
    "edges": [
        {"source_id": "user", "relation": "plays", "target_id": "dungeons_and_dragons", "source_fact_ids": ["f1"]},
        {"source_id": "user", "relation": "hobby", "target_id": "running", "source_fact_ids": ["f2", "f3"]},
        {"source_id": "user", "relation": "likes", "target_id": "python", "source_fact_ids": ["f4"]},
        {"source_id": "user", "relation": "sibling_of", "target_id": "sam", "source_fact_ids": []},
    ],
}


def test_select_by_marker_not_content():
    hits = purge.select_calibration_facts(_DOCS)
    assert [h["id"] for h in hits] == ["f1", "f2"]  # f3 says "running" but is real


def test_edges_split_into_fully_synthetic_and_mixed():
    full, mixed = purge.select_calibration_edges(_GRAPH, ["f1", "f2"])
    assert [(e["source_id"], e["target_id"]) for e in full] == [("user", "dungeons_and_dragons")]
    assert [(e["target_id"], e["synthetic_fact_ids"]) for e in mixed] == [("running", ["f2"])]


def test_orphan_candidates_exclude_user_and_still_connected_nodes():
    full, _ = purge.select_calibration_edges(_GRAPH, ["f1", "f2"])
    assert purge.orphan_node_candidates(_GRAPH, full) == ["dungeons_and_dragons"]
