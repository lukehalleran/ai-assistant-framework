"""Metric-aware similarity normalization for the wiki FAISS search layer.

Regression coverage for project_wiki_faiss_l2_metric_mismatch: the wiki index is
built with an L2 metric (build_faiss_index.py), so FAISS returns SQUARED
DISTANCES (smaller = closer). The query layer used to store those verbatim as
"similarity" and sort descending, which surfaced the FARTHEST neighbors first
and inverted every downstream 0-1 cosine threshold. _to_similarity() now maps
L2 distance -> cosine (1 - d/2) and passes inner-product through.

These tests build tiny synthetic indexes (no 2 GB wiki file required).
"""

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from knowledge.semantic_search import SemanticSearchIndex


def _normed(rows):
    a = np.asarray(rows, dtype="float32")
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return np.ascontiguousarray(a, dtype="float32")


# doc0 == query (cos 1.0), doc1 cos 0.8, doc2/doc3 cos 0.0
DOCS = _normed([
    [1.0, 0.0, 0.0, 0.0],
    [0.8, 0.6, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])
QUERY = _normed([[1.0, 0.0, 0.0, 0.0]])


def _make_index(metric):
    idx = (faiss.IndexFlatIP if metric == faiss.METRIC_INNER_PRODUCT
           else faiss.IndexFlatL2)(DOCS.shape[1])
    idx.add(DOCS)
    return idx


def _stub_search_instance(metric):
    inst = SemanticSearchIndex()
    inst.index = _make_index(metric)
    inst._metric = int(inst.index.metric_type)
    inst._total_rows = DOCS.shape[0]
    inst.loaded = True
    inst._encode_query = lambda query: QUERY.copy()
    inst._read_rows = lambda indices, columns=None: {
        i: {"text": f"doc{i}", "title": f"title{i}"} for i in indices
    }
    return inst


# --- _to_similarity unit behavior ---

def test_to_similarity_l2_conversion():
    inst = SemanticSearchIndex()
    inst._metric = faiss.METRIC_L2
    assert inst._to_similarity(0.0) == pytest.approx(1.0)    # identical
    assert inst._to_similarity(2.0) == pytest.approx(0.0)    # orthogonal
    assert inst._to_similarity(4.0) == pytest.approx(-1.0)   # opposite
    assert inst._to_similarity(10.0) == -1.0                 # clamped


def test_to_similarity_ip_passthrough():
    inst = SemanticSearchIndex()
    inst._metric = faiss.METRIC_INNER_PRODUCT
    assert inst._to_similarity(0.83) == pytest.approx(0.83)
    assert inst._to_similarity(-0.2) == pytest.approx(-0.2)


def test_to_similarity_unknown_metric_passthrough():
    inst = SemanticSearchIndex()
    inst._metric = None  # not loaded
    assert inst._to_similarity(0.42) == pytest.approx(0.42)


# --- end-to-end search ordering ---

@pytest.mark.parametrize("metric", [faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT])
def test_search_orders_nearest_first(metric):
    inst = _stub_search_instance(metric)
    results = inst.search("anything", k=4)
    assert results, "expected results"

    titles = [r["title"] for r in results]
    sims = [r["similarity"] for r in results]

    # closest doc (identical to query) is first, with cosine ~1.0 — NOT inverted
    assert titles[0] == "title0"
    assert sims[0] == pytest.approx(1.0, abs=1e-4)
    # cos-0.8 doc is second
    assert titles[1] == "title1"
    assert sims[1] == pytest.approx(0.8, abs=1e-4)
    # descending (best first) and within cosine range
    assert sims == sorted(sims, reverse=True)
    assert all(-1.0 - 1e-6 <= s <= 1.0 + 1e-6 for s in sims)


def test_search_l2_not_inverted_regression():
    """Pre-fix the L2 path returned the farthest neighbor first with a squared
    distance mislabeled as similarity. Guard the polarity directly."""
    inst = _stub_search_instance(faiss.METRIC_L2)
    results = inst.search("anything", k=4)
    assert results[0]["title"] == "title0"                       # closest first
    assert results[0]["similarity"] > results[-1]["similarity"]  # not flipped
    # the mislabeled-distance failure mode would have put sims[0] near 0 (dist),
    # not near 1 (cosine); pin that it reads as a real similarity.
    assert results[0]["similarity"] > 0.9
