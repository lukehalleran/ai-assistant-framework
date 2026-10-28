"""Behavioural sibling for ``TestThirdRetestClosures.
test_search_pubmed_threads_anchors_to_ranker`` in
tests/unit/test_insight_sweep_assistant_directed_exclusion.py (BC-63: it read
knowledge/pubmed_search.py as text and grepped the ``rank_pubmed_rows`` call).

Incident date: 2026-09-06 (a broadened rung anchored on its own first word;
farm-labor / bed-rest abstracts passed). Bug classes: BC-63, BC-62. THE deployed
``search_pubmed`` is driven with a fake HTTP layer and a recording ranker.
"""
import pytest

import knowledge.pubmed_search as ps

ESEARCH = "<eSearchResult><IdList><Id>111</Id></IdList></eSearchResult>"


class _Resp:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        return None


def _install(monkeypatch):
    async def fake_get(client, url, *a, **k):
        return _Resp(ESEARCH if "esearch" in url else "<PubmedArticleSet/>")

    seen = {}

    def fake_rank(rows, query, **kw):
        seen["query"] = query
        seen["kw"] = kw
        return ["ranked"]

    monkeypatch.setattr(ps, "_get_with_retry", fake_get)
    monkeypatch.setattr(ps, "parse_pubmed_articles", lambda text: [{"pmid": "111"}])
    monkeypatch.setattr(ps, "rank_pubmed_rows", fake_rank)
    return seen


@pytest.mark.asyncio
async def test_anchor_terms_and_synonyms_reach_the_ranker(monkeypatch):
    seen = _install(monkeypatch)
    out = await ps.search_pubmed(
        "rest AND days", max_results=3, anchor_terms=["fatigue", "recovery"],
        concept_synonyms={"rest": ["bed rest", "downtime"]},
    )
    assert out == ["ranked"]
    assert seen["kw"]["anchor_terms"] == ["fatigue", "recovery"]
    assert seen["kw"]["concept_synonyms"] == {"rest": ["bed rest", "downtime"]}
    assert seen["kw"]["max_results"] == 3


@pytest.mark.asyncio
async def test_defaults_still_reach_the_ranker_as_none(monkeypatch):
    seen = _install(monkeypatch)
    await ps.search_pubmed("rest AND days")
    assert seen["kw"]["anchor_terms"] is None
    assert seen["kw"]["concept_synonyms"] is None


@pytest.mark.asyncio
async def test_no_ids_never_reaches_the_ranker(monkeypatch):
    seen = _install(monkeypatch)

    async def empty(client, url, *a, **k):
        return _Resp("<eSearchResult><IdList/></eSearchResult>")

    monkeypatch.setattr(ps, "_get_with_retry", empty)
    out = await ps.search_pubmed("rest AND days", anchor_terms=["x"])
    assert "kw" not in seen
    assert getattr(out, "status", None) == "no_results"
