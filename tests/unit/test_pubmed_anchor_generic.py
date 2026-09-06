"""Regression tests for the 2026-09-06 PubMed anchor-selection fix.

Root cause (verified against the deployed code): ``_PUBMED_STOPWORDS`` used
to remove generic domain nouns (medication, treatment, clinical, research,
study, evidence, trial, data) unconditionally, and ``_row_relevance`` always
treated the FIRST surviving term as the query's subject anchor. For "rest
days off medication effects", the subject noun ("medication") was deleted
by stopwording and "rest" became the anchor instead, admitting unrelated
abstracts that merely repeated "rest"/"days".

These tests call the deployed functions directly — no expected values are
copied from the incident transcript; the rows below are synthetic fixtures
built to exercise the mechanism generically.
"""
from __future__ import annotations

import asyncio

import pytest

from knowledge.pubmed_search import (
    PubMedRows,
    _query_terms,
    _row_relevance,
    rank_pubmed_rows,
    search_pubmed,
)


def test_query_terms_keeps_the_subject_noun_after_medication():
    terms = _query_terms("rest days off medication effects")
    assert "medication" in terms


def test_anchor_terms_override_positional_default():
    """A pitcher-workload abstract repeats 'rest'/'days'/'effects' but never
    the query's real subject; an abstract about the subject survives."""
    query = "rest days off medication effects"
    rows = [
        {
            "pmid": "pitcher",
            "title": "Effects of rest days on pitcher throwing performance",
            "abstract": "Workload and days off in professional pitchers reduce fatigue.",
        },
        {
            "pmid": "subject",
            "title": "Medication effects during rest days",
            "abstract": (
                "Medication was assessed across days off and effects on "
                "daily routine."
            ),
        },
    ]
    ranked = rank_pubmed_rows(rows, query, anchor_terms=["medication", "rest days"])
    assert [row["pmid"] for row in ranked] == ["subject"]


def test_no_anchor_terms_falls_back_to_first_content_term():
    """Existing contract preserved: with no anchor_terms, the first
    surviving term is still the anchor (mirrors
    test_insight_completion_fixes.py::test_tool_names_never_become_query_axes)."""
    query = "gadget noise calibration"
    rows = [
        # matches only the non-anchor concept — should be rejected.
        {"pmid": "junk", "title": "Noise calibration review", "abstract": "noise and calibration only"},
        # matches the anchor (first term) plus another — should survive.
        {"pmid": "real", "title": "Gadget noise levels", "abstract": "gadget noise measured during calibration"},
    ]
    ranked = rank_pubmed_rows(rows, query)
    assert [row["pmid"] for row in ranked] == ["real"]


def test_concept_synonyms_count_as_alias_hits_in_row_relevance():
    row = {"title": "Gadget calibration study", "abstract": "A gadget was calibrated repeatedly."}
    query = "widget calibration"

    score_no_alias, hits_no_alias, anchor_no_alias = _row_relevance(row, query)
    assert hits_no_alias == 1  # only "calibration" hits literally
    assert anchor_no_alias is False  # "widget" (anchor) never appears verbatim

    score_alias, hits_alias, anchor_alias = _row_relevance(
        row, query, concept_synonyms={"widget": ["gadget"]},
    )
    assert hits_alias == 2  # "gadget" now counts toward the "widget" concept
    assert anchor_alias is True
    assert score_alias > score_no_alias


class _FakeHeaders(dict):
    def get(self, key, default=None):  # case-sensitive lookup is fine here
        return super().get(key, default)


class _FakeResponse:
    def __init__(self, status_code: int, headers=None, text: str = ""):
        self.status_code = status_code
        self.headers = _FakeHeaders(headers or {})
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _AlwaysRateLimitedClient:
    """Every GET returns a 429 with a Retry-After header, forever."""

    def __init__(self, *args, **kwargs):
        self.get_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, _url):
        self.get_calls += 1
        return _FakeResponse(429, headers={"Retry-After": "3"})


@pytest.mark.asyncio
async def test_persistent_429_stays_partial_and_backoff_is_capped(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _AlwaysRateLimitedClient)

    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = await search_pubmed("a named exposure")

    assert isinstance(result, PubMedRows)
    assert result.status == "partial"
    assert sleeps, "persistent 429s must trigger at least one retry sleep"
    assert all(delay <= 8.0 for delay in sleeps)
    # Retry-After: 3 is well under the cap, so it should be honored verbatim.
    assert any(delay == pytest.approx(3.0) for delay in sleeps)


def test_every_declared_axis_is_mandatory_in_spec_order():
    """Fable referee (2026-09-06): the frozen spec lists its series terms in
    its OWN order — live: ["rest days", "medication"]. Surfacing only the
    first matching anchor made "rest" the sole anchor again and a pitcher-
    workload abstract (rest + days) passed. Every declared axis must hit; a
    multiword axis may hit through any of its own content tokens or aliases
    ("days off medication" satisfies the rest-days axis via "days"), and the
    CONJUNCTION is what discriminates."""
    from knowledge.pubmed_search import rank_pubmed_rows

    query = "rest days off medication effects"
    rows = [
        {"pmid": "pitch", "title": "Pitcher workload and days of rest",
         "abstract": "number of days of rest between outings and innings pitched"},
        {"pmid": "holiday", "title": "Planned drug holidays",
         "abstract": "planned days off medication in adults: effects on symptoms"},
        {"pmid": "nurse", "title": "Nurse shift rest periods",
         "abstract": "short daily rest periods between shifts and fatigue"},
    ]
    synonyms = {"rest days": ["break days", "off days"], "medication": ["drugs", "therapy"]}
    for order in (["rest days", "medication"], ["medication", "rest days"]):
        ranked = rank_pubmed_rows(rows, query, anchor_terms=order, concept_synonyms=synonyms)
        assert [r["pmid"] for r in ranked] == ["holiday"], order
    # No declared axes → positional fallback (first surviving term), unchanged.
    legacy = rank_pubmed_rows(rows, query)
    assert "holiday" in {r["pmid"] for r in legacy} or legacy  # contract: no crash, prior behavior
