"""2026-10-08 (class: BC-05, BC-47, BC-72): the heuristic's active-suppression
veto is decided BEFORE the LLM call when there is no pattern-analysis
candidate; an unparseable classifier answer is labelled as such."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import utils.web_search_trigger as wst
from utils.web_search_trigger import WebSearchDecision, WebSearchDepth

QUERY = "I feel like the weather has been really grey lately and I am tired"


def _suppression():
    return WebSearchDecision(
        should_search=False, depth=WebSearchDepth.QUICK, confidence=0.3,
        reason="matched suppression", matched_keywords=["x"],
        matched_patterns=["i feel"], search_terms=[], num_searches=0, source="heuristic",
    )


def _manager(response):
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=response)
    return mm


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    wst._llm_trigger_cache.clear()
    monkeypatch.setattr(wst, "LLM_FIRST_ENABLED", True)
    monkeypatch.setattr(wst, "should_search_heuristic", lambda q: _suppression())
    monkeypatch.setattr(wst, "_looks_like_pattern_candidate", lambda q: False)
    monkeypatch.setattr(wst, "quick_prefilter_should_skip", lambda q: False)


async def _run(mm):
    return await wst.analyze_for_web_search_llm(
        query=QUERY, model_manager=mm, remaining_credits=50.0, web_search_enabled=True)


@pytest.mark.asyncio
async def test_suppression_vetoes_without_consulting_llm():
    mm = _manager("{}")
    decision = await _run(mm)
    mm.generate_once.assert_not_awaited()
    assert decision.should_search is False
    assert decision.reason.startswith("Heuristic veto")
    assert decision.reason.endswith("LLM not consulted")
    assert decision.source == "heuristic"
    # cached: a repeat stays LLM-free
    again = await _run(mm)
    mm.generate_once.assert_not_awaited()
    assert again.reason == decision.reason


@pytest.mark.asyncio
async def test_pattern_candidate_still_consults_llm(monkeypatch):
    monkeypatch.setattr(wst, "_looks_like_pattern_candidate", lambda q: True)
    mm = _manager(json.dumps({
        "should_search": True, "confidence": 0.9, "reason": "r",
        "search_terms": ["grey weather"], "search_depth": "quick", "num_searches": 1,
    }))
    decision = await _run(mm)
    mm.generate_once.assert_awaited()
    assert decision.should_search is False
    assert decision.reason.startswith("Heuristic veto")
    assert decision.reason.endswith("LLM overridden")


@pytest.mark.asyncio
async def test_unparseable_vs_unavailable_labels(monkeypatch):
    # No active suppression -> the LLM is consulted.
    neutral = WebSearchDecision(
        should_search=False, depth=WebSearchDepth.QUICK, confidence=0.1,
        reason="No strong indicators", matched_keywords=["x"], matched_patterns=[],
        search_terms=[], num_searches=0, source="heuristic")
    monkeypatch.setattr(wst, "should_search_heuristic", lambda q: neutral)
    bad = await _run(_manager("this is not json"))
    assert bad.source == "fallback"
    assert bad.reason.startswith("Classifier output unparseable; ")
    wst._llm_trigger_cache.clear()
    empty = await _run(_manager(""))
    assert empty.source == "fallback"
    assert empty.reason.startswith("Classifier unavailable; ")
