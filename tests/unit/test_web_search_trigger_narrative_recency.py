"""Recency words inside first-person narration are not a world-recency signal.

Incident 2026-10-24 (class: BC-01 keyword scored without its clause's subject;
BC-05 arm precedence -- a high keyword confidence skipped the classifier): a
personal check-in scored two STRONG recency keywords (conf 0.80), skipped the
LLM trigger and paid for four Tavily sub-queries. Drives the deployed
should_search_heuristic / analyze_for_web_search_llm; the classifier is a fake.
"""

import textwrap
from unittest.mock import AsyncMock

import pytest

import utils.web_search_trigger as trigger

INCIDENT = (
    "Alright. Yeah so that was mostly just me checking that things came through "
    "okay after the update last night. Nothing urgent on my end right now, I'm "
    "mostly just poking around and seeing how everything feels after the changes. "
    "I'll probably get back into the homework stuff later today once I've had some "
    "coffee and settled in a bit. Anyway, how are things looking on your side?"
)
VARIANT = (
    "Ok so I mostly just rested after the long shift last night, nothing big on my "
    "side right now, I'll probably tackle the reading later today. "
    "How's your day going?"
)
# Narrative that still carries high keyword confidence (two fast-changing
# topics + a recent-year mention = 0.90) so the item-2 fast path is exercised.
NARRATIVE_HIGH_KEYWORD = (
    "I spent the evening reading about bitcoin and ethereum, nothing big on my "
    "side, and I'll go through my 2026 budget charts later once I settle in with "
    "some coffee"
)
LOOKUPS = [
    "what's the weather today",
    "latest news on the election",
    "price of bitcoin today",
]


def _wrapped(text):
    return textwrap.indent(textwrap.fill(text, 40), "    ")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(trigger, "_llm_trigger_cache", {})
    monkeypatch.setattr(trigger, "_semantic_search_boost", lambda q: 0)
    monkeypatch.setattr(trigger, "LLM_FIRST_ENABLED", True)


def _llm(should_search):
    return trigger.LLMSearchTriggerResponse(
        should_search=should_search, confidence=0.9, reason="synthetic verdict",
        search_terms=["bitcoin"] if should_search else [], search_depth="quick",
        num_searches=1,
    )


@pytest.mark.parametrize("fmt", [lambda t: t, _wrapped], ids=["clean", "wrapped"])
@pytest.mark.parametrize("text", [INCIDENT, VARIANT])
def test_narration_recency_words_do_not_search(text, fmt):
    d = trigger.should_search_heuristic(fmt(text))
    assert d.should_search is False
    assert d.confidence < 0.5
    assert not any(k in d.matched_keywords for k in ("today", "right now"))


@pytest.mark.parametrize("fmt", [lambda t: t, _wrapped], ids=["clean", "wrapped"])
@pytest.mark.parametrize("query", LOOKUPS)
def test_world_recency_lookups_still_search(query, fmt):
    assert trigger.should_search_heuristic(fmt(query)).should_search is True


@pytest.mark.parametrize("fmt", [lambda t: t, _wrapped], ids=["clean", "wrapped"])
@pytest.mark.parametrize("query", [
    "I'm wondering what's happening in Gaza right now",
    "I wonder what the weather is doing right now",
])
def test_first_person_frame_with_embedded_question_keeps_recency(query, fmt):
    d = trigger.should_search_heuristic(fmt(query))
    assert d.confidence > 0
    assert "right now" in d.matched_keywords


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", [lambda t: t, _wrapped], ids=["clean", "wrapped"])
async def test_narrative_with_high_keyword_confidence_consults_classifier(monkeypatch, fmt):
    text = fmt(NARRATIVE_HIGH_KEYWORD)
    assert trigger.should_search_heuristic(text).confidence >= 0.7  # premise
    classifier = AsyncMock(return_value=_llm(False))
    monkeypatch.setattr(trigger, "_classify_with_llm_unified_shared", classifier)
    decision = await trigger.analyze_for_web_search_llm(text, model_manager=object())
    classifier.assert_awaited_once()
    assert decision.should_search is False


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [INCIDENT, VARIANT])
async def test_incident_never_searches_even_if_classifier_unavailable(monkeypatch, text):
    classifier = AsyncMock(return_value=None)
    monkeypatch.setattr(trigger, "_classify_with_llm_unified_shared", classifier)
    decision = await trigger.analyze_for_web_search_llm(text, model_manager=object())
    assert decision.should_search is False


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["latest news on the election", "price of bitcoin today"])
async def test_world_lookup_keeps_fast_path(monkeypatch, query):
    classifier = AsyncMock(side_effect=AssertionError("classifier must not run"))
    monkeypatch.setattr(trigger, "_classify_with_llm_unified_shared", classifier)
    decision = await trigger.analyze_for_web_search_llm(query, model_manager=object())
    classifier.assert_not_awaited()
    assert decision.should_search is True


def test_narrative_predicate_boundaries():
    assert trigger.is_first_person_narrative(INCIDENT) is True
    assert trigger.is_first_person_narrative("latest news on the election") is False
    # a world question appended to narration is an information request
    assert trigger.is_first_person_narrative(
        "I spent the evening reading about bitcoin and ethereum, nothing big on my "
        "side right now. Who won the election?") is False
