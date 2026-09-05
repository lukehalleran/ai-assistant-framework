"""Public-news verification survives unavailable classifiers; no live services."""

from unittest.mock import AsyncMock

import pytest

import utils.web_search_trigger as trigger


QUERY = "Farage caught doing crimes this week? Meeting with Russian proxies and receiving illegal money. This has to be releated"


@pytest.fixture(autouse=True)
def isolated_trigger(monkeypatch):
    monkeypatch.setattr(trigger, "_llm_trigger_cache", {})
    monkeypatch.setattr(trigger, "_semantic_search_boost", lambda query: 0)
    monkeypatch.setattr(trigger, "LLM_FIRST_ENABLED", True)


@pytest.mark.parametrize("query", [
    QUERY, "Did the minister resign this week?",
    "Has Acme been charged recently?", "The company recalled batteries yesterday. Is that true?",
    "Can you check if Farage was arrested this week?",
    "Did you see that the president was charged this week?",
    "Could you look up whether parliament passed the sanctions bill this week",
    "Can you check if the company announced the merger this week?",
    "Did the company report earnings this week? What was announced?",
])
def test_dated_public_questions_require_evidence(query):
    assert trigger.requires_fresh_public_evidence(query)
    assert trigger.should_search_heuristic(query).should_search


@pytest.mark.parametrize("query", [
    "Can you check if Farage was arrested this week?",
    "Did you see that the president was charged this week?",
    "Could you look up whether parliament passed the sanctions bill this week",
])
def test_second_person_request_wrapper_routes_via_freshness_rule(query):
    assert trigger.should_search_heuristic(query).source == "freshness_rule"


@pytest.mark.parametrize("query", [
    "I felt tired this week", "Did I mention my meeting this week?",
    "Check my syllabus for this week's homework", "What should I do this week?",
    "My court appointment is today. What did I tell you about it?",
    "The government announced this last century. Explain the history.",
    "Is that confirmed?",  # Needs prior-topic resolution, not standalone guessing.
    "Don't search: did the minister resign this week?",
    "What is a company merger?",
    "What did you say yesterday about the courts?",
    "Can you check my email from the court this week?",
    "Is the homework due this week? The syllabus reports Sep 13.",
    "Can you check what my company announced this week about our schedule?",
])
def test_personal_mixed_static_and_unresolved_queries_use_existing_routing(query):
    assert not trigger.requires_fresh_public_evidence(query)


@pytest.mark.asyncio
async def test_exact_incident_does_not_need_classifier(monkeypatch):
    classifier = AsyncMock(side_effect=AssertionError("unnecessary model call"))
    monkeypatch.setattr(trigger, "_classify_with_llm_unified", classifier)
    decision = await trigger.analyze_for_web_search_llm(QUERY, model_manager=object())
    assert decision.should_search
    assert decision.search_terms == [QUERY]
    classifier.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    QUERY, "Can you check if Farage was arrested this week?",
])
async def test_agentic_gate_routes_exact_incident_without_model(query):
    from core.agentic.gate import evaluate_agentic_gate
    decision = await evaluate_agentic_gate(query)
    assert decision.should_trigger
    assert "web_search" in decision.modes
    assert decision.search_terms == [query]
    assert not decision.skip_initial_search


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [{"web_search_enabled": False}, {"crisis_level": "HIGH"}, {"crisis_level": "MEDIUM"}])
async def test_cached_positive_cannot_bypass_current_policy(policy):
    initial = await trigger.analyze_for_web_search_llm(QUERY, model_manager=object())
    assert initial.should_search and trigger._llm_trigger_cache
    restricted = await trigger.analyze_for_web_search_llm(QUERY, model_manager=object(), **policy)
    assert not restricted.should_search


@pytest.mark.asyncio
async def test_failed_classifier_is_not_reported_as_successful_classification(monkeypatch):
    monkeypatch.setattr(trigger, "_classify_with_llm_unified", AsyncMock(return_value=None))
    decision = await trigger.analyze_for_web_search_llm("Recent developments in optics?", model_manager=object())
    assert decision.source == "fallback"
    assert "Classifier unavailable" in decision.reason
