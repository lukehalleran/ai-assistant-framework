"""Regression coverage for the 2026-09-10 web-search trigger gap.

All model and provider boundaries are fakes; this module makes no live calls
and writes no adaptive state.
"""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import utils.web_search_trigger as trigger
from core.agentic.gate import _is_info_seeking
from core.prompt.formatter import PromptFormatter
from core.prompt.gatherer_web import WebSearchMixin
from utils.retrieval_outcome import outcome_status


T1 = "The president says he will pay everyone 5000 if his party wins midterm. Oh boy"
T2 = "Uhm. Please investigate thank you"
PRIOR_CLAIM = (
    "User: BREAKING: IRGC says it struck a US carrier in the Gulf on Sept 5\n"
    "Daemon: That is a serious claim and should be verified."
)


@pytest.fixture(autouse=True)
def isolated_trigger(monkeypatch):
    monkeypatch.setattr(trigger, "_llm_trigger_cache", {})
    monkeypatch.setattr(trigger, "_semantic_search_boost", lambda query: 0)
    monkeypatch.setattr(trigger, "LLM_FIRST_ENABLED", True)


@pytest.mark.parametrize(
    "query",
    [
        T1,
        "The minister announced a new tax yesterday",
        "Congress passed the sanctions bill",
        "Police arrested the mayor",
    ],
)
def test_public_actor_statement_positive_shapes(query):
    assert trigger.public_actor_statement(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "I feel like the president is terrible",
        "my company announced layoffs",
        "nice",
        "thanks",
        "I feel way more fucked up then unusual do when I go just one night poor sleep.",
        "Ugh. Either I'm dying or super fucking constipated.",
        "Yeah that's like a warning sign I watch for.",
    ],
)
def test_public_actor_statement_rejects_personal_and_vent_shapes(query):
    assert trigger.public_actor_statement(query) is False


def test_fresh_public_question_remains_owned_by_freshness_rule():
    query = "The president was charged this week?"
    assert trigger.public_actor_statement(query) is False
    assert trigger.should_search_heuristic(query).source == "freshness_rule"


def test_statement_requests_classifier_without_forcing_search():
    decision = trigger.should_search_heuristic(T1)
    assert decision.should_search is False
    assert decision.consult_classifier is True
    assert decision.confidence == 0.0
    assert decision.matched_patterns == []
    assert "public-actor statement" in decision.reason.lower()


def _llm_response(should_search: bool):
    return trigger.LLMSearchTriggerResponse(
        should_search=should_search,
        confidence=0.9,
        reason="synthetic classifier verdict",
        search_terms=["president 5000 midterm claim"] if should_search else [],
        search_depth="quick",
        num_searches=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("should_search", [True, False])
async def test_statement_consults_classifier_without_teaching(monkeypatch, should_search):
    classifier = AsyncMock(return_value=_llm_response(should_search))
    monkeypatch.setattr(trigger, "_classify_with_llm_unified_shared", classifier)
    store = MagicMock()
    with patch("utils.adaptive_exemplars.get_store", return_value=store):
        decision = await trigger.analyze_for_web_search_llm(T1, model_manager=object())

    classifier.assert_awaited_once()
    assert decision.should_search is should_search
    assert decision.source == "llm"
    if not should_search:
        store.record.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "I feel way more fucked up then unusual do when I go just one night poor sleep. Is the benedryl still affecting me?",
        "Hey",
    ],
)
async def test_personal_state_and_greeting_still_skip_classifier(monkeypatch, query):
    classifier = AsyncMock(side_effect=AssertionError("classifier should not run"))
    monkeypatch.setattr(trigger, "_classify_with_llm_unified_shared", classifier)
    await trigger.analyze_for_web_search_llm(query, model_manager=object())
    classifier.assert_not_awaited()


@pytest.mark.parametrize(
    "query",
    ["Uhm. Please investigate thank you", "verify this", "fact check that", "look into it please", "is this real?"],
)
def test_verification_request_positive_shapes(query):
    assert trigger.is_verification_request(query) is True


@pytest.mark.parametrize(
    "query",
    ["I'll investigate later", "investigate how memory gating works in this assistant architecture today", "ok"],
)
def test_verification_request_negative_shapes(query):
    assert trigger.is_verification_request(query) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation_context, expected_calls", [(PRIOR_CLAIM, 1), (None, 0)])
async def test_verification_request_only_consults_with_prior_context(
    monkeypatch, conversation_context, expected_calls
):
    classifier = AsyncMock(return_value=_llm_response(True))
    monkeypatch.setattr(trigger, "_classify_with_llm_unified_shared", classifier)
    await trigger.analyze_for_web_search_llm(
        T2, model_manager=object(), conversation_context=conversation_context
    )
    assert classifier.await_count == expected_calls


def test_gate_treats_investigate_as_info_seeking():
    assert _is_info_seeking("Please investigate thank you") is True


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ({"web_search_decision": {"triggered": False, "reason": "No strong indicators"}}, "web_search=ON(not triggered)"),
        ({"web_search_decision": {"triggered": True, "results": 0}}, "web_search=ON(0 results)"),
        ({"web_search_decision": {"triggered": True, "results": 3}}, "web_search=ON(3 results)"),
        ({"web_search_decision": {"triggered": True, "error": "timeout"}}, "web_search=ON(error)"),
        ({"web_search_results": None}, "web_search=ON(no search this turn)"),
        ({"web_search_results": ""}, "web_search=ON(no search this turn)"),
        ({"web_search_results": []}, "web_search=ON(no search this turn)"),
    ],
)
def test_feature_inventory_reports_search_decision_honestly(monkeypatch, context, expected):
    monkeypatch.setattr("config.app_config.WEB_SEARCH_ENABLED", True)
    formatter = PromptFormatter(token_manager=MagicMock(), time_manager=None)
    out = formatter._build_feature_inventory(context)
    assert expected in out
    assert "web_search=ON(0)" not in out


def _gatherer(decision, *, error=None):
    manager = MagicMock()
    manager.is_available.return_value = True
    manager.multi_search = AsyncMock(
        side_effect=error,
        return_value=NS(has_results=True, pages=[NS(url="https://example.test")], total_credits_used=1, from_cache=False),
    )
    gatherer = WebSearchMixin.__new__(WebSearchMixin)
    gatherer.web_search_manager = manager
    gatherer.web_search_trigger_llm = AsyncMock(return_value=decision)
    gatherer.web_search_trigger = None
    gatherer.model_manager = object()
    gatherer.memory_id_map = {}
    return gatherer


@pytest.mark.asyncio
async def test_gatherer_exposes_not_triggered_decision():
    decision = trigger.WebSearchDecision(
        should_search=False,
        depth=trigger.WebSearchDepth.QUICK,
        confidence=0.0,
        reason="No strong indicators",
        matched_keywords=[],
        matched_patterns=[],
        source="heuristic",
    )
    gatherer = _gatherer(decision)
    assert await gatherer._get_web_search_results(T1) is None
    assert gatherer.last_web_decision == {
        "triggered": False,
        "source": "heuristic",
        "reason": "No strong indicators",
        "confidence": 0.0,
        "results": None,
        "error": None,
        # 2026-09-12 evidence receipt (review F4): additive fields — a
        # budget-blocked need survives, with the budget the decision used.
        "requested": False,
        "blocked": None,
        "budget_remaining": 100.0,
        "from_cache": False,
    }


@pytest.mark.asyncio
async def test_gatherer_exposes_search_exception():
    decision = trigger.WebSearchDecision(
        should_search=True,
        depth=trigger.WebSearchDepth.QUICK,
        confidence=0.9,
        reason="classifier requested search",
        matched_keywords=[],
        matched_patterns=[],
        source="llm",
    )
    gatherer = _gatherer(decision, error=TimeoutError("synthetic timeout"))
    result = await gatherer._get_web_search_results(T1)
    # CGR-20260913-007 #92 (F8b): a typed failure, not the bare `None` a
    # genuine empty search returns -- the receipt below is unchanged.
    assert outcome_status(result) == ("failed", "TimeoutError")
    assert result == []
    assert gatherer.last_web_decision["triggered"] is True
    assert gatherer.last_web_decision["source"] == "llm"
    assert gatherer.last_web_decision["results"] is None
    assert gatherer.last_web_decision["error"] == "TimeoutError"
