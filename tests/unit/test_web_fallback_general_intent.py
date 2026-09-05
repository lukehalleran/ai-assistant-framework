"""Focused coverage for web-search fallback intent gating.

These tests construct the mixin directly so no model, store, or provider is
initialized.  The manager and trigger are fully mocked as well.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import core.prompt.gatherer_web as gatherer_web
from core.prompt.gatherer_web import WebSearchMixin
from knowledge.web_search_manager import MultiSearchResult, WebPage
from utils.web_search_trigger import WebSearchDecision, WebSearchDepth


QUERY = "Farage caught doing crimes this week meeting Russian proxies illegal money?"


def _decision(should_search: bool) -> WebSearchDecision:
    return WebSearchDecision(
        should_search=should_search,
        depth=WebSearchDepth.QUICK,
        confidence=0.9 if should_search else 0.1,
        reason="mock trigger",
        matched_keywords=[],
        matched_patterns=[],
    )


@pytest.fixture
def mocked_gatherer():
    manager = MagicMock()
    manager.is_available.return_value = True
    result = MultiSearchResult(
        original_query=QUERY,
        pages=[WebPage(url="https://example.test/news", title="News", content="facts")],
    )
    manager.multi_search = AsyncMock(return_value=result)

    trigger = MagicMock()
    gatherer = WebSearchMixin.__new__(WebSearchMixin)
    gatherer.web_search_manager = manager
    gatherer.web_search_trigger = trigger
    # Exercise the shared synchronous trigger without constructing a model.
    gatherer.web_search_trigger_llm = None
    gatherer.model_manager = MagicMock()
    gatherer.memory_id_map = {}
    return gatherer, manager, trigger


@pytest.mark.asyncio
@pytest.mark.parametrize("intent_type", ["general", "unknown"])
async def test_ambiguous_intents_consult_positive_shared_trigger(
    mocked_gatherer, intent_type
):
    gatherer, manager, trigger = mocked_gatherer
    trigger.return_value = _decision(True)

    result = await gatherer._get_web_search_results(QUERY, intent_type=intent_type)

    assert result is manager.multi_search.return_value
    trigger.assert_called_once_with(QUERY)
    manager.multi_search.assert_awaited_once()


@pytest.mark.asyncio
async def test_general_negative_shared_trigger_does_not_search(mocked_gatherer):
    gatherer, manager, trigger = mocked_gatherer
    trigger.return_value = _decision(False)

    result = await gatherer._get_web_search_results(QUERY, intent_type="general")

    assert result is None
    trigger.assert_called_once_with(QUERY)
    manager.multi_search.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "crisis_level"),
    [(False, None), (True, "HIGH"), (True, "MEDIUM")],
)
async def test_disabled_or_crisis_paths_do_not_call_model_or_network(
    mocked_gatherer, monkeypatch, enabled, crisis_level
):
    gatherer, manager, trigger = mocked_gatherer
    monkeypatch.setattr(gatherer_web, "WEB_SEARCH_ENABLED", enabled)
    gatherer.web_search_trigger_llm = AsyncMock(side_effect=AssertionError("model called"))
    gatherer.web_search_trigger = MagicMock(side_effect=AssertionError("trigger called"))

    result = await gatherer._get_web_search_results(
        QUERY, crisis_level=crisis_level, intent_type="general"
    )

    assert result is None
    manager.is_available.assert_not_called()
    manager.multi_search.assert_not_awaited()
    gatherer.web_search_trigger.assert_not_called()
    gatherer.web_search_trigger_llm.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent_type", ["casual_social", "meta_conversational", "emotional_support"]
)
async def test_explicit_non_search_intents_remain_hard_skips(
    mocked_gatherer, intent_type
):
    gatherer, manager, trigger = mocked_gatherer
    trigger.return_value = _decision(True)

    result = await gatherer._get_web_search_results(QUERY, intent_type=intent_type)

    assert result is None
    trigger.assert_not_called()
    manager.is_available.assert_not_called()
    manager.multi_search.assert_not_awaited()
