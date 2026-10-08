"""2026-10-08 (class: BC-46, BC-25): an elliptical follow-up whose proposed
search terms add no resolved subject never searches (live: "President one of
them" -> 3 generic searches, 6 credits). Synthetic text only."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import utils.web_search_trigger as wst

T6_QUERY = "President one of them"
T6_TERMS = ["President news October 2026",
            "current president statements October 2026",
            "President one of them news"]


def test_generic_terms_on_referential_query_are_suppressed():
    assert wst.terms_lack_resolved_referent(T6_QUERY, T6_TERMS) is True


def test_terms_with_resolved_subject_pass():
    assert wst.terms_lack_resolved_referent(
        T6_QUERY, ["Trump comments Cornell Jane Doe case"]) is False


def test_non_referential_query_is_untouched():
    assert wst.terms_lack_resolved_referent(
        "president election results", ["president election results October 2026"]) is False


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    wst._llm_trigger_cache.clear()
    monkeypatch.setattr(wst, "quick_prefilter_should_skip", lambda q: False)
    monkeypatch.setattr(wst.institution_resolver, "scope_identity_terms",
                        lambda terms, *a, **k: terms)


async def _run(terms):
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=json.dumps({
        "should_search": True, "confidence": 0.9, "reason": "r",
        "search_terms": terms,
    }))
    return await wst._classify_with_llm_unified(
        T6_QUERY, mm, conversation_context="User: hello\nAssistant: hi")


@pytest.mark.asyncio
async def test_llm_path_suppresses_unresolved_referent():
    parsed = await _run(T6_TERMS)
    assert parsed.should_search is False and parsed.search_terms == []


@pytest.mark.asyncio
async def test_llm_path_keeps_resolved_terms():
    parsed = await _run(["Trump comments Cornell Jane Doe case"])
    assert parsed.should_search is True
