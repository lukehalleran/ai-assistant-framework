"""2026-10-08 (class: BC-05, BC-58, BC-01): an explicit search request is never
a continuation answer, the conversational-opener prefilter matches whole
words, and a bare imperative lookup routes to web search on the prior exchange.
Synthetic text only."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.agentic.gate import evaluate_agentic_gate
from utils.query_checker import is_continuation_answer
from utils.web_search_trigger import quick_prefilter_should_skip

QUESTION_REPLY = "I do not have notes on that. Which source did you mean?"


@pytest.fixture(autouse=True)
def _budget_ok(monkeypatch):
    import utils.web_search_trigger as wst
    monkeypatch.setattr(wst, "paid_search_block_reason", lambda *a, **k: None)


class _Corpus:
    def get_recent_memories(self, n):
        return [{"query": "What did the mayor say about the bridge?",
                 "response": "I am not sure about the bridge statement."}]


# --- S2.1 -------------------------------------------------------------------

def test_search_request_is_not_a_continuation_answer():
    assert is_continuation_answer("Yeah search it", QUESTION_REPLY) is False


def test_real_continuation_answers_still_are():
    assert is_continuation_answer("yes the second one", QUESTION_REPLY) is True
    assert is_continuation_answer("amplification", QUESTION_REPLY) is True


# --- S2.2 -------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "north korea news", "history of rome", "nobel prize 2026",
    "hire trends", "okay google it", "yeah search it",
])
def test_whole_word_opener_or_request_is_not_skipped(q):
    assert quick_prefilter_should_skip(q) is False


@pytest.mark.parametrize("q", ["ok", "yeah thanks", "no worries", "hello there", "sure thing"])
def test_pleasantries_still_skipped(q):
    assert quick_prefilter_should_skip(q) is True


# --- S2.3 -------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("q", ["search it", "Yeah search it", "look that up", "google it"])
async def test_bare_lookup_imperative_routes_to_web(q):
    d = await evaluate_agentic_gate(q, corpus_manager=_Corpus())
    assert d.should_trigger
    assert "web_search" in d.modes
    assert d.search_terms == []
    assert d.skip_initial_search is True


@pytest.mark.asyncio
async def test_bare_imperative_without_prior_exchange_has_no_arm():
    d = await evaluate_agentic_gate("search it", corpus_manager=None)
    assert "web_search" not in d.modes


@pytest.mark.asyncio
async def test_negated_bare_imperative_does_not_fire():
    d = await evaluate_agentic_gate("don't search it", corpus_manager=_Corpus())
    assert "web_search" not in d.modes
