"""Behavioural siblings for the builder's retrieval-override wiring (source-text rows).

Incident dates: 2026-08-29 (calendar-turn memory top-up refilled 1 -> 30 ungated
memories), 2026-09-03 (casual/emotional intents zero passive email, graph and notes
via override keys; rollup/mood notes gated on the turn). Class: BC-63 (a grep for
the call text stands in for the behaviour), BC-24 (floor vs intent cap).

Each test drives THE deployed ``UnifiedPromptBuilder.build_prompt`` (same harness as
test_independent_prompt_audit.full_builder) with recording collaborators and asserts
what the gatherer was ASKED, so renaming a variable cannot break it but dropping the
wiring does. Fixture text is synthetic; each live-style query is asserted clean and
wrapped/indented.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits


def _builder(monkeypatch, recent=()):
    builder = full_builder(monkeypatch, list(recent))
    g = builder.context_gatherer
    pool = list(recent)
    g._get_recent_conversations = AsyncMock(side_effect=lambda n=len(pool): pool[:n])
    g._get_semantic_memories = AsyncMock(return_value=[])
    g.get_relevant_emails = AsyncMock(return_value=[])
    g.get_graph_context = AsyncMock(return_value=[])
    g.get_personal_notes = AsyncMock(return_value=[])
    return builder


def _recents(n):
    return [{"query": f"Synthetic question number {i} about topic{i}",
             "response": f"Synthetic answer {i}.",
             "timestamp": f"2026-10-2{i % 9}T10:00:00"} for i in range(n)]


# ---------------------------------------------------------------- memory top-up
@pytest.mark.asyncio
async def test_memory_topup_is_bounded_by_floor_not_intent_cap(monkeypatch):
    """Intent cap 20, no gated memories: the top-up adds MEMORY_TOPUP_FLOOR, not 20."""
    from core.prompt.builder import MEMORY_TOPUP_FLOOR
    builder = _builder(monkeypatch, _recents(30))
    ro = retrieval_limits() | {"max_mems": 20}
    result = await builder.build_prompt("Estimate the homework effort", retrieval_overrides=ro)
    assert "_build_time" in result, "builder returned its error fallback"
    assert len(result["memories"]) == MEMORY_TOPUP_FLOOR


@pytest.mark.asyncio
async def test_memory_topup_never_exceeds_a_smaller_intent_cap(monkeypatch):
    builder = _builder(monkeypatch, _recents(30))
    ro = retrieval_limits() | {"max_mems": 1}
    result = await builder.build_prompt("Estimate the homework effort", retrieval_overrides=ro)
    assert len(result["memories"]) == 1


# ------------------------------------------------------- passive email / graph keys
@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "did the advisor reply about registration?",
    "    did the advisor reply\n    about registration?\n   ",
])
async def test_max_relevant_emails_override_reaches_the_gatherer(monkeypatch, query):
    builder = _builder(monkeypatch)
    monkeypatch.setattr("config.app_config.EMAIL_PASSIVE_CONTEXT_ENABLED", True)
    await builder.build_prompt(query, retrieval_overrides=retrieval_limits() | {"max_relevant_emails": 2})
    builder.context_gatherer.get_relevant_emails.assert_awaited_once()
    assert builder.context_gatherer.get_relevant_emails.await_args.args[1] == 2


@pytest.mark.asyncio
async def test_zero_relevant_emails_override_skips_the_gatherer(monkeypatch):
    builder = _builder(monkeypatch)
    monkeypatch.setattr("config.app_config.EMAIL_PASSIVE_CONTEXT_ENABLED", True)
    await builder.build_prompt("did the advisor reply about registration?",
                               retrieval_overrides=retrieval_limits() | {"max_relevant_emails": 0})
    builder.context_gatherer.get_relevant_emails.assert_not_awaited()


@pytest.mark.asyncio
async def test_max_graph_sentences_override_reaches_the_gatherer(monkeypatch):
    builder = _builder(monkeypatch)
    await builder.build_prompt("Estimate the homework effort",
                               retrieval_overrides=retrieval_limits() | {"max_graph_sentences": 4})
    builder.context_gatherer.get_graph_context.assert_awaited_once_with("Estimate the homework effort", 4)


@pytest.mark.asyncio
async def test_zero_graph_sentences_override_skips_the_gatherer(monkeypatch):
    builder = _builder(monkeypatch)
    await builder.build_prompt("Estimate the homework effort", retrieval_overrides=retrieval_limits())
    builder.context_gatherer.get_graph_context.assert_not_awaited()


# ------------------------------------------------------ personal-notes gate flags
async def _notes_kwargs(monkeypatch, query):
    builder = _builder(monkeypatch)
    ro = retrieval_limits() | {"max_personal_notes": 3}
    await builder.build_prompt(query, retrieval_overrides=ro)
    builder.context_gatherer.get_personal_notes.assert_awaited_once()
    return builder.context_gatherer.get_personal_notes.await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "what did March look like?",
    "    what did March\n    look like?\n  ",
])
async def test_question_shaped_query_allows_rollup_notes(monkeypatch, query):
    assert (await _notes_kwargs(monkeypatch, query))["allow_rollups"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "Biscuit chased a moth",
    "    Biscuit chased\n    a moth\n  ",
])
async def test_plain_statement_withholds_rollup_notes(monkeypatch, query):
    assert (await _notes_kwargs(monkeypatch, query))["allow_rollups"] is False


@pytest.mark.asyncio
async def test_neutral_statement_withholds_mood_section_notes(monkeypatch):
    assert (await _notes_kwargs(monkeypatch, "Biscuit chased a moth"))["allow_mood_sections"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "I feel hopeless and I can't stop crying",
    "    I feel hopeless and\n    I can't stop crying\n  ",
])
async def test_emotional_message_allows_mood_section_notes(monkeypatch, query):
    assert (await _notes_kwargs(monkeypatch, query))["allow_mood_sections"] is True
