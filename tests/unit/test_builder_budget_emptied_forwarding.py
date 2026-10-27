"""The deployed build path forwards `_budget_emptied` to the agentic reuse gate.

Incident date: 2026-10-08 (the flag was added to TokenManager; the builder's
final-assembly dict forwards it only through one `if context.get(...)` branch
and the only coverage was `inspect.getsource(...)` containing the key).
Class: BC-25 (a consumer blind to what the budget removed) / BC-47.

Behavioural: drives `UnifiedPromptBuilder.build_prompt` (THE deployed builder,
real TokenManager budget trim) with a gatherer whose Obsidian-notes section is
bigger than the whole budget, then feeds the RESULT to
`AgenticSearchController._first_unmet_retrieval_key` -- the consumer. Replaces
the source-text test `test_builder_forwards_the_key`.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.agentic.controller import AgenticSearchController
from core.agentic.types import AgenticSearchSession
from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits

RECENT = [{"query": "Synthetic question", "response": "Short answer."}]
BIG_NOTE = {"content": "word " * 600, "metadata": {"title": "Synthetic note"}}


def _controller():
    manager = MagicMock()
    manager.api_models = {}
    return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())


def _builder(monkeypatch, notes, budget):
    builder = full_builder(monkeypatch, RECENT, budget=budget)
    builder.context_gatherer.get_personal_notes = AsyncMock(return_value=notes)
    return builder


def _limits():
    return retrieval_limits() | {"max_personal_notes": 3}


@pytest.mark.asyncio
async def test_emptied_section_reaches_the_reuse_gate(monkeypatch):
    builder = _builder(monkeypatch, [dict(BIG_NOTE)], budget=150)
    result = await builder.build_prompt("Synthetic question", retrieval_overrides=_limits())
    assert "_build_time" in result, "builder returned its error fallback"
    assert not result["personal_notes"], "precondition: the budget emptied the notes"
    assert result["_budget_emptied"] == ["personal_notes"]
    session = AgenticSearchSession(query="q")  # no decision block -> the emptied rule decides
    session.decision_rendered_keys = frozenset(result.keys())  # isolate the emptied rule
    assert _controller()._first_unmet_retrieval_key(result, session) == "personal_notes"


@pytest.mark.asyncio
async def test_untrimmed_section_sets_no_flag_same_builder(monkeypatch):
    # same builder shape, notes small enough to fit: nothing emptied, nothing forwarded
    builder = _builder(monkeypatch, [{"content": "tiny note", "metadata": {"title": "T"}}], budget=10000)
    result = await builder.build_prompt("Synthetic question", retrieval_overrides=_limits())
    assert result["personal_notes"]
    assert "_budget_emptied" not in result
    session = AgenticSearchSession(query="q")
    session.decision_rendered_keys = frozenset(result.keys())  # everything present was rendered
    assert _controller()._first_unmet_retrieval_key(result, session) is None
