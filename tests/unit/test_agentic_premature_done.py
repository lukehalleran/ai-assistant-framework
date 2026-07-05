"""Regression: a premature <done/> on round 1 must not end the agentic loop.

glm-5.2 (observed 2026-06-28, conv 0f6d70c7) signaled <done/> on round 1 without
running a single tool, so the loop gathered nothing and the final synthesis
produced a promissory non-answer ("Let me check what you've been up to…"). The
controller now nudges once when done arrives before anything is gathered, forcing
real tool use before accepting completion.
"""
import pytest
from unittest.mock import MagicMock

from core.agentic.controller import AgenticSearchController
from core.agentic.types import SearchDecision


@pytest.fixture
def controller():
    manager = MagicMock()
    manager.api_models = {}
    # sandbox_manager defaults to None → no sandbox dependency in the loop.
    return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())


class _Res:
    """Minimal stand-in for a _dispatch_single result (see controller.py:830-854)."""
    def __init__(self, formatted_context="MEMORY: relevant context."):
        self.start_events = []
        self.end_events = []
        self.round_data = object()          # truthy, not None → recorded as a round
        self.formatted_context = formatted_context
        self.memory_collection = "conversations"
        self.is_expand = False
        self.decision = SearchDecision()    # wants_search defaults False → no relaxation


@pytest.mark.asyncio
async def test_premature_done_round1_nudges_then_gathers(controller, monkeypatch):
    calls = {"n": 0}
    captured = {}

    async def fake_decision(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Bail immediately with nothing gathered.
            return [SearchDecision(is_done=True)]
        if calls["n"] == 2:
            # After the nudge, actually search memory.
            return [SearchDecision(
                wants_memory_search=True,
                memory_query="synthesis system",
                memory_collection="conversations",
            )]
        return [SearchDecision(is_done=True)]

    async def fake_dispatch(*a, **k):
        return _Res("MEMORY: yesterday you built the doc co-occurrence oracle.")

    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["session"] = session
        yield "Yesterday you built the doc co-occurrence oracle."

    monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
    monkeypatch.setattr(controller, "_dispatch_single", fake_dispatch)
    monkeypatch.setattr(controller, "_generate_final_response", fake_final)

    out = []
    async for ev in controller.run_agentic_search(
        query="Check out what I did yesterday with the synthesis system",
        system_prompt="sys",
        model_name="glm-5.2",
        initial_search_terms=[],
        skip_initial_search=True,
    ):
        out.append(ev)

    session = captured["session"]
    # The round-1 done was NOT honored — a one-shot nudge was injected...
    assert session._done_nudge_sent is True
    assert "have not gathered any information yet" in session.accumulated_context
    # ...the model then ran a real search before the loop accepted done...
    assert calls["n"] >= 3
    assert len(session.rounds) >= 1
    # ...and the final answer is grounded in what was gathered.
    text = "".join(c for c in out if isinstance(c, str))
    assert "doc co-occurrence oracle" in text


@pytest.mark.asyncio
async def test_done_honored_immediately_when_context_already_gathered(controller, monkeypatch):
    """A done after context exists is honored straight away — the guard is round-1-only."""
    calls = {"n": 0}
    captured = {}

    async def fake_decision(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [SearchDecision(
                wants_memory_search=True,
                memory_query="x",
                memory_collection="conversations",
            )]
        return [SearchDecision(is_done=True)]

    async def fake_dispatch(*a, **k):
        return _Res("MEMORY: relevant context.")

    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["session"] = session
        yield "Answer."

    monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
    monkeypatch.setattr(controller, "_dispatch_single", fake_dispatch)
    monkeypatch.setattr(controller, "_generate_final_response", fake_final)

    async for _ in controller.run_agentic_search(
        query="q", system_prompt="sys", model_name="glm-5.2",
        initial_search_terms=[], skip_initial_search=True,
    ):
        pass

    session = captured["session"]
    # Round 1 gathered context, round 2 signaled done → honored, no nudge.
    assert getattr(session, "_done_nudge_sent", False) is False
    assert session.model_signaled_done is True


@pytest.mark.asyncio
async def test_web_mode_no_seed_answer_nudged_to_search(controller, monkeypatch):
    """Web-routed query with no distilled seed terms: a tool-less first answer
    (answering purely from priors, zero web results) gets one nudge to actually
    search. Regression for the removed initial_search_terms=[query] fallback."""
    calls = {"n": 0}
    captured = {}

    async def fake_decision(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Answers from priors instead of searching — has answer text, so the
            # premature-done guard alone would have honored it.
            return [SearchDecision(is_done=True, partial_response="Python 3.13 is the latest.")]
        if calls["n"] == 2:
            return [SearchDecision(wants_search=True, search_query="latest python release")]
        return [SearchDecision(is_done=True)]

    async def fake_dispatch(*a, **k):
        return _Res("WEB: Python 3.14 released 2026-06-01.")

    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["session"] = session
        yield "Python 3.14, released June 2026."

    monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
    monkeypatch.setattr(controller, "_dispatch_single", fake_dispatch)
    monkeypatch.setattr(controller, "_generate_final_response", fake_final)

    out = []
    async for ev in controller.run_agentic_search(
        query="what's the latest python release?",
        system_prompt="sys",
        model_name="glm-5.2",
        initial_search_terms=[],      # trigger fired but distilled nothing
        skip_initial_search=False,    # pure web mode
    ):
        out.append(ev)

    session = captured["session"]
    assert session._web_mode_no_seed is True
    assert session._web_nudge_sent is True
    assert "needs fresh information from the web" in session.accumulated_context
    # The model then ran a real search before done was accepted.
    assert calls["n"] >= 3
    assert len(session.rounds) >= 1


@pytest.mark.asyncio
async def test_web_mode_no_seed_tool_use_not_nudged(controller, monkeypatch):
    """If the loop's first decision already calls a tool, no web nudge fires."""
    calls = {"n": 0}
    captured = {}

    async def fake_decision(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [SearchDecision(wants_search=True, search_query="latest python release")]
        return [SearchDecision(is_done=True)]

    async def fake_dispatch(*a, **k):
        return _Res("WEB: Python 3.14 released 2026-06-01.")

    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["session"] = session
        yield "Answer."

    monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
    monkeypatch.setattr(controller, "_dispatch_single", fake_dispatch)
    monkeypatch.setattr(controller, "_generate_final_response", fake_final)

    async for _ in controller.run_agentic_search(
        query="q", system_prompt="sys", model_name="glm-5.2",
        initial_search_terms=[], skip_initial_search=False,
    ):
        pass

    session = captured["session"]
    assert getattr(session, "_web_nudge_sent", False) is False
