"""2026-10-12: `self_reference_note` covered attachments and pastes (gui/handlers.py)
but not pages the agentic loop fetches with fetch_url, so a third-party page that
describes "Daemon" could be read as the assistant's own claims or instructions.
2026-10-26 (lane R2): the controller appends the DEPLOYED `self_reference_note`
beside every fetch_url result where it enters the loop (round-1 direct fetch and the
in-loop dispatch). Prompt-only: `SearchRound.summary` keeps the raw page.

Incident date: 2026-10-12. class: BC-46.
Fixture pages are invented (a fictional "Atlas" assistant name is injected through
ASSISTANT_NAME so the test never depends on the owner's configured name).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config import app_config
from core.agentic.controller import AgenticSearchController
from core.agentic.types import SearchDecision
from utils.attachment_audit import SELF_REFERENCE_NOTE_PREFIX

NAME = "Atlas"
URL = "https://pages.example.test/atlas-writeup"
NAMING_PAGE = (
    "Project notes from the maker\n"
    "Atlas is my side project: a memory-backed assistant I run at home. " + ("padding text " * 40)
)
ORDINARY_PAGE = "Notes on kiln firing schedules and glaze chemistry for hobby potters. " + ("padding text " * 40)


def _indented(text):
    return "\n".join("    " + line for line in text.splitlines())


@pytest.fixture(autouse=True)
def _assistant_name(monkeypatch):
    monkeypatch.setattr(app_config, "ASSISTANT_NAME", NAME)


@pytest.fixture
def controller():
    manager = MagicMock()
    manager.api_models = {}
    return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())


async def _round1(controller, monkeypatch, page):
    async def fake_fetch(url):
        return page

    monkeypatch.setattr(controller._tool_executor, "_execute_fetch_url", fake_fetch)
    captured = {}

    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["session"] = session
        yield "Answer."

    monkeypatch.setattr(controller, "_generate_final_response", fake_final)
    async for _ in controller.run_agentic_search(
        query=f"what does this say {URL}", system_prompt="sys", model_name="glm-5.2",
        initial_search_terms=[], initial_urls=[URL], fetch_fastpath=True,
    ):
        pass
    return captured["session"]


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [lambda t: t, _indented], ids=["clean", "wrapped"])
async def test_round1_page_naming_assistant_gets_note(controller, monkeypatch, wrap):
    session = await _round1(controller, monkeypatch, wrap(NAMING_PAGE))
    assert SELF_REFERENCE_NOTE_PREFIX in session.accumulated_context
    assert f'names "{NAME}"' in session.accumulated_context
    # raw page text kept as-is in the stored round summary: no note leaks into tool content
    assert SELF_REFERENCE_NOTE_PREFIX not in session.rounds[0].summary


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [lambda t: t, _indented], ids=["clean", "wrapped"])
async def test_round1_ordinary_page_gets_no_note(controller, monkeypatch, wrap):
    session = await _round1(controller, monkeypatch, wrap(ORDINARY_PAGE))
    assert SELF_REFERENCE_NOTE_PREFIX not in session.accumulated_context
    assert "kiln firing" in session.accumulated_context


@pytest.mark.asyncio
async def test_round1_fetch_error_gets_no_note(controller, monkeypatch):
    async def fake_fetch(url):
        raise RuntimeError("boom")

    monkeypatch.setattr(controller._tool_executor, "_execute_fetch_url", fake_fetch)
    captured = {}

    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["session"] = session
        yield "Answer."

    monkeypatch.setattr(controller, "_generate_final_response", fake_final)
    async for _ in controller.run_agentic_search(
        query=f"read {URL}", system_prompt="sys", model_name="glm-5.2",
        initial_search_terms=[], initial_urls=[URL],
    ):
        pass
    assert SELF_REFERENCE_NOTE_PREFIX not in captured["session"].accumulated_context


@pytest.mark.asyncio
@pytest.mark.parametrize("page,expect_note", [(NAMING_PAGE, True), (ORDINARY_PAGE, False)],
                         ids=["names_assistant", "ordinary"])
async def test_in_loop_fetch_result_is_noted_through_the_deployed_dispatch(controller, monkeypatch, page, expect_note):
    """The model-chosen fetch (decision round, not the user's pasted link) goes through
    ToolExecutor._dispatch_fetch_url -> the controller's accumulate step."""
    async def fake_fetch(url):
        return page

    monkeypatch.setattr(controller._tool_executor, "_execute_fetch_url", fake_fetch)
    monkeypatch.setattr(controller, "detect_protocol", lambda m: __import__("core.agentic.types", fromlist=["SearchProtocol"]).SearchProtocol.NATIVE_TOOLS)
    calls = {"n": 0}

    async def fake_decision(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return [SearchDecision(wants_fetch_url=True, fetch_url=URL, fetch_url_reason="read it")]
        return [SearchDecision(wants_answer=True, partial_response="Done.")]

    monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
    captured = {}

    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["session"] = session
        yield "Answer."

    monkeypatch.setattr(controller, "_generate_final_response", fake_final)
    async for _ in controller.run_agentic_search(
        query="summarise it", system_prompt="sys", model_name="test-model",
        initial_search_terms=[], skip_initial_search=True,
    ):
        pass
    session = captured["session"]
    assert "[FETCHED URL" in session.accumulated_context
    assert (SELF_REFERENCE_NOTE_PREFIX in session.accumulated_context) is expect_note
    assert all(SELF_REFERENCE_NOTE_PREFIX not in (r.summary or "") for r in session.rounds)
