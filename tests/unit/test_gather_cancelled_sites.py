"""Regression: one cancellation test per site routed through
utils.async_results (2026-09-16, class BCP-0916-T2).

Before this batch, a cancelled child of an ``asyncio.gather(...,
return_exceptions=True)`` — a ``CancelledError`` INSTANCE, not raised, sitting
in the results list — reached typed code that assumed ``isinstance(r,
Exception)`` was the only non-value shape. Each test here reproduces one
site's exact old failure mode and asserts the survivor result is used
instead.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agentic.controller import AgenticSearchController
from core.agentic.types import SearchDecision
from core.email.gmail_provider import GmailProvider
from core.email.provider import EmailMessage
from core.response_generator import ResponseGenerator
from knowledge.synthesis_generator import SynthesisGenerator


# ---------------------------------------------------------------------------
# P1 — core/agentic/controller.py: run_agentic_search tool dispatch
# ---------------------------------------------------------------------------

class _Res:
    """Minimal stand-in for a _dispatch_single result (see controller.py)."""

    def __init__(self, formatted_context="MEMORY: relevant context."):
        self.start_events = []
        self.end_events = []
        self.round_data = object()          # truthy, not None → recorded as a round
        self.formatted_context = formatted_context
        self.memory_collection = "conversations"
        self.is_expand = False
        self.decision = SearchDecision()    # wants_search defaults False → no relaxation


@pytest.fixture
def controller():
    manager = MagicMock()
    manager.api_models = {}
    return AgenticSearchController(model_manager=manager, web_search_manager=MagicMock())


@pytest.mark.asyncio
async def test_controller_skips_cancelled_tool_dispatch(controller, monkeypatch):
    """Two parallel tool dispatches in round 1, one of which is cancelled.

    Before 2026-09-16 this raised
    ``AttributeError: 'CancelledError' object has no attribute 'start_events'``
    because the old filter was ``isinstance(tr, Exception)`` — CancelledError
    is a BaseException, not an Exception — which killed the whole agentic
    generator for the turn instead of skipping the one cancelled tool.
    """
    calls = {"n": 0}
    captured = {}

    async def fake_decision(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [
                SearchDecision(
                    wants_memory_search=True,
                    memory_query="first",
                    memory_collection="conversations",
                ),
                SearchDecision(
                    wants_memory_search=True,
                    memory_query="second",
                    memory_collection="conversations",
                ),
            ]
        return [SearchDecision(is_done=True)]

    async def fake_dispatch(decision, round_num, session, crisis_level, sandbox_session):
        if decision.memory_query == "first":
            raise asyncio.CancelledError()
        return _Res("MEMORY: survivor context.")

    async def fake_final(query, system_prompt, model_name, session, initial_context=None):
        captured["session"] = session
        yield "Answer."

    monkeypatch.setattr(controller, "_get_model_decision", fake_decision)
    monkeypatch.setattr(controller, "_dispatch_single", fake_dispatch)
    monkeypatch.setattr(controller, "_generate_final_response", fake_final)

    out = []
    async for ev in controller.run_agentic_search(
        query="q", system_prompt="sys", model_name="glm-5.2",
        initial_search_terms=[], skip_initial_search=True,
    ):
        out.append(ev)

    # The generator completed (no exception) and yielded the final text.
    text = "".join(c for c in out if isinstance(c, str))
    assert "Answer." in text

    session = captured["session"]
    # Only the survivor tool's round was recorded — the cancelled one was
    # logged and skipped, not raised.
    assert len(session.rounds) == 1
    # ...and its formatted context reached the session's accumulated context.
    assert "MEMORY: survivor context." in session.accumulated_context


# ---------------------------------------------------------------------------
# core/response_generator.py — best-of / duel / ensemble
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_best_of_skips_cancelled_candidate():
    mm = MagicMock()

    async def fake_generate_once(*, prompt, model_name, system_prompt, max_tokens, temperature):
        if temperature == 0.2:
            raise asyncio.CancelledError()
        return "Paris is the capital of France."

    mm.generate_once = fake_generate_once
    rg = ResponseGenerator(model_manager=mm)

    result = await rg.generate_best_of(
        prompt="p",
        model_name="m",
        system_prompt="s",
        question_text="capital of France?",
        n=2,
        temps=(0.2, 0.7),
    )
    assert result == "Paris is the capital of France."


@pytest.mark.asyncio
async def test_duel_skips_cancelled_model():
    mm = MagicMock()

    async def fake_generate_once(*, prompt, model_name, system_prompt, max_tokens, temperature):
        if model_name == "a":
            raise asyncio.CancelledError()
        return "Answer B."

    mm.generate_once = fake_generate_once
    rg = ResponseGenerator(model_manager=mm)

    async def fake_judge(**kwargs):
        return {"winner": "B", "score_A": 0, "score_B": 9}

    with patch.object(rg, "_llm_judge_compare", side_effect=fake_judge):
        result = await rg.generate_duel_and_judge(
            prompt="p", model_a="a", model_b="b", judge_model="j",
            system_prompt="s", question_text="q",
        )
    assert result["answer"] == "Answer B."


@pytest.mark.asyncio
async def test_ensemble_skips_cancelled_generator_model():
    mm = MagicMock()

    async def fake_generate_once(*, prompt, model_name, system_prompt, max_tokens, temperature):
        if model_name == "a":
            raise asyncio.CancelledError()
        return "b's answer."

    mm.generate_once = fake_generate_once
    rg = ResponseGenerator(model_manager=mm)

    result = await rg.generate_best_of_ensemble(
        prompt="p",
        generator_models=["a", "b"],
        system_prompt="s",
        question_text="q",
        n_total=2,
        temps=(0.7,),
        selector_models=None,
    )
    assert result == "b's answer."


# ---------------------------------------------------------------------------
# core/email/gmail_provider.py — search() and recent()
# ---------------------------------------------------------------------------

def _mock_google_auth():
    mock_creds = MagicMock()
    mock_creds.token = "t"
    mock_auth = MagicMock()
    mock_auth.is_authenticated = True
    mock_auth.get_credentials.return_value = mock_creds
    mock_auth.has_scope.return_value = True
    return mock_auth


def _list_resp(messages):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"messages": messages}
    return resp


async def _fake_fetch_survivor_only(token, message_id, sem):
    if message_id == "msg1":
        raise asyncio.CancelledError()
    return EmailMessage(
        provider="gmail",
        message_id="msg2",
        subject="Survivor",
        date="2026-09-01T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_gmail_search_skips_cancelled_message(monkeypatch):
    """Before 2026-09-16: the CancelledError reached message_timestamp(),
    raised AttributeError, and the outer `except Exception` swallowed it —
    search() silently returned [] instead of the one survivor message.
    """
    provider = GmailProvider()
    monkeypatch.setattr(provider, "_fetch_message_metadata", _fake_fetch_survivor_only)

    with patch("core.actions.google_auth.get_google_auth", return_value=_mock_google_auth()), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock,
               return_value=_list_resp([{"id": "msg1"}, {"id": "msg2"}])):
        result = await provider.search("test")

    assert len(result) == 1
    assert result[0].message_id == "msg2"


@pytest.mark.asyncio
async def test_gmail_recent_skips_cancelled_message(monkeypatch):
    provider = GmailProvider()
    monkeypatch.setattr(provider, "_fetch_message_metadata", _fake_fetch_survivor_only)

    with patch("core.actions.google_auth.get_google_auth", return_value=_mock_google_auth()), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock,
               return_value=_list_resp([{"id": "msg1"}, {"id": "msg2"}])):
        result = await provider.recent(window_days=7, limit=25)

    assert len(result) == 1
    assert result[0].message_id == "msg2"


# ---------------------------------------------------------------------------
# knowledge/synthesis_generator.py — generate_candidates articulation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesis_generator_skips_cancelled_pair():
    gen = SynthesisGenerator(chroma_store=MagicMock(), model_manager=MagicMock())

    cand = object()

    def fake_sample_personal(n):
        return [{"id": "p1"}, {"id": "p2"}]

    def fake_sample_wiki(n):
        return [{"id": "w1"}, {"id": "w2"}]

    def fake_form_pairs(personal, wiki, max_pairs):
        return [("PAIR_CANCEL",), ("PAIR_SURVIVOR",)]

    async def fake_articulate(pair, semaphore):
        if pair == ("PAIR_CANCEL",):
            raise asyncio.CancelledError()
        return cand

    with patch.object(gen, "_sample_personal_entities", side_effect=fake_sample_personal), \
         patch.object(gen, "_sample_wiki_articles", side_effect=fake_sample_wiki), \
         patch.object(gen, "_form_pairs", side_effect=fake_form_pairs), \
         patch.object(gen, "_articulate_and_package", side_effect=fake_articulate), \
         patch("config.app_config.SYNTHESIS_GENERATOR_ENABLED", True), \
         patch("config.app_config.SYNTHESIS_GENERATOR_LLM_CONCURRENCY", 3), \
         patch("config.app_config.SYNTHESIS_GENERATOR_MIN_GRAPH_NODES", 20):
        result = await gen.generate_candidates(count=2)

    assert result == [cand]


# ---------------------------------------------------------------------------
# gui/handlers.py — _find_email_draft (BC-14: fallback parameter made real)
# ---------------------------------------------------------------------------

_DRAFT_A = (
    "Here's the weekly summary draft:\n\n"
    "- Completed the quarterly report and shared it with the whole team\n"
    "- Reviewed the team OKRs and flagged two at-risk items for follow-up\n"
    "- Scheduled the cross-team offsite for next month\n"
    "- Followed up with three vendors about renewal pricing options"
)

_DRAFT_B = (
    "Draft for the volunteer newsletter:\n\n"
    "- Organized the supply closet and logged the remaining inventory\n"
    "- Reached out to two new sponsors about the spring fundraiser\n"
    "- Booked the community hall for the next planning meeting\n"
    "- Sent thank-you notes to last month's donors"
)

assert len(_DRAFT_A) > 200 and _DRAFT_A.count("\n") >= 3
assert len(_DRAFT_B) > 200 and _DRAFT_B.count("\n") >= 3


def test_find_email_draft_short_meta_commentary_fallback_no_history_none():
    from gui.handlers import _find_email_draft
    assert _find_email_draft([], "Sure, I'll send that right away.") is None


def test_find_email_draft_uses_fallback_when_history_empty():
    from gui.handlers import _find_email_draft
    draft = _find_email_draft([], _DRAFT_A)
    assert draft is not None
    assert draft.startswith("- Completed the quarterly report")
    assert "Here's the weekly summary draft:" not in draft


def test_find_email_draft_falls_back_to_history_when_fallback_is_meta():
    from gui.handlers import _find_email_draft
    history = [{"role": "assistant", "content": _DRAFT_B}]
    draft = _find_email_draft(history, "Sure, I'll send that right away.")
    assert draft is not None
    assert draft.startswith("- Organized the supply closet")


def test_find_email_draft_prefers_fallback_over_history_draft():
    from gui.handlers import _find_email_draft
    history = [{"role": "assistant", "content": _DRAFT_B}]
    draft = _find_email_draft(history, _DRAFT_A)
    assert draft is not None
    assert draft.startswith("- Completed the quarterly report")
