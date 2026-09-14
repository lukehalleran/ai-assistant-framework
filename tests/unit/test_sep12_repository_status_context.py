"""Post-push dump: casual commit reports must receive current repository evidence."""

import inspect
import subprocess
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.prompt.builder import UnifiedPromptBuilder, _apply_self_report_trim
from core.prompt.context_gatherer import ContextGatherer
from core.prompt.token_manager import TokenManager
from utils.repository_context import is_repository_status_report


LIVE_QUERY = "Yeah we chilling. Managed to push 6 commits in Uber lol"
GAME_QUERY = "Remind me about a game called terra Invicta Jordan keeps talking about it"


def _surface(text, wrapped):
    return textwrap.fill(text, width=29, subsequent_indent="  ") if wrapped else text


@pytest.mark.parametrize("wrapped", [False, True])
def test_report_retains_git_but_trims_personal_context(wrapped):
    query = _surface(LIVE_QUERY, wrapped)
    limits = _apply_self_report_trim({}, query, "CONVERSATIONAL")
    assert limits.get("max_git_commits", 10) == 10
    assert limits["max_mems"] == 4
    assert limits["max_user_uploads"] == 0
    # A caller's explicit disabling remains authoritative.
    assert _apply_self_report_trim({"max_git_commits": 0}, query, None)["max_git_commits"] == 0


@pytest.mark.parametrize("query, expected", [
    ("I pushed six commits today", True),
    ("I updated my repository today", True),
    ("I finished another pull request today", True),
    ("I finally committed to taking a break", False),
    ("He commits crimes", False),
    ("I am pushing through today", False),
    ("I reviewed my commitments today", False),
    ("What did I commit last week?", False),
    (GAME_QUERY, False),
])
def test_repository_evidence_requires_domain_and_status(query, expected):
    assert is_repository_status_report(query) is expected


@pytest.fixture
def git_log(monkeypatch):
    """Fake only git's read transport; parse and gather through deployed code."""
    records = [
        f"{i:08x}{'a' * 32}|||fix: repository change {i}|||Long body.|||1 hour ago|||"
        "2026-09-12T19:30:00-05:00|||Test Author\x00"
        for i in range(6, 0, -1)
    ]
    run = MagicMock(return_value=SimpleNamespace(returncode=0, stdout="\n".join(records)))
    monkeypatch.setattr("knowledge.git_memory.subprocess.run", run)
    monkeypatch.setattr("config.app_config.GIT_MEMORY_ENABLED", True)
    return run


def _gatherer():
    g = ContextGatherer.__new__(ContextGatherer)
    g.memory_coordinator = SimpleNamespace(chroma_store=None)
    g.memory_id_map = {}
    return g


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_live_git_works_without_an_index_and_renders(git_log, wrapped):
    g = _gatherer()
    commits = await g.get_git_commits(_surface(LIVE_QUERY, wrapped))
    assert len(commits) == 6
    assert len(g.memory_id_map) == 6
    assert all("Long body" not in c["content"] for c in commits)
    git_log.assert_called_once()
    assert git_log.call_args.args[0][0:2] == ["git", "log"]
    assert git_log.call_args.kwargs["timeout"] == 5


@pytest.mark.asyncio
async def test_disabled_git_does_not_read_repository(git_log, monkeypatch):
    monkeypatch.setattr("config.app_config.GIT_MEMORY_ENABLED", False)
    assert await _gatherer().get_git_commits(LIVE_QUERY) == []
    git_log.assert_not_called()


@pytest.mark.asyncio
async def test_history_question_keeps_index_route(git_log):
    g = _gatherer()
    old = {"id": "old", "content": "Commit: old fix", "metadata": {}}
    chroma = MagicMock()
    chroma.collections = {"procedural": object()}
    chroma.get_recent.return_value = [old]
    chroma.query_collection.return_value = [old]
    g.memory_coordinator.chroma_store = chroma
    assert await g.get_git_commits("What did I commit last week?") == [old]
    git_log.assert_not_called()


@pytest.mark.asyncio
async def test_git_timeout_does_not_substitute_stale_index(git_log):
    git_log.side_effect = subprocess.TimeoutExpired("git log", 5)
    assert await _gatherer().get_git_commits(LIVE_QUERY) == []


def _builder(git_log):
    mm = MagicMock()
    mm.get_active_model_name.return_value = "test-model"
    tokenizer = MagicMock()
    tokenizer.count_tokens.side_effect = lambda text, *a, **k: max(1, len(text) // 4)
    coordinator = MagicMock()
    coordinator.get_summaries.return_value = []
    coordinator.get_reflections.return_value = []
    coordinator.chroma_store = None
    b = UnifiedPromptBuilder(memory_coordinator=coordinator, model_manager=mm,
                             tokenizer_manager=tokenizer, token_budget=10000)
    g = MagicMock(spec=ContextGatherer)
    for name in dir(ContextGatherer):
        if inspect.iscoroutinefunction(getattr(ContextGatherer, name, None)):
            setattr(g, name, AsyncMock(return_value=[]))
    g.get_narrative_context.return_value = ""
    g.get_user_profile_context.return_value = ""
    g._get_web_search_results.return_value = None
    g.get_visual_memories.return_value = {"text_results": [], "images": []}
    g.memory_id_map = {}
    g.last_web_decision = None
    actual = _gatherer()
    g.get_git_commits.side_effect = actual.get_git_commits
    # Immediate live exchange: keeps the original surface/context trigger,
    # with unrelated personal names removed from the public fixture.
    g._get_recent_conversations.return_value = [{
        "query": "Ok actually in the Uber now",
        "response": "Good — glad you got out the door. Whatever the mood does from here, "
                    "you didn't let tonight get taken from you.",
        "timestamp": "2026-09-12T19:03:00-05:00",
    }]
    b.context_gatherer = g
    b._hygiene.context_gatherer = g
    b.time_manager = None
    return b


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_builder_to_prompt_contains_current_commit_subjects(git_log, wrapped):
    b = _builder(git_log)
    query = _surface(LIVE_QUERY, wrapped)
    context = await b.build_prompt(query, intent_type="general", stm_summary={
        "topic": "Uber ride", "user_question": "User pushed six commits during the ride.",
        "tone": "casual", "reference_type": "recall",
    })
    assert len(context["git_commits"]) == 6
    prompt = b._assemble_prompt(context=context, user_input=query)
    assert "[PROJECT COMMIT HISTORY] n=6" in prompt
    for i in range(1, 7):
        assert f"fix: repository change {i}" in prompt
    assert "Commit dates are not push dates" in prompt
    assert "get_git_commits" not in prompt


@pytest.mark.asyncio
async def test_commit_records_survive_competing_history_within_budget(git_log):
    commits = await _gatherer().get_git_commits(LIVE_QUERY)
    b = _builder(git_log)
    manager = TokenManager(b.model_manager, b.token_manager.tokenizer_manager, token_budget=1000)
    context = {"git_commits": commits, "recent_conversations": [
        {"query": "Old topic", "response": "Older conversation content. " * 200}
        for _ in range(10)
    ]}
    trimmed = manager._manage_token_budget(context)
    assert len(trimmed["git_commits"]) == 6
    usage = sum(manager.get_token_count(manager._extract_text(item), "test-model")
                for rows in trimmed.values() for item in rows)
    assert usage <= 1000
