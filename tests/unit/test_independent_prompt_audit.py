"""Deterministic regressions for the independent September 4 audit.

No live stores, embedders, or providers are constructed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
import asyncio

import pytest

from core.prompt.builder import UnifiedPromptBuilder
from core.prompt.token_manager import TokenManager


class CharacterTokenizer:
    """Token-dense input: one token per character, including punctuation."""

    def count_tokens(self, text, model_name):
        return len(text)


def manager(budget=10000):
    return TokenManager(
        SimpleNamespace(get_active_model_name=lambda: "audit"),
        CharacterTokenizer(), budget,
    )


@pytest.mark.parametrize("budget", [0, 8, 64, 600])
def test_middle_out_obeys_actual_token_cap(budget):
    tm = manager()
    text = "A" * 4000 + "END"
    result = tm._middle_out(text, budget, force=True)
    assert tm.get_token_count(result, "audit") <= budget
    if budget >= 64:
        assert result.startswith("A") and result.endswith("END")


def test_high_priority_live_context_precedes_low_priority_memories():
    tm = manager(100)
    result = tm._manage_token_budget({
        "memories": [{"content": "M" * 100}],
        "google_calendar": [{"content": "C" * 50}],
        "web_search_results": [{"content": "W" * 50}],
    })
    assert result["web_search_results"] == [{"content": "W" * 50}]
    assert result["google_calendar"] == [{"content": "C" * 50}]
    assert result["memories"] == []


def test_oversized_string_is_accounted_before_admitting_lower_priority_items():
    tm = manager(100)
    result = tm._manage_token_budget({
        "user_profile": "P" * 90,
        "narrative_state": "N" * 90,
        "memories": [{"content": "M" * 10}],
    })
    assert tm._prompt_token_usage <= 100
    assert result["user_profile"] == "P" * 90


def compression_builder():
    builder = UnifiedPromptBuilder.__new__(UnifiedPromptBuilder)
    builder.model_manager = SimpleNamespace(
        get_active_model_name=lambda: "audit",
        generate_once=AsyncMock(return_value="Compressed evidence with provenance."),
    )
    builder.token_manager = manager()
    builder._llm_compress_cache = {}
    return builder


@pytest.mark.asyncio
async def test_conversation_compression_preserves_speakers_without_llm(monkeypatch):
    monkeypatch.setattr("core.prompt.builder.LLM_COMPRESSION_ENABLED", True)
    builder = compression_builder()
    entry = {"query": "Q" * 1600, "response": "R" * 1000, "id": "turn"}
    result = await builder._llm_compress_oversized({"recent_conversations": [entry]})
    builder.model_manager.generate_once.assert_not_awaited()
    assert result["recent_conversations"][0] == entry
    capped = builder.token_manager._manage_token_budget(result)["recent_conversations"][0]
    assert capped["query"].startswith("Q")
    assert capped["response"].startswith("R")
    assert len(capped["query"]) <= 600
    assert len(capped["response"]) <= 800


@pytest.mark.asyncio
async def test_compressor_updates_the_field_it_actually_read(monkeypatch):
    monkeypatch.setattr("core.prompt.builder.LLM_COMPRESSION_ENABLED", True)
    builder = compression_builder()
    original = {"content": "", "text": "X" * 2000, "id": "doc"}
    result = await builder._llm_compress_oversized({"memories": [original]})
    updated = result["memories"][0]
    assert updated["text"] == "Compressed evidence with provenance."
    assert updated["content"] == ""
    assert original["text"] == "X" * 2000


@pytest.mark.asyncio
async def test_compressor_rejects_expansion(monkeypatch):
    monkeypatch.setattr("core.prompt.builder.LLM_COMPRESSION_ENABLED", True)
    builder = compression_builder()
    builder.model_manager.generate_once.return_value = "B" * 2500
    result = await builder._llm_compress_oversized({"memories": [{"content": "A" * 2000}]})
    assert result["memories"][0]["content"] == "A" * 2000


def full_builder(monkeypatch, recent, budget=1200):
    """Run the deployed full builder without optional stores or models."""
    builder = compression_builder()
    builder.token_manager = manager(budget)
    builder.time_manager = None
    builder.model_manager.active_model_name = "audit"
    builder.memory_coordinator = SimpleNamespace(
        scorer=None, chroma_store=None, corpus_manager=None,
        get_summaries=lambda count: [], get_reflections=AsyncMock(return_value=[]),
    )
    builder._skill_activation_policy = None
    builder._should_use_light_path = lambda *args: False
    builder._is_continuation_answer = lambda *args: False
    builder._hygiene_and_caps = AsyncMock(side_effect=lambda context, **kw: context)
    builder.context_gatherer = SimpleNamespace(
        memory_id_map={}, clear_memory_id_map=lambda: None,
        get_narrative_context=lambda: "",
        _get_recent_conversations=AsyncMock(return_value=recent),
        get_user_profile_context=AsyncMock(return_value=""),
        _get_web_search_results=AsyncMock(return_value=[]),
    )
    for name in ("GOOGLE_CALENDAR_ENABLED", "EMAIL_PASSIVE_CONTEXT_ENABLED", "DAEMON_NOTES_ENABLED"):
        monkeypatch.setattr("config.app_config." + name, False)
    monkeypatch.setattr("core.prompt.builder.LLM_COMPRESSION_ENABLED", False)
    return builder


def retrieval_limits():
    return dict.fromkeys([
        "max_mems", "max_summaries", "max_reflections", "max_dreams", "max_semantic",
        "max_wiki", "max_skills", "max_proposals", "max_git_commits",
        "max_surfaced_threads", "max_reference_docs", "max_user_uploads",
        "max_proactive", "max_visual_memories", "max_personal_notes", "max_graph_sentences",
    ], 0) | {"max_recent": 2}


@pytest.mark.asyncio
async def test_full_builder_floors_cannot_bypass_final_budget(monkeypatch):
    entries = [{"query": "Q" * 20000, "response": "R" * 1000}]
    builder = full_builder(monkeypatch, entries)
    result = await builder.build_prompt("Estimate the homework effort", retrieval_overrides=retrieval_limits())
    assert "_build_time" in result, "builder must not silently return its error fallback"
    actual = sum(len(builder.token_manager._extract_text(item)) for item in result["recent_conversations"])
    assert actual <= builder.token_manager.token_budget
    assert entries[0]["query"] == "Q" * 20000, "stored source must remain intact"


@pytest.mark.asyncio
async def test_cancelled_builder_drains_retrieval_before_resetting_shared_state(monkeypatch):
    builder = full_builder(monkeypatch, [])
    entered, finished = asyncio.Event(), asyncio.Event()

    async def slow_retrieval(*args):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            finished.set()

    builder.context_gatherer._get_recent_conversations = slow_retrieval
    task = asyncio.create_task(builder.build_prompt("Estimate the homework effort", retrieval_overrides=retrieval_limits()))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set(), "retrieval outlived its cancelled request"


@pytest.mark.asyncio
async def test_lightweight_followup_caps_previous_attachment_turn(monkeypatch):
    entries = [{"query": "Q" * 400000, "response": "A short answer."}]
    builder = full_builder(monkeypatch, entries, budget=2000)
    result = await builder._build_lightweight_context("thanks")
    assert len(result["recent_conversations"][0]["query"]) <= 600
    assert result["recent_conversations"][0]["response"] == "A short answer."


@pytest.mark.asyncio
async def test_retrieved_self_notes_and_codebase_changes_reach_context(monkeypatch):
    builder = full_builder(monkeypatch, [])
    monkeypatch.setattr("config.app_config.DAEMON_NOTES_ENABLED", True)
    monkeypatch.setattr("config.app_config.DAEMON_NOTES_MAX_PER_PROMPT", 2)
    builder.context_gatherer.get_daemon_self_notes = AsyncMock(return_value=[{"content": "Remember to check the result."}])
    builder.time_manager = SimpleNamespace(time_since_previous_message=lambda: "N/A")
    changes = {"since_label": "last session", "committed": []}
    builder.context_gatherer.get_codebase_changes = AsyncMock(return_value=changes)
    result = await builder.build_prompt("Estimate the homework effort", retrieval_overrides=retrieval_limits())
    assert result["daemon_self_notes"] == [{"content": "Remember to check the result."}]
    assert result["codebase_changes"] == changes


def test_large_structured_section_keeps_schema_or_is_dropped():
    tm = manager(1000)
    changes = {"since_label": "last session", "committed": [{"summary": "X" * 10000}]}
    result = tm._manage_token_budget({"codebase_changes": changes})
    assert isinstance(result["codebase_changes"], dict)
    assert tm._prompt_token_usage <= 1000


def test_profile_retains_its_existing_three_thousand_token_allocation():
    tm = manager(10000)
    profile = "P" * 1000 + "Important identity fact." + "P" * 1000
    result = tm._manage_token_budget({"user_profile": profile})
    assert result["user_profile"] == profile


@pytest.mark.parametrize("previous", ["", "old round\n\n---\nold evidence"])
def test_single_oversized_tool_result_is_bounded(previous):
    from core.agentic.controller import AgenticSearchController
    controller = AgenticSearchController.__new__(AgenticSearchController)
    controller.context_budget_tokens = 128
    controller._estimate_tokens = len
    session = SimpleNamespace(accumulated_context=previous)
    controller._append_accumulated(session, "SOURCE: doc-1\n" + "X" * 20000 + "\nEND")
    assert len(session.accumulated_context) <= 128
    assert session.accumulated_context.endswith("END")
