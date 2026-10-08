"""2026-10-08 (class: BC-70): the builder does not schedule the wiki "semantic"
retrieval task when the index is a known-absent (disabled) state; it records
the same outcome the search reports ("index_not_loaded")."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tests.unit.test_independent_prompt_audit import full_builder, retrieval_limits

QUERY = "What is the history of ancient Roman aqueduct engineering?"


async def _build(monkeypatch, index_ok):
    monkeypatch.setattr("knowledge.semantic_search.index_available", lambda: index_ok)
    builder = full_builder(monkeypatch, [{"query": "hi", "response": "hello"}], budget=10000)
    chunks = AsyncMock(return_value=[{"content": "chunk"}])
    builder.context_gatherer._get_semantic_chunks = chunks
    result = await builder.build_prompt(
        QUERY, retrieval_overrides={**retrieval_limits(), "max_semantic": 5})
    return result, chunks


@pytest.mark.asyncio
async def test_absent_index_skips_semantic_task(monkeypatch):
    result, chunks = await _build(monkeypatch, False)
    chunks.assert_not_awaited()
    assert result["_section_outcomes"]["semantic"] == {
        "status": "unavailable", "reason": "index_not_loaded"}
    assert result["semantic_chunks"] == []


@pytest.mark.asyncio
async def test_available_index_runs_semantic_task(monkeypatch):
    result, chunks = await _build(monkeypatch, True)
    chunks.assert_awaited()
    assert result["_section_outcomes"]["semantic"].get("reason") != "index_not_loaded"
