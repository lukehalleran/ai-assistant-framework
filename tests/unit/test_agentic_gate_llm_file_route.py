"""Tier-4 file outcome: a Tier-1 lexical file miss has an LLM fallback.

Incident 2026-10-25 19:09 (live): a request to pull a course document whose
noun was misspelled missed Tier 1's file arm; Tier 4 had no file outcome, the
turn ran tool-less and the reply denied file access that existed.
Classes: BC-15 (routing outcome missing from one tier, sibling of the 09-27
email outcome), BC-27 (lexical primary with no fallback), BC-46 (reply denied
an existing capability).
"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agentic import gate as gate_mod
from core.agentic.gate import evaluate_agentic_gate
from utils.query_checker import is_personal_doc_search
from utils.web_search_trigger import (
    LLMSearchTriggerResponse,
    _build_llm_trigger_prompt,
)

CLEAN = "could you pull the sylabis and confirm the late rule"
WRAPPED = "  could you pull the sylabis\n     and confirm the late rule"
STATEMENT = "I read the sylabis last night and the late rule seemed odd"


def _verdict(file=True):
    return MagicMock(
        should_search=False, search_terms=[],
        needs_memory_search=False, needs_knowledge_search=False,
        needs_document_generation=False, needs_pattern_analysis=False,
        needs_email_search=False, needs_file_search=file,
        file_query="syllabus", blocked_reason="",
    )


async def _gate(text, verdict, monkeypatch, files_on=True):
    from config import app_config
    monkeypatch.setattr(app_config, "FILE_ACCESS_ENABLED", files_on)
    monkeypatch.setattr(app_config, "FILE_ACCESS_APPROVED_FOLDERS",
                        ["docs"] if files_on else [])
    with patch("utils.web_search_trigger.analyze_for_web_search_llm",
               new_callable=AsyncMock, return_value=verdict):
        return await evaluate_agentic_gate(text, model_manager=MagicMock())


def test_tier1_limit_is_documented():
    """The Tier-1 arm is lexical; this lane backs it up rather than widening it."""
    assert is_personal_doc_search("could you pull sylbus and confirm") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [CLEAN, WRAPPED])
async def test_llm_file_verdict_routes_file_tools(text, monkeypatch):
    d = await _gate(text, _verdict(), monkeypatch)
    assert d.should_trigger and "tools" in d.modes
    assert "tier4_llm_files" in d.trigger_source
    assert d.veto_exempt is True


@pytest.mark.asyncio
async def test_file_access_unconfigured_no_route(monkeypatch):
    d = await _gate(CLEAN, _verdict(), monkeypatch, files_on=False)
    assert not d.should_trigger


@pytest.mark.asyncio
async def test_statement_shape_no_route(monkeypatch):
    d = await _gate(STATEMENT, _verdict(), monkeypatch)
    assert not d.should_trigger


@pytest.mark.asyncio
async def test_mock_attribute_not_true_no_route(monkeypatch):
    v = _verdict()
    v.needs_file_search = MagicMock()  # auto-vivified, truthy, not True
    d = await _gate(CLEAN, v, monkeypatch)
    assert not d.should_trigger


@pytest.mark.asyncio
async def test_false_verdict_no_route(monkeypatch):
    d = await _gate(CLEAN, _verdict(file=False), monkeypatch)
    assert not d.should_trigger


def _payload(**kw):
    p = {"should_search": False, "search_terms": []}
    p.update(kw)
    return json.dumps(p)


def test_parser_defaults_without_keys():
    r = LLMSearchTriggerResponse.parse(_payload())
    assert r.needs_file_search is False and r.file_query == ""


def test_parser_reads_keys():
    r = LLMSearchTriggerResponse.parse(
        _payload(needs_file_search=True, file_query="syllabus"))
    assert r.needs_file_search is True and r.file_query == "syllabus"


def test_parser_rejects_wrong_type():
    assert LLMSearchTriggerResponse.parse(_payload(needs_file_search="true")) is None


def test_prompt_teaches_file_fields():
    prompt = _build_llm_trigger_prompt("open my lab report", "2026-10-25")
    assert "FILE SEARCH CRITERIA" in prompt
    assert '"needs_file_search"' in prompt and '"file_query"' in prompt


def test_tier4_mark_registered():
    assert "tier4_llm_files" in gate_mod.TRIGGER_SOURCES


def test_operating_principles_offer_matches_file_markers():
    text = Path("config/prompts/operating_principles.txt").read_text()
    assert "do not say the capability is missing" in text
    assert "Want me to pull" in text
    assert any(m in "want me to pull <name>?" for m in gate_mod.FILE_OFFER_MARKERS)


@pytest.mark.asyncio
async def test_offer_then_affirmation_routes_files():
    corpus = MagicMock()
    corpus.get_recent_memories.return_value = [
        {"query": "the syllabus", "response": "Want me to pull the syllabus?"}]
    corpus.corpus = corpus.get_recent_memories.return_value
    d = await evaluate_agentic_gate("yes", corpus_manager=corpus)
    assert d.should_trigger and "tools" in d.modes
