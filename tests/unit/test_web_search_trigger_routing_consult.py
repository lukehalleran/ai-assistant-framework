"""
Lane DOCR2 (2026-10-26 live incident, classes: BC-27, BC-58).

PR #107 gave the Tier-4 classifier a `needs_file_search` outcome, but the live
[test] probe "could you pull up my <misspelled document noun> ..." still ran
tool-less: `_analyze_for_web_search_llm` returned the WEB heuristic result
without calling the LLM ("Skipping LLM: heuristic confident no-search"), so no
non-web outcome (memory/knowledge/doc/email/file) was ever evaluated on a
message without web keywords. #107's tests stubbed analyze_for_web_search_llm
and never reached the shortcut.

These tests drive the DEPLOYED `analyze_for_web_search_llm` and the DEPLOYED
`evaluate_agentic_gate` with a FAKE model manager (only `generate_once` is
faked -- nothing in the trigger is stubbed).
"""
import json

import pytest

import utils.web_search_trigger as wst
from core.agentic.gate import evaluate_agentic_gate
from utils.web_search_trigger import analyze_for_web_search_llm

CLEAN = "could you pull up my sylabis and check the late rule"
WRAPPED = "  could you pull up my sylabis\n     and check the late rule"
STATEMENT = "I read the sylabis last night"


class FakeMM:
    """Only generate_once is faked; the real prompt/parse/blend path runs."""

    def __init__(self, **overrides):
        payload = {
            "should_search": False, "confidence": 0.9, "reason": "user file",
            "search_terms": [], "search_depth": "quick", "num_searches": 0,
            "needs_memory_search": False, "needs_knowledge_search": False,
            "needs_document_generation": False, "needs_pattern_analysis": False,
            "document_topic": "", "document_type": "", "document_source": "",
            "needs_email_search": False, "email_query": "",
            "needs_file_search": True, "file_query": "syllabus",
        }
        payload.update(overrides)
        self.payload = payload
        self.calls = 0

    async def generate_once(self, prompt, **kw):
        self.calls += 1
        return json.dumps(self.payload)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    wst._llm_trigger_cache.clear()
    wst._llm_trigger_inflight.clear()
    from config import app_config
    monkeypatch.setattr(app_config, "FILE_ACCESS_ENABLED", True)
    monkeypatch.setattr(app_config, "FILE_ACCESS_APPROVED_FOLDERS", ["docs"])
    yield
    wst._llm_trigger_cache.clear()


async def _analyze(text, mm, **kw):
    kw.setdefault("web_search_enabled", True)
    kw.setdefault("remaining_credits", 100)
    return await analyze_for_web_search_llm(query=text, model_manager=mm, **kw)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [CLEAN, WRAPPED])
async def test_incident_gate_routes_files_via_real_trigger(text):
    mm = FakeMM()
    d = await evaluate_agentic_gate(text, model_manager=mm)
    assert mm.calls == 1
    assert d.should_trigger is True
    assert "tier4_llm_files" in (d.reason + str(getattr(d, "trigger_source", "")))


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [CLEAN, WRAPPED])
async def test_default_call_still_short_circuits(text):
    mm = FakeMM()
    d = await _analyze(text, mm)
    assert mm.calls == 0
    assert d.needs_file_search is False


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [CLEAN, WRAPPED])
async def test_routing_call_consults_llm(text):
    mm = FakeMM()
    d = await _analyze(text, mm, consult_for_routing=True)
    assert mm.calls == 1
    assert d.needs_file_search is True and d.llm_consulted is True


@pytest.mark.asyncio
async def test_statement_does_not_pay_for_the_call():
    mm = FakeMM()
    d = await evaluate_agentic_gate(STATEMENT, model_manager=mm)
    assert mm.calls == 0
    assert d.should_trigger is False


@pytest.mark.asyncio
async def test_web_toggle_off_routing_keeps_outcomes_drops_web():
    mm = FakeMM(should_search=True, search_terms=["late policy"], num_searches=1,
                confidence=0.95)
    d = await _analyze(CLEAN, mm, web_search_enabled=False, consult_for_routing=True)
    assert mm.calls == 1
    assert d.should_search is False and not d.search_terms
    assert d.needs_file_search is True


@pytest.mark.asyncio
async def test_web_toggle_off_default_unchanged():
    mm = FakeMM()
    d = await _analyze(CLEAN, mm, web_search_enabled=False)
    assert mm.calls == 0 and d.should_search is False


@pytest.mark.asyncio
async def test_heuristic_cache_entry_does_not_satisfy_routing_call():
    mm = FakeMM()
    first = await _analyze(CLEAN, mm)            # heuristic-only, cached
    assert mm.calls == 0 and first.llm_consulted is False
    d = await _analyze(CLEAN, mm, consult_for_routing=True)
    assert mm.calls == 1 and d.needs_file_search is True


@pytest.mark.asyncio
async def test_llm_cache_entry_satisfies_routing_and_plain_calls():
    mm = FakeMM()
    await _analyze(CLEAN, mm, consult_for_routing=True)
    assert mm.calls == 1
    again = await _analyze(CLEAN, mm, consult_for_routing=True)
    plain = await _analyze(CLEAN, mm)
    assert mm.calls == 1
    assert again.needs_file_search and plain.needs_file_search
