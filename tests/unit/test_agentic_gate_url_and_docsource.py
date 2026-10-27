"""Behavioural siblings for two source-text assertions on
``core.agentic.gate.evaluate_agentic_gate`` (BC-63, BC-62).

* tests/unit/test_url_detect_gate.py::test_gate_uses_shared_helper grepped the
  gate source for ``"'http://' in"`` / ``contains_url(``. Incident 2026-09-28
  (BC-01): a glued ``xhttp://a.b`` counted as a URL. Here the gate is DRIVEN.
* tests/unit/test_doc_conversation_source.py::TestGateCarriesSource grepped the
  gate source for the ``"source"`` dict key. Incident 2026-08-24: the Tier-4
  ``document_source`` must reach ``doc_gen_intent["source"]``. Here the gate is
  DRIVEN with a fake LLM trigger decision.
Each live text is asserted clean and indented/wrapped.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agentic.gate import evaluate_agentic_gate


def _wrap(text):
    head, _, tail = text.rpartition(" ")
    return "    " + head + "\n    " + tail + "\n   "


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [False, True])
async def test_real_url_routes_to_url_arm(wrap):
    text = "check out https://example.com for details"
    d = await evaluate_agentic_gate(_wrap(text) if wrap else text)
    assert d.should_trigger
    assert d.trigger_source == "tier1_url"


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [False, True])
async def test_glued_scheme_is_not_a_url(wrap):
    text = "I keep the notes in xhttp://a.b for later"
    d = await evaluate_agentic_gate(_wrap(text) if wrap else text)
    assert d.trigger_source != "tier1_url"


def _llm_docgen(source):
    return MagicMock(
        should_search=False, search_terms=[], needs_memory_search=False,
        needs_knowledge_search=False, needs_document_generation=True,
        document_topic="therapy notes", document_type="summary",
        document_source=source,
    )


async def _tier4(text, decision):
    with patch("utils.web_search_trigger.analyze_for_web_search_llm",
               new_callable=AsyncMock, return_value=decision):
        return await evaluate_agentic_gate(text, model_manager=MagicMock())


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [False, True])
async def test_tier4_docgen_threads_conversation_source(wrap):
    text = "please put what we covered tonight into something I can keep"
    d = await _tier4(_wrap(text) if wrap else text, _llm_docgen("conversation"))
    assert d.doc_gen_intent is not None, d
    assert d.doc_gen_intent["source"] == "conversation"
    assert d.doc_gen_intent["topic"] == "therapy notes"


@pytest.mark.asyncio
async def test_tier4_docgen_empty_source_is_none():
    d = await _tier4("please put what we covered tonight into something I can keep",
                     _llm_docgen(""))
    assert d.doc_gen_intent is not None
    assert d.doc_gen_intent["source"] is None
