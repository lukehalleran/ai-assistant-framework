"""2026-10-15 (class: BC-58, BC-06): the last two whole-message scans in the gate.

G1: the intent-veto exemption (`_explicit_kw`) matched 'search'/'look up' anywhere
    in a long paste, so a mid-paste word made the turn veto-exempt.
G2: the Tier-4 LLM trigger classifier (a paid call) received the ENTIRE paste.
Both now use the Tier-1 head/tail scope. Drives the DEPLOYED `evaluate_agentic_gate`
with a recording fake for `analyze_for_web_search_llm`; synthetic text.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.agentic.gate import (
    REQUEST_CONTINUATION_MAX_WORDS,
    _INCIDENTAL_HIT_EDGE_CHARS,
    _tier4_query_text,
    evaluate_agentic_gate,
)

_FILLER = (
    "Section {n}. Finished the simulation writeup and revised the plots for the "
    "second part. Fixed the labels on the axes and cleaned the notes for the group. "
)


def _body(n=40):
    return "".join(_FILLER.format(n=i) for i in range(n))


def _no_search_decision():
    return SimpleNamespace(
        should_search=False, search_terms=[], needs_memory_search=False,
        needs_knowledge_search=False, needs_document_generation=False,
        needs_pattern_analysis=False,
    )


def _recording_llm(record):
    async def _fake(query, **kw):
        record.append(query)
        return _no_search_decision()
    return _fake


@pytest.mark.asyncio
async def test_midpaste_search_word_is_not_veto_exempt():
    body = _body()
    half = len(body) // 2
    text = body[:half] + " I will search my notes sometime. " + body[half:]
    assert len(text.split()) > 600
    d = await evaluate_agentic_gate(text)
    assert d.veto_exempt is False


@pytest.mark.asyncio
async def test_head_search_request_in_long_paste_is_veto_exempt():
    d = await evaluate_agentic_gate("Please search for the STAT syllabus. " + _body())
    assert d.veto_exempt is True


@pytest.mark.asyncio
async def test_short_lookup_request_is_veto_exempt():
    d = await evaluate_agentic_gate("can you look up the syllabus for me")
    assert d.veto_exempt is True


@pytest.mark.asyncio
async def test_tier4_receives_head_and_tail_of_a_long_paste():
    text = "HEADMARK " + _body() + " TAILMARK done."
    assert len(text.split()) > 600
    seen = []
    with patch("utils.web_search_trigger.analyze_for_web_search_llm", _recording_llm(seen)):
        await evaluate_agentic_gate(text, model_manager=MagicMock(), corpus_manager=None)
    assert seen, "Tier-4 classifier was not reached"
    q = seen[0]
    assert len(q) <= 2 * _INCIDENTAL_HIT_EDGE_CHARS + 80
    assert q.startswith("HEADMARK") and q.rstrip().endswith("TAILMARK done.")
    assert "words of pasted material omitted" in q


@pytest.mark.asyncio
async def test_tier4_short_message_unchanged():
    text = "I wonder how the bridge held up after the storm yesterday evening"
    assert len(text.split()) <= REQUEST_CONTINUATION_MAX_WORDS
    seen = []
    with patch("utils.web_search_trigger.analyze_for_web_search_llm", _recording_llm(seen)):
        await evaluate_agentic_gate(text, model_manager=MagicMock(), corpus_manager=None)
    if seen:
        assert seen == [text]


def test_tier4_query_text_marker_count_and_whole_words():
    text = _body(20)
    out = _tier4_query_text(text)
    head, rest = out.split("\n[… ", 1)
    n_str, tail = rest.split(" words of pasted material omitted …]\n")
    assert text.startswith(head) and text.endswith(tail)
    assert int(n_str) == len(text.split()) - len(head.split()) - len(tail.split())
    assert head.split()[-1] in text.split() and tail.split()[0] in text.split()
