"""Behavioural siblings for orchestrator / generator turn-flow wiring (source-text rows).

Incident dates: 2026-09-04 (document trigger fired on an explicitly negated "do not
save a document"), 2026-07-25 (pacing mark moved out of ResponseGenerator), 2026-10-08
(streamed chunks re-joined with a guessed space), 2026-08-23 (agentic bypass debug
record lacked section_outcomes). Class: BC-63 (a grep for the call text stands in for
the behaviour), BC-04 (wiring dropped at a call site).

Each test runs THE deployed ``process_user_query`` / ``ResponseGenerator`` /
``BestOfHandler._fallback_streaming`` with stub collaborators and asserts the OUTCOME
(what ran, what text came out, what the debug record says). Fixture text is synthetic;
the document requests are asserted clean AND wrapped/indented.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.best_of_handler import BestOfHandler
from core.response_generator import ResponseGenerator
from tests.unit.test_process_user_query import (
    _agentic_stream, _make_context, _make_flow_orch,
)

NEGATED = ("Please make a plan for the trip. Order by date. "
           "Do not save a document for this. Plain list in the reply is fine.")
WRAPPED_NEGATED = ("    Please make a plan for the trip. Order by date.\n"
                   "    Do not save a document for this.\n    Plain list in the reply is fine.\n  ")
REQUESTED = "Please save a document summarizing the trip plan for my sister"
WRAPPED_REQUESTED = "    Please save a document\n    summarizing the trip plan for my sister\n  "


class _CapturingController:
    """Agentic controller stand-in that records the query it was handed."""
    def __init__(self, *chunks):
        self.queries = []
        self._chunks = chunks

    def run_agentic_search(self, *args, **kwargs):
        self.queries.append(kwargs.get("query"))
        chunks = self._chunks

        async def _gen():
            for c in chunks:
                yield c
        return _gen()


def _no_web_search(monkeypatch):
    decision = MagicMock()
    decision.should_search = False
    decision.search_terms = []
    monkeypatch.setattr("utils.web_search_trigger.analyze_for_web_search_llm",
                        AsyncMock(return_value=decision))


# ------------------------------------------------- document intent via the shared detector
@pytest.mark.asyncio
@pytest.mark.parametrize("text", [REQUESTED, WRAPPED_REQUESTED])
async def test_document_request_takes_the_document_route(monkeypatch, text):
    _no_web_search(monkeypatch)
    ctrl = _CapturingController("Saved ", "it")
    orch = _make_flow_orch(agentic_controller=ctrl)
    reply, debug = await orch.process_user_query(text, use_agentic_search=True)
    assert debug.get("document_generation_used") is True
    assert ctrl.queries and "generate_document tool" in ctrl.queries[0]
    assert reply == "Saved it"


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [NEGATED, WRAPPED_NEGATED])
async def test_negated_document_request_does_not_take_the_document_route(monkeypatch, text):
    _no_web_search(monkeypatch)
    ctrl = _CapturingController("Should not run")
    orch = _make_flow_orch(agentic_controller=ctrl)
    _, debug = await orch.process_user_query(text, use_agentic_search=True)
    assert not debug.get("document_generation_used")
    assert not any("generate_document tool" in (q or "") for q in ctrl.queries)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [REQUESTED, NEGATED])
async def test_agentic_gate_reaches_the_same_detector(text):
    """Gate Tier 3 reports a document intent exactly when the shared detector does."""
    from core.agentic.gate import evaluate_agentic_gate
    from knowledge.document_generator import detect_document_intent
    decision = await evaluate_agentic_gate(text)
    assert bool(decision.doc_gen_intent) == bool(detect_document_intent(text))
    assert bool(decision.doc_gen_intent) == (text == REQUESTED)


# -------------------------------------------------- agentic bypass debug record
@pytest.mark.asyncio
async def test_agentic_turn_debug_record_carries_section_outcomes(monkeypatch):
    outcomes = {"memories": {"status": "succeeded", "reason": ""},
                "personal_notes": {"status": "not_checked", "reason": "disabled"}}
    decision = MagicMock()
    decision.should_search = True
    decision.search_terms = ["weather today"]
    monkeypatch.setattr("utils.web_search_trigger.analyze_for_web_search_llm",
                        AsyncMock(return_value=decision))
    orch = _make_flow_orch(agentic_controller=_agentic_stream("Agentic ", "answer"))
    orch.build_full_prompt = AsyncMock(return_value=("PROMPT", "SYSTEM", {"_section_outcomes": outcomes}))
    text, debug = await orch.process_user_query("search the web for weather", use_agentic_search=True)
    assert text == "Agentic answer"
    assert debug["section_outcomes"] == outcomes
    assert "task_timings" in debug  # sibling key from the same site


# ---------------------------------------------------- streamed text joins byte-for-byte
LIVE_DELTAS = [" but the evidence", ' points to "', 'less worse,"', ' not "would', "n't have", ' happened."']
EXPECTED = "".join(LIVE_DELTAS)


class _Stream:
    def __init__(self, parts):
        self._parts = parts

    def __aiter__(self):
        async def _gen():
            for i, p in enumerate(self._parts):
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = p
                chunk.choices[0].delta.reasoning_content = ""
                chunk.choices[0].finish_reason = "stop" if i == len(self._parts) - 1 else None
                yield chunk
        return _gen()


def _real_generator(parts):
    """THE deployed ResponseGenerator over a fake provider stream of raw deltas."""
    model_manager = MagicMock()

    async def _generate_async(*a, **k):
        return _Stream(parts)

    model_manager.generate_async = _generate_async
    model_manager.generate_once = AsyncMock(return_value="full text")
    time_manager = MagicMock()
    time_manager.elapsed_since_last.return_value = "2 m"
    return ResponseGenerator(model_manager=model_manager, time_manager=time_manager), time_manager


@pytest.mark.asyncio
async def test_cli_route_joins_generator_chunks_byte_for_byte():
    orch = _make_flow_orch()
    orch.response_generator, _ = _real_generator(LIVE_DELTAS)
    text, _ = await orch.process_user_query("tell me what you think")
    assert text == EXPECTED.strip()
    assert 'points to "less worse," not "would' in text  # the glued-quote shape of the incident


@pytest.mark.asyncio
async def test_best_of_fallback_joins_generator_chunks_byte_for_byte():
    handler = BestOfHandler.__new__(BestOfHandler)
    handler.response_generator, _ = _real_generator(LIVE_DELTAS)
    result = await handler._fallback_streaming("prompt", "model", "system", None)
    assert result.mode == "fallback"
    assert result.response == EXPECTED.strip()


# ------------------------------------------------------------- pacing mark ownership
@pytest.mark.asyncio
async def test_response_generator_never_marks_the_query_time():
    gen, time_manager = _real_generator(["Hello", " there"])
    chunks = [c async for c in gen.generate_streaming_response("q", "m")]
    assert "".join(chunks) == "Hello there"
    assert await gen.generate_full("q", "m") == "full text"
    time_manager.mark_query_time.assert_not_called()
